#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tokenisation and batch planning for ModernProst inference (numpy only).

Two planners, matching the two inference engines in
:mod:`pholdlib.modernprost.inference`:

``plan_padded_batches``
    One protein per row, rows right-padded to the batch's longest protein
    (rounded up to ``pad_multiple``). A batch closes when *rows x padded
    length* would exceed the token budget, so the budget bounds what the GPU
    actually computes, padding included. Proteins are taken longest-first, so
    each batch holds similar lengths and padding stays small.

``plan_packed_batches``
    Many proteins concatenated into one row of exactly ``capacity`` tokens.
    Each bin is filled from the long end of the length-sorted list and topped
    up from the short end, so the only waste is the tail of each bin (less
    than the shortest remaining protein).

Tokenisation is done once for the whole input with a 256-entry lookup table
built from the tokenizer's own vocabulary. ModernProst's tokenizer maps one
character to one id with no special tokens, so the table reproduces it
exactly, without calling the Hugging Face tokenizer per batch.
"""

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Chunk marker embedded in a temporary item id (see inference.chunk_sequence).
CHUNK_SEP = "__chunk"

# Characters replaced by X before tokenisation, as the original engine did.
_REPLACE_WITH_X = "UZO"


def chunk_sequence(seq: str, max_len: int):
    """Yield ``(start, subseq)`` splitting *seq* into near-equal chunks.

    Chunks are non-overlapping and cover the sequence exactly; sizes differ by
    at most one residue so no chunk is a tiny remainder. Splitting evenly
    rather than taking ``max_len``-sized prefixes keeps every chunk in a
    similar context regime, which matters because a 20-residue tail chunk
    would be predicted with almost no structural context.
    """
    L = len(seq)
    n_chunks = math.ceil(L / max_len)
    base = L // n_chunks
    remainder = L % n_chunks

    start = 0
    for i in range(n_chunks):
        size = base + (1 if i < remainder else 0)
        yield start, seq[start : start + size]
        start += size


def build_token_lut(vocab: Dict[str, int], unk_id: int) -> np.ndarray:
    """Byte -> token id table equivalent to the ModernProst tokenizer.

    Every single-character vocabulary entry maps to its id; every other byte
    (lowercase, ``*``, gaps, ...) maps to ``unk_id``, as the tokenizer's
    ``_convert_token_to_id`` does. U, Z and O are mapped to X here, mirroring
    the replacement the original engine applied before tokenising.
    """
    lut = np.full(256, unk_id, dtype=np.uint8)
    for token, idx in vocab.items():
        if len(token) == 1 and ord(token) < 256:
            lut[ord(token)] = idx
    if "X" in vocab:
        for ch in _REPLACE_WITH_X:
            lut[ord(ch)] = vocab["X"]
    return lut


def lut_from_tokenizer(tokenizer) -> np.ndarray:
    """:func:`build_token_lut` for a loaded ModernProst tokenizer."""
    vocab = tokenizer.get_vocab()
    return build_token_lut(vocab, vocab[str(tokenizer.unk_token)])


@dataclass
class TokenisedItems:
    """All inference items (proteins, or chunks of long proteins), tokenised
    once into one flat uint8 buffer.

    Item *k* occupies ``tokens[offsets[k]:offsets[k + 1]]``. Chunks of one
    protein are consecutive items, so a chunked protein's outputs are one
    contiguous slice of any per-token array laid out like ``tokens``.

    Attributes:
        seq_ids: original sequence id per protein, in input order.
        item_protein: index into ``seq_ids`` for each item.
        lengths: residues per item.
        offsets: start of each item in the flat buffer (len = n_items + 1).
        tokens: flat token ids.
        protein_offsets: start of each protein in the flat buffer
            (len = n_proteins + 1); a protein spans all of its chunks.
        n_chunked: number of proteins that were split into chunks.
    """

    seq_ids: List[str]
    item_protein: np.ndarray
    lengths: np.ndarray
    offsets: np.ndarray
    tokens: np.ndarray
    protein_offsets: np.ndarray
    n_chunked: int

    @property
    def n_items(self) -> int:
        return len(self.lengths)

    @property
    def n_tokens(self) -> int:
        return int(self.offsets[-1])

    @classmethod
    def from_sequences(
        cls,
        seq_dict: Sequence[Tuple[str, str, int]],
        lut: np.ndarray,
        chunk_len: int,
    ) -> "TokenisedItems":
        """Chunk long sequences and tokenise everything in one pass.

        Args:
            seq_dict: ``(seq_id, sequence, length)`` tuples in any order.
            lut: table from :func:`build_token_lut`.
            chunk_len: sequences longer than this are split with
                :func:`chunk_sequence`.
        """
        seq_ids: List[str] = []
        item_protein: List[int] = []
        lengths: List[int] = []
        pieces: List[str] = []
        n_chunked = 0
        for p, (seq_id, seq, _) in enumerate(seq_dict):
            seq_ids.append(seq_id)
            if len(seq) > chunk_len:
                n_chunked += 1
                for _, sub in chunk_sequence(seq, chunk_len):
                    item_protein.append(p)
                    lengths.append(len(sub))
                    pieces.append(sub)
            else:
                item_protein.append(p)
                lengths.append(len(seq))
                pieces.append(seq)

        lengths_arr = np.asarray(lengths, dtype=np.int64)
        offsets = np.zeros(len(lengths_arr) + 1, dtype=np.int64)
        np.cumsum(lengths_arr, out=offsets[1:])
        raw = "".join(pieces).encode("latin-1", errors="replace")
        tokens = lut[np.frombuffer(raw, dtype=np.uint8)]

        item_protein_arr = np.asarray(item_protein, dtype=np.int64)
        protein_offsets = np.zeros(len(seq_ids) + 1, dtype=np.int64)
        if len(item_protein_arr):
            per_protein = np.bincount(item_protein_arr, weights=lengths_arr,
                                      minlength=len(seq_ids)).astype(np.int64)
            np.cumsum(per_protein, out=protein_offsets[1:])
        return cls(seq_ids, item_protein_arr, lengths_arr, offsets, tokens,
                   protein_offsets, n_chunked)


def _roundup(x: int, multiple: int) -> int:
    if multiple and multiple > 1:
        return ((x + multiple - 1) // multiple) * multiple
    return x


def plan_padded_batches(
    lengths: np.ndarray,
    token_budget: int,
    pad_multiple: int = 8,
    max_batch: Optional[int] = None,
    max_scores: Optional[int] = None,
) -> List[np.ndarray]:
    """Group items into padded batches bounded by computed tokens.

    Items are visited longest-first. A batch closes when adding the next item
    would make ``rows x padded_length`` exceed ``token_budget`` (the padded
    length is set by the batch's first, longest item), when ``max_batch``
    rows are reached, or when ``rows x padded_length**2`` would exceed
    ``max_scores``. The last bound is for devices whose attention
    materialises the full score matrix (MPS, CPU): it keeps batches of long
    proteins small while short proteins still fill the token budget. An item
    longer than the budget gets a batch of its own.

    Returns:
        A list of item-index arrays, longest batches first.
    """
    order = np.argsort(-np.asarray(lengths), kind="stable")
    batches: List[np.ndarray] = []
    start = 0
    n = len(order)
    while start < n:
        lpad = _roundup(int(lengths[order[start]]), pad_multiple)
        rows = max(1, token_budget // max(lpad, 1))
        if max_batch:
            rows = min(rows, max_batch)
        if max_scores:
            rows = max(1, min(rows, max_scores // max(lpad * lpad, 1)))
        end = min(n, start + rows)
        batches.append(order[start:end])
        start = end
    return batches


def plan_packed_batches(lengths: np.ndarray, capacity: int) -> List[np.ndarray]:
    """Bin items into rows of exactly ``capacity`` tokens.

    Each bin takes items from the long end of the length-sorted list until the
    next one does not fit, then fills the remaining space from the short end.
    O(n); the free space left in a bin is smaller than the shortest item not
    yet placed.

    Raises:
        ValueError: if an item is longer than ``capacity``.
    """
    lengths = np.asarray(lengths)
    if len(lengths) and int(lengths.max()) > capacity:
        raise ValueError(
            f"item of {int(lengths.max())} residues exceeds pack capacity {capacity}"
        )
    order = np.argsort(-lengths, kind="stable")
    i, j = 0, len(order) - 1
    bins: List[np.ndarray] = []
    while i <= j:
        cur: List[int] = []
        free = capacity
        while i <= j and lengths[order[i]] <= free:
            cur.append(order[i])
            free -= int(lengths[order[i]])
            i += 1
        while j >= i and lengths[order[j]] <= free:
            cur.append(order[j])
            free -= int(lengths[order[j]])
            j -= 1
        bins.append(np.asarray(cur, dtype=np.int64))
    return bins


def batch_token_count(lengths: np.ndarray, batches: Iterable[np.ndarray],
                      pad_multiple: int = 8) -> int:
    """Tokens the GPU computes for padded batches (rows x padded length)."""
    total = 0
    for b in batches:
        if len(b):
            total += len(b) * _roundup(int(lengths[b].max()), pad_multiple)
    return total
