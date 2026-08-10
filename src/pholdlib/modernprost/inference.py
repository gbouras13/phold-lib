#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ModernProst 3Di + 12-state batch inference engine.

Adapted from foldseek2-plm's ``infer-pure`` inference loop, restructured to
match :mod:`pholdlib.prostt5.inference` so phold and baktfold can drive either
backend through the same call shape.

Two tasks are supported:

``classification``
    argmax the 3Di (20-class) and 12st (12-class) heads into per-residue class
    indices, plus per-residue max softmax probabilities. Feeds the FASTA
    writers and the combined ``_ss`` Foldseek DB.

``pssm``
    keep the full per-residue softmax distributions — ``(L, 20)`` and
    ``(L, 12)`` float arrays. Feeds the Foldseek profile DBs. Memory is ~32x
    the classification path per residue, which is why it is opt-in.

Differences from the ProstT5 engine that matter when reading this code:

* No ``<AA2fold>`` prefix token and no EOS, so logit position *i* is residue
  *i* — there is no index shift and no attention-mask patching.
* Sequences longer than ``chunk_len`` are split into near-equal chunks,
  inferred independently and stitched back together, because attention cost
  grows quadratically in the global-attention layers.
* Batches are padded up to a multiple of 64 so Tensor Cores hit peak on the
  padded dimension.
"""

import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

from pholdlib.databases.modernprost import (
    CLASSIFICATION_TASK,
    PSSM_TASK,
    VALID_TASKS,
)
from pholdlib.prostt5.device import shard_round_robin
from pholdlib.prostt5.model import toCPU

# Sequences longer than this are split before inference. 1568 is the value
# phold settled on for ModernProst: comfortably inside the model's 8192-token
# positional range while keeping the global-attention layers' quadratic cost
# bounded on the long tail of phage proteins.
DEFAULT_CHUNK_LEN = 1568

# Chunk marker embedded in a temporary sequence id. Chosen to be absurd enough
# that it cannot collide with a real CDS id from Pharokka/Bakta/NCBI.
_CHUNK_SEP = "__chunk"


@dataclass
class ModernProstResult:
    """Everything one inference run produces.

    A dataclass rather than the ProstT5 engine's 4-tuple because ModernProst
    emits two heads plus two optional profile stores — a 7-tuple would be
    unreadable at the call site.

    Attributes:
        predictions_3di / predictions_12st: ``{seq_id: (pred, mean_prob, all_prob)}``.
            ``pred`` is a byte array of class indices; ``mean_prob`` is a float
            percentage rounded to 2 dp; ``all_prob`` is a float32 array of
            per-residue max probabilities in ``[0, 1]``, or None when
            ``output_probs`` is False. Empty for the ``pssm`` task.
        profiles_3di / profiles_12st: ``{seq_id: ndarray (L, 20 | 12)}`` softmax
            profiles. Empty for the ``classification`` task.
        embeddings_per_residue / embeddings_per_protein: optional embedding
            stores, keyed by seq_id.
        fail_ids: sequence IDs that could not be processed.
    """

    predictions_3di: Dict[str, Tuple] = field(default_factory=dict)
    predictions_12st: Dict[str, Tuple] = field(default_factory=dict)
    profiles_3di: Dict[str, np.ndarray] = field(default_factory=dict)
    profiles_12st: Dict[str, np.ndarray] = field(default_factory=dict)
    embeddings_per_residue: Dict[str, np.ndarray] = field(default_factory=dict)
    embeddings_per_protein: Dict[str, np.ndarray] = field(default_factory=dict)
    fail_ids: List[str] = field(default_factory=list)

    def update(self, other: "ModernProstResult") -> None:
        """Merge another result in place (used to combine multi-GPU shards)."""
        self.predictions_3di.update(other.predictions_3di)
        self.predictions_12st.update(other.predictions_12st)
        self.profiles_3di.update(other.profiles_3di)
        self.profiles_12st.update(other.profiles_12st)
        self.embeddings_per_residue.update(other.embeddings_per_residue)
        self.embeddings_per_protein.update(other.embeddings_per_protein)
        self.fail_ids.extend(other.fail_ids)


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


def predictions_from_profiles(
    profiles: Dict[str, np.ndarray],
    output_probs: bool = True,
) -> Dict[str, Tuple]:
    """Collapse softmax profiles into the ``(pred, mean_prob, all_prob)`` shape.

    The ``pssm`` task keeps full distributions and so produces no argmax
    predictions, but downstream consumers still want a representative state
    string and a per-residue confidence — phold, for example, reports a mean
    confidence per CDS and masks low-confidence residues in the amino-acid
    FASTA regardless of which task ran. Taking the argmax and its probability
    here gives exactly the values the ``classification`` task would have
    produced from the same logits, without a second forward pass.
    """
    out: Dict[str, Tuple] = {}
    for seq_id, profile in profiles.items():
        pred = np.argmax(profile, axis=-1).astype(np.byte)
        probs = np.max(profile, axis=-1).astype(np.float32)
        out[seq_id] = _finalise(pred, probs, output_probs)
    return out


def run_modernprost_inference(
    seq_dict: List[Tuple[str, str, int]],
    model,
    tokenizer,
    device: torch.device,
    task: str = CLASSIFICATION_TASK,
    max_residues: int = 50000,
    max_seq_len: int = 30000,
    max_batch: int = 10000,
    chunk_len: int = DEFAULT_CHUNK_LEN,
    output_probs: bool = True,
    save_per_residue_embeddings: bool = False,
    save_per_protein_embeddings: bool = False,
    pad_multiple: int = 64,
    desc: str = "Predicting 3Di + 12st",
    progress_callback: Optional[Callable[[int], None]] = None,
) -> ModernProstResult:
    """
    Run ModernProst on a list of protein sequences.

    Args:
        seq_dict: Pre-sorted (descending by length) list of
            ``(seq_id, sequence, length)`` tuples. Non-standard amino acids are
            replaced with X internally, so callers need not pre-clean.
        model: ModernProst model from :func:`get_modernprost_model`.
        tokenizer: matching tokenizer from :func:`get_modernprost_model`.
        device: Torch device the model lives on.
        task: ``"classification"`` or ``"pssm"``.
        max_residues: Max total residues per batch before flushing.
        max_seq_len: Any single chunk longer than this triggers an immediate flush.
        max_batch: Max number of sequences (post-chunking) per batch.
        chunk_len: Sequences longer than this are split before inference.
        output_probs: When False, per-residue probability arrays are dropped
            from the result (the mean is still computed). Ignored for ``pssm``.
        save_per_residue_embeddings: Collect per-residue ``last_hidden_state``.
        save_per_protein_embeddings: Collect mean-pooled per-protein embeddings.
        pad_multiple: Pad each batch's length dimension up to a multiple of
            this. Set to 0 or 1 to disable.
        desc: tqdm progress bar description.
        progress_callback: Optional ``progress_callback(n)`` invoked after each
            flushed batch. When provided, the internal tqdm bar is disabled —
            used by the multi-GPU dispatcher to drive a single shared bar.

    Returns:
        A :class:`ModernProstResult`.
    """
    task = str(task).lower()
    if task not in VALID_TASKS:
        raise ValueError(f"task must be one of {VALID_TASKS}, got {task!r}")
    is_pssm = task == PSSM_TASK

    result = ModernProstResult()
    want_embeddings = save_per_residue_embeddings or save_per_protein_embeddings

    # ── expand long sequences into chunks ───────────────────────────────────
    # Chunk items carry a temporary "<id>__chunk<N>" key and are stitched back
    # together after the loop. Chunk order within an id is recovered from N,
    # not from batch completion order, so multi-GPU interleaving is safe.
    items: List[Tuple[str, str, int]] = []
    chunked_ids: set = set()
    for seq_id, seq, _ in seq_dict:
        seq = seq.replace("U", "X").replace("Z", "X").replace("O", "X")
        if len(seq) > chunk_len:
            chunked_ids.add(seq_id)
            for ci, (_, subseq) in enumerate(chunk_sequence(seq, chunk_len)):
                items.append((f"{seq_id}{_CHUNK_SEP}{ci}", subseq, len(subseq)))
        else:
            items.append((seq_id, seq, len(seq)))

    if chunked_ids:
        logger.info(
            f"{len(chunked_ids)} sequence(s) longer than {chunk_len} residues "
            f"were split into chunks for inference and reassembled afterwards."
        )

    # Per-chunk stores, keyed by base id then chunk index.
    chunk_store_3di: Dict[str, Dict[int, Tuple]] = {}
    chunk_store_12st: Dict[str, Dict[int, Tuple]] = {}
    chunk_profiles_3di: Dict[str, Dict[int, np.ndarray]] = {}
    chunk_profiles_12st: Dict[str, Dict[int, np.ndarray]] = {}
    chunk_emb_residue: Dict[str, Dict[int, np.ndarray]] = {}

    use_internal_bar = progress_callback is None
    pbar = tqdm(total=len(items), desc=desc) if use_internal_bar else None

    batch: List[Tuple[str, str, int]] = []
    n_res_batch = 0

    for idx, (item_id, seq, slen) in enumerate(items, 1):
        batch.append((item_id, seq, slen))
        n_res_batch += slen

        flush = (
            len(batch) >= max_batch
            or n_res_batch >= max_residues
            or idx == len(items)
            or slen > max_seq_len
        )
        if not flush:
            continue

        item_ids, seqs, seq_lens = zip(*batch)
        batch_n = len(batch)
        batch = []
        batch_residues = n_res_batch
        n_res_batch = 0

        try:
            logits_3di, logits_12st, hidden = _forward_batch(
                model, tokenizer, device, seqs, pad_multiple, want_embeddings
            )
        except RuntimeError as exc:
            logger.warning(
                f"RuntimeError during ModernProst inference — batch residues: "
                f"{batch_residues}, batch size: {batch_n}, ids: {item_ids} ({exc})"
            )
            result.fail_ids.extend(_base_ids(item_ids))
            _tick(pbar, progress_callback, batch_n)
            continue

        try:
            if is_pssm:
                profiles_3di = toCPU(torch.softmax(logits_3di.float(), dim=-1))
                profiles_12st = toCPU(torch.softmax(logits_12st.float(), dim=-1))
            else:
                preds_3di = toCPU(torch.argmax(logits_3di, dim=-1)).astype(np.byte)
                preds_12st = toCPU(torch.argmax(logits_12st, dim=-1)).astype(np.byte)
                # .float() before toCPU — numpy has no bfloat16 dtype, and fp16
                # means would accumulate visible error over a long protein.
                probs_3di = toCPU(
                    torch.softmax(logits_3di.float(), dim=-1).max(dim=-1).values
                )
                probs_12st = toCPU(
                    torch.softmax(logits_12st.float(), dim=-1).max(dim=-1).values
                )

            for i, item_id in enumerate(item_ids):
                L = seq_lens[i]
                base_id, chunk_idx = _split_chunk_id(item_id)

                if want_embeddings:
                    _store_embeddings(
                        result,
                        chunk_emb_residue,
                        hidden,
                        i,
                        L,
                        base_id,
                        chunk_idx,
                        save_per_residue_embeddings,
                        save_per_protein_embeddings,
                    )

                if is_pssm:
                    prof_3di = profiles_3di[i, :L, :]
                    prof_12st = profiles_12st[i, :L, :]
                    if chunk_idx is None:
                        result.profiles_3di[base_id] = prof_3di
                        result.profiles_12st[base_id] = prof_12st
                    else:
                        chunk_profiles_3di.setdefault(base_id, {})[chunk_idx] = prof_3di
                        chunk_profiles_12st.setdefault(base_id, {})[chunk_idx] = prof_12st
                    continue

                p3, pr3 = preds_3di[i, :L], probs_3di[i, :L]
                p12, pr12 = preds_12st[i, :L], probs_12st[i, :L]

                if chunk_idx is None:
                    result.predictions_3di[base_id] = _finalise(p3, pr3, output_probs)
                    result.predictions_12st[base_id] = _finalise(p12, pr12, output_probs)
                else:
                    # Per-chunk means are meaningless; the mean is recomputed
                    # from the concatenated full-length arrays after the loop.
                    chunk_store_3di.setdefault(base_id, {})[chunk_idx] = (p3, pr3)
                    chunk_store_12st.setdefault(base_id, {})[chunk_idx] = (p12, pr12)

        except IndexError as exc:
            logger.warning(
                f"IndexError while decoding ModernProst output for {item_ids} ({exc})"
            )
            result.fail_ids.extend(_base_ids(item_ids))

        _tick(pbar, progress_callback, batch_n)

    if pbar is not None:
        pbar.close()

    # ── stitch chunked sequences back together ──────────────────────────────
    _concat_prediction_chunks(chunk_store_3di, result.predictions_3di, output_probs)
    _concat_prediction_chunks(chunk_store_12st, result.predictions_12st, output_probs)
    _concat_profile_chunks(chunk_profiles_3di, result.profiles_3di)
    _concat_profile_chunks(chunk_profiles_12st, result.profiles_12st)

    if save_per_residue_embeddings:
        for base_id, chunks in chunk_emb_residue.items():
            result.embeddings_per_residue[base_id] = np.concatenate(
                [chunks[ci] for ci in sorted(chunks)], axis=0
            )
    if save_per_protein_embeddings:
        # A chunked protein's mean must be recomputed over the whole sequence;
        # averaging per-chunk means would weight short chunks too heavily.
        for base_id, chunks in chunk_emb_residue.items():
            full = np.concatenate([chunks[ci] for ci in sorted(chunks)], axis=0)
            result.embeddings_per_protein[base_id] = full.mean(axis=0)

    # De-duplicate fail ids: a chunked protein can fail in more than one batch.
    result.fail_ids = list(dict.fromkeys(result.fail_ids))

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _forward_batch(
    model,
    tokenizer,
    device: torch.device,
    seqs: Tuple[str, ...],
    pad_multiple: int,
    want_hidden: bool,
):
    """Tokenise, pad, and run one forward pass.

    Returns ``(logits_3di, logits_12st, last_hidden_state_or_None)``, all still
    on ``device``.
    """
    inputs = tokenizer(
        list(seqs),
        add_special_tokens=False,
        padding="longest",
        truncation=False,
        return_tensors="pt",
    )
    # ModernProst has no token_type embedding; a stray key would be forwarded
    # into the model as **kwargs and silently ignored, but drop it explicitly.
    inputs.pop("token_type_ids", None)

    if pad_multiple and pad_multiple > 1:
        cur_len = inputs["input_ids"].shape[1]
        pad_to = ((cur_len + pad_multiple - 1) // pad_multiple) * pad_multiple
        if pad_to != cur_len:
            extra = pad_to - cur_len
            inputs["input_ids"] = F.pad(
                inputs["input_ids"], (0, extra), value=tokenizer.pad_token_id
            )
            inputs["attention_mask"] = F.pad(inputs["attention_mask"], (0, extra), value=0)

    input_ids = inputs["input_ids"].to(device, non_blocking=True)
    attention_mask = inputs["attention_mask"].to(device, non_blocking=True)

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    hidden = outputs.last_hidden_state if want_hidden else None
    return logits["3di"], logits["12st"], hidden


def _finalise(pred: np.ndarray, probs: np.ndarray, output_probs: bool) -> Tuple:
    """Build the ``(pred, mean_prob, all_prob)`` tuple for one sequence."""
    mean_prob = round(100 * float(np.mean(probs)), 2) if probs.size else 0.0
    return (pred, mean_prob, probs if output_probs else None)


def _split_chunk_id(item_id: str) -> Tuple[str, Optional[int]]:
    """Split ``"<id>__chunk<N>"`` into ``("<id>", N)``; plain ids give ``(id, None)``.

    Uses rsplit so a CDS id that itself contains ``__chunk`` still resolves to
    the marker this function appended, not one embedded in the user's id.
    """
    if _CHUNK_SEP not in item_id:
        return item_id, None
    base, _, tag = item_id.rpartition(_CHUNK_SEP)
    try:
        return base, int(tag)
    except ValueError:
        return item_id, None


def _base_ids(item_ids: Tuple[str, ...]) -> List[str]:
    """Map batch item ids back to their originating sequence ids."""
    return [_split_chunk_id(i)[0] for i in item_ids]


def _tick(pbar, progress_callback, n: int) -> None:
    """Advance whichever progress reporter is active."""
    if pbar is not None:
        pbar.update(n)
    if progress_callback is not None:
        progress_callback(n)


def _store_embeddings(
    result: ModernProstResult,
    chunk_emb_residue: Dict[str, Dict[int, np.ndarray]],
    hidden: torch.Tensor,
    i: int,
    L: int,
    base_id: str,
    chunk_idx: Optional[int],
    save_per_residue: bool,
    save_per_protein: bool,
) -> None:
    """Extract and store this batch row's embeddings."""
    try:
        emb = hidden[i, :L].detach().float().cpu().numpy()
    except Exception:
        logger.warning(f"Saving embeddings failed for {base_id}")
        return

    if chunk_idx is not None:
        # Both the per-residue and the per-protein output need the full
        # concatenation, so chunks are always stored per-residue and reduced
        # after stitching.
        chunk_emb_residue.setdefault(base_id, {})[chunk_idx] = emb
        return

    if save_per_residue:
        result.embeddings_per_residue[base_id] = emb
    if save_per_protein:
        result.embeddings_per_protein[base_id] = emb.mean(axis=0)


def _concat_prediction_chunks(
    store: Dict[str, Dict[int, Tuple]],
    sink: Dict[str, Tuple],
    output_probs: bool,
) -> None:
    """Concatenate per-chunk predictions in chunk order and recompute the mean."""
    for base_id, chunks in store.items():
        order = sorted(chunks)
        preds = np.concatenate([chunks[ci][0] for ci in order])
        probs = np.concatenate([chunks[ci][1] for ci in order])
        sink[base_id] = _finalise(preds, probs, output_probs)


def _concat_profile_chunks(
    store: Dict[str, Dict[int, np.ndarray]],
    sink: Dict[str, np.ndarray],
) -> None:
    """Concatenate per-chunk profiles along the residue axis, in chunk order."""
    for base_id, chunks in store.items():
        sink[base_id] = np.concatenate([chunks[ci] for ci in sorted(chunks)], axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-GPU dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def run_modernprost_inference_multi_gpu(
    seq_dict: List[Tuple[str, str, int]],
    devices: List[str],
    model_dir: Path,
    model_name: str,
    half_precision: bool,
    threads: int,
    task: str = CLASSIFICATION_TASK,
    check_fn: Optional[Callable] = None,
    zenodo_fn: Optional[Callable] = None,
    max_residues: int = 50000,
    max_seq_len: int = 30000,
    max_batch: int = 10000,
    chunk_len: int = DEFAULT_CHUNK_LEN,
    output_probs: bool = True,
    save_per_residue_embeddings: bool = False,
    save_per_protein_embeddings: bool = False,
    pad_multiple: int = 64,
    desc: str = "Predicting 3Di + 12st",
) -> ModernProstResult:
    """Run ModernProst inference across one or more devices.

    With a single device this loads the model and calls
    :func:`run_modernprost_inference` directly. With several, the pre-sorted
    ``seq_dict`` is sharded round-robin, one worker thread per device runs its
    shard, and the results are merged.

    As in the ProstT5 dispatcher, all N model copies are loaded **sequentially
    on the main thread** before any worker starts: HuggingFace's
    ``from_pretrained`` is not thread-safe, and concurrent loads of the same
    checkpoint can hand the second thread a model whose parameters live on the
    ``meta`` device, so the subsequent ``.to(device)`` raises
    "Cannot copy out of meta tensor". Loads are serialised (~N x single-load
    wall clock); inference stays fully parallel.

    Args:
        seq_dict: Pre-sorted (descending by length) ``(seq_id, sequence, length)``.
        devices: Device strings from :func:`pholdlib.prostt5.device.parse_gpus`.
        model_dir / model_name / half_precision / threads / check_fn / zenodo_fn:
            passed to :func:`get_modernprost_model`.
        Everything else is passed straight through to
        :func:`run_modernprost_inference`.

    Returns:
        A merged :class:`ModernProstResult`. Dict ordering is not guaranteed;
        callers must re-order using their own key list.

    Raises:
        ValueError: if ``devices`` is empty.
        RuntimeError: if any input sequence is neither predicted nor recorded
            as a failure — that would otherwise surface downstream as a
            silently masked CDS.
    """
    if not devices:
        raise ValueError("devices must contain at least one entry")

    from pholdlib.modernprost.model import count_parameters, get_modernprost_model

    common = dict(
        task=task,
        max_residues=max_residues,
        max_seq_len=max_seq_len,
        max_batch=max_batch,
        chunk_len=chunk_len,
        output_probs=output_probs,
        save_per_residue_embeddings=save_per_residue_embeddings,
        save_per_protein_embeddings=save_per_protein_embeddings,
        pad_multiple=pad_multiple,
    )

    # ── single-device fast path ─────────────────────────────────────────────
    if len(devices) == 1:
        dev = devices[0]
        is_cpu = dev == "cpu"
        model, tokenizer, torch_device = get_modernprost_model(
            model_dir,
            model_name,
            cpu=is_cpu,
            threads=threads,
            check_fn=check_fn,
            zenodo_fn=zenodo_fn,
            device=None if is_cpu else dev,
            half_precision=half_precision,
        )
        logger.info(f"ModernProst total parameters: {count_parameters(model):,}")
        return run_modernprost_inference(
            seq_dict, model, tokenizer, torch_device, desc=desc, **common
        )

    # ── multi-GPU path ──────────────────────────────────────────────────────
    n_devices = len(devices)
    shards = shard_round_robin(seq_dict, n_devices)
    logger.info(
        f"Multi-GPU ModernProst inference across {n_devices} device(s): {devices}. "
        f"Total sequences: {len(seq_dict)}; per-shard sizes: "
        f"{[len(s) for s in shards]} (round-robin after global length-sort)."
    )

    logger.info(f"Loading {n_devices} model instances (sequential, main thread)…")
    loaded = []
    for dev in devices:
        is_cpu = dev == "cpu"
        model, tokenizer, torch_device = get_modernprost_model(
            model_dir,
            model_name,
            cpu=is_cpu,
            threads=threads,
            check_fn=check_fn,
            zenodo_fn=zenodo_fn,
            device=None if is_cpu else dev,
            half_precision=half_precision,
        )
        loaded.append((dev, model, tokenizer, torch_device))

    progress_lock = threading.Lock()
    pbar = tqdm(total=len(seq_dict), desc=desc)

    def _on_batch(n: int) -> None:
        with progress_lock:
            pbar.update(n)

    def _worker(dev, model, tokenizer, torch_device, shard) -> ModernProstResult:
        return run_modernprost_inference(
            shard,
            model,
            tokenizer,
            torch_device,
            desc=f"{desc} [{dev}]",
            progress_callback=_on_batch,
            **common,
        )

    merged = ModernProstResult()
    try:
        with ThreadPoolExecutor(max_workers=n_devices) as pool:
            futures = {
                pool.submit(_worker, dev, model, tokenizer, torch_device, shard): dev
                for (dev, model, tokenizer, torch_device), shard in zip(loaded, shards)
                if shard  # skip empty shards (rare: len(seq_dict) < n_devices)
            }
            for fut in as_completed(futures):
                dev = futures[fut]
                try:
                    merged.update(fut.result())
                except Exception:
                    logger.exception(f"Worker on {dev} failed")
                    raise
    finally:
        pbar.close()

    # ── completeness check ──────────────────────────────────────────────────
    # Every input seq_id must appear either as a prediction/profile or in
    # fail_ids. A silent drop would make downstream code mask the CDS as
    # "XXXX…" with no explanation of why.
    accounted = (
        set(merged.predictions_3di) | set(merged.profiles_3di) | set(merged.fail_ids)
    )
    dropped = {sid for sid, _, _ in seq_dict} - accounted
    if dropped:
        raise RuntimeError(
            f"Multi-GPU ModernProst inference dropped {len(dropped)} sequence(s) "
            f"without recording them in fail_ids — likely a bug in batch error "
            f"handling. First few: {sorted(dropped)[:10]}"
        )

    return merged
