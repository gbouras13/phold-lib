#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Output utilities for ModernProst 3Di + 12-state predictions.

Two alphabets are in play:

* **3Di** — Foldseek's 20-state structural alphabet, ``ACDEFGHIKLMNPQRSTVWY``.
  Identical to the ProstT5 path's :data:`pholdlib.prostt5.output.SS_MAPPING`,
  with class index 20 reserved for a low-confidence masked residue (``X``).
* **12st** — a 12-state secondary-structure alphabet, ``ACDEFGHIKLMN``. It has
  **no mask state**: every residue must carry one of the twelve letters,
  because the combined Foldseek encoding packs both alphabets into one byte
  (``c = 3di_index * 12 + ss12_index``, so ``3di_index`` must stay < 20).

That asymmetry is why :func:`write_3di_fasta` takes a ``mask_threshold`` and
:func:`write_12st_fasta` does not. Masking the 3Di string to ``X`` produces a
FASTA that :func:`pholdlib.modernprost.foldseek_db.generate_combined_foldseek_db`
will reject — deliberately, since the mask cannot be encoded. Callers building
a combined DB should write the 3Di FASTA unmasked and apply masking to the
amino-acid FASTA instead.
"""

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from loguru import logger


THREEDI_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
SS12_ALPHABET = "ACDEFGHIKLMN"

# 3Di class index -> letter. Index 20 is the masked/low-confidence residue and
# is not produced by the model — it is only written by write_3di_fasta() when
# mask_threshold > 0.
THREEDI_MAPPING: Dict[int, str] = {i: ch for i, ch in enumerate(THREEDI_ALPHABET)}
THREEDI_MAPPING[20] = "X"
THREEDI_MASK_INDEX = 20

# 12-state class index -> letter. No mask entry: see the module docstring.
SS12_MAPPING: Dict[int, str] = {i: ch for i, ch in enumerate(SS12_ALPHABET)}


def decode_3di(class_indices: Iterable[int]) -> str:
    """Decode 3Di class indices (0–20) to a 3Di string."""
    return "".join(THREEDI_MAPPING[int(i)] for i in class_indices)


def decode_12st(class_indices: Iterable[int]) -> str:
    """Decode 12-state class indices (0–11) to a 12-state string."""
    return "".join(SS12_MAPPING[int(i)] for i in class_indices)


# ─────────────────────────────────────────────────────────────────────────────
# FASTA writers
# ─────────────────────────────────────────────────────────────────────────────

def write_3di_fasta(
    predictions: Dict[str, Tuple],
    out_path: Path,
    original_keys: List[str],
    header_map: Optional[Dict[str, str]] = None,
    mask_threshold: float = 0,
) -> None:
    """Write 3Di predictions to a FASTA file.

    Args:
        predictions: Flat ``{seq_id: (pred, mean_prob, all_prob)}`` where
            ``pred`` is a byte array of 3Di class indices and ``all_prob`` is a
            float array of per-residue max softmax probabilities in ``[0, 1]``.
        out_path: Output FASTA path.
        original_keys: Sequence IDs in the order they should be written.
        header_map: Optional ``{seq_id: header}`` override. When omitted the
            seq_id is used verbatim — phold passes ``contig_id:cds_id`` headers
            through here.
        mask_threshold: Residues whose probability (0–100 scale) falls below
            this are written as ``X``. Pass 0 (the default) to disable, which
            is required when the FASTA feeds a combined 3Di+12st Foldseek DB.

    Zero-length predictions are skipped (phold issue #47).
    """
    mask_prop = mask_threshold / 100
    written = 0

    with open(out_path, "w+") as out_f:
        for seq_id in original_keys:
            entry = predictions.get(seq_id)
            if entry is None:
                continue
            pred, _, all_prob = entry
            if len(pred) == 0:
                continue

            if mask_prop > 0 and all_prob is not None:
                pred = pred.copy()
                pred[np.asarray(all_prob).ravel() < mask_prop] = THREEDI_MASK_INDEX

            header = header_map.get(seq_id, seq_id) if header_map else seq_id
            out_f.write(f">{header}\n{decode_3di(pred)}\n")
            written += 1

    logger.info(f"Finished writing {written} 3Di sequences to {out_path}")


def write_12st_fasta(
    predictions: Dict[str, Tuple],
    out_path: Path,
    original_keys: List[str],
    header_map: Optional[Dict[str, str]] = None,
) -> None:
    """Write 12-state predictions to a FASTA file.

    Args are as for :func:`write_3di_fasta`, minus ``mask_threshold`` — the
    12-state alphabet has no mask symbol (see the module docstring).
    """
    written = 0

    with open(out_path, "w+") as out_f:
        for seq_id in original_keys:
            entry = predictions.get(seq_id)
            if entry is None:
                continue
            pred, _, _ = entry
            if len(pred) == 0:
                continue

            header = header_map.get(seq_id, seq_id) if header_map else seq_id
            out_f.write(f">{header}\n{decode_12st(pred)}\n")
            written += 1

    logger.info(f"Finished writing {written} 12-state sequences to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Probability writers
# ─────────────────────────────────────────────────────────────────────────────

def write_mean_probs(
    predictions: Dict[str, Tuple],
    output_path: Path,
    original_keys: List[str],
) -> None:
    """Write one ``seq_id,mean_prob`` line per sequence.

    ``mean_prob`` is the mean per-residue max softmax probability as a
    percentage, rounded to 2 dp.
    """
    with open(output_path, "w+") as out_f:
        for seq_id in original_keys:
            if seq_id not in predictions:
                logger.warning(f"Missing ModernProst mean confidence for {seq_id}")
                continue
            _, mean_prob, _ = predictions[seq_id]
            out_f.write(f"{seq_id},{mean_prob}\n")


def write_all_probs(
    predictions: Dict[str, Tuple],
    output_path: Path,
    original_keys: List[str],
) -> None:
    """Write per-residue probabilities as JSONL, one object per sequence.

    Each line is ``{"seq_id": ..., "probability": [...]}`` with probabilities
    on a 0–100 scale rounded to 2 dp, matching the ProstT5 path's format.
    """
    with open(output_path, "w+") as out_f:
        for seq_id in original_keys:
            if seq_id not in predictions:
                logger.warning(f"Missing ModernProst confidence for {seq_id}")
                continue
            _, _, all_probs = predictions[seq_id]
            if all_probs is None:
                continue
            probs = (np.asarray(all_probs) * 100).flatten().tolist()
            out_f.write(
                json.dumps({"seq_id": seq_id, "probability": [round(p, 2) for p in probs]})
                + "\n"
            )


def write_fail_ids(fail_ids: List[str], out_path: Path) -> None:
    """Write a list of failed sequence IDs to a TSV file."""
    if not fail_ids:
        return
    with open(out_path, "w", newline="") as f:
        csv.writer(f, delimiter="\t").writerows([[str(i)] for i in fail_ids])


# ─────────────────────────────────────────────────────────────────────────────
# Human-readable profile writers (pssm task)
# ─────────────────────────────────────────────────────────────────────────────

def format_profile(profile: np.ndarray, alphabet: str) -> str:
    """Format one ``(L, K)`` probability matrix as a fixed-width text block.

    Byte-compatible with foldseek2-plm's ``output_profile_3di.txt`` /
    ``output_profile_12st.txt``, which is also the input format understood by
    :func:`pholdlib.modernprost.foldseek_db.parse_profiles`.
    """
    header = " ".join(f"{state:>6}" for state in alphabet)
    rows = [" ".join(f"{float(value):.4f}" for value in row) for row in profile]
    return "\n".join([header] + rows)


def write_profiles_text(
    profiles: Dict[str, np.ndarray],
    out_path: Path,
    alphabet: str,
    original_keys: List[str],
) -> None:
    """Write per-residue profiles as human-readable text blocks.

    Args:
        profiles: Flat ``{seq_id: ndarray (L, K)}`` of softmax probabilities.
        out_path: Output text path.
        alphabet: ``THREEDI_ALPHABET`` or ``SS12_ALPHABET``; must match the
            number of columns in each profile.
        original_keys: Sequence IDs in the order they should be written.
    """
    with open(out_path, "w") as fh:
        for seq_id in original_keys:
            profile = profiles.get(seq_id)
            if profile is None:
                continue
            print(f"Query profile of sequence {seq_id}", file=fh)
            print(format_profile(profile, alphabet), file=fh)
            print("", file=fh)

    logger.info(f"Finished writing profile predictions to {out_path}")
