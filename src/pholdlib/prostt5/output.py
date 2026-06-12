#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared output utilities for ProstT5 predictions."""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger


SS_MAPPING = {
    0: "A", 1: "C", 2: "D", 3: "E", 4: "F",
    5: "G", 6: "H", 7: "I", 8: "K", 9: "L",
    10: "M", 11: "N", 12: "P", 13: "Q", 14: "R",
    15: "S", 16: "T", 17: "V", 18: "W", 19: "Y",
    20: "X",  # masked low-confidence residue
}


def mask_low_confidence_aa(sequence: str, scores, threshold: float = 0.5) -> str:
    """Replace amino acids whose ProstT5 confidence score is below *threshold* with 'X'.

    *scores* is a numpy array of shape (1, L) or (L,), or an equivalent nested list.
    Uses a numpy byte-mask — O(L) with no per-residue Python loop or float() call.
    """
    score_arr = np.asarray(scores, dtype=np.float64).ravel()
    mask = score_arr < threshold
    if not mask.any():
        return sequence
    buf = np.frombuffer(bytearray(sequence, "ascii"), dtype=np.uint8).copy()
    buf[mask] = 88  # ord("X")
    return buf.tobytes().decode("ascii")


def write_probs(
    predictions: Dict[str, Tuple],
    output_path_mean: Path,
    output_path_all: Optional[Path],
    original_keys: List[str],
) -> None:
    """
    Write ProstT5 mean and per-residue probabilities to disk.

    Args:
        predictions: Flat dict {seq_id: (labels, mean_prob, all_probs)}.
                     Callers with nested (contig -> seq) dicts should flatten
                     before calling.
        output_path_mean: CSV output path — one "seq_id,mean_prob" line per sequence.
        output_path_all: JSONL output path for per-residue probabilities.
                         Pass None to skip.
        original_keys: Sequence IDs in the order they should appear in the output.
    """
    with open(output_path_mean, "w+") as out_f:
        for seq_id in original_keys:
            if seq_id not in predictions:
                logger.warning(f"Missing ProstT5 mean confidence for {seq_id}")
                continue
            _, mean_prob, _ = predictions[seq_id]
            out_f.write(f"{seq_id},{mean_prob}\n")

    if output_path_all is None:
        return

    with open(output_path_all, "w+") as out_f:
        for seq_id in original_keys:
            if seq_id not in predictions:
                logger.warning(f"Missing ProstT5 confidence for {seq_id}")
                continue
            _, _, all_probs = predictions[seq_id]
            all_probs = all_probs * 100
            if isinstance(all_probs, np.ndarray):
                all_probs_list = all_probs.flatten().tolist()
            else:
                all_probs_list = all_probs
            out_f.write(
                json.dumps({"seq_id": seq_id, "probability": [round(p, 2) for p in all_probs_list]})
                + "\n"
            )


def write_fail_ids(fail_ids: List[str], out_path: Path) -> None:
    """Write a list of failed sequence IDs to a TSV file."""
    if not fail_ids:
        return
    with open(out_path, "w", newline="") as f:
        csv.writer(f, delimiter="\t").writerows([[str(i)] for i in fail_ids])
