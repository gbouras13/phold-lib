#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Device selection, parsing and sharding utilities for multi-GPU ProstT5 inference.

This module contains *pure* helper functions — no torch model loading,
no subprocess calls. Centralising them here keeps the device-resolution
policy in one place and lets phold/baktfold call it identically.

The high-level rule (see plan luminous-rolling-pie.md §1):
    --cpu          > --gpus (explicit)  > auto-detect all CUDA > mps > xpu > cpu

`--gpus` only applies when CUDA is available; MPS and XPU stay single-device.
"""

from typing import List, Optional, Tuple, TypeVar

import torch
from loguru import logger


T = TypeVar("T")


# ─────────────────────────────────────────────────────────────────────────────
# GPU enumeration
# ─────────────────────────────────────────────────────────────────────────────

def get_available_cuda_devices() -> List[str]:
    """Return all visible CUDA devices as torch device strings.

    Returns an empty list if CUDA is unavailable or no devices are found.
    Respects ``CUDA_VISIBLE_DEVICES`` because ``torch.cuda.device_count()``
    already accounts for it.
    """
    if not torch.cuda.is_available():
        return []
    n = torch.cuda.device_count()
    return [f"cuda:{i}" for i in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# --gpus parsing & device-list resolution
# ─────────────────────────────────────────────────────────────────────────────

def parse_gpus(cpu: bool, gpus: Optional[str]) -> List[str]:
    """Resolve the final torch device list to use for inference.

    Args:
        cpu:  ``--cpu`` flag from the CLI. When True, always returns ``["cpu"]``
              regardless of ``gpus`` (with a warning if gpus was explicit).
        gpus: Raw ``--gpus`` CLI string, e.g. ``"0,2"`` or ``None``.
              ``None``/empty → auto-detect.

    Returns:
        A list of torch device strings such as ``["cuda:0", "cuda:1"]``,
        ``["mps"]``, ``["xpu:0"]``, or ``["cpu"]``. The list is never empty.

    Raises:
        ValueError: if ``gpus`` contains a non-integer entry or an index
                    that exceeds ``torch.cuda.device_count()``.
    """
    # 1. --cpu wins
    if cpu:
        if gpus:
            logger.warning(
                f"--cpu was set; ignoring --gpus={gpus!r}"
            )
        return ["cpu"]

    # 2. explicit --gpus
    if gpus:
        if not torch.cuda.is_available():
            raise ValueError(
                f"--gpus={gpus!r} was set but CUDA is not available on this machine. "
                "Remove --gpus, or pass --cpu to force CPU inference."
            )
        n_available = torch.cuda.device_count()
        try:
            idxs = [int(s.strip()) for s in gpus.split(",") if s.strip() != ""]
        except ValueError as e:
            raise ValueError(
                f"--gpus={gpus!r} must be a comma-separated list of integers (e.g. '0,2')."
            ) from e
        if not idxs:
            raise ValueError(f"--gpus={gpus!r} parsed to an empty list.")
        for i in idxs:
            if i < 0 or i >= n_available:
                raise ValueError(
                    f"--gpus={gpus!r} refers to CUDA device {i}, but only "
                    f"{n_available} device(s) are visible (valid indices: "
                    f"0..{n_available - 1})."
                )
        # de-dup while preserving order
        seen = set()
        unique = [i for i in idxs if not (i in seen or seen.add(i))]
        return [f"cuda:{i}" for i in unique]

    # 3. auto-detect: prefer CUDA (all), then MPS, then XPU, else CPU
    cuda_devs = get_available_cuda_devices()
    if cuda_devs:
        return cuda_devs
    if torch.backends.mps.is_available():
        return ["mps"]
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return ["xpu:0"]

    logger.warning("No GPU detected, but --cpu was not specified. Falling back to CPU.")
    return ["cpu"]


# ─────────────────────────────────────────────────────────────────────────────
# Foldseek subprocess env helper
# ─────────────────────────────────────────────────────────────────────────────

def cuda_visible_devices_value(devices: List[str]) -> Optional[str]:
    """Return the ``CUDA_VISIBLE_DEVICES`` string for a foldseek subprocess.

    Args:
        devices: Output of :func:`parse_gpus`.

    Returns:
        ``"0,2"`` (etc.) for a CUDA device list, or ``None`` when the device
        list contains no CUDA entries (CPU / MPS / XPU). Foldseek's GPU mode
        is CUDA-only, so non-CUDA lists need no env override.
    """
    idxs = [d.split(":", 1)[1] for d in devices if d.startswith("cuda:")]
    if not idxs:
        return None
    return ",".join(idxs)


# ─────────────────────────────────────────────────────────────────────────────
# Sharding
# ─────────────────────────────────────────────────────────────────────────────

def shard_round_robin(seq_dict: List[T], n: int) -> List[List[T]]:
    """Split ``seq_dict`` into ``n`` shards by round-robin interleaving.

    The input is assumed to already be sorted (typically length-descending).
    Round-robin preserves that ordering inside each shard while balancing
    total residues per shard.

    Example::

        >>> shard_round_robin(['A','B','C','D','E'], 2)
        [['A', 'C', 'E'], ['B', 'D']]

    Args:
        seq_dict: List of sequence tuples ``(seq_id, sequence, length)``,
                  pre-sorted descending by length.
        n: Number of shards. Must be ≥ 1.

    Returns:
        A list of ``n`` shards. Empty shards may occur if ``len(seq_dict) < n``.
    """
    if n < 1:
        raise ValueError(f"n must be ≥ 1, got {n}")
    return [seq_dict[i::n] for i in range(n)]
