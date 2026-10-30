#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ModernProst model and tokenizer loading.

ModernProst checkpoints are HuggingFace ``custom_code`` models: their
``config.json`` maps ``AutoModel`` / ``AutoConfig`` / ``AutoTokenizer`` onto
``modeling_modernprost_3di12st.py`` and friends inside the repo, so loading
them requires ``trust_remote_code=True``.

The loaded model is a ``ModernProst3Di12StModel``. Its forward returns a
``ModernProst3Di12StOutput`` whose ``logits`` is a dict::

    {"3di": (B, L, 20), "12st": (B, L, 12)}

alongside ``last_hidden_state`` (B, L, H), which is post-final-norm and is what
both heads were trained on directly.

Tokenisation is deliberately plain: one vocab id per residue, right-padded with
``<pad>``, and **no BOS/EOS is ever added** (the tokenizer overrides
``build_inputs_with_special_tokens`` to a no-op). So unlike ProstT5 — which
needs an ``<AA2fold>`` prefix token stripped from the output — logit position
*i* corresponds to residue *i* with no offset.
"""

import contextlib
import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from loguru import logger
from transformers import AutoModel, AutoTokenizer, logging, utils

from pholdlib.databases.modernprost import resolve_modernprost_model


def get_modernprost_model(
    model_dir: Path,
    model_name: str,
    cpu: bool,
    threads: int,
    check_fn: Optional[Callable] = None,
    zenodo_fn: Optional[Callable] = None,
    device: Optional[str] = None,
    half_precision: bool = False,
) -> Tuple[AutoModel, AutoTokenizer, torch.device]:
    """
    Load a ModernProst model and tokenizer, downloading if necessary.

    Mirrors :func:`pholdlib.prostt5.model.get_T5_model` so that callers can
    swap between the ProstT5 and ModernProst backends with the same argument
    shape, with two additions: there is no separate CNN checkpoint to load
    (the 3Di and 12st heads are part of the model), and ``half_precision`` is
    applied here rather than by the caller, because the cast must happen before
    the model is handed out.

    Args:
        model_dir: Directory where the model is cached (HuggingFace cache root).
        model_name: Short registry name (e.g. ``"modernprost-50M"``) or full
            HuggingFace id (e.g. ``"gbouras13/modernprost-50M"``).
        cpu: Force CPU even when a GPU is available.
        threads: Number of CPU threads for torch.
        check_fn: Optional callable ``(model_dir, model_name) -> bool``
            returning True when the model needs to be (re)downloaded. Provided
            by the calling tool so tool-specific MD5 dicts can be used.
        zenodo_fn: Optional callable ``(model_dir, logdir, threads)`` that
            downloads the model from a Zenodo backup, called when the
            HuggingFace download fails.
        device: Optional torch device string (e.g. ``"cuda:1"``). When set,
            bypasses auto-detection and forces the model onto this device —
            used by the multi-GPU dispatcher to put one model copy per GPU.
            ``cpu`` overrides ``device`` when True.
        half_precision: Cast the model to fp16 after loading. Silently ignored
            on CPU, where half-precision ops are slow or unimplemented (see
            phold issue #123).

    Returns:
        ``(model, tokenizer, device)`` — the device is returned explicitly so
        callers never need a module-level global.
    """
    spec = resolve_modernprost_model(model_name)
    hf_name = spec.hf_name

    torch.set_num_threads(threads)

    if cpu:
        torch_device = torch.device("cpu")
        dev_name = "cpu"
    elif device is not None:
        # Caller has already resolved which device to use (multi-GPU path).
        torch_device = torch.device(device)
        dev_name = device
    else:
        if torch.cuda.is_available():
            torch_device = torch.device("cuda:0")
            dev_name = "cuda:0"
        elif torch.backends.mps.is_available():
            torch_device = torch.device("mps")
            dev_name = "mps"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            torch_device = torch.device("xpu:0")
            dev_name = "xpu:0"
        else:
            torch_device = torch.device("cpu")
            dev_name = "cpu"
            logger.warning("No available GPU found, but --cpu was not specified")
            logger.warning("ModernProst will be run with CPU only")

    logger.info(f"Using device: {dev_name}")
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading {hf_name} ({spec.approx_params} parameters) from: {model_dir}")
    logger.info(f"If {hf_name} is not found in {model_dir}, it will be downloaded")

    localfile = True
    need_download = False

    if check_fn is not None:
        need_download = check_fn(model_dir, hf_name)
        if need_download:
            localfile = False
            logger.info(
                f"{hf_name} not found or corrupt. Downloading from Hugging Face "
                f"(~{spec.download_mb} MB)"
            )

    logging.set_verbosity_error()
    utils.logging.disable_progress_bar()

    try:
        model = _from_pretrained_quiet(
            hf_name,
            model_dir=model_dir,
            force_download=need_download,
            local_files_only=localfile,
        ).to(torch_device)

    except Exception as e:
        if need_download and zenodo_fn is not None:
            logger.warning("Hugging Face download failed. Trying Zenodo backup.")
            logdir = Path(model_dir) / "logdir"
            zenodo_fn(model_dir, logdir, threads)
            model = _from_pretrained_quiet(
                hf_name,
                model_dir=model_dir,
                force_download=False,
                local_files_only=True,
            ).to(torch_device)
        else:
            raise RuntimeError(f"ModernProst model loading failed: {e}") from e

    tokenizer = AutoTokenizer.from_pretrained(
        hf_name,
        trust_remote_code=True,
        cache_dir=f"{model_dir}/",
    )

    if half_precision:
        if torch_device.type == "cpu":
            # Matches the ProstT5 path: fp16 on CPU is slow and some ops are
            # unimplemented. Keep fp32 rather than failing deep inside a batch.
            logger.info("CPU device — keeping ModernProst in full precision.")
        else:
            model = model.half()

    model = model.eval()
    logger.info(f"{hf_name} loaded")

    return model, tokenizer, torch_device


def _from_pretrained_quiet(
    hf_name: str,
    model_dir: Path,
    force_download: bool,
    local_files_only: bool,
) -> AutoModel:
    """``AutoModel.from_pretrained`` with the remote code's stdout suppressed.

    The custom modeling module prints architecture/config banners on import and
    construction. Those land in the middle of phold's loguru output and look
    like errors to users, so stdout is redirected for the duration of the load.
    stderr is left alone — real warnings and tracebacks must still surface.
    """
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        return AutoModel.from_pretrained(
            hf_name,
            trust_remote_code=True,
            cache_dir=f"{model_dir}/",
            force_download=force_download,
            local_files_only=local_files_only,
        )


def count_parameters(model) -> int:
    """Total number of parameters, for a one-line log after loading."""
    return sum(p.numel() for p in model.parameters())
