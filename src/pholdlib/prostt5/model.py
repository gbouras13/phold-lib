#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ProstT5 model definitions and loading utilities.

Code adapted from @mheinzinger
https://github.com/mheinzinger/ProstT5/blob/main/scripts/predict_3Di_encoderOnly.py
"""

from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import torch
from loguru import logger
from torch import nn
from transformers import T5EncoderModel, T5Tokenizer, logging, utils


def _accelerate_available() -> bool:
    """Whether HuggingFace ``accelerate`` is importable.

    ``accelerate`` is a hard dependency of pholdlib, but check at runtime
    so a broken/incomplete install degrades gracefully rather than crashing
    at import time.
    """
    try:
        import accelerate  # noqa: F401
        return True
    except ImportError:
        return False


class CNN(nn.Module):
    """Two-layer convolutional prediction head used on top of ProstT5 embeddings."""

    def __init__(self):
        super(CNN, self).__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(1024, 32, kernel_size=(7, 1), padding=(3, 0)),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Conv2d(32, 20, kernel_size=(7, 1), padding=(3, 0)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B x L x F) — batch, sequence length, embedding dim (1024)
        Returns:
            (B x N x L) — logits over 20 3Di classes per residue
        """
        x = x.permute(0, 2, 1).unsqueeze(dim=-1)   # (B x F x L x 1)
        Yhat = self.classifier(x)                    # (B x N x L x 1)
        return Yhat.squeeze(dim=-1)                  # (B x N x L)


def get_T5_model(
    model_dir: Path,
    model_name: str,
    cpu: bool,
    threads: int,
    check_fn: Optional[Callable] = None,
    zenodo_fn: Optional[Callable] = None,
    device: Optional[str] = None,
) -> Tuple[T5EncoderModel, T5Tokenizer, torch.device]:
    """
    Load a ProstT5 encoder and tokenizer, downloading if necessary.

    Args:
        model_dir: Directory where the model is cached.
        model_name: HuggingFace model identifier (e.g. "Rostlab/ProstT5_fp16").
        cpu: Force CPU even when a GPU is available.
        threads: Number of CPU threads for torch.
        check_fn: Optional callable(model_dir, model_name) -> bool that returns
                  True when the model needs to be (re)downloaded. Provided by the
                  calling tool so that tool-specific MD5 dicts can be used.
        zenodo_fn: Optional callable(model_dir, logdir, threads) that downloads
                   the model from a Zenodo backup. Called when the HuggingFace
                   download fails. Provided by the calling tool.
        device: Optional torch device string (e.g. ``"cuda:1"``). When set,
                bypasses the auto-detection chain and forces the model onto
                this device — used by the multi-GPU dispatcher to put one
                model copy per GPU. ``cpu`` overrides ``device`` when True.

    Returns:
        (model, vocab, device) — the device is returned explicitly so callers
        never need a module-level global.
    """
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
        elif torch.xpu.is_available():
            # Intel XPU (Data Center GPU / Arc) support — PR #129
            torch_device = torch.device("xpu:0")
            dev_name = "xpu:0"
        else:
            torch_device = torch.device("cpu")
            dev_name = "cpu"
            logger.warning("No available GPU found, but --cpu was not specified")
            logger.warning("ProstT5 will be run with CPU only")

    # Rebind for the rest of the function (existing code uses `device`).
    device = torch_device

    logger.info(f"Using device: {dev_name}")
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading T5 from: {model_dir}/{model_name}")
    logger.info(f"If {model_dir}/{model_name} is not found, it will be downloaded")

    localfile = True
    need_download = False

    if check_fn is not None:
        need_download = check_fn(model_dir, model_name)
        if need_download:
            localfile = False
            logger.info("ProstT5 not found or corrupt. Downloading from Hugging Face")

    logging.set_verbosity_error()
    utils.logging.disable_progress_bar()

    # ── pick a load path ────────────────────────────────────────────────────
    # Two known-good strategies, each with different trade-offs:
    #
    #   FAST PATH  (preferred): pass ``device_map=str(device)`` to
    #   ``from_pretrained``. Requires the ``accelerate`` library. Loads
    #   weights directly onto the target device via mmap'd safetensors —
    #   no CPU intermediate, no meta-tensor materialisation issue. ~30 s
    #   for ProstT5 on a GPU.
    #
    #   SLOW PATH  (fallback): pass ``low_cpu_mem_usage=False`` and call
    #   ``.to(device)`` after. Works without accelerate and on every
    #   transformers version, but allocates the full model in CPU RAM as
    #   fp32 first then copies to the device. Can take ~10 min for ProstT5
    #   on a network filesystem.
    #
    # ``accelerate`` is a hard dependency of pholdlib but the check is at
    # runtime so a broken install degrades to "slow but correct" instead
    # of crashing.
    use_fast_path = device.type == "cuda" and _accelerate_available()
    if use_fast_path:
        load_kwargs = dict(
            cache_dir=f"{model_dir}/",
            force_download=need_download,
            local_files_only=localfile,
            device_map=str(device),
        )
    else:
        if device.type == "cuda":
            logger.warning(
                "accelerate is not installed; falling back to the slow ProstT5 "
                "load path (~10 min instead of ~30 s on GPU). Install with "
                "`pip install accelerate` to restore fast loading."
            )
        load_kwargs = dict(
            cache_dir=f"{model_dir}/",
            force_download=need_download,
            local_files_only=localfile,
            low_cpu_mem_usage=False,
        )

    def _do_load(extra: dict) -> T5EncoderModel:
        kwargs = {**load_kwargs, **extra}
        m = T5EncoderModel.from_pretrained(model_name, **kwargs)
        # device_map already places the model on the target device, so .to()
        # is unnecessary (and slightly wasteful) in the fast path.
        if not use_fast_path:
            m = m.to(device)
        return m

    try:
        model = _do_load({})

    except Exception as e:
        if need_download and zenodo_fn is not None:
            logger.warning("Hugging Face download failed. Trying Zenodo backup.")
            logdir = Path(model_dir) / "logdir"
            zenodo_fn(model_dir, logdir, threads)
            model = _do_load({"force_download": False, "local_files_only": True})
        else:
            raise RuntimeError(f"ProstT5 model loading failed: {e}") from e

    vocab = T5Tokenizer.from_pretrained(
        model_name,
        cache_dir=f"{model_dir}/",
        do_lower_case=False,
        use_fast=False,
    )

    if cpu:
        # Force FP32 on CPU — avoids errors with half-precision ops on CPU
        # See https://github.com/gbouras13/baktfold/issues/28
        #     https://github.com/gbouras13/phold/issues/123
        model = model.float()

    model = model.eval()
    logger.info(f"{model_name} loaded")

    return model, vocab, device


def load_predictor(checkpoint_path: Union[str, Path], device: torch.device) -> CNN:
    """
    Load CNN prediction head weights from a checkpoint file.

    Args:
        checkpoint_path: Path to the .pt or state-dict checkpoint.
        device: Target device (returned by get_T5_model).

    Returns:
        CNN model in eval mode on the given device.
    """
    model = CNN()
    state = torch.load(checkpoint_path, map_location=device)

    if Path(checkpoint_path).suffix == ".pt":
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)

    return model.eval().to(device)


def device_synchronize(device: torch.device) -> None:
    """Synchronize the given device — handles CUDA, MPS, XPU and CPU safely.

    Used by autotune timing loops to get accurate wall-clock measurements
    per batch without relying on device-specific synchronize calls at each
    call site.  PR #129 added XPU support; this centralises the logic so
    phold/autotune.py and baktfold/autotune.py can both import it.
    """
    dtype = device.type
    if dtype == "cuda":
        torch.cuda.synchronize(device)
    elif dtype == "mps":
        torch.mps.synchronize()
    elif dtype == "xpu":
        torch.xpu.synchronize(device)
    # cpu: no-op


def toCPU(tensor: torch.Tensor):
    """Move a tensor to CPU and return a NumPy array, squeezing trailing singleton dims."""
    if len(tensor.shape) > 1:
        return tensor.detach().cpu().squeeze(dim=-1).numpy()
    return tensor.detach().cpu().numpy()
