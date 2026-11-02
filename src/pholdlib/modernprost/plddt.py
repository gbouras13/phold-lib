#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-residue pLDDT prediction with the ModernProst pLDDT heads.

Each ModernProst repo on Hugging Face ships an optional ``plddt_head/``
subfolder (``head.safetensors`` + ``head_config.json``): a small network
trained on top of the *frozen* backbone. It reads the output of every encoder
layer, mixes them with learned softmax weights, applies two depthwise-conv
blocks and predicts pLDDT as the expected value over 50 bins. The backbone
itself is unchanged, so loading the model is unaffected by the head.

The predicted pLDDT is used to mask low-confidence residues of the ``-pssm``
models' profiles (``write_profile_foldseek_dbs(..., plddt=...,
plddt_threshold=...)``).

The module classes below are copied unchanged from the training script
(``foldseek2-plm/plddt_head/train_plddt_head.py``) so the saved state dicts load
as they are; ``plddt_head/plddt_head.py`` in each Hugging Face repo is the same
code.

The head runs as a separate pass of the padded engine (encoder only, layer
outputs captured with forward hooks): the packed engine concatenates proteins,
and the head's convolutions would read across protein boundaries there.
"""

import json
import math
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from safetensors.torch import load_file

from pholdlib.databases.modernprost import resolve_modernprost_model

HEAD_SUBFOLDER = "plddt_head"
HEAD_FILES = ("head_config.json", "head.safetensors")


class LayerMix(nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(n_layers))
        self.gamma = nn.Parameter(torch.ones(()))

    def forward(self, hidden_states):
        w = torch.softmax(self.weights, dim=0)
        x = sum(w[i] * F.layer_norm(h.float(), h.shape[-1:]) for i, h in enumerate(hidden_states))
        return self.gamma * x


class DepthwiseConv1d(nn.Module):
    """Same params and output as nn.Conv1d(d, d, k, padding=k//2, groups=d) for odd k, on (B, L, d)
    input, computed as a sum of k shifted copies (MIOpen fails on depthwise conv1d on ROCm)."""

    def __init__(self, d, kernel):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d, 1, kernel))
        self.bias = nn.Parameter(torch.empty(d))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.uniform_(self.bias, -1 / math.sqrt(kernel), 1 / math.sqrt(kernel))

    def forward(self, x):
        k, L = self.weight.shape[-1], x.shape[1]
        xp = F.pad(x, (0, 0, k // 2, k // 2))
        w = self.weight[:, 0, :]
        return self.bias + sum(xp[:, j : j + L] * w[:, j] for j in range(k))


class ConvBlock(nn.Module):
    """ConvNeXt-style: depthwise conv (local context) -> LN -> MLP, residual."""

    def __init__(self, d, kernel, expand=2):
        super().__init__()
        self.dwconv = DepthwiseConv1d(d, kernel)
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, expand * d)
        self.fc2 = nn.Linear(expand * d, d)

    def forward(self, x, mask):
        h = self.dwconv(x * mask)
        h = self.fc2(F.gelu(self.fc1(self.norm(h))))
        return (x + h) * mask


class AttnBlock(nn.Module):
    """Pre-norm transformer block: global self-attention over the protein (padding masked) -> MLP."""

    def __init__(self, d, n_heads, expand=2):
        super().__init__()
        assert d % n_heads == 0, f"head_dim={d} must be divisible by attn_heads={n_heads}"
        self.n_heads = n_heads
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.attn_out = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, expand * d)
        self.fc2 = nn.Linear(expand * d, d)

    def forward(self, x, mask):
        B, L, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(B, L, 3, self.n_heads, d // self.n_heads).permute(2, 0, 3, 1, 4)
        keep = mask[..., 0].bool()[:, None, None, :]
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=keep)
        x = x + self.attn_out(h.transpose(1, 2).reshape(B, L, d))
        x = x + self.fc2(F.gelu(self.fc1(self.norm2(x))))
        return x * mask


class PlddtHead(nn.Module):
    def __init__(self, hidden_size, n_layers, d=128, kernel=9, n_blocks=2, n_bins=50, n_attn_layers=0,
                 attn_heads=4):
        super().__init__()
        self.mix = LayerMix(n_layers)
        self.proj = nn.Linear(hidden_size, d)
        self.blocks = nn.ModuleList(ConvBlock(d, kernel) for _ in range(n_blocks))
        self.attn_blocks = nn.ModuleList(AttnBlock(d, attn_heads) for _ in range(n_attn_layers))
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, n_bins)
        self.n_bins = n_bins
        self.register_buffer("bin_centers", (torch.arange(n_bins) + 0.5) * (100.0 / n_bins), persistent=False)

    def forward(self, hidden_states, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        x = self.proj(self.mix(hidden_states)) * mask
        for block in self.blocks:
            x = block(x, mask)
        for block in self.attn_blocks:
            x = block(x, mask)
        return self.out(self.norm(x)).float()  # (B, L, n_bins)

    def predict(self, logits):
        if self.n_bins == 1:
            return torch.sigmoid(logits.squeeze(-1)) * 100.0
        return torch.softmax(logits, dim=-1) @ self.bin_centers


def _head_paths(model_name: str, model_dir: Union[str, Path], local_files_only: bool) -> Dict[str, Path]:
    """Locate the head files: a local model directory, or the repo's ``plddt_head/`` at the pinned revision."""
    local = Path(model_name) / HEAD_SUBFOLDER
    if local.is_dir():
        return {f: local / f for f in HEAD_FILES}
    from huggingface_hub import hf_hub_download

    spec = resolve_modernprost_model(model_name)
    return {
        f: Path(hf_hub_download(spec.hf_name, f, subfolder=HEAD_SUBFOLDER, cache_dir=f"{model_dir}/",
                                revision=spec.revision, local_files_only=local_files_only))
        for f in HEAD_FILES
    }


def load_plddt_head(
    model_name: str,
    model_dir: Union[str, Path],
    device: torch.device,
    local_files_only: bool = False,
) -> PlddtHead:
    """Load the pLDDT head of *model_name* (registry name, HF id or local model dir).

    The head is downloaded into *model_dir* (the same HF cache the model uses)
    and kept in fp32 whatever the backbone's precision.
    """
    paths = _head_paths(model_name, model_dir, local_files_only)
    config = json.loads(paths["head_config.json"].read_text())
    head = PlddtHead(config["hidden_size"], len(config["layers"]), d=config["head_dim"], kernel=config["kernel"],
                     n_blocks=config["n_blocks"], n_bins=config["n_bins"],
                     n_attn_layers=config.get("n_attn_layers", 0), attn_heads=config.get("attn_heads", 4))
    head.load_state_dict(load_file(str(paths["head.safetensors"])))
    head.layers = list(config["layers"])  # 1-based encoder layers the head reads
    head.hidden_size = config["hidden_size"]
    logger.info(f"Loaded pLDDT head for {model_name} ({sum(p.numel() for p in head.parameters()):,} parameters)")
    return head.to(device).eval()


@torch.no_grad()
def predict_plddt(
    model,
    tokenizer,
    head: PlddtHead,
    sequences: Dict[str, str],
    device: torch.device,
    max_residues: int = 16384,
    chunk_len: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Predict per-residue pLDDT (0-100) for ``{seq_id: amino-acid sequence}``.

    Proteins longer than *chunk_len* (default ``DEFAULT_CHUNK_LEN``) are
    predicted in windows and stitched back together. Batches are length-sorted
    and hold about *max_residues* padded tokens.
    """
    from pholdlib.modernprost.inference import DEFAULT_CHUNK_LEN

    if model.config.hidden_size != head.hidden_size:
        raise ValueError(f"pLDDT head expects hidden size {head.hidden_size}, model has {model.config.hidden_size}")
    chunk_len = chunk_len or DEFAULT_CHUNK_LEN

    pieces = []  # (seq_id, window start, window)
    for seq_id, seq in sequences.items():
        seq = seq.upper()
        for begin in range(0, max(len(seq), 1), chunk_len):
            pieces.append((seq_id, begin, seq[begin : begin + chunk_len]))
    pieces.sort(key=lambda p: -len(p[2]))

    captured = {}

    def hook(i):
        def fn(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            captured[i] = h.sum(dim=2) if h.dim() == 4 else h  # hyper-connection streams -> (B, L, H)

        return fn

    layer_idx = [l - 1 for l in head.layers]
    handles = [model.encoder.layers[i].register_forward_hook(hook(i)) for i in layer_idx]
    windows: Dict[str, list] = {seq_id: [] for seq_id in sequences}
    try:
        start = 0
        while start < len(pieces):
            width = max(len(pieces[start][2]), 1)
            n = max(1, max_residues // width)
            batch = pieces[start : start + n]
            start += n
            enc = tokenizer([s for _, _, s in batch], add_special_tokens=False, padding=True, return_tensors="pt")
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            captured.clear()
            model.encoder(ids, mask)
            pred = head.predict(head([captured[i] for i in layer_idx], mask)).float().cpu().numpy()
            for row, (seq_id, begin, s) in enumerate(batch):
                windows[seq_id].append((begin, pred[row, : len(s)]))
    finally:
        for h in handles:
            h.remove()

    out = {}
    for seq_id, seq in sequences.items():
        parts = [p for _, p in sorted(windows[seq_id], key=lambda w: w[0])]
        out[seq_id] = np.concatenate(parts).astype(np.float32)
    return out
