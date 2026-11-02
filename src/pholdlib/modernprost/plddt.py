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

The predicted pLDDT is used to mask low-confidence residues of the query DBs
(``write_profile_foldseek_dbs`` / ``generate_combined_foldseek_db`` with
``plddt=..., plddt_threshold=...``).

Normally the head runs inside the main inference pass:
``run_modernprost_inference(..., plddt_head=head)`` returns ``result.plddt``
next to the 3Di / 12-state outputs, at the cost of the head alone. The layer
mix is accumulated layer by layer, already projected to the head's width
(:meth:`PlddtHead.layer_term`), so no per-layer or hidden-size-wide state is
kept. In the packed engine, where several proteins share
one row, the convolutions take the per-token protein id (``seg``) and never
read across a protein boundary, which gives exactly the per-protein result.
:func:`predict_plddt` is a standalone pass for callers that already have
predictions.

The module classes are the training script's
(``foldseek2-plm/plddt_head/train_plddt_head.py``) with the same parameters, so
the saved state dicts load as they are; ``seg`` is an addition that changes
nothing when absent.
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


def _same_segment(seg, shift):
    """(B, L) bool: is position ``t + shift`` in the same protein as ``t`` (False off the ends)?"""
    L = seg.shape[1]
    out = torch.zeros_like(seg, dtype=torch.bool)
    if shift >= 0:
        out[:, : L - shift] = seg[:, shift:] == seg[:, : L - shift]
    else:
        out[:, -shift:] = seg[:, : L + shift] == seg[:, -shift:]
    return out


class DepthwiseConv1d(nn.Module):
    """Same params and output as nn.Conv1d(d, d, k, padding=k//2, groups=d) for odd k, on (B, L, d)
    input, computed as a sum of k shifted copies (MIOpen fails on depthwise conv1d on ROCm).

    With ``seg`` (B, L), a per-token protein id, neighbours from another protein count as zero padding,
    so a packed row gives each protein exactly its own padded convolution."""

    def __init__(self, d, kernel):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d, 1, kernel))
        self.bias = nn.Parameter(torch.empty(d))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.uniform_(self.bias, -1 / math.sqrt(kernel), 1 / math.sqrt(kernel))

    def forward(self, x, seg=None):
        k, L = self.weight.shape[-1], x.shape[1]
        xp = F.pad(x, (0, 0, k // 2, k // 2))
        w = self.weight[:, 0, :]
        if seg is None:
            return self.bias + sum(xp[:, j : j + L] * w[:, j] for j in range(k))
        return self.bias + sum(
            xp[:, j : j + L] * w[:, j] * _same_segment(seg, j - k // 2).unsqueeze(-1).to(x.dtype)
            for j in range(k)
        )


class ConvBlock(nn.Module):
    """ConvNeXt-style: depthwise conv (local context) -> LN -> MLP, residual."""

    def __init__(self, d, kernel, expand=2):
        super().__init__()
        self.dwconv = DepthwiseConv1d(d, kernel)
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, expand * d)
        self.fc2 = nn.Linear(expand * d, d)

    def forward(self, x, mask, seg=None):
        h = self.dwconv(x * mask, seg)
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

    def forward(self, x, mask, seg=None):
        B, L, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(B, L, 3, self.n_heads, d // self.n_heads).permute(2, 0, 3, 1, 4)
        keep = mask[..., 0].bool()[:, None, None, :]  # (B, 1, 1, L): attend to real residues only
        if seg is not None:
            keep = keep & (seg[:, None, :, None] == seg[:, None, None, :])
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
        return self.from_mixed(self.mix(hidden_states), attention_mask)

    def from_mixed(self, mixed, attention_mask, seg=None):
        """Logits from the layer mix (``self.mix`` output). ``seg``: per-token protein id for packed rows."""
        return self.from_projected(F.linear(mixed, self.proj.weight), attention_mask, seg)

    def layer_term(self, h, w):
        """One layer's share of the projected mix: ``w * layer_norm(h) @ proj.weight.T``, (B, L, d) float32.

        ``proj`` is linear, so ``proj(sum_i w_i LN(h_i)) = sum_i layer_term(h_i, w_i) + proj.bias``:
        accumulating these needs a d-wide buffer (128) instead of a hidden-size one (448 / 1536).
        The layer norm is fp32; with a half-precision backbone the projection runs in its dtype (tensor
        cores, fp32 accumulation) and the term is returned as fp32."""
        x = F.layer_norm(h.float(), h.shape[-1:])
        if h.dtype in (torch.float16, torch.bfloat16):
            return w * F.linear(x.to(h.dtype), self.proj.weight.to(h.dtype)).float()
        return w * F.linear(x, self.proj.weight)

    def from_projected(self, projected, attention_mask, seg=None):
        """Logits from the summed ``layer_term`` outputs (the projected mix without the bias)."""
        mask = attention_mask.unsqueeze(-1).float()
        x = (projected + self.proj.bias) * mask
        for block in self.blocks:
            x = block(x, mask, seg)
        for block in self.attn_blocks:
            x = block(x, mask, seg)
        return self.out(self.norm(x)).float()  # (B, L, n_bins)

    def predict(self, logits):
        if self.n_bins == 1:
            return torch.sigmoid(logits.squeeze(-1)) * 100.0
        return torch.softmax(logits, dim=-1) @ self.bin_centers

    def mix_weights(self) -> Dict[int, torch.Tensor]:
        """{0-based encoder layer index: its mix weight (gamma folded in)}."""
        w = torch.softmax(self.mix.weights, dim=0) * self.mix.gamma
        return {layer - 1: w[i] for i, layer in enumerate(self.layers)}


class LayerMixAccumulator:
    """Forward hooks that build the head's (projected) layer mix while the backbone runs.

    Each hooked layer adds its :meth:`PlddtHead.layer_term` into one float32
    (B, L, d) buffer, ``projected``, so nothing hidden-size wide is kept
    across layers. Finish with ``head.from_projected(acc.projected, mask)``.
    Use as a context manager around the forward; the hooks are removed on exit.
    """

    def __init__(self, model, head: PlddtHead):
        if int(model.config.hidden_size) != head.hidden_size:
            raise ValueError(f"pLDDT head expects hidden size {head.hidden_size}, "
                             f"model has {model.config.hidden_size}")
        self.layers = model.encoder.layers
        self.head = head
        self.projected = None
        self._handles = []

    def _hook(self, w):
        def fn(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            if h.dim() == 4:  # hyper-connection streams (B, L, n, H) -> (B, L, H)
                h = h.sum(dim=2)
            term = self.head.layer_term(h, w)
            self.projected = term if self.projected is None else self.projected + term

        return fn

    def __enter__(self):
        self.projected = None
        with torch.no_grad():
            weights = self.head.mix_weights()
        self._handles = [self.layers[i].register_forward_hook(self._hook(w)) for i, w in weights.items()]
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


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
    """Standalone pLDDT pass (0-100 per residue) for ``{seq_id: amino-acid sequence}``.

    Prefer ``run_modernprost_inference(..., plddt_head=head)``, which predicts
    pLDDT in the same pass as the 3Di / 12-state outputs. This pass tokenises
    and chunks exactly as inference does (``TokenisedItems``: the same token
    table, and proteins longer than *chunk_len*, default ``DEFAULT_CHUNK_LEN``,
    split into near-equal chunks), so both give the same values. Batches are
    length-sorted and hold about *max_residues* padded tokens.
    """
    from pholdlib.modernprost.batching import TokenisedItems, lut_from_tokenizer
    from pholdlib.modernprost.inference import DEFAULT_CHUNK_LEN

    items = TokenisedItems.from_sequences([(k, v, len(v)) for k, v in sequences.items()],
                                          lut_from_tokenizer(tokenizer), chunk_len or DEFAULT_CHUNK_LEN)
    acc = LayerMixAccumulator(model, head)
    out = np.zeros(items.n_tokens, dtype=np.float32)
    order = np.argsort(-items.lengths, kind="stable")
    order = order[items.lengths[order] > 0]
    start = 0
    while start < len(order):
        width = int(items.lengths[order[start]])
        batch = order[start : start + max(1, max_residues // width)]
        start += len(batch)
        ids = np.zeros((len(batch), width), dtype=np.int64)
        for r, k in enumerate(batch):
            o, L = int(items.offsets[k]), int(items.lengths[k])
            ids[r, :L] = items.tokens[o : o + L]
        ids_t = torch.from_numpy(ids).to(device)
        mask = ids_t != 0  # <pad> is id 0
        with acc:
            model.encoder(ids_t, mask)
        pred = head.predict(head.from_projected(acc.projected, mask)).float().cpu().numpy()
        for r, k in enumerate(batch):
            o, L = int(items.offsets[k]), int(items.lengths[k])
            out[o : o + L] = pred[r, :L]
    po = items.protein_offsets
    return {seq_id: out[po[p] : po[p + 1]].copy() for p, seq_id in enumerate(items.seq_ids)}
