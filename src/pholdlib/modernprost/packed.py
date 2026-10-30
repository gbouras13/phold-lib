#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sequence-packed ModernProst forward pass using FlexAttention.

Many proteins are concatenated into one row of a fixed number of tokens.
Positions restart at 0 for every protein, and attention is restricted so that
the packed row computes exactly what one-protein-per-row inference computes:

* global layers: a token attends to every token of its own protein;
* sliding layers: a token attends to tokens of its own protein within
  ``|i - j| <= sliding_window``.

The masks are expressed as a FlexAttention ``BlockMask`` built directly from
protein boundaries, so no dense mask is materialised and fully masked
128 x 128 tiles are skipped. That gives the sliding layers their real
O(L x window) cost, where the Hugging Face model, which passes a dense L x L
boolean mask to SDPA, pays O(L^2) in every layer.

Because every batch has the same shape, the whole forward can be compiled
once with ``torch.compile`` and reused; that fusion of the many small fp32
elementwise ops (RoPE, LayerNorm, SwiGLU, NOBLE) is where most of the speed
comes from on a model this small.

The forward reuses the loaded model's own modules and its own ``_apply_rope``,
so the weights and numerics are those of the checkpoint; only attention
masking and position ids differ from the padded path. In fp32 the two agree
to ~1e-6.
"""

import inspect
import sys
from typing import Callable, Optional, Tuple

import numpy as np
import torch

from pholdlib.modernprost.batching import TokenisedItems

# Size of the FlexAttention mask tiles. Proteins and the sliding window are
# mapped onto tiles of this many query x key tokens.
DEFAULT_BLOCK = 128

# Config switches the packed forward does not implement. Checkpoints that set
# any of them fall back to the padded engine.
_UNSUPPORTED_FLAGS = ("use_diff_attn", "use_attn_gate", "value_residual", "use_qk_norm")

# Tests set this to exercise the packed forward on CPU with eager
# FlexAttention (correct but far too slow for real inputs).
_ALLOW_CPU = False


def packed_engine_status(device: torch.device, model=None) -> Tuple[bool, str]:
    """Whether the packed engine can run here, and if not, why.

    Needs a CUDA or ROCm device (FlexAttention's GPU kernels are Triton),
    PyTorch with ``flex_attention`` and ``BlockMask.from_kv_blocks`` (>= 2.5),
    Triton, and a checkpoint whose architecture the packed forward covers.
    """
    on_gpu = torch.device(device).type == "cuda"
    if not on_gpu and not _ALLOW_CPU:
        return False, f"device {device} is not a CUDA/ROCm GPU"
    try:
        from torch.nn.attention.flex_attention import BlockMask, flex_attention  # noqa: F401
    except ImportError:
        return False, f"torch {torch.__version__} has no FlexAttention (needs >= 2.5)"
    if not hasattr(BlockMask, "from_kv_blocks"):
        return False, f"torch {torch.__version__} BlockMask lacks from_kv_blocks"
    if on_gpu:
        try:
            import triton  # noqa: F401
        except ImportError:
            return False, "triton is not installed"
    if model is not None:
        cfg = model.config
        for flag in _UNSUPPORTED_FLAGS:
            if getattr(cfg, flag, False):
                return False, f"checkpoint uses {flag}, which the packed forward does not implement"
        if getattr(cfg, "hyper_connections", 1) != 1 or getattr(cfg, "loop_count", 1) != 1:
            return False, "checkpoint uses hyper-connections or looped depth"
    return True, ""


def _raise_compile_limit(n: int) -> None:
    """Allow one compiled entry per device (one PackedEncoder per GPU).

    Every encoder compiles the same method code object, so each extra device
    is a dynamo recompile; the default limit of 8 would be hit on a full
    Setonix node (8 GCDs).
    """
    cfg = torch._dynamo.config
    for name in ("recompile_limit", "cache_size_limit"):
        if hasattr(cfg, name) and getattr(cfg, name) < n:
            setattr(cfg, name, n)


class PackedEncoder:
    """Runs a loaded ModernProst model on packed rows of ``capacity`` tokens.

    Args:
        model: model from :func:`pholdlib.modernprost.model.get_modernprost_model`.
        device: device the model lives on.
        capacity: tokens per packed row (multiple of ``block``). 64k is a good
            default; it must be at least as long as the longest chunk.
        block: FlexAttention tile size.
        compile: ``"full"`` compiles the whole forward (fastest, ~30 s the
            first time), ``"flex"`` compiles only the attention kernel.
        n_encoders: total encoders that will be built in this process (one per
            GPU), so the dynamo recompile limit can be raised to fit.
    """

    def __init__(
        self,
        model,
        device: torch.device,
        capacity: int = 65536,
        block: int = DEFAULT_BLOCK,
        compile: str = "full",
        n_encoders: int = 1,
    ):
        from torch.nn.attention.flex_attention import BlockMask, flex_attention

        if capacity % block:
            raise ValueError(f"capacity {capacity} must be a multiple of block {block}")
        ok, why = packed_engine_status(device, model)
        if not ok:
            raise RuntimeError(f"packed engine unavailable: {why}")

        self.model = model
        self.device = torch.device(device)
        self.T = capacity
        self.BS = block
        cfg = model.config
        self.window = int(cfg.sliding_window)
        self.layer_is_full = [t == "full" for t in cfg.attn_layer_types]
        self.BlockMask = BlockMask
        self._from_kv_takes_lengths = "seq_lengths" in inspect.signature(BlockMask.from_kv_blocks).parameters
        # The checkpoint's own RoPE helper, so numerics match the padded path.
        self._apply_rope = sys.modules[type(model).__module__]._apply_rope

        # Per-token protein id, read by the mask functions. It is overwritten
        # in place for every batch so the same mask functions, and the graphs
        # compiled around them, are reused.
        self.doc = torch.zeros(self.T, dtype=torch.int32, device=self.device)
        doc, w = self.doc, self.window

        def same_protein(b, h, q, kv):
            return doc[q] == doc[kv]

        def same_protein_window(b, h, q, kv):
            return (doc[q] == doc[kv]) & ((q - kv).abs() <= w)

        self.full_mod, self.slide_mod = same_protein, same_protein_window
        self.nqb = self.T // self.BS
        self._ar_blocks = torch.arange(self.nqb, device=self.device, dtype=torch.int32)
        self._q_first = torch.arange(self.nqb, device=self.device, dtype=torch.int64) * self.BS
        self._q_last = self._q_first + self.BS - 1

        if compile == "full":
            _raise_compile_limit(max(8, 2 * n_encoders))
            self._flex = flex_attention
            self.forward = torch.compile(self._forward, dynamic=False)
        elif compile == "flex":
            self._flex = torch.compile(flex_attention, dynamic=False)
            self.forward = self._forward
        elif compile == "none":
            # Eager FlexAttention is a slow reference implementation; useful
            # only for tests on CPU-sized inputs.
            self._flex = flex_attention
            self.forward = self._forward
        else:
            raise ValueError(f"compile must be 'full', 'flex' or 'none', got {compile!r}")

    # ── forward ──────────────────────────────────────────────────────────────

    def _forward(self, ids, pos, bm_full, bm_slide):
        """Packed equivalent of ``ModernProst3Di12StModel.forward``.

        Returns ``(logits_3di, logits_12st, last_hidden_state)`` for the whole
        row, shaped ``(1, T, 20)``, ``(1, T, 12)`` and ``(1, T, H)``.
        """
        m = self.model
        enc = m.encoder
        h = enc.embeddings(ids)
        cos_f, sin_f = enc.rotary_emb(h, pos, "full")
        cos_s, sin_s = enc.rotary_emb(h, pos, "sliding")
        for layer, is_full in zip(enc.layers, self.layer_is_full):
            cos, sin, bm = (cos_f, sin_f, bm_full) if is_full else (cos_s, sin_s, bm_slide)
            attn = layer.attn
            x = layer.attn_norm(h)
            B, L, _ = x.shape
            qkv = attn.Wqkv(x).view(B, L, 3, attn.num_heads, attn.head_dim)
            q, k, v = qkv.unbind(dim=2)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            q, k = self._apply_rope(q, k, cos, sin)
            o = self._flex(q, k, v, block_mask=bm, scale=attn.attn_scale)
            o = o.transpose(1, 2).contiguous().view(B, L, -1)
            h = h + attn.Wo(o)
            h = h + layer.mlp(layer.mlp_norm(h))
        h = enc.final_norm(h)
        mult = m.mup_output_mult
        return m.head_3di(h) * mult, m.head_12st(h) * mult, h

    # ── inputs ───────────────────────────────────────────────────────────────

    def host_inputs(self, items: TokenisedItems, batch: np.ndarray):
        """CPU-side arrays for one packed row.

        Returns ``(ids, pos, doc, start, end, n_real)``: token ids, per-protein
        positions, per-token protein index, and for every token the start and
        end (exclusive) of the span it may attend to. Padding after the last
        protein is split into ``block``-sized pseudo-proteins that attend only
        to themselves, so they cost one tile each.
        """
        T, BS = self.T, self.BS
        lens = items.lengths[batch]
        n_real = int(lens.sum())
        if n_real > T:
            raise ValueError(f"packed batch of {n_real} tokens exceeds capacity {T}")
        ids = np.zeros(T, dtype=np.uint8)
        o = 0
        toks, offs = items.tokens, items.offsets
        for idx in batch:
            s, L = int(offs[idx]), int(items.lengths[idx])
            ids[o:o + L] = toks[s:s + L]
            o += L
        starts = np.zeros(len(batch), dtype=np.int64)
        if len(batch) > 1:
            np.cumsum(lens[:-1], out=starts[1:])
        doc = np.empty(T, dtype=np.int32)
        pos = np.zeros(T, dtype=np.int32)
        start = np.empty(T, dtype=np.int64)
        end = np.empty(T, dtype=np.int64)
        doc[:n_real] = np.repeat(np.arange(len(batch), dtype=np.int32), lens)
        pos[:n_real] = np.arange(n_real, dtype=np.int64) - np.repeat(starts, lens)
        start[:n_real] = np.repeat(starts, lens)
        end[:n_real] = np.repeat(starts + lens, lens)
        if n_real < T:
            t = np.arange(n_real, T, dtype=np.int64)
            blk = t // BS
            doc[n_real:] = (-1 - blk).astype(np.int32)
            start[n_real:] = np.maximum(blk * BS, n_real)
            end[n_real:] = np.minimum((blk + 1) * BS, T)
        return ids, pos, doc, start, end, n_real

    def _block_mask(self, lo, hi, mask_mod):
        """BlockMask whose query tile *i* visits key tiles covering [lo[i], hi[i])."""
        BS, n = self.BS, self.nqb
        lo_b = lo // BS
        hi_b = (hi - 1) // BS
        num = (hi_b - lo_b + 1).to(torch.int32)
        idx = (lo_b[:, None] + self._ar_blocks[None, :]).clamp_max(n - 1).to(torch.int32)
        kwargs = dict(BLOCK_SIZE=BS, mask_mod=mask_mod)
        if self._from_kv_takes_lengths:
            kwargs["seq_lengths"] = (self.T, self.T)
        return self.BlockMask.from_kv_blocks(num[None, None], idx[None, None], **kwargs)

    def run_batch(self, items: TokenisedItems, batch: np.ndarray, pin: bool = True):
        """Launch the forward for one packed row (asynchronous on GPU).

        Returns ``(logits_3di, logits_12st, hidden, n_real)`` with the row's
        real tokens first; slice ``[:n_real]`` before reading.
        """
        ids, pos, doc, start, end, n_real = self.host_inputs(items, batch)
        dev = self.device

        pin = pin and dev.type == "cuda"

        def up(a):
            t = torch.from_numpy(a)
            if pin:
                t = t.pin_memory()
            return t.to(dev, non_blocking=pin)

        ids_g = up(ids).long()[None]
        pos_g = up(pos).long()[None]
        self.doc.copy_(up(doc))
        st_g, en_g = up(start), up(end)
        s_first, e_last = st_g[self._q_first], en_g[self._q_last]
        w = self.window
        bm_full = self._block_mask(s_first, e_last, self.full_mod)
        bm_slide = self._block_mask(torch.maximum(s_first, self._q_first - w),
                                    torch.minimum(e_last, self._q_last + w + 1), self.slide_mod)
        with torch.inference_mode():
            l3, l12, hid = self.forward(ids_g, pos_g, bm_full, bm_slide)
        return l3[0], l12[0], hid[0], n_real
