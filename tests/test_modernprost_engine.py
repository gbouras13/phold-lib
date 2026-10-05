"""Unit tests for the ModernProst inference engines (CPU only, no download).

* engine / compile selection on CPU, NVIDIA and ROCm devices;
* output assembly: chunk stitching, failures, per-protein means;
* the packed engine's FlexAttention masks: analytic block ranges against the
  dense masks, and packed attention against per-protein SDPA in float64.
"""

import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pholdlib.modernprost import batching  # noqa: E402
from pholdlib.modernprost import inference  # noqa: E402
from pholdlib.modernprost import packed  # noqa: E402

VOCAB = {"<pad>": 0, "</s>": 1, "<unk>": 2, **{c: i + 3 for i, c in enumerate("ALGVSREDTIPKFQNYMHWCXBOUZ")},
         "<mask>": 28}
LUT = batching.build_token_lut(VOCAB, 2)

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention  # noqa: F401
    HAS_FLEX = hasattr(BlockMask, "from_kv_blocks")
except ImportError:  # torch < 2.5
    HAS_FLEX = False


def fake_model(window=64, layer_types=("full", "sliding", "sliding")):
    cfg = types.SimpleNamespace(sliding_window=window, attn_layer_types=list(layer_types),
                                hidden_size=16, use_diff_attn=False, use_attn_gate=False,
                                value_residual=False, use_qk_norm=False, hyper_connections=1,
                                loop_count=1)
    return FakeModel(cfg)


class FakeModel:
    """Just enough of a ModernProst model for PackedEncoder's constructor."""

    def __init__(self, config):
        self.config = config


def _apply_rope(q, k, cos, sin):  # PackedEncoder looks this up on the model's module
    return q, k


# ===========================================================================
# Engine selection
# ===========================================================================

class TestChooseEngine:
    @pytest.fixture
    def gpu(self, monkeypatch):
        """A CUDA device on which the packed engine is reported available."""
        monkeypatch.setattr(packed, "packed_engine_status", lambda device, model=None: (True, ""))
        monkeypatch.setattr(inference, "is_rocm", lambda: False)
        return torch.device("cuda")

    BIG = 10 ** 9
    SMALL = 1000

    def test_cpu_is_always_eager_padded(self):
        dev = torch.device("cpu")
        assert inference._choose_engine("auto", "auto", dev, fake_model(), self.BIG) == ("padded", "none")

    def test_nvidia_large_input_packs_and_compiles(self, gpu):
        assert inference._choose_engine("auto", "auto", gpu, fake_model(), self.BIG) == ("packed", "full")

    def test_small_input_skips_compile(self, gpu):
        assert inference._choose_engine("auto", "auto", gpu, fake_model(), self.SMALL) == ("padded", "none")

    def test_rocm_large_input_compiles_padded(self, gpu, monkeypatch):
        monkeypatch.setattr(inference, "is_rocm", lambda: True)
        assert inference._choose_engine("auto", "auto", gpu, fake_model(), self.BIG) == ("padded", "dynamic")

    def test_threshold_env_override(self, gpu, monkeypatch):
        monkeypatch.setenv("PHOLDLIB_PACKED_MIN_RESIDUES", "500")
        assert inference._choose_engine("auto", "auto", gpu, fake_model(), self.SMALL) == ("packed", "full")

    def test_threshold_depends_on_model_size(self, monkeypatch):
        monkeypatch.delenv("PHOLDLIB_PACKED_MIN_RESIDUES", raising=False)
        small = torch.nn.Linear(10, 10)
        big = types.SimpleNamespace(parameters=lambda: [torch.empty(300_000_000, dtype=torch.int8)])
        assert inference.packed_min_residues(small) == inference.PACKED_MIN_RESIDUES
        assert inference.packed_min_residues(big) == inference.PACKED_MIN_RESIDUES_LARGE_MODEL
        assert inference.packed_min_residues(None) == inference.PACKED_MIN_RESIDUES

    def test_explicit_requests(self, gpu):
        m = fake_model()
        assert inference._choose_engine("padded", "on", gpu, m, self.SMALL) == ("padded", "dynamic")
        assert inference._choose_engine("padded", "auto", gpu, m, self.BIG) == ("padded", "dynamic")
        assert inference._choose_engine("auto", "off", gpu, m, self.BIG) == ("padded", "none")
        assert inference._choose_engine("packed", "auto", gpu, m, self.SMALL) == ("packed", "full")
        assert inference._choose_engine("packed", "flex", gpu, m, self.SMALL) == ("packed", "flex")

    def test_packed_without_compile_on_gpu_falls_back(self, gpu):
        # eager FlexAttention would build a dense (T x T) score matrix
        assert inference._choose_engine("packed", "off", gpu, fake_model(), self.BIG) == ("padded", "none")

    def test_packed_unavailable_falls_back(self, monkeypatch):
        monkeypatch.setattr(packed, "packed_engine_status", lambda device, model=None: (False, "no triton"))
        dev = torch.device("cuda")
        # still a large GPU input, so the padded fallback is compiled
        assert inference._choose_engine("packed", "auto", dev, fake_model(), self.BIG) == ("padded", "dynamic")
        assert inference._choose_engine("packed", "auto", dev, fake_model(), self.SMALL) == ("padded", "none")

    def test_rejects_unknown_values(self):
        with pytest.raises(ValueError):
            inference._choose_engine("fast", "auto", torch.device("cpu"), fake_model(), 1)
        with pytest.raises(ValueError):
            inference._choose_engine("auto", "maybe", torch.device("cpu"), fake_model(), 1)

    def test_unsupported_architecture_is_not_packed(self):
        m = fake_model()
        m.config.use_diff_attn = True
        ok, why = packed.packed_engine_status(torch.device("cuda"), m)
        assert not ok


# ===========================================================================
# Output assembly
# ===========================================================================

class TestOutputStore:
    def _items(self):
        seqs = [("a", "MKTIIAL", 7), ("long", "A" * 25, 25), ("b", "MK", 2)]
        return batching.TokenisedItems.from_sequences(seqs, LUT, chunk_len=10)

    def _fill(self, store, items, value_fn):
        for idx in range(items.n_items):
            o, L = int(items.offsets[idx]), int(items.lengths[idx])
            host = value_fn(o, L)
            store.write(np.array([idx]), host)

    def test_classification_stitches_chunks(self):
        items = self._items()
        assert items.lengths.tolist() == [7, 9, 8, 8, 2]
        store = inference._OutputStore(items, "classification", want_hidden=False, hidden_size=4)
        self._fill(store, items, lambda o, L: {
            "p3": (np.arange(o, o + L) % 20).astype(np.int8), "p12": np.zeros(L, np.int8),
            "q3": np.full(L, 0.5, np.float32), "q12": np.full(L, 0.25, np.float32)})
        res = store.finalise(output_probs=True, per_residue=False, per_protein=False)
        assert set(res.predictions_3di) == {"a", "long", "b"}
        pred, mean, probs = res.predictions_3di["long"]
        assert pred.tolist() == [i % 20 for i in range(7, 32)] and pred.dtype == np.int8
        assert mean == 50.0 and probs.dtype == np.float32 and len(probs) == 25
        assert res.predictions_12st["b"][1] == 25.0

    def test_output_probs_false_keeps_mean(self):
        items = self._items()
        store = inference._OutputStore(items, "classification", want_hidden=False, hidden_size=4)
        self._fill(store, items, lambda o, L: {"p3": np.zeros(L, np.int8), "p12": np.zeros(L, np.int8),
                                               "q3": np.ones(L, np.float32), "q12": np.ones(L, np.float32)})
        pred, mean, probs = store.finalise(False, False, False).predictions_3di["a"]
        assert probs is None and mean == 100.0

    def test_failed_chunk_fails_whole_protein(self):
        items = self._items()
        store = inference._OutputStore(items, "pssm", want_hidden=True, hidden_size=4)
        store.fail(np.array([2]))  # one chunk of "long"
        res = store.finalise(True, per_residue=True, per_protein=True)
        assert res.fail_ids == ["long"]
        assert set(res.profiles_3di) == {"a", "b"}
        assert res.profiles_3di["a"].shape == (7, 20) and res.profiles_12st["a"].shape == (7, 12)
        assert res.embeddings_per_residue["b"].shape == (2, 4)
        assert res.embeddings_per_protein["b"].shape == (4,)

    def test_segment_means(self):
        values = np.arange(10, dtype=np.float32)
        offsets = np.array([0, 3, 3, 10])
        assert inference._segment_means(values, offsets).tolist() == [1.0, 0.0, 6.0]
        v2 = np.arange(20, dtype=np.float32).reshape(10, 2)
        assert inference._segment_means(v2, offsets)[2].tolist() == [12.0, 13.0]


# ===========================================================================
# Packed attention masks
# ===========================================================================

@pytest.mark.skipif(not HAS_FLEX, reason="needs torch >= 2.5 FlexAttention")
class TestPackedMasks:
    W = 64

    @pytest.fixture
    def encoder_factory(self, monkeypatch):
        monkeypatch.setattr(packed, "_ALLOW_CPU", True)

        def make(T):
            return packed.PackedEncoder(fake_model(self.W), torch.device("cpu"), capacity=T,
                                        compile="none")
        return make

    @staticmethod
    def _items(lengths):
        seqs = [(f"p{i}", "A" * int(L), int(L)) for i, L in enumerate(lengths)]
        return batching.TokenisedItems.from_sequences(seqs, LUT, chunk_len=10 ** 6)

    def _masks(self, enc, items, batch):
        ids, pos, doc, start, end, n_real = enc.host_inputs(items, batch)
        enc.doc.copy_(torch.from_numpy(doc))
        st, en = torch.from_numpy(start), torch.from_numpy(end)
        qf, ql = enc._q_first, enc._q_last
        s_first, e_last = st[qf], en[ql]
        bm_full = enc._block_mask(s_first, e_last, enc.full_mod)
        bm_slide = enc._block_mask(torch.maximum(s_first, qf - self.W),
                                   torch.minimum(e_last, ql + self.W + 1), enc.slide_mod)
        return bm_full, bm_slide, torch.from_numpy(doc), pos, n_real

    @pytest.mark.parametrize("lengths,T", [
        ([1568, 1568, 300, 5, 1, 64, 65, 127, 128, 129, 2000], 8192),   # tile-edge cases
        ([1], 128), ([128], 128), ([127, 1], 256), ([3000], 4096),
    ])
    def test_block_ranges_cover_dense_masks_exactly(self, encoder_factory, lengths, T):
        lengths = np.array(lengths)
        lengths = lengths[: np.searchsorted(np.cumsum(lengths), T, side="right")]
        items = self._items(lengths)
        enc = encoder_factory(T)
        bm_full, bm_slide, doc, _, _ = self._masks(enc, items, np.arange(len(lengths)))
        for bm, window in ((bm_full, None), (bm_slide, self.W)):
            def mm(b, h, q, kv):
                m = doc[q] == doc[kv]
                return m if window is None else m & ((q - kv).abs() <= window)
            dense = create_block_mask(mm, 1, 1, T, T, device="cpu", BLOCK_SIZE=enc.BS).to_dense()[0, 0].bool()
            ours = bm.to_dense()[0, 0].bool()
            assert torch.equal(dense, ours)

    def test_positions_restart_per_protein(self, encoder_factory):
        items = self._items([5, 3])
        enc = encoder_factory(256)
        _, pos, doc, start, end, n_real = enc.host_inputs(items, np.array([0, 1]))
        assert n_real == 8
        assert pos[:8].tolist() == [0, 1, 2, 3, 4, 0, 1, 2]
        assert doc[:8].tolist() == [0] * 5 + [1] * 3
        assert (doc[8:] < 0).all()  # padding is its own pseudo-protein

    def test_packed_attention_equals_per_protein_sdpa(self, encoder_factory):
        rng = np.random.default_rng(0)
        lengths = rng.integers(20, 400, size=10)
        T = int(-(-lengths.sum() // 128) * 128) + 128
        items = self._items(lengths)
        enc = encoder_factory(T)
        bm_full, bm_slide, doc, _, n_real = self._masks(enc, items, np.arange(len(lengths)))
        H, D = 4, 32
        q = torch.randn(1, H, T, D, dtype=torch.float64)
        k, v = torch.randn_like(q), torch.randn_like(q)
        out_full = flex_attention(q, k, v, block_mask=bm_full, scale=D ** -0.5)
        out_slide = flex_attention(q, k, v, block_mask=bm_slide, scale=D ** -0.5)
        start = 0
        for L in lengths.tolist():
            sl = slice(start, start + L)
            p = torch.arange(L)
            ref_full = torch.nn.functional.scaled_dot_product_attention(q[:, :, sl], k[:, :, sl], v[:, :, sl],
                                                                        scale=D ** -0.5)
            ref_slide = torch.nn.functional.scaled_dot_product_attention(
                q[:, :, sl], k[:, :, sl], v[:, :, sl], attn_mask=(p[:, None] - p[None, :]).abs() <= self.W,
                scale=D ** -0.5)
            assert (out_full[:, :, sl] - ref_full).abs().max().item() < 1e-10
            assert (out_slide[:, :, sl] - ref_slide).abs().max().item() < 1e-10
            start += L
