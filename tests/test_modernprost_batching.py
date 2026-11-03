"""Unit tests for pholdlib.modernprost.batching (tokenisation and planning).

numpy only: no model download, no torch, no GPU.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).parent.parent / "src" / "pholdlib"


def _load(mod_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _SRC / mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_b = _load("modernprost/batching.py", "pholdlib.modernprost.batching")

# The 29-token ModernProst vocabulary (tokenization_modernprost_3di12st.py).
VOCAB = {
    "<pad>": 0, "</s>": 1, "<unk>": 2,
    "A": 3, "L": 4, "G": 5, "V": 6, "S": 7, "R": 8, "E": 9, "D": 10,
    "T": 11, "I": 12, "P": 13, "K": 14, "F": 15, "Q": 16, "N": 17,
    "Y": 18, "M": 19, "H": 20, "W": 21, "C": 22, "X": 23, "B": 24,
    "O": 25, "U": 26, "Z": 27, "<mask>": 28,
}


def reference_ids(seq: str):
    """What the original engine fed the model: U/Z/O -> X, then the
    tokenizer's per-character lookup with <unk> for anything else."""
    seq = seq.replace("U", "X").replace("Z", "X").replace("O", "X")
    return [VOCAB.get(c, VOCAB["<unk>"]) for c in seq]


@pytest.fixture(scope="module")
def lut():
    return _b.build_token_lut(VOCAB, VOCAB["<unk>"])


class FakeTokenizer:
    unk_token = "<unk>"

    def get_vocab(self):
        return dict(VOCAB)


# ===========================================================================
# Tokenisation
# ===========================================================================

class TestTokenLUT:
    @pytest.mark.parametrize("seq", [
        "MKTIIALSYIFCLVFA",
        "ACDEFGHIKLMNPQRSTVWY",
        "MKUZOXB",          # replaced / rare residues
        "mkt*-.? 1",        # lowercase, stop, gap and junk -> <unk>
        "",
    ])
    def test_matches_reference(self, lut, seq):
        raw = np.frombuffer(seq.encode("latin-1"), dtype=np.uint8)
        assert lut[raw].tolist() == reference_ids(seq)

    def test_never_emits_pad(self, lut):
        # <pad> (0) marks padding in the attention mask; no byte may map to it
        assert 0 not in set(lut.tolist())

    def test_from_tokenizer(self):
        assert np.array_equal(_b.lut_from_tokenizer(FakeTokenizer()),
                              _b.build_token_lut(VOCAB, VOCAB["<unk>"]))


class TestTokenisedItems:
    def _seqs(self):
        return [
            ("short", "MKTIIALSYIF", 11),
            ("long", "A" * 700 + "C" * 701, 1401),     # 3 chunks at chunk_len 500
            ("edge", "M" * 500, 500),                  # exactly chunk_len: not split
            ("empty", "", 0),
        ]

    def test_layout(self, lut):
        items = _b.TokenisedItems.from_sequences(self._seqs(), lut, chunk_len=500)
        assert items.seq_ids == ["short", "long", "edge", "empty"]
        assert items.n_chunked == 1
        assert items.lengths.tolist() == [11, 467, 467, 467, 500, 0]
        assert items.item_protein.tolist() == [0, 1, 1, 1, 2, 3]
        assert items.offsets[-1] == items.n_tokens == 11 + 1401 + 500
        # each protein is one contiguous slice spanning all of its chunks
        assert items.protein_offsets.tolist() == [0, 11, 1412, 1912, 1912]

    def test_tokens_match_reference(self, lut):
        seqs = self._seqs()
        items = _b.TokenisedItems.from_sequences(seqs, lut, chunk_len=500)
        for p, (_, seq, _) in enumerate(seqs):
            s, e = items.protein_offsets[p], items.protein_offsets[p + 1]
            assert items.tokens[s:e].tolist() == reference_ids(seq)

    def test_chunks_match_chunk_sequence(self, lut):
        seq = "".join("ACDEFGHIKLMNPQRSTVWY"[i % 20] for i in range(3401))
        items = _b.TokenisedItems.from_sequences([("p", seq, len(seq))], lut, chunk_len=1568)
        expected = [len(c) for _, c in _b.chunk_sequence(seq, 1568)]
        assert items.lengths.tolist() == expected


# ===========================================================================
# Planning
# ===========================================================================

def _random_lengths(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(rng.lognormal(5.0, 0.8, n).astype(np.int64), 1, 1568)


class TestPlanPadded:
    @pytest.mark.parametrize("budget,pad", [(4096, 8), (65536, 8), (20000, 64)])
    def test_budget_and_coverage(self, budget, pad):
        lengths = _random_lengths()
        batches = _b.plan_padded_batches(lengths, budget, pad)
        flat = np.concatenate(batches)
        assert sorted(flat.tolist()) == list(range(len(lengths)))
        for b in batches:
            lpad = -(-int(lengths[b].max()) // pad) * pad
            assert len(b) * lpad <= budget or len(b) == 1

    def test_longest_first_and_sorted_within(self):
        lengths = _random_lengths()
        batches = _b.plan_padded_batches(lengths, 16384, 8)
        firsts = [int(lengths[b[0]]) for b in batches]
        assert firsts == sorted(firsts, reverse=True)
        for b in batches:
            assert lengths[b].tolist() == sorted(lengths[b].tolist(), reverse=True)

    def test_max_batch(self):
        batches = _b.plan_padded_batches(np.full(100, 10), 10 ** 6, 8, max_batch=7)
        assert max(len(b) for b in batches) == 7

    def test_score_cap_limits_long_batches_only(self):
        lengths = np.array([1600] * 50 + [100] * 2000)
        batches = _b.plan_padded_batches(lengths, 65536, 8, max_scores=10 * 1600 * 1600)
        long_rows = [len(b) for b in batches if lengths[b[0]] == 1600]
        short_rows = [len(b) for b in batches if lengths[b[0]] == 100]
        assert max(long_rows) == 10             # capped by scores
        assert max(short_rows) == 65536 // 104  # still filled by tokens

    def test_oversized_item_gets_own_batch(self):
        batches = _b.plan_padded_batches(np.array([5000, 10, 10]), 1024, 8)
        assert [len(b) for b in batches] == [1, 2]

    def test_token_count(self):
        lengths = np.array([10, 9, 3])
        batches = [np.array([0, 1]), np.array([2])]
        assert _b.batch_token_count(lengths, batches, 8) == 2 * 16 + 8


class TestPlanPacked:
    @pytest.mark.parametrize("capacity", [2048, 65536])
    def test_capacity_and_coverage(self, capacity):
        lengths = _random_lengths()
        bins = _b.plan_packed_batches(lengths, capacity)
        flat = np.concatenate(bins)
        assert sorted(flat.tolist()) == list(range(len(lengths)))
        assert all(int(lengths[b].sum()) <= capacity for b in bins)

    def test_bins_are_nearly_full(self):
        lengths = _random_lengths(5000)
        capacity = 8192
        bins = _b.plan_packed_batches(lengths, capacity)
        fill = [int(lengths[b].sum()) / capacity for b in bins[:-1]]
        assert min(fill) > 0.98

    def test_rejects_item_longer_than_capacity(self):
        with pytest.raises(ValueError):
            _b.plan_packed_batches(np.array([10, 300]), 256)

    def test_empty(self):
        assert _b.plan_packed_batches(np.array([], dtype=np.int64), 1024) == []
