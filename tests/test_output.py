"""Unit tests for pholdlib.prostt5.output.

These tests load output.py directly via importlib to avoid triggering the
pholdlib.prostt5 package __init__ (which eagerly imports model.py → transformers).
No model download or GPU is required.
"""

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Load output.py directly, bypassing pholdlib/prostt5/__init__.py so we avoid
# the transformers import that lives in model.py.
# ---------------------------------------------------------------------------
_SRC = Path(__file__).parent.parent / "src" / "pholdlib"


def _load_output():
    spec = importlib.util.spec_from_file_location(
        "pholdlib.prostt5.output",
        _SRC / "prostt5" / "output.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_output = _load_output()
SS_MAPPING = _output.SS_MAPPING
mask_low_confidence_aa = _output.mask_low_confidence_aa
write_probs = _output.write_probs
write_fail_ids = _output.write_fail_ids


# ===========================================================================
# SS_MAPPING
# ===========================================================================

class TestSSMapping:
    def test_size(self):
        assert len(SS_MAPPING) == 21  # indices 0-19 (standard 3Di) + 20 (masked)

    def test_standard_letters_cover_alphabet(self):
        standard_letters = {SS_MAPPING[i] for i in range(20)}
        assert standard_letters == set("ACDEFGHIKLMNPQRSTVWY")

    def test_masked_residue_is_X(self):
        assert SS_MAPPING[20] == "X"

    def test_all_values_are_single_characters(self):
        for idx, ch in SS_MAPPING.items():
            assert isinstance(ch, str) and len(ch) == 1, f"index {idx}: {ch!r}"

    def test_indices_are_integers(self):
        for k in SS_MAPPING:
            assert isinstance(k, int)


# ===========================================================================
# mask_low_confidence_aa
# ===========================================================================

class TestMaskLowConfidenceAA:
    """scores is expected to be a sequence-of-sequences, e.g. [[0.9, 0.8, ...]].

    The function calls zip(sequence, *scores), so a shape-(1, L) list/array
    unpacks correctly to a single length-L iterable.
    """

    def test_all_above_threshold_unchanged(self):
        seq = "ACDEF"
        scores = [[0.9, 0.8, 0.7, 0.6, 0.55]]
        assert mask_low_confidence_aa(seq, scores, threshold=0.5) == "ACDEF"

    def test_all_below_threshold_become_X(self):
        seq = "ACDEF"
        scores = [[0.1, 0.2, 0.3, 0.4, 0.49]]
        assert mask_low_confidence_aa(seq, scores, threshold=0.5) == "XXXXX"

    def test_mixed_positions(self):
        seq = "ACDE"
        scores = [[0.6, 0.4, 0.7, 0.3]]
        result = mask_low_confidence_aa(seq, scores, threshold=0.5)
        assert result == "AXDX"

    def test_boundary_is_exclusive_below(self):
        # score == threshold → kept; score < threshold → masked
        seq = "AC"
        scores = [[0.5, 0.499]]
        result = mask_low_confidence_aa(seq, scores, threshold=0.5)
        assert result == "AX"

    def test_numpy_scores(self):
        seq = "MKL"
        scores = np.array([[0.9, 0.1, 0.8]])
        result = mask_low_confidence_aa(seq, scores, threshold=0.5)
        assert result == "MXL"

    def test_empty_sequence(self):
        result = mask_low_confidence_aa("", [[]], threshold=0.5)
        assert result == ""

    def test_custom_threshold_zero(self):
        # threshold=0 → nothing masked
        seq = "ACDE"
        scores = [[0.0, 0.0, 0.0, 0.0]]
        assert mask_low_confidence_aa(seq, scores, threshold=0.0) == "ACDE"

    def test_custom_threshold_one(self):
        # threshold=1 → everything below 1.0 is masked
        seq = "ACDE"
        scores = [[0.99, 0.99, 0.99, 0.99]]
        assert mask_low_confidence_aa(seq, scores, threshold=1.0) == "XXXX"


# ===========================================================================
# write_probs
# ===========================================================================

class TestWriteProbs:
    def _preds(self, ids, means, probs=None):
        out = {}
        for i, sid in enumerate(ids):
            arr = np.array([[0.9]], dtype=np.float32) if probs is None else probs[i]
            out[sid] = (np.array([0], dtype=np.byte), means[i], arr)
        return out

    def test_mean_csv_content(self, tmp_path):
        preds = {
            "seq1": (np.array([0, 1, 2], dtype=np.byte), 85.5, None),
            "seq2": (np.array([3, 4], dtype=np.byte), 72.1, None),
        }
        mean_path = tmp_path / "mean.csv"
        write_probs(preds, mean_path, None, ["seq1", "seq2"])

        lines = mean_path.read_text().strip().splitlines()
        assert lines[0] == "seq1,85.5"
        assert lines[1] == "seq2,72.1"

    def test_key_order_respected(self, tmp_path):
        preds = {
            "a": (np.array([0], dtype=np.byte), 10.0, None),
            "b": (np.array([1], dtype=np.byte), 20.0, None),
            "c": (np.array([2], dtype=np.byte), 30.0, None),
        }
        mean_path = tmp_path / "mean.csv"
        write_probs(preds, mean_path, None, ["c", "b", "a"])

        lines = mean_path.read_text().strip().splitlines()
        assert lines[0].startswith("c,")
        assert lines[1].startswith("b,")
        assert lines[2].startswith("a,")

    def test_missing_key_skipped(self, tmp_path):
        preds = {"seq1": (np.array([0], dtype=np.byte), 50.0, None)}
        mean_path = tmp_path / "mean.csv"
        write_probs(preds, mean_path, None, ["seq1", "ghost"])

        lines = mean_path.read_text().strip().splitlines()
        assert len(lines) == 1
        assert lines[0] == "seq1,50.0"

    def test_all_probs_jsonl_written(self, tmp_path):
        all_probs = np.array([[0.9, 0.8, 0.7]], dtype=np.float32)
        preds = {
            "seq1": (np.array([0, 1, 2], dtype=np.byte), 80.0, all_probs),
        }
        mean_path = tmp_path / "mean.csv"
        all_path = tmp_path / "all.jsonl"
        write_probs(preds, mean_path, all_path, ["seq1"])

        with open(all_path) as f:
            record = json.loads(f.readline())

        assert record["seq_id"] == "seq1"
        # probabilities are multiplied by 100 and rounded to 2 dp
        assert record["probability"] == pytest.approx([90.0, 80.0, 70.0], abs=0.01)

    def test_all_probs_multiple_sequences(self, tmp_path):
        preds = {
            "p1": (np.array([0], dtype=np.byte), 90.0, np.array([[1.0]], dtype=np.float32)),
            "p2": (np.array([1], dtype=np.byte), 50.0, np.array([[0.5]], dtype=np.float32)),
        }
        mean_path = tmp_path / "mean.csv"
        all_path = tmp_path / "all.jsonl"
        write_probs(preds, mean_path, all_path, ["p1", "p2"])

        records = [json.loads(line) for line in all_path.read_text().splitlines()]
        assert [r["seq_id"] for r in records] == ["p1", "p2"]
        assert records[0]["probability"] == [100.0]
        assert records[1]["probability"] == [50.0]

    def test_none_all_probs_path_skips_jsonl(self, tmp_path):
        all_probs = np.array([[0.9, 0.8]], dtype=np.float32)
        preds = {"seq1": (np.array([0, 1], dtype=np.byte), 85.0, all_probs)}
        mean_path = tmp_path / "mean.csv"
        write_probs(preds, mean_path, None, ["seq1"])

        assert not any(tmp_path.glob("*.jsonl"))


# ===========================================================================
# write_fail_ids
# ===========================================================================

class TestWriteFailIds:
    def test_empty_list_creates_no_file(self, tmp_path):
        out = tmp_path / "fails.tsv"
        write_fail_ids([], out)
        assert not out.exists()

    def test_single_id(self, tmp_path):
        out = tmp_path / "fails.tsv"
        write_fail_ids(["bad_seq"], out)
        rows = list(csv.reader(out.read_text().splitlines(), delimiter="\t"))
        assert rows == [["bad_seq"]]

    def test_multiple_ids(self, tmp_path):
        out = tmp_path / "fails.tsv"
        ids = ["seq1", "seq2", "seq3"]
        write_fail_ids(ids, out)
        rows = list(csv.reader(out.read_text().splitlines(), delimiter="\t"))
        assert rows == [["seq1"], ["seq2"], ["seq3"]]

    def test_non_string_ids_are_stringified(self, tmp_path):
        out = tmp_path / "fails.tsv"
        write_fail_ids([1, 2, 3], out)
        rows = list(csv.reader(out.read_text().splitlines(), delimiter="\t"))
        assert rows == [["1"], ["2"], ["3"]]
