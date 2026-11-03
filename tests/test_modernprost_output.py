"""Unit tests for pholdlib.modernprost.output.

Loaded directly via importlib so the package __init__ (which imports
model.py → transformers) is not triggered. No model download or GPU required.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).parent.parent / "src" / "pholdlib"


def _load(mod_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _SRC / mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_output = _load("modernprost/output.py", "pholdlib.modernprost.output")

THREEDI_ALPHABET = _output.THREEDI_ALPHABET
SS12_ALPHABET = _output.SS12_ALPHABET
THREEDI_MAPPING = _output.THREEDI_MAPPING
SS12_MAPPING = _output.SS12_MAPPING
THREEDI_MASK_INDEX = _output.THREEDI_MASK_INDEX
decode_3di = _output.decode_3di
decode_12st = _output.decode_12st
format_profile = _output.format_profile
write_3di_fasta = _output.write_3di_fasta
write_12st_fasta = _output.write_12st_fasta
write_mean_probs = _output.write_mean_probs
write_all_probs = _output.write_all_probs
write_profiles_text = _output.write_profiles_text


def _pred(indices, probs=None):
    """Build a (pred, mean_prob, all_prob) tuple like the inference engine emits."""
    arr = np.asarray(indices, dtype=np.byte)
    if probs is None:
        probs = np.ones(len(arr), dtype=np.float32)
    probs = np.asarray(probs, dtype=np.float32)
    mean = round(100 * float(probs.mean()), 2) if probs.size else 0.0
    return (arr, mean, probs)


# ===========================================================================
# Alphabets
# ===========================================================================

class TestAlphabets:
    def test_3di_has_20_states_plus_mask(self):
        assert len(THREEDI_ALPHABET) == 20
        assert len(THREEDI_MAPPING) == 21
        assert THREEDI_MAPPING[THREEDI_MASK_INDEX] == "X"

    def test_12st_has_12_states_and_no_mask(self):
        assert len(SS12_ALPHABET) == 12
        assert len(SS12_MAPPING) == 12
        assert "X" not in SS12_MAPPING.values()

    def test_12st_is_a_prefix_of_3di(self):
        # Both alphabets are drawn from the same letter sequence, which is why
        # a 12-state string is not self-describing without its alphabet.
        assert THREEDI_ALPHABET.startswith(SS12_ALPHABET)

    def test_decode_roundtrip(self):
        assert decode_3di(range(20)) == THREEDI_ALPHABET
        assert decode_12st(range(12)) == SS12_ALPHABET

    def test_decode_3di_accepts_mask_index(self):
        assert decode_3di([0, 20, 1]) == "AXC"

    def test_decode_12st_rejects_out_of_range(self):
        with pytest.raises(KeyError):
            decode_12st([12])


# ===========================================================================
# FASTA writers
# ===========================================================================

class TestWrite3diFasta:
    def test_writes_in_requested_order(self, tmp_path):
        preds = {"b": _pred([0, 1]), "a": _pred([2, 3])}
        out = tmp_path / "3di.fasta"
        write_3di_fasta(preds, out, ["a", "b"])
        assert out.read_text() == ">a\nDE\n>b\nAC\n"

    def test_skips_missing_and_zero_length(self, tmp_path):
        preds = {"a": _pred([0]), "empty": _pred([])}
        out = tmp_path / "3di.fasta"
        write_3di_fasta(preds, out, ["a", "empty", "absent"])
        assert out.read_text() == ">a\nA\n"

    def test_header_map_is_applied(self, tmp_path):
        preds = {"cds_1": _pred([0])}
        out = tmp_path / "3di.fasta"
        write_3di_fasta(preds, out, ["cds_1"], header_map={"cds_1": "contig:cds_1"})
        assert out.read_text().startswith(">contig:cds_1\n")

    def test_masking_below_threshold(self, tmp_path):
        preds = {"a": _pred([0, 1, 2], probs=[0.9, 0.1, 0.8])}
        out = tmp_path / "3di.fasta"
        write_3di_fasta(preds, out, ["a"], mask_threshold=50)
        assert out.read_text() == ">a\nAXD\n"

    def test_masking_does_not_mutate_caller_array(self, tmp_path):
        pred = _pred([0, 1, 2], probs=[0.9, 0.1, 0.8])
        preds = {"a": pred}
        write_3di_fasta(preds, tmp_path / "3di.fasta", ["a"], mask_threshold=50)
        # The inference result is reused to build the AA FASTA afterwards, so
        # masking must not write back into it.
        assert pred[0].tolist() == [0, 1, 2]

    def test_zero_threshold_disables_masking(self, tmp_path):
        preds = {"a": _pred([0, 1], probs=[0.0, 0.0])}
        out = tmp_path / "3di.fasta"
        write_3di_fasta(preds, out, ["a"], mask_threshold=0)
        assert out.read_text() == ">a\nAC\n"


class TestWrite12stFasta:
    def test_writes_12_state_letters(self, tmp_path):
        preds = {"a": _pred([0, 11, 5])}
        out = tmp_path / "12st.fasta"
        write_12st_fasta(preds, out, ["a"])
        assert out.read_text() == ">a\nANG\n"


# ===========================================================================
# Probability writers
# ===========================================================================

class TestProbabilityWriters:
    def test_mean_probs_csv(self, tmp_path):
        preds = {"a": _pred([0], probs=[0.8265]), "b": _pred([1], probs=[0.5])}
        out = tmp_path / "mean.csv"
        write_mean_probs(preds, out, ["a", "b"])
        assert out.read_text() == "a,82.65\nb,50.0\n"

    def test_all_probs_jsonl(self, tmp_path):
        preds = {"a": _pred([0, 1], probs=[0.5, 0.25])}
        out = tmp_path / "all.json"
        write_all_probs(preds, out, ["a"])
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        assert rows == [{"seq_id": "a", "probability": [50.0, 25.0]}]

    def test_all_probs_skips_none(self, tmp_path):
        preds = {"a": (np.array([0], dtype=np.byte), 50.0, None)}
        out = tmp_path / "all.json"
        write_all_probs(preds, out, ["a"])
        assert out.read_text() == ""


# ===========================================================================
# Profile text
# ===========================================================================

class TestProfileText:
    def test_format_profile_header_and_rows(self):
        profile = np.array([[0.5, 0.5], [0.25, 0.75]], dtype=np.float32)
        text = format_profile(profile, "AC")
        lines = text.splitlines()
        assert lines[0].split() == ["A", "C"]
        assert lines[1] == "0.5000 0.5000"
        assert lines[2] == "0.2500 0.7500"

    def test_write_profiles_text_blocks(self, tmp_path):
        profiles = {"a": np.full((2, 12), 1 / 12, dtype=np.float32)}
        out = tmp_path / "prof.txt"
        write_profiles_text(profiles, out, SS12_ALPHABET, ["a", "missing"])
        text = out.read_text()
        assert text.startswith("Query profile of sequence a\n")
        # id line + alphabet row + 2 residue rows + block-separating blank line
        assert len(text.splitlines()) == 5
        # "missing" has no profile, so no second block is emitted.
        assert text.count("Query profile of sequence") == 1
