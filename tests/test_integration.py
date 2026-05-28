"""Integration tests for pholdlib.

These tests exercise the full ProstT5 + CNN pipeline end-to-end.

Requirements
------------
* A working transformers install (pholdENV satisfies this).
* ProstT5_fp16 model weights — downloaded automatically to
  tests/test_data/model_cache on first run (~1.6 GB, fp16).

Run
---
# Simplest — downloads ProstT5 automatically if not cached
pytest tests/test_integration.py --run_integration

# Pre-existing model cache
pytest tests/test_integration.py --run_integration --model_dir /path/to/cache

# Use the real trained phold CNN weights (phold repo ships these)
pytest tests/test_integration.py --run_integration \\
    --checkpoint /path/to/phold/src/phold/cnn/cnn_chkpnt/model.pt

# GPU + multiple threads
pytest tests/test_integration.py --run_integration --gpu_available --threads 8
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.conftest import HAS_TRANSFORMERS

# ---------------------------------------------------------------------------
# Short phage-protein sequences used across tests.
# Real sequences picked to keep runtime short (<60 AA each).
# ---------------------------------------------------------------------------
PHAGE_SEQS = [
    ("prot_A", "MKTIIALSYIFCLVFA",                                   16),
    ("prot_B", "MASMTGGQQMGRDLYDDDDKDHHHHHH",                        27),
    ("prot_C", "MGSSHHHHHHSSGLVPRGSH",                               20),
    ("prot_D", "MVKVYAPASSANMSVGFDVLGAAVTPVDGALLGDVVTVEAAETFSLNNLGRFADKLP", 57),
    ("prot_E", "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLT",        43),
]

# Sanity-check lengths at import time so a typo fails immediately.
for _sid, _seq, _slen in PHAGE_SEQS:
    assert len(_seq) == _slen, f"Length mismatch in PHAGE_SEQS: {_sid} ({len(_seq)} != {_slen})"

# Sorted descending by length (as run_prostt5_inference expects)
PHAGE_SEQS_SORTED = sorted(PHAGE_SEQS, key=lambda x: x[2], reverse=True)

MODEL_NAME = "Rostlab/ProstT5_fp16"


# ===========================================================================
# Helpers
# ===========================================================================

def _skip_unless_integration(run_integration):
    if not run_integration:
        pytest.skip("Pass --run_integration to run integration tests.")
    if not HAS_TRANSFORMERS:
        pytest.skip(
            "transformers not fully importable (e.g. scipy missing). "
            "Use the pholdENV conda environment."
        )


def _load_model(model_dir, gpu_available, threads):
    """Load ProstT5, downloading from HuggingFace if not already cached."""
    from pholdlib.prostt5.model import get_T5_model
    from pholdlib.databases.prostt5 import check_prostT5_download

    return get_T5_model(
        model_dir=model_dir,
        model_name=MODEL_NAME,
        cpu=not gpu_available,
        threads=threads,
        check_fn=check_prostT5_download,
    )


def _make_predictor(device, checkpoint=None):
    """Load CNN predictor from checkpoint, or fall back to random weights.

    A random predictor produces meaningless 3Di class assignments but correct
    shapes, types, and probability ranges, which is sufficient to test the
    inference pipeline in isolation from the model quality.
    """
    from pholdlib.prostt5.model import CNN, load_predictor

    if checkpoint is not None:
        return load_predictor(checkpoint, device)

    cnn = CNN().eval().to(device)
    return cnn


# ===========================================================================
# Session-scoped model fixture — loaded once, shared across all classes
# ===========================================================================

@pytest.fixture(scope="session")
def loaded_model(run_integration, model_dir, gpu_available, threads):
    _skip_unless_integration(run_integration)
    return _load_model(model_dir, gpu_available, threads)


@pytest.fixture(scope="session")
def predictor(loaded_model, checkpoint):
    _, _, device = loaded_model
    return _make_predictor(device, checkpoint)


# ===========================================================================
# Model loading
# ===========================================================================

class TestModelLoading:
    def test_returns_three_values(self, loaded_model):
        assert len(loaded_model) == 3

    def test_device_is_torch_device(self, loaded_model):
        _, _, device = loaded_model
        assert isinstance(device, torch.device)

    def test_model_is_in_eval_mode(self, loaded_model):
        model, _, _ = loaded_model
        assert not model.training

    def test_model_is_on_expected_device(self, loaded_model, gpu_available):
        model, _, device = loaded_model
        if gpu_available:
            assert device.type in ("cuda", "mps")
        else:
            assert device.type == "cpu"

    def test_vocab_tokenises_prefix(self, loaded_model):
        """The tokenizer must handle the '<AA2fold>' prefix used in inference."""
        _, vocab, _ = loaded_model
        tokens = vocab("<AA2fold> M K T", return_tensors="pt")
        assert tokens.input_ids.shape[1] > 0

    def test_cpu_model_is_float32(self, loaded_model, gpu_available):
        """CPU inference forces FP32 to avoid half-precision errors."""
        model, _, device = loaded_model
        if device.type == "cpu":
            param = next(model.parameters())
            assert param.dtype == torch.float32


# ===========================================================================
# Predictor (CNN head)
# ===========================================================================

class TestPredictor:
    def test_predictor_is_cnn(self, predictor):
        from pholdlib.prostt5.model import CNN
        assert isinstance(predictor, CNN)

    def test_predictor_in_eval_mode(self, predictor):
        assert not predictor.training

    def test_predictor_forward_shape(self, loaded_model, predictor):
        _, _, device = loaded_model
        x = torch.randn(1, 10, 1024).to(device)
        with torch.no_grad():
            out = predictor(x)
        assert out.shape == (1, 20, 10)

    def test_checkpoint_weights_loaded(self, predictor, checkpoint):
        """When a real checkpoint is given, weights should differ from random init."""
        if checkpoint is None:
            pytest.skip("No --checkpoint supplied; skipping trained-weight check.")
        from pholdlib.prostt5.model import CNN
        random_cnn = CNN()
        ckpt_bias = next(predictor.parameters()).detach().cpu()
        rand_bias = next(random_cnn.parameters()).detach().cpu()
        assert not torch.equal(ckpt_bias, rand_bias), (
            "Checkpoint weights are identical to random init — checkpoint may not have loaded."
        )


# ===========================================================================
# run_prostt5_inference — core pipeline
# ===========================================================================

@pytest.fixture(scope="module")
def basic_inference(loaded_model, predictor):
    """Run inference once, shared across TestInferenceOutputs tests."""
    from pholdlib.prostt5.inference import run_prostt5_inference
    model, vocab, device = loaded_model
    return run_prostt5_inference(
        seq_dict=PHAGE_SEQS_SORTED,
        model=model,
        vocab=vocab,
        predictor=predictor,
        device=device,
        output_probs=True,
        save_per_residue_embeddings=False,
        save_per_protein_embeddings=False,
    )


class TestInferenceOutputs:
    def test_no_fail_ids(self, basic_inference):
        _, _, _, fail_ids = basic_inference
        assert fail_ids == [], f"Unexpected failures: {fail_ids}"

    def test_all_sequences_predicted(self, basic_inference):
        preds, _, _, _ = basic_inference
        for seq_id, _, _ in PHAGE_SEQS:
            assert seq_id in preds

    def test_prediction_tuple_length(self, basic_inference):
        preds, _, _, _ = basic_inference
        for seq_id, _, _ in PHAGE_SEQS:
            assert len(preds[seq_id]) == 3

    def test_pred_array_length_matches_sequence(self, basic_inference):
        preds, _, _, _ = basic_inference
        for seq_id, _, expected_len in PHAGE_SEQS:
            pred_arr, _, _ = preds[seq_id]
            assert len(pred_arr) == expected_len, (
                f"{seq_id}: expected {expected_len} residues, got {len(pred_arr)}"
            )

    def test_pred_values_are_valid_3di_classes(self, basic_inference):
        preds, _, _, _ = basic_inference
        for seq_id, _, _ in PHAGE_SEQS:
            pred_arr, _, _ = preds[seq_id]
            assert np.all((pred_arr >= 0) & (pred_arr <= 20)), (
                f"{seq_id}: out-of-range 3Di indices: {np.unique(pred_arr)}"
            )

    def test_mean_prob_is_valid_percentage(self, basic_inference):
        preds, _, _, _ = basic_inference
        for seq_id, _, _ in PHAGE_SEQS:
            _, mean_prob, _ = preds[seq_id]
            assert 0.0 <= mean_prob <= 100.0, (
                f"{seq_id}: mean_prob={mean_prob} outside [0, 100]"
            )

    def test_all_probs_shape(self, basic_inference):
        preds, _, _, _ = basic_inference
        for seq_id, _, seq_len in PHAGE_SEQS:
            _, _, all_prob = preds[seq_id]
            assert all_prob is not None
            assert all_prob.shape[-1] == seq_len, (
                f"{seq_id}: all_prob.shape[-1]={all_prob.shape[-1]}, expected {seq_len}"
            )

    def test_all_probs_values_in_unit_interval(self, basic_inference):
        preds, _, _, _ = basic_inference
        for seq_id, _, _ in PHAGE_SEQS:
            _, _, all_prob = preds[seq_id]
            assert np.all(all_prob >= 0.0) and np.all(all_prob <= 1.0), (
                f"{seq_id}: probabilities outside [0, 1]"
            )

    def test_output_probs_false_gives_none(self, loaded_model, predictor):
        from pholdlib.prostt5.inference import run_prostt5_inference
        model, vocab, device = loaded_model
        preds, _, _, _ = run_prostt5_inference(
            seq_dict=PHAGE_SEQS_SORTED[:2],
            model=model, vocab=vocab, predictor=predictor, device=device,
            output_probs=False,
        )
        for seq_id, _, _ in PHAGE_SEQS_SORTED[:2]:
            _, _, all_prob = preds[seq_id]
            assert all_prob is None

    def test_single_sequence_batch(self, loaded_model, predictor):
        from pholdlib.prostt5.inference import run_prostt5_inference
        model, vocab, device = loaded_model
        single = [PHAGE_SEQS_SORTED[0]]
        preds, _, _, fail_ids = run_prostt5_inference(
            seq_dict=single, model=model, vocab=vocab,
            predictor=predictor, device=device,
        )
        assert fail_ids == []
        seq_id, _, seq_len = single[0]
        assert len(preds[seq_id][0]) == seq_len

    def test_nonstandard_aa_substitution(self, loaded_model, predictor):
        """Sequences with U, Z, O should be handled (replaced with X) without errors."""
        from pholdlib.prostt5.inference import run_prostt5_inference
        model, vocab, device = loaded_model
        exotic = [("exotic", "MKTUZO", 6)]
        preds, _, _, fail_ids = run_prostt5_inference(
            seq_dict=exotic, model=model, vocab=vocab,
            predictor=predictor, device=device,
        )
        assert fail_ids == []
        assert "exotic" in preds
        assert len(preds["exotic"][0]) == 6

    def test_result_order_matches_input(self, loaded_model, predictor):
        """predictions dict preserves insertion order matching original_keys."""
        from pholdlib.prostt5.inference import run_prostt5_inference
        model, vocab, device = loaded_model
        preds, _, _, _ = run_prostt5_inference(
            seq_dict=PHAGE_SEQS_SORTED, model=model, vocab=vocab,
            predictor=predictor, device=device,
        )
        assert list(preds.keys()) == [s[0] for s in PHAGE_SEQS_SORTED]


# ===========================================================================
# Embedding outputs
# ===========================================================================

@pytest.fixture(scope="module")
def embedding_inference(loaded_model, predictor):
    """Run inference with embeddings enabled; shared across TestEmbeddingOutputs."""
    from pholdlib.prostt5.inference import run_prostt5_inference
    model, vocab, device = loaded_model
    return run_prostt5_inference(
        seq_dict=PHAGE_SEQS_SORTED,
        model=model, vocab=vocab, predictor=predictor, device=device,
        output_probs=False,
        save_per_residue_embeddings=True,
        save_per_protein_embeddings=True,
    )


class TestEmbeddingOutputs:
    def test_per_residue_keys_match_input(self, embedding_inference):
        _, emb_res, _, _ = embedding_inference
        assert set(emb_res.keys()) == {s[0] for s in PHAGE_SEQS}

    def test_per_residue_shape(self, embedding_inference):
        _, emb_res, _, _ = embedding_inference
        for seq_id, _, seq_len in PHAGE_SEQS:
            assert emb_res[seq_id].shape == (seq_len, 1024), (
                f"{seq_id}: expected ({seq_len}, 1024), got {emb_res[seq_id].shape}"
            )

    def test_per_protein_keys_match_input(self, embedding_inference):
        _, _, emb_prot, _ = embedding_inference
        assert set(emb_prot.keys()) == {s[0] for s in PHAGE_SEQS}

    def test_per_protein_shape(self, embedding_inference):
        _, _, emb_prot, _ = embedding_inference
        for seq_id, _, _ in PHAGE_SEQS:
            assert emb_prot[seq_id].shape == (1024,), (
                f"{seq_id}: expected (1024,), got {emb_prot[seq_id].shape}"
            )

    def test_per_protein_equals_mean_of_per_residue(self, embedding_inference):
        _, emb_res, emb_prot, _ = embedding_inference
        for seq_id, _, _ in PHAGE_SEQS:
            expected = emb_res[seq_id].mean(axis=0)
            np.testing.assert_allclose(
                emb_prot[seq_id], expected, rtol=1e-4,
                err_msg=f"{seq_id}: per-protein ≠ mean(per-residue)",
            )

    def test_embeddings_empty_when_flags_false(self, loaded_model, predictor):
        from pholdlib.prostt5.inference import run_prostt5_inference
        model, vocab, device = loaded_model
        _, emb_res, emb_prot, _ = run_prostt5_inference(
            seq_dict=PHAGE_SEQS_SORTED[:1],
            model=model, vocab=vocab, predictor=predictor, device=device,
            save_per_residue_embeddings=False,
            save_per_protein_embeddings=False,
        )
        assert emb_res == {}
        assert emb_prot == {}


# ===========================================================================
# Output-writing round-trip
# ===========================================================================

class TestOutputRoundTrip:
    def test_write_probs_mean_csv(self, basic_inference, tmp_path):
        from pholdlib.prostt5.output import write_probs
        preds, _, _, _ = basic_inference
        original_keys = [s[0] for s in PHAGE_SEQS_SORTED]

        mean_path = tmp_path / "mean.csv"
        write_probs(preds, mean_path, None, original_keys)

        lines = mean_path.read_text().strip().splitlines()
        assert len(lines) == len(PHAGE_SEQS)
        for line in lines:
            sid, prob_str = line.split(",")
            assert sid in {s[0] for s in PHAGE_SEQS}
            val = float(prob_str)
            assert 0.0 <= val <= 100.0

    def test_write_probs_jsonl(self, basic_inference, tmp_path):
        from pholdlib.prostt5.output import write_probs
        preds, _, _, _ = basic_inference
        original_keys = [s[0] for s in PHAGE_SEQS_SORTED]
        seq_len_map = {sid: slen for sid, _, slen in PHAGE_SEQS}

        mean_path = tmp_path / "mean.csv"
        all_path = tmp_path / "all.jsonl"
        write_probs(preds, mean_path, all_path, original_keys)

        records = [json.loads(l) for l in all_path.read_text().strip().splitlines()]
        assert len(records) == len(PHAGE_SEQS)
        # Records are written in original_keys order (= PHAGE_SEQS_SORTED order)
        assert [r["seq_id"] for r in records] == original_keys
        for rec in records:
            seq_len = seq_len_map[rec["seq_id"]]
            assert len(rec["probability"]) == seq_len
            assert all(0.0 <= p <= 100.0 for p in rec["probability"])

    def test_write_probs_key_order_preserved(self, basic_inference, tmp_path):
        from pholdlib.prostt5.output import write_probs
        preds, _, _, _ = basic_inference
        reversed_keys = [s[0] for s in reversed(PHAGE_SEQS_SORTED)]

        mean_path = tmp_path / "mean_rev.csv"
        write_probs(preds, mean_path, None, reversed_keys)

        written_ids = [l.split(",")[0] for l in mean_path.read_text().strip().splitlines()]
        assert written_ids == reversed_keys

    def test_mask_low_confidence_aa_with_real_probs(self, basic_inference):
        """mask_low_confidence_aa should never produce more X than residues."""
        from pholdlib.prostt5.output import mask_low_confidence_aa
        preds, _, _, _ = basic_inference
        for seq_id, seq, _ in PHAGE_SEQS:
            _, _, all_prob = preds[seq_id]
            masked = mask_low_confidence_aa(seq, all_prob, threshold=0.5)
            assert len(masked) == len(seq)
            assert set(masked).issubset(set(seq) | {"X"})
