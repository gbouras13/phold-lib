"""Integration tests: the ModernProst engines with real checkpoints (CPU).

Opt-in, because they download modernprost-50M and modernprost-50M-pssm
(~210 MB each) into ``--model_dir`` if not cached:

    pytest tests/test_modernprost_integration.py --run_modernprost_integration
    pytest tests/test_modernprost_integration.py --run_modernprost_integration --model_dir /path/to/cache

On CPU the packed engine runs FlexAttention eagerly (slow, so inputs are
small); the point is that the packed forward reproduces the checkpoint's own
padded forward, not speed.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tests.conftest import HAS_TRANSFORMERS  # noqa: E402

pytestmark = pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not importable")

SEQS = [
    ("prot_A", "MKTIIALSYIFCLVFADYKDDDDKMSKGEELFTGVVPILVELDGDVNGHKFSVSGEG"),
    ("prot_B", "MASMTGGQQMGRDLYDDDDKDHHHHHH"),
    ("prot_C", "MVKVYAPASSANMSVGFDVLGAAVTPVDGALLGDVVTVEAAETFSLNNLGRFADKLP"),
    ("odd_chars", "mkt*BXZUO-ACDEFGH"),
    ("one", "M"),
    # longer than the 1568 chunk length: split into two chunks and stitched
    ("long", ("MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTF" * 26)[:1700]),
]
SEQ_DICT = sorted([(i, s, len(s)) for i, s in SEQS], key=lambda x: x[2], reverse=True)


def _load(model_dir, name):
    from pholdlib.databases.modernprost import check_modernprost_download
    from pholdlib.modernprost.model import get_modernprost_model

    return get_modernprost_model(model_dir, name, cpu=True, threads=4,
                                 check_fn=check_modernprost_download)


@pytest.fixture(scope="module")
def models(model_dir, run_modernprost_integration):
    if not run_modernprost_integration:
        pytest.skip("pass --run_modernprost_integration to run")
    return {name: _load(model_dir, name) for name in ("modernprost-50M", "modernprost-50M-pssm")}


@pytest.fixture
def allow_cpu_packed(monkeypatch):
    from pholdlib.modernprost import packed

    monkeypatch.setattr(packed, "_ALLOW_CPU", True)
    ok, why = packed.packed_engine_status(torch.device("cpu"))
    if not ok:
        pytest.skip(f"packed engine unavailable: {why}")


def _run(models, name, engine, **kw):
    from pholdlib.modernprost.inference import run_modernprost_inference

    model, tok, dev = models[name]
    task = "pssm" if name.endswith("pssm") else "classification"
    return run_modernprost_inference(SEQ_DICT, model, tok, dev, task=task, engine=engine,
                                     save_per_protein_embeddings=True, **kw)


class TestPaddedEngine:
    def test_classification_outputs(self, models):
        res = _run(models, "modernprost-50M", "padded")
        assert not res.fail_ids
        for sid, seq in SEQS:
            pred, mean, probs = res.predictions_3di[sid]
            assert len(pred) == len(seq) and pred.dtype == np.int8
            assert probs.dtype == np.float32 and 0.0 <= mean <= 100.0
            assert res.predictions_12st[sid][0].max() < 12
            assert res.embeddings_per_protein[sid].shape == (models["modernprost-50M"][0].config.hidden_size,)

    def test_pssm_profiles_are_distributions(self, models):
        res = _run(models, "modernprost-50M-pssm", "padded")
        for sid, seq in SEQS:
            p3, p12 = res.profiles_3di[sid], res.profiles_12st[sid]
            assert p3.shape == (len(seq), 20) and p12.shape == (len(seq), 12)
            np.testing.assert_allclose(p3.sum(1), 1.0, atol=1e-5)

    def test_independent_of_batching(self, models):
        a = _run(models, "modernprost-50M", "padded", max_residues=100000)
        b = _run(models, "modernprost-50M", "padded", max_residues=64, max_batch=1)
        for sid, _ in SEQS:
            assert np.array_equal(a.predictions_3di[sid][0], b.predictions_3di[sid][0])
            np.testing.assert_allclose(a.predictions_3di[sid][2], b.predictions_3di[sid][2], atol=1e-5)


@pytest.mark.usefixtures("allow_cpu_packed")
class TestPackedEngineMatchesPadded:
    def test_classification(self, models):
        pad = _run(models, "modernprost-50M", "padded")
        pk = _run(models, "modernprost-50M", "packed", compile="none", pack_tokens=2048)
        for sid, _ in SEQS:
            assert np.array_equal(pad.predictions_3di[sid][0], pk.predictions_3di[sid][0])
            assert np.array_equal(pad.predictions_12st[sid][0], pk.predictions_12st[sid][0])
            np.testing.assert_allclose(pad.predictions_3di[sid][2], pk.predictions_3di[sid][2], atol=1e-4)
            np.testing.assert_allclose(pad.embeddings_per_protein[sid], pk.embeddings_per_protein[sid],
                                       atol=1e-4)

    def test_pssm(self, models):
        pad = _run(models, "modernprost-50M-pssm", "padded")
        pk = _run(models, "modernprost-50M-pssm", "packed", compile="none", pack_tokens=2048)
        for sid, _ in SEQS:
            np.testing.assert_allclose(pad.profiles_3di[sid], pk.profiles_3di[sid], atol=1e-4)
            np.testing.assert_allclose(pad.profiles_12st[sid], pk.profiles_12st[sid], atol=1e-4)


class TestMultiDeviceQueue:
    def test_two_cpu_workers_match_single(self, models, model_dir):
        from pholdlib.databases.modernprost import check_modernprost_download
        from pholdlib.modernprost.inference import run_modernprost_inference_multi_gpu

        single = _run(models, "modernprost-50M", "padded")
        multi = run_modernprost_inference_multi_gpu(
            SEQ_DICT, ["cpu", "cpu"], model_dir, "modernprost-50M", half_precision=False, threads=2,
            check_fn=check_modernprost_download, engine="padded", max_residues=512)
        assert not multi.fail_ids
        for sid, _ in SEQS:
            assert np.array_equal(single.predictions_3di[sid][0], multi.predictions_3di[sid][0])
