"""Unit tests for the ModernProst model registry and inference helpers.

The registry tests import ``pholdlib.databases.modernprost`` directly (no
torch). The inference-helper tests need ``pholdlib.modernprost.inference``,
which imports torch, and skip cleanly when torch is unavailable.
"""

import hashlib
import importlib.util
import io
import sys
import tarfile
from pathlib import Path
from unittest.mock import patch

import pytest

_SRC = Path(__file__).parent.parent / "src" / "pholdlib"


def _load(mod_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _SRC / mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# prostt5.py first — modernprost.py does `from pholdlib.databases.prostt5
# import ...`, which otherwise resolves against whatever pholdlib is installed
# in the environment rather than this checkout. Registering the local copy in
# sys.modules keeps the tests bound to the source under test.
_load("databases/prostt5.py", "pholdlib.databases.prostt5")
_registry = _load("databases/modernprost.py", "pholdlib.databases.modernprost")

MODERNPROST_MODELS = _registry.MODERNPROST_MODELS
resolve_modernprost_model = _registry.resolve_modernprost_model
is_modernprost_model = _registry.is_modernprost_model
default_task_for = _registry.default_task_for
check_modernprost_download = _registry.check_modernprost_download
MODERNPROST_SNAPSHOT_FILES = _registry.MODERNPROST_SNAPSHOT_FILES
CLASSIFICATION_TASK = _registry.CLASSIFICATION_TASK
PSSM_TASK = _registry.PSSM_TASK


# ===========================================================================
# Registry
# ===========================================================================

class TestRegistry:
    def test_all_four_published_models_present(self):
        assert set(MODERNPROST_MODELS) == {
            "modernprost-base",
            "modernprost-50M",
            "modernprost-pssm",
            "modernprost-50M-pssm",
        }

    def test_resolves_by_short_name_and_hf_id(self):
        by_short = resolve_modernprost_model("modernprost-50M")
        by_hf = resolve_modernprost_model("gbouras13/modernprost-50M")
        assert by_short is by_hf

    def test_unknown_model_lists_valid_names(self):
        with pytest.raises(KeyError, match="modernprost-50M"):
            resolve_modernprost_model("modernprost-tiny")

    def test_is_modernprost_model(self):
        assert is_modernprost_model("modernprost-base")
        assert is_modernprost_model("gbouras13/modernprost-pssm")
        assert not is_modernprost_model("Rostlab/ProstT5_fp16")

    def test_pssm_models_default_to_the_pssm_task(self):
        assert default_task_for("modernprost-pssm") == PSSM_TASK
        assert default_task_for("modernprost-50M-pssm") == PSSM_TASK

    def test_non_pssm_models_default_to_classification(self):
        assert default_task_for("modernprost-base") == CLASSIFICATION_TASK
        assert default_task_for("modernprost-50M") == CLASSIFICATION_TASK

    def test_cache_subdir_matches_huggingface_layout(self):
        spec = resolve_modernprost_model("modernprost-50M-pssm")
        assert spec.cache_subdir == "models--gbouras13--modernprost-50M-pssm"

    def test_short_name_round_trips(self):
        for name, spec in MODERNPROST_MODELS.items():
            assert spec.short_name == name


# ===========================================================================
# Download / integrity check
# ===========================================================================

def _make_snapshot(root: Path, model: str, commit: str = "abc123", omit=()):
    """Build a minimal HuggingFace cache layout for *model*."""
    cache = root / f"models--gbouras13--{model}"
    (cache / "refs").mkdir(parents=True, exist_ok=True)
    (cache / "refs" / "main").write_text(commit)
    snapshot = cache / "snapshots" / commit
    snapshot.mkdir(parents=True, exist_ok=True)
    for filename in MODERNPROST_SNAPSHOT_FILES:
        if filename in omit:
            continue
        (snapshot / filename).write_text("x")
    return cache, snapshot


class TestCheckDownload:
    def test_complete_snapshot_needs_no_download(self, tmp_path):
        _make_snapshot(tmp_path, "modernprost-50M")
        assert check_modernprost_download(tmp_path, "modernprost-50M") is False

    def test_absent_model_needs_download(self, tmp_path):
        assert check_modernprost_download(tmp_path, "modernprost-50M") is True

    def test_missing_weights_needs_download(self, tmp_path):
        _make_snapshot(tmp_path, "modernprost-50M", omit=("model.safetensors",))
        assert check_modernprost_download(tmp_path, "modernprost-50M") is True

    def test_missing_custom_code_needs_download(self, tmp_path):
        # trust_remote_code loading fails without the modeling module, so a
        # snapshot missing it is as unusable as one missing the weights.
        _make_snapshot(
            tmp_path, "modernprost-50M", omit=("modeling_modernprost_3di12st.py",)
        )
        assert check_modernprost_download(tmp_path, "modernprost-50M") is True

    def test_empty_file_needs_download(self, tmp_path):
        _, snapshot = _make_snapshot(tmp_path, "modernprost-50M")
        (snapshot / "model.safetensors").write_text("")
        assert check_modernprost_download(tmp_path, "modernprost-50M") is True

    def test_dangling_ref_needs_download(self, tmp_path):
        cache, _ = _make_snapshot(tmp_path, "modernprost-50M")
        (cache / "refs" / "main").write_text("does-not-exist")
        assert check_modernprost_download(tmp_path, "modernprost-50M") is True

    def test_explicit_md5_dict_is_used(self, tmp_path):
        import hashlib

        cache = tmp_path / "models--gbouras13--modernprost-50M"
        (cache / "blobs").mkdir(parents=True)
        blob = cache / "blobs" / "deadbeef"
        blob.write_text("payload")
        good = hashlib.md5(b"payload").hexdigest()

        assert (
            check_modernprost_download(
                tmp_path, "modernprost-50M", md5_dict={"blobs": {"deadbeef": good}}
            )
            is False
        )
        assert (
            check_modernprost_download(
                tmp_path, "modernprost-50M", md5_dict={"blobs": {"deadbeef": "0" * 32}}
            )
            is True
        )


# ===========================================================================
# Inference helpers
# ===========================================================================

torch = pytest.importorskip("torch", reason="inference.py imports torch")
_inference = _load("modernprost/inference.py", "pholdlib.modernprost.inference")

chunk_sequence = _inference.chunk_sequence
_split_chunk_id = _inference._split_chunk_id
ModernProstResult = _inference.ModernProstResult


class TestChunkSequence:
    def test_short_sequence_is_one_chunk(self):
        assert [s for _, s in chunk_sequence("MKTII", 10)] == ["MKTII"]

    def test_chunks_cover_sequence_exactly(self):
        seq = "".join("ACDEFGHIKL"[i % 10] for i in range(1000))
        chunks = [s for _, s in chunk_sequence(seq, 300)]
        assert "".join(chunks) == seq

    def test_chunk_sizes_differ_by_at_most_one(self):
        seq = "A" * 1000
        sizes = [len(s) for _, s in chunk_sequence(seq, 300)]
        assert max(sizes) - min(sizes) <= 1
        assert all(size <= 300 for size in sizes)

    def test_starts_are_cumulative(self):
        seq = "A" * 100
        starts = [start for start, _ in chunk_sequence(seq, 30)]
        assert starts == [0, 25, 50, 75]

    def test_exact_multiple(self):
        sizes = [len(s) for _, s in chunk_sequence("A" * 600, 300)]
        assert sizes == [300, 300]


class TestSplitChunkId:
    def test_plain_id(self):
        assert _split_chunk_id("cds_0001") == ("cds_0001", None)

    def test_chunked_id(self):
        assert _split_chunk_id("cds_0001__chunk3") == ("cds_0001", 3)

    def test_id_containing_the_marker_resolves_to_the_appended_one(self):
        assert _split_chunk_id("weird__chunkname__chunk2") == ("weird__chunkname", 2)

    def test_marker_without_an_index_is_left_alone(self):
        assert _split_chunk_id("weird__chunkname") == ("weird__chunkname", None)


class TestModernProstResult:
    def test_update_merges_every_store(self):
        a = ModernProstResult(
            predictions_3di={"x": (1, 2, 3)}, fail_ids=["f1"]
        )
        b = ModernProstResult(
            predictions_3di={"y": (4, 5, 6)},
            predictions_12st={"y": (7, 8, 9)},
            fail_ids=["f2"],
        )
        a.update(b)
        assert set(a.predictions_3di) == {"x", "y"}
        assert set(a.predictions_12st) == {"y"}
        assert a.fail_ids == ["f1", "f2"]

    def test_defaults_are_independent_between_instances(self):
        a, b = ModernProstResult(), ModernProstResult()
        a.fail_ids.append("x")
        assert b.fail_ids == []


# ===========================================================================
# download_zenodo_modernprost — extraction hardening
# ===========================================================================

def _make_tarball(members: dict) -> bytes:
    """Build an in-memory .tar.gz with {filename: content} members."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class TestDownloadZenodoModernProstExtraction:
    """The Zenodo backup path shares prostt5's _safe_extractall.

    Mirrors TestDownloadZenodoProstT5 in tests/test_databases.py; the helper
    itself is covered there, so this only pins that this call site is wired
    to it rather than to a bare extractall.
    """

    def _run(self, model_dir, tarball_bytes):
        def fake_download(url, dest):
            Path(dest).write_bytes(tarball_bytes)

        with patch.object(_registry, "_download_requests", side_effect=fake_download):
            _registry.download_zenodo_modernprost(
                model_dir=model_dir,
                logdir=str(model_dir / "logdir"),
                threads=1,
                backup_url="http://example.com/model.tar.gz",
                backup_md5=hashlib.md5(tarball_bytes).hexdigest(),
                backup_tarball="model.tar.gz",
            )

    def test_extracts_legitimate_tarball(self, tmp_path):
        self._run(tmp_path, _make_tarball({"dummy_model.bin": b"model weights"}))
        assert (tmp_path / "dummy_model.bin").exists()

    def test_traversal_tarball_rejected(self, tmp_path):
        model_dir = tmp_path / "models"
        model_dir.mkdir()

        with pytest.raises(tarfile.TarError):
            self._run(model_dir, _make_tarball({"../escaped.bin": b"pwned"}))

        assert not (tmp_path / "escaped.bin").exists()

    def test_absolute_member_never_escapes_dest(self, tmp_path):
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        victim = tmp_path / "victim.bin"

        try:
            self._run(model_dir, _make_tarball({str(victim): b"pwned"}))
        except tarfile.TarError:
            pass  # 3.8-3.11 manual branch rejects; 3.12+ data filter strips

        assert not victim.exists()
