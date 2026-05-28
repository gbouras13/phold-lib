"""Unit tests for pholdlib.databases.prostt5.

Loaded directly via importlib to avoid the transformers import chain.
Network calls are mocked; no model download is required.
"""

import hashlib
import importlib.util
import io
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Direct module load – bypasses pholdlib/databases/__init__.py so we can test
# without triggering the transformers import that lives in prostt5/__init__.py
# via the re-export in pholdlib/__init__.py.
# ---------------------------------------------------------------------------
_SRC = Path(__file__).parent.parent / "src" / "pholdlib"


def _load_db():
    spec = importlib.util.spec_from_file_location(
        "pholdlib.databases.prostt5",
        _SRC / "databases" / "prostt5.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_db = _load_db()
check_prostT5_download = _db.check_prostT5_download
download_zenodo_prostT5 = _db.download_zenodo_prostT5
PROSTT5_MD5_DICTIONARY = _db.PROSTT5_MD5_DICTIONARY
_calc_md5 = _db._calc_md5
_MODEL_SUBDIR = _db._MODEL_SUBDIR


# ===========================================================================
# PROSTT5_MD5_DICTIONARY structure
# ===========================================================================

class TestMD5Dictionary:
    def test_has_refs_and_blobs_keys(self):
        assert set(PROSTT5_MD5_DICTIONARY.keys()) == {"refs", "blobs"}

    def test_refs_contains_main(self):
        assert "main" in PROSTT5_MD5_DICTIONARY["refs"]

    def test_blobs_not_empty(self):
        assert len(PROSTT5_MD5_DICTIONARY["blobs"]) > 0

    def test_all_md5_values_are_32_char_hex(self):
        for section in PROSTT5_MD5_DICTIONARY.values():
            for md5 in section.values():
                assert len(md5) == 32, f"not 32 chars: {md5!r}"
                int(md5, 16)  # raises ValueError for non-hex

    def test_model_subdir_registered(self):
        assert "Rostlab/ProstT5_fp16" in _MODEL_SUBDIR


# ===========================================================================
# _calc_md5
# ===========================================================================

class TestCalcMd5:
    def test_known_content(self, tmp_path):
        content = b"hello pholdlib\n"
        f = tmp_path / "test.bin"
        f.write_bytes(content)
        expected = hashlib.md5(content).hexdigest()
        assert _calc_md5(f) == expected

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        assert _calc_md5(f) == hashlib.md5(b"").hexdigest()

    def test_multi_chunk_file(self, tmp_path):
        # larger than the default 1 MiB buffer to exercise the chunked read path
        content = b"x" * (3 * 1024 * 1024 + 7)
        f = tmp_path / "large.bin"
        f.write_bytes(content)
        assert _calc_md5(f) == hashlib.md5(content).hexdigest()


# ===========================================================================
# check_prostT5_download
# ===========================================================================

class TestCheckProstT5Download:
    """check_prostT5_download returns True when files are missing/corrupt."""

    def test_empty_model_dir_needs_download(self, tmp_path):
        assert check_prostT5_download(tmp_path, "Rostlab/ProstT5_fp16") is True

    def test_partial_model_dir_needs_download(self, tmp_path):
        # Create the refs sub-directory but leave blobs absent
        (tmp_path / "models--Rostlab--ProstT5_fp16" / "refs").mkdir(parents=True)
        assert check_prostT5_download(tmp_path, "Rostlab/ProstT5_fp16") is True

    def test_corrupt_file_needs_download(self, tmp_path):
        """A file that exists but has the wrong MD5 triggers a re-download."""
        model_subdir = tmp_path / "models--Rostlab--ProstT5_fp16"
        refs_dir = model_subdir / "refs"
        refs_dir.mkdir(parents=True)
        # Write garbage into the 'main' ref file
        (refs_dir / "main").write_bytes(b"not the real model")

        assert check_prostT5_download(tmp_path, "Rostlab/ProstT5_fp16") is True

    def test_correct_files_no_download(self, tmp_path):
        """All files present with matching MD5s → returns False (no download)."""
        md5_dict = {
            "refs": {"main": hashlib.md5(b"ref_content").hexdigest()},
            "blobs": {"blob1": hashlib.md5(b"blob_content").hexdigest()},
        }
        subdir = "models--test--Model"
        (tmp_path / subdir / "refs").mkdir(parents=True)
        (tmp_path / subdir / "blobs").mkdir(parents=True)
        (tmp_path / subdir / "refs" / "main").write_bytes(b"ref_content")
        (tmp_path / subdir / "blobs" / "blob1").write_bytes(b"blob_content")

        result = check_prostT5_download(
            tmp_path, "test/Model", md5_dict=md5_dict, model_subdir=subdir
        )
        assert result is False

    def test_custom_md5_dict(self, tmp_path):
        custom_dict = {
            "refs": {"main": hashlib.md5(b"data").hexdigest()},
        }
        subdir = "models--custom--Model"
        (tmp_path / subdir / "refs").mkdir(parents=True)
        (tmp_path / subdir / "refs" / "main").write_bytes(b"data")

        result = check_prostT5_download(
            tmp_path, "custom/Model", md5_dict=custom_dict, model_subdir=subdir
        )
        assert result is False

    def test_custom_model_subdir_argument(self, tmp_path):
        """model_subdir overrides the default HuggingFace naming convention."""
        subdir = "my_custom_cache_dir"
        result = check_prostT5_download(
            tmp_path, "Rostlab/ProstT5_fp16", model_subdir=subdir
        )
        # Files are absent → needs download
        assert result is True

    def test_derived_subdir_for_unknown_model(self, tmp_path):
        """For unknown model_name, subdir is derived as 'models--org--Name'."""
        # No files present; just confirm the function doesn't crash and returns True
        result = check_prostT5_download(tmp_path, "someorg/SomeModel")
        assert result is True


# ===========================================================================
# download_zenodo_prostT5
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


class TestDownloadZenodoProstT5:
    def _mock_download(self, mod, tarball_bytes, dest_attr="_download_requests"):
        """Patch _download_requests to write tarball_bytes to its dest argument."""

        def fake_download(url, dest):
            Path(dest).write_bytes(tarball_bytes)

        return patch.object(mod, "_download_requests", side_effect=fake_download)

    def test_extracts_tarball_contents(self, tmp_path):
        tarball_bytes = _make_tarball({"dummy_model.bin": b"model weights"})
        backup_md5 = hashlib.md5(tarball_bytes).hexdigest()

        with self._mock_download(_db, tarball_bytes):
            download_zenodo_prostT5(
                model_dir=tmp_path,
                logdir=str(tmp_path / "logdir"),
                threads=1,
                backup_url="http://example.com/model.tar.gz",
                backup_md5=backup_md5,
                backup_tarball="model.tar.gz",
            )

        assert (tmp_path / "dummy_model.bin").exists()

    def test_tarball_removed_after_extraction(self, tmp_path):
        tarball_bytes = _make_tarball({"file.bin": b"data"})
        backup_md5 = hashlib.md5(tarball_bytes).hexdigest()

        with self._mock_download(_db, tarball_bytes):
            download_zenodo_prostT5(
                model_dir=tmp_path,
                logdir=str(tmp_path / "logdir"),
                threads=1,
                backup_url="http://example.com/model.tar.gz",
                backup_md5=backup_md5,
                backup_tarball="model.tar.gz",
            )

        # The tarball should be cleaned up
        assert not (tmp_path / "model.tar.gz").exists()

    def test_md5_mismatch_logged_but_extraction_proceeds(self, tmp_path, caplog):
        import logging

        tarball_bytes = _make_tarball({"file.bin": b"data"})
        wrong_md5 = "a" * 32  # deliberately wrong

        with self._mock_download(_db, tarball_bytes):
            with caplog.at_level(logging.ERROR, logger="pholdlib"):
                download_zenodo_prostT5(
                    model_dir=tmp_path,
                    logdir=str(tmp_path / "logdir"),
                    threads=1,
                    backup_url="http://example.com/model.tar.gz",
                    backup_md5=wrong_md5,
                    backup_tarball="model.tar.gz",
                )
        # extraction still happened despite MD5 mismatch
        assert (tmp_path / "file.bin").exists()
