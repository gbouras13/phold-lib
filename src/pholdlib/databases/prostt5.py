#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ProstT5 model download and integrity-check helpers.

The standard model MD5s (PROSTT5_MD5_DICTIONARY) are shipped here since both
phold and baktfold use the same Rostlab/ProstT5_fp16 weights.

Tool-specific model variants (e.g. phold's fine-tuned gbouras13/ProstT5Phold)
should define their own MD5 dicts and pass them via the md5_dict argument.
"""

import hashlib
import sys
import tarfile
from pathlib import Path
from typing import Dict, Optional

from alive_progress import alive_bar
from loguru import logger
import requests


# MD5 hashes for Rostlab/ProstT5_fp16 — identical in phold and baktfold.
PROSTT5_MD5_DICTIONARY: Dict = {
    "refs": {"main": "962133e8e2bff04ec1768fa58dd788f3"},
    "blobs": {
        "2c19eb6e3b583f52d34b903b5978d3d30b6b7682": "8fd03e945174de0818746ecbde1aad8e",
        "60fe6bb247c90b8545d7b73820cd796ce6dcbd59": "ce32377619690072ced051ec7fc6c5f0",
        "6fc7be92c58e238f20a6cdea5a87b123a4ad35e2": "1deb27443d0d9b90b8e5704e752373e2",
        "74da7b4afcde53faa570114b530c726135bdfcdb813dec3abfb27f9d44db7324": "6ad28d9980aaec37d3935072d204e520",
        "b1a9ffcef73280cc57f090ad6446b4116b574b6c75d83ccc32778282f7f00855": "45f066fc3b0d87fb9f98bb0ddb97a3dc",
        "e9322396e6e75ecf8da41a9527e24dfa4eeea505": "b1cdd31ea50a37bf84cc0d7ef11820c8",
    },
}

# Map model_name -> HuggingFace cache sub-directory name.
_MODEL_SUBDIR: Dict[str, str] = {
    "Rostlab/ProstT5_fp16": "models--Rostlab--ProstT5_fp16",
}


def check_prostT5_download(
    model_dir: Path,
    model_name: str,
    md5_dict: Optional[Dict] = None,
    model_subdir: Optional[str] = None,
) -> bool:
    """
    Check whether a ProstT5 model is present and uncorrupted.

    Args:
        model_dir: Directory where HuggingFace caches models.
        model_name: HuggingFace model identifier.
        md5_dict: MD5 dictionary for this model. Defaults to PROSTT5_MD5_DICTIONARY
                  (Rostlab/ProstT5_fp16). Pass a custom dict for fine-tuned variants.
        model_subdir: Override the HuggingFace cache sub-directory name.
                      If None, derived from _MODEL_SUBDIR or model_name.

    Returns:
        True if the model needs to be (re)downloaded, False if it is intact.
    """
    if md5_dict is None:
        md5_dict = PROSTT5_MD5_DICTIONARY

    if model_subdir is None:
        if model_name in _MODEL_SUBDIR:
            model_subdir = _MODEL_SUBDIR[model_name]
        else:
            # Derive from model_name: "org/Name" -> "models--org--Name"
            model_subdir = "models--" + model_name.replace("/", "--")

    download = False
    for key, nested in md5_dict.items():
        for nested_key, expected_md5 in nested.items():
            file_path = Path(model_dir) / model_subdir / key / nested_key
            if file_path.exists():
                actual_md5 = _calc_md5(file_path)
                if actual_md5 != expected_md5:
                    logger.warning(
                        f"Corrupt model file {file_path}! "
                        f"MD5 should be '{expected_md5}' but is '{actual_md5}'"
                    )
                    download = True
            else:
                logger.warning(f"Model file {file_path} does not exist.")
                download = True

    return download


def download_zenodo_prostT5(
    model_dir: Path,
    logdir: str,
    threads: int,
    backup_url: str,
    backup_md5: str,
    backup_tarball: str,
) -> None:
    """
    Download ProstT5 model from a Zenodo backup and extract it.

    Args:
        model_dir: Destination directory.
        logdir: Unused (kept for API compatibility with existing callers).
        threads: Unused (kept for API compatibility with existing callers).
        backup_url: Direct download URL for the tarball.
        backup_md5: Expected MD5 of the tarball.
        backup_tarball: Filename for the downloaded tarball.
    """
    logger.info(f"Downloading ProstT5 model backup from {backup_url}")
    tarball_path = Path(model_dir) / backup_tarball

    _download_requests(backup_url, tarball_path)

    md5_sum = _calc_md5(tarball_path)
    if md5_sum == backup_md5:
        logger.info(f"ProstT5 model backup file download OK: {md5_sum}")
    else:
        logger.error(
            f"Error: corrupt file! MD5 should be '{backup_md5}' but is '{md5_sum}'"
        )

    logger.info(f"Extracting ProstT5 model backup: {tarball_path} -> {model_dir}")
    try:
        with tarball_path.open("rb") as fh_in, tarfile.open(fileobj=fh_in, mode="r:gz") as tar:
            _safe_extractall(tar, model_dir)
    except OSError as exc:
        logger.warning(f"Encountered OSError: {exc}")
        logger.error(f"Could not extract {tarball_path} to {model_dir}")

    tarball_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_extractall(tar: tarfile.TarFile, dest: Path) -> None:
    """Extract ``tar`` into ``dest`` with path-traversal protection.

    Plain ``TarFile.extractall`` honours member names like ``../../etc/passwd``,
    so a corrupted or tampered-with tarball can write outside ``dest``. The
    mitigation is version-dependent:

    * **Python 3.12+**: delegate to CPython's ``filter="data"``, which strips
      leading separators from absolute names, rejects ``..`` escapes, blocks
      links whose targets leave the destination, and disallows special file
      modes. It becomes the default in 3.14; requesting it explicitly also
      silences the 3.12/3.13 DeprecationWarning.
    * **Python 3.8-3.11**: the equivalent checks by hand — resolve each
      member (and link target) under ``dest`` and reject anything outside.

    Note that the two branches differ on *absolute* member names: the data
    filter strips the leading separator and extracts to ``dest/etc/passwd``,
    whereas the manual branch rejects the member outright. Both keep the
    write inside ``dest``, which is the property that matters.

    Raises:
        tarfile.TarError: if a member or link target escapes ``dest``.

    See https://docs.python.org/3.12/library/tarfile.html#extraction-filters
    and PEP 706 for the full filter contract.
    """
    dest = Path(dest).resolve()

    if sys.version_info >= (3, 12):
        tar.extractall(path=str(dest), filter="data")
        return

    for member in tar.getmembers():
        member_dest = (dest / member.name).resolve()
        try:
            member_dest.relative_to(dest)
        except ValueError:
            raise tarfile.TarError(
                f"Refusing to extract '{member.name}' from tarball: "
                f"resolved path {member_dest} escapes destination {dest}"
            )
        if member.issym() or member.islnk():
            link_target = (member_dest.parent / member.linkname).resolve()
            try:
                link_target.relative_to(dest)
            except ValueError:
                raise tarfile.TarError(
                    f"Refusing to extract link '{member.name}' -> "
                    f"'{member.linkname}' from tarball: target {link_target} "
                    f"escapes destination {dest}"
                )

    tar.extractall(path=str(dest))


def _calc_md5(file_path: Path, buffer_size: int = 1024 * 1024) -> str:
    """Return the MD5 hex digest of file_path."""
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(buffer_size):
            md5.update(chunk)
    return md5.hexdigest()


def _download_requests(url: str, dest: Path) -> None:
    """Stream-download url to dest with a progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(dest, "wb") as fh_out, alive_bar(total, unit="B", scale="SI") as bar:
            for data in response.iter_content(chunk_size=1024 * 1024):
                fh_out.write(data)
                bar(count=len(data))
    except Exception:
        logger.error(f"Could not download file from {url} to {dest}")
        raise
