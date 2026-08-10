#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ModernProst model registry, download and integrity-check helpers.

ModernProst is a ModernBERT-style encoder that maps an amino-acid sequence
directly to Foldseek's 3Di alphabet (20 states) *and* a 12-state secondary
structure alphabet ("12st") in a single forward pass — no ProstT5 encoder and
no separate CNN prediction head.

Four checkpoints are published on HuggingFace, all sharing the same
``modernprost_3di12st`` custom architecture (``ModernProst3Di12StModel``) and
the same 29-token residue vocabulary:

============================  =========  ==========  ==================
model                         size       task        Foldseek query DB
============================  =========  ==========  ==================
``modernprost-base``          ~1.0 B     argmax      combined ``_ss``
``modernprost-50M``           ~53 M      argmax      combined ``_ss``
``modernprost-pssm``          ~1.0 B     profiles    ``_profile_ss*``
``modernprost-50M-pssm``      ~53 M      profiles    ``_profile_ss*``
============================  =========  ==========  ==================

The ``-pssm`` checkpoints are trained to produce calibrated per-residue
distributions rather than confident argmax states, so their softmax output is
used directly as a Foldseek profile. The others are used in "classification"
mode: argmax 3Di + argmax 12st written as FASTA strings.

Integrity checking mirrors :mod:`pholdlib.databases.prostt5`: an MD5
dictionary keyed by HuggingFace cache blob path. Where no MD5 dictionary is
known for a model, the check degrades to a presence check of the essential
snapshot files, which still catches the interrupted-download case that
motivated the MD5 check (phold issue #44) without pinning blob hashes that
change on every re-upload.
"""

import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from loguru import logger

from pholdlib.databases.prostt5 import _calc_md5, _download_requests, _safe_extractall


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CLASSIFICATION_TASK = "classification"
PSSM_TASK = "pssm"
VALID_TASKS = (CLASSIFICATION_TASK, PSSM_TASK)


@dataclass(frozen=True)
class ModernProstModel:
    """Static metadata for one published ModernProst checkpoint.

    Attributes:
        hf_name: Full HuggingFace repo id, as passed to ``from_pretrained``.
        default_task: ``"classification"`` (argmax 3Di + 12st strings) or
            ``"pssm"`` (per-residue softmax profiles). Callers may override,
            but this is what the checkpoint was trained for.
        hidden_size / num_hidden_layers / num_attention_heads: architecture,
            duplicated here only so ``phold predict`` can log which model it
            is about to download before any weights are fetched.
        approx_params: human-readable parameter count for log messages.
        download_mb: approximate on-disk size of ``model.safetensors``, used
            to warn before a multi-GB download starts.
    """

    hf_name: str
    default_task: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    approx_params: str
    download_mb: int

    @property
    def cache_subdir(self) -> str:
        """HuggingFace cache directory name, e.g. ``models--gbouras13--modernprost-50M``."""
        return "models--" + self.hf_name.replace("/", "--")

    @property
    def short_name(self) -> str:
        """The registry key, i.e. the HF repo name without the owner prefix."""
        return self.hf_name.split("/", 1)[1]


MODERNPROST_MODELS: Dict[str, ModernProstModel] = {
    "modernprost-base": ModernProstModel(
        hf_name="gbouras13/modernprost-base",
        default_task=CLASSIFICATION_TASK,
        hidden_size=1536,
        num_hidden_layers=36,
        num_attention_heads=24,
        approx_params="~1.0B",
        download_mb=4164,
    ),
    "modernprost-50M": ModernProstModel(
        hf_name="gbouras13/modernprost-50M",
        default_task=CLASSIFICATION_TASK,
        hidden_size=448,
        num_hidden_layers=24,
        num_attention_heads=14,
        approx_params="~53M",
        download_mb=210,
    ),
    "modernprost-pssm": ModernProstModel(
        hf_name="gbouras13/modernprost-pssm",
        default_task=PSSM_TASK,
        hidden_size=1536,
        num_hidden_layers=36,
        num_attention_heads=24,
        approx_params="~1.0B",
        download_mb=4164,
    ),
    "modernprost-50M-pssm": ModernProstModel(
        hf_name="gbouras13/modernprost-50M-pssm",
        default_task=PSSM_TASK,
        hidden_size=448,
        num_hidden_layers=24,
        num_attention_heads=14,
        approx_params="~53M",
        download_mb=210,
    ),
}


def resolve_modernprost_model(name: str) -> ModernProstModel:
    """Look up a ModernProst checkpoint by short name or full HuggingFace id.

    Both ``"modernprost-50M"`` and ``"gbouras13/modernprost-50M"`` resolve to
    the same entry, so CLI flags and config files can use whichever is more
    readable without the caller normalising first.

    Raises:
        KeyError: with the full list of valid names, if *name* is unknown.
    """
    if name in MODERNPROST_MODELS:
        return MODERNPROST_MODELS[name]

    for spec in MODERNPROST_MODELS.values():
        if name == spec.hf_name:
            return spec

    raise KeyError(
        f"Unknown ModernProst model {name!r}. Valid models: "
        f"{', '.join(sorted(MODERNPROST_MODELS))}"
    )


def is_modernprost_model(name: str) -> bool:
    """True when *name* refers to a registered ModernProst checkpoint.

    Lets callers branch between the ProstT5 and ModernProst code paths from a
    single ``--model`` string without a try/except around the resolver.
    """
    try:
        resolve_modernprost_model(name)
    except KeyError:
        return False
    return True


def default_task_for(name: str) -> str:
    """Return the task (``"classification"`` / ``"pssm"``) a model was trained for."""
    return resolve_modernprost_model(name).default_task


# ---------------------------------------------------------------------------
# Integrity checks
# ---------------------------------------------------------------------------

# Essential files inside a ModernProst HuggingFace snapshot. All four models
# ship the same custom-code file set (the model_type is `modernprost_3di12st`,
# so `trust_remote_code=True` loads these .py files, not a built-in class).
MODERNPROST_SNAPSHOT_FILES = (
    "config.json",
    "model.safetensors",
    "configuration_modernprost_3di12st.py",
    "modeling_modernprost_3di12st.py",
    "tokenization_modernprost_3di12st.py",
    "tokenizer_config.json",
)

# Per-model MD5 dictionaries, keyed by short model name, in the same
# {"refs": {...}, "blobs": {...}} shape as PROSTT5_MD5_DICTIONARY. Models
# absent from this mapping fall back to the snapshot presence check in
# check_modernprost_download() — see this module's docstring.
MODERNPROST_MD5_DICTIONARIES: Dict[str, Dict] = {}


def check_modernprost_download(
    model_dir: Path,
    model_name: str,
    md5_dict: Optional[Dict] = None,
) -> bool:
    """Check whether a ModernProst checkpoint is present and uncorrupted.

    Args:
        model_dir: Directory HuggingFace caches models into (phold passes the
            phold database directory so model and DB live together).
        model_name: Short name or full HuggingFace id.
        md5_dict: Blob MD5 dictionary for this checkpoint. Defaults to the
            registered dictionary for *model_name*, if one is known.

    Returns:
        True if the model needs to be (re)downloaded, False if it is intact.
    """
    spec = resolve_modernprost_model(model_name)
    cache_root = Path(model_dir) / spec.cache_subdir

    if md5_dict is None:
        md5_dict = MODERNPROST_MD5_DICTIONARIES.get(spec.short_name)

    if md5_dict:
        download = False
        for key, nested in md5_dict.items():
            for nested_key, expected_md5 in nested.items():
                file_path = cache_root / key / nested_key
                if not file_path.exists():
                    logger.warning(f"Model file {file_path} does not exist.")
                    download = True
                    continue
                actual_md5 = _calc_md5(file_path)
                if actual_md5 != expected_md5:
                    logger.warning(
                        f"Corrupt model file {file_path}! "
                        f"MD5 should be '{expected_md5}' but is '{actual_md5}'"
                    )
                    download = True
        return download

    # No MD5 dictionary registered: fall back to checking that the resolved
    # snapshot actually contains every file `from_pretrained` will need.
    return not _snapshot_is_complete(cache_root)


def _snapshot_is_complete(cache_root: Path) -> bool:
    """True when the HF cache at *cache_root* holds a complete snapshot.

    Resolves ``refs/main`` to a commit hash and checks every entry of
    :data:`MODERNPROST_SNAPSHOT_FILES` exists and is non-empty under
    ``snapshots/<hash>/``. Symlinked blobs are followed, so a snapshot whose
    blob was truncated or removed reads as incomplete.
    """
    ref = cache_root / "refs" / "main"
    if not ref.is_file():
        logger.warning(f"No cached ModernProst snapshot found at {cache_root}")
        return False

    try:
        commit = ref.read_text().strip()
    except OSError as exc:
        logger.warning(f"Could not read {ref}: {exc}")
        return False

    snapshot = cache_root / "snapshots" / commit
    if not snapshot.is_dir():
        logger.warning(f"ModernProst snapshot directory {snapshot} does not exist.")
        return False

    for filename in MODERNPROST_SNAPSHOT_FILES:
        path = snapshot / filename
        try:
            if not path.is_file() or path.stat().st_size == 0:
                logger.warning(f"Model file {path} is missing or empty.")
                return False
        except OSError as exc:
            # Broken symlink into a deleted blob raises here rather than
            # returning False from is_file() on some filesystems.
            logger.warning(f"Could not stat {path}: {exc}")
            return False

    return True


# ---------------------------------------------------------------------------
# Zenodo backup download
# ---------------------------------------------------------------------------

def download_zenodo_modernprost(
    model_dir: Path,
    logdir: str,
    threads: int,
    backup_url: str,
    backup_md5: str,
    backup_tarball: str,
) -> None:
    """Download a ModernProst checkpoint from a Zenodo backup and extract it.

    Same contract as :func:`pholdlib.databases.prostt5.download_zenodo_prostT5`
    — the tarball is expected to contain a ``models--gbouras13--<name>/``
    directory so it drops straight into the HuggingFace cache layout.

    Args:
        model_dir: Destination directory (the HuggingFace cache root).
        logdir: Unused (kept for API compatibility with existing callers).
        threads: Unused (kept for API compatibility with existing callers).
        backup_url: Direct download URL for the tarball.
        backup_md5: Expected MD5 of the tarball.
        backup_tarball: Filename for the downloaded tarball.
    """
    logger.info(f"Downloading ModernProst model backup from {backup_url}")
    tarball_path = Path(model_dir) / backup_tarball

    _download_requests(backup_url, tarball_path)

    md5_sum = _calc_md5(tarball_path)
    if md5_sum == backup_md5:
        logger.info(f"ModernProst model backup file download OK: {md5_sum}")
    else:
        logger.error(
            f"Error: corrupt file! MD5 should be '{backup_md5}' but is '{md5_sum}'"
        )

    logger.info(f"Extracting ModernProst model backup: {tarball_path} -> {model_dir}")
    try:
        with tarball_path.open("rb") as fh_in, tarfile.open(fileobj=fh_in, mode="r:gz") as tar:
            _safe_extractall(tar, model_dir)
    except OSError as exc:
        logger.warning(f"Encountered OSError: {exc}")
        logger.error(f"Could not extract {tarball_path} to {model_dir}")

    tarball_path.unlink(missing_ok=True)
