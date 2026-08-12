"""ModernProst 3Di + 12-state inference and Foldseek database construction.

See :mod:`pholdlib.databases.modernprost` for the model registry and
:mod:`pholdlib.modernprost.inference` for the inference engine.

Imports here are **lazy** (PEP 562). ``model`` and ``inference`` pull in
``transformers`` and ``torch``, which cost seconds on a local disk and can cost
minutes on a shared cluster filesystem. Importing them eagerly would make every
consumer pay that price merely to reach the database builders in
``foldseek_db`` / ``profile_db``, which need nothing heavier than numpy — and,
worse, would charge it to unrelated commands: ``from
pholdlib.modernprost.foldseek_db import ...`` executes this module first, so an
eager import chain here reaches anything that touches Foldseek DB construction,
including a bare ``--help``.

``from pholdlib.modernprost import get_modernprost_model`` still works and still
imports torch — it just does so at first attribute access rather than at package
import. Submodules can also be imported directly
(``from pholdlib.modernprost.output import write_3di_fasta``) to bypass this
module's ``__getattr__`` entirely.
"""

from importlib import import_module
from typing import Any

# Public name -> submodule that defines it. Kept as data so __getattr__ stays a
# dict lookup and __dir__ needs no separate list to drift out of sync.
_EXPORTS = {
    # model (imports transformers)
    "get_modernprost_model": "model",
    "count_parameters": "model",
    # inference (imports torch)
    "run_modernprost_inference": "inference",
    "run_modernprost_inference_multi_gpu": "inference",
    "ModernProstResult": "inference",
    "chunk_sequence": "inference",
    "predictions_from_profiles": "inference",
    "DEFAULT_CHUNK_LEN": "inference",
    # output (numpy only)
    "THREEDI_ALPHABET": "output",
    "SS12_ALPHABET": "output",
    "THREEDI_MAPPING": "output",
    "SS12_MAPPING": "output",
    "THREEDI_MASK_INDEX": "output",
    "decode_3di": "output",
    "decode_12st": "output",
    "format_profile": "output",
    "write_3di_fasta": "output",
    "write_12st_fasta": "output",
    "write_mean_probs": "output",
    "write_all_probs": "output",
    "write_fail_ids": "output",
    "write_profiles_text": "output",
    # foldseek dbs (numpy only)
    "generate_combined_foldseek_db": "foldseek_db",
    "generate_sequence_foldseek_db": "foldseek_db",
    "encode_combined": "foldseek_db",
    "read_fasta": "foldseek_db",
    "build_lookup": "foldseek_db",
    "sort_index_file": "foldseek_db",
    "subprocess_tsv2db_runner": "foldseek_db",
    "DBTYPE_SS_12ST": "foldseek_db",
    # profile dbs (numpy only)
    "write_profile_foldseek_dbs": "profile_db",
    "compute_log_pssm_3di": "profile_db",
    "compute_log_pssm_12st": "profile_db",
    "to_profile_record": "profile_db",
    "aa_profile_record": "profile_db",
    "parse_profiles": "profile_db",
    "AA_ALPHABET": "profile_db",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve a public name by importing its submodule on first access."""
    try:
        submodule = _EXPORTS[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None

    value = getattr(import_module(f"{__name__}.{submodule}"), name)
    # Cache on the module so subsequent lookups skip __getattr__ entirely.
    globals()[name] = value
    return value


def __dir__() -> list:
    return sorted(set(globals()) | set(_EXPORTS))
