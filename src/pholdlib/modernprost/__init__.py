"""ModernProst 3Di + 12-state inference and Foldseek database construction.

See :mod:`pholdlib.databases.modernprost` for the model registry and
:mod:`pholdlib.modernprost.inference` for the inference engine.
"""

from pholdlib.modernprost.model import count_parameters, get_modernprost_model
from pholdlib.modernprost.inference import (
    DEFAULT_CHUNK_LEN,
    ModernProstResult,
    chunk_sequence,
    predictions_from_profiles,
    run_modernprost_inference,
    run_modernprost_inference_multi_gpu,
)
from pholdlib.modernprost.output import (
    SS12_ALPHABET,
    SS12_MAPPING,
    THREEDI_ALPHABET,
    THREEDI_MAPPING,
    THREEDI_MASK_INDEX,
    decode_12st,
    decode_3di,
    format_profile,
    write_12st_fasta,
    write_3di_fasta,
    write_all_probs,
    write_fail_ids,
    write_mean_probs,
    write_profiles_text,
)
from pholdlib.modernprost.foldseek_db import (
    DBTYPE_SS_12ST,
    build_lookup,
    encode_combined,
    generate_combined_foldseek_db,
    generate_sequence_foldseek_db,
    read_fasta,
    sort_index_file,
    subprocess_tsv2db_runner,
)
from pholdlib.modernprost.profile_db import (
    AA_ALPHABET,
    aa_profile_record,
    compute_log_pssm_12st,
    compute_log_pssm_3di,
    parse_profiles,
    to_profile_record,
    write_profile_foldseek_dbs,
)

__all__ = [
    # model
    "get_modernprost_model",
    "count_parameters",
    # inference
    "run_modernprost_inference",
    "run_modernprost_inference_multi_gpu",
    "ModernProstResult",
    "chunk_sequence",
    "predictions_from_profiles",
    "DEFAULT_CHUNK_LEN",
    # output
    "THREEDI_ALPHABET",
    "SS12_ALPHABET",
    "THREEDI_MAPPING",
    "SS12_MAPPING",
    "THREEDI_MASK_INDEX",
    "decode_3di",
    "decode_12st",
    "format_profile",
    "write_3di_fasta",
    "write_12st_fasta",
    "write_mean_probs",
    "write_all_probs",
    "write_fail_ids",
    "write_profiles_text",
    # foldseek dbs
    "generate_combined_foldseek_db",
    "generate_sequence_foldseek_db",
    "encode_combined",
    "read_fasta",
    "build_lookup",
    "sort_index_file",
    "subprocess_tsv2db_runner",
    "DBTYPE_SS_12ST",
    # profile dbs
    "write_profile_foldseek_dbs",
    "compute_log_pssm_3di",
    "compute_log_pssm_12st",
    "to_profile_record",
    "aa_profile_record",
    "parse_profiles",
    "AA_ALPHABET",
]
