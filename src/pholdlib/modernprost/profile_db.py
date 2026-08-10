#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Foldseek *profile* database construction from ModernProst softmax profiles.

The ``-pssm`` ModernProst checkpoints emit calibrated per-residue distributions
rather than confident argmax states. Instead of collapsing those to a single
letter, they are written straight into Foldseek's HMM-profile database format,
so the search scores against the full distribution.

Three databases are produced for one query set:

``<db>_profile_ss``
    3Di profile — the ``(L, 20)`` softmax turned into a log-odds PSSM against
    the 3Di background distribution.
``<db>_profile_ss12``
    12-state profile — the ``(L, 12)`` softmax packed into the same fixed
    20-column layout MMseqs uses, with the trailing eight columns zeroed.
``<db>_profile``
    Amino-acid profile — BLOSUM62 rows for the query residues, i.e. a plain
    sequence expressed in profile form so all three DBs share a record layout.

Each record is 25 bytes per residue: 20 profile scores (int8), then the query
residue index, the consensus residue index, an Neff byte and two reserved
zero bytes.

Ported from ``foldseek2_plm.utils.inference``.

Amino-acid ordering
-------------------
Profile columns and the query/consensus byte both index the 21-letter MMseqs
alphabet ``ACDEFGHIKLMNPQRSTVWYX``, which is the BLOSUM62 header order. Note
this differs from foldseek2-plm, whose ``AAs`` list interposes ``O`` at index
12 and so shifts ``P``..``Y`` by one relative to the matrix columns it indexes;
that shift is not reproduced here. ``X`` is also given a real profile block
here rather than being skipped, so the amino-acid profile always has exactly
the same length as the 3Di and 12-state profiles for the same record — with
X-skipping, any masked residue silently desynchronised the three databases.
"""

import shutil
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from loguru import logger

from pholdlib.modernprost.foldseek_db import (
    create_single_shard_alias,
    sort_index_file,
    write_dbtype,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# BLOSUM62 in 1/2 bit units, as shipped with MMseqs2/Foldseek.
BLOSUM62_STR = """
# BLOSUM62 in 1/2 Bit
# Background (precomputed optional): 0.07422 0.02469 0.05363 0.05431 0.04742 0.07415 0.02621 0.06792 0.05815 0.09891 0.02499 0.04465 0.03854 0.03426 0.05161 0.05723 0.05089 0.07292 0.01303 0.03228 0.00001
# Lambda     (precomputed optional): 0.34657
   A       C       D       E       F       G       H       I       K       L       M       N       P       Q       R       S       T       V       W       Y       X
A  3.9291 -0.4085 -1.7534 -0.8639 -2.2101  0.1596 -1.6251 -1.3218 -0.7340 -1.4646 -0.9353 -1.5307 -0.8143 -0.8040 -1.4135  1.1158 -0.0454 -0.1894 -2.5269 -1.7640 -1.0000
C -0.4085  8.5821 -3.4600 -3.6125 -2.3755 -2.5004 -2.9878 -1.2277 -3.0363 -1.2775 -1.4198 -2.6598 -2.7952 -2.9019 -3.3892 -0.8750 -0.8667 -0.8077 -2.3041 -2.4071 -1.0000
D -1.7534 -3.4600  5.7742  1.5103 -3.4839 -1.3135 -1.1189 -3.1212 -0.7018 -3.6057 -3.0585  1.2717 -1.4801 -0.3134 -1.6058 -0.2610 -1.0507 -3.1426 -4.2143 -3.0650 -1.0000
E -0.8639 -3.6125  1.5103  4.9028 -3.1924 -2.1102 -0.1177 -3.1944  0.7753 -2.8465 -1.9980 -0.2680 -1.1162  1.8546 -0.1154 -0.1469 -0.8633 -2.4423 -2.8354 -2.0205 -1.0000
F -2.2101 -2.3755 -3.4839 -3.1924  6.0461 -3.1074 -1.2342 -0.1609 -3.0787  0.4148  0.0126 -2.9940 -3.5973 -3.1644 -2.7863 -2.3690 -2.1076 -0.8490  0.9176  2.9391 -1.0000
G  0.1596 -2.5004 -1.3135 -2.1102 -3.1074  5.5633 -2.0409 -3.7249 -1.5280 -3.6270 -2.6766 -0.4228 -2.1335 -1.7852 -2.3041 -0.2925 -1.5754 -3.1387 -2.4915 -3.0398 -1.0000
H -1.6251 -2.9878 -1.1189 -0.1177 -1.2342 -2.0409  7.5111 -3.2316 -0.7210 -2.7867 -1.5513  0.5785 -2.1609  0.4480 -0.2499 -0.8816 -1.6859 -3.1175 -2.3422  1.6926 -1.0000
I -1.3218 -1.2277 -3.1212 -3.1944 -0.1609 -3.7249 -3.2316  3.9985 -2.6701  1.5216  1.1268 -3.2170 -2.7567 -2.7696 -2.9902 -2.3482 -0.7176  2.5470 -2.5805 -1.3314 -1.0000
K -0.7340 -3.0363 -0.7018  0.7753 -3.0787 -1.5280 -0.7210 -2.6701  4.5046 -2.4468 -1.3547 -0.1790 -1.0136  1.2726  2.1087 -0.2034 -0.6696 -2.2624 -2.9564 -1.8200 -1.0000
L -1.4646 -1.2775 -3.6057 -2.8465  0.4148 -3.6270 -2.7867  1.5216 -2.4468  3.8494  1.9918 -3.3789 -2.8601 -2.1339 -2.1546 -2.4426 -1.1975  0.7884 -1.6319 -1.0621 -1.0000
M -0.9353 -1.4198 -3.0585 -1.9980  0.0126 -2.6766 -1.5513  1.1268 -1.3547  1.9918  5.3926 -2.1509 -2.4764 -0.4210 -1.3671 -1.4809 -0.6663  0.6872 -1.4248 -0.9949 -1.0000
N -1.5307 -2.6598  1.2717 -0.2680 -2.9940 -0.4228  0.5785 -3.2170 -0.1790 -3.3789 -2.1509  5.6532 -2.0004  0.0017 -0.4398  0.6009 -0.0461 -2.8763 -3.6959 -2.0818 -1.0000
P -0.8143 -2.7952 -1.4801 -1.1162 -3.5973 -2.1335 -2.1609 -2.7567 -1.0136 -2.8601 -2.4764 -2.0004  7.3646 -1.2819 -2.1086 -0.8090 -1.0753 -2.3487 -3.6542 -2.9198 -1.0000
Q -0.8040 -2.9019 -0.3134  1.8546 -3.1644 -1.7852  0.4480 -2.7696  1.2726 -2.1339 -0.4210  0.0017 -1.2819  5.2851  0.9828 -0.1011 -0.6753 -2.1984 -1.9465 -1.4211 -1.0000
R -1.4135 -3.3892 -1.6058 -0.1154 -2.7863 -2.3041 -0.2499 -2.9902  2.1087 -2.1546 -1.3671 -0.4398 -2.1086  0.9828  5.4735 -0.7648 -1.1223 -2.5026 -2.6794 -1.6939 -1.0000
S  1.1158 -0.8750 -0.2610 -0.1469 -2.3690 -0.2925 -0.8816 -2.3482 -0.2034 -2.4426 -1.4809  0.6009 -0.8090 -0.1011 -0.7648  3.8844  1.3811 -1.6462 -2.7519 -1.6858 -1.0000
T -0.0454 -0.8667 -1.0507 -0.8633 -2.1076 -1.5754 -1.6859 -0.7176 -0.6696 -1.1975 -0.6663 -0.0461 -1.0753 -0.6753 -1.1223  1.3811  4.5453 -0.0555 -2.4289 -1.6060 -1.0000
V -0.1894 -0.8077 -3.1426 -2.4423 -0.8490 -3.1387 -3.1175  2.5470 -2.2624  0.7884  0.6872 -2.8763 -2.3487 -2.1984 -2.5026 -1.6462 -0.0555  3.7689 -2.8343 -1.2075 -1.0000
W -2.5269 -2.3041 -4.2143 -2.8354  0.9176 -2.4915 -2.3422 -2.5805 -2.9564 -1.6319 -1.4248 -3.6959 -3.6542 -1.9465 -2.6794 -2.7519 -2.4289 -2.8343 10.5040  2.1542 -1.0000
Y -1.7640 -2.4071 -3.0650 -2.0205  2.9391 -3.0398  1.6926 -1.3314 -1.8200 -1.0621 -0.9949 -2.0818 -2.9198 -1.4211 -1.6939 -1.6858 -1.6060 -1.2075  2.1542  6.5950 -1.0000
X -1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000
"""

# MMseqs' amino-acid ordering — the BLOSUM62 header order. See module docstring.
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AA_TO_INDEX: Dict[str, int] = {aa: i for i, aa in enumerate(AA_ALPHABET)}

# 3Di background state frequencies (Foldseek's mat3di), first 20 entries.
BACKGROUND_3DI = np.asarray(
    [
        0.0489372, 0.0306991, 0.101049, 0.0329671, 0.0276149,
        0.0416262, 0.0452521, 0.030876, 0.0297251, 0.0607036,
        0.0150238, 0.0215826, 0.0783843, 0.0512926, 0.0264886,
        0.0610702, 0.0201311, 0.215998, 0.0310265, 0.0295417,
    ],
    dtype=np.float32,
)

# 12-state background frequencies, padded to 20 columns with a tiny epsilon so
# the unused columns produce a saturated -128 score rather than a divide-by-zero.
BACKGROUND_12ST = np.asarray(
    [
        0.0412689151029924, 0.0231671389192358, 0.0599445079508294,
        0.0796249244089918, 0.113519805597932, 0.0845763487053319,
        0.124117111554957, 0.105288913376871, 0.150162613848461,
        0.0882703257027577, 0.0471395802906669, 0.082901814540972,
        1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06,
    ],
    dtype=np.float32,
)

# MMseqs stores PSSM scores in 1/8-bit units.
BIT_FACTOR = 8

# Bytes per residue in a profile DB record: 20 scores + query + consensus +
# Neff + 2 reserved.
PROFILE_RECORD_STRIDE = 25

# Foldseek DB type for an HMM profile database.
DBTYPE_HMM_PROFILE = 2


# ---------------------------------------------------------------------------
# PSSM computation
# ---------------------------------------------------------------------------

def _log_odds_pssm(profile: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Convert an ``(L, 20)`` probability matrix to an int8 log-odds PSSM.

    Positions whose probability is zero would give ``log2(0) = -inf``; they are
    assigned the floor value instead. Rounding is half-away-from-zero to match
    MMseqs' C++ ``(int)(x + 0.5)`` / ``(int)(x - 0.5)`` behaviour, which differs
    from numpy's banker's rounding on exact .5 values.
    """
    odds = profile / background[None, :]

    log_probs = np.empty_like(odds, dtype=np.float32)
    log_probs.fill(-128.0)
    positive = odds > 0.0
    log_probs[positive] = np.log2(odds[positive])

    pssm = log_probs * BIT_FACTOR
    pssm = np.where(pssm < 0, pssm - 0.5, pssm + 0.5)
    return np.clip(pssm, -128, 127).astype(np.int8)


def compute_log_pssm_3di(profile: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build the 3Di PSSM and consensus from an ``(L, 20)`` softmax profile.

    Returns:
        ``(pssm_flat, consensus)`` — the PSSM flattened to ``L * 20`` int8
        values, and the per-residue argmax state as int8.
    """
    profile = np.asarray(profile, dtype=np.float32)
    if profile.ndim != 2 or profile.shape[1] != 20:
        raise ValueError(f"Expected a 20-column 3Di profile, got shape {profile.shape}")

    consensus = np.argmax(profile, axis=1).astype(np.int8)
    return _log_odds_pssm(profile, BACKGROUND_3DI).ravel(), consensus


def compute_log_pssm_12st(profile: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build the 12-state PSSM and consensus from an ``(L, 12)`` softmax profile.

    The 12 columns are packed into MMseqs' fixed 20-column layout; the
    remaining eight columns are zero and score as the floor value.
    """
    profile = np.asarray(profile, dtype=np.float32)
    if profile.ndim != 2 or profile.shape[1] != 12:
        raise ValueError(
            f"Expected a 12-column 12-state profile, got shape {profile.shape}"
        )

    profile20 = np.zeros((profile.shape[0], 20), dtype=np.float32)
    profile20[:, :12] = profile
    consensus = np.argmax(profile, axis=1).astype(np.int8)
    return _log_odds_pssm(profile20, BACKGROUND_12ST).ravel(), consensus


def to_profile_record(
    pssm: np.ndarray,
    query: np.ndarray,
    consensus: np.ndarray,
    neff: int = 1,
) -> bytes:
    """Pack a flattened PSSM into a Foldseek profile-DB record.

    Args:
        pssm: ``L * 20`` int8 log-odds scores.
        query: ``L`` query-residue indices.
        consensus: ``L`` consensus-residue indices.
        neff: Effective-sequence-number byte, constant for a single-sequence
            profile.

    Returns:
        ``L * 25`` bytes.
    """
    L = len(consensus)
    if len(query) != L:
        raise ValueError(f"query has {len(query)} entries but consensus has {L}")
    if len(pssm) != L * 20:
        raise ValueError(f"pssm has {len(pssm)} entries, expected {L * 20}")

    # Build the record as an (L, 25) uint8 matrix and flatten, rather than
    # looping in Python — a 50k-CDS genome is ~15M residues.
    record = np.zeros((L, PROFILE_RECORD_STRIDE), dtype=np.uint8)
    record[:, :20] = np.ascontiguousarray(pssm, dtype=np.int8).reshape(L, 20).view(np.uint8)
    record[:, 20] = np.asarray(query, dtype=np.int64) & 0xFF
    record[:, 21] = np.asarray(consensus, dtype=np.int64) & 0xFF
    record[:, 22] = neff & 0xFF
    # columns 23, 24 stay zero (reserved)
    return record.tobytes()


# ---------------------------------------------------------------------------
# Amino-acid profile DB (BLOSUM62 rows for the query sequence)
# ---------------------------------------------------------------------------

def parse_substitution_matrix(matrix_str: str = BLOSUM62_STR) -> Dict[str, List[float]]:
    """Parse a substitution matrix into ``{residue: [20 scores]}``.

    Only the first 20 score columns are kept — the trailing ``X`` column is
    outside the 20-wide profile layout.
    """
    matrix: Dict[str, List[float]] = {}
    header_seen = False

    for line in matrix_str.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if not header_seen:
            header_seen = True
            continue
        tokens = line.split()
        matrix[tokens[0].upper()] = [float(v) for v in tokens[1:21]]

    return matrix


def _precompute_aa_blocks(
    matrix: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, bytes]:
    """Precompute the 25-byte profile record for every amino acid.

    Scores are scaled by 4 (matching MMseqs' sequence-as-profile conversion)
    before rounding and clipping to int8.
    """
    if matrix is None:
        matrix = parse_substitution_matrix()

    blocks: Dict[str, bytes] = {}
    for aa, row in matrix.items():
        if aa not in AA_TO_INDEX:
            continue

        scaled = np.asarray(row, dtype=np.float32) * 4.0
        scaled = np.where(scaled < 0, scaled - 0.5, scaled + 0.5)
        scaled = np.clip(scaled, -128, 127).astype(np.int8)

        idx = AA_TO_INDEX[aa]
        block = bytearray(PROFILE_RECORD_STRIDE)
        block[0:20] = scaled.view(np.uint8).tobytes()
        block[20] = idx
        block[21] = idx
        block[22] = 0
        blocks[aa] = bytes(block)

    return blocks


_AA_BLOCKS: Dict[str, bytes] = {}


def aa_profile_record(sequence: str) -> bytes:
    """Build the amino-acid profile record for one sequence.

    Residues outside the 21-letter alphabet (``B``, ``J``, ``O``, ``U``, ``Z``
    and anything lowercase) are encoded as ``X`` rather than dropped, so the
    record always has exactly ``len(sequence)`` residues and stays aligned with
    the 3Di and 12-state records for the same key.
    """
    global _AA_BLOCKS
    if not _AA_BLOCKS:
        _AA_BLOCKS = _precompute_aa_blocks()

    unknown = _AA_BLOCKS["X"]
    out = bytearray()
    for res in sequence:
        out.extend(_AA_BLOCKS.get(res.upper(), unknown))
    return bytes(out)


# ---------------------------------------------------------------------------
# Database assembly
# ---------------------------------------------------------------------------

def write_profile_foldseek_dbs(
    profiles_3di: Dict[str, np.ndarray],
    profiles_12st: Dict[str, np.ndarray],
    aa_sequences: Dict[str, str],
    lookup: Dict[str, str],
    output_dir: Path,
    db_name: str,
    source_db: Path,
    order: Optional[Sequence[str]] = None,
) -> Dict[str, Path]:
    """Write the 3Di, 12-state and amino-acid Foldseek profile databases.

    Args:
        profiles_3di: ``{seq_id: ndarray (L, 20)}`` softmax profiles.
        profiles_12st: ``{seq_id: ndarray (L, 12)}`` softmax profiles.
        aa_sequences: ``{seq_id: amino-acid sequence}`` for the same records.
        lookup: ``{seq_id: numeric DB key}``, e.g. from
            :func:`pholdlib.modernprost.foldseek_db.build_lookup`.
        output_dir: Directory the profile DBs are written into.
        db_name: DB basename; produces ``<db_name>_profile``,
            ``<db_name>_profile_ss`` and ``<db_name>_profile_ss12``.
        source_db: Prefix of the sequence DB that supplies the header
            database — ``<source_db>_h``, ``_h.index`` and ``_h.dbtype`` are
            copied next to the profile DBs.
        order: Optional explicit record order. Defaults to ``profiles_3di``'s
            insertion order. Records are index-sorted afterwards regardless, so
            this only affects data-file layout.

    Returns:
        ``{"profile_ss": ..., "profile_ss12": ..., "profile": ...}`` paths.

    Raises:
        ValueError: on a missing 12-state profile, a missing amino-acid
            sequence, a length disagreement between the three, or a seq_id with
            no entry in *lookup*.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seq_ids = list(order) if order is not None else list(profiles_3di)

    ss_db = out_dir / f"{db_name}_profile_ss"
    ss12_db = out_dir / f"{db_name}_profile_ss12"
    aa_db = out_dir / f"{db_name}_profile"

    parts_ss: List[bytes] = []
    parts_ss12: List[bytes] = []
    parts_aa: List[bytes] = []
    index_ss: List[str] = []
    index_ss12: List[str] = []
    index_aa: List[str] = []
    offset_ss = offset_ss12 = offset_aa = 0

    for seq_id in seq_ids:
        prof_3di = profiles_3di.get(seq_id)
        if prof_3di is None:
            continue

        key = lookup.get(seq_id)
        if key is None:
            raise ValueError(
                f"No numeric Foldseek key for {seq_id!r}; it is missing from the "
                "query database lookup."
            )

        prof_12st = profiles_12st.get(seq_id)
        if prof_12st is None:
            raise ValueError(
                f"No 12-state profile for {seq_id!r}; cannot write the "
                "profile_ss12 database."
            )

        aa_seq = aa_sequences.get(seq_id)
        if aa_seq is None:
            raise ValueError(f"No amino-acid sequence for {seq_id!r}.")

        L = prof_3di.shape[0]
        if prof_12st.shape[0] != L or len(aa_seq) != L:
            raise ValueError(
                f"Length mismatch for {seq_id!r}: 3Di={L} "
                f"12st={prof_12st.shape[0]} aa={len(aa_seq)}"
            )

        pssm_3di, consensus_3di = compute_log_pssm_3di(prof_3di)
        pssm_12st, consensus_12st = compute_log_pssm_12st(prof_12st)

        # The query byte carries the consensus state for the structural
        # profiles — there is no separate "query" 3Di/12st residue to record.
        buf_ss = to_profile_record(pssm_3di, consensus_3di, consensus_3di)
        buf_ss12 = to_profile_record(pssm_12st, consensus_12st, consensus_12st)
        buf_aa = aa_profile_record(aa_seq)

        parts_ss.append(buf_ss)
        index_ss.append(f"{key}\t{offset_ss}\t{len(buf_ss)}")
        offset_ss += len(buf_ss)

        parts_ss12.append(buf_ss12)
        index_ss12.append(f"{key}\t{offset_ss12}\t{len(buf_ss12)}")
        offset_ss12 += len(buf_ss12)

        parts_aa.append(buf_aa)
        index_aa.append(f"{key}\t{offset_aa}\t{len(buf_aa)}")
        offset_aa += len(buf_aa)

    if not parts_ss:
        raise ValueError("No profile database records were produced.")

    for path, parts, index_lines in (
        (ss_db, parts_ss, index_ss),
        (ss12_db, parts_ss12, index_ss12),
        (aa_db, parts_aa, index_aa),
    ):
        path.write_bytes(b"".join(parts))
        Path(f"{path}.index").write_text("\n".join(index_lines) + "\n")
        sort_index_file(Path(f"{path}.index"))
        create_single_shard_alias(path)

    copy_header_db(source_db, out_dir, db_name)

    for path in (ss_db, ss12_db, aa_db):
        write_dbtype(Path(f"{path}.dbtype"), DBTYPE_HMM_PROFILE)

    logger.info(f"3Di Foldseek profile DB written to {ss_db}")
    logger.info(f"12-state Foldseek profile DB written to {ss12_db}")
    logger.info(f"Amino-acid Foldseek profile DB written to {aa_db}")

    return {"profile_ss": ss_db, "profile_ss12": ss12_db, "profile": aa_db}


def copy_header_db(source_db: Path, output_dir: Path, db_name: str) -> None:
    """Copy a sequence DB's header database next to the profile DBs.

    Foldseek resolves query names through ``<db>_h``; the profile DBs reuse the
    same numeric keys as *source_db*, so its header DB can be copied verbatim.
    """
    source_db = Path(source_db)
    out_dir = Path(output_dir)

    for suffix in ("_h", "_h.index", "_h.dbtype"):
        src = Path(f"{source_db}{suffix}")
        if not src.exists():
            raise ValueError(
                f"Header database file {src} does not exist; cannot build the "
                "Foldseek profile database."
            )
        shutil.copy2(src, out_dir / f"{db_name}_profile{suffix}")


def parse_profiles(path: Path) -> List[Tuple[str, np.ndarray]]:
    """Parse a human-readable profile text file back into arrays.

    Reads the format written by
    :func:`pholdlib.modernprost.output.write_profiles_text`, so a profile run
    can be inspected, edited, and rebuilt into a DB without re-running
    inference.

    Returns:
        A list of ``(seq_id, profile)`` pairs in file order.
    """
    header_pattern = re.compile(r"Query profile of sequence\s+(.*)")

    profiles: List[Tuple[str, np.ndarray]] = []
    current_id: Optional[str] = None
    rows: List[List[float]] = []
    skip_alphabet_line = False

    def _flush() -> None:
        if current_id is not None and rows:
            profiles.append((current_id, np.asarray(rows, dtype=np.float32)))

    with open(path, "r") as fh:
        for line in fh:
            match = header_pattern.match(line)
            if match:
                _flush()
                current_id = match.group(1).strip()
                rows = []
                skip_alphabet_line = True
                continue

            stripped = line.strip()
            if not stripped:
                continue
            if skip_alphabet_line:
                # The line after the header is the alphabet row.
                skip_alphabet_line = False
                continue
            rows.append([float(v) for v in stripped.split()])

    _flush()
    return profiles
