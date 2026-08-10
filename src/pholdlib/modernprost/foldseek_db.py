#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Foldseek query-database construction for combined 3Di + 12-state predictions.

Foldseek's 12-state-aware mode packs both alphabets into the single ``_ss``
structure database, one byte per residue::

    c = threeDi_index * 12 + ss12_index      # 0 <= c <= 239
    threeDi_index = c // 12
    ss12_index    = c % 12

with ``threeDi_index`` indexing ``ACDEFGHIKLMNPQRSTVWY`` and ``ss12_index``
indexing ``ACDEFGHIKLMN``. The database is searched with ``--ss-12st 1``.

Because those byte values include 0x00, 0x09 and 0x0A — NUL, tab and newline —
the combined string cannot be fed through ``foldseek tsv2db`` as text. So the
DB is built in two steps, matching foldseek2-plm's ``generate_foldseek_db``:

1. ``tsv2db`` the plain 3Di letters into ``<db>_ss``, which lays out records
   and writes a matching ``.index``.
2. Reopen ``<db>_ss`` and overwrite each record's payload in place with the
   packed bytes.

Step 2 here seeks using the offsets in ``<db>_ss.index`` rather than assuming
records were emitted in input order, so a future ``tsv2db`` that reorders or
pads records cannot silently corrupt the DB.

Ported from ``foldseek2_plm.utils.generate_foldseek_db``.
"""

import os
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from loguru import logger

from pholdlib.modernprost.output import SS12_ALPHABET, THREEDI_ALPHABET

# Foldseek --output-dbtype values.
DBTYPE_AMINO_ACIDS = 0
DBTYPE_GENERIC = 12
# The structure/SS database type used for the combined 3Di+12st alphabet.
# Matches foldseek2-plm's verified `generate_foldseek_db`.
DBTYPE_SS_12ST = 2097152

THREEDI_INDEX: Dict[str, int] = {ch: i for i, ch in enumerate(THREEDI_ALPHABET)}
SS12_INDEX: Dict[str, int] = {ch: i for i, ch in enumerate(SS12_ALPHABET)}

# One Foldseek DB record is "<payload>\n\0", so the index length is payload+2.
_RECORD_SUFFIX_LEN = 2

Tsv2DbRunner = Callable[[Path, Path, int], None]


def subprocess_tsv2db_runner(foldseek: str) -> Tsv2DbRunner:
    """Build a ``tsv2db`` runner that shells out to the *foldseek* binary.

    phold passes its own runner instead, so ``foldseek tsv2db`` calls go
    through phold's ExternalTool wrapper and land in its log directory.
    """

    def _run(in_tsv: Path, out_db: Path, dbtype: int) -> None:
        subprocess.run(
            [str(foldseek), "tsv2db", str(in_tsv), str(out_db),
             "--output-dbtype", str(dbtype)],
            check=True,
        )

    return _run


def encode_combined(seq_3di: str, seq_12st: str, seq_id: str = "") -> bytes:
    """Pack a 3Di string and a 12-state string into combined Foldseek bytes.

    Raises:
        ValueError: on a length mismatch, or on a character outside either
            alphabet. In particular a masked ``X`` in the 3Di string is
            rejected: the combined encoding has no 21st 3Di state, so masking
            must be applied to the amino-acid FASTA instead (see
            :mod:`pholdlib.modernprost.output`).
    """
    if len(seq_3di) != len(seq_12st):
        raise ValueError(
            f"Length mismatch for {seq_id or '<unnamed>'}: "
            f"3di={len(seq_3di)} 12st={len(seq_12st)}"
        )

    out = bytearray(len(seq_3di))
    for j, (ch3, ch12) in enumerate(zip(seq_3di, seq_12st)):
        try:
            three = THREEDI_INDEX[ch3]
        except KeyError:
            hint = (
                " — 'X' is a masked low-confidence residue, which the combined "
                "3Di+12st encoding cannot represent. Write the 3Di FASTA with "
                "mask_threshold=0 and mask the amino-acid FASTA instead."
                if ch3 == "X" else ""
            )
            raise ValueError(
                f"Unknown 3Di character {ch3!r} in {seq_id or '<unnamed>'} "
                f"at position {j}{hint}"
            ) from None
        try:
            ss = SS12_INDEX[ch12]
        except KeyError:
            raise ValueError(
                f"Unknown 12-state character {ch12!r} in "
                f"{seq_id or '<unnamed>'} at position {j}"
            ) from None
        out[j] = three * 12 + ss

    return bytes(out)


def read_fasta(path: Path) -> Dict[str, str]:
    """Read a FASTA file into an ordered ``{id: sequence}`` dict.

    A deliberately minimal parser so pholdlib does not depend on BioPython.
    The record id is the header up to the first whitespace, matching
    ``SeqIO.parse(...).id``. Sequence lines are concatenated, so wrapped FASTA
    is handled; sequences are *not* upper-cased, because the 3Di path relies on
    case being preserved.
    """
    sequences: Dict[str, str] = {}
    current_id: Optional[str] = None
    chunks: List[str] = []

    with open(path, "r") as fh:
        for line in fh:
            line = line.rstrip("\n\r")
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(chunks)
                current_id = line[1:].split(None, 1)[0] if len(line) > 1 else ""
                chunks = []
            elif line:
                chunks.append(line.strip())

    if current_id is not None:
        sequences[current_id] = "".join(chunks)

    return sequences


def generate_combined_foldseek_db(
    fasta_aa: Path,
    fasta_3di: Path,
    fasta_12st: Path,
    foldseek_db_path: Path,
    db_name: str,
    tsv2db_runner: Tsv2DbRunner,
    keep_tsv: bool = False,
) -> Path:
    """Build a Foldseek query DB with the combined 3Di + 12-state alphabet.

    Args:
        fasta_aa: Amino-acid FASTA. Defines which sequences make it into the
            DB and in what order.
        fasta_3di: 3Di FASTA, written with ``mask_threshold=0``.
        fasta_12st: 12-state FASTA.
        foldseek_db_path: Directory the DB files are written into.
        db_name: DB basename; produces ``<db_name>``, ``<db_name>_ss``,
            ``<db_name>_h`` and ``<db_name>.lookup``.
        tsv2db_runner: Callable ``(in_tsv, out_db, dbtype)`` that runs
            ``foldseek tsv2db``. See :func:`subprocess_tsv2db_runner`.
        keep_tsv: Leave the intermediate TSVs on disk for debugging.

    Returns:
        The DB prefix path (``foldseek_db_path / db_name``).

    Raises:
        ValueError: if a sequence is missing a 3Di or 12-state string, if
            lengths disagree, or if either string contains an out-of-alphabet
            character.
    """
    out_dir = Path(foldseek_db_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    sequences_aa = read_fasta(fasta_aa)
    sequences_3di = read_fasta(fasta_3di)
    sequences_12st = read_fasta(fasta_12st)

    for rid in list(sequences_3di):
        if rid not in sequences_aa:
            logger.warning(f"Ignoring 3Di entry {rid}, not present in the amino-acid FASTA")
            del sequences_3di[rid]
    for rid in list(sequences_12st):
        if rid not in sequences_aa:
            logger.warning(
                f"Ignoring 12-state entry {rid}, not present in the amino-acid FASTA"
            )
            del sequences_12st[rid]

    # Drop AA entries with no structural prediction rather than aborting the
    # whole run — a handful of CDS can fail inference (OOM, zero length) and
    # the remaining thousands should still be searchable. This mirrors
    # generate_foldseek_db_from_aa_3di's behaviour in phold.
    for rid in list(sequences_aa):
        missing = []
        if rid not in sequences_3di:
            missing.append("3Di")
        if rid not in sequences_12st:
            missing.append("12-state")
        if missing:
            logger.warning(
                f"Entry {rid} has no corresponding {' or '.join(missing)} string; "
                "removing it from the Foldseek database"
            )
            del sequences_aa[rid]

    if not sequences_aa:
        raise ValueError(
            "No sequences left to build a Foldseek database from — every entry "
            "was missing a 3Di or 12-state prediction."
        )

    db_prefix = out_dir / db_name
    ss_db = out_dir / f"{db_name}_ss"
    header_db = out_dir / f"{db_name}_h"

    tmp_dir = out_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    aa_tsv = tmp_dir / "aa.tsv"
    threedi_tsv = tmp_dir / "3di.tsv"
    header_tsv = tmp_dir / "header.tsv"
    lookup_path = out_dir / f"{db_name}.lookup"

    # key -> (seq_id, combined bytes), used to patch _ss after tsv2db.
    combined_by_key: Dict[str, Tuple[str, bytes]] = {}

    with open(aa_tsv, "w") as f_aa, open(threedi_tsv, "w") as f_3di, \
            open(header_tsv, "w") as f_header, open(lookup_path, "w") as f_lookup:
        for i, rid in enumerate(sequences_aa, start=1):
            aa = sequences_aa[rid]
            d3 = sequences_3di[rid]
            s12 = sequences_12st[rid]

            if len(aa) != len(d3) or len(aa) != len(s12):
                raise ValueError(
                    f"Length mismatch for {rid}: aa={len(aa)} "
                    f"3di={len(d3)} 12st={len(s12)}"
                )

            combined_by_key[str(i)] = (rid, encode_combined(d3, s12, rid))

            f_aa.write(f"{i}\t{aa}\n")
            f_3di.write(f"{i}\t{d3}\n")
            f_header.write(f"{i}\t{rid}\n")
            f_lookup.write(f"{i}\t{rid}\t0\n")

    tsv2db_runner(aa_tsv, db_prefix, DBTYPE_AMINO_ACIDS)
    tsv2db_runner(threedi_tsv, ss_db, DBTYPE_SS_12ST)
    tsv2db_runner(header_tsv, header_db, DBTYPE_GENERIC)

    patch_ss_db_with_combined(ss_db, combined_by_key)

    if not keep_tsv:
        for path in (aa_tsv, threedi_tsv, header_tsv):
            path.unlink(missing_ok=True)
        # Only remove tmp/ if we put nothing else in it.
        try:
            tmp_dir.rmdir()
        except OSError:
            pass

    logger.info(
        f"Created Foldseek DBs: {db_prefix}, {ss_db} (combined 3Di+12st), {header_db}"
    )
    logger.info(
        "Combined encoding: c = 3di_index * 12 + ss12_index; "
        "search this DB with --ss-12st 1"
    )
    return db_prefix


def patch_ss_db_with_combined(
    ss_db: Path,
    combined_by_key: Dict[str, Tuple[str, bytes]],
) -> None:
    """Overwrite each ``_ss`` record's 3Di letters with combined 3Di+12st bytes.

    Seeks by the offsets recorded in ``<ss_db>.index`` — order-independent, so
    it does not matter how ``tsv2db`` chose to lay records out.

    Args:
        ss_db: Path to the ``_ss`` database file written by ``tsv2db``.
        combined_by_key: ``{db_key: (seq_id, combined_bytes)}``.

    Raises:
        ValueError: if the index is missing an expected key, or if a record's
            on-disk payload length disagrees with the combined bytes — either
            would mean the DB and the predictions have drifted apart.
    """
    index_path = Path(f"{ss_db}.index")
    if not index_path.is_file():
        raise ValueError(f"Foldseek did not produce an index at {index_path}")

    entries = []
    with open(index_path, "r") as fh:
        for line_no, line in enumerate(fh, 1):
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            key, offset, length = parts[0], int(parts[1]), int(parts[2])
            if key not in combined_by_key:
                raise ValueError(
                    f"{index_path}:{line_no} refers to DB key {key!r}, which was "
                    "not among the sequences written to the TSV."
                )
            entries.append((key, offset, length))

    if len(entries) != len(combined_by_key):
        raise ValueError(
            f"{index_path} lists {len(entries)} records but "
            f"{len(combined_by_key)} sequences were written."
        )

    with open(ss_db, "rb+") as fh:
        for key, offset, length in entries:
            seq_id, combined = combined_by_key[key]
            payload_len = length - _RECORD_SUFFIX_LEN
            if payload_len != len(combined):
                raise ValueError(
                    f"Record length mismatch for {seq_id} (DB key {key}): "
                    f"database holds {payload_len} residues, predictions have "
                    f"{len(combined)}."
                )
            fh.seek(offset)
            fh.write(combined)


def generate_sequence_foldseek_db(
    aa_sequences: Dict[str, str],
    foldseek_db_path: Path,
    db_name: str,
    tsv2db_runner: Tsv2DbRunner,
    keep_tsv: bool = False,
) -> Tuple[Path, Dict[str, str]]:
    """Build a plain amino-acid Foldseek DB and return its lookup.

    Used by the profile (``pssm``) path, which needs a header database and a
    stable ``sequence_id -> numeric key`` mapping before it can lay out the
    profile databases.

    Args:
        aa_sequences: Ordered ``{seq_id: amino-acid sequence}``. Keys are
            numbered from 1 in iteration order.
        foldseek_db_path: Directory the DB files are written into.
        db_name: DB basename; produces ``<db_name>``, ``<db_name>_h`` and
            ``<db_name>.lookup``.
        tsv2db_runner: Callable ``(in_tsv, out_db, dbtype)``.
        keep_tsv: Leave the intermediate TSVs on disk for debugging.

    Returns:
        ``(db_prefix, lookup)`` where lookup maps seq_id to its numeric key.
    """
    out_dir = Path(foldseek_db_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    db_prefix = out_dir / db_name
    header_db = out_dir / f"{db_name}_h"
    lookup_path = out_dir / f"{db_name}.lookup"

    tmp_dir = out_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    aa_tsv = tmp_dir / "aa.tsv"
    header_tsv = tmp_dir / "header.tsv"

    lookup: Dict[str, str] = {}
    with open(aa_tsv, "w") as f_aa, open(header_tsv, "w") as f_header, \
            open(lookup_path, "w") as f_lookup:
        for i, (seq_id, aa) in enumerate(aa_sequences.items(), start=1):
            lookup[seq_id] = str(i)
            f_aa.write(f"{i}\t{aa}\n")
            f_header.write(f"{i}\t{seq_id}\n")
            f_lookup.write(f"{i}\t{seq_id}\t0\n")

    tsv2db_runner(aa_tsv, db_prefix, DBTYPE_AMINO_ACIDS)
    tsv2db_runner(header_tsv, header_db, DBTYPE_GENERIC)

    if not keep_tsv:
        for path in (aa_tsv, header_tsv):
            path.unlink(missing_ok=True)
        try:
            tmp_dir.rmdir()
        except OSError:
            pass

    logger.info(f"Created amino-acid Foldseek DB {db_prefix} ({len(lookup)} sequences)")
    return db_prefix, lookup


def write_dbtype(path: Path, dbtype: int) -> None:
    """Write a Foldseek ``.dbtype`` file — a single little-endian int32."""
    Path(path).write_bytes(int(dbtype).to_bytes(4, byteorder="little", signed=True))


def create_single_shard_alias(data_path: Path) -> Path:
    """Create the ``<db>.0`` shard alias Foldseek expects for a single-shard DB.

    Hard-linked where the filesystem allows it so a multi-GB DB is not
    duplicated; falls back to a copy across filesystems.
    """
    data_path = Path(data_path)
    shard_path = Path(f"{data_path}.0")
    if shard_path.exists():
        shard_path.unlink()
    try:
        os.link(data_path, shard_path)
    except OSError:
        import shutil

        shutil.copy2(data_path, shard_path)
    return shard_path


def sort_index_file(index_path: Path) -> None:
    """Sort a Foldseek ``.index`` numerically by key, in place.

    Foldseek requires index keys in ascending order. Falls back to a
    lexicographic sort if the keys are not integers.
    """
    index_path = Path(index_path)
    with open(index_path, "r") as fh:
        lines = fh.readlines()

    try:
        lines.sort(key=lambda x: int(x.split()[0]))
    except (ValueError, IndexError):
        lines.sort()

    with open(index_path, "w") as fh:
        fh.writelines(lines)


def build_lookup(lookup_path: Path) -> Dict[str, str]:
    """Parse a Foldseek/MMseqs ``.lookup`` into ``{sequence_id: numeric_key}``.

    The file's columns are ``key<TAB>id<TAB>file_number``; this inverts the
    first two so predictions keyed by FASTA id can find their DB key.
    """
    lookup: Dict[str, str] = {}
    with open(lookup_path, "r") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                lookup[parts[1]] = parts[0]
    return lookup


def write_lookup(lookup: Dict[str, str], lookup_path: Path) -> None:
    """Write a Foldseek/MMseqs ``.lookup`` from ``{sequence_id: numeric_key}``."""
    with open(lookup_path, "w") as fh:
        for seq_id, key in lookup.items():
            fh.write(f"{key}\t{seq_id}\t0\n")
