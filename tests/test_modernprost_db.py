"""Unit tests for the Foldseek database builders in pholdlib.modernprost.

Covers the combined 3Di+12st byte packing, the ``_ss`` patch-by-index step,
and the profile-DB record layout. Foldseek itself is stubbed with a pure-Python
``tsv2db`` that reproduces its on-disk format, so these run with no external
binary.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

_SRC = Path(__file__).parent.parent / "src" / "pholdlib"


def _load(mod_path: str, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _SRC / mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# output.py first — foldseek_db.py imports the alphabets from it.
_output = _load("modernprost/output.py", "pholdlib.modernprost.output")
_fsdb = _load("modernprost/foldseek_db.py", "pholdlib.modernprost.foldseek_db")
_pdb = _load("modernprost/profile_db.py", "pholdlib.modernprost.profile_db")

encode_combined = _fsdb.encode_combined
read_fasta = _fsdb.read_fasta
build_lookup = _fsdb.build_lookup
sort_index_file = _fsdb.sort_index_file
write_dbtype = _fsdb.write_dbtype
generate_combined_foldseek_db = _fsdb.generate_combined_foldseek_db
generate_sequence_foldseek_db = _fsdb.generate_sequence_foldseek_db
DBTYPE_SS_12ST = _fsdb.DBTYPE_SS_12ST

compute_log_pssm_3di = _pdb.compute_log_pssm_3di
compute_log_pssm_12st = _pdb.compute_log_pssm_12st
to_profile_record = _pdb.to_profile_record
aa_profile_record = _pdb.aa_profile_record
write_profile_foldseek_dbs = _pdb.write_profile_foldseek_dbs
parse_profiles = _pdb.parse_profiles
AA_ALPHABET = _pdb.AA_ALPHABET
PROFILE_RECORD_STRIDE = _pdb.PROFILE_RECORD_STRIDE


# ---------------------------------------------------------------------------
# Foldseek tsv2db stub
# ---------------------------------------------------------------------------

def fake_tsv2db(in_tsv: Path, out_db: Path, dbtype: int) -> None:
    """Reproduce ``foldseek tsv2db``: records of ``<payload>\\n\\0`` plus an index."""
    payloads = []
    for line in Path(in_tsv).read_text().splitlines():
        if not line:
            continue
        key, value = line.split("\t", 1)
        payloads.append((key, value))

    data = bytearray()
    index_lines = []
    for key, value in payloads:
        offset = len(data)
        record = value.encode("ascii") + b"\n\x00"
        data.extend(record)
        index_lines.append(f"{key}\t{offset}\t{len(record)}")

    Path(out_db).write_bytes(bytes(data))
    Path(f"{out_db}.index").write_text("\n".join(index_lines) + "\n")
    write_dbtype(Path(f"{out_db}.dbtype"), dbtype)


def _write_fasta(path: Path, seqs: dict) -> None:
    with open(path, "w") as fh:
        for k, v in seqs.items():
            fh.write(f">{k}\n{v}\n")


# ===========================================================================
# Combined encoding
# ===========================================================================

class TestEncodeCombined:
    def test_packing_formula(self):
        # 3Di 'A'=0 'C'=1 ; 12st 'A'=0 'N'=11
        assert list(encode_combined("AC", "AN")) == [0, 1 * 12 + 11]

    def test_full_range_roundtrip(self):
        seq_3di = _output.THREEDI_ALPHABET
        seq_12st = (_output.SS12_ALPHABET * 2)[:20]
        packed = encode_combined(seq_3di, seq_12st)
        assert all(0 <= c <= 239 for c in packed)
        assert "".join(_output.THREEDI_ALPHABET[c // 12] for c in packed) == seq_3di
        assert "".join(_output.SS12_ALPHABET[c % 12] for c in packed) == seq_12st

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            encode_combined("AAA", "AA", "seq1")

    def test_masked_3di_rejected_with_actionable_message(self):
        with pytest.raises(ValueError, match="masked low-confidence residue"):
            encode_combined("AX", "AA", "seq1")

    def test_unknown_12st_character_rejected(self):
        with pytest.raises(ValueError, match="Unknown 12-state character"):
            encode_combined("AA", "AZ", "seq1")


# ===========================================================================
# FASTA reader
# ===========================================================================

class TestReadFasta:
    def test_multiline_and_id_truncation(self, tmp_path):
        p = tmp_path / "in.fasta"
        p.write_text(">a description here\nMKT\nIIA\n>b\nQQ\n")
        assert read_fasta(p) == {"a": "MKTIIA", "b": "QQ"}

    def test_preserves_order_and_case(self, tmp_path):
        p = tmp_path / "in.fasta"
        p.write_text(">z\nAc\n>a\nGg\n")
        assert list(read_fasta(p)) == ["z", "a"]
        assert read_fasta(p)["z"] == "Ac"


# ===========================================================================
# Combined DB construction
# ===========================================================================

class TestCombinedDb:
    def _build(self, tmp_path, aa, d3i, s12):
        _write_fasta(tmp_path / "aa.fasta", aa)
        _write_fasta(tmp_path / "3di.fasta", d3i)
        _write_fasta(tmp_path / "12st.fasta", s12)
        return generate_combined_foldseek_db(
            tmp_path / "aa.fasta",
            tmp_path / "3di.fasta",
            tmp_path / "12st.fasta",
            tmp_path / "db",
            "q",
            fake_tsv2db,
        )

    def test_ss_payload_is_patched_with_combined_bytes(self, tmp_path):
        aa = {"s1": "MKTI", "s2": "MASMT"}
        d3i = {"s1": "DVQA", "s2": "PPDDV"}
        s12 = {"s1": "ACDE", "s2": "AACCD"}
        self._build(tmp_path, aa, d3i, s12)

        raw = (tmp_path / "db" / "q_ss").read_bytes()
        lookup = build_lookup(tmp_path / "db" / "q.lookup")
        index = {}
        for line in (tmp_path / "db" / "q_ss.index").read_text().splitlines():
            key, offset, length = line.split("\t")
            index[key] = (int(offset), int(length))

        for sid in aa:
            offset, length = index[lookup[sid]]
            payload = raw[offset : offset + length - 2]
            assert payload == encode_combined(d3i[sid], s12[sid], sid)
            # The record terminator must survive the patch.
            assert raw[offset + length - 2 : offset + length] == b"\n\x00"

    def test_dbtype_marks_the_12st_structure_db(self, tmp_path):
        aa, d3i, s12 = {"s1": "MK"}, {"s1": "DV"}, {"s1": "AC"}
        self._build(tmp_path, aa, d3i, s12)
        raw = (tmp_path / "db" / "q_ss.dbtype").read_bytes()
        assert int.from_bytes(raw, "little", signed=True) == DBTYPE_SS_12ST

    def test_lookup_maps_ids_to_keys_from_one(self, tmp_path):
        aa = {"s1": "MK", "s2": "QQ"}
        self._build(tmp_path, aa, {"s1": "DV", "s2": "AA"}, {"s1": "AC", "s2": "AA"})
        assert build_lookup(tmp_path / "db" / "q.lookup") == {"s1": "1", "s2": "2"}

    def test_entry_without_prediction_is_dropped_not_fatal(self, tmp_path):
        aa = {"s1": "MK", "s2": "QQ"}
        self._build(tmp_path, aa, {"s1": "DV"}, {"s1": "AC"})
        assert build_lookup(tmp_path / "db" / "q.lookup") == {"s1": "1"}

    def test_all_entries_missing_predictions_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No sequences left"):
            self._build(tmp_path, {"s1": "MK"}, {}, {})

    def test_length_disagreement_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Length mismatch"):
            self._build(tmp_path, {"s1": "MKT"}, {"s1": "DV"}, {"s1": "AC"})

    def test_intermediate_tsvs_are_cleaned_up(self, tmp_path):
        self._build(tmp_path, {"s1": "MK"}, {"s1": "DV"}, {"s1": "AC"})
        assert not (tmp_path / "db" / "tmp" / "aa.tsv").exists()


# ===========================================================================
# Index sorting
# ===========================================================================

class TestSortIndexFile:
    def test_numeric_not_lexicographic(self, tmp_path):
        p = tmp_path / "x.index"
        p.write_text("10\t0\t5\n2\t5\t5\n1\t10\t5\n")
        sort_index_file(p)
        assert [line.split("\t")[0] for line in p.read_text().splitlines()] == [
            "1",
            "2",
            "10",
        ]


# ===========================================================================
# PSSM maths
# ===========================================================================

class TestPssm:
    def test_3di_shape_and_dtype(self):
        profile = np.full((7, 20), 1 / 20, dtype=np.float32)
        pssm, consensus = compute_log_pssm_3di(profile)
        assert pssm.shape == (7 * 20,)
        assert pssm.dtype == np.int8
        assert consensus.shape == (7,)
        assert consensus.dtype == np.int8

    def test_3di_rejects_wrong_width(self):
        with pytest.raises(ValueError, match="20-column"):
            compute_log_pssm_3di(np.zeros((3, 12), dtype=np.float32))

    def test_12st_pads_to_20_columns(self):
        profile = np.full((4, 12), 1 / 12, dtype=np.float32)
        pssm, consensus = compute_log_pssm_12st(profile)
        assert pssm.shape == (4 * 20,)
        # The 8 padding columns hold zero probability, so they saturate low.
        padded = pssm.reshape(4, 20)[:, 12:]
        assert (padded == -128).all()

    def test_12st_rejects_wrong_width(self):
        with pytest.raises(ValueError, match="12-column"):
            compute_log_pssm_12st(np.zeros((3, 20), dtype=np.float32))

    def test_consensus_is_argmax(self):
        profile = np.zeros((3, 12), dtype=np.float32)
        profile[0, 5] = profile[1, 0] = profile[2, 11] = 1.0
        _, consensus = compute_log_pssm_12st(profile)
        assert consensus.tolist() == [5, 0, 11]

    def test_zero_probability_saturates_at_floor(self):
        profile = np.zeros((1, 20), dtype=np.float32)
        profile[0, 0] = 1.0
        pssm, _ = compute_log_pssm_3di(profile)
        assert pssm[1:20].tolist() == [-128] * 19
        # A state 20x above background scores well above zero.
        assert pssm[0] > 0


# ===========================================================================
# Profile record layout
# ===========================================================================

class TestProfileRecord:
    def test_record_stride(self):
        pssm = np.zeros(3 * 20, dtype=np.int8)
        query = np.array([1, 2, 3])
        consensus = np.array([4, 5, 6])
        record = to_profile_record(pssm, query, consensus)
        assert len(record) == 3 * PROFILE_RECORD_STRIDE

    def test_trailing_bytes(self):
        pssm = np.zeros(20, dtype=np.int8)
        record = to_profile_record(pssm, np.array([7]), np.array([9]), neff=1)
        assert record[20] == 7  # query
        assert record[21] == 9  # consensus
        assert record[22] == 1  # neff
        assert record[23:25] == b"\x00\x00"  # reserved

    def test_negative_scores_are_stored_as_twos_complement(self):
        pssm = np.full(20, -1, dtype=np.int8)
        record = to_profile_record(pssm, np.array([0]), np.array([0]))
        assert record[0] == 0xFF

    def test_length_validation(self):
        with pytest.raises(ValueError, match="pssm has"):
            to_profile_record(np.zeros(19, dtype=np.int8), np.array([0]), np.array([0]))
        with pytest.raises(ValueError, match="query has"):
            to_profile_record(
                np.zeros(20, dtype=np.int8), np.array([0, 1]), np.array([0])
            )


class TestAaProfileRecord:
    def test_one_block_per_residue(self):
        assert len(aa_profile_record("MKTII")) == 5 * PROFILE_RECORD_STRIDE

    def test_x_is_encoded_not_skipped(self):
        # X-skipping would desynchronise the AA profile from the 3Di profile
        # for any masked residue, which is the bug this guards against.
        assert len(aa_profile_record("MXK")) == 3 * PROFILE_RECORD_STRIDE
        assert aa_profile_record("X")[20] == AA_ALPHABET.index("X")

    def test_unknown_residues_fall_back_to_x(self):
        assert aa_profile_record("O") == aa_profile_record("X")
        assert aa_profile_record("m") == aa_profile_record("M")

    def test_query_and_consensus_index_the_mmseqs_alphabet(self):
        # 'P' is column 12 of the BLOSUM62 header used for the profile scores;
        # query/consensus must agree with that indexing.
        assert aa_profile_record("P")[20] == 12
        assert aa_profile_record("P")[21] == 12


# ===========================================================================
# Profile DB assembly
# ===========================================================================

class TestWriteProfileDbs:
    def _fixture(self, tmp_path):
        aa = {"s1": "MKTI", "s2": "QQPPG"}
        source, lookup = generate_sequence_foldseek_db(
            aa, tmp_path / "prof", "q", fake_tsv2db
        )
        rng = np.random.default_rng(0)
        p3 = {k: rng.dirichlet(np.ones(20), len(v)).astype(np.float32) for k, v in aa.items()}
        p12 = {k: rng.dirichlet(np.ones(12), len(v)).astype(np.float32) for k, v in aa.items()}
        return aa, source, lookup, p3, p12

    def test_all_three_dbs_have_matching_record_sizes(self, tmp_path):
        aa, source, lookup, p3, p12 = self._fixture(tmp_path)
        paths = write_profile_foldseek_dbs(
            p3, p12, aa, lookup, tmp_path / "prof", "q", source
        )
        expected = sum(len(v) for v in aa.values()) * PROFILE_RECORD_STRIDE
        for path in paths.values():
            assert path.stat().st_size == expected

    def test_dbtype_and_shard_alias_written(self, tmp_path):
        aa, source, lookup, p3, p12 = self._fixture(tmp_path)
        paths = write_profile_foldseek_dbs(
            p3, p12, aa, lookup, tmp_path / "prof", "q", source
        )
        for path in paths.values():
            assert Path(f"{path}.0").exists()
            raw = Path(f"{path}.dbtype").read_bytes()
            assert int.from_bytes(raw, "little", signed=True) == _pdb.DBTYPE_HMM_PROFILE

    def test_header_db_is_copied(self, tmp_path):
        aa, source, lookup, p3, p12 = self._fixture(tmp_path)
        write_profile_foldseek_dbs(p3, p12, aa, lookup, tmp_path / "prof", "q", source)
        for suffix in ("_h", "_h.index", "_h.dbtype"):
            assert (tmp_path / "prof" / f"q_profile{suffix}").exists()

    def test_missing_12st_profile_raises(self, tmp_path):
        aa, source, lookup, p3, p12 = self._fixture(tmp_path)
        del p12["s1"]
        with pytest.raises(ValueError, match="No 12-state profile"):
            write_profile_foldseek_dbs(
                p3, p12, aa, lookup, tmp_path / "prof", "q", source
            )

    def test_missing_lookup_key_raises(self, tmp_path):
        aa, source, lookup, p3, p12 = self._fixture(tmp_path)
        del lookup["s1"]
        with pytest.raises(ValueError, match="No numeric Foldseek key"):
            write_profile_foldseek_dbs(
                p3, p12, aa, lookup, tmp_path / "prof", "q", source
            )

    def test_length_disagreement_raises(self, tmp_path):
        aa, source, lookup, p3, p12 = self._fixture(tmp_path)
        p3["s1"] = p3["s1"][:-1]
        with pytest.raises(ValueError, match="Length mismatch"):
            write_profile_foldseek_dbs(
                p3, p12, aa, lookup, tmp_path / "prof", "q", source
            )

    def test_indexes_are_key_sorted(self, tmp_path):
        aa, source, lookup, p3, p12 = self._fixture(tmp_path)
        paths = write_profile_foldseek_dbs(
            p3, p12, aa, lookup, tmp_path / "prof", "q", source
        )
        for path in paths.values():
            keys = [
                int(line.split("\t")[0])
                for line in Path(f"{path}.index").read_text().splitlines()
            ]
            assert keys == sorted(keys)


# ===========================================================================
# Profile text roundtrip
# ===========================================================================

class TestParseProfiles:
    def test_roundtrip_through_text(self, tmp_path):
        profiles = {
            "a": np.array([[0.25, 0.75], [0.5, 0.5]], dtype=np.float32),
            "b": np.array([[0.1, 0.9]], dtype=np.float32),
        }
        out = tmp_path / "prof.txt"
        _output.write_profiles_text(profiles, out, "AC", ["a", "b"])

        parsed = dict(parse_profiles(out))
        assert list(parsed) == ["a", "b"]
        for key, expected in profiles.items():
            assert np.allclose(parsed[key], expected, atol=1e-4)
