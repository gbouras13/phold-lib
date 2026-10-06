# pholdlib

Shared protein-language-model inference library for [phold](https://github.com/gbouras13/phold) and [baktfold](https://github.com/gbouras13/baktfold).

pholdlib provides the common pLM → Foldseek layer so that this logic is consistent and not duplicated. It supports two backends:

- **ProstT5** (`pholdlib.prostt5`) — the ProstT5 encoder plus a CNN prediction head, emitting Foldseek 3Di.
- **ModernProst** (`pholdlib.modernprost`) — a single ModernBERT-style encoder emitting Foldseek **3Di and a 12-state secondary-structure alphabet** in one forward pass, plus the Foldseek query-database construction for both (combined `_ss` sequence DBs and 3Di/12-state profile DBs).

## Citation

If you use pholdlib in your research, please cite:


> Bouras G., Grigson S.R., Mirdita M., Heinzinger M., Papudeshi B.,  
> Mallawaarachchi V., Green R., Kim S.R., Mihalia V., Psaltis A.J.,  
> Wormald P-J., Vreugde S., Steinegger M., Edwards R.A.  
>  
> *Protein Structure Informed Bacteriophage Genome Annotation with Phold*  
> **Nucleic Acids Research**, Volume 54, Issue 1, 13 January 2026  
> https://doi.org/10.1093/nar/gkaf1448

## License

MIT

## Installation

```bash
pip install pholdlib
```

With test dependencies:

```bash
pip install "pholdlib[test]"
```

## Quick start

```python
from pathlib import Path
import torch

from pholdlib.prostt5.model import get_T5_model, load_predictor
from pholdlib.prostt5.inference import run_prostt5_inference
from pholdlib.prostt5.output import SS_MAPPING, write_probs, write_fail_ids

# 1. Load ProstT5 + CNN predictor head
model, vocab, device = get_T5_model(
    model_dir=Path("/path/to/model_cache"),
    model_name="Rostlab/ProstT5_fp16",
    cpu=True,
    threads=4,
)
predictor = load_predictor("/path/to/cnn_checkpoint.pt", device)

# 2. Prepare sequences: list of (id, sequence, length), sorted descending by length
seqs = [
    ("protein_A", "MKTIIALSYIFCLVFA", 16),
    ("protein_B", "MASMTGGQQMGRDLY", 15),
]

# 3. Run inference
predictions, emb_residue, emb_protein, fail_ids = run_prostt5_inference(
    seq_dict=seqs,
    model=model,
    vocab=vocab,
    predictor=predictor,
    device=device,
    output_probs=True,
    save_per_residue_embeddings=False,
    save_per_protein_embeddings=False,
)

# 4. Decode predictions
for seq_id, (pred_array, mean_prob, all_probs) in predictions.items():
    threedi = "".join(SS_MAPPING[int(c)] for c in pred_array)
    print(f"{seq_id}: {threedi}  (mean confidence {mean_prob:.1f}%)")
```

## API

### `pholdlib.prostt5.model`

| Function / Class | Description |
|---|---|
| `get_T5_model(model_dir, model_name, cpu, threads, check_fn, zenodo_fn)` | Load ProstT5 encoder + tokenizer; optionally download via HuggingFace with a Zenodo fallback. Returns `(model, vocab, device)`. |
| `load_predictor(checkpoint_path, device)` | Load a CNN prediction head from a `.pt` checkpoint or state-dict file. Returns the CNN in eval mode. |
| `CNN` | Two-layer Conv2d head: `(B, L, 1024)` embeddings → `(B, 20, L)` 3Di logits. |
| `toCPU(tensor)` | Detach a tensor, move to CPU, and return a NumPy array. |

### `pholdlib.prostt5.inference`

| Function | Description |
|---|---|
| `run_prostt5_inference(seq_dict, model, vocab, predictor, device, ...)` | Batch inference over a list of `(id, sequence, length)` tuples. Returns `(predictions, embeddings_per_residue, embeddings_per_protein, fail_ids)`. |

`predictions` is a dict `{seq_id: (pred_array, mean_prob, all_prob)}`:

- `pred_array` — `np.byte` array of 3Di class indices (0–19; 20 = masked).
- `mean_prob` — mean per-residue confidence, 0–100.
- `all_prob` — `float32` array of shape `(1, L)`, or `None` when `output_probs=False`.

Key batching parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `max_residues` | 100,000 | Max total residues per batch before flush |
| `max_seq_len` | 30,000 | Any single sequence longer than this triggers an immediate flush |
| `max_batch` | 10,000 | Max sequences per batch |

### `pholdlib.prostt5.output`

| Function / Constant | Description |
|---|---|
| `SS_MAPPING` | `{0…20: char}` — maps 3Di class index to single-letter code (20 → `'X'` masked). |
| `mask_low_confidence_aa(sequence, scores, threshold)` | Replace residues whose confidence score is below `threshold` with `'X'`. `scores` should be a shape-`(1, L)` array or equivalent. |
| `write_probs(predictions, output_path_mean, output_path_all, original_keys)` | Write mean probabilities to a CSV and per-residue probabilities to a JSONL file. |
| `write_fail_ids(fail_ids, out_path)` | Write a list of failed sequence IDs to a TSV. No-ops on an empty list. |

### `pholdlib.databases.prostt5`

| Function / Constant | Description |
|---|---|
| `PROSTT5_MD5_DICTIONARY` | MD5 hashes for `Rostlab/ProstT5_fp16` model files. |
| `check_prostT5_download(model_dir, model_name, md5_dict, model_subdir)` | Returns `True` if the model is absent or corrupt and needs to be (re)downloaded. |
| `download_zenodo_prostT5(model_dir, logdir, threads, backup_url, backup_md5, backup_tarball)` | Download and extract a ProstT5 tarball from a Zenodo backup URL. |

## ModernProst (3Di + 12-state)

ModernProst maps an amino-acid sequence directly to Foldseek's 3Di alphabet **and** a 12-state secondary-structure alphabet in a single forward pass — no ProstT5 encoder, no separate CNN head. Four checkpoints are published:

| Model | Size | Default task | Query DB built |
|---|---|---|---|
| `modernprost-base` | ~1.0 B | `classification` | combined `_ss` |
| `modernprost-50M` | ~53 M | `classification` | combined `_ss` |
| `modernprost-pssm` | ~1.0 B | `pssm` | `_profile_ss` / `_profile_ss12` / `_profile` |
| `modernprost-50M-pssm` | ~53 M | `pssm` | `_profile_ss` / `_profile_ss12` / `_profile` |

The `-pssm` checkpoints are trained to emit calibrated per-residue distributions, which are written straight into Foldseek profile databases rather than being collapsed to an argmax letter.

Requires `transformers >= 4.51` (the checkpoints are HuggingFace `custom_code` models loaded with `trust_remote_code=True`; tested with 5.9, 5.15 and 5.17). Older releases load the classification checkpoints but not the `-pssm` ones, whose config import sits in a `try/except` that the pre-4.51 import scanner misreads as a missing PyPI package. The ProstT5 backend still works on older `transformers`.

```python
from pathlib import Path

from pholdlib.modernprost import (
    get_modernprost_model, run_modernprost_inference,
    write_3di_fasta, write_12st_fasta, generate_combined_foldseek_db,
    subprocess_tsv2db_runner,
)

model, tokenizer, device = get_modernprost_model(
    model_dir=Path("/path/to/model_cache"),
    model_name="modernprost-50M",
    cpu=False, threads=4, half_precision=True,
)

seqs = [("protein_A", "MKTIIALSYIFCLVFA", 16)]   # (id, sequence, length), any order
result = run_modernprost_inference(seqs, model, tokenizer, device, task="classification")

keys = [s[0] for s in seqs]
# NOTE mask_threshold must stay 0 when the FASTA feeds a combined 3Di+12st DB.
write_3di_fasta(result.predictions_3di, "3di.fasta", keys, mask_threshold=0)
write_12st_fasta(result.predictions_12st, "12st.fasta", keys)

generate_combined_foldseek_db(
    "aa.fasta", "3di.fasta", "12st.fasta",
    foldseek_db_path="foldseek_db", db_name="query",
    tsv2db_runner=subprocess_tsv2db_runner("foldseek"),
)
```

### `pholdlib.modernprost.model` / `.inference`

| Function | Description |
|---|---|
| `get_modernprost_model(model_dir, model_name, cpu, threads, check_fn, zenodo_fn, device, half_precision)` | Load a ModernProst model + tokenizer. Returns `(model, tokenizer, device)`. |
| `run_modernprost_inference(seq_dict, model, tokenizer, device, task=..., engine="auto", compile="auto", max_residues=65536, ...)` | Batch inference with one loaded model. Returns a `ModernProstResult`. |
| `run_modernprost_inference_multi_gpu(seq_dict, devices, model_dir, model_name, ...)` | Tokenises and plans once; one worker per device pulls batches from a shared queue into one set of output buffers. Workers are threads for eager runs; with `torch.compile` on several GPUs they are processes (this one drives the first GPU, spawned workers the rest, sharing buffers through shared memory), because compilation is serialised within a process (~20-25 s per A100) but runs in parallel across processes. |

`ModernProstResult` fields:

- `predictions_3di` / `predictions_12st` — `{seq_id: (pred, mean_prob, all_prob)}`, `classification` task only.
- `profiles_3di` / `profiles_12st` — `{seq_id: ndarray (L, 20 | 12)}` softmax profiles, `pssm` task only.
- `embeddings_per_residue` / `embeddings_per_protein`, `fail_ids`.

Sequences longer than `chunk_len` are split into near-equal chunks, inferred independently, and stitched back together; the reported mean confidence is recomputed over the full-length sequence. Returned arrays are views into per-run buffers: copy one before modifying it in place.

#### Inference engines

The input is tokenised once (a 256-entry lookup table built from the tokenizer's vocabulary) and run through one of two engines. Both reduce the logits on the GPU, copy back only real tokens, and overlap host-side bookkeeping with the next batch.

| engine | how | where |
|---|---|---|
| `padded` | one protein per row, longest first, rows padded to a multiple of 8; a batch is bounded by *computed* tokens (`rows x padded length <= max_residues`; default 65,536 on CUDA/ROCm, 4,096 on Apple MPS and CPU, where attention materialises the score matrix and small batches run faster) | any device; the checkpoint's own forward |
| `packed` | proteins concatenated into rows of `pack_tokens` (65,536) tokens; per-protein RoPE positions; FlexAttention block masks keep each protein to itself and give the sliding-window layers their real O(L x window) cost; compiled once | CUDA or ROCm GPU, torch >= 2.5, Triton |

`engine="auto"` with `compile="auto"` picks, for inputs of at least 20M residues (5M for the ~1B checkpoints, whose per-residue savings repay compiling sooner; both doubled on ROCm, where compiling saves less): `packed` + `torch.compile` on NVIDIA GPUs, `padded` + `torch.compile(dynamic=True)` on AMD/ROCm GPUs (FlexAttention's Triton kernels are slow on MI250X), and the eager `padded` engine for smaller inputs and on CPU/MPS. Set the threshold with `PHOLDLIB_PACKED_MIN_RESIDUES`, or choose explicitly (`engine="padded"|"packed"`, `compile="on"|"off"`).

Single-GPU throughput, ModernProst-50M, 111,158 phage proteins (24.7M residues):

| setup | A100-40GB | MI250X GCD |
|---|---|---|
| pholdlib 0.2 (`max_batch=20`) | 139k residues/s | 108k residues/s |
| `padded`, eager | 298k | 205k |
| `padded`, compiled | 510k | 344k |
| `packed`, compiled | 623k | 293k |

Compiling costs ~30-60 s on the first batch (cold cache). Keep compile caches between runs with `TORCHINDUCTOR_CACHE_DIR` / `TRITON_CACHE_DIR`, on node-local disk on clusters (never on Lustre).

`torch.compile` needs Triton's JIT toolchain. The pip wheels bundle it. conda-forge's Triton instead finds `ptxas` through the *activated* env and builds small launcher modules with the system gcc against `cuda.h`, which an env holding only the CUDA runtime does not have: add the CUDA headers (e.g. conda-forge `cuda-driver-dev`) or point `CPATH` at a CUDA toolkit's `include/`. If compiling fails for any reason, that device falls back to the eager `padded` engine with a warning instead of failing the run.

### `pholdlib.modernprost.output`

| Function / Constant | Description |
|---|---|
| `THREEDI_ALPHABET` / `SS12_ALPHABET` | `"ACDEFGHIKLMNPQRSTVWY"` and `"ACDEFGHIKLMN"`. |
| `write_3di_fasta(...)` / `write_12st_fasta(...)` | FASTA writers. Only the 3Di writer takes a `mask_threshold`. |
| `write_mean_probs(...)` / `write_all_probs(...)` | Mean-confidence CSV and per-residue JSONL. |
| `write_profiles_text(profiles, out_path, alphabet, original_keys)` | Human-readable profile blocks (`pssm` task). |

### `pholdlib.modernprost.foldseek_db` / `.profile_db`

| Function | Description |
|---|---|
| `generate_combined_foldseek_db(fasta_aa, fasta_3di, fasta_12st, foldseek_db_path, db_name, tsv2db_runner)` | Build the combined 3Di+12st query DB. |
| `generate_sequence_foldseek_db(aa_sequences, foldseek_db_path, db_name, tsv2db_runner)` | Build a plain AA DB + lookup, used as the source for profile DBs. |
| `write_profile_foldseek_dbs(profiles_3di, profiles_12st, aa_sequences, lookup, output_dir, db_name, source_db)` | Build `_profile_ss`, `_profile_ss12` and `_profile`. |
| `encode_combined(seq_3di, seq_12st)` | Pack one residue pair per byte: `c = 3di_index * 12 + ss12_index`. |

Search a combined or profile DB with `--ss-12st 1`; profile DBs additionally want `--sort-by-structure-bits 0`.

**Masking caveat.** The combined encoding has no 21st 3Di state, so a masked `X` cannot be represented — `encode_combined` rejects it with an actionable error. Write the 3Di FASTA with `mask_threshold=0` and apply confidence masking to the amino-acid FASTA instead.

### `pholdlib.databases.modernprost`

| Function / Constant | Description |
|---|---|
| `MODERNPROST_MODELS` | Registry of the four published checkpoints. |
| `resolve_modernprost_model(name)` | Look up by short name or full HuggingFace id. |
| `is_modernprost_model(name)` / `default_task_for(name)` | Branch between backends; get a checkpoint's trained task. |
| `check_modernprost_download(model_dir, model_name, md5_dict)` | `True` when the model needs (re)downloading. Falls back to a snapshot completeness check when no MD5 dict is registered. |
| `download_zenodo_modernprost(...)` | Zenodo backup download + extract. |

## Notes for downstream tools

- **`check_fn` / `zenodo_fn` hooks** — `get_T5_model` accepts optional callables for model integrity checking and Zenodo fallback download. Tool-specific implementations (e.g. phold's `check_prostT5_download`) are passed in; pholdlib ships the base `Rostlab/ProstT5_fp16` MD5 dict and a generic `download_zenodo_prostT5` that callers configure with their own backup URL.
- **FP32 on CPU** — `get_T5_model` automatically casts the model to `float()` when `cpu=True` to avoid errors with half-precision operations on CPU.
- **Sequence pre-processing** — `run_prostt5_inference` replaces `U`, `Z`, and `O` with `X` internally; callers do not need to sanitise sequences beforehand.

## Testing

```bash
# Install test dependencies
pip install "pholdlib[test]"

# Unit tests — no model download, no GPU required (completes in ~1 s)
pytest tests/

# Integration tests — ProstT5 is downloaded automatically on first run
# (~1.6 GB fp16) into tests/test_data/model_cache/
pytest tests/ --run_integration

# Point at a pre-existing model cache
pytest tests/ --run_integration --model_dir /path/to/model_cache

# Use the real trained phold CNN weights (ships with the phold repo)
pytest tests/ --run_integration \
    --checkpoint /path/to/phold/src/phold/cnn/cnn_chkpnt/model.pt

# With GPU + multiple threads
pytest tests/ --run_integration --gpu_available --threads 8
```

### Test organisation

| File | Contents |
|---|---|
| `tests/test_output.py` | `SS_MAPPING`, `mask_low_confidence_aa`, `write_probs`, `write_fail_ids` |
| `tests/test_databases.py` | MD5 dict structure, `_calc_md5`, `check_prostT5_download`, `download_zenodo_prostT5` |
| `tests/test_model.py` | `CNN` forward pass and architecture, `toCPU`, `load_predictor` |
| `tests/test_integration.py` | Model loading, tokenizer, predictor, batch inference, per-residue/protein embeddings, output round-trip |
| `tests/test_modernprost_output.py` | 3Di/12-state alphabets, FASTA + probability + profile-text writers, masking |
| `tests/test_modernprost_db.py` | Combined 3Di+12st byte packing, `_ss` patch-by-index, PSSM maths, profile-DB record layout |
| `tests/test_modernprost_registry.py` | Model registry resolution, snapshot integrity check, chunking and chunk-id helpers |

The ModernProst database tests stub `foldseek tsv2db` with a pure-Python equivalent, so the whole unit suite still runs with no external binary.

Unit tests load module files directly via `importlib` to avoid triggering the `transformers` import chain in `pholdlib/prostt5/__init__.py`, so they run without a working HuggingFace / scipy install. `conftest.py` also mocks `transformers` automatically if it cannot be imported, keeping the PyTorch-only `CNN` and `toCPU` tests green in broken environments.

Integration tests are gated behind `--run_integration`. When no `--checkpoint` is supplied they use a randomly initialised CNN predictor — predictions are meaningless but shapes, types, and probability ranges are all verified. Pass `--checkpoint` to run against the real trained weights.

