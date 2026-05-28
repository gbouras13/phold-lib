# pholdlib

Shared ProstT5 inference library for [phold](https://github.com/gbouras13/phold) and [baktfold](https://github.com/gbouras13/baktfold).

pholdlib provides the common pLM Foldseek 3Di inference layer — this includes loading the ProstT5 protein language model & running the CNN prediction head and writing 3Di outputs and probabilities — so that this logic is consistent and not duplicated.

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

Unit tests load module files directly via `importlib` to avoid triggering the `transformers` import chain in `pholdlib/prostt5/__init__.py`, so they run without a working HuggingFace / scipy install. `conftest.py` also mocks `transformers` automatically if it cannot be imported, keeping the PyTorch-only `CNN` and `toCPU` tests green in broken environments.

Integration tests are gated behind `--run_integration`. When no `--checkpoint` is supplied they use a randomly initialised CNN predictor — predictions are meaningless but shapes, types, and probability ranges are all verified. Pass `--checkpoint` to run against the real trained weights.

