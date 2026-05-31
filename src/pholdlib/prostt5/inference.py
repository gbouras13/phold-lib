#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared ProstT5 batch inference engine.

Code adapted from @mheinzinger
https://github.com/mheinzinger/ProstT5/blob/main/scripts/predict_3Di_encoderOnly.py
"""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

from pholdlib.prostt5.device import shard_round_robin
from pholdlib.prostt5.model import CNN, get_T5_model, load_predictor, toCPU

PROSTT5_PREFIX = "<AA2fold>"


def run_prostt5_inference(
    seq_dict: List[Tuple[str, str, int]],
    model,
    vocab,
    predictor: CNN,
    device: torch.device,
    max_residues: int = 100000,
    max_seq_len: int = 30000,
    max_batch: int = 10000,
    output_probs: bool = True,
    save_per_residue_embeddings: bool = False,
    save_per_protein_embeddings: bool = False,
    desc: str = "Predicting 3Di",
    progress_callback: Optional[Callable[[int], None]] = None,
) -> Tuple[Dict, Dict, Dict, List]:
    """
    Run ProstT5 + CNN prediction head on a list of protein sequences.

    Args:
        seq_dict: Pre-sorted (descending by length) list of (seq_id, sequence, length)
                  tuples. Sequences should already have non-standard AAs replaced with X.
        model: ProstT5 encoder (from get_T5_model).
        vocab: ProstT5 tokenizer (from get_T5_model).
        predictor: CNN prediction head (from load_predictor).
        device: Torch device (from get_T5_model).
        max_residues: Max total residues per batch before flushing.
        max_seq_len: Any single sequence longer than this triggers an immediate flush.
        max_batch: Max number of sequences per batch.
        output_probs: If True, per-residue probabilities are stored in predictions.
                      If False, the third element of each tuple is None.
        save_per_residue_embeddings: If True, per-residue embeddings are collected.
        save_per_protein_embeddings: If True, mean per-protein embeddings are collected.
        desc: tqdm progress bar description.
        progress_callback: Optional callable invoked as ``progress_callback(n)``
                           after each flushed batch, where ``n`` is the number
                           of sequences in that batch. When provided, the
                           internal tqdm bar is disabled — used by the
                           multi-GPU dispatcher to drive a single shared bar.

    Returns:
        predictions: {seq_id: (pred_array, mean_prob, all_prob_or_None)}
                     pred_array is a byte numpy array of 3Di class indices (0–19).
                     mean_prob is a float (0–100).
                     all_prob is a float32 numpy array of shape (1, L) or None.
        embeddings_per_residue: {seq_id: ndarray (L x 1024)} — empty if flag is False.
        embeddings_per_protein: {seq_id: ndarray (1024,)} — empty if flag is False.
        fail_ids: List of seq_ids that could not be processed.
    """
    predictions: Dict = {}
    embeddings_per_residue: Dict = {}
    embeddings_per_protein: Dict = {}
    batch_predictions: Dict = {}
    batch_embeddings_per_residue: Dict = {}
    batch_embeddings_per_protein: Dict = {}
    fail_ids: List[str] = []

    batch: List = []

    # When a progress_callback is supplied (multi-GPU dispatcher path) we
    # silence the per-shard tqdm bar — the caller drives a single shared bar.
    use_internal_bar = progress_callback is None
    iterator = (
        tqdm(seq_dict, desc=desc)
        if use_internal_bar
        else seq_dict
    )

    for seq_idx, (pdb_id, seq, slen) in enumerate(iterator, 1):
        seq = seq.replace("U", "X").replace("Z", "X").replace("O", "X")
        seq_len = len(seq)
        seq_tokenized = PROSTT5_PREFIX + " " + " ".join(list(seq))
        batch.append((pdb_id, seq_tokenized, seq_len))

        n_res_batch = sum(s_len for _, _, s_len in batch) + seq_len
        flush = (
            len(batch) >= max_batch
            or n_res_batch >= max_residues
            or seq_idx == len(seq_dict)
            or seq_len > max_seq_len
        )

        if not flush:
            continue

        pdb_ids, seqs, seq_lens = zip(*batch)
        batch = []
        _batch_n = len(pdb_ids)

        token_encoding = vocab(
            seqs,
            add_special_tokens=True,
            padding="longest",
            truncation=False,
            return_tensors="pt",
        ).to(device)

        try:
            with torch.no_grad():
                embedding_repr = model(
                    token_encoding.input_ids,
                    attention_mask=token_encoding.attention_mask,
                )
        except RuntimeError:
            logger.warning(f"RuntimeError during embedding — batch residues: {n_res_batch}, seq_len: {seq_len}, ids: {pdb_ids}")
            fail_ids.extend(pdb_ids)
            if progress_callback is not None:
                progress_callback(_batch_n)
            continue

        try:
            # Mask out the special token appended at the end of each sequence,
            # accounting for the <AA2fold> prefix token at position 0.
            for idx, s_len in enumerate(seq_lens):
                token_encoding.attention_mask[idx, s_len + 1] = 0

            residue_embedding = embedding_repr.last_hidden_state.detach()
            residue_embedding = residue_embedding * token_encoding.attention_mask.unsqueeze(dim=-1)
            residue_embedding = residue_embedding[:, 1:]  # drop prefix token embedding

            prediction = predictor(residue_embedding)
            probabilities = toCPU(torch.max(F.softmax(prediction, dim=1), dim=1, keepdim=True)[0])
            prediction = toCPU(torch.max(prediction, dim=1, keepdim=True)[1]).astype(np.byte)

            for batch_idx, identifier in enumerate(pdb_ids):
                s_len = seq_lens[batch_idx]

                if save_per_residue_embeddings or save_per_protein_embeddings:
                    try:
                        emb = embedding_repr.last_hidden_state[batch_idx, 1: s_len + 1]
                        if save_per_residue_embeddings:
                            batch_embeddings_per_residue[identifier] = emb.detach().cpu().numpy().squeeze()
                        if save_per_protein_embeddings:
                            batch_embeddings_per_protein[identifier] = emb.mean(dim=0).detach().cpu().numpy().squeeze()
                    except Exception:
                        logger.warning(f"Saving embeddings failed for {identifier}")

                pred = prediction[batch_idx, :, 0:s_len].squeeze()
                # Clean 2dp Python float — the preferred representation.
                # NOTE — bioconda phold v1.2.5 behaviour (numpy ≥2 artefact):
                #   In numpy ≥2, round(np.float32_scalar, 2) returns np.float32 rather
                #   than a Python float.  That np.float32 value formats in an f-string as
                #   its full IEEE-754 float32 repr, e.g. "82.69000244140625" instead of
                #   "82.69".  The two values differ by <3e-6, purely a formatting artefact.
                #   compare_outputs.py accounts for this with a tight abs_tol on mean-prob
                #   CSVs so the test suite passes regardless of numpy version.
                mean_prob = round(100 * float(np.mean(probabilities[batch_idx, :, 0:s_len])), 2)

                if output_probs:
                    all_prob = probabilities[batch_idx, :, 0:s_len]
                    batch_predictions[identifier] = (pred, mean_prob, all_prob)
                else:
                    batch_predictions[identifier] = (pred, mean_prob, None)

                try:
                    len(batch_predictions[identifier][0])
                except Exception:
                    logger.warning(f"{identifier} prediction has length 0")
                    fail_ids.append(identifier)
                    continue

                if s_len != len(batch_predictions[identifier][0]):
                    logger.warning(
                        f"Length mismatch for {identifier}: got {len(batch_predictions[identifier][0])}, expected {s_len}"
                    )

        except IndexError:
            logger.warning(f"IndexError during prediction for {pdb_id} (L={seq_len})")
            fail_ids.extend(pdb_ids)
            if progress_callback is not None:
                progress_callback(_batch_n)
            continue

        # Successful batch — notify the shared progress bar (multi-GPU path).
        if progress_callback is not None:
            progress_callback(_batch_n)

    # Merge batch dicts into final dicts, preserving original key order.
    # Callers build seq_dict from their own ordered source and pass the
    # original key list to write_probs for ordered output.
    predictions.update(batch_predictions)
    if save_per_residue_embeddings:
        embeddings_per_residue.update(batch_embeddings_per_residue)
    if save_per_protein_embeddings:
        embeddings_per_protein.update(batch_embeddings_per_protein)

    return predictions, embeddings_per_residue, embeddings_per_protein, fail_ids


# ─────────────────────────────────────────────────────────────────────────────
# Multi-GPU dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def run_prostt5_inference_multi_gpu(
    seq_dict: List[Tuple[str, str, int]],
    devices: List[str],
    model_dir: Path,
    model_name: str,
    checkpoint_path: Path,
    half_precision: bool,
    threads: int,
    check_fn: Optional[Callable] = None,
    zenodo_fn: Optional[Callable] = None,
    max_residues: int = 100000,
    max_seq_len: int = 30000,
    max_batch: int = 10000,
    output_probs: bool = True,
    save_per_residue_embeddings: bool = False,
    save_per_protein_embeddings: bool = False,
    desc: str = "Predicting 3Di",
) -> Tuple[Dict, Dict, Dict, List]:
    """Run ProstT5 inference across one or more devices.

    When ``len(devices) == 1`` this loads the model on the single device and
    calls :func:`run_prostt5_inference` directly — byte-identical to the
    pre-multi-GPU code path. When ``len(devices) > 1`` the sorted
    ``seq_dict`` is sharded round-robin across the devices, one
    :class:`ThreadPoolExecutor` worker per device runs inference on its
    shard, and the four return values are merged.

    The model is loaded *inside* each worker thread so that ``.to(device)``
    executes on the right CUDA context — never load N copies in the main
    thread and hand them out.

    Args:
        seq_dict: Pre-sorted (descending by length) list of
                  ``(seq_id, sequence, length)`` tuples.
        devices: List of torch device strings from
                 :func:`pholdlib.prostt5.device.parse_gpus`. Must be non-empty.
        model_dir / model_name / checkpoint_path: As for :func:`get_T5_model`
                  and :func:`load_predictor`.
        half_precision: If True, cast both ProstT5 and CNN to fp16 after
                  loading. Forced off on CPU devices.
        threads: torch CPU-thread count (passed through to ``get_T5_model``).
        check_fn / zenodo_fn: Optional tool-specific download callbacks
                  (same contract as :func:`get_T5_model`).
        max_residues / max_seq_len / max_batch / output_probs /
        save_per_residue_embeddings / save_per_protein_embeddings / desc:
                  Passed straight through to :func:`run_prostt5_inference`.

    Returns:
        (predictions, embeddings_per_residue, embeddings_per_protein, fail_ids)
        — merged across all shards. Order in the dicts is not guaranteed;
        callers must re-order using their own ``original_keys`` list.
    """
    if not devices:
        raise ValueError("devices must contain at least one entry")

    # ── single-device fast path: byte-identical to pre-multi-GPU behaviour ──
    if len(devices) == 1:
        dev = devices[0]
        is_cpu = dev == "cpu"
        model, vocab, torch_device = get_T5_model(
            model_dir, model_name,
            cpu=is_cpu,
            threads=threads,
            check_fn=check_fn,
            zenodo_fn=zenodo_fn,
            device=None if is_cpu else dev,
        )
        predictor = load_predictor(checkpoint_path, torch_device)

        if half_precision and not is_cpu:
            model = model.half()
            predictor = predictor.half()

        return run_prostt5_inference(
            seq_dict,
            model, vocab, predictor, torch_device,
            max_residues=max_residues,
            max_seq_len=max_seq_len,
            max_batch=max_batch,
            output_probs=output_probs,
            save_per_residue_embeddings=save_per_residue_embeddings,
            save_per_protein_embeddings=save_per_protein_embeddings,
            desc=desc,
        )

    # ── multi-GPU path ──────────────────────────────────────────────────────
    n_devices = len(devices)
    shards = shard_round_robin(seq_dict, n_devices)
    logger.info(
        f"Multi-GPU 3Di inference across {n_devices} device(s): {devices}. "
        f"Total sequences: {len(seq_dict)}; per-shard sizes: "
        f"{[len(s) for s in shards]} (round-robin after global length-sort)."
    )

    # Single shared progress bar driven by every worker.
    progress_lock = threading.Lock()
    pbar = tqdm(total=len(seq_dict), desc=desc)

    def _on_batch(n: int) -> None:
        with progress_lock:
            pbar.update(n)

    def _worker(device_str: str, shard: List[Tuple[str, str, int]]):
        """Load model on `device_str` and run inference on `shard`."""
        is_cpu = device_str == "cpu"
        model, vocab, torch_device = get_T5_model(
            model_dir, model_name,
            cpu=is_cpu,
            threads=threads,
            check_fn=check_fn,
            zenodo_fn=zenodo_fn,
            device=None if is_cpu else device_str,
        )
        predictor = load_predictor(checkpoint_path, torch_device)

        if half_precision and not is_cpu:
            model = model.half()
            predictor = predictor.half()

        return run_prostt5_inference(
            shard,
            model, vocab, predictor, torch_device,
            max_residues=max_residues,
            max_seq_len=max_seq_len,
            max_batch=max_batch,
            output_probs=output_probs,
            save_per_residue_embeddings=save_per_residue_embeddings,
            save_per_protein_embeddings=save_per_protein_embeddings,
            desc=f"{desc} [{device_str}]",
            progress_callback=_on_batch,
        )

    merged_preds: Dict = {}
    merged_emb_res: Dict = {}
    merged_emb_prot: Dict = {}
    merged_fails: List[str] = []

    try:
        with ThreadPoolExecutor(max_workers=n_devices) as pool:
            futures = {
                pool.submit(_worker, dev, shard): dev
                for dev, shard in zip(devices, shards)
                if shard  # skip empty shards (rare: len(seq_dict) < n_devices)
            }
            for fut in as_completed(futures):
                dev = futures[fut]
                try:
                    preds, emb_res, emb_prot, fails = fut.result()
                except Exception:
                    logger.exception(f"Worker on {dev} failed")
                    raise
                merged_preds.update(preds)
                merged_emb_res.update(emb_res)
                merged_emb_prot.update(emb_prot)
                merged_fails.extend(fails)
    finally:
        pbar.close()

    return merged_preds, merged_emb_res, merged_emb_prot, merged_fails
