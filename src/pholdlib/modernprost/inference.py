#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ModernProst 3Di + 12-state batch inference engine.

Two tasks are supported:

``classification``
    argmax the 3Di (20-class) and 12st (12-class) heads into per-residue class
    indices, plus per-residue max softmax probabilities. Feeds the FASTA
    writers and the combined ``_ss`` Foldseek DB.

``pssm``
    keep the full per-residue softmax distributions — ``(L, 20)`` and
    ``(L, 12)`` float arrays. Feeds the Foldseek profile DBs.

The input is tokenised once with a lookup table (see
:mod:`pholdlib.modernprost.batching`) and then run through one of two engines:

``padded``
    One protein per row, longest first, rows padded to a multiple of 8, and a
    batch bounded by *computed* tokens (rows x padded length <= budget).
    Works on every device and uses the checkpoint's own forward.

``packed``
    Proteins concatenated into fixed rows of ``pack_tokens`` tokens, with
    FlexAttention masks that keep each protein to itself and give the sliding
    layers their real windowed cost; the forward is compiled once. Needs a
    CUDA or ROCm GPU with PyTorch >= 2.5 and Triton (see
    :mod:`pholdlib.modernprost.packed`).

``engine="auto"`` with ``compile="auto"`` picks, for inputs of at least
20M residues (5M for the 1B checkpoints, both doubled on ROCm; see
:func:`packed_min_residues`):
the packed engine on NVIDIA GPUs (fastest there), the padded engine under
``torch.compile`` on AMD/ROCm GPUs (Triton's FlexAttention kernels are slow on
MI250X, so packing does not pay), and the eager padded engine everywhere
else. Smaller inputs use the eager padded engine, because compiling costs
tens of seconds.

On GPUs both engines do the softmax / argmax on the device, copy back only
real (unpadded) tokens once per batch, and overlap the host-side bookkeeping
for one batch with the GPU work of the next.

Sequences longer than ``chunk_len`` are split into near-equal chunks, inferred
independently and stitched back together; there is no ``<AA2fold>`` prefix and
no EOS, so logit position *i* is residue *i*.
"""

import os
import queue as queue_mod
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from pholdlib.databases.modernprost import CLASSIFICATION_TASK, PSSM_TASK, VALID_TASKS
from pholdlib.modernprost.batching import (
    CHUNK_SEP,
    TokenisedItems,
    chunk_sequence,
    lut_from_tokenizer,
    plan_packed_batches,
    plan_padded_batches,
)

# Sequences longer than this are split before inference. 1568 is the value
# phold settled on for ModernProst: comfortably inside the model's 8192-token
# positional range while keeping the global-attention layers' quadratic cost
# bounded on the long tail of phage proteins.
DEFAULT_CHUNK_LEN = 1568

# Default padded-token budget per batch (rows x padded length) for the padded
# engine, and tokens per packed row for the packed engine.
DEFAULT_TOKEN_BUDGET = 65536
DEFAULT_PACK_TOKENS = 65536

# compile="auto" compiles (and engine="auto" may pack) only for inputs of at
# least this many residues: below it the one-off compile (~30-60 s cold)
# costs more than it saves. Measured break-even against the eager padded
# engine on an A100: ~17M residues for the 50M checkpoints (which save
# ~1.7 us/residue) and ~6M for the 1B checkpoints (~7 us/residue). On ROCm
# (MI250X) the compiled padded engine saves less and compiles slower
# (break-even ~26M for 50M), so the thresholds double there.
# Override with PHOLDLIB_PACKED_MIN_RESIDUES.
PACKED_MIN_RESIDUES = 20_000_000
PACKED_MIN_RESIDUES_LARGE_MODEL = 5_000_000
_ROCM_THRESHOLD_FACTOR = 2
_LARGE_MODEL_PARAMS = 200_000_000

# Kept for compatibility: the chunk marker used in temporary item ids.
_CHUNK_SEP = CHUNK_SEP

ENGINES = ("auto", "padded", "packed")

__all__ = [
    "DEFAULT_CHUNK_LEN",
    "DEFAULT_PACK_TOKENS",
    "DEFAULT_TOKEN_BUDGET",
    "ENGINES",
    "ModernProstResult",
    "PACKED_MIN_RESIDUES",
    "chunk_sequence",
    "predictions_from_profiles",
    "run_modernprost_inference",
    "run_modernprost_inference_multi_gpu",
]


@dataclass
class ModernProstResult:
    """Everything one inference run produces.

    Attributes:
        predictions_3di / predictions_12st: ``{seq_id: (pred, mean_prob, all_prob)}``.
            ``pred`` is a byte array of class indices; ``mean_prob`` is a float
            percentage rounded to 2 dp; ``all_prob`` is a float32 array of
            per-residue max probabilities in ``[0, 1]``, or None when
            ``output_probs`` is False. Empty for the ``pssm`` task.
        profiles_3di / profiles_12st: ``{seq_id: ndarray (L, 20 | 12)}`` softmax
            profiles. Empty for the ``classification`` task.
        embeddings_per_residue / embeddings_per_protein: optional embedding
            stores, keyed by seq_id.
        fail_ids: sequence IDs that could not be processed.

    Arrays are views into per-run buffers; copy one before modifying it in
    place if other entries must stay untouched.
    """

    predictions_3di: Dict[str, Tuple] = field(default_factory=dict)
    predictions_12st: Dict[str, Tuple] = field(default_factory=dict)
    profiles_3di: Dict[str, np.ndarray] = field(default_factory=dict)
    profiles_12st: Dict[str, np.ndarray] = field(default_factory=dict)
    embeddings_per_residue: Dict[str, np.ndarray] = field(default_factory=dict)
    embeddings_per_protein: Dict[str, np.ndarray] = field(default_factory=dict)
    fail_ids: List[str] = field(default_factory=list)

    def update(self, other: "ModernProstResult") -> None:
        """Merge another result in place."""
        self.predictions_3di.update(other.predictions_3di)
        self.predictions_12st.update(other.predictions_12st)
        self.profiles_3di.update(other.profiles_3di)
        self.profiles_12st.update(other.profiles_12st)
        self.embeddings_per_residue.update(other.embeddings_per_residue)
        self.embeddings_per_protein.update(other.embeddings_per_protein)
        self.fail_ids.extend(other.fail_ids)


def predictions_from_profiles(
    profiles: Dict[str, np.ndarray],
    output_probs: bool = True,
) -> Dict[str, Tuple]:
    """Collapse softmax profiles into the ``(pred, mean_prob, all_prob)`` shape.

    The ``pssm`` task keeps full distributions and so produces no argmax
    predictions, but downstream consumers still want a representative state
    string and a per-residue confidence — phold, for example, reports a mean
    confidence per CDS and masks low-confidence residues in the amino-acid
    FASTA regardless of which task ran. Taking the argmax and its probability
    here gives exactly the values the ``classification`` task would have
    produced from the same logits, without a second forward pass.
    """
    out: Dict[str, Tuple] = {}
    for seq_id, profile in profiles.items():
        pred = np.argmax(profile, axis=-1).astype(np.byte)
        probs = np.max(profile, axis=-1).astype(np.float32)
        out[seq_id] = _finalise(pred, probs, output_probs)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Public entry points
# ─────────────────────────────────────────────────────────────────────────────

def run_modernprost_inference(
    seq_dict: List[Tuple[str, str, int]],
    model,
    tokenizer,
    device: torch.device,
    task: str = CLASSIFICATION_TASK,
    max_residues: int = DEFAULT_TOKEN_BUDGET,
    max_seq_len: int = 30000,
    max_batch: Optional[int] = None,
    chunk_len: int = DEFAULT_CHUNK_LEN,
    output_probs: bool = True,
    save_per_residue_embeddings: bool = False,
    save_per_protein_embeddings: bool = False,
    pad_multiple: int = 8,
    desc: str = "Predicting 3Di + 12st",
    progress_callback: Optional[Callable[[int], None]] = None,
    engine: str = "auto",
    compile: str = "auto",
    pack_tokens: int = DEFAULT_PACK_TOKENS,
) -> ModernProstResult:
    """
    Run ModernProst on a list of protein sequences with one loaded model.

    Args:
        seq_dict: ``(seq_id, sequence, length)`` tuples, in any order.
            Non-standard amino acids are handled by the tokenizer table, so
            callers need not pre-clean.
        model / tokenizer: from :func:`get_modernprost_model`.
        device: torch device the model lives on.
        task: ``"classification"`` or ``"pssm"``.
        max_residues: padded engine only: computed-token budget per batch
            (rows x padded length). Padding is counted, so this bounds memory.
        max_seq_len: unused; kept for call-site compatibility (chunking keeps
            every item at or below ``chunk_len``).
        max_batch: padded engine only: optional cap on rows per batch.
        chunk_len: sequences longer than this are split before inference.
        output_probs: when False, per-residue probability arrays are dropped
            from the result (the mean is still computed). Ignored for ``pssm``.
        save_per_residue_embeddings / save_per_protein_embeddings: collect
            ``last_hidden_state`` per residue and/or its per-protein mean.
        pad_multiple: padded engine: round each batch's length up to this.
        desc: progress bar description.
        progress_callback: optional ``progress_callback(n_items)`` invoked
            after each batch; when given, the internal bar is disabled.
        engine: ``"padded"``, ``"packed"`` or ``"auto"`` (see the module
            docstring for what auto picks on each kind of device).
        compile: ``"auto"`` (compile on a GPU for inputs of at least
            :func:`packed_min_residues` residues), ``"on"`` or ``"off"``. Experts can
            name the mode: ``"full"`` / ``"flex"`` (packed), ``"dynamic"``
            (padded under ``torch.compile(dynamic=True)``) or ``"none"``.
        pack_tokens: packed engine: tokens per packed row.

    Returns:
        A :class:`ModernProstResult`.
    """
    task = _check_task(task)
    items = TokenisedItems.from_sequences(seq_dict, lut_from_tokenizer(tokenizer), chunk_len)
    _log_chunking(items, chunk_len)
    device = torch.device(device)
    chosen, mode = _choose_engine(engine, compile, device, model, items.n_tokens)
    pack_tokens = _fit_pack_tokens(items, chosen, pack_tokens)
    store = _OutputStore(items, task, save_per_residue_embeddings or save_per_protein_embeddings,
                         hidden_size=int(model.config.hidden_size))
    batches = _plan(items, chosen, max_residues, pad_multiple, max_batch, pack_tokens)
    runner = _make_runner(chosen, model, device, mode, pack_tokens, pad_multiple,
                          n_encoders=1, want_hidden=store.want_hidden)

    pbar = None
    if progress_callback is None:
        pbar = tqdm(total=items.n_items, desc=desc)
        progress_callback = pbar.update
    _report_empty(items, progress_callback)
    try:
        _drive(runner, items, iter(batches), store, progress_callback)
    finally:
        if pbar is not None:
            pbar.close()
    return store.finalise(output_probs, save_per_residue_embeddings, save_per_protein_embeddings)


def run_modernprost_inference_multi_gpu(
    seq_dict: List[Tuple[str, str, int]],
    devices: List[str],
    model_dir: Path,
    model_name: str,
    half_precision: bool,
    threads: int,
    task: str = CLASSIFICATION_TASK,
    check_fn: Optional[Callable] = None,
    zenodo_fn: Optional[Callable] = None,
    max_residues: int = DEFAULT_TOKEN_BUDGET,
    max_seq_len: int = 30000,
    max_batch: Optional[int] = None,
    chunk_len: int = DEFAULT_CHUNK_LEN,
    output_probs: bool = True,
    save_per_residue_embeddings: bool = False,
    save_per_protein_embeddings: bool = False,
    pad_multiple: int = 8,
    desc: str = "Predicting 3Di + 12st",
    engine: str = "auto",
    compile: str = "auto",
    pack_tokens: int = DEFAULT_PACK_TOKENS,
) -> ModernProstResult:
    """Load the model on each device and run inference across all of them.

    The input is tokenised and planned once; one worker per device then pulls
    batches from a shared queue until it is empty, so a slower or busier GPU
    simply takes fewer batches. Outputs go straight into one shared set of
    buffers (batches write disjoint slices).

    Workers are threads when nothing is compiled: model copies are loaded
    **sequentially on the main thread** first (HuggingFace's
    ``from_pretrained`` is not thread-safe; concurrent loads can leave a model
    on the ``meta`` device). With ``torch.compile`` on several GPUs, workers
    are processes instead: compilation is serialised inside one process
    (~20-25 s per GPU on an A100, so four GPUs would spend ~90 s compiling
    one after another), whereas separate processes compile in parallel. This
    process drives the first device; spawned workers drive the others and
    share the token and output buffers through shared memory.

    Args:
        devices: device strings from :func:`pholdlib.prostt5.device.parse_gpus`.
        model_dir / model_name / half_precision / threads / check_fn / zenodo_fn:
            passed to :func:`get_modernprost_model`.
        Everything else as in :func:`run_modernprost_inference`.

    Returns:
        A :class:`ModernProstResult`.

    Raises:
        ValueError: if ``devices`` is empty.
        RuntimeError: if any input sequence is neither predicted nor recorded
            as a failure — that would otherwise surface downstream as a
            silently masked CDS.
    """
    if not devices:
        raise ValueError("devices must contain at least one entry")

    from pholdlib.modernprost.model import count_parameters, get_modernprost_model

    task = _check_task(task)
    t_start = time.perf_counter()

    def load(dev: str, first: bool):
        is_cpu = dev == "cpu"
        # only the first load checks / downloads the checkpoint; the rest
        # read the cache it left behind
        return get_modernprost_model(
            model_dir,
            model_name,
            cpu=is_cpu,
            threads=threads,
            check_fn=check_fn if first else None,
            zenodo_fn=zenodo_fn if first else None,
            device=None if is_cpu else dev,
            half_precision=half_precision,
        )

    model0, tokenizer, device0 = load(devices[0], True)
    logger.info(f"ModernProst total parameters: {count_parameters(model0):,}")

    items = TokenisedItems.from_sequences(seq_dict, lut_from_tokenizer(tokenizer), chunk_len)
    _log_chunking(items, chunk_len)
    chosen, mode = _choose_engine(engine, compile, device0, model0, items.n_tokens)
    pack_tokens = _fit_pack_tokens(items, chosen, pack_tokens)
    use_processes = (len(devices) > 1 and mode != "none"
                     and all(str(d).startswith("cuda") for d in devices))
    want_hidden = save_per_residue_embeddings or save_per_protein_embeddings
    store = _OutputStore(items, task, want_hidden, hidden_size=int(model0.config.hidden_size),
                         shared=use_processes)
    batches = _plan(items, chosen, max_residues, pad_multiple, max_batch, pack_tokens)
    if len(devices) > 1:
        logger.info(
            f"Multi-GPU ModernProst inference across {len(devices)} device(s): {devices}. "
            f"{len(seq_dict)} sequences in {len(batches)} {chosen} batches from a shared queue "
            f"({'one process' if use_processes else 'one thread'} per device)."
        )

    pbar = tqdm(total=items.n_items, desc=desc)
    lock = threading.Lock()

    def on_batch(n: int) -> None:
        with lock:
            pbar.update(n)

    _report_empty(items, on_batch)
    # warm-up and workers change the CUDA current device; give the caller its
    # own back afterwards
    restore = torch.cuda.current_device() if torch.cuda.is_available() else None
    try:
        if use_processes:
            worker_args = dict(model_dir=model_dir, model_name=model_name,
                               half_precision=half_precision,
                               threads=max(1, threads // len(devices)), task=task, engine=chosen,
                               mode=mode, pack_tokens=pack_tokens, pad_multiple=pad_multiple,
                               want_hidden=want_hidden)
            t_run = _run_with_processes(model0, device0, devices, items, store, batches,
                                        worker_args, on_batch)
        else:
            loaded = [(model0, device0)] + [load(dev, False)[0::2] for dev in devices[1:]]
            logger.info(f"Loaded {len(loaded)} ModernProst model cop{'y' if len(loaded) == 1 else 'ies'} "
                        f"in {time.perf_counter() - t_start:.1f} s")
            t_run = _run_with_threads(loaded, items, store, batches, chosen, mode, pack_tokens,
                                      pad_multiple, on_batch)
    finally:
        pbar.close()
        if restore is not None:
            torch.cuda.set_device(restore)
    t_done = time.perf_counter()
    result = store.finalise(output_probs, save_per_residue_embeddings, save_per_protein_embeddings)
    logger.info(
        f"ModernProst: {items.n_tokens:,} residues on {len(devices)} device(s) in "
        f"{time.perf_counter() - t_start:.1f} s (main inference phase {t_done - t_run:.1f} s, "
        f"result assembly {time.perf_counter() - t_done:.1f} s)"
    )

    # Every input seq_id must appear either as a prediction/profile or in
    # fail_ids. A silent drop would make downstream code mask the CDS as
    # "XXXX…" with no explanation of why.
    accounted = set(result.predictions_3di) | set(result.profiles_3di) | set(result.fail_ids)
    dropped = {sid for sid, _, _ in seq_dict} - accounted
    if dropped:
        raise RuntimeError(
            f"ModernProst inference dropped {len(dropped)} sequence(s) without recording "
            f"them in fail_ids — likely a bug in batch error handling. "
            f"First few: {sorted(dropped)[:10]}"
        )
    return result


def _run_with_threads(loaded, items, store, batches, engine, mode, pack_tokens, pad_multiple,
                      on_batch) -> float:
    """Drive every loaded (model, device) from one shared queue, one thread
    per device. Returns the time the main phase started."""
    queue = _BatchQueue(batches)
    runners = []
    for model, dev in loaded:
        _set_device(dev)
        runner = _make_runner(engine, model, dev, mode, pack_tokens, pad_multiple,
                              n_encoders=len(loaded), want_hidden=store.want_hidden)
        # A compiled engine compiles on its first batches: do that here, one
        # device at a time, on real batches. The longest and the shortest
        # batch bracket every shape a dynamic-shape (padded) graph will see.
        if runner.compiled:
            t_warm = time.perf_counter()
            _drive(runner, items, queue.take_extremes(), store, on_batch)
            logger.info(f"Compiled ModernProst on {dev} in {time.perf_counter() - t_warm:.1f} s "
                        f"(first two batches included)")
        runners.append(runner)
    t_run = time.perf_counter()

    if len(runners) == 1:
        _drive(runners[0], items, queue.iterate(), store, on_batch)
    else:
        def work(runner):
            _set_device(runner.device)
            _drive(runner, items, queue.iterate(), store, on_batch)

        with ThreadPoolExecutor(max_workers=len(runners)) as pool:
            for fut in [pool.submit(work, r) for r in runners]:
                fut.result()
    return t_run


def _run_with_processes(model0, device0, devices, items: TokenisedItems, store: "_OutputStore",
                        batches: List[np.ndarray], worker_args: dict, on_batch) -> float:
    """Drive ``devices[0]`` here and every other device from a spawned worker
    process, all pulling from one shared batch counter. Returns the time the
    main phase started."""
    import torch.multiprocessing as tmp

    ctx = tmp.get_context("spawn")
    counter = ctx.Value("q", 0, lock=False)
    counter_lock = ctx.Lock()
    msgs = ctx.Queue()
    shared_items = {
        "tokens": torch.from_numpy(items.tokens).share_memory_(),
        "offsets": torch.from_numpy(items.offsets).share_memory_(),
        "lengths": torch.from_numpy(items.lengths).share_memory_(),
        "item_protein": torch.from_numpy(items.item_protein).share_memory_(),
    }
    procs = []
    for rank, dev in enumerate(devices[1:], start=1):
        p = ctx.Process(
            target=_process_worker,
            args=(rank, str(dev), worker_args, shared_items, store.tensors, batches,
                  counter, counter_lock, msgs),
            daemon=True,
        )
        p.start()
        procs.append(p)

    finished: Dict[int, tuple] = {}

    def listen():
        while len(finished) < len(procs):
            try:
                msg = msgs.get(timeout=1.0)
            except queue_mod.Empty:
                if not any(p.is_alive() for p in procs):
                    break
                continue
            kind = msg[0]
            if kind == "progress":
                on_batch(msg[1])
            elif kind == "log":
                logger.log(msg[1], msg[2])
            else:  # "done" / "error"
                finished[msg[1]] = msg

    listener = threading.Thread(target=listen, daemon=True)
    listener.start()

    t_run = time.perf_counter()
    _set_device(device0)
    runner = _make_runner(worker_args["engine"], model0, device0, worker_args["mode"],
                          worker_args["pack_tokens"], worker_args["pad_multiple"], n_encoders=1,
                          want_hidden=worker_args["want_hidden"])
    _drive(runner, items, _SharedBatchQueue(batches, counter, counter_lock).iterate(), store, on_batch)

    for p in procs:
        p.join()
    listener.join()
    for rank, p in enumerate(procs, start=1):
        msg = finished.get(rank)
        if msg is None:
            logger.warning(f"ModernProst worker for {devices[rank]} exited with code {p.exitcode} "
                           "without reporting; its unfinished batches are rerun here.")
        elif msg[0] == "error":
            logger.warning(f"ModernProst worker for {devices[rank]} failed; its unfinished batches "
                           f"are rerun here:\n{msg[2]}")
        else:
            for b in msg[2]:
                store.fail(b)

    # a worker that died mid-batch leaves items neither written nor failed
    leftover = store.unfinished()
    if len(leftover):
        logger.warning(f"Rerunning {len(leftover)} unfinished item(s) on {device0}.")
        if worker_args["engine"] == "packed":
            sub = plan_packed_batches(items.lengths[leftover], worker_args["pack_tokens"])
        else:
            sub = plan_padded_batches(items.lengths[leftover], DEFAULT_TOKEN_BUDGET,
                                      worker_args["pad_multiple"])
        _drive(runner, items, iter([leftover[b] for b in sub]), store, on_batch)
    return t_run


class _SharedBatchQueue:
    """The plan plus a 'next batch' counter shared between processes."""

    def __init__(self, batches: List[np.ndarray], counter, lock):
        self._batches = batches
        self._counter = counter
        self._lock = lock

    def iterate(self) -> Iterator[np.ndarray]:
        while True:
            with self._lock:
                k = self._counter.value
                self._counter.value = k + 1
            if k >= len(self._batches):
                return
            yield self._batches[k]


def _process_worker(rank: int, device: str, args: dict, shared_items: dict, shared_out: dict,
                    batches: List[np.ndarray], counter, counter_lock, msgs) -> None:
    """Spawned worker: load the model on ``device``, then pull batches from the
    shared counter and write outputs into the shared buffers. Warnings are
    forwarded to the parent's log; nothing large travels through ``msgs``."""
    logger.remove()
    logger.add(lambda m: msgs.put(("log", m.record["level"].name, f"[{device}] {m.record['message']}")),
               level="WARNING")
    try:
        from pholdlib.modernprost.model import get_modernprost_model

        torch.cuda.set_device(device)
        model, _, dev = get_modernprost_model(
            Path(args["model_dir"]), args["model_name"], cpu=False, threads=args["threads"],
            device=device, half_precision=args["half_precision"],
        )
        n = {k: v.numpy() for k, v in shared_items.items()}
        items = TokenisedItems(seq_ids=[], item_protein=n["item_protein"], lengths=n["lengths"],
                               offsets=n["offsets"], tokens=n["tokens"],
                               protein_offsets=np.zeros(1, dtype=np.int64), n_chunked=0)
        store = _OutputStore.attach(items, args["task"], args["want_hidden"], shared_out)
        runner = _make_runner(args["engine"], model, dev, args["mode"], args["pack_tokens"],
                              args["pad_multiple"], n_encoders=1, want_hidden=args["want_hidden"])
        _drive(runner, items, _SharedBatchQueue(batches, counter, counter_lock).iterate(), store,
               lambda k: msgs.put(("progress", k)))
        msgs.put(("done", rank, store.failed_items))
    except BaseException:  # noqa: BLE001 - report everything to the parent
        msgs.put(("error", rank, traceback.format_exc()))


# ─────────────────────────────────────────────────────────────────────────────
# Engine selection and planning
# ─────────────────────────────────────────────────────────────────────────────

def _check_task(task: str) -> str:
    task = str(task).lower()
    if task not in VALID_TASKS:
        raise ValueError(f"task must be one of {VALID_TASKS}, got {task!r}")
    return task


def _log_chunking(items: TokenisedItems, chunk_len: int) -> None:
    if items.n_chunked:
        logger.info(
            f"{items.n_chunked} sequence(s) longer than {chunk_len} residues "
            f"were split into chunks for inference and reassembled afterwards."
        )


def packed_min_residues(model=None) -> int:
    """Residue count from which ``compile="auto"`` compiles (and
    ``engine="auto"`` may pack): ``PHOLDLIB_PACKED_MIN_RESIDUES`` if set,
    else 20M for the ~50M-parameter checkpoints and 5M for the ~1B ones,
    doubled on ROCm."""
    env = os.environ.get("PHOLDLIB_PACKED_MIN_RESIDUES")
    if env:
        try:
            return int(env)
        except ValueError:
            logger.warning(f"Ignoring non-integer PHOLDLIB_PACKED_MIN_RESIDUES={env!r}")
    try:
        n_params = sum(p.numel() for p in model.parameters())
    except (AttributeError, TypeError):
        n_params = 0
    base = PACKED_MIN_RESIDUES_LARGE_MODEL if n_params >= _LARGE_MODEL_PARAMS else PACKED_MIN_RESIDUES
    return base * _ROCM_THRESHOLD_FACTOR if is_rocm() else base


COMPILE_MODES = ("auto", "on", "off", "full", "flex", "dynamic", "none")


def is_rocm() -> bool:
    """True on a ROCm build of PyTorch (never infer this from compute
    capability: ROCm reports (9, 0) on MI250X, the same as an H100)."""
    return getattr(torch.version, "hip", None) is not None


def _choose_engine(engine: str, compile: str, device: torch.device, model,
                   n_tokens: int) -> Tuple[str, str]:
    """Resolve the user's ``engine`` / ``compile`` request to a concrete
    ``(engine, compile_mode)``.

    compile modes: ``"full"`` / ``"flex"`` (packed engine), ``"dynamic"``
    (padded engine under ``torch.compile(dynamic=True)``) and ``"none"``.
    ``"on"`` means full for packed and dynamic for padded; ``"off"`` = none;
    ``"auto"`` compiles only on a GPU and only for inputs of at least
    :func:`packed_min_residues` residues (which depends on model size).
    """
    engine = str(engine).lower()
    compile = str(compile).lower() if compile is not None else "auto"
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES}, got {engine!r}")
    if compile not in COMPILE_MODES:
        raise ValueError(f"compile must be one of {COMPILE_MODES}, got {compile!r}")

    on_gpu = torch.device(device).type == "cuda"
    threshold = packed_min_residues(model)
    large = n_tokens >= threshold
    want_compile = compile in ("on", "full", "flex", "dynamic") or (compile == "auto" and on_gpu and large)

    from pholdlib.modernprost.packed import packed_engine_status

    packed_ok, why = packed_engine_status(device, model)
    if engine == "auto":
        # packed only where its Triton attention is fast (NVIDIA); on ROCm the
        # padded engine under torch.compile is faster (measured on MI250X)
        engine = "packed" if (want_compile and packed_ok and not is_rocm()) else "padded"
    if engine == "packed" and not packed_ok:
        logger.warning(f"Packed ModernProst engine unavailable ({why}); using the padded engine.")
        engine = "padded"

    mode = "none"
    if engine == "packed":
        mode = compile if compile in ("full", "flex", "none") else ("none" if compile == "off" else "full")
        if mode == "none" and on_gpu:
            # uncompiled FlexAttention falls back to a dense reference
            # implementation: a (pack_tokens x pack_tokens) score matrix
            logger.warning("The packed ModernProst engine needs torch.compile on a GPU; "
                           "using the eager padded engine.")
            engine, want_compile = "padded", False
    if engine == "padded":
        # "full"/"flex" name packed modes; the padded equivalent is dynamic
        mode = "dynamic" if want_compile else "none"
    if mode != "none" and compile == "auto":
        logger.info(
            f"{n_tokens:,} residues >= {threshold:,}: using the {engine} ModernProst engine "
            f"with torch.compile ({mode}); the first batch compiles, then it runs faster."
        )
    return engine, mode


def _fit_pack_tokens(items: TokenisedItems, engine: str, pack_tokens: int) -> int:
    """Packed rows must hold the longest item; grow them (to a multiple of
    the FlexAttention tile) when ``chunk_len`` exceeds ``pack_tokens``."""
    if engine != "packed" or not items.n_items:
        return pack_tokens
    from pholdlib.modernprost.packed import DEFAULT_BLOCK

    longest = int(items.lengths.max())
    if longest <= pack_tokens:
        return pack_tokens
    fitted = -(-longest // DEFAULT_BLOCK) * DEFAULT_BLOCK
    logger.info(f"Longest item is {longest} residues; packing {fitted} tokens per row instead of {pack_tokens}.")
    return fitted


def _plan(items: TokenisedItems, engine: str, max_residues: int, pad_multiple: int,
          max_batch: Optional[int], pack_tokens: int) -> List[np.ndarray]:
    """Batches of item indices; zero-length items are never sent to the model."""
    lengths = items.lengths
    nonempty = np.flatnonzero(lengths > 0)
    if engine == "packed":
        sub = plan_packed_batches(lengths[nonempty], pack_tokens)
    else:
        budget = max_residues if max_residues and max_residues > 0 else DEFAULT_TOKEN_BUDGET
        sub = plan_padded_batches(lengths[nonempty], budget, pad_multiple, max_batch)
    return [nonempty[b] for b in sub]


class _BatchQueue:
    """Thread-safe queue over a fixed plan, consumed from the front (plus a
    warm-up helper that takes one batch from each end)."""

    def __init__(self, batches: List[np.ndarray]):
        self._batches = batches
        self._next = 0
        self._end = len(batches)
        self._lock = threading.Lock()

    def _pop(self, back: bool = False) -> Optional[np.ndarray]:
        with self._lock:
            if self._next >= self._end:
                return None
            if back:
                self._end -= 1
                return self._batches[self._end]
            b = self._batches[self._next]
            self._next += 1
            return b

    def take_extremes(self) -> Iterator[np.ndarray]:
        for back in (False, True):
            b = self._pop(back)
            if b is None:
                return
            yield b

    def iterate(self) -> Iterator[np.ndarray]:
        while True:
            b = self._pop()
            if b is None:
                return
            yield b



def _report_empty(items: TokenisedItems, progress: Callable[[int], None]) -> None:
    """Zero-length items are never batched; count them as done up front."""
    n_empty = int((items.lengths == 0).sum())
    if n_empty:
        progress(n_empty)


def _set_device(device: torch.device) -> None:
    # Each worker thread must target its own GPU: CUDA's "current device" is
    # per thread and defaults to cuda:0, and events/streams created without an
    # explicit device follow it.
    if device.type == "cuda":
        torch.cuda.set_device(device)


# ─────────────────────────────────────────────────────────────────────────────
# Runners: launch one batch, return device tensors for its real tokens
# ─────────────────────────────────────────────────────────────────────────────

class _PaddedRunner:
    """One protein per row, through the checkpoint's own forward."""

    def __init__(self, model, device: torch.device, pad_multiple: int, compile: str,
                 want_hidden: bool):
        self.device = device
        self.pad_multiple = pad_multiple
        self.compiled = compile == "dynamic"
        self.forward = torch.compile(model, dynamic=True) if self.compiled else model
        self.pin = device.type == "cuda"
        self.want_hidden = want_hidden

    def launch(self, items: TokenisedItems, batch: np.ndarray):
        lens = items.lengths[batch]
        B = len(batch)
        Lp = int(lens.max())
        if self.pad_multiple and self.pad_multiple > 1:
            Lp = ((Lp + self.pad_multiple - 1) // self.pad_multiple) * self.pad_multiple
        arr = np.zeros((B, Lp), dtype=np.uint8)
        toks, offs = items.tokens, items.offsets
        for r, idx in enumerate(batch):
            o, L = int(offs[idx]), int(items.lengths[idx])
            arr[r, :L] = toks[o:o + L]
        starts = np.zeros(B, dtype=np.int64)
        if B > 1:
            np.cumsum(lens[:-1], out=starts[1:])
        n_real = int(lens.sum())
        # flat (row * Lp + col) index of every real token, in batch order;
        # gathering with it needs no device sync, unlike boolean-mask indexing
        sel = np.repeat(np.arange(B, dtype=np.int64) * Lp - starts, lens) + np.arange(n_real, dtype=np.int64)

        ids = _to_device(arr, self.device, self.pin).long()
        sel_g = _to_device(sel, self.device, self.pin)
        mask = ids != 0  # <pad> is id 0; every residue, even unknown, is >= 2
        with torch.inference_mode():
            out = self.forward(input_ids=ids, attention_mask=mask)
            l3 = out.logits["3di"].reshape(B * Lp, -1).index_select(0, sel_g)
            l12 = out.logits["12st"].reshape(B * Lp, -1).index_select(0, sel_g)
            hid = None
            if self.want_hidden:
                hid = out.last_hidden_state.reshape(B * Lp, -1).index_select(0, sel_g)
        return l3, l12, hid


class _PackedRunner:
    """Packed rows through :class:`pholdlib.modernprost.packed.PackedEncoder`."""

    def __init__(self, model, device: torch.device, mode: str, pack_tokens: int, n_encoders: int):
        from pholdlib.modernprost.packed import PackedEncoder

        self.encoder = PackedEncoder(model, device, capacity=pack_tokens, compile=mode,
                                     n_encoders=n_encoders)
        self.device = device
        self.compiled = mode in ("full", "flex")

    def launch(self, items: TokenisedItems, batch: np.ndarray):
        l3, l12, hid, n_real = self.encoder.run_batch(items, batch)
        return l3[:n_real], l12[:n_real], hid[:n_real]


class _FallbackRunner:
    """A compiled runner that permanently drops to the eager padded engine if
    a batch fails for any reason other than running out of memory — typically
    a torch.compile / Triton failure on an unsupported stack (for example a
    conda Triton that cannot find ``ptxas`` because the env is not
    activated). Without this, every batch would fail and every protein would
    land in ``fail_ids``."""

    def __init__(self, primary, make_fallback: Callable[[], "_PaddedRunner"]):
        self.primary = primary
        self.make_fallback = make_fallback
        self.device = primary.device
        self.compiled = True
        self._fell_back = False

    def launch(self, items: TokenisedItems, batch: np.ndarray):
        if self._fell_back:
            return self.primary.launch(items, batch)
        try:
            return self.primary.launch(items, batch)
        except Exception as exc:  # noqa: BLE001
            if _is_oom(exc):
                raise
            logger.warning(
                f"Compiled ModernProst engine failed on {self.device} ({type(exc).__name__}: "
                f"{str(exc)[:300]}); switching this device to the eager padded engine."
            )
            self.primary = self.make_fallback()
            self.compiled = False
            self._fell_back = True
            return self.primary.launch(items, batch)


def _make_runner(engine: str, model, device: torch.device, mode: str, pack_tokens: int,
                 pad_multiple: int, n_encoders: int, want_hidden: bool):
    """Runner for the resolved ``(engine, mode)``. Compiled runners are
    wrapped so a compile failure on the first batch drops that device to the
    eager padded engine instead of failing every batch."""
    def eager_padded():
        return _PaddedRunner(model, device, pad_multiple, "none", want_hidden)

    if engine == "packed":
        try:
            primary = _PackedRunner(model, device, mode, pack_tokens, n_encoders)
        except Exception as exc:  # noqa: BLE001 - any setup failure falls back
            logger.warning(f"Packed ModernProst engine failed to start ({exc}); using the padded engine.")
            return eager_padded()
    else:
        primary = _PaddedRunner(model, device, pad_multiple, mode, want_hidden)
    if primary.compiled:
        return _FallbackRunner(primary, eager_padded)
    return primary


def _to_device(a: np.ndarray, device: torch.device, pin: bool) -> torch.Tensor:
    t = torch.from_numpy(a)
    if pin:
        return t.pin_memory().to(device, non_blocking=True)
    return t.to(device)


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.OutOfMemoryError) if hasattr(torch, "OutOfMemoryError") \
        else "out of memory" in str(exc).lower()


# ─────────────────────────────────────────────────────────────────────────────
# Driving batches and collecting outputs
# ─────────────────────────────────────────────────────────────────────────────

class _OutputStore:
    """Per-token output buffers for every item, laid out like ``items.tokens``.

    A protein's chunks are consecutive items, so every protein's outputs are
    one contiguous slice ``[protein_offsets[p], protein_offsets[p + 1])`` and
    stitching chunks needs no extra work. Batches only ever write disjoint
    slices, so several worker threads can fill the same store.
    """

    def __init__(self, items: TokenisedItems, task: str, want_hidden: bool, hidden_size: int,
                 shared: bool = False):
        n = items.n_tokens
        if task == PSSM_TASK:
            spec = {"P3": ((n, 20), torch.float32), "P12": ((n, 12), torch.float32)}
        else:
            spec = {"p3": ((n,), torch.int8), "p12": ((n,), torch.int8),
                    "q3": ((n,), torch.float32), "q12": ((n,), torch.float32)}
        if want_hidden:
            spec["H"] = ((n, hidden_size), torch.float32)
        # per-item completion flag (zero-length items never reach the model)
        spec["done"] = ((items.n_items,), torch.uint8)
        # torch tensors so they can live in shared memory for worker processes
        self.tensors = {k: torch.zeros(shape, dtype=dt) for k, (shape, dt) in spec.items()}
        self.tensors["done"][torch.from_numpy(items.lengths == 0)] = 1
        if shared:
            for t in self.tensors.values():
                t.share_memory_()
        self._init(items, task, want_hidden)

    def _init(self, items: TokenisedItems, task: str, want_hidden: bool) -> None:
        self.items = items
        self.task = task
        self.want_hidden = want_hidden
        self.arrays = {k: t.numpy() for k, t in self.tensors.items() if k != "done"}
        self.done = self.tensors["done"].numpy()
        self.failed_items: List[np.ndarray] = []
        self._fail_lock = threading.Lock()

    @classmethod
    def attach(cls, items: TokenisedItems, task: str, want_hidden: bool,
               tensors: Dict[str, torch.Tensor]) -> "_OutputStore":
        """A store over another process's shared buffers."""
        store = cls.__new__(cls)
        store.tensors = tensors
        store._init(items, task, want_hidden)
        return store

    def device_outputs(self, l3, l12, hid) -> Dict[str, torch.Tensor]:
        """Reduce logits on the device to exactly what is copied back."""
        if self.task == PSSM_TASK:
            out = {"P3": torch.softmax(l3.float(), dim=-1), "P12": torch.softmax(l12.float(), dim=-1)}
        else:
            # .float() before softmax: numpy has no bfloat16, and fp16 means
            # would accumulate visible error over a long protein.
            q3, p3 = torch.softmax(l3.float(), dim=-1).max(dim=-1)
            q12, p12 = torch.softmax(l12.float(), dim=-1).max(dim=-1)
            out = {"p3": p3.to(torch.int8), "p12": p12.to(torch.int8), "q3": q3, "q12": q12}
        if self.want_hidden:
            out["H"] = hid.float()
        return out

    def write(self, batch: np.ndarray, host: Dict[str, np.ndarray]) -> None:
        offs, lens = self.items.offsets, self.items.lengths
        lo = 0
        for idx in batch:
            o, L = int(offs[idx]), int(lens[idx])
            for k, arr in self.arrays.items():
                arr[o:o + L] = host[k][lo:lo + L]
            lo += L
        self.done[batch] = 1

    def unfinished(self) -> np.ndarray:
        """Items neither written nor recorded as failed."""
        failed = np.zeros(len(self.done), dtype=bool)
        for b in self.failed_items:
            failed[b] = True
        return np.flatnonzero((self.done == 0) & ~failed)

    def fail(self, batch: np.ndarray) -> None:
        with self._fail_lock:
            self.failed_items.append(np.asarray(batch))

    def finalise(self, output_probs: bool, per_residue: bool, per_protein: bool) -> ModernProstResult:
        items = self.items
        result = ModernProstResult()
        failed = np.zeros(len(items.seq_ids), dtype=bool)
        for b in self.failed_items:
            failed[items.item_protein[b]] = True
        po = items.protein_offsets
        a = self.arrays
        if self.task != PSSM_TASK:
            # per-protein mean confidence in one vectorised pass
            means3 = _segment_means(a["q3"], po)
            means12 = _segment_means(a["q12"], po)
        if per_protein:
            emb_means = _segment_means(a["H"], po)
        for p, seq_id in enumerate(items.seq_ids):
            if failed[p]:
                result.fail_ids.append(seq_id)
                continue
            s, e = int(po[p]), int(po[p + 1])
            if self.task == PSSM_TASK:
                result.profiles_3di[seq_id] = a["P3"][s:e]
                result.profiles_12st[seq_id] = a["P12"][s:e]
            else:
                result.predictions_3di[seq_id] = (
                    a["p3"][s:e], round(100 * float(means3[p]), 2), a["q3"][s:e] if output_probs else None)
                result.predictions_12st[seq_id] = (
                    a["p12"][s:e], round(100 * float(means12[p]), 2), a["q12"][s:e] if output_probs else None)
            if per_residue:
                result.embeddings_per_residue[seq_id] = a["H"][s:e]
            if per_protein:
                result.embeddings_per_protein[seq_id] = emb_means[p].astype(np.float32)
        return result


def _segment_means(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Mean of ``values[offsets[i]:offsets[i+1]]`` along axis 0 for every i.

    Accumulates in float64; empty segments give 0.
    """
    n = len(offsets) - 1
    out_shape = (n,) + values.shape[1:]
    sums = np.zeros(out_shape, dtype=np.float64)
    counts = np.diff(offsets)
    nonempty = counts > 0
    if values.shape[0] and nonempty.any():
        sums[nonempty] = np.add.reduceat(values, offsets[:-1][nonempty], axis=0, dtype=np.float64)
    return sums / np.maximum(counts, 1).reshape((-1,) + (1,) * (values.ndim - 1))


class _HostCopy:
    """Device -> host copy of one batch's outputs, asynchronous on CUDA."""

    def __init__(self, gpu: Dict[str, torch.Tensor]):
        dev = next(iter(gpu.values())).device
        self.event = None
        if dev.type == "cuda":
            self.host = {}
            for k, v in gpu.items():
                h = torch.empty(v.shape, dtype=v.dtype, pin_memory=True)
                h.copy_(v, non_blocking=True)
                self.host[k] = h
            # Record on the tensors' own device stream; a bare record() would
            # use the calling thread's current device, which in a worker
            # thread driving another GPU leaves the copies unguarded.
            self.event = torch.cuda.Event()
            self.event.record(torch.cuda.current_stream(dev))
        else:
            self.host = {k: v.cpu() for k, v in gpu.items()}

    def wait(self) -> Dict[str, np.ndarray]:
        if self.event is not None:
            self.event.synchronize()
        return {k: v.numpy() for k, v in self.host.items()}


def _drive(runner, items: TokenisedItems, batches: Iterable[np.ndarray], store: _OutputStore,
           progress: Callable[[int], None]) -> None:
    """Run batches through ``runner`` and write their outputs into ``store``.

    The host-side copy and bookkeeping of batch *i* happen while the GPU runs
    batch *i + 1*. A failing batch is recorded and skipped (its proteins end
    up in ``fail_ids``), as the previous engine did.
    """
    pending: Optional[Tuple[np.ndarray, _HostCopy]] = None

    def flush(entry):
        batch, copy = entry
        try:
            host = copy.wait()
        except RuntimeError as exc:
            _record_failure(store, items, batch, exc)
        else:
            store.write(batch, host)
        progress(len(batch))

    for batch in batches:
        try:
            l3, l12, hid = runner.launch(items, batch)
            copy = _HostCopy(store.device_outputs(l3, l12, hid))
        except RuntimeError as exc:
            _record_failure(store, items, batch, exc)
            progress(len(batch))
            continue
        if pending is not None:
            flush(pending)
        pending = (batch, copy)
    if pending is not None:
        flush(pending)


def _record_failure(store: _OutputStore, items: TokenisedItems, batch: np.ndarray, exc: Exception) -> None:
    ids = sorted({items.seq_ids[p] for p in items.item_protein[batch]}) if items.seq_ids else []
    logger.warning(
        f"RuntimeError during ModernProst inference — batch of {len(batch)} items, "
        f"{int(items.lengths[batch].sum())} residues"
        + (f"; first ids: {ids[:5]}" if ids else "") + f" ({exc})"
    )
    store.fail(batch)
    if _is_oom(exc) and torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers kept for callers and tests
# ─────────────────────────────────────────────────────────────────────────────

def _finalise(pred: np.ndarray, probs: np.ndarray, output_probs: bool) -> Tuple:
    """Build the ``(pred, mean_prob, all_prob)`` tuple for one sequence."""
    mean_prob = round(100 * float(np.mean(probs)), 2) if probs.size else 0.0
    return (pred, mean_prob, probs if output_probs else None)


def _split_chunk_id(item_id: str) -> Tuple[str, Optional[int]]:
    """Split ``"<id>__chunk<N>"`` into ``("<id>", N)``; plain ids give ``(id, None)``.

    Uses rsplit so a CDS id that itself contains ``__chunk`` still resolves to
    the marker this function appended, not one embedded in the user's id.
    """
    if _CHUNK_SEP not in item_id:
        return item_id, None
    base, _, tag = item_id.rpartition(_CHUNK_SEP)
    try:
        return base, int(tag)
    except ValueError:
        return item_id, None
