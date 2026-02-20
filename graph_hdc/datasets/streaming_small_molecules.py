"""
Small Molecule Streaming Dataset for FlowEdgeDecoder.

Generates training data by sampling from a pre-defined pool of small molecules
(loaded from a CSV file), encoding with HyperNet, and feeding to the model
via a producer-consumer architecture.

The pool is a static CSV at ``data/small_molecules.csv`` with columns
``smiles,source`` built from ZINC (<12 heavy atoms) and QM9 (exactly 9 heavy
atoms).
"""

from __future__ import annotations

import csv
import logging
import multiprocessing as mp
import random
import time
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch, Data
from tqdm.auto import tqdm

from graph_hdc.datasets.streaming_fragments import (
    ZINC_ATOM_TO_IDX,
    mol_to_flow_data,
    mol_to_zinc_data,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Small Molecule Pool
# =============================================================================


class SmallMoleculePool:
    """
    Pool of small molecule SMILES loaded from a CSV file.

    The CSV must have a header row with at least a ``smiles`` column.
    An optional ``source`` column indicates the origin dataset.

    Usage::

        pool = SmallMoleculePool("data/small_molecules.csv")
        smiles = pool.sample()
    """

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        self.smiles_list: List[str] = []
        self._load()

    def _load(self) -> None:
        """Read the CSV and populate the SMILES list."""
        self.smiles_list = []
        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                smiles = row["smiles"].strip()
                if smiles:
                    self.smiles_list.append(smiles)
        logger.info(
            f"Loaded {len(self.smiles_list)} SMILES from {self.csv_path}"
        )

    def sample(self) -> str:
        """Return a uniformly random SMILES from the pool."""
        return random.choice(self.smiles_list)

    @property
    def size(self) -> int:
        return len(self.smiles_list)

    def __len__(self) -> int:
        return len(self.smiles_list)


# =============================================================================
# Worker Process
# =============================================================================


def _small_mol_worker_process(
    worker_id: int,
    smiles_list: List[str],
    output_queue: mp.Queue,
    stop_event: mp.Event,
    hypernet_checkpoint_path: str,
    max_nodes: int = 50,
    log_interval: int = 1000,
    encoding_batch_size: int = 32,
) -> None:
    """
    Worker process that samples small molecules, encodes with HyperNet, and
    pushes serialised data to the output queue.

    This is a standalone function (not a method) to avoid pickling issues with
    ``mp.Process`` on the ``spawn`` start method.

    Args:
        worker_id: Unique identifier for logging.
        smiles_list: Flat list of SMILES strings to sample from.
        output_queue: Shared queue for pushing serialised samples.
        stop_event: Event signalling graceful shutdown.
        hypernet_checkpoint_path: Path to saved HyperNet checkpoint.
        max_nodes: Discard molecules exceeding this heavy-atom count.
        log_interval: Print profiling stats every N generated samples.
        encoding_batch_size: Number of molecules to batch-encode at once.
    """
    import sys
    import time as time_module

    def log_msg(msg: str) -> None:
        print(f"[SmallMolWorker {worker_id}] {msg}", file=sys.stderr, flush=True)

    log_msg("Starting worker process...")

    # -- Stage 1: core imports -----------------------------------------------
    try:
        log_msg("[init 1/5] Importing rdkit...")
        from rdkit import Chem, RDLogger as _RDLogger
        _RDLogger.DisableLog('rdApp.*')
        log_msg("[init 1/5] rdkit OK")
    except Exception as e:
        log_msg(f"FATAL [init 1/5] rdkit import failed: {e}")
        import traceback; traceback.print_exc()
        return

    # -- Stage 2: graph_hdc imports ------------------------------------------
    try:
        log_msg("[init 2/5] Importing graph_hdc.hypernet...")
        from graph_hdc.hypernet import load_hypernet
        log_msg("[init 2/5] graph_hdc imports OK")
    except Exception as e:
        log_msg(f"FATAL [init 2/5] graph_hdc import failed: {e}")
        import traceback; traceback.print_exc()
        return

    # -- Stage 3: checkpoint validation --------------------------------------
    try:
        import os
        log_msg(f"[init 3/5] Checking checkpoint: {hypernet_checkpoint_path}")
        if not os.path.exists(hypernet_checkpoint_path):
            log_msg(f"FATAL [init 3/5] Checkpoint not found: {hypernet_checkpoint_path}")
            return
        ckpt_size_mb = os.path.getsize(hypernet_checkpoint_path) / (1024 ** 2)
        log_msg(f"[init 3/5] Checkpoint exists ({ckpt_size_mb:.1f} MB)")
    except Exception as e:
        log_msg(f"FATAL [init 3/5] Checkpoint check failed: {e}")
        import traceback; traceback.print_exc()
        return

    # -- Stage 4: load HyperNet ----------------------------------------------
    try:
        log_msg("[init 4/5] Loading HyperNet...")
        hypernet = load_hypernet(hypernet_checkpoint_path, device="cpu")
        hypernet.eval()
        log_msg("[init 4/5] HyperNet loaded OK")
    except Exception as e:
        log_msg(f"FATAL [init 4/5] HyperNet loading failed: {e}")
        import traceback; traceback.print_exc()
        return

    # -- Stage 5: optional RW augmentation -----------------------------------
    try:
        _use_rw = hasattr(hypernet, "rw_config") and hypernet.rw_config.enabled
        if _use_rw:
            log_msg("[init 5/5] Setting up RW augmentation...")
            from graph_hdc.utils.rw_features import augment_data_with_rw
            _rw_k = hypernet.rw_config.k_values
            _rw_bins = hypernet.rw_config.num_bins
            _rw_boundaries = hypernet.rw_config.bin_boundaries
            _rw_clip_range = hypernet.rw_config.clip_range
            bin_mode = "quantile" if _rw_boundaries else ("clipped" if _rw_clip_range else "uniform")
            log_msg(f"[init 5/5] RW augmentation enabled: k={_rw_k}, bins={_rw_bins}, mode={bin_mode}")
        else:
            log_msg("[init 5/5] No RW augmentation")
    except Exception as e:
        log_msg(f"FATAL [init 5/5] RW augmentation setup failed: {e}")
        import traceback; traceback.print_exc()
        return

    log_msg(
        f"Worker ready: pool={len(smiles_list)}, "
        f"batch_size={encoding_batch_size}, max_nodes={max_nodes}"
    )

    iteration = 0
    retry_count = 0
    sanitize_fail_count = 0  # Molecules that failed RDKit SanitizeMol
    disconnected_count = 0   # Molecules with disconnected components

    # Profiling accumulators
    profile_sample = 0.0
    profile_validation = 0.0
    profile_conversion = 0.0
    profile_hdc = 0.0
    profile_serialize = 0.0
    profile_queue = 0.0

    log_msg(f"Ready, pool size={len(smiles_list)}")

    while not stop_event.is_set():
        try:
            # --- Collect a batch of valid molecules ---
            zinc_data_list: list = []
            flow_data_list: list = []

            while len(zinc_data_list) < encoding_batch_size and not stop_event.is_set():
                t0 = time_module.perf_counter()
                smiles = random.choice(smiles_list)
                mol = Chem.MolFromSmiles(smiles)
                t1 = time_module.perf_counter()
                profile_sample += t1 - t0

                if mol is None:
                    retry_count += 1
                    continue

                if mol.GetNumHeavyAtoms() > max_nodes:
                    retry_count += 1
                    continue

                # Connectivity and validity checks
                t0 = time_module.perf_counter()
                canon_smi = Chem.MolToSmiles(mol, canonical=True)
                if "." in canon_smi:
                    t1 = time_module.perf_counter()
                    profile_validation += t1 - t0
                    disconnected_count += 1
                    retry_count += 1
                    continue
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    t1 = time_module.perf_counter()
                    profile_validation += t1 - t0
                    sanitize_fail_count += 1
                    retry_count += 1
                    continue
                t1 = time_module.perf_counter()
                profile_validation += t1 - t0

                t0 = time_module.perf_counter()
                zinc_data = mol_to_zinc_data(mol)
                if zinc_data is None:
                    t1 = time_module.perf_counter()
                    profile_conversion += t1 - t0
                    retry_count += 1
                    continue

                flow_data = mol_to_flow_data(mol)
                if flow_data is None or flow_data.edge_index.numel() == 0:
                    t1 = time_module.perf_counter()
                    profile_conversion += t1 - t0
                    retry_count += 1
                    continue
                t1 = time_module.perf_counter()
                profile_conversion += t1 - t0

                # Augment with RW features if needed
                if _use_rw:
                    zinc_data = augment_data_with_rw(zinc_data, k_values=_rw_k, num_bins=_rw_bins, bin_boundaries=_rw_boundaries, clip_range=_rw_clip_range)
                    # Extend flow_data node features with one-hot RW bins
                    rw_bin_cols = zinc_data.x[:, 5:]  # (n, len(k_values))
                    rw_onehot_parts = []
                    for col_idx in range(rw_bin_cols.size(1)):
                        rw_onehot_parts.append(
                            F.one_hot(rw_bin_cols[:, col_idx].long(), num_classes=_rw_bins).float()
                        )
                    flow_data.x = torch.cat([flow_data.x] + rw_onehot_parts, dim=-1)

                zinc_data_list.append(zinc_data)
                flow_data_list.append(flow_data)

            if stop_event.is_set() or not zinc_data_list:
                break

            # --- Batched HyperNet encoding ---
            t0 = time_module.perf_counter()
            with torch.no_grad():
                zinc_batch = Batch.from_data_list(zinc_data_list)
                hdc_out = hypernet.forward(zinc_batch)

                order_zero = hdc_out["node_terms"]
                order_n = hdc_out["graph_embedding"]
                hdc_vectors = torch.cat([order_zero, order_n], dim=-1).float()
            t1 = time_module.perf_counter()
            profile_hdc += t1 - t0

            # --- Serialize and push ---
            for i, flow_data in enumerate(flow_data_list):
                if stop_event.is_set():
                    break

                t0 = time_module.perf_counter()
                hdc_vector = hdc_vectors[i : i + 1].clone().detach()
                serialized_data = {
                    "x": flow_data.x.numpy(),
                    "edge_index": flow_data.edge_index.numpy(),
                    "edge_attr": flow_data.edge_attr.numpy(),
                    "hdc_vector": hdc_vector.numpy(),
                    "smiles": flow_data.smiles,
                }
                t1 = time_module.perf_counter()
                profile_serialize += t1 - t0

                t0 = time_module.perf_counter()
                while not stop_event.is_set():
                    try:
                        output_queue.put(serialized_data, timeout=1.0)
                        break
                    except Exception:
                        continue
                t1 = time_module.perf_counter()
                profile_queue += t1 - t0

                if stop_event.is_set():
                    break

                iteration += 1

            # Periodic logging
            if iteration >= log_interval and iteration % log_interval < encoding_batch_size:
                total_time = (
                    profile_sample + profile_validation + profile_conversion
                    + profile_hdc + profile_serialize + profile_queue
                )
                if total_time > 0:
                    sps = iteration / total_time
                    total_attempts = iteration + retry_count
                    log_msg(
                        f"Generated {iteration} samples, {retry_count} retries, "
                        f"{sps:.1f} samples/sec\n"
                        f"  Sample: {100*profile_sample/total_time:.1f}% | "
                        f"Valid: {100*profile_validation/total_time:.1f}% | "
                        f"Conv: {100*profile_conversion/total_time:.1f}% | "
                        f"HDC: {100*profile_hdc/total_time:.1f}% | "
                        f"Ser: {100*profile_serialize/total_time:.1f}% | "
                        f"Queue: {100*profile_queue/total_time:.1f}%\n"
                        f"  Rejected: {disconnected_count} disconnected, "
                        f"{sanitize_fail_count} invalid "
                        f"({100 * (disconnected_count + sanitize_fail_count) / max(total_attempts, 1):.1f}% of attempts)"
                    )

        except Exception as e:
            log_msg(f"Error during generation: {e}")
            import traceback
            traceback.print_exc()
            retry_count += 1
            continue

    log_msg(f"Stopped after {iteration} samples")


# =============================================================================
# Streaming Dataset
# =============================================================================


class SmallMoleculeStreamingDataset(torch.utils.data.IterableDataset):
    """
    IterableDataset that generates HDC-encoded samples by sampling from a
    :class:`SmallMoleculePool`.

    Manages worker processes analogous to
    :class:`~graph_hdc.datasets.streaming_fragments.StreamingFragmentDataset`.
    """

    def __init__(
        self,
        smiles_pool: SmallMoleculePool,
        hypernet_checkpoint_path: str | Path,
        buffer_size: int = 2000,
        num_workers: int = 1,
        max_nodes: int = 50,
        prefill_fraction: float = 0.1,
        log_interval: int = 1000,
        encoding_batch_size: int = 32,
    ):
        super().__init__()
        self.smiles_pool = smiles_pool
        self.hypernet_checkpoint_path = Path(hypernet_checkpoint_path)
        self._buffer_size = buffer_size
        self.num_workers = num_workers
        self.max_nodes = max_nodes
        self.prefill_fraction = prefill_fraction
        self.log_interval = log_interval
        self.encoding_batch_size = encoding_batch_size

        self._queue: Optional[mp.Queue] = None
        self._stop_event: Optional[mp.Event] = None
        self._workers: list = []
        self._started = False

    # -- diagnostics ---------------------------------------------------------

    def _format_worker_diagnostics(self) -> str:
        """Collect exit codes from dead workers."""
        lines = []
        for i, w in enumerate(self._workers):
            code = w.exitcode
            if code is None:
                status = "still running"
            elif code == 0:
                status = "exited normally (init error caught — check stderr)"
            elif code < 0:
                import signal as sig_mod

                try:
                    sig_name = sig_mod.Signals(-code).name
                except (ValueError, AttributeError):
                    sig_name = f"signal {-code}"
                status = f"killed by {sig_name}"
                if code == -9:
                    status += " (likely OOM)"
            else:
                status = f"exited with code {code}"
            lines.append(f"  Worker {i}: {status}")
        return "\n".join(lines)

    # -- public interface expected by MixedStreamingDataLoader ---------------

    @property
    def buffer_size(self) -> int:
        return self._buffer_size

    @property
    def current_buffer_size(self) -> int:
        if self._queue is None:
            return 0
        return self._queue.qsize()

    def start_workers(self) -> None:
        """Spawn worker processes using the ``spawn`` multiprocessing context."""
        if self._started:
            logger.warning("SmallMolecule workers already started")
            return

        try:
            ctx = mp.get_context("spawn")
        except ValueError:
            ctx = mp

        self._queue = ctx.Queue(maxsize=self._buffer_size)
        self._stop_event = ctx.Event()

        # Hide GPU from worker processes — they only need CPU for HDC encoding.
        # See streaming_fragments.py for detailed explanation.
        import os
        _prev_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        self._workers = []
        for i in range(self.num_workers):
            worker = ctx.Process(
                target=_small_mol_worker_process,
                args=(
                    i,
                    self.smiles_pool.smiles_list,
                    self._queue,
                    self._stop_event,
                    str(self.hypernet_checkpoint_path),
                    self.max_nodes,
                    self.log_interval,
                    self.encoding_batch_size,
                ),
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)

        # Restore parent's CUDA visibility
        if _prev_cuda is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = _prev_cuda

        logger.info(f"Started {self.num_workers} small-molecule workers")
        print(
            f"[SmallMolDataset] Started {self.num_workers} workers "
            f"(pool={self.smiles_pool.size}, batch={self.encoding_batch_size})",
            flush=True,
        )

        # Prefill
        target_size = int(self._buffer_size * self.prefill_fraction)
        print(
            f"[SmallMolDataset] Waiting for buffer to reach {target_size} samples...",
            flush=True,
        )
        prefill_timeout = 120  # seconds
        start_time = time.time()
        samples_received = 0

        while samples_received < target_size:
            elapsed = time.time() - start_time
            if elapsed > prefill_timeout:
                alive_workers = [w for w in self._workers if w.is_alive()]
                if not alive_workers:
                    raise RuntimeError(
                        "All small-molecule workers died during prefill.\n"
                        + self._format_worker_diagnostics()
                    )
                print(
                    f"[SmallMolDataset] Warning: Prefill timeout after {elapsed:.1f}s, "
                    f"but {len(alive_workers)} workers still alive. Continuing...",
                    flush=True,
                )
                break

            try:
                serialized_data = self._queue.get(timeout=5.0)
                # Put it back — use timeout to avoid deadlocking when workers
                # fill the freed slot before we can return the sample.
                try:
                    self._queue.put(serialized_data, timeout=1.0)
                except Exception:
                    pass  # Queue full — workers refilled it, sample lost but OK
                samples_received += 1
                if samples_received % 100 == 0:
                    print(
                        f"[SmallMolDataset] Prefill progress: "
                        f"{samples_received}/{target_size}",
                        flush=True,
                    )
            except Exception:
                alive = sum(1 for w in self._workers if w.is_alive())
                if alive == 0:
                    raise RuntimeError(
                        "All small-molecule workers died during prefill.\n"
                        + self._format_worker_diagnostics()
                    )

        print(
            f"[SmallMolDataset] Buffer prefilled with ~{samples_received} samples",
            flush=True,
        )
        self._started = True

    def stop_workers(self) -> None:
        """Gracefully stop all workers."""
        if not self._started:
            return

        print("[SmallMolDataset] Stopping workers...", flush=True)

        if self._stop_event is not None:
            self._stop_event.set()

        # Drain queue to unblock workers stuck on put()
        if self._queue is not None:
            drained = 0
            while True:
                try:
                    self._queue.get_nowait()
                    drained += 1
                except Exception:
                    break
            if drained > 0:
                print(f"[SmallMolDataset] Drained {drained} items from queue", flush=True)

        # Join workers with periodic queue draining
        for i, worker in enumerate(self._workers):
            for _ in range(10):
                worker.join(timeout=1.0)
                if not worker.is_alive():
                    break
                if self._queue is not None:
                    try:
                        while True:
                            self._queue.get_nowait()
                    except Exception:
                        pass

            if worker.is_alive():
                print(f"[SmallMolDataset] Worker {i} did not stop, terminating", flush=True)
                worker.terminate()
                worker.join(timeout=2)

        # Final queue cleanup
        if self._queue is not None:
            try:
                while True:
                    self._queue.get_nowait()
            except Exception:
                pass
            try:
                self._queue.close()
                self._queue.join_thread()
            except Exception:
                pass

        self._workers = []
        self._started = False
        print("[SmallMolDataset] All workers stopped", flush=True)

    def __iter__(self) -> Iterator[Data]:
        """Yield samples from the queue."""
        if not self._started:
            self.start_workers()

        while True:
            try:
                serialized_data = self._queue.get(timeout=30)
                data = Data(
                    x=torch.from_numpy(serialized_data["x"]),
                    edge_index=torch.from_numpy(serialized_data["edge_index"]),
                    edge_attr=torch.from_numpy(serialized_data["edge_attr"]),
                    hdc_vector=torch.from_numpy(serialized_data["hdc_vector"]),
                    smiles=serialized_data["smiles"],
                )
                yield data
            except Exception as e:
                alive = sum(1 for w in self._workers if w.is_alive())
                if alive == 0:
                    logger.error("All small-molecule workers died")
                    break
                logger.warning(f"Queue get timeout: {e}")

    def __del__(self) -> None:
        self.stop_workers()
