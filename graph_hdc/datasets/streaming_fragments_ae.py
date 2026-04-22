"""
Simplified streaming fragment dataset for autoencoder training.

Produces only HyperNet-encoded vectors (edge_terms + graph_terms),
skipping the flow_data features needed by the edge decoder.  This is
faster than the full streaming pipeline since it avoids computing
24-dim one-hot features, edge attributes, and their serialization.

Usage:
    from graph_hdc.datasets.streaming_fragments import FragmentLibrary
    from graph_hdc.datasets.streaming_fragments_ae import (
        StreamingAEDataset,
        StreamingAEDataLoader,
    )

    library = FragmentLibrary(min_atoms=2, max_atoms=30)
    library.build_from_dataset(zinc_train)

    dataset = StreamingAEDataset(
        fragment_library=library,
        hypernet_checkpoint_path="/path/to/encoder.ckpt",
    )
    loader = StreamingAEDataLoader(dataset, batch_size=512, steps_per_epoch=1000)

    # Use in training loop
    for batch in loader:
        # batch.node_terms: [B, hv_dim]  (actually edge_terms)
        # batch.graph_terms: [B, hv_dim]
        ...

    loader.stop()
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from graph_hdc.datasets.streaming_fragments import (
    FragmentLibrary,
    mol_to_zinc_data,
    strip_dummy_atoms,
)


# =============================================================================
# WORKER PROCESS
# =============================================================================


def _ae_worker_process(
    fragment_library: FragmentLibrary,
    hypernet_checkpoint_path: str,
    output_queue: mp.Queue,
    stop_event: mp.Event,
    fragments_range: Tuple[int, int],
    max_nodes: int,
    encoding_batch_size: int,
    log_interval: int,
    worker_id: int,
):
    """Worker process for streaming AE data generation.

    Simplified pipeline (vs edge decoder worker):
    1. Sample & combine BRICS fragments
    2. Convert to zinc_data for HyperNet
    3. Batch-encode with HyperNet
    4. Extract edge_terms + graph_terms (skip flow_data entirely)
    5. Serialize and push to queue
    """
    import random
    import time as time_module

    from rdkit import Chem
    from rdkit import RDLogger as _RDLogger

    _RDLogger.DisableLog("rdApp.*")

    def log_msg(msg: str) -> None:
        print(f"[AE Worker {worker_id}] {msg}", file=sys.stderr, flush=True)

    log_msg("Starting worker process...")

    try:
        from graph_hdc.hypernet import load_hypernet

        log_msg(f"Loading HyperNet from {hypernet_checkpoint_path}")
        hypernet = load_hypernet(hypernet_checkpoint_path, device="cpu")
        hypernet.eval()

        # Check if RW augmentation is needed
        _use_rw = hasattr(hypernet, "rw_config") and hypernet.rw_config.enabled
        if _use_rw:
            from graph_hdc.utils.rw_features import augment_data_with_rw_np

            _rw_k = hypernet.rw_config.k_values
            _rw_bins = hypernet.rw_config.num_bins
            _rw_boundaries = hypernet.rw_config.bin_boundaries
            _rw_clip_range = hypernet.rw_config.clip_range
            log_msg(f"RW augmentation enabled: k={_rw_k}, bins={_rw_bins}")

        log_msg(f"HyperNet loaded (batch_size={encoding_batch_size})")

    except Exception as ex:
        log_msg(f"FATAL: Failed to load HyperNet: {ex}")
        import traceback

        traceback.print_exc()
        return

    iteration = 0
    rej_combine = 0   # BRICS combination returned None
    rej_size = 0      # Too many atoms
    rej_disconn = 0   # Disconnected fragments
    rej_sanitize = 0  # RDKit sanitization failed
    rej_convert = 0   # mol_to_zinc_data failed

    profile_fragment = 0.0
    profile_validation = 0.0
    profile_conversion = 0.0
    profile_rw = 0.0
    profile_hdc_batch = 0.0  # Batch.from_data_list
    profile_hdc_forward = 0.0  # hypernet.forward
    profile_hdc_extract = 0.0  # detach + cpu
    profile_queue = 0.0

    log_msg("Worker ready, starting generation loop")

    while not stop_event.is_set():
        try:
            # === Collect a batch of valid molecules ===
            zinc_data_list: list = []
            smiles_list: list[str] = []

            while len(zinc_data_list) < encoding_batch_size and not stop_event.is_set():
                # Fragment sampling + BRICS combination
                t0 = time_module.perf_counter()
                n_frags = random.randint(fragments_range[0], fragments_range[1])
                fragments = fragment_library.sample_fragments(n_frags)

                if n_frags == 1:
                    mol = strip_dummy_atoms(fragments[0])
                else:
                    mol = fragment_library.combine_fragments(fragments)
                    # combine_fragments only removes dummies at each join
                    # point — unused attachment points remain as * atoms.
                    # Strip them so mol_to_zinc_data doesn't reject.
                    if mol is not None:
                        mol = strip_dummy_atoms(mol)

                t1 = time_module.perf_counter()
                profile_fragment += t1 - t0

                if mol is None:
                    rej_combine += 1
                    continue

                # Validation
                t0 = time_module.perf_counter()

                if mol.GetNumAtoms() > max_nodes:
                    profile_validation += time_module.perf_counter() - t0
                    rej_size += 1
                    continue

                smiles = Chem.MolToSmiles(mol, canonical=True)
                if "." in smiles:
                    profile_validation += time_module.perf_counter() - t0
                    rej_disconn += 1
                    continue

                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    profile_validation += time_module.perf_counter() - t0
                    rej_sanitize += 1
                    continue

                profile_validation += time_module.perf_counter() - t0

                # Data conversion (zinc_data only — no flow_data needed)
                t0 = time_module.perf_counter()
                zinc_data = mol_to_zinc_data(mol)
                if zinc_data is None or zinc_data.edge_index.numel() == 0:
                    profile_conversion += time_module.perf_counter() - t0
                    rej_convert += 1
                    continue
                profile_conversion += time_module.perf_counter() - t0

                # RW augmentation if needed
                if _use_rw:
                    t0 = time_module.perf_counter()
                    zinc_data = augment_data_with_rw_np(
                        zinc_data,
                        k_values=_rw_k,
                        num_bins=_rw_bins,
                        bin_boundaries=_rw_boundaries,
                        clip_range=_rw_clip_range,
                    )
                    profile_rw += time_module.perf_counter() - t0

                zinc_data_list.append(zinc_data)
                smiles_list.append(smiles)

            if stop_event.is_set() or not zinc_data_list:
                break

            # === Collect edge pairs for incremental codebook ===
            batch_edge_pairs: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
            for zd in zinc_data_list:
                node_tuples = [
                    tuple(int(v) for v in row)
                    for row in zd.x.tolist()
                ]
                for e_idx in range(zd.edge_index.size(1)):
                    src = zd.edge_index[0, e_idx].item()
                    dst = zd.edge_index[1, e_idx].item()
                    batch_edge_pairs.add((node_tuples[src], node_tuples[dst]))

            # === Batched HyperNet encoding ===
            t0 = time_module.perf_counter()
            zinc_batch = Batch.from_data_list(zinc_data_list)
            t1 = time_module.perf_counter()
            profile_hdc_batch += t1 - t0

            with torch.no_grad():
                hdc_out = hypernet.forward(zinc_batch)
            t2 = time_module.perf_counter()
            profile_hdc_forward += t2 - t1

            edge_terms = hdc_out["edge_terms"].detach().cpu()
            graph_terms = hdc_out["graph_embedding"].detach().cpu()
            t3 = time_module.perf_counter()
            profile_hdc_extract += t3 - t2

            # === Serialize and push ===
            for i in range(len(zinc_data_list)):
                if stop_event.is_set():
                    break

                serialized = {
                    # edge_terms go into node_terms slot (the AE swap)
                    "node_terms": edge_terms[i].numpy(),
                    "graph_terms": graph_terms[i].numpy(),
                    "smiles": smiles_list[i],
                    # Attach edge pairs once per encoding batch (first sample)
                    "edge_pairs": batch_edge_pairs if i == 0 else None,
                }

                t0 = time_module.perf_counter()
                while not stop_event.is_set():
                    try:
                        output_queue.put(serialized, timeout=1.0)
                        break
                    except Exception:
                        continue
                profile_queue += time_module.perf_counter() - t0

                iteration += 1

            # Periodic logging
            if iteration > 0 and iteration % log_interval == 0:
                total_time = (
                    profile_fragment
                    + profile_validation
                    + profile_conversion
                    + profile_rw
                    + profile_hdc_batch
                    + profile_hdc_forward
                    + profile_hdc_extract
                    + profile_queue
                )
                total_rej = rej_combine + rej_size + rej_disconn + rej_sanitize + rej_convert
                if total_time > 0:
                    # Absolute times for the detailed breakdown
                    samples_per_sec = iteration / total_time
                    log_msg(
                        f"Generated {iteration} samples ({samples_per_sec:.0f} samples/s) | "
                        f"rejected {total_rej} "
                        f"(combine={rej_combine} size={rej_size} "
                        f"disconn={rej_disconn} sanitize={rej_sanitize} "
                        f"convert={rej_convert}) | "
                        f"frag={100 * profile_fragment / total_time:.0f}% "
                        f"val={100 * profile_validation / total_time:.0f}% "
                        f"conv={100 * profile_conversion / total_time:.0f}% "
                        f"rw={100 * profile_rw / total_time:.0f}% "
                        f"hdc_batch={100 * profile_hdc_batch / total_time:.0f}% "
                        f"hdc_fwd={100 * profile_hdc_forward / total_time:.0f}% "
                        f"hdc_ext={100 * profile_hdc_extract / total_time:.0f}% "
                        f"queue={100 * profile_queue / total_time:.0f}%"
                    )
                    log_msg(
                        f"  Absolute: frag={profile_fragment:.1f}s "
                        f"val={profile_validation:.1f}s "
                        f"conv={profile_conversion:.1f}s "
                        f"rw={profile_rw:.1f}s "
                        f"hdc_batch={profile_hdc_batch:.1f}s "
                        f"hdc_fwd={profile_hdc_forward:.1f}s "
                        f"hdc_ext={profile_hdc_extract:.1f}s "
                        f"queue={profile_queue:.1f}s "
                        f"total={total_time:.1f}s"
                    )

        except Exception as ex:
            log_msg(f"ERROR: {ex}")
            import traceback

            traceback.print_exc(file=sys.stderr)
            continue

    log_msg(f"Worker stopped after {iteration} samples")


# =============================================================================
# STREAMING DATASET
# =============================================================================


class StreamingAEDataset:
    """Streaming dataset for AE training on BRICS-generated molecules.

    Workers generate molecules from BRICS fragments, encode with HyperNet,
    and produce Data objects with ``node_terms`` (edge_terms) and
    ``graph_terms`` only.

    Args:
        fragment_library: Built FragmentLibrary instance.
        hypernet_checkpoint_path: Path to HyperNet checkpoint for workers.
        buffer_size: Maximum queue size.
        num_workers: Number of worker processes.
        fragments_range: (min, max) fragments per generated molecule.
        max_nodes: Maximum atoms per molecule.
        encoding_batch_size: Molecules per HyperNet batch in workers.
        prefill_fraction: Fill buffer to this fraction before yielding.
        log_interval: Workers log every N generated samples.
    """

    def __init__(
        self,
        fragment_library: FragmentLibrary,
        hypernet_checkpoint_path: str,
        buffer_size: int = 2000,
        num_workers: int = 2,
        fragments_range: Tuple[int, int] = (1, 4),
        max_nodes: int = 40,
        encoding_batch_size: int = 32,
        prefill_fraction: float = 0.1,
        log_interval: int = 1000,
    ):
        self.fragment_library = fragment_library
        self.hypernet_checkpoint_path = str(hypernet_checkpoint_path)
        self.buffer_size = buffer_size
        self.num_workers = num_workers
        self.fragments_range = fragments_range
        self.max_nodes = max_nodes
        self.encoding_batch_size = encoding_batch_size
        self.prefill_fraction = prefill_fraction
        self.log_interval = log_interval

        self._queue: mp.Queue | None = None
        self._workers: list[mp.Process] = []
        self._stop_event: mp.Event | None = None
        self._started = False

    def start(self):
        """Start worker processes and prefill the buffer."""
        if self._started:
            return

        ctx = mp.get_context("spawn")
        self._stop_event = ctx.Event()
        self._queue = ctx.Queue(maxsize=self.buffer_size)

        # Hide GPUs from workers (they use CPU only)
        cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        for wid in range(self.num_workers):
            p = ctx.Process(
                target=_ae_worker_process,
                args=(
                    self.fragment_library,
                    self.hypernet_checkpoint_path,
                    self._queue,
                    self._stop_event,
                    self.fragments_range,
                    self.max_nodes,
                    self.encoding_batch_size,
                    self.log_interval,
                    wid,
                ),
                daemon=True,
            )
            p.start()
            self._workers.append(p)

        # Restore CUDA env
        if cuda_env is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_env
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        self._started = True

        # Prefill
        target = int(self.buffer_size * self.prefill_fraction)
        if target > 0:
            print(
                f"[StreamingAE] Prefilling buffer to {target}...",
                file=sys.stderr,
                flush=True,
            )
            deadline = time.time() + 120
            while self._queue.qsize() < target and time.time() < deadline:
                alive = any(w.is_alive() for w in self._workers)
                if not alive:
                    raise RuntimeError("All AE streaming workers died during prefill")
                time.sleep(0.5)
            print(
                f"[StreamingAE] Buffer at {self._queue.qsize()}",
                file=sys.stderr,
                flush=True,
            )

    def stop(self):
        """Stop all worker processes and drain the queue."""
        if not self._started:
            return

        if self._stop_event is not None:
            self._stop_event.set()

        # Drain queue to unblock workers stuck on put
        if self._queue is not None:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except Exception:
                    break

        for w in self._workers:
            w.join(timeout=5.0)
            if w.is_alive():
                w.terminate()

        if self._queue is not None:
            try:
                self._queue.close()
                self._queue.join_thread()
            except Exception:
                pass

        self._workers = []
        self._started = False

    def __iter__(self):
        while True:
            try:
                d = self._queue.get(timeout=30)
            except Exception:
                alive = any(w.is_alive() for w in self._workers)
                if not alive:
                    raise RuntimeError("All AE streaming workers have died")
                continue

            data = Data(
                node_terms=torch.from_numpy(d["node_terms"]),
                graph_terms=torch.from_numpy(d["graph_terms"]),
                smiles=d["smiles"],
            )
            # Attach edge pairs as a plain Python attribute (not a tensor,
            # so PyG batching ignores it).  Only the first sample per
            # worker encoding batch carries a non-None value.
            data._edge_pairs = d.get("edge_pairs")
            yield data


# =============================================================================
# DATALOADER
# =============================================================================


class StreamingAEDataLoader:
    """DataLoader for StreamingAEDataset.

    Yields batched PyG Batch objects for a fixed number of steps per epoch.

    Args:
        dataset: StreamingAEDataset instance.
        batch_size: Number of molecules per batch.
        steps_per_epoch: Number of batches per epoch.
    """

    def __init__(
        self,
        dataset: StreamingAEDataset,
        batch_size: int = 512,
        steps_per_epoch: int = 1000,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self._iter = None
        self._epoch_edge_pairs: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        if not self.dataset._started:
            self.dataset.start()
        if self._iter is None:
            self._iter = iter(self.dataset)

        self._epoch_edge_pairs = set()

        for _ in range(self.steps_per_epoch):
            batch_list = []
            for _ in range(self.batch_size):
                data = next(self._iter)
                ep = getattr(data, "_edge_pairs", None)
                if ep is not None:
                    self._epoch_edge_pairs.update(ep)
                batch_list.append(data)
            yield Batch.from_data_list(batch_list)

    def drain_edge_pairs(self) -> set[tuple[tuple[int, ...], tuple[int, ...]]]:
        """Return edge pairs accumulated this epoch and clear the buffer."""
        pairs = self._epoch_edge_pairs
        self._epoch_edge_pairs = set()
        return pairs

    def stop(self):
        """Stop the underlying streaming dataset."""
        self.dataset.stop()

    def test_iteration(self, num_batches: int = 2):
        """Smoke test: pull a few batches to verify the pipeline works."""
        if not self.dataset._started:
            self.dataset.start()
        if self._iter is None:
            self._iter = iter(self.dataset)

        for i in range(num_batches):
            batch_list = []
            for _ in range(self.batch_size):
                batch_list.append(next(self._iter))
            batch = Batch.from_data_list(batch_list)
            print(
                f"[StreamingAE] Test batch {i}: {batch.num_graphs} graphs, "
                f"node_terms={batch.node_terms.shape}",
                file=sys.stderr,
                flush=True,
            )
