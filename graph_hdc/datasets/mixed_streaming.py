"""
Mixed Streaming DataLoader for combining multiple streaming data sources.

Provides a generic framework for mixing samples from *N* streaming sources
with configurable weights.  Each source is an
:class:`~torch.utils.data.IterableDataset` that manages its own worker
processes and multiprocessing queue.

Usage::

    from graph_hdc.datasets.mixed_streaming import MixedStreamingDataLoader, StreamingSource

    loader = MixedStreamingDataLoader(
        sources=[
            StreamingSource("fragments", fragment_dataset, weight=0.9),
            StreamingSource("small_molecules", small_mol_dataset, weight=0.1),
        ],
        batch_size=16,
        steps_per_epoch=1000,
    )

    for batch in loader:
        ...  # PyG Batch
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional

from torch_geometric.data import Batch, Data

logger = logging.getLogger(__name__)


@dataclass
class StreamingSource:
    """
    A named streaming data source with a relative sampling weight.

    The *dataset* object must implement the following informal protocol:

    - ``start_workers() -> None``
    - ``stop_workers() -> None``
    - ``__iter__() -> Iterator[Data]``
    - ``current_buffer_size: int``  (property)
    - ``buffer_size: int``          (property)

    Both :class:`~graph_hdc.datasets.streaming_fragments.StreamingFragmentDataset`
    and :class:`~graph_hdc.datasets.streaming_small_molecules.SmallMoleculeStreamingDataset`
    satisfy this protocol.
    """

    name: str
    dataset: Any
    weight: float = 1.0


class MixedStreamingDataLoader:
    """
    DataLoader that mixes samples from multiple streaming sources according to
    configurable weights.

    Implements the same interface as
    :class:`~graph_hdc.datasets.streaming_fragments.StreamingFragmentDataLoader`:

    - ``__iter__()`` yields :class:`~torch_geometric.data.Batch` objects
    - ``__len__()`` returns *steps_per_epoch*
    - ``stop()`` cleans up all sources
    - ``test_iteration()`` runs a smoke test
    - ``get_buffer_stats()`` returns per-source buffer statistics
    """

    def __init__(
        self,
        sources: List[StreamingSource],
        batch_size: int = 32,
        steps_per_epoch: int = 1000,
        collate_fn: Optional[Callable] = None,
        log_buffer_interval: int = 100,
    ):
        if not sources:
            raise ValueError("At least one StreamingSource is required")

        self.sources = sources
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.collate_fn = collate_fn or Batch.from_data_list
        self.log_buffer_interval = log_buffer_interval

        # Normalise weights to probabilities
        total_weight = sum(s.weight for s in sources)
        if total_weight <= 0:
            raise ValueError("Total source weight must be positive")
        self._source_probs = [s.weight / total_weight for s in sources]

        self._iterators: List[Optional[Iterator[Data]]] = [None] * len(sources)
        self._buffer_samples: Dict[str, List[int]] = {s.name: [] for s in sources}
        self._batch_count: int = 0

    # --------------------------------------------------------------------- #
    # Core iteration
    # --------------------------------------------------------------------- #

    def __iter__(self) -> Iterator[Batch]:
        """Yield batches for one epoch, mixing sources by weight."""
        # Lazily initialise iterators
        for i, source in enumerate(self.sources):
            if self._iterators[i] is None:
                self._iterators[i] = iter(source.dataset)

        for _step in range(self.steps_per_epoch):
            batch_data: List[Data] = []

            # Choose a source for every sample in the batch
            chosen_indices = random.choices(
                range(len(self.sources)),
                weights=self._source_probs,
                k=self.batch_size,
            )

            for source_idx in chosen_indices:
                try:
                    data = next(self._iterators[source_idx])
                except StopIteration:
                    # Restart iterator (streaming should never exhaust, but be safe)
                    self._iterators[source_idx] = iter(
                        self.sources[source_idx].dataset
                    )
                    data = next(self._iterators[source_idx])
                batch_data.append(data)

            self._batch_count += 1

            # Periodic buffer-level logging
            if (
                self.log_buffer_interval > 0
                and self._batch_count % self.log_buffer_interval == 0
            ):
                parts = []
                for source in self.sources:
                    buf_size = source.dataset.current_buffer_size
                    buf_max = source.dataset.buffer_size
                    pct = 100 * buf_size / buf_max if buf_max > 0 else 0
                    self._buffer_samples[source.name].append(buf_size)
                    parts.append(f"{source.name}={buf_size}/{buf_max} ({pct:.1f}%)")
                print(
                    f"[MixedLoader] Batch {self._batch_count}: {' | '.join(parts)}",
                    flush=True,
                )

            yield self.collate_fn(batch_data)

    def __len__(self) -> int:
        return self.steps_per_epoch

    # --------------------------------------------------------------------- #
    # Lifecycle
    # --------------------------------------------------------------------- #

    def stop(self) -> None:
        """Stop all streaming sources."""
        for source in self.sources:
            try:
                source.dataset.stop_workers()
            except Exception as e:
                logger.warning(f"Error stopping source '{source.name}': {e}")

    # --------------------------------------------------------------------- #
    # Diagnostics
    # --------------------------------------------------------------------- #

    def test_iteration(self, num_batches: int = 3) -> bool:
        """
        Smoke-test that the mixed loader can produce batches.

        Args:
            num_batches: Number of test batches to pull.

        Returns:
            ``True`` if successful; raises on failure.
        """
        print(f"[MixedLoader] Testing {num_batches} batches...", flush=True)

        for i, source in enumerate(self.sources):
            if self._iterators[i] is None:
                self._iterators[i] = iter(source.dataset)

        for i in range(num_batches):
            batch_data: List[Data] = []
            chosen_indices = random.choices(
                range(len(self.sources)),
                weights=self._source_probs,
                k=self.batch_size,
            )

            for source_idx in chosen_indices:
                try:
                    data = next(self._iterators[source_idx])
                except StopIteration:
                    self._iterators[source_idx] = iter(
                        self.sources[source_idx].dataset
                    )
                    data = next(self._iterators[source_idx])
                batch_data.append(data)

            batch = self.collate_fn(batch_data)

            # Report per-source counts
            source_counts: Dict[str, int] = {}
            for idx in chosen_indices:
                name = self.sources[idx].name
                source_counts[name] = source_counts.get(name, 0) + 1

            print(
                f"[MixedLoader] Test batch {i + 1}/{num_batches}: "
                f"{batch.num_graphs} graphs, "
                f"x shape: {batch.x.shape}, "
                f"hdc_vector shape: {batch.hdc_vector.shape}, "
                f"source mix: {source_counts}",
                flush=True,
            )

        print("[MixedLoader] Test successful!", flush=True)
        return True

    def get_buffer_stats(self) -> Dict[str, dict]:
        """
        Return per-source buffer utilisation statistics.

        Returns:
            Dict keyed by source name, each value a dict with
            ``min``, ``max``, ``mean``, ``samples``, ``utilization_pct``.
        """
        stats: Dict[str, dict] = {}
        for source in self.sources:
            samples = self._buffer_samples.get(source.name, [])
            if not samples:
                stats[source.name] = {
                    "min": 0,
                    "max": 0,
                    "mean": 0.0,
                    "samples": [],
                    "utilization_pct": 0.0,
                }
            else:
                mean_size = sum(samples) / len(samples)
                stats[source.name] = {
                    "min": min(samples),
                    "max": max(samples),
                    "mean": mean_size,
                    "samples": samples[-10:],
                    "utilization_pct": 100 * mean_size / source.dataset.buffer_size
                    if source.dataset.buffer_size > 0
                    else 0.0,
                }
        return stats
