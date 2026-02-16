"""
Tests for small molecule streaming dataset and mixed streaming dataloader.
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from typing import Iterator, List
from unittest.mock import MagicMock

import pytest
import torch
from torch_geometric.data import Batch, Data

from graph_hdc.datasets.mixed_streaming import MixedStreamingDataLoader, StreamingSource
from graph_hdc.datasets.streaming_small_molecules import SmallMoleculePool


# =============================================================================
# Fixtures
# =============================================================================

SAMPLE_SMILES = [
    ("c1ccccc1", "zinc"),          # benzene
    ("CCO", "zinc"),               # ethanol
    ("CC(=O)O", "zinc"),           # acetic acid
    ("c1ccncc1", "qm9"),           # pyridine
    ("C1CCCCC1", "qm9"),           # cyclohexane
    ("CC=O", "qm9"),               # acetaldehyde
]


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Create a temporary small_molecules.csv."""
    csv_path = tmp_path / "small_molecules.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", "source"])
        for smiles, source in SAMPLE_SMILES:
            writer.writerow([smiles, source])
    return csv_path


# =============================================================================
# SmallMoleculePool Tests
# =============================================================================


class TestSmallMoleculePool:
    """Tests for SmallMoleculePool CSV loading."""

    def test_load_from_csv(self, sample_csv: Path):
        pool = SmallMoleculePool(sample_csv)
        assert pool.size == len(SAMPLE_SMILES)

    def test_len(self, sample_csv: Path):
        pool = SmallMoleculePool(sample_csv)
        assert len(pool) == len(SAMPLE_SMILES)

    def test_sample_returns_valid_smiles(self, sample_csv: Path):
        pool = SmallMoleculePool(sample_csv)
        known = {s for s, _ in SAMPLE_SMILES}
        for _ in range(20):
            s = pool.sample()
            assert s in known

    def test_all_smiles_loaded(self, sample_csv: Path):
        pool = SmallMoleculePool(sample_csv)
        loaded = set(pool.smiles_list)
        expected = {s for s, _ in SAMPLE_SMILES}
        assert loaded == expected

    def test_empty_csv_raises(self, tmp_path: Path):
        csv_path = tmp_path / "empty.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["smiles", "source"])
        pool = SmallMoleculePool(csv_path)
        assert pool.size == 0
        with pytest.raises(IndexError):
            pool.sample()

    def test_real_csv_loads(self):
        """Test loading the actual data/small_molecules.csv if it exists."""
        real_path = Path("data/small_molecules.csv")
        if not real_path.exists():
            pytest.skip("data/small_molecules.csv not found")
        pool = SmallMoleculePool(real_path)
        assert pool.size > 0
        # Should have at least 1000 molecules
        assert pool.size > 1000


# =============================================================================
# Data Format Compatibility Tests
# =============================================================================


class TestDataFormatCompatibility:
    """Verify that small molecules produce valid FlowEdgeDecoder data."""

    def test_qm9_smiles_produces_valid_flow_data(self):
        """QM9 molecules should convert to valid 24-dim + 5-class format."""
        from rdkit import Chem

        from graph_hdc.datasets.streaming_fragments import mol_to_flow_data

        # QM9-like small molecules
        qm9_smiles = ["c1ccncc1", "C1CCCC1", "CC(=O)O", "CCN"]
        for smi in qm9_smiles:
            mol = Chem.MolFromSmiles(smi)
            assert mol is not None, f"Failed to parse {smi}"
            data = mol_to_flow_data(mol)
            assert data is not None, f"mol_to_flow_data returned None for {smi}"
            # Node features: 24-dim one-hot
            assert data.x.shape[1] == 24, f"Expected 24-dim x, got {data.x.shape[1]}"
            # Edge attributes: 5-class one-hot
            assert data.edge_attr.shape[1] == 5, f"Expected 5-class edges, got {data.edge_attr.shape[1]}"
            # Has edges
            assert data.edge_index.numel() > 0

    def test_zinc_small_smiles_produces_valid_flow_data(self):
        """Small ZINC molecules should also convert correctly."""
        from rdkit import Chem

        from graph_hdc.datasets.streaming_fragments import mol_to_flow_data

        zinc_smiles = ["c1ccccc1", "CCO", "CC(=O)O", "c1ccc(O)cc1"]
        for smi in zinc_smiles:
            mol = Chem.MolFromSmiles(smi)
            assert mol is not None
            data = mol_to_flow_data(mol)
            assert data is not None
            assert data.x.shape[1] == 24
            assert data.edge_attr.shape[1] == 5

    def test_mol_to_zinc_data_works_for_qm9_atoms(self):
        """QM9 atom types {C, N, O, F} are a subset of ZINC's atom set."""
        from rdkit import Chem

        from graph_hdc.datasets.streaming_fragments import mol_to_zinc_data

        mol = Chem.MolFromSmiles("c1ccncc1")  # pyridine: C and N
        data = mol_to_zinc_data(mol)
        assert data is not None
        # ZINC format: 5 raw features per atom
        assert data.x.shape[1] == 5


# =============================================================================
# Mock Streaming Source for MixedStreamingDataLoader Tests
# =============================================================================


class MockStreamingDataset:
    """
    Minimal mock that satisfies the StreamingSource protocol.

    Yields Data objects with a ``source_tag`` attribute for verifying
    which source produced each sample.
    """

    def __init__(self, tag: str, num_atoms: int = 5):
        self.tag = tag
        self.num_atoms = num_atoms
        self._buffer_size = 100
        self._started = False

    @property
    def buffer_size(self) -> int:
        return self._buffer_size

    @property
    def current_buffer_size(self) -> int:
        return 50

    def start_workers(self) -> None:
        self._started = True

    def stop_workers(self) -> None:
        self._started = False

    def __iter__(self) -> Iterator[Data]:
        n = self.num_atoms
        while True:
            x = torch.randn(n, 24)
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
            edge_attr = torch.zeros(2, 5)
            edge_attr[:, 1] = 1.0  # single bond
            hdc_vector = torch.randn(1, 512)
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                hdc_vector=hdc_vector,
                smiles="MOCK",
                source_tag=self.tag,
            )
            yield data


# =============================================================================
# MixedStreamingDataLoader Tests
# =============================================================================


class TestMixedStreamingDataLoader:
    """Tests for MixedStreamingDataLoader."""

    def test_single_source(self):
        """Works with a single source."""
        ds = MockStreamingDataset("only")
        loader = MixedStreamingDataLoader(
            sources=[StreamingSource("only", ds, weight=1.0)],
            batch_size=4,
            steps_per_epoch=3,
            log_buffer_interval=0,
        )
        batches = list(loader)
        assert len(batches) == 3
        for batch in batches:
            assert batch.num_graphs == 4

    def test_two_sources_mixing(self):
        """Two sources are mixed approximately according to weights."""
        ds_a = MockStreamingDataset("A", num_atoms=3)
        ds_b = MockStreamingDataset("B", num_atoms=7)
        loader = MixedStreamingDataLoader(
            sources=[
                StreamingSource("A", ds_a, weight=0.8),
                StreamingSource("B", ds_b, weight=0.2),
            ],
            batch_size=100,
            steps_per_epoch=50,
            log_buffer_interval=0,
        )

        count_a = 0
        count_b = 0
        for batch in loader:
            # Distinguish sources by number of nodes per graph
            # A has 3 atoms, B has 7 atoms
            sizes = []
            ptr = batch.ptr
            for i in range(batch.num_graphs):
                sizes.append((ptr[i + 1] - ptr[i]).item())
            count_a += sum(1 for s in sizes if s == 3)
            count_b += sum(1 for s in sizes if s == 7)

        total = count_a + count_b
        ratio_a = count_a / total
        # With 80/20 weights, expect ~0.8 for A
        assert 0.7 < ratio_a < 0.9, f"Source A ratio {ratio_a:.3f} not near 0.8"

    def test_len(self):
        ds = MockStreamingDataset("x")
        loader = MixedStreamingDataLoader(
            sources=[StreamingSource("x", ds, weight=1.0)],
            batch_size=4,
            steps_per_epoch=42,
        )
        assert len(loader) == 42

    def test_stop_calls_all_sources(self):
        ds_a = MockStreamingDataset("A")
        ds_b = MockStreamingDataset("B")
        loader = MixedStreamingDataLoader(
            sources=[
                StreamingSource("A", ds_a, weight=0.5),
                StreamingSource("B", ds_b, weight=0.5),
            ],
            batch_size=2,
            steps_per_epoch=1,
        )
        # Consume one batch to initialise iterators
        next(iter(loader))
        loader.stop()
        assert not ds_a._started
        assert not ds_b._started

    def test_empty_sources_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            MixedStreamingDataLoader(sources=[], batch_size=4, steps_per_epoch=1)

    def test_get_buffer_stats_empty(self):
        ds = MockStreamingDataset("x")
        loader = MixedStreamingDataLoader(
            sources=[StreamingSource("x", ds, weight=1.0)],
            batch_size=4,
            steps_per_epoch=1,
        )
        stats = loader.get_buffer_stats()
        assert "x" in stats
        assert stats["x"]["mean"] == 0.0

    def test_weight_normalisation(self):
        """Weights don't need to sum to 1."""
        ds_a = MockStreamingDataset("A", num_atoms=3)
        ds_b = MockStreamingDataset("B", num_atoms=7)
        loader = MixedStreamingDataLoader(
            sources=[
                StreamingSource("A", ds_a, weight=9),
                StreamingSource("B", ds_b, weight=1),
            ],
            batch_size=4,
            steps_per_epoch=1,
            log_buffer_interval=0,
        )
        # Should not raise; weights normalised internally
        batches = list(loader)
        assert len(batches) == 1
