"""Tests for the explicit feature_bins constructor path and domain integration."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, ClassVar

import pytest
import torch
from torch_geometric.data import Batch, Data

from graph_hdc.hypernet.encoder import HyperNet
from graph_hdc.hypernet.configs import DSHDCConfig, get_config


def _make_batch(x: torch.Tensor, edge_index: torch.Tensor) -> Batch:
    """Create a single-graph Batch from node features and edge index."""
    data = Data(x=x, edge_index=edge_index)
    return Batch.from_data_list([data])


# ===================================================================
# Creation tests
# ===================================================================

class TestCreationFromFeatureBins:

    def test_creation_from_feature_bins(self):
        hn = HyperNet(feature_bins=[4, 5, 3, 5])
        assert hn.hv_dim == 256
        assert hn.depth == 3
        assert hn.base_dataset is None

    def test_creation_from_feature_bins_with_options(self):
        hn = HyperNet(
            feature_bins=[4, 5, 3, 5],
            hv_dim=128,
            seed=7,
            normalize=True,
            dtype="float32",
        )
        assert hn.hv_dim == 128
        assert hn.seed == 7
        assert hn.normalize is True
        assert hn._dtype == torch.float32

    def test_creation_rejects_both_config_and_bins(self):
        config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
        with pytest.raises(ValueError, match="Cannot specify both"):
            HyperNet(config, feature_bins=[4, 5, 3, 5])

    def test_creation_rejects_neither(self):
        with pytest.raises(ValueError, match="Either 'config' or 'feature_bins'"):
            HyperNet()

    def test_feature_bins_property(self):
        hn = HyperNet(feature_bins=[4, 5, 3, 5])
        assert hn.feature_bins == [4, 5, 3, 5]

    def test_base_dataset_is_none(self):
        hn = HyperNet(feature_bins=[4, 5, 3, 5])
        assert hn.base_dataset is None

    def test_depth_override(self):
        hn = HyperNet(feature_bins=[4, 5, 3, 5], hypernet_depth=5)
        assert hn.depth == 5

    def test_depth_positional_override(self):
        """The positional `depth` arg still works with feature_bins."""
        hn = HyperNet(None, 7, feature_bins=[4, 5, 3, 5])
        assert hn.depth == 7


# ===================================================================
# Encoding equivalence tests
# ===================================================================

class TestEncodingEquivalence:

    def test_encoding_matches_config_qm9(self):
        """Explicit bins with same params as QM9 config produce identical output."""
        config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
        hn_config = HyperNet(config)

        hn_explicit = HyperNet(
            feature_bins=[4, 5, 3, 5],
            hv_dim=256,
            seed=42,
            hypernet_depth=3,
            normalize=True,
            dtype="float64",
            device="cpu",
        )

        # Methane: single carbon atom
        x = torch.tensor([[0, 0, 0, 3]])
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        batch = _make_batch(x, edge_index)

        out_config = hn_config.forward(batch, normalize=True)
        out_explicit = hn_explicit.forward(batch, normalize=True)

        torch.testing.assert_close(
            out_config["graph_embedding"],
            out_explicit["graph_embedding"],
        )

    def test_encoding_matches_config_zinc(self):
        """Explicit bins with same params as ZINC config produce identical output."""
        config = get_config("ZINC_SMILES_HRR_256_F64_5G1NG4")
        hn_config = HyperNet(config)

        hn_explicit = HyperNet(
            feature_bins=[9, 6, 3, 4, 2],
            hv_dim=256,
            seed=42,
            hypernet_depth=4,
            normalize=True,
            dtype="float64",
            device="cpu",
        )

        # Single atom with features in ZINC format
        x = torch.tensor([[1, 1, 0, 2, 1]])
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        batch = _make_batch(x, edge_index)

        out_config = hn_config.forward(batch, normalize=True)
        out_explicit = hn_explicit.forward(batch, normalize=True)

        torch.testing.assert_close(
            out_config["graph_embedding"],
            out_explicit["graph_embedding"],
        )


# ===================================================================
# Synthetic graph tests (domain-agnostic)
# ===================================================================

class TestSyntheticGraphs:

    def test_two_feature_graph(self):
        """Encode a graph with feature_bins=[3, 4] (color + size)."""
        hn = HyperNet(feature_bins=[3, 4], seed=1)

        # Two connected nodes with different features
        x = torch.tensor([[0, 3], [2, 1]])
        edge_index = torch.tensor([[0, 1], [1, 0]])
        batch = _make_batch(x, edge_index)

        out = hn.forward(batch)
        assert out["graph_embedding"].shape == (1, 256)

    def test_multi_feature_graph(self):
        """Encode with feature_bins=[3, 2] (color + shape)."""
        hn = HyperNet(feature_bins=[3, 2], seed=1)

        x = torch.tensor([[0, 1], [2, 0]])
        edge_index = torch.tensor([[0, 1], [1, 0]])
        batch = _make_batch(x, edge_index)

        out = hn.forward(batch)
        assert out["graph_embedding"].shape == (1, 256)

    def test_deterministic_with_seed(self):
        """Same seed produces identical output."""
        x = torch.tensor([[1, 0], [0, 1]])
        edge_index = torch.tensor([[0, 1], [1, 0]])
        batch = _make_batch(x, edge_index)

        hn1 = HyperNet(feature_bins=[3, 2], seed=42)
        hn2 = HyperNet(feature_bins=[3, 2], seed=42)

        out1 = hn1.forward(batch)
        out2 = hn2.forward(batch)
        torch.testing.assert_close(out1["graph_embedding"], out2["graph_embedding"])


# ===================================================================
# Domain integration tests
# ===================================================================

class TestFromDomain:

    def test_from_domain_qm9(self):
        from graph_hdc.domains.molecular.domain import QM9MolecularDomain

        hn = HyperNet.from_domain(QM9MolecularDomain())
        assert hn.feature_bins == [4, 5, 3, 5]

    def test_from_domain_zinc(self):
        from graph_hdc.domains.molecular.domain import ZINCMolecularDomain

        hn = HyperNet.from_domain(ZINCMolecularDomain())
        assert hn.feature_bins == [9, 6, 3, 4, 2]

    def test_from_domain_with_kwargs(self):
        from graph_hdc.domains.molecular.domain import QM9MolecularDomain

        hn = HyperNet.from_domain(QM9MolecularDomain(), hv_dim=128, seed=7)
        assert hn.hv_dim == 128
        assert hn.seed == 7
        assert hn.feature_bins == [4, 5, 3, 5]

    def test_from_domain_custom(self):
        """from_domain works with any object that has feature_bins."""
        from graph_hdc.domains.base import GraphDomain, DomainResult, OneHotEncoder

        class TinyDomain(GraphDomain):
            feature_schema: ClassVar[dict[str, Any]] = {
                "color": OneHotEncoder(["R", "G", "B"]),
            }

            def process(self, data):
                import networkx as nx
                return nx.Graph()

            def unprocess(self, graph):
                return DomainResult(domain_object=None, is_valid=False, canonical_key=None)

            def visualize(self, ax, obj, **kwargs):
                pass

        hn = HyperNet.from_domain(TinyDomain())
        assert hn.feature_bins == [3]


# ===================================================================
# Save/load tests
# ===================================================================

class TestSaveLoad:

    def test_save_load_roundtrip(self):
        hn = HyperNet(feature_bins=[4, 5, 3, 5], seed=42)
        x = torch.tensor([[0, 0, 0, 3]])
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        batch = _make_batch(x, edge_index)

        out_before = hn.forward(batch)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            hn.save(path)
            hn_loaded = HyperNet.load(path)
            out_after = hn_loaded.forward(batch)
            torch.testing.assert_close(
                out_before["graph_embedding"],
                out_after["graph_embedding"],
            )
        finally:
            Path(path).unlink()

    def test_save_load_preserves_feature_bins(self):
        hn = HyperNet(feature_bins=[3, 2], seed=1)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            hn.save(path)
            hn_loaded = HyperNet.load(path)
            assert hn_loaded.feature_bins == [3, 2]
        finally:
            Path(path).unlink()

    def test_save_load_base_dataset_none(self):
        hn = HyperNet(feature_bins=[4, 5, 3, 5], seed=42)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            hn.save(path)
            hn_loaded = HyperNet.load(path)
            assert hn_loaded.base_dataset is None
        finally:
            Path(path).unlink()

    def test_save_includes_feature_bins_key(self):
        """The saved state includes the feature_bins key for inspection."""
        hn = HyperNet(feature_bins=[4, 5, 3, 5], seed=42)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            hn.save(path)
            state = torch.load(path, map_location="cpu", weights_only=False)
            assert state["config"]["feature_bins"] == [4, 5, 3, 5]
            assert state["config"]["base_dataset"] is None
        finally:
            Path(path).unlink()


# ===================================================================
# Pruning tests
# ===================================================================

class TestPruning:

    def test_prune_with_observed_features(self):
        observed = {(0, 0, 0, 0), (1, 1, 1, 1), (2, 2, 2, 2)}
        hn = HyperNet(
            feature_bins=[4, 5, 3, 5],
            seed=42,
            prune_codebook=True,
            observed_node_features=observed,
        )
        assert hn.nodes_codebook.shape[0] == 3

    def test_prune_without_observed_raises(self):
        with pytest.raises(ValueError, match="prune_codebook=True requires"):
            HyperNet(
                feature_bins=[4, 5, 3, 5],
                seed=42,
                prune_codebook=True,
            )
