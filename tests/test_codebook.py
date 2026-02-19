"""
Tests for node and edge codebook initialization and limitation.

This module verifies that:
1. The node codebook matches the expected size for the config (full or pruned)
2. The edge codebook has a quadratic relationship with the node codebook
3. Encoding and decoding works correctly with the codebooks
"""

import math

import pytest
import torch

from graph_hdc.hypernet.configs import get_config, DecoderSettings, Features
from graph_hdc.hypernet.encoder import HyperNet
from graph_hdc.datasets.utils import get_split, get_dataset_info


def _get_node_bins(config):
    """Extract node feature bins from a DSHDCConfig."""
    return next(iter(config.node_feature_configs.values())).bins

# force_cpu fixture is now in conftest.py


class TestQM9Codebook:
    """Tests for QM9 dataset codebook initialization."""

    @pytest.fixture
    def qm9_config(self):
        config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
        config.device = "cpu"
        return config

    @pytest.fixture
    def qm9_hypernet(self, qm9_config):
        return HyperNet(qm9_config)

    def test_node_codebook_size_matches_config(self, qm9_hypernet, qm9_config):
        """Verify node codebook size is consistent with prune_codebook setting."""
        full_size = math.prod(_get_node_bins(qm9_config))

        actual_size = qm9_hypernet.nodes_codebook.shape[0]
        if qm9_config.prune_codebook:
            assert actual_size < full_size, (
                f"Node codebook should be pruned. Full: {full_size}, Actual: {actual_size}"
            )
        else:
            assert actual_size == full_size, (
                f"Node codebook should be full Cartesian product. "
                f"Full: {full_size}, Actual: {actual_size}"
            )

        # Verify shape is [size, hv_dim]
        assert qm9_hypernet.nodes_codebook.shape[1] == qm9_config.hv_dim

    def test_node_codebook_contains_dataset_features(self, qm9_hypernet):
        """Verify all node features from the dataset are represented in codebook."""
        dataset_info = get_dataset_info("qm9")

        for node_tuple in dataset_info.node_features:
            assert node_tuple in qm9_hypernet.nodes_indexer.tuple_to_idx, (
                f"Node feature {node_tuple} from qm9 dataset not in indexer"
            )

    def test_nodes_indexer_consistency(self, qm9_hypernet):
        """Verify nodes indexer is consistent with codebook."""
        codebook_size = qm9_hypernet.nodes_codebook.shape[0]
        indexer_size = qm9_hypernet.nodes_indexer.size()

        assert codebook_size == indexer_size, (
            f"Codebook size {codebook_size} must equal indexer size {indexer_size}"
        )

        # Verify indexer can round-trip all indices
        for idx in range(indexer_size):
            node_tuple = qm9_hypernet.nodes_indexer.get_tuple(idx)
            recovered_idx = qm9_hypernet.nodes_indexer.get_idx(node_tuple)
            assert idx == recovered_idx, f"Index {idx} → {node_tuple} → {recovered_idx}"

    def test_edge_codebook_quadratic_relation(self, qm9_hypernet, qm9_config):
        """Verify edge codebook has quadratic relationship with node codebook."""
        node_count = qm9_hypernet.nodes_codebook.shape[0]
        expected_edge_count = node_count * node_count

        # Edges codebook shape should be [N^2, hv_dim]
        assert qm9_hypernet.edges_codebook.shape == (expected_edge_count, qm9_config.hv_dim), (
            f"Expected edges codebook shape ({expected_edge_count}, {qm9_config.hv_dim}), "
            f"got {qm9_hypernet.edges_codebook.shape}"
        )

    def test_edges_indexer_consistency(self, qm9_hypernet):
        """Verify edges indexer is consistent with edge codebook."""
        edge_codebook_size = qm9_hypernet.edges_codebook.shape[0]
        edge_indexer_size = qm9_hypernet.edges_indexer.size()

        assert edge_codebook_size == edge_indexer_size, (
            f"Edge codebook size {edge_codebook_size} must equal edge indexer size {edge_indexer_size}"
        )

        # Verify indexer dimensions match node indexer
        node_count = qm9_hypernet.nodes_indexer.size()
        assert edge_indexer_size == node_count * node_count

    def test_encode_decode_sample(self, qm9_hypernet):
        """Test encoding and decoding a sample from QM9 dataset."""
        train_ds = get_split("train", dataset="qm9")
        data = train_ds[0].clone()
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device="cpu")

        # Encode
        with torch.no_grad():
            output = qm9_hypernet.forward(data, normalize=True)

        # Verify encoding output
        assert "graph_embedding" in output
        assert "edge_terms" in output
        assert output["graph_embedding"].shape == (1, qm9_hypernet.hv_dim)
        assert output["edge_terms"].shape == (1, qm9_hypernet.hv_dim)

        # Decode
        decoder_settings = DecoderSettings.get_default_for("qm9")
        decoder_settings.top_k = 1
        result = qm9_hypernet.decode_graph(
            edge_term=output["edge_terms"][0],
            graph_term=output["graph_embedding"][0],
            decoder_settings=decoder_settings,
        )

        # Verify decoding returns valid result
        assert hasattr(result, "nx_graphs")
        assert hasattr(result, "cos_similarities")


class TestCodebookLimitationProcess:
    """Tests verifying the codebook limitation process works correctly."""

    def test_qm9_codebook_size_is_reasonable(self):
        """Sanity check: codebook size should match config expectations."""
        config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
        config.device = "cpu"
        hypernet = HyperNet(config)

        node_size = hypernet.nodes_codebook.shape[0]
        full_size = math.prod(_get_node_bins(config))

        if config.prune_codebook:
            # Pruned: QM9 typically has ~50-60 unique node types
            assert 30 < node_size < 150, f"QM9 node codebook size {node_size} seems unusual"
        else:
            assert node_size == full_size, (
                f"Unpruned codebook should be full Cartesian product ({full_size}), got {node_size}"
            )

    def test_all_dataset_node_features_in_codebook(self):
        """Verify all node features from dataset are represented in codebook."""
        config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
        config.device = "cpu"
        hypernet = HyperNet(config)

        dataset_info = get_dataset_info("qm9")

        # All node features from dataset should be in indexer
        for node_tuple in dataset_info.node_features:
            assert node_tuple in hypernet.nodes_indexer.tuple_to_idx, (
                f"Node feature {node_tuple} from qm9 dataset not in indexer"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
