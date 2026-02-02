"""
Smoke tests for HDC encoding functionality.
"""
import tempfile
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from graph_hdc.hypernet.configs import get_config
from graph_hdc.hypernet.encoder import HyperNet


def test_get_config():
    """Test configuration retrieval."""

    qm9_config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
    assert qm9_config is not None
    assert qm9_config.hv_dim == 256
    assert qm9_config.base_dataset == "qm9"

    zinc_config = get_config("ZINC_SMILES_HRR_256_F64_5G1NG4")
    assert zinc_config is not None
    assert zinc_config.hv_dim == 256
    assert zinc_config.base_dataset == "zinc"


def test_hypernet_creation():
    """Test HyperNet encoder creation."""
    config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
    # Ensure CPU device
    config.device = "cpu"
    hypernet = HyperNet(config)

    assert hypernet is not None
    assert hypernet.hv_dim == 256
    assert hypernet.base_dataset == "qm9"


def test_molecule_encoding():
    """Test encoding a simple molecule (ethane: C-C)."""

    config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
    config.device = "cpu"
    hypernet = HyperNet(config)

    # Create a simple PyG graph (ethane: C-C)
    # Two carbons with degree 1, each with 3 hydrogens
    # Features: [atom_type, degree-1, formal_charge, total_Hs]
    x = torch.tensor([
        [0.0, 0.0, 0.0, 3.0],  # C with degree 1
        [0.0, 0.0, 0.0, 3.0],  # C with degree 1
    ], device="cpu")
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device="cpu")

    data = Data(x=x, edge_index=edge_index)
    data.batch = torch.zeros(2, dtype=torch.long, device="cpu")

    # Encode
    with torch.no_grad():
        output = hypernet.forward(data, normalize=True)

    assert "graph_embedding" in output
    assert "edge_terms" in output
    assert output["graph_embedding"].shape == (1, 256)
    assert output["edge_terms"].shape == (1, 256)


def test_decode_order_zero():
    """Test order-zero decoding (node retrieval)."""

    config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
    config.device = "cpu"
    hypernet = HyperNet(config)

    # Ethane: C-C
    # Two carbons with degree 1, each with 3 hydrogens
    x = torch.tensor([
        [0.0, 0.0, 0.0, 3.0],  # C with degree 1
        [0.0, 0.0, 0.0, 3.0],  # C with degree 1
    ], device="cpu")
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device="cpu")

    data = Data(x=x, edge_index=edge_index)
    data.batch = torch.zeros(2, dtype=torch.long, device="cpu")

    with torch.no_grad():
        output = hypernet.forward(data, normalize=False)
        graph_embedding = output["graph_embedding"]

        # Decode nodes - returns similarity scores (one per node type in codebook)
        # We use graph_embedding which aggregates all node information
        decoded_nodes = hypernet.decode_order_zero(graph_embedding[0])

    # decode_order_zero returns similarity scores for all possible node types
    # The output should be a tensor with shape matching the node codebook size
    assert decoded_nodes is not None
    assert isinstance(decoded_nodes, torch.Tensor)


def test_decode_order_zero_iterative():
    """Test order-zero iterative decoding (node retrieval via unbinding)."""
    from collections import Counter
    from graph_hdc.utils.helpers import scatter_hd

    config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
    config.device = "cpu"
    hypernet = HyperNet(config)

    # Ethane: C-C
    # Two carbons with degree 1, each with 3 hydrogens
    # Features: [atom_type, degree-1, formal_charge, total_Hs]
    x = torch.tensor([
        [0.0, 0.0, 0.0, 3.0],  # C with degree 1 (degree_idx=0), 3 H
        [0.0, 0.0, 0.0, 3.0],  # C with degree 1 (degree_idx=0), 3 H
    ], device="cpu")
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device="cpu")

    data = Data(x=x, edge_index=edge_index)
    data.batch = torch.zeros(2, dtype=torch.long, device="cpu")

    with torch.no_grad():
        # Encode node features to hypervectors (without message passing)
        data = hypernet.encode_properties(data)
        # Compute pure order-0 embedding (bundled nodes, no message passing)
        order_zero_embedding = scatter_hd(src=data.node_hv, index=data.batch, op="bundle")

        # Decode using iterative method
        decoded_nodes = hypernet.decode_order_zero_iterative(order_zero_embedding[0])

    # Should decode exactly 2 carbon nodes
    assert len(decoded_nodes) == 2, f"Expected 2 nodes, got {len(decoded_nodes)}"

    # Both should be the same node type: (atom=0, degree=0, charge=0, H=3)
    expected_tuple = (0, 0, 0, 3)
    node_counter = Counter(decoded_nodes)
    assert node_counter[expected_tuple] == 2, f"Expected 2 carbons with features {expected_tuple}, got {node_counter}"


def test_decode_order_zero_iterative_counter():
    """Test order-zero iterative counter decoding matches original method."""
    from collections import Counter
    from graph_hdc.utils.helpers import scatter_hd

    config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
    config.device = "cpu"
    hypernet = HyperNet(config)

    # Ethane: C-C (use same molecule as other tests since node types are limited)
    # Two carbons with degree 1, each with 3 hydrogens
    # Features: [atom_type, degree-1, formal_charge, total_Hs]
    x = torch.tensor([
        [0.0, 0.0, 0.0, 3.0],  # C with degree 1 (degree_idx=0), 3 H
        [0.0, 0.0, 0.0, 3.0],  # C with degree 1 (degree_idx=0), 3 H
    ], device="cpu")
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device="cpu")

    data = Data(x=x, edge_index=edge_index)
    data.batch = torch.zeros(2, dtype=torch.long, device="cpu")

    with torch.no_grad():
        # Encode node features to hypervectors (without message passing)
        data = hypernet.encode_properties(data)
        # Compute pure order-0 embedding (bundled nodes, no message passing)
        order_zero_embedding = scatter_hd(src=data.node_hv, index=data.batch, op="bundle")

        # Decode using both methods
        original_counter = hypernet.decode_order_zero_counter(order_zero_embedding)
        iterative_counter = hypernet.decode_order_zero_counter_iterative(order_zero_embedding)

    # Both methods should produce results for batch index 0
    assert 0 in original_counter, "Original method should have batch 0"
    assert 0 in iterative_counter, "Iterative method should have batch 0"

    # Ground truth: {(0, 0, 0, 3): 2} - two carbons with same features
    expected = Counter({(0, 0, 0, 3): 2})

    # Both methods should match ground truth
    original_result = original_counter[0]
    iterative_result = iterative_counter[0]

    assert original_result == expected, f"Original: expected {expected}, got {original_result}"
    assert iterative_result == expected, f"Iterative: expected {expected}, got {iterative_result}"



def test_hypernet_save_load():
    """Test HyperNet save and load preserves all codebooks exactly."""

    # Create HyperNet from config
    config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
    config.device = "cpu"
    original = HyperNet(config)

    # Save to temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "hypernet.pt"
        original.save(save_path)

        # Load from file
        loaded = HyperNet.load(save_path, device="cpu")

        # ─────────────────────────── Config attributes ───────────────────────────
        assert loaded.hv_dim == original.hv_dim, "hv_dim mismatch"
        assert loaded.depth == original.depth, "depth mismatch"
        assert loaded.vsa == original.vsa, "vsa mismatch"
        assert loaded.seed == original.seed, "seed mismatch"
        assert loaded.normalize == original.normalize, "normalize mismatch"
        assert loaded.base_dataset == original.base_dataset, "base_dataset mismatch"
        assert loaded._dtype == original._dtype, "dtype mismatch"

        # ─────────────────────────── Nodes codebook ───────────────────────────
        assert loaded.nodes_codebook.shape == original.nodes_codebook.shape, "nodes_codebook shape mismatch"
        assert torch.allclose(loaded.nodes_codebook, original.nodes_codebook, atol=1e-6), \
            "nodes_codebook values mismatch"

        # ─────────────────────────── Edges codebook ───────────────────────────
        assert loaded.edges_codebook.shape == original.edges_codebook.shape, "edges_codebook shape mismatch"
        assert torch.allclose(loaded.edges_codebook, original.edges_codebook, atol=1e-6), \
            "edges_codebook values mismatch"

        # ─────────────────────────── Indexers ───────────────────────────
        assert loaded.nodes_indexer.size() == original.nodes_indexer.size(), "nodes_indexer size mismatch"
        assert loaded.edges_indexer.size() == original.edges_indexer.size(), "edges_indexer size mismatch"

        # ─────────────────────────── Node encoder map ───────────────────────────
        assert set(loaded.node_encoder_map.keys()) == set(original.node_encoder_map.keys()), \
            "node_encoder_map keys mismatch"

        for feat in original.node_encoder_map:
            orig_enc, orig_range = original.node_encoder_map[feat]
            load_enc, load_range = loaded.node_encoder_map[feat]

            assert load_range == orig_range, f"Index range mismatch for {feat}"
            assert load_enc.dim == orig_enc.dim, f"Encoder dim mismatch for {feat}"
            assert load_enc.vsa == orig_enc.vsa, f"Encoder vsa mismatch for {feat}"
            assert load_enc.num_categories == orig_enc.num_categories, f"Encoder num_categories mismatch for {feat}"
            assert torch.allclose(load_enc.codebook, orig_enc.codebook, atol=1e-6), \
                f"Encoder codebook mismatch for {feat}"

        # ─────────────────────────── Functional test ───────────────────────────
        # Encode a molecule with both and verify same output
        x = torch.tensor([
            [0.0, 0.0, 0.0, 3.0],  # C with degree 1
            [0.0, 0.0, 0.0, 3.0],  # C with degree 1
        ], device="cpu")
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device="cpu")
        data = Data(x=x, edge_index=edge_index)
        data.batch = torch.zeros(2, dtype=torch.long, device="cpu")

        with torch.no_grad():
            orig_output = original.forward(data, normalize=True)
            load_output = loaded.forward(data, normalize=True)

        assert torch.allclose(orig_output["graph_embedding"], load_output["graph_embedding"], atol=1e-6), \
            "graph_embedding output mismatch after load"
        assert torch.allclose(orig_output["edge_terms"], load_output["edge_terms"], atol=1e-6), \
            "edge_terms output mismatch after load"


def test_hypernet_deterministic_initialization():
    """Test that HyperNet initialization is deterministic with same seed."""

    config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
    config.device = "cpu"
    config.seed = 42

    # Create two HyperNets with same seed
    hypernet1 = HyperNet(config)
    hypernet2 = HyperNet(config)

    # They should have identical codebooks
    assert torch.allclose(hypernet1.nodes_codebook, hypernet2.nodes_codebook, atol=1e-6), \
        "Same seed should produce identical nodes_codebook"

    # The node encoder codebooks should also match
    for feat in hypernet1.node_encoder_map:
        enc1, _ = hypernet1.node_encoder_map[feat]
        enc2, _ = hypernet2.node_encoder_map[feat]
        assert torch.allclose(enc1.codebook, enc2.codebook, atol=1e-6), \
            f"Same seed should produce identical encoder codebook for {feat}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
