"""
Tests for MultiHyperNet ensemble encoder.
"""

import tempfile
from collections import Counter
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch, Data

from graph_hdc.hypernet.configs import get_config
from graph_hdc.hypernet.encoder import HyperNet
from graph_hdc.hypernet.multi_hypernet import MultiHyperNet
from graph_hdc.utils.helpers import scatter_hd


# ─────────────────── Helpers ───────────────────


def _make_ethane(device="cpu") -> Data:
    """Create a simple ethane (C-C) PyG Data object."""
    x = torch.tensor(
        [[0.0, 0.0, 0.0, 3.0], [0.0, 0.0, 0.0, 3.0]], device=device
    )
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=device)
    data = Data(x=x, edge_index=edge_index)
    data.batch = torch.zeros(2, dtype=torch.long, device=device)
    return data


def _make_methanol(device="cpu") -> Data:
    """Create methanol (C-O) PyG Data object.

    Features: [atom_type, degree-1, formal_charge, total_Hs]
    C: atom_type=0, degree-1=0, charge=0, Hs=3
    O: atom_type=2, degree-1=0, charge=0, Hs=1
    """
    x = torch.tensor(
        [[0.0, 0.0, 0.0, 3.0], [2.0, 0.0, 0.0, 1.0]], device=device
    )
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=device)
    data = Data(x=x, edge_index=edge_index)
    data.batch = torch.zeros(2, dtype=torch.long, device=device)
    return data


def _make_batch(device="cpu") -> Batch:
    """Create a batch of two molecules."""
    ethane = _make_ethane(device)
    ethane.batch = torch.zeros(2, dtype=torch.long, device=device)

    methanol = _make_methanol(device)
    methanol.batch = torch.zeros(2, dtype=torch.long, device=device)

    return Batch.from_data_list([ethane, methanol])


# ─────────────────── Construction Tests ───────────────────


def test_creation_from_list():
    """Test MultiHyperNet creation from a list of HyperNets."""
    config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
    config.device = "cpu"

    hn1 = HyperNet(config)
    config2 = config
    config2.seed = 123
    hn2 = HyperNet(config2)

    multi = MultiHyperNet([hn1, hn2])

    assert multi.num_hypernets == 2
    assert multi.hv_dim == 256
    assert multi.ensemble_graph_dim == 512
    assert multi.base_dataset == "qm9"


def test_from_config_factory():
    """Test the from_config convenience factory."""
    config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
    config.device = "cpu"

    seeds = [42, 123, 456]
    multi = MultiHyperNet.from_config(config, seeds=seeds)

    assert multi.num_hypernets == 3
    assert multi.hv_dim == 256
    assert multi.ensemble_graph_dim == 3 * 256


def test_creation_requires_nonempty():
    """Test that empty list raises ValueError."""
    with pytest.raises(ValueError, match="at least one"):
        MultiHyperNet([])


def test_mixed_dims():
    """Test that MultiHyperNet supports mixed hv_dim."""
    from dataclasses import replace as dc_replace

    config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
    config.device = "cpu"

    cfg_256 = dc_replace(config, hv_dim=256, seed=42)
    cfg_128 = dc_replace(config, hv_dim=128, seed=99)
    hn_256 = HyperNet(cfg_256)
    hn_128 = HyperNet(cfg_128)

    multi = MultiHyperNet([hn_256, hn_128])
    assert multi.hv_dim == 256  # primary
    assert multi.ensemble_graph_dim == 384  # 256 + 128

    data = _make_ethane()
    with torch.no_grad():
        out = multi.forward(data)
    assert out["graph_embedding"].shape == (1, 384)
    assert out["edge_terms"].shape == (1, 256)  # primary dim


def test_repr():
    """Test string representation."""
    config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
    config.device = "cpu"
    multi = MultiHyperNet.from_config(config, seeds=[42, 99])

    r = repr(multi)
    assert "K=2" in r
    assert "ensemble_graph_dim=512" in r


def test_from_dim_depth_pairs():
    """Test the from_dim_depth_pairs factory."""
    config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
    config.device = "cpu"

    multi = MultiHyperNet.from_dim_depth_pairs(
        config,
        dim_depth_pairs=[(256, 3), (128, 5)],
        base_seed=10,
    )

    assert multi.num_hypernets == 2
    assert multi._hypernets[0].hv_dim == 256
    assert multi._hypernets[0].depth == 3
    assert multi._hypernets[0].seed == 10
    assert multi._hypernets[1].hv_dim == 128
    assert multi._hypernets[1].depth == 5
    assert multi._hypernets[1].seed == 11
    assert multi.ensemble_graph_dim == 384


# ─────────────────── Forward Pass Tests ───────────────────


def test_forward_output_shape_single():
    """Test forward output shape for a single molecule."""
    config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
    config.device = "cpu"
    multi = MultiHyperNet.from_config(config, seeds=[42, 123])

    data = _make_ethane()
    with torch.no_grad():
        out = multi.forward(data)

    assert "graph_embedding" in out
    assert "edge_terms" in out
    assert out["graph_embedding"].shape == (1, 512)  # 2 * 256
    assert out["edge_terms"].shape == (1, 256)  # from primary only


def test_forward_output_shape_triple():
    """Test forward output shape with 3 sub-HyperNets."""
    config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
    config.device = "cpu"
    multi = MultiHyperNet.from_config(config, seeds=[42, 123, 456])

    data = _make_ethane()
    with torch.no_grad():
        out = multi.forward(data)

    assert out["graph_embedding"].shape == (1, 768)  # 3 * 256
    assert out["edge_terms"].shape == (1, 256)


def test_forward_batch():
    """Test forward with a batch of molecules."""
    config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
    config.device = "cpu"
    multi = MultiHyperNet.from_config(config, seeds=[42, 123])

    batch = _make_batch()
    with torch.no_grad():
        out = multi.forward(batch)

    assert out["graph_embedding"].shape == (2, 512)
    assert out["edge_terms"].shape == (2, 256)


def test_forward_does_not_mutate_input():
    """Test that forward doesn't modify the input data's node_hv."""
    config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
    config.device = "cpu"
    multi = MultiHyperNet.from_config(config, seeds=[42, 123])

    data = _make_ethane()
    # Pre-encode with primary to set node_hv
    data = multi.encode_properties(data)
    original_node_hv = data.node_hv.clone()

    with torch.no_grad():
        multi.forward(data)

    # node_hv should be unchanged (forward clones internally)
    assert torch.allclose(data.node_hv, original_node_hv), \
        "forward() should not mutate input data's node_hv"


# ─────────────────── Embedding Quality Tests ───────────────────


def test_different_seeds_produce_different_embeddings():
    """Test that different seeds produce uncorrelated graph embeddings."""
    config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
    config.device = "cpu"
    multi = MultiHyperNet.from_config(config, seeds=[42, 123])

    data = _make_ethane()
    with torch.no_grad():
        out = multi.forward(data)

    embedding = out["graph_embedding"][0]  # (512,)
    part_1 = embedding[:256]
    part_2 = embedding[256:]

    # Cosine similarity between the two perspectives should be low
    cos_sim = torch.nn.functional.cosine_similarity(
        part_1.unsqueeze(0), part_2.unsqueeze(0)
    ).item()
    assert abs(cos_sim) < 0.3, (
        f"Different seeds should produce near-uncorrelated embeddings, "
        f"got cosine similarity {cos_sim:.4f}"
    )


def test_first_partition_matches_primary():
    """Test that the first partition of the ensemble matches the primary HyperNet alone."""
    config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
    config.device = "cpu"

    hn_primary = HyperNet(config)
    multi = MultiHyperNet.from_config(config, seeds=[42, 123])

    data = _make_ethane()
    with torch.no_grad():
        primary_out = hn_primary.forward(data.clone())
        multi_out = multi.forward(data.clone())

    primary_emb = primary_out["graph_embedding"][0]
    multi_first = multi_out["graph_embedding"][0, :256]

    assert torch.allclose(primary_emb, multi_first, atol=1e-5), \
        "First partition of ensemble should match primary HyperNet"


def test_edge_terms_match_primary():
    """Test that edge_terms match primary HyperNet's edge_terms."""
    config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
    config.device = "cpu"

    hn_primary = HyperNet(config)
    multi = MultiHyperNet.from_config(config, seeds=[42, 123])

    data = _make_ethane()
    with torch.no_grad():
        primary_out = hn_primary.forward(data.clone())
        multi_out = multi.forward(data.clone())

    assert torch.allclose(
        primary_out["edge_terms"], multi_out["edge_terms"], atol=1e-5
    ), "edge_terms should match primary"


def test_deterministic_output():
    """Test that MultiHyperNet produces identical output on repeated calls."""
    config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
    config.device = "cpu"
    multi = MultiHyperNet.from_config(config, seeds=[42, 123])

    data = _make_ethane()
    with torch.no_grad():
        out1 = multi.forward(data.clone())
        out2 = multi.forward(data.clone())

    assert torch.allclose(
        out1["graph_embedding"], out2["graph_embedding"], atol=1e-6
    ), "Repeated calls should produce identical output"


def test_different_molecules_produce_different_embeddings():
    """Test that different molecules produce distinct ensemble embeddings."""
    config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
    config.device = "cpu"
    multi = MultiHyperNet.from_config(config, seeds=[42, 123])

    ethane = _make_ethane()
    methanol = _make_methanol()

    with torch.no_grad():
        out_ethane = multi.forward(ethane)
        out_methanol = multi.forward(methanol)

    emb_e = out_ethane["graph_embedding"][0]
    emb_m = out_methanol["graph_embedding"][0]

    cos_sim = torch.nn.functional.cosine_similarity(
        emb_e.unsqueeze(0), emb_m.unsqueeze(0)
    ).item()
    assert cos_sim < 0.95, (
        f"Different molecules should have distinguishable embeddings, "
        f"got cosine similarity {cos_sim:.4f}"
    )


# ─────────────────── encode_properties Tests ───────────────────


def test_encode_properties_uses_primary():
    """Test that encode_properties delegates to the primary HyperNet."""
    config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
    config.device = "cpu"

    hn_primary = HyperNet(config)
    multi = MultiHyperNet.from_config(config, seeds=[42, 123])

    data_primary = _make_ethane()
    data_multi = _make_ethane()

    with torch.no_grad():
        hn_primary.encode_properties(data_primary)
        multi.encode_properties(data_multi)

    assert torch.allclose(data_primary.node_hv, data_multi.node_hv, atol=1e-6), \
        "encode_properties should use the primary HyperNet"


# ─────────────────── Decoding Delegation Tests ───────────────────


def test_decode_order_zero_delegates():
    """Test decode_order_zero delegates to primary."""
    config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
    config.device = "cpu"

    hn_primary = HyperNet(config)
    multi = MultiHyperNet.from_config(config, seeds=[42, 123])

    data = _make_ethane()
    with torch.no_grad():
        data = hn_primary.encode_properties(data)
        order_zero = scatter_hd(src=data.node_hv, index=data.batch, op="bundle")

        primary_decoded = hn_primary.decode_order_zero(order_zero[0])
        multi_decoded = multi.decode_order_zero(order_zero[0])

    assert torch.equal(primary_decoded, multi_decoded), \
        "decode_order_zero should produce identical results to primary"


def test_decode_order_zero_iterative_delegates():
    """Test decode_order_zero_iterative delegates to primary."""
    config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
    config.device = "cpu"

    hn_primary = HyperNet(config)
    multi = MultiHyperNet.from_config(config, seeds=[42, 123])

    data = _make_ethane()
    with torch.no_grad():
        data = hn_primary.encode_properties(data)
        order_zero = scatter_hd(src=data.node_hv, index=data.batch, op="bundle")

        primary_nodes = hn_primary.decode_order_zero_iterative(order_zero[0])
        multi_nodes = multi.decode_order_zero_iterative(order_zero[0])

    assert primary_nodes == multi_nodes, \
        "decode_order_zero_iterative should produce identical results"


# ─────────────────── Property Delegation Tests ───────────────────


def test_codebook_properties_delegate():
    """Test that codebook properties delegate to primary."""
    config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
    config.device = "cpu"

    hn_primary = HyperNet(config)
    multi = MultiHyperNet([hn_primary])

    assert multi.nodes_codebook is hn_primary.nodes_codebook
    assert multi.nodes_indexer is hn_primary.nodes_indexer
    assert multi.edges_codebook is hn_primary.edges_codebook
    assert multi.edges_indexer is hn_primary.edges_indexer
    assert multi.base_dataset == hn_primary.base_dataset
    assert multi.vsa == hn_primary.vsa
    assert multi.depth == hn_primary.depth
    assert multi.normalize == hn_primary.normalize


# ─────────────────── Integration with preprocess workflow ───────────────────


def test_preprocess_workflow_compatibility():
    """
    Test that MultiHyperNet works in the same workflow as
    preprocess_for_flow_edge_decoder (encode_properties → order_zero → forward → order_n).
    """
    config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
    config.device = "cpu"
    multi = MultiHyperNet.from_config(config, seeds=[42, 123, 456])

    data = _make_ethane()
    data_for_hdc = data.clone()

    with torch.no_grad():
        # Step 1: encode_properties (from primary)
        data_for_hdc = multi.encode_properties(data_for_hdc)
        assert hasattr(data_for_hdc, "node_hv")
        assert data_for_hdc.node_hv.shape == (2, 256)

        # Step 2: order_zero from primary's node_hv
        order_zero = scatter_hd(
            src=data_for_hdc.node_hv, index=data_for_hdc.batch, op="bundle"
        )
        assert order_zero.shape == (1, 256)

        # Step 3: forward for ensemble order_N
        hdc_out = multi.forward(data_for_hdc)
        order_n = hdc_out["graph_embedding"]
        assert order_n.shape == (1, 768)  # 3 * 256

        # Step 4: concatenate [order_0 | order_N]
        hdc_concat = torch.cat([order_zero, order_n], dim=-1)
        assert hdc_concat.shape == (1, 256 + 768)  # 1024 total


# ─────────────────── Save / Load Tests ───────────────────


def test_save_load_roundtrip():
    """Test that save → load produces identical outputs."""
    config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
    config.device = "cpu"
    original = MultiHyperNet.from_config(config, seeds=[42, 123])

    data = _make_ethane()
    with torch.no_grad():
        original_out = original.forward(data.clone())

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "multi_hypernet.pt"
        original.save(save_path)

        loaded = MultiHyperNet.load(save_path, device="cpu")

    assert loaded.num_hypernets == original.num_hypernets
    assert loaded.hv_dim == original.hv_dim
    assert loaded.ensemble_graph_dim == original.ensemble_graph_dim
    assert loaded.base_dataset == original.base_dataset

    with torch.no_grad():
        loaded_out = loaded.forward(data.clone())

    assert torch.allclose(
        original_out["graph_embedding"],
        loaded_out["graph_embedding"],
        atol=1e-5,
    ), "graph_embedding mismatch after save/load"

    assert torch.allclose(
        original_out["edge_terms"],
        loaded_out["edge_terms"],
        atol=1e-5,
    ), "edge_terms mismatch after save/load"


def test_save_load_preserves_codebooks():
    """Test that save/load preserves all sub-HyperNet codebooks."""
    config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
    config.device = "cpu"
    original = MultiHyperNet.from_config(config, seeds=[42, 123])

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "multi_hypernet.pt"
        original.save(save_path)
        loaded = MultiHyperNet.load(save_path, device="cpu")

    for i, (orig_hn, load_hn) in enumerate(
        zip(original._hypernets, loaded._hypernets)
    ):
        assert torch.allclose(
            orig_hn.nodes_codebook, load_hn.nodes_codebook, atol=1e-6
        ), f"HyperNet {i}: nodes_codebook mismatch"
        assert torch.allclose(
            orig_hn.edges_codebook, load_hn.edges_codebook, atol=1e-6
        ), f"HyperNet {i}: edges_codebook mismatch"
        assert orig_hn.seed == load_hn.seed, f"HyperNet {i}: seed mismatch"


def test_load_rejects_non_multi_checkpoint():
    """Test that load rejects a file that isn't a MultiHyperNet."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_path = Path(tmpdir) / "bad.pt"
        torch.save({"type": "SomethingElse"}, bad_path)

        with pytest.raises(ValueError, match="not a MultiHyperNet"):
            MultiHyperNet.load(bad_path)


# ─────────────────── K=1 edge case ───────────────────


def test_single_hypernet_matches_original():
    """Test that K=1 MultiHyperNet produces same output as plain HyperNet."""
    config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
    config.device = "cpu"

    hn = HyperNet(config)
    multi = MultiHyperNet([hn])

    data = _make_ethane()
    with torch.no_grad():
        hn_out = hn.forward(data.clone())
        multi_out = multi.forward(data.clone())

    assert multi_out["graph_embedding"].shape == (1, 256)
    assert torch.allclose(
        hn_out["graph_embedding"], multi_out["graph_embedding"], atol=1e-6
    ), "K=1 ensemble should match plain HyperNet exactly"
    assert torch.allclose(
        hn_out["edge_terms"], multi_out["edge_terms"], atol=1e-6
    )


# ─────────────────── load_hypernet dispatch ───────────────────


def test_load_hypernet_dispatches_to_hypernet():
    """load_hypernet returns a HyperNet when given a HyperNet checkpoint."""
    from graph_hdc.hypernet import load_hypernet

    config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
    config.device = "cpu"
    hn = HyperNet(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "hn.ckpt"
        hn.save(path)
        loaded = load_hypernet(path, device="cpu")

    assert isinstance(loaded, HyperNet)
    assert not isinstance(loaded, MultiHyperNet)
    assert loaded.hv_dim == hn.hv_dim


def test_load_hypernet_dispatches_to_multi():
    """load_hypernet returns a MultiHyperNet when given a MultiHyperNet checkpoint."""
    from graph_hdc.hypernet import load_hypernet

    config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
    config.device = "cpu"
    multi = MultiHyperNet.from_config(config, seeds=[42, 123])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "multi.ckpt"
        multi.save(path)
        loaded = load_hypernet(path, device="cpu")

    assert isinstance(loaded, MultiHyperNet)
    assert loaded.num_hypernets == 2


def test_load_hypernet_produces_identical_output():
    """load_hypernet produces encoders with identical outputs to direct load."""
    from graph_hdc.hypernet import load_hypernet

    config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
    config.device = "cpu"
    multi = MultiHyperNet.from_config(config, seeds=[42, 123])

    data = _make_ethane()
    with torch.no_grad():
        original_out = multi.forward(data.clone())

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "multi.ckpt"
        multi.save(path)
        loaded = load_hypernet(path, device="cpu")

    with torch.no_grad():
        loaded_out = loaded.forward(data.clone())

    assert torch.allclose(
        original_out["graph_embedding"],
        loaded_out["graph_embedding"],
        atol=1e-6,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
