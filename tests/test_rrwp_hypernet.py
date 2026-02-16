"""Tests for RRWPHyperNet — split RRWP encoding variant."""

import math
import tempfile
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch, Data

from graph_hdc.hypernet.configs import (
    RWConfig,
    create_config_with_rw,
)
from graph_hdc.hypernet.encoder import HyperNet
from graph_hdc.hypernet.rrwp_hypernet import RRWPHyperNet
from graph_hdc.utils.rw_features import augment_data_with_rw


# ── Helpers ──────────────────────────────────────────────────────────

QM9_BASE_BINS = [4, 5, 3, 5]
QM9_K_VALUES = (2, 4)
QM9_NUM_BINS = 5


def _qm9_rw_config():
    return RWConfig(enabled=True, k_values=QM9_K_VALUES, num_bins=QM9_NUM_BINS)


def _qm9_rrwp_config(hv_dim=256):
    return create_config_with_rw("qm9", hv_dim=hv_dim, rw_config=_qm9_rw_config())


def _qm9_base_config(hv_dim=256):
    """Base QM9 config without RW features (same seed)."""
    return create_config_with_rw("qm9", hv_dim=hv_dim, rw_config=RWConfig())


def _make_molecule(base_features, edge_index, k_values=QM9_K_VALUES, num_bins=QM9_NUM_BINS):
    """Create a Data object with RW features appended."""
    x = torch.tensor(base_features, dtype=torch.float)
    ei = torch.tensor(edge_index, dtype=torch.long)
    data = Data(x=x, edge_index=ei)
    data.batch = torch.zeros(x.size(0), dtype=torch.long)
    data = augment_data_with_rw(data, k_values=k_values, num_bins=num_bins)
    return data


def _ethane():
    """Ethane-like: two carbon nodes connected."""
    return _make_molecule(
        base_features=[[0, 0, 0, 3], [0, 0, 0, 3]],
        edge_index=[[0, 1], [1, 0]],
    )


def _methanol():
    """Methanol-like: C-O."""
    return _make_molecule(
        base_features=[[0, 0, 0, 3], [2, 0, 0, 1]],
        edge_index=[[0, 1], [1, 0]],
    )


def _propane():
    """Propane-like: C-C-C (3-node path)."""
    return _make_molecule(
        base_features=[[0, 0, 0, 3], [0, 1, 0, 2], [0, 0, 0, 3]],
        edge_index=[[0, 1, 1, 2], [1, 0, 2, 1]],
    )


# ── Construction Tests ───────────────────────────────────────────────


class TestCreation:

    def test_creation_with_rw_enabled(self):
        config = _qm9_rrwp_config()
        net = RRWPHyperNet(config)
        assert isinstance(net, RRWPHyperNet)
        assert isinstance(net, HyperNet)

    def test_creation_fails_without_rw(self):
        config = _qm9_base_config()
        with pytest.raises(ValueError, match="rw_config.enabled"):
            RRWPHyperNet(config)

    def test_both_codebooks_exist(self):
        net = RRWPHyperNet(_qm9_rrwp_config())
        assert hasattr(net, "nodes_codebook")
        assert hasattr(net, "nodes_codebook_full")

    def test_codebook_dimensions(self):
        config = _qm9_rrwp_config()
        net = RRWPHyperNet(config, depth=1)
        # Base: prod([4,5,3,5]) = 300 (before pruning)
        # Full: prod([4,5,3,5,5,5]) = 7500 (before pruning)
        assert net.nodes_codebook.shape[1] == config.hv_dim
        assert net.nodes_codebook_full.shape[1] == config.hv_dim

    def test_codebook_shapes_differ(self):
        """Full codebook should be larger than base due to RRWP dims."""
        net = RRWPHyperNet(_qm9_rrwp_config(), depth=1)
        # Without pruning, full > base because of extra RRWP categories
        assert net.nodes_codebook_full.shape[0] != net.nodes_codebook.shape[0]

    def test_both_indexers_exist(self):
        net = RRWPHyperNet(_qm9_rrwp_config())
        assert hasattr(net, "nodes_indexer")
        assert hasattr(net, "nodes_indexer_full")

    def test_rw_config_preserved(self):
        rw = _qm9_rw_config()
        net = RRWPHyperNet(create_config_with_rw("qm9", hv_dim=256, rw_config=rw))
        assert net.rw_config.enabled is True
        assert net.rw_config.k_values == QM9_K_VALUES
        assert net.rw_config.num_bins == QM9_NUM_BINS

    def test_num_rw_dims(self):
        net = RRWPHyperNet(_qm9_rrwp_config())
        assert net._num_rw_dims == len(QM9_K_VALUES)

    def test_creation_no_pruning(self):
        config = _qm9_rrwp_config()
        config.prune_codebook = False
        net = RRWPHyperNet(config)
        # Without pruning, sizes are the full Cartesian product
        base_size = math.prod(QM9_BASE_BINS)
        full_size = math.prod(QM9_BASE_BINS + [QM9_NUM_BINS] * len(QM9_K_VALUES))
        assert net.nodes_codebook.shape[0] == base_size
        assert net.nodes_codebook_full.shape[0] == full_size

    def test_creation_with_observed_features(self):
        config = _qm9_rrwp_config()
        config.prune_codebook = True
        # Two full tuples: (atom, degree, charge, H, rw1, rw2)
        observed = {(0, 0, 0, 3, 2, 1), (2, 0, 0, 1, 3, 0)}
        net = RRWPHyperNet(config, observed_node_features=observed)
        assert net.nodes_codebook_full.shape[0] == 2
        # Base tuples: {(0,0,0,3), (2,0,0,1)}
        assert net.nodes_codebook.shape[0] == 2


# ── Encode Properties Tests ──────────────────────────────────────────


class TestEncodeProperties:

    def test_sets_both_node_hvs(self):
        net = RRWPHyperNet(_qm9_rrwp_config())
        data = _ethane()
        data = net.encode_properties(data)
        assert hasattr(data, "node_hv")
        assert hasattr(data, "node_hv_full")

    def test_node_hv_shapes(self):
        net = RRWPHyperNet(_qm9_rrwp_config())
        data = _ethane()
        data = net.encode_properties(data)
        num_nodes = data.x.size(0)
        assert data.node_hv.shape == (num_nodes, net.hv_dim)
        assert data.node_hv_full.shape == (num_nodes, net.hv_dim)

    def test_original_x_preserved(self):
        net = RRWPHyperNet(_qm9_rrwp_config())
        data = _ethane()
        x_before = data.x.clone()
        net.encode_properties(data)
        assert torch.equal(data.x, x_before)

    def test_hvs_differ(self):
        """Base and full HVs should differ (different codebooks)."""
        net = RRWPHyperNet(_qm9_rrwp_config())
        data = _ethane()
        data = net.encode_properties(data)
        assert not torch.equal(data.node_hv, data.node_hv_full)


# ── Forward Pass Tests ───────────────────────────────────────────────


class TestForward:

    def test_output_keys(self):
        net = RRWPHyperNet(_qm9_rrwp_config())
        data = _ethane()
        result = net.forward(data)
        assert set(result.keys()) == {"graph_embedding", "node_terms", "edge_terms"}

    def test_output_shapes(self):
        net = RRWPHyperNet(_qm9_rrwp_config())
        data = _ethane()
        result = net.forward(data)
        for key in ("graph_embedding", "node_terms", "edge_terms"):
            assert result[key].shape == (1, net.hv_dim)

    def test_batch_output_shapes(self):
        net = RRWPHyperNet(_qm9_rrwp_config())
        d1 = _ethane()
        d2 = _methanol()
        batch = Batch.from_data_list([d1, d2])
        result = net.forward(batch)
        for key in ("graph_embedding", "node_terms", "edge_terms"):
            assert result[key].shape == (2, net.hv_dim)

    def test_node_terms_use_full_codebook(self):
        """node_terms should differ from what a base HyperNet would produce."""
        config = _qm9_rrwp_config()
        rrwp_net = RRWPHyperNet(config)

        base_config = _qm9_base_config()
        base_net = HyperNet(base_config)

        data_rrwp = _ethane()
        data_base = _make_molecule(
            base_features=[[0, 0, 0, 3], [0, 0, 0, 3]],
            edge_index=[[0, 1], [1, 0]],
            k_values=QM9_K_VALUES,
            num_bins=QM9_NUM_BINS,
        )
        # Strip RW features for base forward
        data_base_only = Data(
            x=data_base.x[:, :4],
            edge_index=data_base.edge_index,
            batch=data_base.batch,
        )

        result_rrwp = rrwp_net.forward(data_rrwp)
        result_base = base_net.forward(data_base_only)

        # node_terms should differ (RRWP uses full codebook)
        assert not torch.allclose(
            result_rrwp["node_terms"], result_base["node_terms"]
        )

    def test_deterministic_output(self):
        net = RRWPHyperNet(_qm9_rrwp_config())
        data1 = _ethane()
        data2 = _ethane()
        r1 = net.forward(data1)
        r2 = net.forward(data2)
        for key in ("graph_embedding", "node_terms", "edge_terms"):
            assert torch.allclose(r1[key], r2[key])

    def test_different_molecules_differ(self):
        net = RRWPHyperNet(_qm9_rrwp_config())
        r1 = net.forward(_ethane())
        r2 = net.forward(_methanol())
        assert not torch.allclose(r1["node_terms"], r2["node_terms"])

    def test_three_node_molecule(self):
        net = RRWPHyperNet(_qm9_rrwp_config())
        data = _propane()
        result = net.forward(data)
        for key in ("graph_embedding", "node_terms", "edge_terms"):
            assert result[key].shape == (1, net.hv_dim)
            assert torch.isfinite(result[key]).all()


# ── Decoding Tests ───────────────────────────────────────────────────


class TestDecoding:

    def test_decode_order_zero_uses_full_codebook(self):
        config = _qm9_rrwp_config()
        config.prune_codebook = False
        net = RRWPHyperNet(config)
        data = _ethane()
        result = net.forward(data)

        decoded = net.decode_order_zero(result["node_terms"])
        # Output size should match full codebook
        full_size = math.prod(QM9_BASE_BINS + [QM9_NUM_BINS] * len(QM9_K_VALUES))
        assert decoded.shape[-1] == full_size

    def test_decode_order_zero_iterative_returns_full_tuples(self):
        config = _qm9_rrwp_config()
        config.prune_codebook = False
        net = RRWPHyperNet(config)
        data = _ethane()
        result = net.forward(data)

        decoded = net.decode_order_zero_iterative(result["node_terms"].squeeze(0))
        if decoded:
            # Each decoded tuple should have base + RW dimensions
            expected_dims = len(QM9_BASE_BINS) + len(QM9_K_VALUES)
            assert len(decoded[0]) == expected_dims

    def test_decode_order_zero_counter(self):
        config = _qm9_rrwp_config()
        config.prune_codebook = False
        net = RRWPHyperNet(config)
        data = _ethane()
        result = net.forward(data)

        counters = net.decode_order_zero_counter(result["node_terms"])
        # Should return a dict
        assert isinstance(counters, dict)

    def test_edge_decoding_raises(self):
        net = RRWPHyperNet(_qm9_rrwp_config())
        dummy = torch.randn(net.hv_dim)
        with pytest.raises(NotImplementedError):
            net.decode_order_one(dummy, {})
        with pytest.raises(NotImplementedError):
            net.decode_order_one_no_node_terms(dummy)
        with pytest.raises(NotImplementedError):
            net.decode_graph(dummy, dummy)
        with pytest.raises(NotImplementedError):
            net.decode_graph_greedy(dummy, dummy)


# ── Save/Load Tests ──────────────────────────────────────────────────


class TestSaveLoad:

    def test_save_creates_file(self):
        net = RRWPHyperNet(_qm9_rrwp_config())
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rrwp_net.pt"
            net.save(path)
            assert path.exists()

    def test_load_roundtrip(self):
        net = RRWPHyperNet(_qm9_rrwp_config())
        data = _ethane()
        result_before = net.forward(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rrwp_net.pt"
            net.save(path)
            loaded = RRWPHyperNet.load(path)

        data2 = _ethane()
        result_after = loaded.forward(data2)

        for key in ("graph_embedding", "node_terms", "edge_terms"):
            assert torch.allclose(result_before[key], result_after[key])

    def test_load_preserves_codebooks(self):
        net = RRWPHyperNet(_qm9_rrwp_config())
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rrwp_net.pt"
            net.save(path)
            loaded = RRWPHyperNet.load(path)

        assert torch.equal(net.nodes_codebook.cpu(), loaded.nodes_codebook.cpu())
        assert torch.equal(
            net.nodes_codebook_full.cpu(), loaded.nodes_codebook_full.cpu()
        )

    def test_load_preserves_rw_config(self):
        net = RRWPHyperNet(_qm9_rrwp_config())
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rrwp_net.pt"
            net.save(path)
            loaded = RRWPHyperNet.load(path)

        assert loaded.rw_config.enabled is True
        assert loaded.rw_config.k_values == net.rw_config.k_values
        assert loaded.rw_config.num_bins == net.rw_config.num_bins

    def test_load_preserves_type(self):
        net = RRWPHyperNet(_qm9_rrwp_config())
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rrwp_net.pt"
            net.save(path)
            loaded = RRWPHyperNet.load(path)

        assert isinstance(loaded, RRWPHyperNet)
        assert isinstance(loaded, HyperNet)

    def test_load_hypernet_dispatches_to_rrwp(self):
        from graph_hdc.hypernet import load_hypernet

        net = RRWPHyperNet(_qm9_rrwp_config())
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rrwp_net.pt"
            net.save(path)
            loaded = load_hypernet(path)

        assert isinstance(loaded, RRWPHyperNet)

    def test_load_rejects_non_rrwp_checkpoint(self):
        base_net = HyperNet(_qm9_base_config())
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "base_net.pt"
            base_net.save(path)
            with pytest.raises(ValueError, match="does not contain RRWP"):
                RRWPHyperNet.load(path)


# ── Seed Reproducibility Tests ───────────────────────────────────────


class TestReproducibility:

    def test_same_seed_same_codebooks(self):
        config = _qm9_rrwp_config()
        net1 = RRWPHyperNet(config)
        net2 = RRWPHyperNet(config)
        assert torch.equal(net1.nodes_codebook, net2.nodes_codebook)
        assert torch.equal(net1.nodes_codebook_full, net2.nodes_codebook_full)

    def test_same_seed_same_forward(self):
        config = _qm9_rrwp_config()
        net1 = RRWPHyperNet(config)
        net2 = RRWPHyperNet(config)
        data1 = _ethane()
        data2 = _ethane()
        r1 = net1.forward(data1)
        r2 = net2.forward(data2)
        for key in ("graph_embedding", "node_terms", "edge_terms"):
            assert torch.allclose(r1[key], r2[key])
