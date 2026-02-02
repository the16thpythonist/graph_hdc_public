"""
Unit tests for FlowEdgeDecoder model.

Tests individual components with mock data to ensure the experiment pipeline works correctly.
"""

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from copy import deepcopy

from graph_hdc.models.flow_edge_decoder import (
    # Constants
    NODE_FEATURE_DIM,
    NODE_FEATURE_BINS,
    NUM_EDGE_CLASSES,
    ZINC_ATOM_TYPES,
    BOND_TYPES,
    # Helper functions
    get_bond_type_idx,
    node_tuple_to_onehot,
    node_tuples_to_onehot,
    raw_features_to_onehot,
    # Classes
    EdgeOnlyLoss,
    DistributionNodes,
    FlowEdgeDecoder,
    FlowEdgeDecoderConfig,
    # Preprocessing
    compute_edge_marginals,
    compute_node_counts,
)


# =============================================================================
# Fixtures for mock data
# =============================================================================

@pytest.fixture
def mock_edge_marginals():
    """Create mock edge marginals (probability distribution over edge types)."""
    # Typical distribution: mostly no-edge, some single bonds, few others
    marginals = torch.tensor([0.7, 0.2, 0.05, 0.02, 0.03])
    assert marginals.sum().isclose(torch.tensor(1.0))
    return marginals


@pytest.fixture
def mock_node_counts():
    """Create mock node count distribution."""
    # Distribution over graph sizes 0-15
    counts = torch.zeros(16)
    counts[5:10] = torch.tensor([10, 30, 40, 15, 5], dtype=torch.float)
    return counts


@pytest.fixture
def small_model(mock_edge_marginals, mock_node_counts):
    """Create a small FlowEdgeDecoder for testing."""
    return FlowEdgeDecoder(
        num_node_classes=NODE_FEATURE_DIM,
        num_edge_classes=NUM_EDGE_CLASSES,
        hdc_dim=64,  # Small for fast tests
        n_layers=2,
        hidden_dim=32,
        hidden_mlp_dim=64,
        n_heads=2,
        dropout=0.0,
        noise_type="marginal",
        edge_marginals=mock_edge_marginals,
        node_counts=mock_node_counts,
        max_nodes=15,
        extra_features_type="rrwp",
        rrwp_steps=5,
        lr=1e-4,
        weight_decay=0.0,
        train_time_distortion="identity",
        sample_steps=3,
        eta=0.0,
        omega=0.0,
        sample_time_distortion="identity",
    )


@pytest.fixture
def mock_dense_batch():
    """Create mock dense batch data for testing."""
    bs, n, dx, de = 2, 8, NODE_FEATURE_DIM, NUM_EDGE_CLASSES

    # Node features: 24-dim concatenated one-hot (9 atom + 6 degree + 3 charge + 4 Hs + 2 ring)
    # For simplicity, set one feature per category
    X = torch.zeros(bs, n, dx)
    # First graph: 5 carbons (C=1 in ZINC) + 2 nitrogens (N=5 in ZINC)
    for i in range(5):
        X[0, i, 1] = 1   # atom_type: C (index 1 in 9-class)
        X[0, i, 9] = 1   # degree: 0 (offset 9)
        X[0, i, 15] = 1  # charge: neutral (offset 15)
        X[0, i, 18] = 1  # Hs: 0 (offset 18)
        X[0, i, 22] = 1  # ring: not in ring (offset 22)
    for i in range(5, 7):
        X[0, i, 5] = 1   # atom_type: N (index 5 in 9-class)
        X[0, i, 9] = 1   # degree: 0
        X[0, i, 15] = 1  # charge: neutral
        X[0, i, 18] = 1  # Hs: 0
        X[0, i, 22] = 1  # ring: not in ring
    # Second graph: 4 carbons + 2 oxygens (O=6 in ZINC)
    for i in range(4):
        X[1, i, 1] = 1   # atom_type: C
        X[1, i, 9] = 1   # degree: 0
        X[1, i, 15] = 1  # charge: neutral
        X[1, i, 18] = 1  # Hs: 0
        X[1, i, 22] = 1  # ring: not in ring
    for i in range(4, 6):
        X[1, i, 6] = 1   # atom_type: O (index 6 in 9-class)
        X[1, i, 9] = 1   # degree: 0
        X[1, i, 15] = 1  # charge: neutral
        X[1, i, 18] = 1  # Hs: 0
        X[1, i, 22] = 1  # ring: not in ring

    # Edge features: one-hot encoded, symmetric
    E = torch.zeros(bs, n, n, de)
    # First graph: some single bonds
    E[0, 0, 1, 1] = E[0, 1, 0, 1] = 1  # single bond
    E[0, 1, 2, 1] = E[0, 2, 1, 1] = 1  # single bond
    E[0, 2, 3, 2] = E[0, 3, 2, 2] = 1  # double bond
    # Fill no-edge for rest
    for i in range(n):
        for j in range(n):
            if i != j and E[0, i, j].sum() == 0:
                E[0, i, j, 0] = 1
            if i != j and E[1, i, j].sum() == 0:
                E[1, i, j, 0] = 1
    # Second graph: some bonds
    E[1, 0, 1, 1] = E[1, 1, 0, 1] = 1
    E[1, 1, 2, 1] = E[1, 2, 1, 1] = 1

    # Node mask
    node_mask = torch.zeros(bs, n, dtype=torch.bool)
    node_mask[0, :7] = True  # 7 valid nodes in first graph
    node_mask[1, :6] = True  # 6 valid nodes in second graph

    # HDC vectors
    hdc_vectors = torch.randn(bs, 64)

    return {
        "X": X,
        "E": E,
        "node_mask": node_mask,
        "hdc_vectors": hdc_vectors,
    }


@pytest.fixture
def mock_pyg_batch(mock_dense_batch):
    """Create mock PyG Batch from dense data."""
    data_list = []

    for i in range(mock_dense_batch["X"].shape[0]):
        mask = mock_dense_batch["node_mask"][i]
        n_nodes = mask.sum().item()

        # Extract valid nodes
        x = mock_dense_batch["X"][i, :n_nodes]

        # Extract edges from dense format
        E = mock_dense_batch["E"][i, :n_nodes, :n_nodes]
        edge_index_list = []
        edge_attr_list = []

        for src in range(n_nodes):
            for dst in range(n_nodes):
                if src != dst and E[src, dst].argmax() != 0:  # Has an edge
                    edge_index_list.append([src, dst])
                    edge_attr_list.append(E[src, dst])

        if edge_index_list:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t()
            edge_attr = torch.stack(edge_attr_list)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_attr = torch.zeros(0, NUM_EDGE_CLASSES)

        # Also include original_x for HDC-guided sampling tests (raw 5-dim features)
        original_x = torch.zeros(n_nodes, 5)
        original_x[:, 0] = x[:, :9].argmax(dim=-1).float()  # atom type from first 9 dims

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            hdc_vector=mock_dense_batch["hdc_vectors"][i:i+1],  # Keep as (1, dim)
            original_x=original_x,
        )
        data_list.append(data)

    return Batch.from_data_list(data_list)


# =============================================================================
# Test Constants
# =============================================================================

def test_constants():
    """Test that constants are correctly defined."""
    assert NODE_FEATURE_DIM == 24  # 9 + 6 + 3 + 4 + 2
    assert NUM_EDGE_CLASSES == 5
    assert len(ZINC_ATOM_TYPES) == 9
    assert len(BOND_TYPES) == 5
    assert len(NODE_FEATURE_BINS) == 5
    assert sum(NODE_FEATURE_BINS) == NODE_FEATURE_DIM


# =============================================================================
# Test Helper Functions
# =============================================================================

class TestNodeTupleConversion:
    """Tests for node tuple to one-hot conversion functions."""

    def test_single_tuple_to_onehot(self):
        """Test converting a single node tuple to one-hot."""
        # (C atom=1, degree=1, neutral=0, 2 Hs, not in ring=0)
        t = (1, 1, 0, 2, 0)
        onehot = node_tuple_to_onehot(t)
        assert onehot.shape == (NODE_FEATURE_DIM,)
        assert onehot.sum() == 5  # One hot per feature

    def test_single_tuple_to_onehot_with_device(self):
        """Test converting with specified device."""
        t = (1, 1, 0, 2, 0)
        device = torch.device("cpu")
        onehot = node_tuple_to_onehot(t, device=device)
        assert onehot.device == device

    def test_multiple_tuples_to_onehot(self):
        """Test converting multiple tuples."""
        tuples = [(1, 1, 0, 2, 0), (5, 2, 0, 1, 1)]  # C and N
        onehot = node_tuples_to_onehot(tuples)
        assert onehot.shape == (2, NODE_FEATURE_DIM)

    def test_empty_tuples_list(self):
        """Test empty list returns empty tensor."""
        onehot = node_tuples_to_onehot([])
        assert onehot.shape == (0, NODE_FEATURE_DIM)

    def test_raw_features_to_onehot(self):
        """Test converting raw features tensor to one-hot."""
        # Create raw features (n, 5)
        raw = torch.tensor([
            [1, 1, 0, 2, 0],  # C
            [5, 2, 0, 1, 1],  # N
        ], dtype=torch.float32)
        onehot = raw_features_to_onehot(raw)
        assert onehot.shape == (2, NODE_FEATURE_DIM)
        # Each row should sum to 5 (one per feature category)
        assert torch.allclose(onehot.sum(dim=1), torch.tensor([5.0, 5.0]))


class TestGetBondTypeIdx:
    """Tests for get_bond_type_idx function."""

    def test_no_bond(self):
        """Test None bond returns 0."""
        assert get_bond_type_idx(None) == 0

    def test_bond_types(self):
        """Test standard bond types using real molecule."""
        from rdkit import Chem
        # Create a molecule with different bond types
        mol = Chem.MolFromSmiles("C=CC#N")  # has single, double, triple
        bonds = list(mol.GetBonds())

        # Find bonds of each type
        bond_types_found = {}
        for bond in bonds:
            bt = bond.GetBondType()
            idx = get_bond_type_idx(bond)
            bond_types_found[bt] = idx

        assert bond_types_found[Chem.BondType.SINGLE] == 1
        assert bond_types_found[Chem.BondType.DOUBLE] == 2
        assert bond_types_found[Chem.BondType.TRIPLE] == 3

    def test_aromatic_bond(self):
        """Test aromatic bond type."""
        from rdkit import Chem
        mol = Chem.MolFromSmiles("c1ccccc1")  # benzene
        bond = mol.GetBondBetweenAtoms(0, 1)
        assert get_bond_type_idx(bond) == 4


# =============================================================================
# Test EdgeOnlyLoss
# =============================================================================

class TestEdgeOnlyLoss:
    """Tests for EdgeOnlyLoss class."""

    def test_creation(self):
        """Test loss function creation."""
        loss_fn = EdgeOnlyLoss()
        assert loss_fn is not None

    def test_forward_shape(self, mock_dense_batch):
        """Test loss computation returns scalar."""
        loss_fn = EdgeOnlyLoss()

        bs, n = mock_dense_batch["X"].shape[:2]
        de = NUM_EDGE_CLASSES

        # Use requires_grad=True to enable gradient computation
        pred_E = torch.randn(bs, n, n, de, requires_grad=True)
        true_E = mock_dense_batch["E"]
        node_mask = mock_dense_batch["node_mask"]

        loss = loss_fn(pred_E, true_E, node_mask)

        assert loss.shape == ()
        assert loss.requires_grad

    def test_perfect_prediction_low_loss(self, mock_dense_batch):
        """Test that perfect prediction gives low loss."""
        loss_fn = EdgeOnlyLoss()

        true_E = mock_dense_batch["E"]
        node_mask = mock_dense_batch["node_mask"]

        # Create "perfect" predictions by using true labels with high confidence
        pred_E = torch.zeros_like(true_E)
        pred_E[true_E == 1] = 10.0  # High logit for correct class
        pred_E[true_E == 0] = -10.0  # Low logit for wrong class

        loss = loss_fn(pred_E, true_E, node_mask)

        assert loss.item() < 0.1  # Should be very low

    def test_masked_nodes_ignored(self):
        """Test that masked nodes don't contribute to loss."""
        loss_fn = EdgeOnlyLoss()

        bs, n, de = 2, 5, NUM_EDGE_CLASSES

        # All nodes valid
        pred_E = torch.randn(bs, n, n, de)
        true_E = F.one_hot(torch.randint(0, de, (bs, n, n)), de).float()
        node_mask_full = torch.ones(bs, n, dtype=torch.bool)

        # Only first 3 nodes valid
        node_mask_partial = torch.zeros(bs, n, dtype=torch.bool)
        node_mask_partial[:, :3] = True

        loss_full = loss_fn(pred_E, true_E, node_mask_full)
        loss_partial = loss_fn(pred_E, true_E, node_mask_partial)

        # Losses should be different due to different valid edges
        assert not torch.isclose(loss_full, loss_partial)


# =============================================================================
# Test DistributionNodes
# =============================================================================

class TestDistributionNodes:
    """Tests for DistributionNodes class."""

    def test_creation(self):
        """Test distribution creation."""
        dist = DistributionNodes()
        assert dist is not None

    def test_creation_with_histogram(self, mock_node_counts):
        """Test distribution creation with histogram."""
        dist = DistributionNodes(mock_node_counts)
        assert dist.histogram.sum().isclose(torch.tensor(1.0))

    def test_sample_shape(self, mock_node_counts):
        """Test sampling returns correct shape."""
        dist = DistributionNodes(mock_node_counts)
        samples = dist.sample(10, torch.device("cpu"))
        assert samples.shape == (10,)

    def test_sample_in_range(self, mock_node_counts):
        """Test samples are within valid range."""
        dist = DistributionNodes(mock_node_counts)
        samples = dist.sample(100, torch.device("cpu"))
        assert samples.min() >= 0
        assert samples.max() < len(mock_node_counts)


# =============================================================================
# Test FlowEdgeDecoder Creation
# =============================================================================

class TestFlowEdgeDecoderCreation:
    """Tests for FlowEdgeDecoder model creation."""

    def test_creation(self, small_model):
        """Test model creation."""
        assert small_model is not None
        assert isinstance(small_model, FlowEdgeDecoder)

    def test_model_attributes(self, small_model):
        """Test model has expected attributes."""
        assert small_model.num_node_classes == NODE_FEATURE_DIM
        assert small_model.num_edge_classes == NUM_EDGE_CLASSES
        assert small_model.hdc_dim == 64
        assert hasattr(small_model, "model")  # GraphTransformer
        assert hasattr(small_model, "train_loss")  # EdgeOnlyLoss
        assert hasattr(small_model, "limit_dist")  # LimitDistribution

    def test_input_output_dims(self, small_model):
        """Test input/output dimensions are set correctly."""
        assert "X" in small_model.input_dims
        assert "E" in small_model.input_dims
        assert "y" in small_model.input_dims

        assert small_model.output_dims["X"] == NODE_FEATURE_DIM
        assert small_model.output_dims["E"] == NUM_EDGE_CLASSES


# =============================================================================
# Test FlowEdgeDecoder Forward Pass
# =============================================================================

class TestFlowEdgeDecoderForward:
    """Tests for FlowEdgeDecoder forward pass."""

    def test_forward_output_shape(self, small_model, mock_dense_batch):
        """Test forward pass returns correct shapes."""
        small_model.eval()

        X = mock_dense_batch["X"]
        E = mock_dense_batch["E"]
        hdc = mock_dense_batch["hdc_vectors"]
        node_mask = mock_dense_batch["node_mask"]

        # Use _apply_noise to properly create noisy data (handles HDC projection)
        noisy_data = small_model._apply_noise(X, E, hdc, node_mask)

        extra_data = small_model._compute_extra_data(noisy_data)

        with torch.no_grad():
            pred = small_model.forward(noisy_data, extra_data, node_mask)

        bs, n = X.shape[:2]
        assert pred.X.shape == (bs, n, NODE_FEATURE_DIM)
        assert pred.E.shape == (bs, n, n, NUM_EDGE_CLASSES)

    def test_forward_no_nan(self, small_model, mock_dense_batch):
        """Test forward pass produces no NaN values."""
        small_model.eval()

        X = mock_dense_batch["X"]
        E = mock_dense_batch["E"]
        hdc = mock_dense_batch["hdc_vectors"]
        node_mask = mock_dense_batch["node_mask"]

        # Use _apply_noise to properly create noisy data
        noisy_data = small_model._apply_noise(X, E, hdc, node_mask)

        extra_data = small_model._compute_extra_data(noisy_data)

        with torch.no_grad():
            pred = small_model.forward(noisy_data, extra_data, mock_dense_batch["node_mask"])

        assert not torch.isnan(pred.X).any()
        assert not torch.isnan(pred.E).any()
        assert not torch.isinf(pred.X).any()
        assert not torch.isinf(pred.E).any()


# =============================================================================
# Test FlowEdgeDecoder._apply_noise
# =============================================================================

class TestApplyNoise:
    """Tests for _apply_noise method."""

    def test_apply_noise_output_keys(self, small_model, mock_dense_batch):
        """Test _apply_noise returns expected keys."""
        noisy_data = small_model._apply_noise(
            mock_dense_batch["X"],
            mock_dense_batch["E"],
            mock_dense_batch["hdc_vectors"],
            mock_dense_batch["node_mask"],
        )

        assert "t" in noisy_data
        assert "X_t" in noisy_data
        assert "E_t" in noisy_data
        assert "y_t" in noisy_data
        assert "node_mask" in noisy_data

    def test_apply_noise_nodes_unchanged(self, small_model, mock_dense_batch):
        """Test that nodes are not modified (edge-only noise)."""
        noisy_data = small_model._apply_noise(
            mock_dense_batch["X"],
            mock_dense_batch["E"],
            mock_dense_batch["hdc_vectors"],
            mock_dense_batch["node_mask"],
        )

        # X_t should be identical to input X
        assert torch.allclose(noisy_data["X_t"], mock_dense_batch["X"])

    def test_apply_noise_edges_one_hot(self, small_model, mock_dense_batch):
        """Test that noisy edges are valid one-hot."""
        noisy_data = small_model._apply_noise(
            mock_dense_batch["X"],
            mock_dense_batch["E"],
            mock_dense_batch["hdc_vectors"],
            mock_dense_batch["node_mask"],
        )

        E_t = noisy_data["E_t"]

        # Should sum to 1 along last dim (where mask is valid)
        node_mask = mock_dense_batch["node_mask"]
        edge_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)

        # Check valid edges sum to 1
        edge_sums = E_t.sum(dim=-1)
        assert torch.allclose(edge_sums[edge_mask], torch.ones_like(edge_sums[edge_mask]))

    def test_apply_noise_time_in_range(self, small_model, mock_dense_batch):
        """Test that sampled time is in [0, 1]."""
        noisy_data = small_model._apply_noise(
            mock_dense_batch["X"],
            mock_dense_batch["E"],
            mock_dense_batch["hdc_vectors"],
            mock_dense_batch["node_mask"],
        )

        t = noisy_data["t"]
        assert (t >= 0).all()
        assert (t <= 1).all()


# =============================================================================
# Test FlowEdgeDecoder._compute_extra_data
# =============================================================================

class TestComputeExtraData:
    """Tests for _compute_extra_data method."""

    def test_compute_extra_data_output_type(self, small_model, mock_dense_batch):
        """Test _compute_extra_data returns PlaceHolder."""
        from defog.core import PlaceHolder

        noisy_data = {
            "t": torch.tensor([[0.5], [0.5]]),
            "X_t": mock_dense_batch["X"],
            "E_t": mock_dense_batch["E"],
            "y_t": mock_dense_batch["hdc_vectors"],
            "node_mask": mock_dense_batch["node_mask"],
        }

        extra_data = small_model._compute_extra_data(noisy_data)

        assert isinstance(extra_data, PlaceHolder)
        assert hasattr(extra_data, "X")
        assert hasattr(extra_data, "E")
        assert hasattr(extra_data, "y")

    def test_compute_extra_data_shapes(self, small_model, mock_dense_batch):
        """Test extra data has compatible shapes."""
        bs, n = mock_dense_batch["X"].shape[:2]

        noisy_data = {
            "t": torch.tensor([[0.5], [0.5]]),
            "X_t": mock_dense_batch["X"],
            "E_t": mock_dense_batch["E"],
            "y_t": mock_dense_batch["hdc_vectors"],
            "node_mask": mock_dense_batch["node_mask"],
        }

        extra_data = small_model._compute_extra_data(noisy_data)

        assert extra_data.X.shape[0] == bs
        assert extra_data.X.shape[1] == n
        assert extra_data.E.shape[:3] == (bs, n, n)
        assert extra_data.y.shape[0] == bs


# =============================================================================
# Test FlowEdgeDecoder._sample_step
# =============================================================================

class TestSampleStep:
    """Tests for _sample_step method."""

    def test_sample_step_output(self, small_model, mock_dense_batch):
        """Test _sample_step returns correct structure."""
        small_model.eval()

        X = mock_dense_batch["X"]
        E = mock_dense_batch["E"]
        hdc = mock_dense_batch["hdc_vectors"]
        node_mask = mock_dense_batch["node_mask"]

        # Project HDC through condition_mlp (same as sample() does)
        with torch.no_grad():
            y = small_model.condition_mlp(hdc)

        t = torch.tensor([[0.3], [0.3]])
        s = torch.tensor([[0.4], [0.4]])

        with torch.no_grad():
            X_s, E_s, y_s, pred_E = small_model._sample_step(t, s, X, E, y, node_mask)

        assert X_s.shape == X.shape
        assert E_s.shape == E.shape
        assert y_s.shape == y.shape
        assert pred_E.shape == E.shape  # pred_E has same shape as E

    def test_sample_step_nodes_unchanged(self, small_model, mock_dense_batch):
        """Test that nodes don't change during sampling step."""
        small_model.eval()

        X = mock_dense_batch["X"]
        E = mock_dense_batch["E"]
        hdc = mock_dense_batch["hdc_vectors"]
        node_mask = mock_dense_batch["node_mask"]

        # Project HDC through condition_mlp (same as sample() does)
        with torch.no_grad():
            y = small_model.condition_mlp(hdc)

        t = torch.tensor([[0.3], [0.3]])
        s = torch.tensor([[0.4], [0.4]])

        with torch.no_grad():
            X_s, E_s, y_s, pred_E = small_model._sample_step(t, s, X, E, y, node_mask)

        # Nodes should be unchanged
        assert torch.allclose(X_s, X)

    def test_sample_step_edges_valid(self, small_model, mock_dense_batch):
        """Test that edges remain valid one-hot after step."""
        small_model.eval()

        X = mock_dense_batch["X"]
        E = mock_dense_batch["E"]
        hdc = mock_dense_batch["hdc_vectors"]
        node_mask = mock_dense_batch["node_mask"]

        # Project HDC through condition_mlp (same as sample() does)
        with torch.no_grad():
            y = small_model.condition_mlp(hdc)

        t = torch.tensor([[0.3], [0.3]])
        s = torch.tensor([[0.4], [0.4]])

        with torch.no_grad():
            X_s, E_s, y_s, pred_E = small_model._sample_step(t, s, X, E, y, node_mask)

        # Edges should be one-hot
        edge_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
        edge_sums = E_s.sum(dim=-1)
        assert torch.allclose(edge_sums[edge_mask], torch.ones_like(edge_sums[edge_mask]))


# =============================================================================
# Test FlowEdgeDecoder.sample
# =============================================================================

class TestSample:
    """Tests for sample method."""

    def test_sample_returns_list(self, small_model, mock_dense_batch):
        """Test sample returns list of Data objects."""
        small_model.eval()

        samples = small_model.sample(
            hdc_vectors=mock_dense_batch["hdc_vectors"],
            node_features=mock_dense_batch["X"],
            node_mask=mock_dense_batch["node_mask"],
            sample_steps=2,
            show_progress=False,
        )

        assert isinstance(samples, list)
        assert len(samples) == 2  # batch size
        assert all(isinstance(s, Data) for s in samples)

    def test_sample_correct_node_counts(self, small_model, mock_dense_batch):
        """Test sampled graphs have correct node counts."""
        small_model.eval()

        samples = small_model.sample(
            hdc_vectors=mock_dense_batch["hdc_vectors"],
            node_features=mock_dense_batch["X"],
            node_mask=mock_dense_batch["node_mask"],
            sample_steps=2,
            show_progress=False,
        )

        # Node counts should match mask
        expected_counts = mock_dense_batch["node_mask"].sum(dim=1).tolist()
        actual_counts = [s.num_nodes for s in samples]

        assert actual_counts == expected_counts

    def test_sample_edges_exist(self, small_model, mock_dense_batch):
        """Test that sampled graphs have edges."""
        small_model.eval()

        samples = small_model.sample(
            hdc_vectors=mock_dense_batch["hdc_vectors"],
            node_features=mock_dense_batch["X"],
            node_mask=mock_dense_batch["node_mask"],
            sample_steps=2,
            show_progress=False,
        )

        # At least some samples should have edges
        total_edges = sum(s.edge_index.shape[1] for s in samples)
        assert total_edges > 0


# =============================================================================
# Test FlowEdgeDecoder Training/Validation Steps
# =============================================================================

class TestTrainingValidation:
    """Tests for training and validation steps."""

    def test_training_step_returns_loss(self, small_model, mock_pyg_batch):
        """Test training step returns loss dict."""
        result = small_model.training_step(mock_pyg_batch, 0)

        assert result is not None
        assert "loss" in result
        assert result["loss"].requires_grad

    def test_training_step_loss_finite(self, small_model, mock_pyg_batch):
        """Test training step produces finite loss."""
        result = small_model.training_step(mock_pyg_batch, 0)

        assert torch.isfinite(result["loss"])

    def test_validation_step_returns_loss(self, small_model, mock_pyg_batch):
        """Test validation step returns loss dict."""
        result = small_model.validation_step(mock_pyg_batch, 0)

        assert result is not None
        assert "val_loss" in result

    def test_validation_step_loss_detached(self, small_model, mock_pyg_batch):
        """Test validation loss is detached (for deepcopy)."""
        result = small_model.validation_step(mock_pyg_batch, 0)

        assert not result["val_loss"].requires_grad

    def test_validation_step_deepcopy_works(self, small_model, mock_pyg_batch):
        """Test validation result can be deepcopied (required by Lightning)."""
        result = small_model.validation_step(mock_pyg_batch, 0)

        # This should not raise
        copied = deepcopy(result)
        assert "val_loss" in copied


# =============================================================================
# Test Utility Functions
# =============================================================================

class TestComputeEdgeMarginals:
    """Tests for compute_edge_marginals function."""

    def test_returns_tensor(self, mock_pyg_batch):
        """Test function returns tensor."""
        data_list = mock_pyg_batch.to_data_list()
        marginals = compute_edge_marginals(data_list)

        assert isinstance(marginals, torch.Tensor)

    def test_correct_shape(self, mock_pyg_batch):
        """Test marginals have correct shape."""
        data_list = mock_pyg_batch.to_data_list()
        marginals = compute_edge_marginals(data_list)

        assert marginals.shape == (NUM_EDGE_CLASSES,)

    def test_sums_to_one(self, mock_pyg_batch):
        """Test marginals sum to 1."""
        data_list = mock_pyg_batch.to_data_list()
        marginals = compute_edge_marginals(data_list)

        assert marginals.sum().isclose(torch.tensor(1.0))

    def test_all_non_negative(self, mock_pyg_batch):
        """Test all marginals are non-negative."""
        data_list = mock_pyg_batch.to_data_list()
        marginals = compute_edge_marginals(data_list)

        assert (marginals >= 0).all()


class TestComputeNodeCounts:
    """Tests for compute_node_counts function."""

    def test_returns_tensor(self, mock_pyg_batch):
        """Test function returns tensor."""
        data_list = mock_pyg_batch.to_data_list()
        counts = compute_node_counts(data_list)

        assert isinstance(counts, torch.Tensor)

    def test_counts_match_data(self, mock_pyg_batch):
        """Test counts match actual graph sizes."""
        data_list = mock_pyg_batch.to_data_list()
        counts = compute_node_counts(data_list)

        # Check that counts for actual sizes are positive
        for data in data_list:
            n = data.num_nodes
            assert counts[n] > 0


# =============================================================================
# Test HDC Vector Handling
# =============================================================================

class TestHDCVectorHandling:
    """Tests for HDC vector handling in batching."""

    def test_hdc_vector_batch_shape(self, mock_pyg_batch):
        """Test HDC vectors batch correctly."""
        # HDC vectors should be (batch_size, hdc_dim) after batching
        assert mock_pyg_batch.hdc_vector.dim() == 2
        assert mock_pyg_batch.hdc_vector.shape[0] == 2  # batch size

    def test_get_hdc_vectors_from_batch(self, small_model, mock_pyg_batch):
        """Test _get_hdc_vectors_from_batch extracts correct shape."""
        hdc = small_model._get_hdc_vectors_from_batch(mock_pyg_batch)

        assert hdc.dim() == 2
        assert hdc.shape[0] == 2  # batch size
        assert hdc.shape[1] == 64  # hdc_dim

    def test_hdc_vectors_not_hrrtensor(self, mock_pyg_batch):
        """Test HDC vectors are regular tensors, not HRRTensor."""
        # This is important to avoid deepcopy issues
        assert type(mock_pyg_batch.hdc_vector) == torch.Tensor


# =============================================================================
# Test Model Save/Load
# =============================================================================

class TestSaveLoad:
    """Tests for model save/load functionality."""

    def test_save_creates_file(self, small_model, tmp_path):
        """Test save creates checkpoint file."""
        path = tmp_path / "test_model.ckpt"
        saved_path = small_model.save(str(path))

        assert saved_path.exists()

    def test_load_restores_model(self, small_model, tmp_path):
        """Test load restores model correctly."""
        path = tmp_path / "test_model.ckpt"
        small_model.save(str(path))

        # Load uses hyperparameters stored in checkpoint
        loaded = FlowEdgeDecoder.load(str(path))

        assert loaded is not None
        assert loaded.hdc_dim == small_model.hdc_dim


# =============================================================================
# Test Configuration
# =============================================================================

class TestConfiguration:
    """Tests for FlowEdgeDecoderConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FlowEdgeDecoderConfig()

        assert config.num_node_classes == NODE_FEATURE_DIM
        assert config.num_edge_classes == NUM_EDGE_CLASSES
        assert config.hdc_dim == 512

    def test_config_to_model(self, mock_edge_marginals, mock_node_counts):
        """Test creating model from config."""
        config = FlowEdgeDecoderConfig(
            hdc_dim=64,
            n_layers=2,
            hidden_dim=32,
        )

        model = FlowEdgeDecoder(
            num_node_classes=config.num_node_classes,
            num_edge_classes=config.num_edge_classes,
            hdc_dim=config.hdc_dim,
            n_layers=config.n_layers,
            hidden_dim=config.hidden_dim,
            hidden_mlp_dim=config.hidden_mlp_dim,
            n_heads=config.n_heads,
            dropout=config.dropout,
            noise_type=config.noise_type,
            edge_marginals=mock_edge_marginals,
            node_counts=mock_node_counts,
            max_nodes=config.max_nodes,
            extra_features_type=config.extra_features_type,
            rrwp_steps=config.rrwp_steps,
            lr=config.lr,
            weight_decay=config.weight_decay,
            train_time_distortion=config.train_time_distortion,
            sample_steps=config.sample_steps,
            eta=config.eta,
            omega=config.omega,
            sample_time_distortion=config.sample_time_distortion,
        )

        assert model.hdc_dim == 64


# =============================================================================
# End-to-End Decoding Test
# =============================================================================

class TestEndToEndDecoding:
    """End-to-end tests for the full encoding -> decoding pipeline."""

    @pytest.fixture
    def hypernet_and_model(self, mock_edge_marginals, mock_node_counts):
        """Create HyperNet encoder and FlowEdgeDecoder for end-to-end testing."""
        from collections import OrderedDict
        import math
        from graph_hdc.hypernet.configs import DSHDCConfig, FeatureConfig, Features, IndexRange
        from graph_hdc.hypernet.feature_encoders import CombinatoricIntegerEncoder
        from graph_hdc.hypernet.types import VSAModel
        from graph_hdc.hypernet.encoder import HyperNet

        # Create HyperNet config for ZINC (5 features: atom, degree, charge, Hs, ring)
        node_feature_config = FeatureConfig(
            count=math.prod([9, 6, 3, 4, 2]),  # 9*6*3*4*2 = 1296
            encoder_cls=CombinatoricIntegerEncoder,
            index_range=IndexRange((0, 5)),
            bins=[9, 6, 3, 4, 2],
        )
        hdc_config = DSHDCConfig(
            name='ZINC_HRR_64_test',
            hv_dim=64,  # Small for testing
            vsa=VSAModel.HRR,
            base_dataset='zinc',
            hypernet_depth=2,
            device='cpu',
            seed=42,
            normalize=True,
            dtype='float32',  # Use float32 to match model
            node_feature_configs=OrderedDict([(Features.NODE_FEATURES, node_feature_config)]),
        )
        hypernet = HyperNet(hdc_config)

        # Create FlowEdgeDecoder
        # Note: hdc_dim = 2 * hv_dim because preprocess concatenates order_0 and order_N
        model = FlowEdgeDecoder(
            num_node_classes=NODE_FEATURE_DIM,
            num_edge_classes=NUM_EDGE_CLASSES,
            hdc_dim=128,  # 2 * hv_dim (64)
            n_layers=2,
            hidden_dim=32,
            hidden_mlp_dim=64,
            n_heads=2,
            dropout=0.0,
            noise_type="marginal",
            edge_marginals=mock_edge_marginals,
            node_counts=mock_node_counts,
            max_nodes=15,
            extra_features_type="rrwp",
            rrwp_steps=5,
            lr=1e-4,
            weight_decay=0.0,
            train_time_distortion="identity",
            sample_steps=3,
            eta=0.0,
            omega=0.0,
            sample_time_distortion="identity",
        )

        return hypernet, model

    @pytest.fixture
    def sample_molecule_data(self):
        """Create sample molecule data for testing."""
        from graph_hdc.datasets.utils import get_split

        # Get a few real molecules from ZINC
        train_ds = get_split('train', dataset='zinc')
        return train_ds[:3]

    def test_encode_decode_pipeline(self, hypernet_and_model, sample_molecule_data):
        """Test full encode -> decode pipeline produces valid output."""
        from graph_hdc.models.flow_edge_decoder import preprocess_dataset
        from defog.core import to_dense

        hypernet, model = hypernet_and_model
        model.eval()

        # Preprocess molecules (encodes with HyperNet)
        processed = preprocess_dataset(
            sample_molecule_data,
            hypernet,
            device=torch.device("cpu"),
            show_progress=False,
        )

        assert len(processed) > 0, "Should have at least one valid molecule"

        # Create batch
        batch = Batch.from_data_list(processed)

        # Get dense format for sampling
        dense_data, node_mask = to_dense(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )

        # Decode (sample edges given HDC vectors and node features)
        with torch.no_grad():
            samples = model.sample(
                hdc_vectors=batch.hdc_vector,
                node_features=dense_data.X,
                node_mask=node_mask,
                sample_steps=3,
                show_progress=False,
            )

        # Verify output
        assert len(samples) == len(processed), "Should get same number of samples"

        for i, sample in enumerate(samples):
            # Check sample is valid Data object
            assert isinstance(sample, Data)
            assert hasattr(sample, 'x')
            assert hasattr(sample, 'edge_index')
            assert hasattr(sample, 'edge_attr')

            # Check node count matches
            expected_nodes = node_mask[i].sum().item()
            assert sample.num_nodes == expected_nodes, f"Sample {i} has wrong node count"

            # Check edge_index is valid (indices within range)
            if sample.edge_index.numel() > 0:
                assert sample.edge_index.max() < sample.num_nodes, "Edge indices out of range"
                assert sample.edge_index.min() >= 0, "Negative edge indices"

    def test_decoded_edges_are_symmetric(self, hypernet_and_model, sample_molecule_data):
        """Test that decoded edges form symmetric adjacency (undirected graph)."""
        from graph_hdc.models.flow_edge_decoder import preprocess_dataset
        from defog.core import to_dense

        hypernet, model = hypernet_and_model
        model.eval()

        processed = preprocess_dataset(
            sample_molecule_data,
            hypernet,
            device=torch.device("cpu"),
            show_progress=False,
        )

        batch = Batch.from_data_list(processed)
        dense_data, node_mask = to_dense(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )

        with torch.no_grad():
            samples = model.sample(
                hdc_vectors=batch.hdc_vector,
                node_features=dense_data.X,
                node_mask=node_mask,
                sample_steps=3,
                show_progress=False,
            )

        for sample in samples:
            if sample.edge_index.numel() == 0:
                continue

            # Build adjacency set
            edges = set()
            for i in range(sample.edge_index.shape[1]):
                src, dst = sample.edge_index[0, i].item(), sample.edge_index[1, i].item()
                edges.add((src, dst))

            # Check symmetry
            for src, dst in list(edges):
                assert (dst, src) in edges, f"Edge ({src}, {dst}) has no reverse edge"

    def test_preprocessing_creates_valid_hdc_vectors(self, hypernet_and_model, sample_molecule_data):
        """Test that preprocessing creates valid HDC vectors."""
        from graph_hdc.models.flow_edge_decoder import preprocess_dataset

        hypernet, model = hypernet_and_model

        processed = preprocess_dataset(
            sample_molecule_data,
            hypernet,
            device=torch.device("cpu"),
            show_progress=False,
        )

        for data in processed:
            # Check HDC vector exists and has correct properties
            assert hasattr(data, 'hdc_vector')
            assert data.hdc_vector.dim() == 2, "HDC vector should be 2D for batching"
            # HDC vector is [order_0 | order_N] = 2 * hv_dim = 128
            assert data.hdc_vector.shape == (1, 128), f"Wrong shape: {data.hdc_vector.shape}"

            # Check it's a regular tensor (not HRRTensor)
            assert type(data.hdc_vector) == torch.Tensor

            # Check values are finite
            assert torch.isfinite(data.hdc_vector).all(), "HDC vector has non-finite values"

    def test_preprocessing_creates_valid_node_features(self, hypernet_and_model, sample_molecule_data):
        """Test that preprocessing creates valid 24-dim node features (concatenated one-hot)."""
        from graph_hdc.models.flow_edge_decoder import preprocess_dataset

        hypernet, model = hypernet_and_model

        processed = preprocess_dataset(
            sample_molecule_data,
            hypernet,
            device=torch.device("cpu"),
            show_progress=False,
        )

        for data in processed:
            # Check node features are 24-dim concatenated one-hot
            assert data.x.shape[1] == NODE_FEATURE_DIM, f"Wrong feature dim: {data.x.shape[1]}"

            # Each row should sum to 5 (one-hot per feature: atom, degree, charge, Hs, ring)
            row_sums = data.x.sum(dim=1)
            expected_sum = torch.full_like(row_sums, 5.0)
            assert torch.allclose(row_sums, expected_sum), "Node features not valid concatenated one-hot"

    def test_preprocessing_creates_valid_edge_features(self, hypernet_and_model, sample_molecule_data):
        """Test that preprocessing creates valid 5-class edge features."""
        from graph_hdc.models.flow_edge_decoder import preprocess_dataset

        hypernet, model = hypernet_and_model

        processed = preprocess_dataset(
            sample_molecule_data,
            hypernet,
            device=torch.device("cpu"),
            show_progress=False,
        )

        for data in processed:
            if data.edge_attr is None or data.edge_attr.numel() == 0:
                continue

            # Check edge features are one-hot
            assert data.edge_attr.shape[1] == NUM_EDGE_CLASSES, f"Wrong edge dim: {data.edge_attr.shape[1]}"

            # Each row should sum to 1 (one-hot)
            row_sums = data.edge_attr.sum(dim=1)
            assert torch.allclose(row_sums, torch.ones_like(row_sums)), "Edge features not one-hot"

    def test_training_on_real_data(self, hypernet_and_model, sample_molecule_data):
        """Test that training step works on real preprocessed data."""
        from graph_hdc.models.flow_edge_decoder import preprocess_dataset

        hypernet, model = hypernet_and_model

        processed = preprocess_dataset(
            sample_molecule_data,
            hypernet,
            device=torch.device("cpu"),
            show_progress=False,
        )

        batch = Batch.from_data_list(processed)

        # Run training step
        result = model.training_step(batch, 0)

        assert result is not None
        assert "loss" in result
        assert torch.isfinite(result["loss"]), "Training loss is not finite"
        assert result["loss"].requires_grad, "Training loss should have gradients"

    def test_validation_on_real_data(self, hypernet_and_model, sample_molecule_data):
        """Test that validation step works on real preprocessed data."""
        from graph_hdc.models.flow_edge_decoder import preprocess_dataset

        hypernet, model = hypernet_and_model

        processed = preprocess_dataset(
            sample_molecule_data,
            hypernet,
            device=torch.device("cpu"),
            show_progress=False,
        )

        batch = Batch.from_data_list(processed)

        # Run validation step
        result = model.validation_step(batch, 0)

        assert result is not None
        assert "val_loss" in result
        assert torch.isfinite(result["val_loss"]), "Validation loss is not finite"

        # Should be deepcopy-able
        copied = deepcopy(result)
        assert "val_loss" in copied


# =============================================================================
# HDC-Guided Sampling Tests
# =============================================================================

class TestHDCGuidedSampling:
    """Tests for HDC-guided sampling methods."""

    @pytest.fixture
    def hdc_guidance_setup(self, small_model, mock_dense_batch):
        """Setup for HDC guidance tests."""
        model = small_model
        X = mock_dense_batch["X"]
        E = mock_dense_batch["E"]
        node_mask = mock_dense_batch["node_mask"]
        hdc_vectors = mock_dense_batch["hdc_vectors"]

        # Create mock original node features (5-dim: atom_type, degree, charge, Hs, ring)
        bs, n = X.shape[:2]
        original_node_features = torch.zeros(bs, n, 5)
        # Set atom type from first 9 dims of 24-dim encoding
        original_node_features[:, :, 0] = X[:, :, :9].argmax(dim=-1).float()

        return model, X, E, node_mask, hdc_vectors, original_node_features

    def test_sample_k_candidates_shape(self, hdc_guidance_setup):
        """Test that _sample_k_candidates produces correct shape."""
        model, X, E, node_mask, hdc_vectors, original_node_features = hdc_guidance_setup
        bs, n = X.shape[:2]
        de = NUM_EDGE_CLASSES
        K = 4

        # Create mock pred_E
        pred_E = torch.rand(bs, n, n, de)
        pred_E = pred_E / pred_E.sum(dim=-1, keepdim=True)  # Normalize

        candidates = model._sample_k_candidates(pred_E, K, node_mask)

        assert candidates.shape == (K, bs, n, n), f"Expected shape ({K}, {bs}, {n}, {n}), got {candidates.shape}"

    def test_sample_k_candidates_symmetry(self, hdc_guidance_setup):
        """Test that _sample_k_candidates produces symmetric edge matrices."""
        model, X, E, node_mask, hdc_vectors, original_node_features = hdc_guidance_setup
        bs, n = X.shape[:2]
        de = NUM_EDGE_CLASSES
        K = 4

        pred_E = torch.rand(bs, n, n, de)
        pred_E = pred_E / pred_E.sum(dim=-1, keepdim=True)

        candidates = model._sample_k_candidates(pred_E, K, node_mask)

        # Check symmetry
        for k in range(K):
            for b in range(bs):
                candidate = candidates[k, b]
                assert torch.allclose(candidate, candidate.T), f"Candidate {k} batch {b} is not symmetric"

    def test_sample_k_candidates_valid_classes(self, hdc_guidance_setup):
        """Test that _sample_k_candidates produces valid edge class indices."""
        model, X, E, node_mask, hdc_vectors, original_node_features = hdc_guidance_setup
        bs, n = X.shape[:2]
        de = NUM_EDGE_CLASSES
        K = 4

        pred_E = torch.rand(bs, n, n, de)
        pred_E = pred_E / pred_E.sum(dim=-1, keepdim=True)

        candidates = model._sample_k_candidates(pred_E, K, node_mask)

        assert candidates.min() >= 0, "Edge classes should be >= 0"
        assert candidates.max() < de, f"Edge classes should be < {de}"

    def test_compute_Rhdc_shape(self, hdc_guidance_setup):
        """Test that _compute_Rhdc produces correct shape."""
        model, X, E, node_mask, hdc_vectors, original_node_features = hdc_guidance_setup
        bs, n = X.shape[:2]
        de = NUM_EDGE_CLASSES

        # Create mock inputs
        E_guide = torch.zeros(bs, n, n, de)
        E_guide[:, :, :, 0] = 1  # All no-edges

        E_t_label = torch.zeros(bs, n, n, dtype=torch.long)
        E_t_label[:, 0, 1] = 1  # Some single bonds
        E_t_label[:, 1, 0] = 1

        Z_t_E = torch.ones(bs, n, n) * de
        pt_at_Et = torch.ones(bs, n, n) * 0.5

        R_hdc = model._compute_Rhdc(E_guide, E_t_label, Z_t_E, pt_at_Et, gamma=1.0)

        assert R_hdc.shape == (bs, n, n, de), f"Expected shape ({bs}, {n}, {n}, {de}), got {R_hdc.shape}"

    def test_compute_Rhdc_zero_gamma(self, hdc_guidance_setup):
        """Test that _compute_Rhdc returns zeros when gamma=0."""
        model, X, E, node_mask, hdc_vectors, original_node_features = hdc_guidance_setup
        bs, n = X.shape[:2]
        de = NUM_EDGE_CLASSES

        E_guide = torch.rand(bs, n, n, de)
        E_guide = F.one_hot(E_guide.argmax(dim=-1), num_classes=de).float()

        E_t_label = torch.randint(0, de, (bs, n, n))
        Z_t_E = torch.ones(bs, n, n) * de
        pt_at_Et = torch.ones(bs, n, n) * 0.5

        R_hdc = model._compute_Rhdc(E_guide, E_t_label, Z_t_E, pt_at_Et, gamma=0.0)

        assert torch.allclose(R_hdc, torch.zeros_like(R_hdc)), "R_hdc should be zero when gamma=0"

    def test_compute_Rhdc_mask_correctness(self, hdc_guidance_setup):
        """Test that _compute_Rhdc masks correctly when E_guide == E_t."""
        model, X, E, node_mask, hdc_vectors, original_node_features = hdc_guidance_setup
        bs, n = X.shape[:2]
        de = NUM_EDGE_CLASSES

        # Create E_guide and E_t_label to be identical
        E_t_label = torch.zeros(bs, n, n, dtype=torch.long)
        E_guide = F.one_hot(E_t_label, num_classes=de).float()

        Z_t_E = torch.ones(bs, n, n) * de
        pt_at_Et = torch.ones(bs, n, n) * 0.5

        R_hdc = model._compute_Rhdc(E_guide, E_t_label, Z_t_E, pt_at_Et, gamma=1.0)

        # Should be zero because E_guide == E_t (mask is 0)
        assert torch.allclose(R_hdc, torch.zeros_like(R_hdc)), "R_hdc should be zero when E_guide == E_t"

    def test_compute_dfm_variables_for_edges_shape(self, hdc_guidance_setup):
        """Test that _compute_dfm_variables_for_edges produces correct shapes."""
        model, X, E, node_mask, hdc_vectors, original_node_features = hdc_guidance_setup
        bs, n = X.shape[:2]

        t = torch.tensor([[0.5]] * bs)
        E_t_label = torch.zeros(bs, n, n, dtype=torch.long)
        E_1_sampled = torch.randint(0, NUM_EDGE_CLASSES, (bs, n, n))

        dfm_vars = model._compute_dfm_variables_for_edges(t, E_t_label, E_1_sampled)

        assert "pt_vals_at_Et" in dfm_vars
        assert "Z_t_E" in dfm_vars
        assert dfm_vars["pt_vals_at_Et"].shape == (bs, n, n)
        assert dfm_vars["Z_t_E"].shape == (bs, n, n)

    def test_select_best_candidate_picks_closest(self, hdc_guidance_setup):
        """Test that _select_best_candidate picks the candidate closest to target."""
        model, X, E, node_mask, hdc_vectors, original_node_features = hdc_guidance_setup
        bs, n = X.shape[:2]

        K = 4
        hdc_dim = 32  # Small for test

        # Create candidates where one is clearly closest to target
        target_order_n = torch.randn(bs, hdc_dim)
        target_order_n = F.normalize(target_order_n, dim=-1)

        candidate_hdcs = torch.randn(K, bs, hdc_dim)
        candidate_hdcs = F.normalize(candidate_hdcs, dim=-1)

        # Make candidate 2 identical to target
        candidate_hdcs[2] = target_order_n

        candidates = torch.randint(0, NUM_EDGE_CLASSES, (K, bs, n, n))

        E_guide = model._select_best_candidate(candidate_hdcs, target_order_n, candidates)

        # E_guide should match candidates[2] (the closest one)
        expected = F.one_hot(candidates[2], num_classes=NUM_EDGE_CLASSES).float()
        assert torch.allclose(E_guide, expected), "Should select candidate 2 (identical to target)"


class TestHDCGuidedSamplingWithHyperNet:
    """Integration tests for HDC-guided sampling with real HyperNet and ZINC data."""

    @pytest.fixture
    def hypernet_for_hdc(self):
        """Create HyperNet for HDC guidance tests."""
        from graph_hdc.hypernet.configs import get_config
        from graph_hdc.hypernet.encoder import HyperNet

        config = get_config("ZINC_SMILES_HRR_256_F64_5G1NG4")
        config.device = "cpu"
        return HyperNet(config)

    @pytest.fixture
    def model_for_hdc(self, mock_edge_marginals, mock_node_counts):
        """Create FlowEdgeDecoder matching HyperNet dimensions."""
        return FlowEdgeDecoder(
            num_node_classes=NODE_FEATURE_DIM,
            num_edge_classes=NUM_EDGE_CLASSES,
            hdc_dim=512,  # 256 * 2 for [order_0 | order_N]
            n_layers=2,
            hidden_dim=32,
            hidden_mlp_dim=64,
            n_heads=2,
            dropout=0.0,
            noise_type="marginal",
            edge_marginals=mock_edge_marginals,
            node_counts=mock_node_counts,
            max_nodes=15,
            extra_features_type="rrwp",
            rrwp_steps=5,
            sample_steps=3,
            eta=0.0,
            omega=0.0,
        )

    @pytest.fixture
    def real_zinc_data(self):
        """Load real ZINC data for testing."""
        from rdkit import Chem
        from graph_hdc.datasets.zinc_smiles import mol_to_data

        # Create simple molecules that are valid in ZINC
        smiles_list = ["C", "CC", "CCC", "CCO"]
        data_list = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                data = mol_to_data(mol)
                data_list.append(data)
        return data_list

    def test_encode_candidates_with_real_data(self, hypernet_for_hdc, model_for_hdc, real_zinc_data):
        """Test _encode_candidates_to_order_n with real ZINC data."""
        hypernet = hypernet_for_hdc
        model = model_for_hdc

        # Use real data
        data = real_zinc_data[1]  # CC (ethane)
        n = data.x.size(0)
        bs = 1
        K = 2

        # Create candidates
        candidates = torch.randint(0, NUM_EDGE_CLASSES, (K, bs, n, n))
        for k in range(K):
            candidates[k, 0] = torch.triu(candidates[k, 0], diagonal=1)
            candidates[k, 0] = candidates[k, 0] + candidates[k, 0].T

        # Use real node features (5-dim raw features for ZINC)
        original_node_features = data.x.unsqueeze(0)  # (1, n, 5)
        node_mask = torch.ones(bs, n, dtype=torch.bool)

        order_n = model._encode_candidates_to_order_n(
            candidates, original_node_features, node_mask, hypernet
        )

        expected_hdc_dim = 256
        assert order_n.shape == (K, bs, expected_hdc_dim), f"Expected ({K}, {bs}, {expected_hdc_dim}), got {order_n.shape}"

    def test_sample_with_hdc_guidance_real_data(self, hypernet_for_hdc, model_for_hdc, real_zinc_data):
        """Test sample_with_hdc_guidance with real ZINC data."""
        from graph_hdc.utils.helpers import scatter_hd

        hypernet = hypernet_for_hdc
        model = model_for_hdc

        # Use real data
        data = real_zinc_data[1]  # CC (ethane)
        n = data.x.size(0)
        bs = 1

        # Compute HDC embedding for target
        data_for_hdc = data.clone()
        data_for_hdc.batch = torch.zeros(n, dtype=torch.long)
        with torch.no_grad():
            data_for_hdc = hypernet.encode_properties(data_for_hdc)
            order_zero = scatter_hd(src=data_for_hdc.node_hv, index=data_for_hdc.batch, op="bundle")
            output = hypernet.forward(data_for_hdc)
            order_n = output["graph_embedding"]
            hdc_vectors = torch.cat([order_zero, order_n], dim=-1)  # (1, 512)

        # Convert node features to 24-dim one-hot
        node_features = raw_features_to_onehot(data.x).unsqueeze(0)  # (1, n, 24)

        node_mask = torch.ones(bs, n, dtype=torch.bool)
        original_node_features = data.x.unsqueeze(0)  # (1, n, 5)

        samples = model.sample_with_hdc_guidance(
            hdc_vectors=hdc_vectors,
            node_features=node_features,
            node_mask=node_mask,
            original_node_features=original_node_features,
            hypernet=hypernet,
            gamma=1.0,
            num_candidates=2,
            sample_steps=2,
            show_progress=False,
        )

        assert len(samples) == bs
        assert samples[0].x is not None
        assert samples[0].edge_index is not None

    def test_sample_with_hdc_guidance_gamma_zero_runs(self, hypernet_for_hdc, model_for_hdc, real_zinc_data):
        """Test that sample_with_hdc_guidance runs with gamma=0 using real data."""
        from graph_hdc.utils.helpers import scatter_hd

        hypernet = hypernet_for_hdc
        model = model_for_hdc

        data = real_zinc_data[2]  # CCC (propane) - has edges
        n = data.x.size(0)
        bs = 1

        # Compute HDC embedding
        data_for_hdc = data.clone()
        data_for_hdc.batch = torch.zeros(n, dtype=torch.long)
        with torch.no_grad():
            data_for_hdc = hypernet.encode_properties(data_for_hdc)
            order_zero = scatter_hd(src=data_for_hdc.node_hv, index=data_for_hdc.batch, op="bundle")
            output = hypernet.forward(data_for_hdc)
            order_n = output["graph_embedding"]
            hdc_vectors = torch.cat([order_zero, order_n], dim=-1)

        # Convert node features to 24-dim one-hot
        node_features = raw_features_to_onehot(data.x).unsqueeze(0)  # (1, n, 24)

        node_mask = torch.ones(bs, n, dtype=torch.bool)
        original_node_features = data.x.unsqueeze(0)  # (1, n, 5)

        samples = model.sample_with_hdc_guidance(
            hdc_vectors=hdc_vectors,
            node_features=node_features,
            node_mask=node_mask,
            original_node_features=original_node_features,
            hypernet=hypernet,
            gamma=0.0,
            num_candidates=2,
            sample_steps=2,
            show_progress=False,
        )

        assert len(samples) == bs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
