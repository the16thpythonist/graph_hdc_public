"""
Unit tests for TransformerEdgeDecoder model.

Tests individual components with mock data to ensure the model works correctly.
TransformerEdgeDecoder is simpler than FlowEdgeDecoder - no iterative denoising,
no time embedding, single forward pass.
"""

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from copy import deepcopy

from graph_hdc.models.flow_edge_decoder import (
    # Constants (reused)
    NODE_FEATURE_DIM,
    NUM_EDGE_CLASSES,
    ZINC_ATOM_TYPES,
    BOND_TYPES,
    EdgeOnlyLoss,
    raw_features_to_onehot,
)
from graph_hdc.models.transformer_edge_decoder import (
    TransformerEdgeDecoder,
    TransformerEdgeDecoderConfig,
)


# =============================================================================
# Fixtures for mock data
# =============================================================================

@pytest.fixture
def small_model():
    """Create a small TransformerEdgeDecoder for testing."""
    return TransformerEdgeDecoder(
        num_node_classes=NODE_FEATURE_DIM,
        num_edge_classes=NUM_EDGE_CLASSES,
        hdc_dim=64,  # Small for fast tests
        condition_dim=32,
        n_layers=2,
        hidden_dim=32,
        hidden_mlp_dim=64,
        n_heads=2,
        dropout=0.0,
        max_nodes=15,
        extra_features_type="rrwp",
        rrwp_steps=5,
        lr=1e-4,
        weight_decay=0.0,
    )


@pytest.fixture
def mock_dense_batch():
    """Create mock dense batch data for testing."""
    bs, n, dx, de = 2, 8, NODE_FEATURE_DIM, NUM_EDGE_CLASSES

    # Node features: one-hot encoded atom types
    X = torch.zeros(bs, n, dx)
    X[0, :5, 0] = 1  # 5 carbon atoms in first graph
    X[0, 5:7, 1] = 1  # 2 nitrogen atoms
    X[1, :4, 0] = 1  # 4 carbons in second graph
    X[1, 4:6, 2] = 1  # 2 oxygen atoms

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

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            hdc_vector=mock_dense_batch["hdc_vectors"][i:i+1],  # Keep as (1, dim)
        )
        data_list.append(data)

    return Batch.from_data_list(data_list)


# =============================================================================
# Test TransformerEdgeDecoder Creation
# =============================================================================

class TestTransformerEdgeDecoderCreation:
    """Tests for TransformerEdgeDecoder model creation."""

    def test_creation(self, small_model):
        """Test model creation."""
        assert small_model is not None
        assert isinstance(small_model, TransformerEdgeDecoder)

    def test_model_attributes(self, small_model):
        """Test model has expected attributes."""
        assert small_model.num_node_classes == NODE_FEATURE_DIM
        assert small_model.num_edge_classes == NUM_EDGE_CLASSES
        assert small_model.hdc_dim == 64
        assert hasattr(small_model, "model")  # GraphTransformer
        assert hasattr(small_model, "train_loss")  # EdgeOnlyLoss
        assert hasattr(small_model, "condition_mlp")  # HDC conditioning MLP

    def test_input_output_dims(self, small_model):
        """Test input/output dimensions are set correctly."""
        assert "X" in small_model.input_dims
        assert "E" in small_model.input_dims
        assert "y" in small_model.input_dims

        assert small_model.output_dims["X"] == NODE_FEATURE_DIM
        assert small_model.output_dims["E"] == NUM_EDGE_CLASSES

    def test_no_time_in_y_dims(self, small_model):
        """Test that y dimension does NOT include time embedding."""
        # In FlowEdgeDecoder: y = 1 + extra + condition_dim
        # In TransformerEdgeDecoder: y = extra + condition_dim (no +1 for time)
        extra_dims = small_model.extra_features.output_dims()
        expected_y_dim = extra_dims["y"] + small_model.condition_dim
        assert small_model.input_dims["y"] == expected_y_dim


# =============================================================================
# Test TransformerEdgeDecoder Forward Pass
# =============================================================================

class TestTransformerEdgeDecoderForward:
    """Tests for TransformerEdgeDecoder forward pass."""

    def test_forward_output_shape(self, small_model, mock_dense_batch):
        """Test forward pass returns correct shapes."""
        small_model.eval()

        X = mock_dense_batch["X"]
        hdc = mock_dense_batch["hdc_vectors"]
        node_mask = mock_dense_batch["node_mask"]

        with torch.no_grad():
            pred = small_model.forward(X, hdc, node_mask)

        bs, n = X.shape[:2]
        assert pred.X.shape == (bs, n, NODE_FEATURE_DIM)
        assert pred.E.shape == (bs, n, n, NUM_EDGE_CLASSES)

    def test_forward_no_nan(self, small_model, mock_dense_batch):
        """Test forward pass produces no NaN values."""
        small_model.eval()

        X = mock_dense_batch["X"]
        hdc = mock_dense_batch["hdc_vectors"]
        node_mask = mock_dense_batch["node_mask"]

        with torch.no_grad():
            pred = small_model.forward(X, hdc, node_mask)

        assert not torch.isnan(pred.X).any()
        assert not torch.isnan(pred.E).any()
        assert not torch.isinf(pred.X).any()
        assert not torch.isinf(pred.E).any()

    def test_forward_single_shot(self, small_model, mock_dense_batch):
        """Test that forward is a single pass (no iterative denoising)."""
        small_model.eval()

        X = mock_dense_batch["X"]
        hdc = mock_dense_batch["hdc_vectors"]
        node_mask = mock_dense_batch["node_mask"]

        # Forward should work directly without any time parameter
        with torch.no_grad():
            pred = small_model.forward(X, hdc, node_mask)

        # Should get predictions directly
        assert pred.E is not None


# =============================================================================
# Test Fully-Connected Edge Creation
# =============================================================================

class TestFullyConnectedEdges:
    """Tests for _create_fully_connected_edges method."""

    def test_creates_correct_shape(self, small_model):
        """Test that fully connected edges have correct shape."""
        bs, n = 2, 8
        device = torch.device("cpu")

        E = small_model._create_fully_connected_edges(bs, n, device)

        assert E.shape == (bs, n, n, NUM_EDGE_CLASSES)

    def test_all_no_edge_class(self, small_model):
        """Test that all edges are initialized as class 0 (no-edge)."""
        bs, n = 2, 8
        device = torch.device("cpu")

        E = small_model._create_fully_connected_edges(bs, n, device)

        # Class 0 should be 1, others should be 0
        assert (E[..., 0] == 1).all()
        assert (E[..., 1:] == 0).all()

    def test_one_hot_encoding(self, small_model):
        """Test that edges are valid one-hot encoding."""
        bs, n = 2, 8
        device = torch.device("cpu")

        E = small_model._create_fully_connected_edges(bs, n, device)

        # Each edge should sum to 1
        edge_sums = E.sum(dim=-1)
        assert torch.allclose(edge_sums, torch.ones_like(edge_sums))


# =============================================================================
# Test TransformerEdgeDecoder Training/Validation Steps
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
# Test Sample and Predict Methods
# =============================================================================

class TestSampleAndPredict:
    """Tests for sample and predict_edges methods."""

    def test_sample_returns_list(self, small_model, mock_dense_batch):
        """Test sample returns list of Data objects."""
        small_model.eval()

        samples = small_model.sample(
            hdc_vectors=mock_dense_batch["hdc_vectors"],
            node_features=mock_dense_batch["X"],
            node_mask=mock_dense_batch["node_mask"],
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
        )

        # Node counts should match mask
        expected_counts = mock_dense_batch["node_mask"].sum(dim=1).tolist()
        actual_counts = [s.num_nodes for s in samples]

        assert actual_counts == expected_counts

    def test_predict_edges_shape(self, small_model, mock_dense_batch):
        """Test predict_edges returns correct shape."""
        small_model.eval()

        E_labels, _ = small_model.predict_edges(
            hdc_vectors=mock_dense_batch["hdc_vectors"],
            node_features=mock_dense_batch["X"],
            node_mask=mock_dense_batch["node_mask"],
            return_probs=False,
        )

        bs, n = mock_dense_batch["X"].shape[:2]
        assert E_labels.shape == (bs, n, n)

    def test_predict_edges_with_probs(self, small_model, mock_dense_batch):
        """Test predict_edges returns probabilities when requested."""
        small_model.eval()

        E_labels, E_probs = small_model.predict_edges(
            hdc_vectors=mock_dense_batch["hdc_vectors"],
            node_features=mock_dense_batch["X"],
            node_mask=mock_dense_batch["node_mask"],
            return_probs=True,
        )

        bs, n = mock_dense_batch["X"].shape[:2]
        assert E_labels.shape == (bs, n, n)
        assert E_probs is not None
        assert E_probs.shape == (bs, n, n, NUM_EDGE_CLASSES)

    def test_predict_edges_symmetric(self, small_model, mock_dense_batch):
        """Test that predicted edges are symmetric."""
        small_model.eval()

        E_labels, _ = small_model.predict_edges(
            hdc_vectors=mock_dense_batch["hdc_vectors"],
            node_features=mock_dense_batch["X"],
            node_mask=mock_dense_batch["node_mask"],
            return_probs=False,
        )

        # Check symmetry
        assert torch.allclose(E_labels, E_labels.transpose(1, 2))


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

        loaded = TransformerEdgeDecoder.load(str(path))

        assert loaded is not None
        assert loaded.hdc_dim == small_model.hdc_dim
        assert loaded.condition_dim == small_model.condition_dim

    def test_load_matches_original(self, small_model, mock_dense_batch, tmp_path):
        """Test loaded model produces same outputs as original."""
        path = tmp_path / "test_model.ckpt"
        small_model.save(str(path))

        loaded = TransformerEdgeDecoder.load(str(path))
        loaded.eval()
        small_model.eval()

        X = mock_dense_batch["X"]
        hdc = mock_dense_batch["hdc_vectors"]
        node_mask = mock_dense_batch["node_mask"]

        with torch.no_grad():
            pred_original = small_model.forward(X, hdc, node_mask)
            pred_loaded = loaded.forward(X, hdc, node_mask)

        assert torch.allclose(pred_original.E, pred_loaded.E)


# =============================================================================
# Test Configuration
# =============================================================================

class TestConfiguration:
    """Tests for TransformerEdgeDecoderConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TransformerEdgeDecoderConfig()

        assert config.num_node_classes == NODE_FEATURE_DIM
        assert config.num_edge_classes == NUM_EDGE_CLASSES
        assert config.hdc_dim == 512

    def test_config_to_model(self):
        """Test creating model from config."""
        config = TransformerEdgeDecoderConfig(
            hdc_dim=64,
            n_layers=2,
            hidden_dim=32,
        )

        model = TransformerEdgeDecoder(
            num_node_classes=config.num_node_classes,
            num_edge_classes=config.num_edge_classes,
            hdc_dim=config.hdc_dim,
            condition_dim=config.condition_dim,
            n_layers=config.n_layers,
            hidden_dim=config.hidden_dim,
            hidden_mlp_dim=config.hidden_mlp_dim,
            n_heads=config.n_heads,
            dropout=config.dropout,
            max_nodes=config.max_nodes,
            extra_features_type=config.extra_features_type,
            rrwp_steps=config.rrwp_steps,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        assert model.hdc_dim == 64


# =============================================================================
# Test HDC Vector Handling
# =============================================================================

class TestHDCVectorHandling:
    """Tests for HDC vector handling in batching."""

    def test_hdc_vector_batch_shape(self, mock_pyg_batch):
        """Test HDC vectors batch correctly."""
        assert mock_pyg_batch.hdc_vector.dim() == 2
        assert mock_pyg_batch.hdc_vector.shape[0] == 2  # batch size

    def test_get_hdc_vectors_from_batch(self, small_model, mock_pyg_batch):
        """Test _get_hdc_vectors_from_batch extracts correct shape."""
        hdc = small_model._get_hdc_vectors_from_batch(mock_pyg_batch)

        assert hdc.dim() == 2
        assert hdc.shape[0] == 2  # batch size
        assert hdc.shape[1] == 64  # hdc_dim


# =============================================================================
# End-to-End Test
# =============================================================================

class TestEndToEnd:
    """End-to-end tests for TransformerEdgeDecoder."""

    @pytest.fixture
    def hypernet_and_model(self):
        """Create HyperNet encoder and TransformerEdgeDecoder for e2e testing."""
        from collections import OrderedDict
        import math
        from graph_hdc.hypernet.configs import DSHDCConfig, FeatureConfig, Features, IndexRange
        from graph_hdc.hypernet.feature_encoders import CombinatoricIntegerEncoder
        from graph_hdc.hypernet.types import VSAModel
        from graph_hdc.hypernet.encoder import HyperNet

        # Create HyperNet config for ZINC (bins: atom=9, degree=6, charge=3, Hs=4, ring=2)
        node_feature_config = FeatureConfig(
            count=math.prod([9, 6, 3, 4, 2]),
            encoder_cls=CombinatoricIntegerEncoder,
            index_range=IndexRange((0, 5)),
            bins=[9, 6, 3, 4, 2],
        )
        hdc_config = DSHDCConfig(
            name='ZINC_HRR_64_test',
            hv_dim=64,
            vsa=VSAModel.HRR,
            base_dataset='zinc',
            hypernet_depth=2,
            device='cpu',
            seed=42,
            normalize=True,
            dtype='float32',
            node_feature_configs=OrderedDict([(Features.NODE_FEATURES, node_feature_config)]),
        )
        hypernet = HyperNet(hdc_config)

        # Create TransformerEdgeDecoder (hdc_dim = 64 * 2 for [order_0 | order_N])
        model = TransformerEdgeDecoder(
            num_node_classes=NODE_FEATURE_DIM,
            num_edge_classes=NUM_EDGE_CLASSES,
            hdc_dim=128,  # 64 * 2
            condition_dim=32,
            n_layers=2,
            hidden_dim=32,
            hidden_mlp_dim=64,
            n_heads=2,
            dropout=0.0,
            max_nodes=15,
            extra_features_type="rrwp",
            rrwp_steps=5,
            lr=1e-4,
            weight_decay=0.0,
        )

        return hypernet, model

    @pytest.fixture
    def sample_molecule_data(self):
        """Create sample molecule data for testing."""
        from graph_hdc.datasets.utils import get_split

        # Use ZINC dataset (ZINC-only node features)
        train_ds = get_split('train', dataset='zinc')
        return train_ds[:3]

    def test_encode_decode_pipeline(self, hypernet_and_model, sample_molecule_data):
        """Test full encode -> decode pipeline produces valid output."""
        from graph_hdc.models.flow_edge_decoder import preprocess_dataset
        from graph_hdc.utils.helpers import scatter_hd
        from defog.core import to_dense

        hypernet, model = hypernet_and_model
        model.eval()

        # Preprocess molecules (encodes with HyperNet)
        # Note: preprocess_dataset expects hdc_dim = hypernet.hv_dim, but we need 2x
        # So we manually encode here
        processed = []
        for data in sample_molecule_data:
            if data.edge_index.numel() == 0:
                continue

            # Encode with HyperNet
            data_copy = data.clone()
            n = data_copy.x.size(0)
            data_copy.batch = torch.zeros(n, dtype=torch.long)

            with torch.no_grad():
                data_copy = hypernet.encode_properties(data_copy)
                order_zero = scatter_hd(src=data_copy.node_hv, index=data_copy.batch, op="bundle")
                output = hypernet.forward(data_copy)
                order_n = output["graph_embedding"]
                hdc_vector = torch.cat([order_zero, order_n], dim=-1).float()

            # Convert to 24-dim concatenated one-hot node features
            x_24dim = raw_features_to_onehot(data.x)

            # Convert edge attributes
            edge_attr = F.one_hot(
                torch.zeros(data.edge_index.size(1), dtype=torch.long),
                num_classes=NUM_EDGE_CLASSES
            ).float()
            if data.edge_index.size(1) > 0:
                # Simple: assume single bonds
                edge_attr[:, 1] = 1
                edge_attr[:, 0] = 0

            new_data = Data(
                x=x_24dim,
                edge_index=data.edge_index,
                edge_attr=edge_attr,
                hdc_vector=hdc_vector,
            )
            processed.append(new_data)

        assert len(processed) > 0

        # Create batch
        batch = Batch.from_data_list(processed)

        # Get dense format for sampling
        dense_data, node_mask = to_dense(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )

        # Decode (predict edges given HDC vectors and node features)
        with torch.no_grad():
            samples = model.sample(
                hdc_vectors=batch.hdc_vector,
                node_features=dense_data.X,
                node_mask=node_mask,
            )

        assert len(samples) == len(processed)

        for i, sample in enumerate(samples):
            assert isinstance(sample, Data)
            assert hasattr(sample, 'x')
            assert hasattr(sample, 'edge_index')

    def test_training_on_real_data(self, hypernet_and_model, sample_molecule_data):
        """Test that training step works on real preprocessed data."""
        from graph_hdc.utils.helpers import scatter_hd

        hypernet, model = hypernet_and_model

        processed = []
        for data in sample_molecule_data:
            if data.edge_index.numel() == 0:
                continue

            data_copy = data.clone()
            n = data_copy.x.size(0)
            data_copy.batch = torch.zeros(n, dtype=torch.long)

            with torch.no_grad():
                data_copy = hypernet.encode_properties(data_copy)
                order_zero = scatter_hd(src=data_copy.node_hv, index=data_copy.batch, op="bundle")
                output = hypernet.forward(data_copy)
                order_n = output["graph_embedding"]
                hdc_vector = torch.cat([order_zero, order_n], dim=-1).float()

            # Convert to 24-dim concatenated one-hot node features
            x_24dim = raw_features_to_onehot(data.x)

            edge_attr = torch.zeros(data.edge_index.size(1), NUM_EDGE_CLASSES)
            edge_attr[:, 1] = 1  # Assume single bonds

            new_data = Data(
                x=x_24dim,
                edge_index=data.edge_index,
                edge_attr=edge_attr,
                hdc_vector=hdc_vector,
            )
            processed.append(new_data)

        batch = Batch.from_data_list(processed)

        result = model.training_step(batch, 0)

        assert result is not None
        assert "loss" in result
        assert torch.isfinite(result["loss"])
        assert result["loss"].requires_grad


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
