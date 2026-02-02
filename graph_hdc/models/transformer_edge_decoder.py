"""
TransformerEdgeDecoder - Single-shot edge predictor conditioned on HDC vectors.

This module implements a direct edge prediction model that generates molecular
edges in a single forward pass through a GraphTransformer, conditioned on:
1. Pre-computed HDC vectors (512-dim default)
2. Fixed node types (7 atom classes)

Unlike FlowEdgeDecoder, this model:
- Uses single forward pass (no iterative denoising)
- No time embedding in global features
- Input edges are all-zeros (fully-connected with "no-edge" class)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch, Data

# DeFoG imports
from defog.core import (
    ExtraFeatures,
    GraphTransformer,
    PlaceHolder,
    dense_to_pyg,
    to_dense,
)

# Reuse constants and EdgeOnlyLoss from flow_edge_decoder
from graph_hdc.models.flow_edge_decoder import (
    ZINC_ATOM_TYPES,
    ZINC_ATOM_TO_IDX,
    ZINC_IDX_TO_ATOM,
    NODE_FEATURE_DIM,
    BOND_TYPES,
    NUM_EDGE_CLASSES,
    EdgeOnlyLoss,
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TransformerEdgeDecoderConfig:
    """Configuration for TransformerEdgeDecoder model."""

    # Data dimensions
    num_node_classes: int = NODE_FEATURE_DIM
    num_edge_classes: int = NUM_EDGE_CLASSES
    hdc_dim: int = 512
    condition_dim: int = 128  # Reduced dimension after MLP

    # Architecture
    n_layers: int = 6
    hidden_dim: int = 256
    hidden_mlp_dim: int = 512
    n_heads: int = 8
    dropout: float = 0.1

    # Graph sizes
    max_nodes: int = 50

    # Extra features
    extra_features_type: str = "rrwp"
    rrwp_steps: int = 10

    # Training
    lr: float = 1e-4
    weight_decay: float = 1e-5


# Dataset-specific default configurations
QM9_TRANSFORMER_DECODER_CONFIG = TransformerEdgeDecoderConfig(
    hdc_dim=512,
    n_layers=6,
    hidden_dim=256,
    max_nodes=15,
)

ZINC_TRANSFORMER_DECODER_CONFIG = TransformerEdgeDecoderConfig(
    hdc_dim=512,
    n_layers=8,
    hidden_dim=384,
    max_nodes=50,
)


# =============================================================================
# TransformerEdgeDecoder Model
# =============================================================================

class TransformerEdgeDecoder(pl.LightningModule):
    """
    Single-shot edge predictor using GraphTransformer conditioned on HDC vectors.

    Unlike FlowEdgeDecoder, this model:
    - Uses single forward pass (no iterative denoising)
    - No time embedding in global features
    - Input edges are all-zeros (no-edge class)

    Training: Given ground-truth nodes + HDC vector, predict edges
    Inference: Decode nodes from HDC order-0, then predict edges

    Parameters
    ----------
    num_node_classes : int
        Number of node classes (default: 7 for C, N, O, F, S, Cl, Br)
    num_edge_classes : int
        Number of edge classes (default: 5 for no-edge, single, double, triple, aromatic)
    hdc_dim : int
        Dimension of HDC conditioning vectors (default: 512)
    condition_dim : int
        Reduced dimension after MLP (default: 128)
    n_layers : int
        Number of transformer layers
    hidden_dim : int
        Hidden dimension for transformer
    hidden_mlp_dim : int
        MLP hidden dimension
    n_heads : int
        Number of attention heads
    dropout : float
        Dropout rate
    max_nodes : int
        Maximum number of nodes
    extra_features_type : str
        Type of extra features ("none", "rrwp", "cycles")
    rrwp_steps : int
        Number of RRWP steps
    lr : float
        Learning rate
    weight_decay : float
        Weight decay
    """

    def __init__(
        self,
        num_node_classes: int = NODE_FEATURE_DIM,
        num_edge_classes: int = NUM_EDGE_CLASSES,
        hdc_dim: int = 512,
        condition_dim: int = 128,
        n_layers: int = 6,
        hidden_dim: int = 256,
        hidden_mlp_dim: int = 512,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_nodes: int = 50,
        extra_features_type: str = "rrwp",
        rrwp_steps: int = 10,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Store dimensions
        self.num_node_classes = num_node_classes
        self.num_edge_classes = num_edge_classes
        self.hdc_dim = hdc_dim
        self.condition_dim = condition_dim
        self.max_nodes = max_nodes

        # Training config
        self.lr = lr
        self.weight_decay = weight_decay

        # Extra features (RRWP)
        self.extra_features = ExtraFeatures(
            feature_type=extra_features_type,
            rrwp_steps=rrwp_steps,
            max_nodes=max_nodes,
        )
        extra_dims = self.extra_features.output_dims()

        # MLP to reduce HDC conditioning dimension
        self.condition_mlp = nn.Sequential(
            nn.Linear(hdc_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, condition_dim),
        )

        # Input dimensions:
        # KEY DIFFERENCE FROM FlowEdgeDecoder:
        # - y: condition_dim + extra_y (NO TIME EMBEDDING - no +1)
        self.input_dims = {
            "X": num_node_classes + extra_dims["X"],
            "E": num_edge_classes + extra_dims["E"],
            "y": extra_dims["y"] + condition_dim,  # No time embedding!
        }

        # Output dimensions
        self.output_dims = {
            "X": num_node_classes,  # Not used but required by GraphTransformer
            "E": num_edge_classes,
            "y": 0,
        }

        # Hidden dimensions
        self.hidden_dims = {
            "dx": hidden_dim,
            "de": hidden_dim // 4,
            "dy": hidden_dim // 4,
            "n_head": n_heads,
            "dim_ffX": hidden_mlp_dim,
            "dim_ffE": hidden_mlp_dim // 4,
            "dim_ffy": hidden_mlp_dim // 4,
        }

        # Create GraphTransformer
        self.model = GraphTransformer(
            n_layers=n_layers,
            input_dims=self.input_dims,
            hidden_mlp_dims={
                "X": hidden_mlp_dim,
                "E": hidden_mlp_dim // 4,
                "y": hidden_mlp_dim // 4,
            },
            hidden_dims=self.hidden_dims,
            output_dims=self.output_dims,
            dropout=dropout,
        )

        # Loss function (edge-only)
        self.train_loss = EdgeOnlyLoss()

    def _create_fully_connected_edges(
        self,
        batch_size: int,
        num_nodes: int,
        device: torch.device,
    ) -> Tensor:
        """
        Create fully-connected edge input with all edges set to class 0 (no-edge).

        Args:
            batch_size: Number of graphs in batch
            num_nodes: Maximum number of nodes
            device: Target device

        Returns:
            E_input: [batch_size, num_nodes, num_nodes, num_edge_classes] one-hot
        """
        # Create all-zeros tensor, then set class 0 to 1 (one-hot for no-edge)
        E_input = torch.zeros(
            batch_size, num_nodes, num_nodes, self.num_edge_classes,
            device=device, dtype=torch.float32
        )
        E_input[..., 0] = 1.0  # All edges initialized as "no-edge" class
        return E_input

    def _compute_extra_data(
        self,
        data_dict: Dict[str, Tensor],
    ) -> PlaceHolder:
        """Compute extra features (RRWP) for the current state."""
        return self.extra_features(data_dict)

    def _get_hdc_vectors_from_batch(self, batch: Batch) -> Tensor:
        """Extract per-graph HDC vectors from batch."""
        if hasattr(batch, "hdc_vector"):
            hdc = batch.hdc_vector
            if hdc.dim() == 2:
                return hdc
            else:
                return hdc.unsqueeze(0)
        else:
            raise ValueError(
                "Batch does not have 'hdc_vector' attribute. "
                "Make sure to preprocess data with preprocess_for_flow_edge_decoder()."
            )

    def forward(
        self,
        X: Tensor,
        hdc_vectors: Tensor,
        node_mask: Tensor,
    ) -> PlaceHolder:
        """
        Single forward pass to predict edges.

        Args:
            X: Node features one-hot (bs, n, num_node_classes)
            hdc_vectors: HDC conditioning vectors (bs, hdc_dim)
            node_mask: Boolean mask for valid nodes (bs, n)

        Returns:
            PlaceHolder with .E containing edge logits (bs, n, n, num_edge_classes)
        """
        bs, n = X.shape[:2]
        device = X.device

        # 1. Create fully-connected edge input (all no-edge)
        E_input = self._create_fully_connected_edges(bs, n, device)

        # 2. Apply node mask to edges
        edge_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
        E_input = E_input * edge_mask.unsqueeze(-1).float()

        # 3. Compute HDC conditioning
        y_cond = self.condition_mlp(hdc_vectors)  # (bs, condition_dim)

        # 4. Compute extra features (needs data dict format for ExtraFeatures)
        # Note: ExtraFeatures expects y_t in data_dict, pass empty tensor
        data_dict = {
            "X_t": X,
            "E_t": E_input,
            "y_t": torch.zeros(bs, 0, device=device),  # Placeholder for ExtraFeatures
            "node_mask": node_mask,
        }
        extra_data = self._compute_extra_data(data_dict)

        # 5. Concatenate inputs
        X_full = torch.cat([X, extra_data.X], dim=-1)
        E_full = torch.cat([E_input, extra_data.E], dim=-1)
        y_full = torch.cat([y_cond, extra_data.y], dim=-1)  # NO time embedding!

        # 6. Forward through GraphTransformer
        pred = self.model(X_full, E_full, y_full, node_mask)

        return pred

    def training_step(
        self,
        batch: Batch,
        batch_idx: int,
    ) -> Optional[Dict[str, Tensor]]:
        """
        Training step.

        Expects batch to have:
        - x: Node features one-hot (7 classes)
        - edge_index: Edge indices
        - edge_attr: Edge features one-hot (5 classes)
        - hdc_vector: Pre-computed HDC embeddings
        """
        if batch.edge_index.numel() == 0:
            return None

        # Convert to dense format
        dense_data, node_mask = to_dense(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
        )
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E

        # Get HDC vectors
        hdc_vectors = self._get_hdc_vectors_from_batch(batch)

        # Forward pass (single shot - no noise, no time)
        pred = self.forward(X, hdc_vectors, node_mask)

        # Compute edge-only loss
        loss = self.train_loss(pred.E, E, node_mask)

        self.log("train/loss", loss.detach(), prog_bar=True, batch_size=batch.num_graphs)
        return {"loss": loss}

    def validation_step(
        self,
        batch: Batch,
        batch_idx: int,
    ) -> Optional[Dict[str, Tensor]]:
        """Validation step."""
        if batch.edge_index.numel() == 0:
            return None

        dense_data, node_mask = to_dense(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
        )
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E

        hdc_vectors = self._get_hdc_vectors_from_batch(batch)
        pred = self.forward(X, hdc_vectors, node_mask)
        loss = self.train_loss(pred.E, E, node_mask)

        self.log("val/loss", loss.detach(), prog_bar=True, batch_size=batch.num_graphs)
        return {"val_loss": loss.detach()}

    @torch.no_grad()
    def sample(
        self,
        hdc_vectors: Tensor,
        node_features: Tensor,
        node_mask: Tensor,
        device: Optional[torch.device] = None,
    ) -> List[Data]:
        """
        Generate edges conditioned on HDC vectors and fixed nodes.

        Single forward pass - no iterative denoising.

        Args:
            hdc_vectors: HDC vectors (num_samples, hdc_dim)
            node_features: Node types one-hot (num_samples, max_n, num_node_classes)
            node_mask: Valid node mask (num_samples, max_n)
            device: Device for computation

        Returns:
            List of PyG Data objects with generated edges
        """
        self.eval()

        device = device or next(self.parameters()).device
        self.to(device)

        hdc_vectors = hdc_vectors.to(device).float()
        X = node_features.to(device).float()
        node_mask = node_mask.to(device)

        # Single forward pass
        pred = self.forward(X, hdc_vectors, node_mask)

        # Get edge predictions (softmax then argmax)
        E_pred = F.softmax(pred.E, dim=-1)

        # Make symmetric (average upper and lower triangle, then argmax)
        E_pred_sym = (E_pred + E_pred.transpose(1, 2)) / 2
        E_labels = torch.argmax(E_pred_sym, dim=-1)

        # Convert to one-hot for dense_to_pyg
        E_onehot = F.one_hot(E_labels, num_classes=self.num_edge_classes).float()

        # Convert to PyG Data objects
        n_nodes = node_mask.sum(dim=1).int()
        samples = dense_to_pyg(X, E_onehot, torch.zeros_like(hdc_vectors[:, :0]), node_mask, n_nodes)

        return samples

    @torch.no_grad()
    def predict_edges(
        self,
        hdc_vectors: Tensor,
        node_features: Tensor,
        node_mask: Tensor,
        return_probs: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Predict edge labels given HDC vectors and node features.

        Args:
            hdc_vectors: HDC vectors (bs, hdc_dim)
            node_features: Node types one-hot (bs, n, num_node_classes)
            node_mask: Valid node mask (bs, n)
            return_probs: If True, also return probability tensor

        Returns:
            E_labels: Predicted edge labels (bs, n, n)
            E_probs: Edge probabilities (bs, n, n, num_edge_classes) if return_probs=True
        """
        self.eval()

        pred = self.forward(node_features, hdc_vectors, node_mask)
        E_probs = F.softmax(pred.E, dim=-1)

        # Make symmetric
        E_probs_sym = (E_probs + E_probs.transpose(1, 2)) / 2
        E_labels = torch.argmax(E_probs_sym, dim=-1)

        if return_probs:
            return E_labels, E_probs_sym
        return E_labels, None

    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer

    def save(self, path: str) -> Path:
        """Save model checkpoint."""
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".ckpt")

        checkpoint = {
            "state_dict": self.state_dict(),
            "hyper_parameters": dict(self.hparams),
        }
        torch.save(checkpoint, path)
        return path

    @classmethod
    def load(
        cls,
        path: str,
        device: Optional[torch.device] = None,
    ) -> "TransformerEdgeDecoder":
        """Load model from checkpoint."""
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".ckpt")

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        hparams = checkpoint["hyper_parameters"]

        model = cls(**hparams)
        model.load_state_dict(checkpoint["state_dict"])

        if device is not None:
            model = model.to(device)

        return model
