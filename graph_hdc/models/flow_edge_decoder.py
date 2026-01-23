"""
FlowEdgeDecoder - Edge-only DeFoG decoder conditioned on HDC vectors.

This module implements a discrete flow matching model that generates molecular
edges conditioned on:
1. Pre-computed HDC vectors (512-dim default)
2. Fixed node types (7 atom classes)

The model inherits from DeFoG's DeFoGModel and overrides the training and
sampling methods to:
- Keep nodes fixed (no noise applied)
- Only denoise edges through the flow matching process
- Use HDC vectors as global conditioning
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

# DeFoG imports (assumes DeFoG is installed)
from defog.core import (
    ExtraFeatures,
    GraphTransformer,
    LimitDistribution,
    PlaceHolder,
    RateMatrixDesigner,
    TimeDistorter,
    dense_to_pyg,
    sample_from_probs,
    to_dense,
)

# =============================================================================
# Constants
# =============================================================================

# 7-class atom type mapping (C, N, O, F, S, Cl, Br)
FLOW_ATOM_TYPES = ["C", "N", "O", "F", "S", "Cl", "Br"]
FLOW_ATOM_TO_IDX = {atom: idx for idx, atom in enumerate(FLOW_ATOM_TYPES)}
FLOW_IDX_TO_ATOM = {idx: atom for atom, idx in FLOW_ATOM_TO_IDX.items()}
NUM_ATOM_CLASSES = 7

# 5-class edge type mapping
BOND_TYPES = ["no_edge", "single", "double", "triple", "aromatic"]
BOND_TYPE_TO_IDX = {
    None: 0,  # No edge
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
    Chem.BondType.AROMATIC: 4,
}
NUM_EDGE_CLASSES = 5

# Dataset-specific atom mappings to 7-class
QM9_TO_7CLASS = {0: 0, 1: 1, 2: 2, 3: 3}  # C, N, O, F -> direct
ZINC_TO_7CLASS = {
    0: 6,   # Br -> 6
    1: 0,   # C -> 0
    2: 5,   # Cl -> 5
    3: 3,   # F -> 3
    # 4: I -> skip
    5: 1,   # N -> 1
    6: 2,   # O -> 2
    # 7: P -> skip
    8: 4,   # S -> 4
}


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class FlowEdgeDecoderConfig:
    """Configuration for FlowEdgeDecoder model."""

    # Data dimensions
    num_node_classes: int = NUM_ATOM_CLASSES
    num_edge_classes: int = NUM_EDGE_CLASSES
    hdc_dim: int = 512

    # Architecture
    n_layers: int = 6
    hidden_dim: int = 256
    hidden_mlp_dim: int = 512
    n_heads: int = 8
    dropout: float = 0.1

    # Noise
    noise_type: str = "marginal"

    # Graph sizes
    max_nodes: int = 50

    # Extra features
    extra_features_type: str = "rrwp"
    rrwp_steps: int = 10

    # Training
    lr: float = 1e-4
    weight_decay: float = 1e-5
    train_time_distortion: str = "identity"

    # Sampling
    sample_steps: int = 100
    eta: float = 0.0
    omega: float = 0.0
    sample_time_distortion: str = "polydec"


QM9_EDGE_DECODER_CONFIG = FlowEdgeDecoderConfig(
    hdc_dim=512,
    n_layers=6,
    hidden_dim=256,
    max_nodes=15,
)

ZINC_EDGE_DECODER_CONFIG = FlowEdgeDecoderConfig(
    hdc_dim=512,
    n_layers=8,
    hidden_dim=384,
    max_nodes=50,
)


# =============================================================================
# Edge-Only Loss
# =============================================================================

class EdgeOnlyLoss(nn.Module):
    """Cross-entropy loss for edges only (nodes are fixed)."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred_E: Tensor,
        true_E: Tensor,
        node_mask: Tensor,
    ) -> Tensor:
        """
        Compute edge-only cross-entropy loss.

        Args:
            pred_E: Predicted edge logits (bs, n, n, de)
            true_E: True edge features one-hot (bs, n, n, de)
            node_mask: Boolean mask (bs, n)

        Returns:
            Scalar loss
        """
        bs, n, _, de = pred_E.shape

        # Create edge mask from node mask (only valid node pairs)
        edge_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)  # (bs, n, n)
        # Exclude diagonal
        diag_mask = ~torch.eye(n, dtype=torch.bool, device=pred_E.device).unsqueeze(0)
        edge_mask = edge_mask & diag_mask

        if not edge_mask.any():
            return torch.tensor(0.0, device=pred_E.device, requires_grad=True)

        # Flatten and select valid edges
        pred_flat = pred_E[edge_mask]  # (num_valid, de)
        true_flat = true_E[edge_mask]  # (num_valid, de)

        # Cross-entropy with soft targets
        log_probs = F.log_softmax(pred_flat, dim=-1)
        loss = -(true_flat * log_probs).sum(dim=-1).mean()

        return loss


# =============================================================================
# Distribution for Node Counts
# =============================================================================

class DistributionNodes:
    """Distribution over graph sizes."""

    def __init__(self, histogram: Optional[Tensor] = None):
        if histogram is None:
            histogram = torch.ones(50)
        self.histogram = histogram / histogram.sum()
        self.n_nodes = torch.arange(len(histogram))

    def sample(self, n_samples: int, device: torch.device) -> Tensor:
        """Sample graph sizes."""
        probs = self.histogram.to(device)
        return torch.multinomial(probs, n_samples, replacement=True)

    def to(self, device: torch.device) -> "DistributionNodes":
        self.histogram = self.histogram.to(device)
        self.n_nodes = self.n_nodes.to(device)
        return self


# =============================================================================
# FlowEdgeDecoder Model
# =============================================================================

class FlowEdgeDecoder(pl.LightningModule):
    """
    Edge-only DeFoG decoder conditioned on HDC vectors.

    This model generates molecular edges given:
    - Pre-computed HDC vectors as global conditioning
    - Fixed node types (not modified during generation)

    The model uses discrete flow matching to denoise edges while keeping
    nodes fixed throughout the process.

    Parameters
    ----------
    num_node_classes : int
        Number of node classes (default: 7 for C, N, O, F, S, Cl, Br)
    num_edge_classes : int
        Number of edge classes (default: 5 for no-edge, single, double, triple, aromatic)
    hdc_dim : int
        Dimension of HDC conditioning vectors (default: 512)
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
    noise_type : str
        Type of noise distribution ("uniform", "marginal", "absorbing")
    edge_marginals : Tensor, optional
        Empirical edge class marginals for marginal noise
    node_counts : Tensor, optional
        Distribution of graph sizes
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
    train_time_distortion : str
        Time distortion for training
    sample_steps : int
        Number of sampling steps
    eta : float
        Stochasticity parameter for sampling
    omega : float
        Target guidance strength for sampling
    sample_time_distortion : str
        Time distortion for sampling
    """

    def __init__(
        self,
        num_node_classes: int = NUM_ATOM_CLASSES,
        num_edge_classes: int = NUM_EDGE_CLASSES,
        hdc_dim: int = 512,
        n_layers: int = 6,
        hidden_dim: int = 256,
        hidden_mlp_dim: int = 512,
        n_heads: int = 8,
        dropout: float = 0.1,
        noise_type: str = "marginal",
        edge_marginals: Optional[Tensor] = None,
        node_counts: Optional[Tensor] = None,
        max_nodes: int = 50,
        extra_features_type: str = "rrwp",
        rrwp_steps: int = 10,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        train_time_distortion: str = "identity",
        sample_steps: int = 100,
        eta: float = 0.0,
        omega: float = 0.0,
        sample_time_distortion: str = "polydec",
    ):
        super().__init__()
        self.save_hyperparameters()

        # Store dimensions
        self.num_node_classes = num_node_classes
        self.num_edge_classes = num_edge_classes
        self.hdc_dim = hdc_dim
        self.max_nodes = max_nodes

        # Training config
        self.lr = lr
        self.weight_decay = weight_decay

        # Sampling config
        self.default_sample_steps = sample_steps
        self.default_eta = eta
        self.default_omega = omega
        self.default_sample_time_distortion = sample_time_distortion

        # Create limit distribution for edges
        # For nodes, use uniform (won't be used since nodes are fixed)
        if edge_marginals is None:
            # Default: mostly no-edge
            edge_marginals = torch.tensor([0.9, 0.05, 0.025, 0.015, 0.01])

        # Create dummy uniform node marginals (required by LimitDistribution but not used)
        # since nodes are fixed in this model
        node_marginals = torch.ones(num_node_classes) / num_node_classes

        self.limit_dist = LimitDistribution(
            noise_type=noise_type,
            num_node_classes=num_node_classes,
            num_edge_classes=num_edge_classes,
            node_marginals=node_marginals,
            edge_marginals=edge_marginals,
        )

        # Node distribution for graph sizes
        self.node_dist = DistributionNodes(node_counts)

        # Extra features
        self.extra_features = ExtraFeatures(
            feature_type=extra_features_type,
            rrwp_steps=rrwp_steps,
            max_nodes=max_nodes,
        )
        extra_dims = self.extra_features.output_dims()

        # Input dimensions: nodes + edges + global (time + HDC)
        self.input_dims = {
            "X": num_node_classes + extra_dims["X"],
            "E": num_edge_classes + extra_dims["E"],
            "y": 1 + extra_dims["y"] + hdc_dim,  # time + extra + HDC
        }

        # Output dimensions
        self.output_dims = {
            "X": num_node_classes,  # Still predict nodes for skip connection
            "E": num_edge_classes,
            "y": 0,
        }

        # Hidden dimensions for transformer
        self.hidden_dims = {
            "dx": hidden_dim,
            "de": hidden_dim // 4,
            "dy": hidden_dim // 4,
            "n_head": n_heads,
            "dim_ffX": hidden_mlp_dim,
            "dim_ffE": hidden_mlp_dim // 4,
            "dim_ffy": hidden_mlp_dim // 4,
        }

        # Create transformer
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

        # Time distorter
        self.time_distorter = TimeDistorter(
            train_distortion=train_time_distortion,
            sample_distortion=sample_time_distortion,
        )

        # Rate matrix designer for sampling
        self.rate_matrix_designer = RateMatrixDesigner(
            rdb="column",
            eta=eta,
            omega=omega,
            limit_dist=self.limit_dist,
        )

    def forward(
        self,
        noisy_data: Dict[str, Tensor],
        extra_data: PlaceHolder,
        node_mask: Tensor,
    ) -> PlaceHolder:
        """
        Forward pass through transformer.

        Args:
            noisy_data: Dict with X_t, E_t, y_t, t
            extra_data: Extra features (RRWP, etc.)
            node_mask: Valid node mask

        Returns:
            PlaceHolder with predictions
        """
        # Concatenate noisy data with extra features
        X = torch.cat([noisy_data["X_t"], extra_data.X], dim=-1)
        E = torch.cat([noisy_data["E_t"], extra_data.E], dim=-1)

        # Global features: time embedding + HDC vector + extra
        t = noisy_data["t"].unsqueeze(-1)  # (bs, 1)
        y = torch.cat([t, noisy_data["y_t"], extra_data.y], dim=-1)

        return self.model(X, E, y, node_mask)

    def _compute_extra_data(
        self,
        noisy_data: Dict[str, Tensor],
    ) -> PlaceHolder:
        """Compute extra features for the current noisy state."""
        # Create placeholder for extra features computation
        X_t = noisy_data["X_t"]
        E_t = noisy_data["E_t"]
        node_mask = noisy_data["node_mask"]

        # Create dense placeholder
        dense = PlaceHolder(
            X=X_t,
            E=E_t,
            y=torch.zeros(X_t.size(0), 0, device=X_t.device),
        )

        return self.extra_features(dense, node_mask)

    def _p_Et_given_E1(
        self,
        E_1_label: Tensor,
        t: Tensor,
    ) -> Tensor:
        """
        Compute p(E_t | E_1) for edge noise interpolation.

        Args:
            E_1_label: Clean edge labels (bs, n, n)
            t: Timestep (bs, 1)

        Returns:
            Probability distribution over edge classes (bs, n, n, de)
        """
        bs, n, _ = E_1_label.shape
        device = E_1_label.device
        de = self.num_edge_classes

        # One-hot encode E_1
        E_1_onehot = F.one_hot(E_1_label, num_classes=de).float()

        # Get limit distribution
        limit_E = self.limit_dist.E.to(device)

        # Reshape t for broadcasting: (bs, 1, 1, 1)
        t_time = t.view(bs, 1, 1, 1)

        # Linear interpolation: p(E_t | E_1) = t * delta(E_t, E_1) + (1-t) * p_0(E_t)
        prob_E = t_time * E_1_onehot + (1 - t_time) * limit_E

        return prob_E

    def _apply_noise(
        self,
        X: Tensor,
        E: Tensor,
        hdc_vectors: Tensor,
        node_mask: Tensor,
        t: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Apply noise to edges only (nodes stay fixed).

        Args:
            X: Clean node features one-hot (bs, n, dx)
            E: Clean edge features one-hot (bs, n, n, de)
            hdc_vectors: HDC conditioning vectors (bs, hdc_dim)
            node_mask: Valid node mask (bs, n)
            t: Optional timestep (bs, 1)

        Returns:
            Dict with noisy data
        """
        bs = X.size(0)
        device = X.device

        # Sample timestep if not provided
        if t is None:
            t_float = self.time_distorter.train_ft(bs, device)
        else:
            t_float = t

        # Nodes: NO NOISE - keep clean
        X_t = X

        # Edges: Apply noise interpolation
        E_1_label = torch.argmax(E, dim=-1)  # (bs, n, n)
        prob_E_t = self._p_Et_given_E1(E_1_label, t_float)

        # Sample noisy edges
        prob_E_flat = prob_E_t.reshape(-1, self.num_edge_classes)
        E_t_label_flat = torch.multinomial(prob_E_flat, 1).squeeze(-1)
        E_t_label = E_t_label_flat.reshape(bs, X.size(1), X.size(1))

        # Make symmetric
        E_t_label = torch.triu(E_t_label, diagonal=1)
        E_t_label = E_t_label + E_t_label.transpose(1, 2)

        # Convert to one-hot
        E_t = F.one_hot(E_t_label, num_classes=self.num_edge_classes).float()

        # Apply node mask to edges
        edge_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
        E_t = E_t * edge_mask.unsqueeze(-1).float()

        # Global features: HDC vector
        y_t = hdc_vectors

        return {
            "t": t_float,
            "X_t": X_t,
            "E_t": E_t,
            "y_t": y_t,
            "node_mask": node_mask,
        }

    def training_step(
        self,
        batch: Batch,
        batch_idx: int,
    ) -> Optional[Dict[str, Tensor]]:
        """
        Training step - only compute edge loss.

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

        # Get HDC vectors from batch
        # HDC vectors are per-graph, need to extract from batch
        hdc_vectors = self._get_hdc_vectors_from_batch(batch)

        # Apply noise to edges only
        noisy_data = self._apply_noise(X, E, hdc_vectors, node_mask)

        # Compute extra features
        extra_data = self._compute_extra_data(noisy_data)

        # Forward pass
        pred = self.forward(noisy_data, extra_data, node_mask)

        # Compute edge-only loss
        loss = self.train_loss(pred.E, E, node_mask)

        self.log("train/loss", loss, prog_bar=True, batch_size=batch.num_graphs)
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
        noisy_data = self._apply_noise(X, E, hdc_vectors, node_mask)
        extra_data = self._compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        loss = self.train_loss(pred.E, E, node_mask)
        self.log("val/loss", loss, prog_bar=True, batch_size=batch.num_graphs)
        return {"val_loss": loss}

    def _get_hdc_vectors_from_batch(self, batch: Batch) -> Tensor:
        """Extract per-graph HDC vectors from batch."""
        # HDC vectors should be stored as batch.hdc_vector with shape (num_graphs, hdc_dim)
        if hasattr(batch, "hdc_vector"):
            hdc = batch.hdc_vector
            if hdc.dim() == 1:
                # Single graph
                return hdc.unsqueeze(0)
            return hdc
        else:
            raise ValueError("Batch does not have 'hdc_vector' attribute. "
                           "Make sure to preprocess data with preprocess_for_flow_edge_decoder().")

    def _sample_step(
        self,
        t: Tensor,
        s: Tensor,
        X_t: Tensor,
        E_t: Tensor,
        y_t: Tensor,
        node_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Sample z_s given z_t - ONLY update edges.

        Args:
            t: Current time (bs, 1)
            s: Next time (bs, 1)
            X_t: Current nodes (fixed, won't change)
            E_t: Current edges one-hot
            y_t: HDC vectors (fixed)
            node_mask: Valid node mask

        Returns:
            (X_s, E_s, y_s) - X_s = X_t (unchanged), E_s = new edges
        """
        bs, n, de = E_t.shape[:2], E_t.shape[-1]
        dt = (s - t)[0]

        # Prepare noisy data dict
        noisy_data = {
            "t": t,
            "X_t": X_t,
            "E_t": E_t,
            "y_t": y_t,
            "node_mask": node_mask,
        }

        # Compute extra features
        extra_data = self._compute_extra_data(noisy_data)

        # Forward pass
        pred = self.forward(noisy_data, extra_data, node_mask)

        # Get edge predictions
        pred_E = F.softmax(pred.E, dim=-1)

        # Compute rate matrices for edges only
        R_t_X, R_t_E = self.rate_matrix_designer.compute_rate_matrices(
            t, node_mask, X_t, E_t, pred.X, pred_E
        )

        # Compute step probabilities for edges
        # p(E_s | E_t) = E_t + R_t * dt
        E_t_label = torch.argmax(E_t, dim=-1)

        # Get transition probabilities
        step_probs_E = R_t_E * dt.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # Add diagonal (stay probability)
        step_probs_E.scatter_(
            -1,
            E_t_label.unsqueeze(-1),
            0.0,
        )
        stay_prob = 1.0 - step_probs_E.sum(dim=-1, keepdim=True).clamp(min=0)
        step_probs_E.scatter_(
            -1,
            E_t_label.unsqueeze(-1),
            stay_prob,
        )

        # Clamp to valid probabilities
        step_probs_E = step_probs_E.clamp(min=0)
        step_probs_E = step_probs_E / step_probs_E.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # At final step, use predicted marginals directly
        if s[0].item() >= 1.0 - 1e-6:
            step_probs_E = pred_E

        # Sample next edge state
        prob_E_flat = step_probs_E.reshape(-1, de)
        E_s_label_flat = torch.multinomial(prob_E_flat, 1).squeeze(-1)
        E_s_label = E_s_label_flat.reshape(bs[0], n, n)

        # Make symmetric
        E_s_label = torch.triu(E_s_label, diagonal=1)
        E_s_label = E_s_label + E_s_label.transpose(1, 2)

        # Convert to one-hot
        E_s = F.one_hot(E_s_label, num_classes=de).float()

        # Apply mask
        edge_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
        E_s = E_s * edge_mask.unsqueeze(-1).float()

        # Nodes stay fixed
        X_s = X_t

        return X_s, E_s, y_t

    @torch.no_grad()
    def sample(
        self,
        hdc_vectors: Tensor,
        node_features: Tensor,
        node_mask: Tensor,
        eta: Optional[float] = None,
        omega: Optional[float] = None,
        sample_steps: Optional[int] = None,
        time_distortion: Optional[str] = None,
        device: Optional[torch.device] = None,
        show_progress: bool = True,
    ) -> List[Data]:
        """
        Generate edges conditioned on HDC vectors and fixed nodes.

        Args:
            hdc_vectors: Pre-computed HDC vectors (num_samples, hdc_dim)
            node_features: Fixed node types one-hot (num_samples, max_n, num_node_classes)
            node_mask: Valid node mask (num_samples, max_n)
            eta: Stochasticity parameter (None = use default)
            omega: Target guidance strength (None = use default)
            sample_steps: Number of denoising steps (None = use default)
            time_distortion: Time distortion type (None = use default)
            device: Device to run on
            show_progress: Whether to show progress bar

        Returns:
            List of PyG Data objects with generated edges
        """
        self.eval()

        # Use defaults if not specified
        if eta is None:
            eta = self.default_eta
        if omega is None:
            omega = self.default_omega
        if sample_steps is None:
            sample_steps = self.default_sample_steps
        if time_distortion is None:
            time_distortion = self.default_sample_time_distortion

        # Update rate matrix designer
        self.rate_matrix_designer.eta = eta
        self.rate_matrix_designer.omega = omega

        # Move to device
        device = device or self.device
        hdc_vectors = hdc_vectors.to(device)
        X = node_features.to(device).float()
        node_mask = node_mask.to(device)

        num_samples = hdc_vectors.size(0)
        n_max = X.size(1)

        # Sample initial noise for edges only
        e_limit = self.limit_dist.E.to(device)
        e_probs = e_limit.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(
            num_samples, n_max, n_max, -1
        )
        prob_E_flat = e_probs.reshape(-1, self.num_edge_classes)
        E_label_flat = torch.multinomial(prob_E_flat, 1).squeeze(-1)
        E_label = E_label_flat.reshape(num_samples, n_max, n_max)

        # Make symmetric
        E_label = torch.triu(E_label, diagonal=1)
        E_label = E_label + E_label.transpose(1, 2)

        # Convert to one-hot
        E = F.one_hot(E_label, num_classes=self.num_edge_classes).float()

        # Apply mask
        edge_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
        E = E * edge_mask.unsqueeze(-1).float()

        # HDC vectors as y
        y = hdc_vectors

        # Sampling loop - only updates edges
        iterator = range(sample_steps)
        if show_progress:
            iterator = tqdm(iterator, desc="Sampling edges")

        for t_int in iterator:
            t_norm = torch.tensor([t_int / sample_steps], device=device)
            s_norm = torch.tensor([(t_int + 1) / sample_steps], device=device)

            # Apply time distortion
            t_norm = self.time_distorter.sample_ft(t_norm, time_distortion)
            s_norm = self.time_distorter.sample_ft(s_norm, time_distortion)

            # Expand to batch size
            t = t_norm.expand(num_samples, 1)
            s = s_norm.expand(num_samples, 1)

            # Sample step - X stays fixed, only E changes
            X, E, y = self._sample_step(t, s, X, E, y, node_mask)

        # Convert to PyG Data objects
        n_nodes = node_mask.sum(dim=1).int()
        samples = dense_to_pyg(X, E, torch.zeros_like(y[:, :0]), node_mask, n_nodes)

        return samples

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
    ) -> "FlowEdgeDecoder":
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


# =============================================================================
# Preprocessing Functions
# =============================================================================

def map_atom_type_to_7class(atom_idx: int, dataset: str) -> int:
    """
    Map dataset-specific atom type index to 7-class encoding.

    Args:
        atom_idx: Atom type index from dataset
        dataset: Dataset name ("qm9" or "zinc")

    Returns:
        7-class atom index, or -1 if unsupported
    """
    if dataset.lower() == "qm9":
        return QM9_TO_7CLASS.get(atom_idx, -1)
    elif dataset.lower() == "zinc":
        return ZINC_TO_7CLASS.get(atom_idx, -1)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_bond_type_idx(bond) -> int:
    """Convert RDKit bond to 5-class index."""
    if bond is None:
        return 0  # No edge
    return BOND_TYPE_TO_IDX.get(bond.GetBondType(), 0)


def preprocess_for_flow_edge_decoder(
    data: Data,
    hypernet,
    dataset: str,
    device: torch.device = None,
) -> Optional[Data]:
    """
    Preprocess a PyG Data object for FlowEdgeDecoder training.

    Adds/replaces:
    - x: 7-class atom type one-hot encoding
    - edge_attr: 5-class bond type one-hot encoding
    - hdc_vector: Pre-computed HDC embedding

    Args:
        data: Original PyG Data object
        hypernet: HyperNet encoder for computing HDC embeddings
        dataset: Dataset name ("qm9" or "zinc")
        device: Device for HDC computation

    Returns:
        Preprocessed Data object, or None if molecule has unsupported atoms or no edges
    """
    device = device or torch.device("cpu")

    # Skip molecules with no edges (single atoms)
    if data.edge_index.numel() == 0:
        return None

    # 1. Map atom types to 7-class encoding
    atom_types = data.x[:, 0].int()  # First column is atom type
    atom_types_7class = []

    for t in atom_types:
        mapped = map_atom_type_to_7class(int(t), dataset)
        if mapped == -1:
            return None  # Skip molecules with unsupported atoms
        atom_types_7class.append(mapped)

    atom_types_7class = torch.tensor(atom_types_7class, dtype=torch.long)
    x_7class = F.one_hot(atom_types_7class, num_classes=NUM_ATOM_CLASSES).float()

    # 2. Create 5-class edge attributes from SMILES
    mol = Chem.MolFromSmiles(data.smiles)
    if mol is None:
        return None

    n_atoms = len(atom_types)

    # Build adjacency with bond types
    edge_attr_list = []
    edge_index = data.edge_index

    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        bond = mol.GetBondBetweenAtoms(src, dst)
        bond_type = get_bond_type_idx(bond)
        edge_attr_list.append(bond_type)

    edge_attr = torch.tensor(edge_attr_list, dtype=torch.long)
    edge_attr_onehot = F.one_hot(edge_attr, num_classes=NUM_EDGE_CLASSES).float()

    # 3. Compute HDC embedding
    data_for_hdc = data.clone().to(device)
    # Add batch attribute for single graph (required by HyperNet)
    if data_for_hdc.batch is None:
        data_for_hdc.batch = torch.zeros(data_for_hdc.x.size(0), dtype=torch.long, device=device)
    with torch.no_grad():
        hdc_out = hypernet.forward(data_for_hdc)
        # Use graph_embedding (combined edge + graph terms)
        if "graph_embedding" in hdc_out:
            hdc_vector = hdc_out["graph_embedding"].squeeze(0).cpu()
        else:
            # Fallback: concatenate edge_terms and graph_terms
            edge_terms = hdc_out.get("edge_terms", torch.zeros(hypernet.hv_dim))
            graph_terms = hdc_out.get("graph_terms", torch.zeros(hypernet.hv_dim))
            hdc_vector = torch.cat([edge_terms, graph_terms], dim=-1).squeeze(0).cpu()

    # 4. Create new Data object
    new_data = Data(
        x=x_7class,
        edge_index=edge_index.clone(),
        edge_attr=edge_attr_onehot,
        hdc_vector=hdc_vector,
        smiles=data.smiles,
        # Keep original data for reference
        original_x=data.x.clone(),
    )

    # Copy other attributes
    if hasattr(data, "logp"):
        new_data.logp = data.logp.clone()
    if hasattr(data, "qed"):
        new_data.qed = data.qed.clone()

    return new_data


def preprocess_dataset(
    dataset,
    hypernet,
    dataset_name: str,
    device: torch.device = None,
    show_progress: bool = True,
) -> List[Data]:
    """
    Preprocess entire dataset for FlowEdgeDecoder.

    Args:
        dataset: PyG dataset or list of Data objects
        hypernet: HyperNet encoder
        dataset_name: Dataset name ("qm9" or "zinc")
        device: Device for HDC computation
        show_progress: Whether to show progress bar

    Returns:
        List of preprocessed Data objects
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hypernet = hypernet.to(device)
    hypernet.eval()

    processed = []
    iterator = dataset
    if show_progress:
        iterator = tqdm(iterator, desc=f"Preprocessing {dataset_name}")

    for data in iterator:
        new_data = preprocess_for_flow_edge_decoder(
            data, hypernet, dataset_name, device
        )
        if new_data is not None:
            processed.append(new_data)

    return processed


def compute_edge_marginals(data_list: List[Data]) -> Tensor:
    """
    Compute empirical edge marginals from preprocessed data.

    Args:
        data_list: List of preprocessed Data objects

    Returns:
        Tensor of shape (5,) with probabilities for each edge class
    """
    edge_counts = torch.zeros(NUM_EDGE_CLASSES)

    for data in data_list:
        n = data.x.size(0)

        # Count edge types from edge_attr
        if data.edge_attr is not None:
            edge_types = data.edge_attr.argmax(dim=-1)
            for et in edge_types:
                edge_counts[et] += 1

        # Count no-edges
        # Total possible edges (undirected): n * (n-1) / 2
        # Actual edges: edge_index.size(1) / 2 (since bidirectional)
        all_pairs = n * (n - 1)  # Counting both directions
        actual_edges = data.edge_index.size(1)
        no_edge_count = all_pairs - actual_edges
        edge_counts[0] += no_edge_count

    # Normalize
    marginals = edge_counts / edge_counts.sum()
    return marginals


def compute_node_counts(data_list: List[Data]) -> Tensor:
    """
    Compute node count distribution.

    Args:
        data_list: List of Data objects

    Returns:
        Tensor with counts for each graph size
    """
    max_n = max(d.x.size(0) for d in data_list)
    counts = torch.zeros(max_n + 1)

    for data in data_list:
        n = data.x.size(0)
        counts[n] += 1

    return counts
