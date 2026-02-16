"""
FlowEdgeDecoder - Edge-only DeFoG decoder conditioned on HDC vectors.

This module implements a discrete flow matching model that generates molecular
edges conditioned on:
1. Pre-computed HDC vectors (512-dim default)
2. Fixed node features (24-dim one-hot: atom type, degree, charge, Hs, ring)

The model inherits from DeFoG's DeFoGModel and overrides the training and
sampling methods to:
- Keep nodes fixed (no noise applied)
- Only denoise edges through the flow matching process
- Use HDC vectors as global conditioning

Node features (ZINC format):
- atom_type: 9 classes (Br, C, Cl, F, I, N, O, P, S)
- degree: 6 classes (0-5)
- formal_charge: 3 classes (0=neutral, 1=+1, 2=-1)
- total_Hs: 4 classes (0-3)
- is_in_ring: 2 classes (0, 1)
Total: 24-dimensional one-hot concatenation
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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

import torchhd

from graph_hdc.utils.helpers import scatter_hd


# =============================================================================
# Time Embedding
# =============================================================================


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for time, similar to transformer positional encodings."""

    def __init__(self, embed_dim: int, max_period: float = 10000.0):
        """
        Args:
            embed_dim: Output embedding dimension
            max_period: Maximum period for the sinusoidal encoding
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.max_period = max_period

    def forward(self, t: Tensor) -> Tensor:
        """
        Embed scalar timesteps into sinusoidal encoding.

        Args:
            t: Timesteps of shape (batch_size, 1) with values in [0, 1]

        Returns:
            Embeddings of shape (batch_size, embed_dim)
        """
        half_dim = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, device=t.device, dtype=t.dtype)
            / half_dim
        )
        # t is (bs, 1), freqs is (half_dim,)
        # Scale t to a larger range for better frequency coverage
        args = t * 1000.0 * freqs.unsqueeze(0)  # (bs, half_dim)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (bs, embed_dim)
        return embedding


# =============================================================================
# Constants
# =============================================================================

# ZINC atom types (9 classes)
ZINC_ATOM_TYPES = ["Br", "C", "Cl", "F", "I", "N", "O", "P", "S"]
ZINC_ATOM_TO_IDX = {atom: idx for idx, atom in enumerate(ZINC_ATOM_TYPES)}
ZINC_IDX_TO_ATOM = {idx: atom for atom, idx in ZINC_ATOM_TO_IDX.items()}

# Node feature dimensions (ZINC)
NUM_ATOM_CLASSES = 9
NUM_DEGREE_CLASSES = 6
NUM_CHARGE_CLASSES = 3
NUM_HS_CLASSES = 4
NUM_RING_CLASSES = 2

# Total one-hot dimension for all node features
NODE_FEATURE_DIM = (
    NUM_ATOM_CLASSES + NUM_DEGREE_CLASSES + NUM_CHARGE_CLASSES +
    NUM_HS_CLASSES + NUM_RING_CLASSES
)  # 24

# Feature bin sizes (for indexing)
NODE_FEATURE_BINS = [
    NUM_ATOM_CLASSES,
    NUM_DEGREE_CLASSES,
    NUM_CHARGE_CLASSES,
    NUM_HS_CLASSES,
    NUM_RING_CLASSES,
]

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


# =============================================================================
# Node Feature Conversion Helpers
# =============================================================================


def get_node_feature_bins(rw_config=None) -> List[int]:
    """
    Get the full list of feature bin sizes, optionally including RW bins.

    Args:
        rw_config: Optional RWConfig. When provided and enabled, appends
                   ``[num_bins] * len(k_values)`` to the base ZINC bins.

    Returns:
        List of class counts per feature position.
        Base: [9, 6, 3, 4, 2] (24-dim).
        With RW(k=(3,6), bins=10): [9, 6, 3, 4, 2, 10, 10] (44-dim).
    """
    bins = list(NODE_FEATURE_BINS)
    if rw_config is not None and getattr(rw_config, "enabled", False):
        bins.extend([rw_config.num_bins] * len(rw_config.k_values))
    return bins


def node_tuple_to_onehot(
    node_tuple: Tuple[int, ...],
    device: Optional[torch.device] = None,
    feature_bins: Optional[List[int]] = None,
) -> Tensor:
    """
    Convert a single node tuple to concatenated one-hot encoding.

    Encodes every position in the tuple according to ``feature_bins``.
    When ``feature_bins`` is ``None``, falls back to the base 5-feature
    ``NODE_FEATURE_BINS`` for backward compatibility.

    Args:
        node_tuple: Tuple of integer feature indices, e.g.
                    (atom_idx, degree, charge, num_hs, is_ring) for base, or
                    (atom_idx, degree, charge, num_hs, is_ring, rw_k3, rw_k6)
                    when RW features are present.
        device: Target device for the tensor.
        feature_bins: List of class counts per position. Length must match the
                      tuple length. Use ``get_node_feature_bins(rw_config)``
                      to obtain this from a HyperNet's RW configuration.

    Returns:
        One-hot tensor of shape (sum(feature_bins),)
    """
    if feature_bins is None:
        feature_bins = NODE_FEATURE_BINS

    parts = []
    for i, num_classes in enumerate(feature_bins):
        parts.append(F.one_hot(torch.tensor(node_tuple[i]), num_classes=num_classes))

    onehot = torch.cat(parts, dim=-1).float()

    if device is not None:
        onehot = onehot.to(device)
    return onehot


def node_tuples_to_onehot(
    node_tuples: List[Tuple[int, ...]],
    device: Optional[torch.device] = None,
    feature_bins: Optional[List[int]] = None,
) -> Tensor:
    """
    Convert list of node tuples to concatenated one-hot tensor.

    Args:
        node_tuples: List of integer-index tuples (one per node).
        device: Target device for the tensor.
        feature_bins: List of class counts per position (see
                      ``node_tuple_to_onehot``). Defaults to
                      ``NODE_FEATURE_BINS``.

    Returns:
        One-hot tensor of shape (n, sum(feature_bins))
    """
    if feature_bins is None:
        feature_bins = NODE_FEATURE_BINS

    total_dim = sum(feature_bins)
    if not node_tuples:
        t = torch.zeros(0, total_dim)
        return t.to(device) if device is not None else t

    onehots = [node_tuple_to_onehot(t, device, feature_bins) for t in node_tuples]
    return torch.stack(onehots, dim=0)


def raw_features_to_onehot(
    raw_features: Tensor,
    device: Optional[torch.device] = None,
    feature_bins: Optional[List[int]] = None,
) -> Tensor:
    """
    Convert raw node features tensor to concatenated one-hot encoding.

    Args:
        raw_features: Tensor of shape (n, num_features) with integer indices.
                      Columns correspond to ``feature_bins`` positions.
        device: Target device for the tensor.
        feature_bins: List of class counts per column. Defaults to
                      ``NODE_FEATURE_BINS``.

    Returns:
        One-hot tensor of shape (n, sum(feature_bins))
    """
    if feature_bins is None:
        feature_bins = NODE_FEATURE_BINS

    total_dim = sum(feature_bins)
    n_atoms = raw_features.size(0)
    x_onehot = torch.zeros(n_atoms, total_dim, dtype=torch.float32)

    offset = 0
    for feat_idx, num_classes in enumerate(feature_bins):
        feat_vals = raw_features[:, feat_idx].long()
        onehot = F.one_hot(feat_vals, num_classes=num_classes).float()
        x_onehot[:, offset:offset + num_classes] = onehot
        offset += num_classes

    if device is not None:
        x_onehot = x_onehot.to(device)
    return x_onehot


def onehot_to_raw_features(
    x_onehot: Tensor,
    feature_bins: Optional[List[int]] = None,
) -> Tensor:
    """
    Reverse concatenated one-hot encoding back to raw integer features.

    Args:
        x_onehot: One-hot tensor of shape (n, sum(feature_bins))
        feature_bins: List of class counts per position. Defaults to
                      ``NODE_FEATURE_BINS`` ([9, 6, 3, 4, 2] = 24-dim).

    Returns:
        Tensor of shape (n, len(feature_bins)) with integer indices.
    """
    if feature_bins is None:
        feature_bins = NODE_FEATURE_BINS

    parts = []
    offset = 0
    for num_classes in feature_bins:
        block = x_onehot[:, offset:offset + num_classes]
        parts.append(block.argmax(dim=-1))
        offset += num_classes

    return torch.stack(parts, dim=-1).float()


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class FlowEdgeDecoderConfig:
    """Configuration for FlowEdgeDecoder model."""

    # Data dimensions
    num_node_classes: int = NODE_FEATURE_DIM
    num_edge_classes: int = NUM_EDGE_CLASSES
    hdc_dim: int = 512
    condition_dim: int = 128  # Reduced dimension after MLP
    time_embed_dim: int = 64  # Dimension for time embedding

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

    # Cross-attention conditioning
    use_cross_attn: bool = False
    cross_attn_tokens: int = 8
    cross_attn_heads: int = 4

    # Per-node HDC codebook embedding
    node_hdc_embed_dim: int = 0  # 0=disabled, >0=project codebook HVs to this dim


# Default config for ZINC dataset (24-dim node features)
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
# HDC Cross-Attention Conditioner
# =============================================================================


class HDCCrossAttentionConditioner(nn.Module):
    """
    Cross-attention conditioning module for HDC vectors.

    Decomposes a raw HDC vector into K learnable tokens, lets node features
    cross-attend to these tokens, then combines node-pair representations
    to produce edge-specific conditioning signals.

    This replaces the broadcast FiLM-only approach with edge-specific
    conditioning: different edges receive different conditioning based on
    which node pair they connect and what the HDC vector encodes.

    Args:
        hdc_dim: Raw HDC vector dimension (e.g. 512)
        node_input_dim: Node feature input dimension (e.g. 34 = 24 node + 10 RRWP)
        edge_cond_dim: Output dimension per edge pair
        num_tokens: Number of HDC tokens to decompose into
        n_heads: Number of attention heads for cross-attention
    """

    def __init__(
        self,
        hdc_dim: int,
        node_input_dim: int,
        edge_cond_dim: int,
        num_tokens: int = 8,
        n_heads: int = 4,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        # Use a projected dimension that is divisible by n_heads
        self.attn_dim = num_tokens * n_heads

        # Project node features to attention dimension
        self.q_proj = nn.Linear(node_input_dim, self.attn_dim)

        # HDC → K tokens of dimension attn_dim
        self.hdc_to_tokens = nn.Sequential(
            nn.Linear(hdc_dim, num_tokens * self.attn_dim),
            nn.ReLU(),
            nn.Linear(num_tokens * self.attn_dim, num_tokens * self.attn_dim),
        )

        # Node cross-attention: Q=projected nodes, K/V=HDC tokens
        self.node_cross_attn = nn.MultiheadAttention(
            embed_dim=self.attn_dim, num_heads=n_heads, batch_first=True
        )
        self.node_norm = nn.LayerNorm(self.attn_dim)

        # Edge projection: concat(node_i, node_j) → edge_cond_dim
        self.edge_proj = nn.Sequential(
            nn.Linear(2 * self.attn_dim, edge_cond_dim),
            nn.ReLU(),
            nn.Linear(edge_cond_dim, edge_cond_dim),
        )

    def forward(
        self,
        X: Tensor,
        hdc_raw: Tensor,
        node_mask: Tensor,
    ) -> Tensor:
        """
        Compute edge-specific conditioning from HDC vector and node features.

        Args:
            X: Node features (bs, n, node_input_dim)
            hdc_raw: Raw HDC vector (bs, hdc_dim)
            node_mask: Valid node mask (bs, n)

        Returns:
            Edge conditioning tensor (bs, n, n, edge_cond_dim)
        """
        bs, n, _ = X.shape

        # 1. Generate HDC tokens
        tokens = self.hdc_to_tokens(hdc_raw)
        tokens = tokens.view(bs, self.num_tokens, self.attn_dim)

        # 2. Project node features and cross-attend to HDC tokens
        Q = self.q_proj(X)  # (bs, n, attn_dim)
        node_cond, _ = self.node_cross_attn(Q, tokens, tokens)
        node_cond = self.node_norm(node_cond)  # (bs, n, attn_dim)
        node_cond = node_cond * node_mask.unsqueeze(-1).float()

        # 3. Edge conditioning from node pairs
        node_i = node_cond.unsqueeze(2).expand(-1, -1, n, -1)  # (bs, n, n, attn_dim)
        node_j = node_cond.unsqueeze(1).expand(-1, n, -1, -1)  # (bs, n, n, attn_dim)
        edge_cond = self.edge_proj(torch.cat([node_i, node_j], dim=-1))

        # Mask invalid edges
        edge_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
        edge_cond = edge_cond * edge_mask.unsqueeze(-1).float()

        return edge_cond


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
        num_node_classes: int = NODE_FEATURE_DIM,
        num_edge_classes: int = NUM_EDGE_CLASSES,
        hdc_dim: int = 512,
        condition_dim: int = 128,
        time_embed_dim: int = 64,
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
        use_cross_attn: bool = False,
        cross_attn_tokens: int = 8,
        cross_attn_heads: int = 4,
        node_hdc_embed_dim: int = 0,
        nodes_codebook: Optional[Tensor] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["nodes_codebook"])

        # Store dimensions
        self.num_node_classes = num_node_classes
        self.num_edge_classes = num_edge_classes
        self.hdc_dim = hdc_dim
        self.condition_dim = condition_dim
        self.time_embed_dim = time_embed_dim
        self.max_nodes = max_nodes

        # Training config
        self.lr = lr
        self.weight_decay = weight_decay

        # Sampling config
        self.default_sample_steps = sample_steps
        self.default_eta = eta
        self.default_omega = omega
        self.default_sample_time_distortion = sample_time_distortion

        # Per-node HDC codebook embedding
        self.node_hdc_embed_dim = node_hdc_embed_dim
        if node_hdc_embed_dim > 0:
            assert nodes_codebook is not None, (
                "nodes_codebook must be provided when node_hdc_embed_dim > 0"
            )
            # Convert to plain Tensor — codebook may be a VSATensor subclass
            # from torchhd which doesn't support deepcopy (breaks PL checkpointing).
            cb = nodes_codebook.detach().clone()
            if type(cb) is not torch.Tensor:
                cb = cb.as_subclass(torch.Tensor)
            self.register_buffer("_nodes_codebook", cb)
            codebook_hv_dim = nodes_codebook.shape[1]
            self.node_hdc_proj = nn.Linear(codebook_hv_dim, node_hdc_embed_dim)
            # Strides for flat index computation from NODE_FEATURE_BINS
            strides = []
            s = 1
            for b in reversed(NODE_FEATURE_BINS):
                strides.append(s)
                s *= b
            strides.reverse()
            self.register_buffer(
                "_codebook_strides",
                torch.tensor(strides, dtype=torch.long),
            )

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

        # Cross-attention conditioning
        self.use_cross_attn = use_cross_attn
        if self.use_cross_attn:
            node_input_dim = num_node_classes + extra_dims["X"] + node_hdc_embed_dim
            edge_cond_dim = hidden_dim // 4  # match de
            self.hdc_conditioner = HDCCrossAttentionConditioner(
                hdc_dim=hdc_dim,
                node_input_dim=node_input_dim,
                edge_cond_dim=edge_cond_dim,
                num_tokens=cross_attn_tokens,
                n_heads=cross_attn_heads,
            )
            cross_attn_edge_dim = edge_cond_dim
        else:
            cross_attn_edge_dim = 0

        # MLP to reduce HDC conditioning dimension
        # Input: concatenated [order_0 | order_N] with dim = hdc_dim (which is 2x base_hdc_dim)
        # Output: condition_dim
        self.condition_mlp = nn.Sequential(
            nn.Linear(hdc_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, condition_dim),
        )

        # Time embedding: sinusoidal encoding + MLP
        # This gives the time signal more expressive power to match the HDC conditioning
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Input dimensions: nodes + edges + global (time_embed + reduced HDC + extra)
        self.input_dims = {
            "X": num_node_classes + extra_dims["X"] + node_hdc_embed_dim,
            "E": num_edge_classes + extra_dims["E"] + cross_attn_edge_dim,
            "y": time_embed_dim + extra_dims["y"] + condition_dim,  # time_embed + extra + reduced HDC
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

        # Container for batch metrics (used by TrainingMetricsCallback)
        self.batch_metrics: Dict[str, Tensor] = {}

    def _create_limit_distribution(
        self,
        noise_type: str,
        edge_marginals: Optional[Tensor] = None,
    ) -> LimitDistribution:
        """
        Create a LimitDistribution with the specified noise type.

        This is used to override the noise type at inference time while keeping
        consistent behavior across initialization and rate matrix computation.

        Args:
            noise_type: "uniform" or "marginal"
            edge_marginals: Edge marginals for "marginal" type (uses stored if None)

        Returns:
            New LimitDistribution instance
        """
        if noise_type == "uniform":
            # Uniform: don't need marginals
            return LimitDistribution(
                noise_type="uniform",
                num_node_classes=self.num_node_classes,
                num_edge_classes=self.num_edge_classes,
                node_marginals=None,
                edge_marginals=None,
            )
        elif noise_type == "marginal":
            # Use stored or provided marginals
            marginals = edge_marginals if edge_marginals is not None else self.limit_dist.E
            node_marginals = torch.ones(self.num_node_classes) / self.num_node_classes
            return LimitDistribution(
                noise_type="marginal",
                num_node_classes=self.num_node_classes,
                num_edge_classes=self.num_edge_classes,
                node_marginals=node_marginals,
                edge_marginals=marginals,
            )
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}. Use 'uniform' or 'marginal'.")

    def _lookup_node_hdc_embeddings(self, X_t: Tensor) -> Tensor:
        """Look up per-node HDC codebook vectors and project them.

        Args:
            X_t: One-hot node features (bs, n, num_node_classes). Only the
                 first NODE_FEATURE_DIM dimensions are used for the lookup.

        Returns:
            Projected embeddings of shape (bs, n, node_hdc_embed_dim).
        """
        x_base = X_t[..., :NODE_FEATURE_DIM]  # (bs, n, 24)
        # Argmax per feature group to recover raw integer indices
        offset = 0
        raw_parts = []
        for num_classes in NODE_FEATURE_BINS:
            raw_parts.append(x_base[..., offset:offset + num_classes].argmax(dim=-1))
            offset += num_classes
        raw = torch.stack(raw_parts, dim=-1)  # (bs, n, 5)
        # Flat index via precomputed strides
        flat_idx = (raw * self._codebook_strides).sum(dim=-1).long()  # (bs, n)
        flat_idx = flat_idx.clamp(0, self._nodes_codebook.shape[0] - 1)
        # Lookup and project (cast to float32 since codebook may be float64)
        node_hdc = self._nodes_codebook[flat_idx].float()  # (bs, n, hv_dim)
        return self.node_hdc_proj(node_hdc)  # (bs, n, node_hdc_embed_dim)

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

        # Per-node HDC codebook embeddings
        if self.node_hdc_embed_dim > 0:
            node_hdc_embed = self._lookup_node_hdc_embeddings(noisy_data["X_t"])
            X = torch.cat([X, node_hdc_embed], dim=-1)

        # Cross-attention edge conditioning
        if self.use_cross_attn and "hdc_raw" in noisy_data:
            edge_cond = self.hdc_conditioner(X, noisy_data["hdc_raw"], node_mask)
            E = torch.cat([E, edge_cond], dim=-1)

        # Global features: time embedding + HDC vector + extra
        t = noisy_data["t"]  # (bs, 1) from time distorter
        t_embed = self.time_mlp(t)  # (bs, time_embed_dim)
        y = torch.cat([t_embed, noisy_data["y_t"], extra_data.y], dim=-1)

        return self.model(X, E, y, node_mask)

    def _compute_extra_data(
        self,
        noisy_data: Dict[str, Tensor],
    ) -> PlaceHolder:
        """Compute extra features for the current noisy state."""
        # ExtraFeatures expects a dict with X_t, E_t, y_t, node_mask
        return self.extra_features(noisy_data)

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

        # Global features: HDC vector reduced through MLP
        y_t = self.condition_mlp(hdc_vectors)

        return {
            "t": t_float,
            "X_t": X_t,
            "E_t": E_t,
            "y_t": y_t,
            "node_mask": node_mask,
            "hdc_raw": hdc_vectors,
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

        # Log detached loss to allow deepcopy in checkpoint callback
        self.log("train/loss", loss.detach(), prog_bar=True, batch_size=batch.num_graphs)

        # Store batch metrics for TrainingMetricsCallback (moved to CPU to avoid GPU accumulation)
        self.batch_metrics = {
            "t": noisy_data["t"].detach().cpu(),
            "pred_classes": pred.E.argmax(dim=-1).detach().cpu(),
            "true_classes": E.argmax(dim=-1).detach().cpu(),
            "node_mask": node_mask.detach().cpu(),
            "loss": loss.detach().cpu(),
        }

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

        # Log and return detached loss (on CPU to avoid accumulation across val batches)
        self.log("val/loss", loss.detach(), prog_bar=True, batch_size=batch.num_graphs)
        return {"val_loss": loss.detach().cpu()}

    def _get_hdc_vectors_from_batch(self, batch: Batch) -> Tensor:
        """Extract per-graph HDC vectors from batch."""
        # HDC vectors should be stored as batch.hdc_vector with shape (num_graphs, hdc_dim)
        # Each graph's hdc_vector is stored as (1, hdc_dim) so batching gives (batch_size, hdc_dim)
        if hasattr(batch, "hdc_vector"):
            hdc = batch.hdc_vector
            # Should already be (batch_size, hdc_dim) from proper PyG batching
            if hdc.dim() == 2:
                return hdc
            else:
                # Fallback for single graph case
                return hdc.unsqueeze(0)
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
        deterministic: bool = False,
        hdc_raw: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Sample z_s given z_t - ONLY update edges.

        Args:
            t: Current time (bs, 1)
            s: Next time (bs, 1)
            X_t: Current nodes (fixed, won't change)
            E_t: Current edges one-hot
            y_t: HDC vectors (fixed)
            node_mask: Valid node mask
            deterministic: If True, use argmax instead of sampling for
                          fully deterministic trajectories given the same initial noise
            hdc_raw: Raw HDC vectors for cross-attention conditioning (bs, hdc_dim).
                    Required when use_cross_attn=True, ignored otherwise.

        Returns:
            (X_s, E_s, y_s, pred_E) - X_s = X_t (unchanged), E_s = new edges,
            pred_E = model's predicted clean edge distribution
        """
        bs, n, _, de = E_t.shape  # E_t is (bs, n, n, de)
        dt = (s - t)[0]

        # Prepare noisy data dict
        noisy_data = {
            "t": t,
            "X_t": X_t,
            "E_t": E_t,
            "y_t": y_t,
            "node_mask": node_mask,
        }
        if hdc_raw is not None:
            noisy_data["hdc_raw"] = hdc_raw

        # Compute extra features
        extra_data = self._compute_extra_data(noisy_data)

        # Forward pass
        pred = self.forward(noisy_data, extra_data, node_mask)

        # Get node and edge predictions as probability distributions
        pred_X = F.softmax(pred.X, dim=-1)
        pred_E = F.softmax(pred.E, dim=-1)

        # Compute rate matrices for edges only
        R_t_X, R_t_E = self.rate_matrix_designer.compute_rate_matrices(
            t, node_mask, X_t, E_t, pred_X, pred_E
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

        # Sample next edge state (or argmax if deterministic)
        prob_E_flat = step_probs_E.reshape(-1, de)
        if deterministic:
            E_s_label_flat = torch.argmax(prob_E_flat, dim=-1)
        else:
            E_s_label_flat = torch.multinomial(prob_E_flat, 1).squeeze(-1)
        E_s_label = E_s_label_flat.reshape(bs, n, n)

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

        return X_s, E_s, y_t, pred_E

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
        noise_type_override: Optional[str] = None,
        device: Optional[torch.device] = None,
        show_progress: bool = True,
        step_callback: Optional[Callable[[int, float, Tensor, Tensor, Tensor, Optional[Tensor]], None]] = None,
        deterministic: bool = False,
        initial_edges: Optional[Tensor] = None,
        start_time: float = 0.0,
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
            noise_type_override: Override noise type for initialization ("uniform" or "marginal").
                                 If None, uses the model's trained noise type.
                                 Use "uniform" to test uniform initialization regardless of training.
            device: Device to run on (defaults to CPU for stability)
            show_progress: Whether to show progress bar
            step_callback: Optional callback invoked at each step with
                          (step, t, X, E, node_mask, pred_E).
                          pred_E is the model's predicted clean edge distribution (None at t=0).
                          Use for capturing intermediate states (e.g., GIF animation).
            deterministic: If True, use argmax instead of sampling at each denoising step
                          for fully deterministic trajectories given the same initial noise.
                          Initial noise is still sampled; use torch seeds for reproducibility.
            initial_edges: Optional pre-computed one-hot edge tensor
                          (num_samples, n_max, n_max, de). When provided, skips noise
                          sampling and uses these edges as the starting state. Used for
                          refinement passes where the output of a previous run serves
                          as initialization.
            start_time: Normalized start time in [0.0, 1.0). When > 0.0, the denoising
                       loop skips all steps before this time. Useful for warm-starting
                       from a partially denoised state. Default 0.0 (full schedule).

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

        # Handle noise type override for initialization
        if noise_type_override is not None:
            # Create temporary limit distribution with override
            sampling_limit_dist = self._create_limit_distribution(noise_type_override)
            # Temporarily swap rate_matrix_designer's limit_dist for consistent behavior
            original_limit_dist = self.rate_matrix_designer.limit_dist
            self.rate_matrix_designer.limit_dist = sampling_limit_dist
        else:
            sampling_limit_dist = self.limit_dist
            original_limit_dist = None  # No swap needed

        # Default to CPU for sampling (more stable, easier to debug)
        device = device or torch.device("cpu")
        self.to(device)
        hdc_vectors = hdc_vectors.to(device).float()  # Ensure float32 for model compatibility
        X = node_features.to(device).float()
        node_mask = node_mask.to(device)

        num_samples = hdc_vectors.size(0)
        n_max = X.size(1)

        # Initialize edges: use provided initial_edges or sample noise
        edge_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
        if initial_edges is not None:
            E = initial_edges.to(device).float()
            E = E * edge_mask.unsqueeze(-1).float()
        else:
            # Sample initial noise for edges only (using possibly overridden limit dist)
            e_limit = sampling_limit_dist.E.to(device)
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
            E = E * edge_mask.unsqueeze(-1).float()

        # Keep raw HDC vectors for cross-attention conditioning
        hdc_raw = hdc_vectors if self.use_cross_attn else None

        # HDC vectors reduced through MLP as y
        y = self.condition_mlp(hdc_vectors)

        # Sampling loop - only updates edges
        iterator = range(sample_steps)
        if show_progress:
            iterator = tqdm(iterator, desc="Sampling edges")

        # Callback for initial state (noisy edges, no prediction yet)
        if step_callback is not None:
            step_callback(0, start_time, X.clone(), E.clone(), node_mask, None)

        for t_int in iterator:
            # Skip steps before start_time (for warm-starting / refinement)
            if t_int / sample_steps < start_time:
                continue

            t_norm = torch.tensor([t_int / sample_steps], device=device)
            s_norm = torch.tensor([(t_int + 1) / sample_steps], device=device)

            # Apply time distortion
            t_norm = self.time_distorter.sample_ft(t_norm, time_distortion)
            s_norm = self.time_distorter.sample_ft(s_norm, time_distortion)

            # Expand to batch size
            t = t_norm.expand(num_samples, 1)
            s = s_norm.expand(num_samples, 1)

            # Sample step - X stays fixed, only E changes
            X, E, y, pred_E = self._sample_step(t, s, X, E, y, node_mask, deterministic, hdc_raw=hdc_raw)

            # Callback after each step (includes model's prediction)
            if step_callback is not None:
                step_callback(t_int + 1, s_norm.item(), X.clone(), E.clone(), node_mask, pred_E.clone())

        # Convert to PyG Data objects
        n_nodes = node_mask.sum(dim=1).int()
        samples = dense_to_pyg(X, E, torch.zeros_like(y[:, :0]), node_mask, n_nodes)

        # Restore original limit_dist if it was overridden
        if original_limit_dist is not None:
            self.rate_matrix_designer.limit_dist = original_limit_dist

        return samples

    def sample_best_of_n(
        self,
        hdc_vectors: Tensor,
        node_features: Tensor,
        node_mask: Tensor,
        num_repetitions: int,
        score_fn: Callable[[Data], float],
        **sample_kwargs,
    ) -> Tuple[Data, float]:
        """
        Generate edges N times in parallel and keep the best result.

        Expands the single-sample inputs to ``num_repetitions`` copies,
        runs a single batched :meth:`sample` call, scores each result with
        ``score_fn``, and returns the sample with the lowest score.

        Args:
            hdc_vectors: HDC conditioning vector for **one** molecule (1, hdc_dim).
            node_features: One-hot node features for **one** molecule (1, n, dx).
            node_mask: Valid-node mask for **one** molecule (1, n).
            num_repetitions: Number of parallel decodings (best-of-N).
            score_fn: Scoring callable applied to each generated ``Data`` object.
                Lower is better. Return ``float('inf')`` to discard a sample.
            **sample_kwargs: Forwarded to :meth:`sample` (eta, omega,
                sample_steps, time_distortion, device, initial_edges, …).
                ``initial_edges``, if present, is automatically expanded to
                match ``num_repetitions``.

        Returns:
            Tuple of (best_sample, best_score).
        """
        # Expand single molecule to num_repetitions copies
        batch_hdc = hdc_vectors.expand(num_repetitions, -1)
        batch_nf = node_features.expand(num_repetitions, -1, -1)
        batch_mask = node_mask.expand(num_repetitions, -1)

        # Expand initial_edges if provided (1, n, n, de) -> (N, n, n, de)
        if "initial_edges" in sample_kwargs and sample_kwargs["initial_edges"] is not None:
            sample_kwargs["initial_edges"] = sample_kwargs["initial_edges"].expand(
                num_repetitions, -1, -1, -1
            )

        all_samples = self.sample(
            hdc_vectors=batch_hdc,
            node_features=batch_nf,
            node_mask=batch_mask,
            **sample_kwargs,
        )

        # Score each repetition and pick the best
        scores = [score_fn(s) for s in all_samples]
        best_idx = int(min(range(num_repetitions), key=lambda i: scores[i]))
        best_sample = all_samples[best_idx]
        best_score = scores[best_idx]
        # Free non-selected GPU samples
        del all_samples, scores
        return best_sample, best_score

    # =========================================================================
    # HDC-Guided Sampling Methods
    # =========================================================================

    def _sample_k_candidates(
        self,
        pred_E: Tensor,
        num_candidates: int,
        node_mask: Tensor,
    ) -> Tensor:
        """
        Sample K candidate edge matrices from predicted distribution.

        Args:
            pred_E: Predicted edge probabilities (bs, n, n, de)
            num_candidates: Number of candidates K to sample
            node_mask: Valid node mask (bs, n)

        Returns:
            Tensor of shape (K, bs, n, n) with edge class indices
        """
        bs, n, _, de = pred_E.shape
        device = pred_E.device

        candidates = []
        for _ in range(num_candidates):
            # Sample from the probability distribution
            prob_flat = pred_E.reshape(-1, de)
            E_label_flat = torch.multinomial(prob_flat, 1).squeeze(-1)
            E_label = E_label_flat.reshape(bs, n, n)

            # Make symmetric (upper triangular + transpose)
            E_label = torch.triu(E_label, diagonal=1)
            E_label = E_label + E_label.transpose(1, 2)

            candidates.append(E_label)

        # Stack to (K, bs, n, n)
        return torch.stack(candidates, dim=0)

    def _encode_candidates_to_order_n(
        self,
        candidates: Tensor,
        original_node_features: Tensor,
        node_mask: Tensor,
        hypernet,
    ) -> Tensor:
        """
        Encode each candidate graph to HDC order_N embedding.

        Args:
            candidates: Edge candidates (K, bs, n, n) class indices
            original_node_features: Original 4-dim features for HyperNet (bs, n, 4)
            node_mask: Valid node mask (bs, n)
            hypernet: HyperNet encoder

        Returns:
            Tensor of shape (K, bs, hdc_dim) with order_N embeddings
        """
        K, bs, n, _ = candidates.shape
        device = candidates.device

        # Flatten K and bs dimensions for batched processing
        # Shape: (K*bs, n, n)
        candidates_flat = candidates.reshape(K * bs, n, n)

        # Expand original_node_features to match: (K*bs, n, 4)
        original_x_expanded = original_node_features.unsqueeze(0).expand(K, -1, -1, -1)
        original_x_flat = original_x_expanded.reshape(K * bs, n, -1)

        # Expand node_mask: (K*bs, n)
        node_mask_expanded = node_mask.unsqueeze(0).expand(K, -1, -1)
        node_mask_flat = node_mask_expanded.reshape(K * bs, n)

        # Build Data objects for each candidate
        data_list = []
        for i in range(K * bs):
            # Get valid nodes for this sample
            valid_nodes = node_mask_flat[i]
            num_valid = valid_nodes.sum().item()

            if num_valid == 0:
                # Empty graph - create dummy
                data = Data(
                    x=torch.zeros(1, original_x_flat.shape[-1], device=device),
                    edge_index=torch.zeros(2, 0, dtype=torch.long, device=device),
                )
                data_list.append(data)
                continue

            # Get node features for valid nodes
            x = original_x_flat[i, :num_valid]

            # Get edge labels for valid nodes only
            E_labels = candidates_flat[i, :num_valid, :num_valid]

            # Build edge_index from non-zero edges (edge label > 0 means bond exists)
            edge_mask = E_labels > 0
            src, dst = torch.where(edge_mask)

            if src.numel() == 0:
                # No edges
                edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
            else:
                edge_index = torch.stack([src, dst], dim=0)

            data = Data(x=x, edge_index=edge_index)
            data_list.append(data)

        # Batch all data objects
        batch = Batch.from_data_list(data_list)
        batch = batch.to(device)

        # Encode with HyperNet
        with torch.no_grad():
            # First encode node properties
            batch = hypernet.encode_properties(batch)
            # Ensure node_hv is on the correct device (codebooks might be on CPU)
            if batch.node_hv.device != device:
                batch.node_hv = batch.node_hv.to(device)

            # Forward pass to get graph embeddings (order_N)
            output = hypernet.forward(batch)
            order_n = output["graph_embedding"]  # (K*bs, hdc_dim)

        # Reshape back to (K, bs, hdc_dim)
        hdc_dim = order_n.shape[-1]
        order_n = order_n.reshape(K, bs, hdc_dim)

        return order_n

    def _select_best_candidate(
        self,
        candidate_hdcs: Tensor,
        target_order_n: Tensor,
        candidates: Tensor,
        hypernet=None,
    ) -> Tensor:
        """
        Select the candidate whose order-N embedding is closest to the target.

        Distances are computed via ``hypernet.calculate_order_n_distance()``,
        which correctly handles both single ``HyperNet`` (one cosine
        distance) and ``MultiHyperNet`` (per-sub-HyperNet averaged cosine
        distance).  When *hypernet* is ``None``, falls back to a plain
        cosine distance over the full embedding.

        Args:
            candidate_hdcs: Candidate order_N embeddings (K, bs, order_n_dim).
            target_order_n: Target order_N (bs, order_n_dim).
            candidates: Edge candidates (K, bs, n, n) class indices.
            hypernet: HyperNet or MultiHyperNet whose
                ``calculate_order_n_distance`` method defines the metric.
                When ``None``, a single cosine distance is used as fallback.

        Returns:
            E_guide: Best candidate edges as one-hot (bs, n, n, de).
        """
        K, bs, order_n_dim = candidate_hdcs.shape
        n = candidates.shape[2]
        device = candidates.device

        # Expand target to match candidates: (K, bs, order_n_dim)
        target_expanded = target_order_n.unsqueeze(0).expand(K, -1, -1)

        # Flatten to (K*bs, order_n_dim) for batched distance computation
        cand_flat = candidate_hdcs.reshape(K * bs, order_n_dim)
        tgt_flat = target_expanded.reshape(K * bs, order_n_dim)

        # Delegate distance computation to the hypernet so that
        # MultiHyperNet's per-chunk averaging is used automatically.
        if hypernet is not None:
            distances_flat = hypernet.calculate_order_n_distance(cand_flat, tgt_flat)
        else:
            distances_flat = 1.0 - F.cosine_similarity(
                cand_flat.float(), tgt_flat.float(), dim=-1,
            )

        distances = distances_flat.reshape(K, bs)  # (K, bs)

        # Find best candidate per batch sample (minimum distance)
        best_indices = torch.argmin(distances, dim=0)  # (bs,)

        # Gather best candidates: (bs, n, n)
        best_edges = torch.zeros(bs, n, n, dtype=torch.long, device=device)
        for b in range(bs):
            best_edges[b] = candidates[best_indices[b].item(), b]

        # Convert to one-hot
        E_guide = F.one_hot(best_edges, num_classes=self.num_edge_classes).float()

        return E_guide

    def _compute_Rhdc(
        self,
        E_guide: Tensor,
        E_t_label: Tensor,
        Z_t_E: Tensor,
        pt_at_Et: Tensor,
        gamma: float,
    ) -> Tensor:
        """
        Compute R^HDC rate matrix term.

        Follows R^TG pattern:
        R^HDC = (gamma / (Z_t_E * pt_at_Et)) * mask * E_guide_onehot

        Where mask = 1 if E_guide != E_t, else 0

        Args:
            E_guide: Best candidate one-hot (bs, n, n, de)
            E_t_label: Current state labels (bs, n, n)
            Z_t_E: Normalization count (bs, n, n)
            pt_at_Et: p(E_t | E_1) at current state (bs, n, n)
            gamma: HDC guidance strength

        Returns:
            R_hdc: (bs, n, n, de) rate matrix contribution
        """
        if gamma == 0:
            return torch.zeros_like(E_guide)

        # Get E_guide labels
        E_guide_label = torch.argmax(E_guide, dim=-1)  # (bs, n, n)

        # Mask: only add guidance when E_guide != E_t
        mask = (E_guide_label != E_t_label).float().unsqueeze(-1)  # (bs, n, n, 1)

        # Denominator: Z_t_E * pt_at_Et
        denom = (Z_t_E * pt_at_Et).unsqueeze(-1)  # (bs, n, n, 1)

        # R^HDC = gamma * E_guide * mask / denom
        R_hdc = (gamma * E_guide * mask) / (denom + 1e-8)

        return R_hdc

    def _compute_dfm_variables_for_edges(
        self,
        t: Tensor,
        E_t_label: Tensor,
        E_1_sampled: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Compute discrete flow matching variables for edges.

        This replicates the logic from RateMatrixDesigner._compute_dfm_variables
        but only for edges, to get Z_t_E and pt_at_Et for R^HDC computation.

        Args:
            t: Current time (bs, 1)
            E_t_label: Current edge labels (bs, n, n)
            E_1_sampled: Sampled clean edge labels (bs, n, n)

        Returns:
            Dict with pt_vals_at_Et and Z_t_E
        """
        device = E_t_label.device
        bs = E_t_label.shape[0]

        # Get limit distribution
        limit_E = self.limit_dist.E.to(device)
        de = len(limit_E)

        # p(E_t | E_1) = t * delta(E_1) + (1-t) * p_0
        t_time = t.squeeze(-1)[:, None, None, None]  # (bs, 1, 1, 1)

        E1_onehot = F.one_hot(E_1_sampled, num_classes=de).float()  # (bs, n, n, de)

        pt_E = t_time * E1_onehot + (1 - t_time) * limit_E[None, None, None, :]
        pt_E = pt_E.clamp(0, 1)

        # Gather at current state
        E_t_label_expanded = E_t_label.unsqueeze(-1)  # (bs, n, n, 1)
        pt_at_Et = pt_E.gather(-1, E_t_label_expanded).squeeze(-1)  # (bs, n, n)

        # Count non-zero probabilities
        Z_t_E = (pt_E > 0).sum(dim=-1).float()  # (bs, n, n)

        return {
            "pt_vals_at_Et": pt_at_Et,
            "Z_t_E": Z_t_E,
        }

    # =========================================================================
    # Soft HDC Gradient Guidance
    # =========================================================================

    @staticmethod
    def _fft_bind(a: Tensor, b: Tensor) -> Tensor:
        """
        HRR circular convolution via FFT (differentiable).

        Equivalent to ``torchhd.bind`` for HRR tensors but operates on plain
        float tensors so that ``torch.autograd.grad`` can backpropagate
        through the operation.

        Args:
            a: Tensor of shape (..., D).
            b: Tensor of shape (..., D).

        Returns:
            Circular convolution result, same shape as inputs.
        """
        return torch.fft.ifft(torch.fft.fft(a) * torch.fft.fft(b)).real

    @staticmethod
    def _soft_hdc_encode(
        soft_A: Tensor,
        node_hvs: Tensor,
        node_mask: Tensor,
        depth: int,
        normalize: bool,
    ) -> Tensor:
        """
        Differentiable soft HDC encoding using a soft adjacency matrix.

        Mirrors the message passing loop of ``HyperNet.forward()``
        (encoder.py L345-362) but replaces the sparse
        ``scatter_hd(messages, srcs, op="bundle")`` with a dense matrix
        multiply ``bmm(soft_A, node_hvs)``.  Since bundle = element-wise
        sum and matrix multiplication is a weighted sum, this is an exact
        relaxation.

        The bind step (circular convolution) is implemented via FFT and is
        fully differentiable.

        Args:
            soft_A: Soft adjacency matrix (bs, n, n), values in [0, 1].
            node_hvs: Pre-computed node hypervectors (bs, n, D).
            node_mask: Valid node mask (bs, n).
            depth: Number of message passing layers (= hypernet.depth).
            normalize: Whether to L2-normalize after each layer.

        Returns:
            Graph embedding (bs, D).
        """
        bs, n, D = node_hvs.shape

        # Build layer list (avoids in-place tensor ops for autograd)
        hv_layers = [node_hvs]  # layer 0

        for layer in range(depth):
            prev_hv = hv_layers[-1]  # (bs, n, D)

            # Soft message passing: weighted sum of neighbor HVs
            aggregated = torch.bmm(soft_A, prev_hv)  # (bs, n, D)

            # Bind: circular convolution of prev_hv with aggregated
            hr = FlowEdgeDecoder._fft_bind(prev_hv, aggregated)  # (bs, n, D)

            if normalize:
                hr = hr / (hr.norm(dim=-1, keepdim=True) + 1e-8)

            hv_layers.append(hr)

        # Multi-bundle across layers: sum over stacked layer dimension
        # Equivalent to torchhd.multibundle for HRR
        stacked = torch.stack(hv_layers, dim=1)  # (bs, depth+1, n, D)
        node_emb = stacked.sum(dim=1)  # (bs, n, D)

        # Mask invalid nodes
        node_emb = node_emb * node_mask.unsqueeze(-1).float()

        # Graph readout: sum over nodes (= scatter_hd with op="bundle")
        graph_emb = node_emb.sum(dim=1)  # (bs, D)

        return graph_emb

    @torch.no_grad()
    def sample_with_soft_hdc_guidance(
        self,
        hdc_vectors: Tensor,
        node_features: Tensor,
        node_mask: Tensor,
        hypernet,
        raw_node_features: Tensor,
        gamma: float = 0.5,
        tau: float = 0.5,
        integration_mode: str = "blend",
        schedule: str = "linear_decay",
        eta: Optional[float] = None,
        omega: Optional[float] = None,
        sample_steps: Optional[int] = None,
        time_distortion: Optional[str] = None,
        noise_type_override: Optional[str] = None,
        device: Optional[torch.device] = None,
        show_progress: bool = True,
        step_callback: Optional[Callable[[int, float, Tensor, Tensor, Tensor, Optional[Tensor]], None]] = None,
        deterministic: bool = False,
        initial_edges: Optional[Tensor] = None,
        start_time: float = 0.0,
    ) -> List[Data]:
        """
        Generate edges with soft HDC gradient guidance.

        At each timestep the model's predicted edge logits are relaxed into
        a differentiable soft adjacency matrix.  A lightweight soft HDC
        encoder (matrix-multiply message passing + FFT circular convolution)
        maps this to a graph embedding.  The cosine distance to the target
        order_N is back-propagated to obtain per-edge, per-class gradients
        that steer sampling toward HDC-consistent configurations.

        Two integration modes are supported:

        * ``"blend"``: the gradient-derived target distribution is mixed
          with the model's prediction *before* rate matrix computation.
        * ``"rate_matrix"``: an additive ``R^HDC`` term constructed from
          the negative gradient is added to the rate matrix *after* the
          standard computation.

        Args:
            hdc_vectors: Full HDC vectors [order_0 | order_N]
                         (num_samples, hdc_dim).
            node_features: Fixed node types one-hot
                           (num_samples, max_n, num_node_classes).
            node_mask: Valid node mask (num_samples, max_n).
            hypernet: HyperNet encoder (used for codebook lookup and depth).
            raw_node_features: Raw integer node features for codebook lookup
                               (num_samples, max_n, num_raw_features).
            gamma: Guidance strength (default 0.5).
            tau: Softmax temperature for soft edge probabilities and for
                 converting the negative gradient into a distribution
                 (default 0.5).
            integration_mode: ``"blend"`` or ``"rate_matrix"`` (default
                              ``"blend"``).
            schedule: Time-dependent strength schedule.  ``"constant"``
                      uses *gamma* at every step; ``"linear_decay"`` uses
                      ``gamma * (1 - t)`` (more guidance early);
                      ``"linear_ramp"`` uses ``gamma * t`` (more guidance
                      late).  Default ``"linear_decay"``.
            eta: Stochasticity parameter (None = use default).
            omega: Target guidance strength (None = use default).
            sample_steps: Number of denoising steps (None = use default).
            time_distortion: Time distortion type (None = use default).
            noise_type_override: Override noise type for initialization.
            device: Device to run on (defaults to CPU).
            show_progress: Whether to show progress bar.
            step_callback: Optional callback invoked at each step with
                           ``(step, t, X, E, node_mask, pred_E)``.
            deterministic: If True, use argmax instead of sampling.
            initial_edges: Optional pre-computed one-hot edge tensor for
                           warm-starting.
            start_time: Normalized start time in [0.0, 1.0).

        Returns:
            List of PyG Data objects with generated edges.
        """
        self.eval()

        # ----- defaults -----
        if eta is None:
            eta = self.default_eta
        if omega is None:
            omega = self.default_omega
        if sample_steps is None:
            sample_steps = self.default_sample_steps
        if time_distortion is None:
            time_distortion = self.default_sample_time_distortion

        self.rate_matrix_designer.eta = eta
        self.rate_matrix_designer.omega = omega

        # Handle noise type override
        if noise_type_override is not None:
            sampling_limit_dist = self._create_limit_distribution(noise_type_override)
            original_limit_dist = self.rate_matrix_designer.limit_dist
            self.rate_matrix_designer.limit_dist = sampling_limit_dist
        else:
            sampling_limit_dist = self.limit_dist
            original_limit_dist = None

        device = device or torch.device("cpu")
        self.to(device)

        hdc_vectors = hdc_vectors.to(device).float()
        X = node_features.to(device).float()
        node_mask = node_mask.to(device)
        raw_node_features = raw_node_features.to(device)

        num_samples = hdc_vectors.size(0)
        n_max = X.size(1)

        # ----- pre-compute node HVs from codebook (one-time) -----
        # For MultiHyperNet, use the primary sub-HyperNet for guidance
        from graph_hdc.hypernet.multi_hypernet import MultiHyperNet
        if isinstance(hypernet, MultiHyperNet):
            primary = hypernet.primary
        else:
            primary = hypernet

        D = primary.hv_dim

        # Extract target order_N for the primary HyperNet only.
        # Layout is [order_0(D) | order_N_primary(D) | order_N_rest(...)]
        target_order_n = hdc_vectors[:, D : 2 * D]  # (num_samples, D)
        depth = primary.depth
        normalize_hdc = primary.normalize
        node_hvs = torch.zeros(num_samples, n_max, D, device=device)

        for b in range(num_samples):
            n_valid = node_mask[b].sum().int().item()
            for i in range(n_valid):
                tup = tuple(raw_node_features[b, i].long().tolist())
                idx = primary.nodes_indexer.get_idx(tup)
                if idx is not None:
                    node_hvs[b, i] = primary.nodes_codebook[idx].float().to(device)
                # else: leave as zeros (unknown node type, no guidance)

        # ----- initialize edges -----
        edge_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
        if initial_edges is not None:
            E = initial_edges.to(device).float()
            E = E * edge_mask.unsqueeze(-1).float()
        else:
            e_limit = sampling_limit_dist.E.to(device)
            e_probs = e_limit.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(
                num_samples, n_max, n_max, -1
            )
            prob_E_flat = e_probs.reshape(-1, self.num_edge_classes)
            E_label_flat = torch.multinomial(prob_E_flat, 1).squeeze(-1)
            E_label = E_label_flat.reshape(num_samples, n_max, n_max)
            E_label = torch.triu(E_label, diagonal=1)
            E_label = E_label + E_label.transpose(1, 2)
            E = F.one_hot(E_label, num_classes=self.num_edge_classes).float()
            E = E * edge_mask.unsqueeze(-1).float()

        hdc_raw = hdc_vectors if self.use_cross_attn else None
        y = self.condition_mlp(hdc_vectors)

        # ----- sampling loop -----
        iterator = range(sample_steps)
        if show_progress:
            iterator = tqdm(iterator, desc="Soft HDC-guided sampling")

        if step_callback is not None:
            step_callback(0, start_time, X.clone(), E.clone(), node_mask, None)

        de = self.num_edge_classes

        for t_int in iterator:
            if t_int / sample_steps < start_time:
                continue

            t_norm = torch.tensor([t_int / sample_steps], device=device)
            s_norm = torch.tensor([(t_int + 1) / sample_steps], device=device)
            t_norm = self.time_distorter.sample_ft(t_norm, time_distortion)
            s_norm = self.time_distorter.sample_ft(s_norm, time_distortion)

            t = t_norm.expand(num_samples, 1)
            s = s_norm.expand(num_samples, 1)
            dt = (s - t)[0]

            # Compute alpha from schedule (use un-distorted fraction)
            t_frac = t_int / sample_steps
            if schedule == "constant":
                alpha = gamma
            elif schedule == "linear_decay":
                alpha = gamma * (1.0 - t_frac)
            elif schedule == "linear_ramp":
                alpha = gamma * t_frac
            else:
                raise ValueError(f"Unknown schedule: {schedule!r}")

            # ----- forward pass -----
            noisy_data = {
                "t": t, "X_t": X, "E_t": E, "y_t": y, "node_mask": node_mask,
            }
            if hdc_raw is not None:
                noisy_data["hdc_raw"] = hdc_raw

            extra_data = self._compute_extra_data(noisy_data)
            pred = self.forward(noisy_data, extra_data, node_mask)

            pred_X = F.softmax(pred.X, dim=-1)
            pred_E_logits = pred.E  # keep raw logits for gradient
            pred_E = F.softmax(pred_E_logits, dim=-1)

            # ----- soft HDC gradient -----
            if alpha > 0:
                with torch.enable_grad():
                    logits_g = pred_E_logits.detach().clone().requires_grad_(True)
                    soft_probs = F.softmax(logits_g / tau, dim=-1)
                    # Soft adjacency: probability of any bond existing
                    soft_A = soft_probs[:, :, :, 1:].sum(dim=-1)  # (bs, n, n)
                    # Symmetrize
                    soft_A = torch.triu(soft_A, diagonal=1)
                    soft_A = soft_A + soft_A.transpose(1, 2)
                    soft_A = soft_A * edge_mask.float()

                    graph_emb = self._soft_hdc_encode(
                        soft_A, node_hvs, node_mask, depth, normalize_hdc,
                    )

                    graph_emb_n = F.normalize(graph_emb, dim=-1)
                    target_n = F.normalize(target_order_n.float(), dim=-1)
                    loss = (1.0 - (graph_emb_n * target_n).sum(dim=-1)).sum()

                    grad = torch.autograd.grad(loss, logits_g)[0]  # (bs, n, n, de)
                # Outside enable_grad: grad is a regular detached tensor

                if integration_mode == "blend":
                    guidance_dist = F.softmax(-grad / tau, dim=-1)
                    pred_E = (1.0 - alpha) * pred_E + alpha * guidance_dist
                    pred_E = pred_E / pred_E.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            # ----- rate matrices -----
            E_t_label = torch.argmax(E, dim=-1)

            R_t_X, R_t_E = self.rate_matrix_designer.compute_rate_matrices(
                t, node_mask, X, E, pred_X, pred_E,
            )

            if alpha > 0 and integration_mode == "rate_matrix":
                grad_n = grad / (grad.norm(dim=-1, keepdim=True).clamp(min=1e-8))
                R_hdc = F.relu(-grad_n) * alpha
                R_hdc.scatter_(-1, E_t_label.unsqueeze(-1), 0.0)
                R_t_E = R_t_E + R_hdc

            # ----- sample next state -----
            step_probs_E = R_t_E * dt.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            step_probs_E.scatter_(-1, E_t_label.unsqueeze(-1), 0.0)
            stay_prob = 1.0 - step_probs_E.sum(dim=-1, keepdim=True).clamp(min=0)
            step_probs_E.scatter_(-1, E_t_label.unsqueeze(-1), stay_prob)

            step_probs_E = step_probs_E.clamp(min=0)
            step_probs_E = step_probs_E / step_probs_E.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            if s[0].item() >= 1.0 - 1e-6:
                step_probs_E = pred_E

            prob_E_flat = step_probs_E.reshape(-1, de)
            if deterministic:
                E_s_label_flat = torch.argmax(prob_E_flat, dim=-1)
            else:
                E_s_label_flat = torch.multinomial(prob_E_flat, 1).squeeze(-1)
            E_s_label = E_s_label_flat.reshape(num_samples, n_max, n_max)

            E_s_label = torch.triu(E_s_label, diagonal=1)
            E_s_label = E_s_label + E_s_label.transpose(1, 2)

            E = F.one_hot(E_s_label, num_classes=de).float()
            E = E * edge_mask.unsqueeze(-1).float()

            if step_callback is not None:
                step_callback(
                    t_int + 1, s_norm.item(),
                    X.clone(), E.clone(), node_mask, pred_E.clone(),
                )

        # ----- finalize -----
        n_nodes = node_mask.sum(dim=1).int()
        samples = dense_to_pyg(X, E, torch.zeros_like(y[:, :0]), node_mask, n_nodes)

        if original_limit_dist is not None:
            self.rate_matrix_designer.limit_dist = original_limit_dist

        return samples

    @torch.no_grad()
    def sample_with_hdc(
        self,
        hdc_vectors: Tensor,
        node_features: Tensor,
        node_mask: Tensor,
        original_node_features: Tensor,
        hypernet,
        distance_threshold: float = 0.001,
        eta: Optional[float] = None,
        omega: Optional[float] = None,
        sample_steps: Optional[int] = None,
        time_distortion: Optional[str] = None,
        noise_type_override: Optional[str] = None,
        device: Optional[torch.device] = None,
        show_progress: bool = True,
    ) -> List[Data]:
        """
        Generate edges with HDC distance tracking and early stopping.

        At each timestep:
        1. Get model prediction and compute deterministic argmax edges
        2. Encode with HyperNet to get order_N
        3. Compute cosine distance to target order_N
        4. Track best match per sample (lowest distance seen)
        5. Early stop if all samples are below distance_threshold

        Returns the best-seen edges (not final edges) for each sample.

        Args:
            hdc_vectors: Full HDC vectors [order_0 | order_N] (num_samples, hdc_dim)
            node_features: Fixed node types one-hot (num_samples, max_n, num_node_classes)
            node_mask: Valid node mask (num_samples, max_n)
            original_node_features: Original node features for HyperNet (num_samples, max_n, feat_dim)
            hypernet: HyperNet encoder for candidate evaluation
            distance_threshold: Cosine distance threshold for early stopping (default: 0.001)
            eta: Stochasticity parameter (None = use default)
            omega: Target guidance strength (None = use default)
            sample_steps: Number of denoising steps (None = use default)
            time_distortion: Time distortion type (None = use default)
            noise_type_override: Override noise type for initialization
            device: Device to run on
            show_progress: Whether to show progress bar

        Returns:
            List of PyG Data objects with best-match edges
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

        # Handle noise type override
        if noise_type_override is not None:
            sampling_limit_dist = self._create_limit_distribution(noise_type_override)
            original_limit_dist = self.rate_matrix_designer.limit_dist
            self.rate_matrix_designer.limit_dist = sampling_limit_dist
        else:
            sampling_limit_dist = self.limit_dist
            original_limit_dist = None

        # Setup device
        device = device or torch.device("cpu")
        self.to(device)
        hypernet = hypernet.to(device)
        hypernet.eval()

        hdc_vectors = hdc_vectors.to(device).float()
        X = node_features.to(device).float()
        node_mask = node_mask.to(device)
        original_node_features = original_node_features.to(device).float()

        num_samples = hdc_vectors.size(0)
        n_max = X.size(1)

        # Extract target order_N.  For a single HyperNet the layout is
        # [order_0(D) | order_N(D)]; for a MultiHyperNet it is
        # [order_0(D) | order_N_1(D1) | … | order_N_K(DK)].
        # In both cases order_N starts at offset ``hypernet.hv_dim``.
        target_order_n = hdc_vectors[:, hypernet.hv_dim:]  # (num_samples, order_n_dim)

        # Initialize best tracking
        best_E = None  # Will store (bs, n, n) edge labels
        best_distances = torch.full((num_samples,), float('inf'), device=device)

        # Sample initial noise for edges
        e_limit = sampling_limit_dist.E.to(device)
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

        # Keep raw HDC vectors for cross-attention conditioning
        hdc_raw = hdc_vectors if self.use_cross_attn else None

        # HDC vectors reduced through MLP as y
        y = self.condition_mlp(hdc_vectors)

        # Sampling loop
        iterator = range(sample_steps)
        if show_progress:
            iterator = tqdm(iterator, desc="HDC sampling with early stop")

        for t_int in iterator:
            t_norm = torch.tensor([t_int / sample_steps], device=device)
            s_norm = torch.tensor([(t_int + 1) / sample_steps], device=device)

            # Apply time distortion
            t_norm = self.time_distorter.sample_ft(t_norm, time_distortion)
            s_norm = self.time_distorter.sample_ft(s_norm, time_distortion)

            # Expand to batch size
            t = t_norm.expand(num_samples, 1)
            s = s_norm.expand(num_samples, 1)

            # Prepare noisy data dict
            noisy_data = {
                "t": t,
                "X_t": X,
                "E_t": E,
                "y_t": y,
                "node_mask": node_mask,
            }
            if hdc_raw is not None:
                noisy_data["hdc_raw"] = hdc_raw

            # Compute extra features
            extra_data = self._compute_extra_data(noisy_data)

            # Forward pass
            pred = self.forward(noisy_data, extra_data, node_mask)

            # Get predictions as probability distributions
            pred_X = F.softmax(pred.X, dim=-1)
            pred_E = F.softmax(pred.E, dim=-1)

            # === HDC Distance Check ===
            # Get deterministic prediction via argmax
            E_pred_label = torch.argmax(pred_E, dim=-1)  # (bs, n, n)

            # Make symmetric
            E_pred_label = torch.triu(E_pred_label, diagonal=1)
            E_pred_label = E_pred_label + E_pred_label.transpose(1, 2)

            # Encode with HyperNet (reuse existing helper with K=1)
            pred_order_n = self._encode_candidates_to_order_n(
                E_pred_label.unsqueeze(0), original_node_features, node_mask, hypernet
            ).squeeze(0)  # (bs, hdc_dim)

            # Compute cosine distance using the hypernet's own metric.
            # For MultiHyperNet this averages per-sub-HyperNet distances.
            distances = hypernet.calculate_order_n_distance(
                pred_order_n, target_order_n,
            )  # (bs,)

            # Update best for samples where this is better
            improved = distances < best_distances
            if improved.any():
                if best_E is None:
                    best_E = E_pred_label.clone()
                best_E[improved] = E_pred_label[improved]
                best_distances[improved] = distances[improved]

            # Early stopping check
            if (best_distances < distance_threshold).all():
                if show_progress:
                    print(f"\nEarly stop at step {t_int + 1}/{sample_steps}, "
                          f"max distance: {best_distances.max().item():.6f}")
                break

            # === Standard Sampling Step (no R^HDC) ===
            # Use _sample_step which handles everything except HDC guidance
            X, E, y, _ = self._sample_step(t, s, X, E, y, node_mask, hdc_raw=hdc_raw)

        # Determine which samples found a good match (below threshold)
        found_good_match = best_distances < distance_threshold

        # Get final sampled state labels
        final_E_label = torch.argmax(E, dim=-1)

        # Combine results:
        # - For samples that found good match: use best_E (best prediction by HDC distance)
        # - For samples that didn't: use final sampled state
        if best_E is None:
            # No improvements found at all - use final state for everything
            result_E = final_E_label
        else:
            result_E = best_E.clone()
            # Override with final sampled state for samples that didn't find good match
            result_E[~found_good_match] = final_E_label[~found_good_match]

        # Convert to one-hot and then to PyG Data
        result_E_onehot = F.one_hot(result_E, num_classes=self.num_edge_classes).float()
        result_E_onehot = result_E_onehot * edge_mask.unsqueeze(-1).float()
        n_nodes = node_mask.sum(dim=1).int()
        samples = dense_to_pyg(X, result_E_onehot, torch.zeros_like(y[:, :0]), node_mask, n_nodes)

        # Restore original limit_dist if overridden
        if original_limit_dist is not None:
            self.rate_matrix_designer.limit_dist = original_limit_dist

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

def get_bond_type_idx(bond) -> int:
    """Convert RDKit bond to 5-class index."""
    if bond is None:
        return 0  # No edge
    return BOND_TYPE_TO_IDX.get(bond.GetBondType(), 0)


def preprocess_for_flow_edge_decoder(
    data: Data,
    hypernet,
    device: torch.device = None,
) -> Optional[Data]:
    """
    Preprocess a PyG Data object for FlowEdgeDecoder training (ZINC only).

    Adds/replaces:
    - x: One-hot encoding (24-dim base, or extended with RW bins when enabled)
    - edge_attr: 5-class bond type one-hot encoding
    - hdc_vector: Pre-computed HDC embedding
    - original_x: Raw feature indices for HDC-guided sampling

    Args:
        data: Original PyG Data object with ZINC features
              data.x should have shape (n, 5) with columns:
              [atom_type, degree, charge, num_hs, is_in_ring]
        hypernet: HyperNet encoder for computing HDC embeddings
        device: Device for HDC computation

    Returns:
        Preprocessed Data object, or None if molecule has no edges
    """
    device = device or torch.device("cpu")

    # Skip molecules with no edges (single atoms)
    if data.edge_index.numel() == 0:
        return None

    # 1. Build node features — base 5-dim ZINC features, optionally extended
    #    with binned RW return probabilities when the encoder uses them.
    rw_config = getattr(hypernet, "rw_config", None)
    feature_bins = get_node_feature_bins(rw_config)

    raw_feats = data.x.clone()  # (n, 5) base ZINC features
    if rw_config is not None and rw_config.enabled:
        from graph_hdc.utils.rw_features import (
            bin_rw_probabilities,
            compute_rw_return_probabilities,
        )
        rw_probs = compute_rw_return_probabilities(
            data.edge_index, data.x.size(0), rw_config.k_values,
        )
        rw_binned = bin_rw_probabilities(rw_probs, rw_config.num_bins, bin_boundaries=rw_config.bin_boundaries, k_values=rw_config.k_values)
        raw_feats = torch.cat([raw_feats, rw_binned], dim=-1)

    x_onehot = raw_features_to_onehot(raw_feats, feature_bins=feature_bins)

    # 2. Create 5-class edge attributes from SMILES
    mol = Chem.MolFromSmiles(data.smiles)
    if mol is None:
        return None

    edge_index = data.edge_index

    # Build adjacency with bond types
    edge_attr_list = []
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        bond = mol.GetBondBetweenAtoms(src, dst)
        bond_type = get_bond_type_idx(bond)
        edge_attr_list.append(bond_type)

    edge_attr = torch.tensor(edge_attr_list, dtype=torch.long)
    edge_attr_onehot = F.one_hot(edge_attr, num_classes=NUM_EDGE_CLASSES).float()

    # 3. Compute HDC embedding: concatenate order-0 and order-N
    data_for_hdc = data.clone().to(device)
    # Add batch attribute for single graph (required by HyperNet)
    if data_for_hdc.batch is None:
        data_for_hdc.batch = torch.zeros(data_for_hdc.x.size(0), dtype=torch.long, device=device)

    # Augment with RW features if the encoder was built with them
    if hasattr(hypernet, "rw_config") and hypernet.rw_config.enabled:
        from graph_hdc.utils.rw_features import augment_data_with_rw

        data_for_hdc = augment_data_with_rw(
            data_for_hdc,
            k_values=hypernet.rw_config.k_values,
            num_bins=hypernet.rw_config.num_bins,
            bin_boundaries=hypernet.rw_config.bin_boundaries,
        )

    with torch.no_grad():
        # forward() handles encode_properties internally and returns:
        # - node_terms: order-0 (bundled node HVs, RRWP-enriched for RRWPHyperNet)
        # - graph_embedding: order-N (message passing result)
        hdc_out = hypernet.forward(data_for_hdc)
        order_zero = hdc_out["node_terms"]
        order_n = hdc_out["graph_embedding"]

        # Concatenate [order_0 | order_N]
        hdc_concat = torch.cat([order_zero, order_n], dim=-1).cpu().float()
        # Convert from HRRTensor subclass to regular tensor
        hdc_vector = torch.Tensor(hdc_concat.tolist())
        if hdc_vector.dim() == 1:
            hdc_vector = hdc_vector.unsqueeze(0)  # Ensure (1, 2*hdc_dim) for proper batching

    # 4. Create new Data object
    new_data = Data(
        x=x_onehot,  # one-hot: base 24-dim, or extended when RW enabled
        edge_index=edge_index.clone(),
        edge_attr=edge_attr_onehot,
        hdc_vector=hdc_vector,
        smiles=data.smiles,
        # Keep original raw features for HDC-guided sampling
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
    device: torch.device = None,
    show_progress: bool = True,
) -> List[Data]:
    """
    Preprocess entire dataset for FlowEdgeDecoder (ZINC only).

    Args:
        dataset: PyG dataset or list of Data objects (ZINC format)
        hypernet: HyperNet encoder
        device: Device for HDC computation
        show_progress: Whether to show progress bar

    Returns:
        List of preprocessed Data objects with 24-dim node features
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hypernet = hypernet.to(device)
    hypernet.eval()

    processed = []
    iterator = dataset
    if show_progress:
        iterator = tqdm(iterator, desc="Preprocessing ZINC")

    for data in iterator:
        new_data = preprocess_for_flow_edge_decoder(data, hypernet, device)
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
