"""
Flow Matching model for learning distributions over HDC graph encodings.

Uses Conditional Flow Matching (CFM) with optional minibatch Optimal Transport
coupling. Built on the Facebook Research ``flow_matching`` library for probability
paths and ODE solving.

The model is agnostic to the internal structure of the input vectors -- it treats
them as flat tensors. It supports optional FiLM-based conditioning (defaulting to
unconditional when condition_dim=0 or when a zero condition is passed).
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from flow_matching.path.affine import CondOTProbPath
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from scipy.optimize import linear_sum_assignment
from torch import Tensor


# =============================================================================
# Time Embedding
# =============================================================================


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for time."""

    def __init__(self, embed_dim: int, max_period: float = 10000.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_period = max_period

    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: (batch_size,) or (batch_size, 1) with values in [0, 1].

        Returns:
            (batch_size, embed_dim)
        """
        if t.dim() == 2:
            t = t.squeeze(-1)

        t = t.unsqueeze(-1)  # (bs, 1)
        half_dim = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, device=t.device, dtype=t.dtype)
            / half_dim
        )
        args = t * 1000.0 * freqs.unsqueeze(0)  # (bs, half_dim)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# =============================================================================
# FiLM Residual Block
# =============================================================================


class FiLMResidualBlock(nn.Module):
    """Residual MLP block with FiLM conditioning from time + condition."""

    def __init__(self, hidden_dim: int, film_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # FiLM: condition -> (scale1, shift1, scale2, shift2)
        self.film_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(film_dim, 4 * hidden_dim),
        )

        # Zero-init so network starts near identity
        nn.init.zeros_(self.film_proj[-1].weight)
        nn.init.zeros_(self.film_proj[-1].bias)

    def forward(self, x: Tensor, film_emb: Tensor) -> Tensor:
        film_params = self.film_proj(film_emb)
        s1, b1, s2, b2 = film_params.chunk(4, dim=-1)

        h = self.norm1(x)
        h = h * (1 + s1) + b1
        h = F.silu(h)
        h = self.linear1(h)
        h = self.dropout(h)

        h = self.norm2(h)
        h = h * (1 + s2) + b2
        h = F.silu(h)
        h = self.linear2(h)
        h = self.dropout(h)

        return x + h


# =============================================================================
# Velocity MLP
# =============================================================================


class VelocityMLP(nn.Module):
    """
    MLP velocity field v_theta(x_t, t, condition) for flow matching.

    Uses FiLM conditioning from time (+ optional condition) embeddings
    applied at each residual block.
    """

    def __init__(
        self,
        data_dim: int,
        hidden_dim: int = 768,
        num_blocks: int = 6,
        time_embed_dim: int = 128,
        condition_dim: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim

        # Input projection
        self.input_proj = nn.Linear(data_dim, hidden_dim)

        # Time embedding: sinusoidal -> MLP
        film_dim = hidden_dim
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, film_dim),
            nn.SiLU(),
            nn.Linear(film_dim, film_dim),
        )

        # Condition embedding (optional)
        if condition_dim > 0:
            self.cond_embed = nn.Sequential(
                nn.Linear(condition_dim, film_dim),
                nn.SiLU(),
                nn.Linear(film_dim, film_dim),
            )
        else:
            self.cond_embed = None

        # Residual FiLM blocks
        self.blocks = nn.ModuleList(
            [FiLMResidualBlock(hidden_dim, film_dim, dropout) for _ in range(num_blocks)]
        )

        # Output projection (zero-initialized)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, data_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        condition: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Predict velocity field at (x_t, t).

        Args:
            x_t: (batch_size, data_dim) noisy data at time t.
            t: (batch_size,) time in [0, 1].
            condition: (batch_size, condition_dim) optional condition.

        Returns:
            (batch_size, data_dim) predicted velocity.
        """
        if t.dim() == 2:
            t = t.squeeze(-1)

        h = self.input_proj(x_t)

        film_emb = self.time_embed(t)
        if self.cond_embed is not None and condition is not None:
            film_emb = film_emb + self.cond_embed(condition)

        for block in self.blocks:
            h = block(h, film_emb)

        h = self.output_norm(h)
        return self.output_proj(h)


# =============================================================================
# Model Wrapper for flow_matching library
# =============================================================================


class VelocityModelWrapper(ModelWrapper):
    """Wraps VelocityMLP for use with flow_matching ODESolver."""

    def __init__(self, velocity_net: VelocityMLP):
        super().__init__(velocity_net)

    def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
        condition = extras.get("condition", None)
        return self.model(x_t=x, t=t, condition=condition)


# =============================================================================
# Minibatch OT Coupling
# =============================================================================


class MinibatchOTCoupler:
    """
    Minibatch Optimal Transport coupling for flow matching.

    Computes the optimal assignment between noise and data samples within a
    minibatch using the Hungarian algorithm on L2 cost, then reorders the
    noise samples accordingly.
    """

    @staticmethod
    @torch.no_grad()
    def couple(x0: Tensor, x1: Tensor) -> Tensor:
        """
        Reorder noise x0 to optimally match data x1 under L2 cost.

        Args:
            x0: (batch_size, dim) noise samples.
            x1: (batch_size, dim) data samples.

        Returns:
            (batch_size, dim) reordered noise.
        """
        cost = torch.cdist(x0, x1, p=2).pow(2)
        cost_np = cost.cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_np)

        # row_ind[i] is paired with col_ind[i].
        # For a square matrix, row_ind = [0, 1, ..., n-1].
        # We need x0_reordered[j] = x0 assigned to x1[j].
        inverse_perm = torch.empty(len(col_ind), dtype=torch.long)
        inverse_perm[col_ind] = torch.arange(len(col_ind))

        return x0[inverse_perm.to(x0.device)]


# =============================================================================
# Conditioning Interface
# =============================================================================


class ConditioningInterface(ABC):
    """
    Abstract interface for molecular property conditioning.

    Each implementation computes a scalar property from a SMILES string.
    Properties are stored on PyG Data objects during preprocessing and
    passed to the flow matching model as condition vectors.
    """

    @abstractmethod
    def evaluate(self, smiles: str) -> float:
        """Compute the property value from a SMILES string."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used as registry key and dict key."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the property."""

    @abstractmethod
    def value_range(self) -> Tuple[float, float]:
        """Expected (min, max) range of the property values."""

    @property
    def condition_dim(self) -> int:
        """Dimension of the condition vector produced by this interface."""
        return 1

    def evaluate_batch(self, smiles_list: Sequence[str]) -> Tensor:
        """Compute property values for a batch of SMILES strings.

        Returns:
            (batch_size, condition_dim) tensor.
        """
        values = [self.evaluate(s) for s in smiles_list]
        return torch.tensor(values, dtype=torch.float32).unsqueeze(-1)

    def sample_condition(self, value: float) -> Tensor:
        """Create a condition tensor for a target property value.

        Args:
            value: Target property value.

        Returns:
            (condition_dim,) tensor.
        """
        return torch.tensor([value], dtype=torch.float32)


class MultiCondition(ConditioningInterface):
    """
    Combines multiple conditioning interfaces into a single multi-dimensional
    condition vector by concatenation.

    This is the standard way to use conditions -- even a single property
    should be wrapped in a MultiCondition for consistency.

    Args:
        conditions: List of ConditioningInterface instances.
    """

    def __init__(self, conditions: List[ConditioningInterface]):
        if not conditions:
            raise ValueError("MultiCondition requires at least one condition.")
        self._conditions = conditions
        self._name_to_idx = {c.name: i for i, c in enumerate(conditions)}

    def evaluate(self, smiles: str) -> float:
        # For single condition, return scalar. For multi, use evaluate_batch.
        if len(self._conditions) == 1:
            return self._conditions[0].evaluate(smiles)
        raise TypeError(
            "MultiCondition with >1 condition does not return a scalar. "
            "Use evaluate_batch() instead."
        )

    def evaluate_multi(self, smiles: str) -> Tensor:
        """Compute all property values for a single SMILES string.

        Returns:
            (condition_dim,) tensor.
        """
        values = [c.evaluate(smiles) for c in self._conditions]
        return torch.tensor(values, dtype=torch.float32)

    def evaluate_batch(self, smiles_list: Sequence[str]) -> Tensor:
        """Compute all property values for a batch of SMILES strings.

        Returns:
            (batch_size, condition_dim) tensor.
        """
        rows = [self.evaluate_multi(s) for s in smiles_list]
        return torch.stack(rows)

    @property
    def name(self) -> str:
        return "+".join(c.name for c in self._conditions)

    @property
    def description(self) -> str:
        parts = [f"{c.name}: {c.description}" for c in self._conditions]
        return "; ".join(parts)

    def value_range(self) -> Tuple[float, float]:
        # Return range of first condition for single, otherwise not meaningful
        if len(self._conditions) == 1:
            return self._conditions[0].value_range()
        raise TypeError(
            "value_range() is ambiguous for MultiCondition. "
            "Access individual conditions via .conditions property."
        )

    def value_ranges(self) -> List[Tuple[float, float]]:
        """Return value ranges for all sub-conditions."""
        return [c.value_range() for c in self._conditions]

    @property
    def condition_dim(self) -> int:
        return sum(c.condition_dim for c in self._conditions)

    @property
    def conditions(self) -> List[ConditioningInterface]:
        return self._conditions

    def sample_condition(
        self,
        values: Union[Dict[str, float], List[float], float],
    ) -> Tensor:
        """Create a condition tensor for target property values.

        Args:
            values: Target values as either:
                - dict mapping condition name to value (preferred)
                - list of values in registration order
                - single float (only valid for single-condition MultiCondition)

        Returns:
            (condition_dim,) tensor.
        """
        if isinstance(values, (int, float)):
            if len(self._conditions) != 1:
                raise ValueError(
                    f"Scalar value only valid for single condition, "
                    f"got {len(self._conditions)} conditions."
                )
            return self._conditions[0].sample_condition(float(values))

        if isinstance(values, dict):
            parts = []
            for c in self._conditions:
                if c.name not in values:
                    raise KeyError(
                        f"Missing condition '{c.name}'. "
                        f"Expected keys: {[c.name for c in self._conditions]}"
                    )
                parts.append(c.sample_condition(values[c.name]))
            return torch.cat(parts)

        if isinstance(values, (list, tuple)):
            if len(values) != len(self._conditions):
                raise ValueError(
                    f"Expected {len(self._conditions)} values, got {len(values)}."
                )
            parts = [
                c.sample_condition(v)
                for c, v in zip(self._conditions, values)
            ]
            return torch.cat(parts)

        raise TypeError(f"Unsupported values type: {type(values)}")


# =============================================================================
# Default Conditioning Implementations
# =============================================================================


class HeavyAtomCount(ConditioningInterface):
    """Number of heavy (non-hydrogen) atoms in the molecule."""

    def evaluate(self, smiles: str) -> float:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0
        return float(mol.GetNumHeavyAtoms())

    @property
    def name(self) -> str:
        return "heavy_atoms"

    @property
    def description(self) -> str:
        return "Number of heavy (non-hydrogen) atoms"

    def value_range(self) -> Tuple[float, float]:
        return (1.0, 50.0)


class CrippenLogP(ConditioningInterface):
    """Crippen LogP (partition coefficient) computed by RDKit."""

    def evaluate(self, smiles: str) -> float:
        from rdkit import Chem
        from rdkit.Chem import Crippen

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0
        return float(Crippen.MolLogP(mol))

    @property
    def name(self) -> str:
        return "logp"

    @property
    def description(self) -> str:
        return "Crippen LogP (octanol-water partition coefficient)"

    def value_range(self) -> Tuple[float, float]:
        return (-5.0, 10.0)


# =============================================================================
# Condition Registry
# =============================================================================

CONDITION_REGISTRY: Dict[str, type] = {}


def register_condition(cls: type) -> type:
    """Register a ConditioningInterface class by its name."""
    instance = cls()
    CONDITION_REGISTRY[instance.name] = cls
    return cls


def get_condition(name: str) -> ConditioningInterface:
    """Look up a conditioning interface by name from the registry."""
    if name not in CONDITION_REGISTRY:
        available = list(CONDITION_REGISTRY.keys())
        raise KeyError(
            f"Unknown condition '{name}'. Available: {available}"
        )
    return CONDITION_REGISTRY[name]()


def build_condition(names: List[str]) -> MultiCondition:
    """Build a MultiCondition from a list of registered condition names.

    Args:
        names: List of condition names from the registry.

    Returns:
        MultiCondition wrapping the requested conditions.
    """
    conditions = [get_condition(n) for n in names]
    return MultiCondition(conditions)


# Register default conditions
register_condition(HeavyAtomCount)
register_condition(CrippenLogP)


# =============================================================================
# Flow Matching Model (Lightning Module)
# =============================================================================


class FlowMatchingModel(pl.LightningModule):
    """
    Flow Matching with optional minibatch OT coupling.

    Learns a continuous normalizing flow from standard Gaussian to a target
    data distribution using conditional flow matching. The model is agnostic
    to the internal structure of input vectors.

    Args:
        data_dim: Dimension of data vectors.
        hidden_dim: Hidden dimension for the velocity MLP.
        num_blocks: Number of residual FiLM blocks.
        time_embed_dim: Dimension of sinusoidal time embedding.
        condition_dim: Raw condition input dimension (0 = unconditional).
        condition_embed_dim: Dimension to project raw conditions into via MLP.
            Only used when condition_dim > 0.
        dropout: Dropout rate in residual blocks.
        use_ot_coupling: Whether to use minibatch OT coupling.
        solver_method: ODE solver method for sampling (euler, midpoint, dopri5).
        default_sample_steps: Default number of ODE steps for sampling.
        lr: Learning rate.
        weight_decay: Weight decay for AdamW.
        warmup_epochs: Number of linear warmup epochs.
    """

    def __init__(
        self,
        data_dim: int = 512,
        hidden_dim: int = 768,
        num_blocks: int = 6,
        time_embed_dim: int = 128,
        condition_dim: int = 0,
        condition_embed_dim: int = 64,
        dropout: float = 0.0,
        use_ot_coupling: bool = True,
        solver_method: str = "midpoint",
        default_sample_steps: int = 100,
        lr: float = 2e-4,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dim = data_dim
        self.condition_dim = condition_dim
        self.use_ot_coupling = use_ot_coupling
        self.solver_method = solver_method
        self.default_sample_steps = default_sample_steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs

        # Condition projection: raw scalars -> learned embedding
        if condition_dim > 0:
            self.condition_proj = nn.Sequential(
                nn.Linear(condition_dim, condition_embed_dim),
                nn.SiLU(),
                nn.Linear(condition_embed_dim, condition_embed_dim),
            )
            velocity_condition_dim = condition_embed_dim
        else:
            self.condition_proj = None
            velocity_condition_dim = 0

        # Velocity network
        self.velocity_net = VelocityMLP(
            data_dim=data_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            time_embed_dim=time_embed_dim,
            condition_dim=velocity_condition_dim,
            dropout=dropout,
        )

        # Wrapper for the flow_matching library's ODESolver
        self.wrapper = VelocityModelWrapper(self.velocity_net)

        # Probability path (CondOT: x_t = t*x_1 + (1-t)*x_0)
        self.path = CondOTProbPath()

        # OT coupler
        self.ot_coupler = MinibatchOTCoupler()

        # Standardization buffers.
        # Initialized to identity transform (mean=0, std=1) so standardize /
        # destandardize are no-ops until set_standardization() is called.
        # No flag needed — the buffers are part of the state dict and survive
        # checkpoint save/load automatically.
        self.register_buffer("data_mean", torch.zeros(data_dim))
        self.register_buffer("data_std", torch.ones(data_dim))


    # -------------------------------------------------------------------------
    # Standardization
    # -------------------------------------------------------------------------

    def set_standardization(self, mean: Tensor, std: Tensor) -> None:
        self.data_mean.copy_(mean)
        self.data_std.copy_(std.clamp(min=1e-8))

    def standardize(self, x: Tensor) -> Tensor:
        return (x - self.data_mean) / self.data_std

    def destandardize(self, z: Tensor) -> Tensor:
        return z * self.data_std + self.data_mean

    # -------------------------------------------------------------------------
    # Condition projection
    # -------------------------------------------------------------------------

    def _project_condition(self, condition: Optional[Tensor]) -> Optional[Tensor]:
        """Project raw condition values through the embedding MLP."""
        if condition is not None and self.condition_proj is not None:
            return self.condition_proj(condition)
        return condition

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    def compute_loss(self, x1: Tensor, condition: Optional[Tensor] = None) -> Tensor:
        """
        Compute flow matching loss with optional OT coupling.

        Args:
            x1: (batch_size, data_dim) data samples.
            condition: (batch_size, condition_dim) optional raw conditioning.

        Returns:
            Scalar MSE loss.
        """
        x1 = self.standardize(x1)
        condition = self._project_condition(condition)
        bs = x1.shape[0]

        # Sample noise
        x0 = torch.randn_like(x1)

        # Minibatch OT coupling
        if self.use_ot_coupling:
            x0_coupled = self.ot_coupler.couple(x0, x1)

            # Store OT diagnostic: ratio of transport cost after/before coupling.
            # Values < 1 confirm OT is reducing path crossings.
            if self.training:
                with torch.no_grad():
                    random_cost = (x0 - x1).pow(2).sum(-1).mean().item()
                    ot_cost = (x0_coupled - x1).pow(2).sum(-1).mean().item()
                    self._ot_cost_ratio = ot_cost / max(random_cost, 1e-8)

            x0 = x0_coupled

        # Sample time
        t = torch.rand(bs, device=x1.device)

        # Get path sample (x_t and target velocity dx_t)
        path_sample = self.path.sample(x_0=x0, x_1=x1, t=t)

        # Predict velocity
        v_pred = self.velocity_net(path_sample.x_t, t, condition)

        # MSE loss against target velocity
        return F.mse_loss(v_pred, path_sample.dx_t)

    def training_step(self, batch, batch_idx) -> Tensor:
        x1 = self._extract_vectors(batch)
        condition = self._extract_condition(batch)
        loss = self.compute_loss(x1, condition)
        self.log(
            "train/loss", loss.detach(), prog_bar=True,
            batch_size=x1.shape[0], on_step=True, on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx) -> Tensor:
        x1 = self._extract_vectors(batch)
        condition = self._extract_condition(batch)
        loss = self.compute_loss(x1, condition)
        self.log(
            "val/loss", loss.detach(), prog_bar=True,
            batch_size=x1.shape[0], on_epoch=True,
        )
        return loss

    # -------------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        condition: Optional[Tensor] = None,
        num_steps: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """
        Generate samples by integrating ODE from noise to data.

        Args:
            num_samples: Number of samples to generate.
            condition: (num_samples, condition_dim) optional conditioning.
            num_steps: ODE integration steps (default from constructor).
            device: Device for generation.

        Returns:
            (num_samples, data_dim) generated vectors in original space.
        """
        device = device or next(self.parameters()).device
        num_steps = num_steps or self.default_sample_steps
        step_size = 1.0 / num_steps

        x0 = torch.randn(num_samples, self.data_dim, device=device)

        if condition is None and self.condition_dim > 0:
            condition = torch.zeros(num_samples, self.condition_dim, device=device)
        condition = self._project_condition(condition)

        solver = ODESolver(velocity_model=self.wrapper)
        x1 = solver.sample(
            x_init=x0,
            step_size=step_size,
            method=self.solver_method,
            time_grid=torch.tensor([0.0, 1.0], device=device),
            return_intermediates=False,
            condition=condition,
        )

        return self.destandardize(x1)

    def encode(
        self,
        x1: Tensor,
        condition: Optional[Tensor] = None,
        num_steps: Optional[int] = None,
    ) -> Tensor:
        """
        Encode data to noise space by integrating reverse ODE.

        Useful for reflow data generation and latent space analysis.

        Args:
            x1: (batch_size, data_dim) data in original space.
            condition: (batch_size, condition_dim) optional.
            num_steps: ODE steps.

        Returns:
            (batch_size, data_dim) corresponding noise vectors.
        """
        num_steps = num_steps or self.default_sample_steps
        step_size = 1.0 / num_steps
        device = x1.device

        x1_std = self.standardize(x1)

        if condition is None and self.condition_dim > 0:
            condition = torch.zeros(x1.shape[0], self.condition_dim, device=device)
        condition = self._project_condition(condition)

        solver = ODESolver(velocity_model=self.wrapper)
        x0 = solver.sample(
            x_init=x1_std,
            step_size=step_size,
            method=self.solver_method,
            time_grid=torch.tensor([1.0, 0.0], device=device),
            return_intermediates=False,
            condition=condition,
        )

        return x0

    # -------------------------------------------------------------------------
    # Likelihood
    # -------------------------------------------------------------------------

    def compute_nll(
        self,
        x1: Tensor,
        condition: Optional[Tensor] = None,
        num_steps: int = 200,
        exact_divergence: bool = False,
    ) -> Tensor:
        """
        Compute negative log-likelihood via the continuous change-of-variables formula.

        Args:
            x1: (batch_size, data_dim) data samples.
            condition: optional conditioning.
            num_steps: Integration steps.
            exact_divergence: If True, compute exact divergence (slow).

        Returns:
            (batch_size,) negative log-likelihood per sample.
        """
        device = x1.device
        x1_std = self.standardize(x1)
        step_size = 1.0 / num_steps

        if condition is None and self.condition_dim > 0:
            condition = torch.zeros(x1.shape[0], self.condition_dim, device=device)
        condition = self._project_condition(condition)

        def log_p0(x: Tensor) -> Tensor:
            return -0.5 * (x.shape[-1] * math.log(2 * math.pi) + (x**2).sum(dim=-1))

        solver = ODESolver(velocity_model=self.wrapper)
        _, log_likelihood = solver.compute_likelihood(
            x_1=x1_std,
            log_p0=log_p0,
            step_size=step_size,
            method="euler",
            time_grid=torch.tensor([1.0, 0.0], device=device),
            exact_divergence=exact_divergence,
            condition=condition,
        )

        nll = -log_likelihood

        # Correct for standardization Jacobian (zero when std=1, i.e. not fitted)
        nll -= self.data_std.log().sum()

        return nll

    # -------------------------------------------------------------------------
    # Data extraction helpers
    # -------------------------------------------------------------------------

    def _extract_vectors(self, batch) -> Tensor:
        """Extract flat HDC vectors from a PyG batch.

        Uses ``[node_terms | graph_terms]`` so the first half contains
        the bundled raw node hypervectors (order-0), which can be
        iteratively decoded back into node identities.
        """
        bs = batch.num_graphs
        node_terms = batch.node_terms.view(bs, -1)
        graph_terms = batch.graph_terms.view(bs, -1)
        # .as_subclass(Tensor) strips HRRTensor (or other subclasses) back to
        # plain Tensor so downstream ops and Lightning deepcopy work correctly.
        return (
            torch.cat([node_terms, graph_terms], dim=-1)
            .float()
            .as_subclass(torch.Tensor)
        )

    def _extract_condition(self, batch) -> Optional[Tensor]:
        """Extract condition from batch. Auto-detects batch.condition attribute.

        Returns None if condition_dim is 0 or no condition attribute is present.
        """
        if self.condition_dim <= 0:
            return None

        if hasattr(batch, "condition") and batch.condition is not None:
            bs = batch.num_graphs
            cond = batch.condition
            if cond.dim() == 1:
                cond = cond.view(bs, -1)
            return cond

        # Fallback: zeros (unconditional behavior with conditional architecture)
        bs = batch.num_graphs
        return torch.zeros(bs, self.condition_dim, device=batch.node_terms.device)

    # -------------------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        if self.warmup_epochs > 0 and self.trainer is not None:
            total_steps = self.trainer.estimated_stepping_batches
            steps_per_epoch = max(1, total_steps // max(1, self.trainer.max_epochs))
            warmup_steps = self.warmup_epochs * steps_per_epoch

            def lr_lambda(step):
                if step < warmup_steps:
                    return step / max(1, warmup_steps)
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

        return optimizer
