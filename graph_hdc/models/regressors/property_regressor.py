"""
Property Regressor for Molecular Properties.

Predicts molecular properties (LogP, QED, SA Score) from HDC graph embeddings.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchmetrics import R2Score


class MolecularProperty(str, Enum):
    """Supported molecular properties for regression."""

    LOGP = "logp"
    QED = "qed"

    @classmethod
    def from_string(cls, s: str) -> "MolecularProperty":
        """Convert string to enum, case-insensitive."""
        s_lower = s.lower()
        for prop in cls:
            if prop.value == s_lower:
                return prop
        raise ValueError(f"Unknown property: {s}. Supported: {[p.value for p in cls]}")


# Activation functions
ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.1),
    "tanh": nn.Tanh,
}

# Normalization layers
NORMALIZATIONS = {
    "lay_norm": nn.LayerNorm,
    "batch_norm": nn.BatchNorm1d,
    "none": None,
}


@dataclass
class RegressorConfig:
    """Configuration for property regressor."""

    input_dim: int = 256
    hidden_dims: tuple[int, ...] = (256, 256)
    activation: str = "gelu"
    norm: str = "none"
    dropout: float = 0.0
    lr: float = 1e-3
    weight_decay: float = 1e-4
    target_property: str = "logp"
    # Training parameters (not used by model, but stored for convenience)
    batch_size: int = 128
    epochs: int = 200


# Best configs for QM9 and ZINC
QM9_LOGP_CONFIG = RegressorConfig(
    input_dim=256,
    hidden_dims=(256, 256),
    activation="gelu",
    norm="none",
    dropout=0.061,
    lr=8.63e-4,
    weight_decay=9.45e-5,
    target_property="logp",
    batch_size=192,
    epochs=200,
)

QM9_QED_CONFIG = RegressorConfig(
    input_dim=256,
    hidden_dims=(512, 256, 128, 32),
    activation="silu",
    norm="lay_norm",
    dropout=0.049,
    lr=6.35e-4,
    weight_decay=2.23e-4,
    target_property="qed",
    batch_size=64,
    epochs=200,
)

ZINC_LOGP_CONFIG = RegressorConfig(
    input_dim=256,
    hidden_dims=(1536, 896, 128),
    activation="silu",
    norm="none",
    dropout=0.127,
    lr=8.64e-5,
    weight_decay=2.80e-5,
    target_property="logp",
    batch_size=352,
    epochs=200,
)

ZINC_QED_CONFIG = RegressorConfig(
    input_dim=256,
    hidden_dims=(1536, 768, 128),
    activation="silu",
    norm="lay_norm",
    dropout=0.178,
    lr=5.36e-4,
    weight_decay=1.36e-6,
    target_property="qed",
    batch_size=224,
    epochs=200,
)


def _instantiate(factory):
    """Instantiate activation/norm factory."""
    return factory() if callable(factory) else factory


def _make_mlp(
    in_dim: int,
    hidden_dims: Iterable[int],
    out_dim: int = 1,
    *,
    activation: str = "gelu",
    dropout: float = 0.0,
    norm: str = "lay_norm",
) -> nn.Sequential:
    """Build MLP with configurable architecture."""
    layers: list[nn.Module] = []
    act_factory = ACTIVATIONS.get(activation, nn.GELU)
    norm_factory = NORMALIZATIONS.get(norm)

    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        if norm_factory is not None:
            layers.append(norm_factory(h))
        layers.append(_instantiate(act_factory))
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h

    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class PropertyRegressor(pl.LightningModule):
    """
    Property regressor for molecular properties.

    Predicts scalar molecular properties from HDC graph embeddings.

    Parameters
    ----------
    input_dim : int
        Input dimension (typically graph_terms dimension = hv_dim)
    hidden_dims : tuple[int, ...]
        Hidden layer dimensions
    activation : str
        Activation function name
    dropout : float
        Dropout probability
    norm : str
        Normalization layer type
    lr : float
        Learning rate
    weight_decay : float
        Weight decay
    target_property : str
        Target property to predict
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: Iterable[int] = (256, 256),
        *,
        activation: str = "gelu",
        dropout: float = 0.0,
        norm: str = "none",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        target_property: str = "logp",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.target_property = MolecularProperty.from_string(target_property)

        self.net = _make_mlp(
            in_dim=input_dim,
            hidden_dims=hidden_dims,
            out_dim=1,
            activation=activation,
            dropout=dropout,
            norm=norm,
        )

        self.loss_fn = nn.MSELoss()
        self.train_r2 = R2Score()
        self.val_r2 = R2Score()
        self.test_r2 = R2Score()

    def on_train_epoch_start(self):
        self.train_r2.reset()

    def on_validation_epoch_start(self):
        self.val_r2.reset()

    def on_test_epoch_start(self):
        self.test_r2.reset()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass on graph embeddings.

        Parameters
        ----------
        x : Tensor
            Graph embeddings of shape [batch_size, input_dim]

        Returns
        -------
        Tensor
            Predicted property values of shape [batch_size]
        """
        return self.net(x).squeeze(-1)

    def forward_batch(self, batch) -> Tensor:
        """Forward pass on PyG batch with graph_terms attribute."""
        B = batch.num_graphs
        D = self.hparams.input_dim
        g = batch.graph_terms.view(B, D)

        # Convert from HRRTensor to regular Tensor (fixes Lightning callback compatibility)
        if hasattr(g, "as_subclass"):
            g = g.as_subclass(torch.Tensor)

        # Cast to module dtype
        if g.dtype.is_floating_point:
            g = g.to(self.dtype)

        return self.forward(g)

    def _get_target(self, batch) -> Tensor:
        """Extract target property from batch."""
        if self.target_property == MolecularProperty.LOGP:
            target = batch.logp
        elif self.target_property == MolecularProperty.QED:
            target = batch.qed
        else:
            raise ValueError(f"Unsupported property: {self.target_property}")

        # Convert from HRRTensor to regular Tensor if needed
        if hasattr(target, "as_subclass"):
            target = target.as_subclass(torch.Tensor)

        return target.to(self.dtype).view(-1)

    def _step(self, batch, stage: str):
        """Generic training/validation/test step."""
        y = self._get_target(batch)
        y_hat = self.forward_batch(batch)
        loss = self.loss_fn(y_hat, y)

        with torch.no_grad():
            mae = F.l1_loss(y_hat, y)
            rmse = torch.sqrt(F.mse_loss(y_hat, y))

        B = batch.num_graphs
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=B)
        self.log(f"{stage}_mae", mae, prog_bar=(stage != "train"), on_step=False, on_epoch=True, batch_size=B)
        self.log(f"{stage}_rmse", rmse, prog_bar=False, on_step=False, on_epoch=True, batch_size=B)

        # R2 metrics
        if stage == "train":
            self.train_r2.update(y_hat.detach(), y.detach())
            self.log("train_r2", self.train_r2, on_step=False, on_epoch=True, prog_bar=False)
        elif stage == "val":
            self.val_r2.update(y_hat.detach(), y.detach())
            self.log("val_r2", self.val_r2, on_step=False, on_epoch=True, prog_bar=True)
        elif stage == "test":
            self.test_r2.update(y_hat.detach(), y.detach())
            self.log("test_r2", self.test_r2, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            foreach=True,
        )

    def on_after_batch_transfer(self, batch, _: int):
        """Cast batch to model dtype for mixed precision training."""

        def cast(x):
            return x.to(self.dtype) if torch.is_tensor(x) and x.dtype.is_floating_point else x

        if isinstance(batch, dict):
            return {k: cast(v) for k, v in batch.items()}
        if isinstance(batch, (list, tuple)):
            return type(batch)(cast(v) for v in batch)
        return cast(batch)

    @classmethod
    def load_from_checkpoint_file(cls, path: str | Path) -> "PropertyRegressor":
        """
        Load model from checkpoint file.

        Handles module path remapping for checkpoints saved with different
        package structures (e.g., 'src' vs 'graph_hdc').
        """
        from graph_hdc.utils.compat import setup_module_aliases
        setup_module_aliases()

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        hparams = checkpoint["hyper_parameters"]
        model = cls(**hparams)
        model.load_state_dict(checkpoint["state_dict"])
        return model
