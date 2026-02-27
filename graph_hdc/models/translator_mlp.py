"""
TranslatorMLP: Deep residual MLP for vector-to-vector translation.

Used to map fingerprint representations (e.g. ECFP4) to HDC hypervectors.
Shared between training and evaluation experiments.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# =============================================================================
# Building blocks
# =============================================================================

ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.1),
    "tanh": nn.Tanh,
}

NORMALIZATIONS = {
    "lay_norm": nn.LayerNorm,
    "batch_norm": nn.BatchNorm1d,
    "none": None,
}


def _make_act(name: str) -> nn.Module:
    factory = ACTIVATIONS.get(name, nn.GELU)
    return factory() if callable(factory) else factory


class ResidualBlock(nn.Module):
    """Linear → Norm → Activation → Dropout with additive skip connection."""

    def __init__(
        self,
        dim: int,
        activation: str = "gelu",
        norm: str = "lay_norm",
        dropout: float = 0.0,
    ):
        super().__init__()
        norm_factory = NORMALIZATIONS.get(norm)
        layers: list[nn.Module] = [nn.Linear(dim, dim)]
        if norm_factory is not None:
            layers.append(norm_factory(dim))
        layers.append(_make_act(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


# =============================================================================
# TranslatorMLP
# =============================================================================


class TranslatorMLP(pl.LightningModule):
    """
    Deep residual MLP that translates a source vector to a target vector.

    Architecture
    ------------
    input_projection: Linear(input_dim → hidden_dims[0])
    For each consecutive pair (h_prev, h_cur) in hidden_dims:
        - If h_cur == h_prev → ResidualBlock(h_cur)
        - Else → Linear(h_prev, h_cur) + Norm + Act + Dropout
    output_projection: Linear(hidden_dims[-1] → output_dim)

    Loss: weighted combination of cosine similarity loss and MSE loss.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Iterable[int] = (1024, 1024, 512, 512),
        *,
        activation: str = "gelu",
        norm: str = "lay_norm",
        dropout: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        cosine_loss_weight: float = 1.0,
        mse_loss_weight: float = 0.1,
        infonce_loss_weight: float = 0.0,
        infonce_temperature: float = 0.07,
        warmup_epochs: int = 10,
        total_epochs: int = 500,
    ):
        super().__init__()
        self.save_hyperparameters()
        hidden_dims = list(hidden_dims)

        # ── build layers ──
        layers: list[nn.Module] = []
        norm_factory = NORMALIZATIONS.get(norm)

        # input projection
        prev = input_dim
        for h in hidden_dims:
            if h == prev:
                layers.append(ResidualBlock(h, activation=activation, norm=norm, dropout=dropout))
            else:
                block: list[nn.Module] = [nn.Linear(prev, h)]
                if norm_factory is not None:
                    block.append(norm_factory(h))
                block.append(_make_act(activation))
                if dropout > 0:
                    block.append(nn.Dropout(dropout))
                layers.append(nn.Sequential(*block))
            prev = h

        # output projection (no activation)
        layers.append(nn.Linear(prev, output_dim))

        self.net = nn.Sequential(*layers)

    # ── forward ──

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    # ── initialization ──

    @torch.no_grad()
    def initialize_procrustes(self, X: Tensor, Y: Tensor) -> float:
        """
        Initialize the output projection via least-squares fit.

        Runs X through all layers except the last to obtain hidden features,
        then solves the least-squares problem  H @ W^T + b = Y  for the
        output Linear layer.

        Parameters
        ----------
        X : Tensor (N, input_dim)
            Representative input samples (e.g. fingerprints).
        Y : Tensor (N, output_dim)
            Corresponding target vectors (e.g. HDC vectors).

        Returns
        -------
        float
            Residual MSE after the linear fit (for logging).
        """
        self.eval()
        device = next(self.parameters()).device
        X = X.to(device)
        Y = Y.to(device)

        # Forward through all layers except the last (output projection)
        backbone = self.net[:-1]
        H = backbone(X)  # (N, hidden_dims[-1])

        # Solve least-squares: [H | 1] @ [W^T; b^T] = Y
        ones = torch.ones(H.size(0), 1, device=device, dtype=H.dtype)
        H_aug = torch.cat([H, ones], dim=1)  # (N, hidden+1)

        # lstsq returns (solution, residuals, rank, singular_values)
        result = torch.linalg.lstsq(H_aug, Y)
        solution = result.solution  # (hidden+1, output_dim)

        W = solution[:-1, :]  # (hidden, output_dim)
        b = solution[-1, :]   # (output_dim,)

        # Copy into the output Linear layer
        output_layer = self.net[-1]
        output_layer.weight.copy_(W.T)
        output_layer.bias.copy_(b)

        # Compute residual MSE
        Y_pred = H_aug @ solution
        residual_mse = F.mse_loss(Y_pred, Y).item()

        self.train()
        return residual_mse

    # ── loss ──

    def _info_nce_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Symmetric InfoNCE (NT-Xent) contrastive loss.

        Within a batch of N (pred, target) pairs, each predicted vector should
        be most similar to its own target.  Computed symmetrically (pred→target
        and target→pred) and averaged, like CLIP.

        This encourages the model to preserve neighborhood structure: similar
        fingerprints should map to similar HDC vectors, and dissimilar ones
        should stay apart.
        """
        pred_norm = F.normalize(pred, dim=-1)
        target_norm = F.normalize(target, dim=-1)

        # Cosine similarity matrix scaled by temperature: (N, N)
        logits = pred_norm @ target_norm.T / self.hparams.infonce_temperature

        # Positive pairs are on the diagonal
        labels = torch.arange(logits.size(0), device=logits.device)

        loss_p2t = F.cross_entropy(logits, labels)
        loss_t2p = F.cross_entropy(logits.T, labels)
        return (loss_p2t + loss_t2p) / 2

    def _compute_loss(self, pred: Tensor, target: Tensor) -> Dict[str, Tensor]:
        cosine_loss = (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()
        mse_loss = F.mse_loss(pred, target)
        combined = (
            self.hparams.cosine_loss_weight * cosine_loss
            + self.hparams.mse_loss_weight * mse_loss
        )

        result = {"cosine_loss": cosine_loss, "mse_loss": mse_loss}

        if self.hparams.infonce_loss_weight > 0 and pred.size(0) > 1:
            infonce = self._info_nce_loss(pred, target)
            combined = combined + self.hparams.infonce_loss_weight * infonce
            result["infonce_loss"] = infonce

        result["loss"] = combined
        return result

    # ── training / validation ──

    def training_step(self, batch, batch_idx):
        fp, hdc = batch
        pred = self.forward(fp)
        losses = self._compute_loss(pred, hdc)
        bs = fp.size(0)
        self.log("train/loss", losses["loss"], prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
        self.log("train/cosine", losses["cosine_loss"], on_step=False, on_epoch=True, batch_size=bs)
        self.log("train/mse", losses["mse_loss"], on_step=False, on_epoch=True, batch_size=bs)
        if "infonce_loss" in losses:
            self.log("train/infonce", losses["infonce_loss"], on_step=False, on_epoch=True, batch_size=bs)
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        fp, hdc = batch
        pred = self.forward(fp)
        losses = self._compute_loss(pred, hdc)
        bs = fp.size(0)
        self.log("val/loss", losses["loss"], prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
        self.log("val/cosine", losses["cosine_loss"], on_step=False, on_epoch=True, batch_size=bs)
        self.log("val/mse", losses["mse_loss"], on_step=False, on_epoch=True, batch_size=bs)
        if "infonce_loss" in losses:
            self.log("val/infonce", losses["infonce_loss"], on_step=False, on_epoch=True, batch_size=bs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        warmup = self.hparams.warmup_epochs
        total = self.hparams.total_epochs

        if warmup <= 0 and total <= 0:
            return optimizer

        def lr_lambda(epoch: int) -> float:
            if warmup > 0 and epoch < warmup:
                # Linear warmup: 0 → 1 over warmup_epochs
                return (epoch + 1) / warmup
            # Cosine annealing: 1 → 0 over remaining epochs
            progress = (epoch - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
