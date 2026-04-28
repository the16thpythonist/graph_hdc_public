"""
TranslatorMLP: Deep SwiGLU residual network for vector-to-vector translation.

Used to map fingerprint representations (e.g. ECFP4) to HDC hypervectors.
Architecture mirrors the autoencoder's ProjectionBlock stack: each hidden
stage is a Linear projection followed by ``resblock_depth`` SwiGLU residual
blocks. The final layer is a plain Linear to the target dimension (so
Procrustes least-squares initialization applies cleanly).
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

from graph_hdc.models.autoencoder import BerHuReconstructionLoss, ProjectionBlock


class TranslatorMLP(pl.LightningModule):
    """
    Deep SwiGLU residual network mapping a source vector to a target vector.

    Architecture
    ------------
    For each h in hidden_dims:
        ProjectionBlock(prev, h, resblock_depth)
          = Linear(prev, h) → [SwiGLUResidualBlock(h)] * resblock_depth
    Final: Linear(hidden_dims[-1], output_dim)

    Loss: BerHu (reverse Huber) only — scale-robust L1/L2 hybrid.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Iterable[int] = (4096, 4096, 4096, 4096),
        *,
        resblock_depth: int = 2,
        berhu_c_fraction: float = 0.2,
        lr: float = 2e-4,
        weight_decay: float = 0.0,
        warmup_epochs: int = 10,
        total_epochs: int = 500,
    ):
        super().__init__()
        self.save_hyperparameters()
        hidden_dims = list(hidden_dims)
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must be non-empty")

        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(ProjectionBlock(prev, h, num_blocks=resblock_depth))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

        self.berhu = BerHuReconstructionLoss(c_fraction=berhu_c_fraction)

        # Per-sample validation stats for the tracking callback. Populated by
        # validation_step, cleared at the start of each validation epoch.
        self._val_per_sample_loss: list[Tensor] = []
        self._val_per_sample_cosine: list[Tensor] = []
        self._val_per_sample_l2: list[Tensor] = []

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    @torch.no_grad()
    def initialize_procrustes(self, X: Tensor, Y: Tensor) -> float:
        """
        Initialize the output projection via least-squares fit.

        Runs X through all layers except the last Linear to obtain hidden
        features, then solves  [H | 1] @ [W^T; b^T] = Y  for the output layer.
        """
        self.eval()
        device = next(self.parameters()).device
        X = X.to(device)
        Y = Y.to(device)

        backbone = self.net[:-1]
        H = backbone(X)

        ones = torch.ones(H.size(0), 1, device=device, dtype=H.dtype)
        H_aug = torch.cat([H, ones], dim=1)

        result = torch.linalg.lstsq(H_aug, Y)
        solution = result.solution

        W = solution[:-1, :]
        b = solution[-1, :]

        output_layer = self.net[-1]
        output_layer.weight.copy_(W.T)
        output_layer.bias.copy_(b)

        Y_pred = H_aug @ solution
        residual_mse = F.mse_loss(Y_pred, Y).item()

        self.train()
        return residual_mse

    def _per_sample_stats(self, pred: Tensor, target: Tensor) -> Dict[str, Tensor]:
        """Compute per-sample loss, cosine similarity, and L2 distance."""
        berhu_ps = self.berhu(pred, target)                                  # [B]
        cos_ps = F.cosine_similarity(pred, target, dim=-1)                   # [B]
        l2_ps = (pred - target).norm(p=2, dim=-1)                            # [B]
        return {"loss": berhu_ps, "cosine": cos_ps, "l2": l2_ps}

    def training_step(self, batch, batch_idx):
        fp, hdc = batch
        pred = self.forward(fp)
        stats = self._per_sample_stats(pred, hdc)
        loss = stats["loss"].mean()
        bs = fp.size(0)

        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
        self.log("train/cosine_sim", stats["cosine"].mean(), on_step=False, on_epoch=True, batch_size=bs)
        self.log("train/l2", stats["l2"].mean(), on_step=False, on_epoch=True, batch_size=bs)
        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_per_sample_loss = []
        self._val_per_sample_cosine = []
        self._val_per_sample_l2 = []

    def validation_step(self, batch, batch_idx):
        fp, hdc = batch
        pred = self.forward(fp)
        stats = self._per_sample_stats(pred, hdc)
        loss = stats["loss"].mean()
        bs = fp.size(0)

        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
        self.log("val/cosine_sim", stats["cosine"].mean(), on_step=False, on_epoch=True, batch_size=bs)
        self.log("val/l2", stats["l2"].mean(), on_step=False, on_epoch=True, batch_size=bs)

        self._val_per_sample_loss.append(stats["loss"].detach().cpu())
        self._val_per_sample_cosine.append(stats["cosine"].detach().cpu())
        self._val_per_sample_l2.append(stats["l2"].detach().cpu())

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
                return (epoch + 1) / warmup
            progress = (epoch - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
