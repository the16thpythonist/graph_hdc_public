"""
(Variational) Autoencoder for HDC vector spaces.

Learns compressed representations of concatenated [edge_terms | graph_terms]
HDC graph encodings. Supports both standard AE and VAE modes with
configurable encoder/decoder architectures.

Loss:
    AE:  mse_weight * MSE + cosine_weight * (1 - cos_sim)
    VAE: mse_weight * MSE + cosine_weight * (1 - cos_sim) + beta * KL
         where beta linearly anneals from 0 to kl_weight over kl_warmup_epochs
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SwiGLUResidualBlock(nn.Module):
    """Pre-norm SwiGLU residual block: ``x + SiLU(W_g LN(x)) * (W_v LN(x))``."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.w_gate = nn.Linear(dim, dim)
        self.w_value = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        return x + F.silu(self.w_gate(h)) * self.w_value(h)


class ProjectionBlock(nn.Module):
    """Linear projection followed by *K* SwiGLU residual blocks.

    Used for both encoder (down-projection) and decoder (up-projection)
    layers -- the class is agnostic to the direction.
    """

    def __init__(self, in_dim: int, out_dim: int, num_blocks: int = 2):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.blocks = nn.ModuleList(
            [SwiGLUResidualBlock(out_dim) for _ in range(num_blocks)]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        return x


class ReconstructionLoss(nn.Module):
    """Base class for per-sample reconstruction losses. Returns shape [B]."""

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError


class MSEReconstructionLoss(ReconstructionLoss):
    """Mean squared error per sample: mean((pred - target)^2, dim=-1)."""

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return (pred - target).pow(2).mean(dim=-1)


class MAEReconstructionLoss(ReconstructionLoss):
    """Mean absolute error per sample: mean(|pred - target|, dim=-1)."""

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return (pred - target).abs().mean(dim=-1)


class BerHuReconstructionLoss(ReconstructionLoss):
    """Reverse Huber (BerHu) loss: L1 for small errors, L2 for large.

    The threshold ``c`` is computed adaptively per batch as
    ``c = c_fraction * max(|residual|)`` (detached from the gradient).

    Per-element:
        |e|                     if |e| <= c
        (e^2 + c^2) / (2c)     if |e| > c

    Then averaged over feature dimensions to produce per-sample losses [B].
    """

    def __init__(self, c_fraction: float = 0.2):
        super().__init__()
        self.c_fraction = c_fraction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = pred - target
        abs_diff = diff.abs()
        c = self.c_fraction * abs_diff.max().detach()
        berhu = torch.where(
            abs_diff <= c,
            abs_diff,
            (diff.pow(2) + c ** 2) / (2 * c + 1e-8),
        )
        return berhu.mean(dim=-1)


class HDCAutoencoder(pl.LightningModule):
    """(Variational) Autoencoder for HDC molecular graph vectors.

    Args:
        data_dim: Input/output dimension (typically 2 * hv_dim).
        latent_dim: Bottleneck dimension.
        encoder_hidden_dims: Hidden layer sizes for the encoder.
        decoder_hidden_dims: Hidden layer sizes for the decoder.
            If None, uses reversed encoder_hidden_dims.
        resblock_depth: Number of SwiGLU residual blocks per projection layer.
        training_target: Which HDC components to reconstruct.
            ``"both"`` (default) uses ``[edge_terms | graph_terms]``,
            ``"edge"`` uses only edge_terms, ``"graph"`` uses only graph_terms.
        recon_loss_type: Reconstruction loss type. ``"mse"`` for mean squared
            error, ``"mae"`` for mean absolute error.
        variational: If True, uses VAE with reparameterization trick.
        mse_weight: Weight for the reconstruction loss component (MSE or MAE).
        cosine_weight: Weight for cosine similarity loss component.
        kl_weight: Maximum beta for KL divergence (VAE only).
        kl_warmup_epochs: Epochs to linearly ramp beta from 0 to kl_weight.
        loss_clamp: If set, clamp per-sample reconstruction loss to this
            maximum before averaging. Prevents outlier samples from causing
            gradient explosions. None means no clamping.
        lr: Learning rate for AdamW.
        weight_decay: Weight decay for AdamW.
        warmup_epochs: Number of linear LR warmup epochs.
    """

    def __init__(
        self,
        data_dim: int,
        latent_dim: int,
        encoder_hidden_dims: List[int],
        decoder_hidden_dims: Optional[List[int]] = None,
        resblock_depth: int = 2,
        training_target: str = "both",
        recon_loss_type: str = "mse",
        berhu_c_fraction: float = 0.2,
        variational: bool = False,
        mse_weight: float = 1.0,
        cosine_weight: float = 1.0,
        kl_weight: float = 1.0,
        kl_warmup_epochs: int = 20,
        loss_clamp: Optional[float] = None,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        warmup_epochs: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters()

        assert training_target in ("both", "edge", "graph"), (
            f"training_target must be 'both', 'edge', or 'graph', got {training_target!r}"
        )
        assert recon_loss_type in ("mse", "mae", "berhu"), (
            f"recon_loss_type must be 'mse', 'mae', or 'berhu', got {recon_loss_type!r}"
        )
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.training_target = training_target
        self.recon_loss_type = recon_loss_type

        if recon_loss_type == "mse":
            self.recon_loss_fn = MSEReconstructionLoss()
        elif recon_loss_type == "mae":
            self.recon_loss_fn = MAEReconstructionLoss()
        elif recon_loss_type == "berhu":
            self.recon_loss_fn = BerHuReconstructionLoss(c_fraction=berhu_c_fraction)
        self.variational = variational
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight
        self.kl_weight = kl_weight
        self.kl_warmup_epochs = kl_warmup_epochs
        self.loss_clamp = loss_clamp
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs

        # ----- Encoder -----
        enc_layers: list[nn.Module] = []
        in_dim = data_dim
        for h in encoder_hidden_dims:
            enc_layers.append(ProjectionBlock(in_dim, h, resblock_depth))
            in_dim = h
        self.encoder_backbone = nn.Sequential(*enc_layers)

        if variational:
            self.fc_mu = nn.Linear(in_dim, latent_dim)
            self.fc_logvar = nn.Linear(in_dim, latent_dim)
        else:
            self.fc_latent = nn.Linear(in_dim, latent_dim)

        # ----- Decoder -----
        dec_dims = (
            decoder_hidden_dims
            if decoder_hidden_dims is not None
            else list(reversed(encoder_hidden_dims))
        )
        dec_layers: list[nn.Module] = []
        in_dim = latent_dim
        for h in dec_dims:
            dec_layers.append(ProjectionBlock(in_dim, h, resblock_depth))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, data_dim))
        self.decoder = nn.Sequential(*dec_layers)

        # Validation latent stats accumulator
        self._val_z_norm_sum: float = 0.0
        self._val_z_sq_sum: Optional[Tensor] = None
        self._val_z_sum: Optional[Tensor] = None
        self._val_z_count: int = 0

    # -----------------------------------------------------------------
    # Core forward methods
    # -----------------------------------------------------------------

    def encode(self, x: Tensor) -> dict:
        """Encode input to latent space."""
        h = self.encoder_backbone(x)
        if self.variational:
            return {"mu": self.fc_mu(h), "logvar": self.fc_logvar(h)}
        return {"z": self.fc_latent(h)}

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """VAE reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent vector to reconstruction."""
        return self.decoder(z)

    def forward(self, x: Tensor) -> dict:
        """Full forward pass.

        Returns:
            Dictionary with keys: x_hat, z, and (for VAE) mu, logvar.
        """
        enc_out = self.encode(x)
        if self.variational:
            z = self.reparameterize(enc_out["mu"], enc_out["logvar"])
            x_hat = self.decode(z)
            return {
                "x_hat": x_hat,
                "z": z,
                "mu": enc_out["mu"],
                "logvar": enc_out["logvar"],
            }
        z = enc_out["z"]
        x_hat = self.decode(z)
        return {"x_hat": x_hat, "z": z}

    def reconstruct(self, x: Tensor) -> Tensor:
        """Deterministic reconstruction (uses mu for VAE, no sampling)."""
        enc_out = self.encode(x)
        z = enc_out["mu"] if self.variational else enc_out["z"]
        return self.decode(z)

    # -----------------------------------------------------------------
    # Loss computation
    # -----------------------------------------------------------------

    def _get_beta(self, epoch: int) -> float:
        """Linear KL annealing schedule."""
        if self.kl_warmup_epochs <= 0:
            return self.kl_weight
        return self.kl_weight * min(1.0, epoch / max(1, self.kl_warmup_epochs))

    def _compute_loss(self, x: Tensor, output: dict, prefix: str) -> Tensor:
        """Compute and log loss components.

        When ``self.loss_clamp`` is set, per-sample reconstruction losses are
        clamped before averaging so that outlier samples cannot cause gradient
        explosions.
        """
        x_hat = output["x_hat"]
        bs = x.shape[0]

        # Per-sample losses [B]
        per_sample_recon_base = self.recon_loss_fn(x_hat, x)
        per_sample_cos_sim = F.cosine_similarity(x_hat, x, dim=-1)
        per_sample_recon = (
            self.mse_weight * per_sample_recon_base
            + self.cosine_weight * (1.0 - per_sample_cos_sim)
        )

        # Clamp outliers before averaging
        if self.loss_clamp is not None:
            n_clamped = (per_sample_recon > self.loss_clamp).sum()
            per_sample_recon = per_sample_recon.clamp(max=self.loss_clamp)
            if n_clamped > 0:
                self.log(
                    f"{prefix}/n_clamped", float(n_clamped),
                    on_step=False, on_epoch=True, batch_size=bs,
                )

        recon_base = per_sample_recon_base.mean()
        cos_sim = per_sample_cos_sim.mean()
        recon_loss = per_sample_recon.mean()

        self.log(f"{prefix}/recon", recon_base, on_step=False, on_epoch=True, batch_size=bs)
        self.log(f"{prefix}/cos_sim", cos_sim, on_step=False, on_epoch=True, batch_size=bs)
        self.log(f"{prefix}/recon_loss", recon_loss, on_step=False, on_epoch=True, batch_size=bs)

        if self.variational:
            kl = -0.5 * torch.mean(
                torch.sum(
                    1 + output["logvar"] - output["mu"].pow(2) - output["logvar"].exp(),
                    dim=-1,
                )
            )
            beta = self._get_beta(self.current_epoch)
            loss = recon_loss + beta * kl
            self.log(f"{prefix}/kl", kl, on_step=False, on_epoch=True, batch_size=bs)
            self.log(f"{prefix}/beta", beta, on_step=False, on_epoch=True, batch_size=bs)
        else:
            loss = recon_loss

        self.log(f"{prefix}/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        return loss

    def per_sample_losses(self, x: Tensor) -> dict:
        """Compute per-sample reconstruction error, cosine similarity, and total loss.

        Returns dict with keys: recon [B], cos_sim [B], loss [B].
        """
        with torch.no_grad():
            output = self(x)
            x_hat = output["x_hat"]
            recon = self.recon_loss_fn(x_hat, x)
            cos_sim = F.cosine_similarity(x_hat, x, dim=-1)
            loss = self.mse_weight * recon + self.cosine_weight * (1.0 - cos_sim)
            return {"recon": recon, "cos_sim": cos_sim, "loss": loss}

    # -----------------------------------------------------------------
    # Data extraction
    # -----------------------------------------------------------------

    def _extract_vectors(self, batch) -> Tensor:
        """Extract HDC vectors from a PyG batch according to ``training_target``.

        After the edge_terms -> node_terms swap in the experiment,
        node_terms contains edge_terms. Returns edge_terms, graph_terms,
        or both concatenated depending on ``self.training_target``.
        """
        bs = batch.num_graphs
        node_terms = batch.node_terms.view(bs, -1).float()
        graph_terms = batch.graph_terms.view(bs, -1).float()
        if self.training_target == "edge":
            return node_terms.as_subclass(torch.Tensor)
        if self.training_target == "graph":
            return graph_terms.as_subclass(torch.Tensor)
        return torch.cat([node_terms, graph_terms], dim=-1).as_subclass(torch.Tensor)

    # -----------------------------------------------------------------
    # Training / validation steps
    # -----------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        x = self._extract_vectors(batch)
        output = self(x)
        return self._compute_loss(x, output, "train")

    def validation_step(self, batch, batch_idx):
        x = self._extract_vectors(batch)
        output = self(x)
        loss = self._compute_loss(x, output, "val")

        # Accumulate latent stats
        z = (output["mu"] if self.variational else output["z"]).detach()
        self._val_z_norm_sum += z.norm(dim=-1).sum().item()
        if self._val_z_sum is None:
            self._val_z_sum = z.sum(dim=0).cpu()
            self._val_z_sq_sum = (z ** 2).sum(dim=0).cpu()
        else:
            self._val_z_sum += z.sum(dim=0).cpu()
            self._val_z_sq_sum += (z ** 2).sum(dim=0).cpu()
        self._val_z_count += z.shape[0]

        return loss

    def on_validation_epoch_start(self):
        self._val_z_norm_sum = 0.0
        self._val_z_sum = None
        self._val_z_sq_sum = None
        self._val_z_count = 0

    def on_validation_epoch_end(self):
        if self._val_z_count > 0:
            n = self._val_z_count
            mean_norm = self._val_z_norm_sum / n
            mean = self._val_z_sum / n
            var = self._val_z_sq_sum / n - mean ** 2
            std_per_dim = var.clamp(min=0).sqrt()
            mean_std = std_per_dim.mean().item()

            self.log("val/z_mean_norm", mean_norm)
            self.log("val/z_mean_std_per_dim", mean_std)

    # -----------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: min(1.0, (epoch + 1) / max(1, self.warmup_epochs)),
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    # -----------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------

    def save(self, path) -> Path:
        """Save model weights and hyperparameters to a checkpoint file."""
        path = Path(path)
        checkpoint = {
            "state_dict": self.state_dict(),
            "hyper_parameters": dict(self.hparams),
        }
        torch.save(checkpoint, path)
        return path

    @classmethod
    def load(cls, path, map_location=None) -> "HDCAutoencoder":
        """Load a model from a checkpoint saved with :meth:`save`."""
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(**checkpoint["hyper_parameters"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model
