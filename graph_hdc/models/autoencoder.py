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

import math
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SwiGLUResidualBlock(nn.Module):
    """Pre-norm SwiGLU residual block (canonical LLaMA/PaLM form):

        ``x + Dropout( W_out( SiLU(W_gate LN(x)) * W_value LN(x) ) )``

    ``ffn_mult`` widens the gate/value matrices internally (canonical
    SwiGLU uses 8/3 ≈ 2.67). ``W_out`` is zero-initialized so the block
    starts as identity, which helps stable training of deep stacks.
    Dropout is applied to the residual delta (after ``W_out``, before the
    skip-add); since ``W_out`` is zero at init the dropped tensor is also
    zero, so dropout's effect ramps in monotonically as the block wakes up.
    """

    def __init__(self, dim: int, ffn_mult: float = 8 / 3, dropout: float = 0.0):
        super().__init__()
        ffn_dim = int(round(ffn_mult * dim))
        ffn_dim = ((ffn_dim + 63) // 64) * 64  # round up to multiple of 64
        self.norm = nn.LayerNorm(dim)
        self.w_gate = nn.Linear(dim, ffn_dim, bias=False)
        self.w_value = nn.Linear(dim, ffn_dim, bias=False)
        self.w_out = nn.Linear(ffn_dim, dim, bias=False)
        nn.init.zeros_(self.w_out.weight)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        delta = self.w_out(F.silu(self.w_gate(h)) * self.w_value(h))
        return x + self.dropout(delta)


class ProjectionBlock(nn.Module):
    """Linear projection followed by *K* SwiGLU residual blocks.

    Used for both encoder (down-projection) and decoder (up-projection)
    layers -- the class is agnostic to the direction.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_blocks: int = 2,
        ffn_mult: float = 8 / 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.blocks = nn.ModuleList(
            [
                SwiGLUResidualBlock(out_dim, ffn_mult=ffn_mult, dropout=dropout)
                for _ in range(num_blocks)
            ]
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

    Architecture is a constant-width "trunk":
        encoder: Linear(data_dim → trunk_dim)
                 → [SwiGLUResidualBlock(trunk_dim)] * n_encoder_blocks
                 → Linear(trunk_dim → latent_dim)        (mu/logvar if VAE)
        decoder: Linear(latent_dim → trunk_dim)
                 → [SwiGLUResidualBlock(trunk_dim)] * n_decoder_blocks
                 → Linear(trunk_dim → data_dim)

    All residual mixing happens at a single fixed width; only the stem and
    head change dimensionality. This is the design used by ResNets and
    modern Transformers and tends to be easier to train and more
    parameter-efficient than a geometric funnel.

    Args:
        data_dim: Input/output dimension (typically 2 * hv_dim).
        latent_dim: Bottleneck dimension.
        trunk_dim: Width of the residual trunk. Both encoder and decoder
            do all their nonlinear mixing at this width. Recommended:
            ``trunk_dim ≥ data_dim`` so the stem isn't an aggressive
            lossy compressor.
        n_encoder_blocks: Number of SwiGLU residual blocks in the encoder
            trunk.
        n_decoder_blocks: Number of SwiGLU residual blocks in the decoder
            trunk.
        ffn_mult: Internal width multiplier for the SwiGLU residual blocks.
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
        free_bits: Per-dimension KL floor in nats (VAE only). If > 0, each
            latent dim's batch-averaged KL is clamped at this value before
            being summed into the loss — dims with KL below ``free_bits``
            pay no penalty. Prevents posterior collapse. Typical values:
            0.1–0.5. ``0.0`` (default) disables the feature.
        loss_clamp: If set, clamp per-sample reconstruction loss to this
            maximum before averaging. Prevents outlier samples from causing
            gradient explosions. None means no clamping.
        lr: Peak learning rate for AdamW (reached at end of warmup).
        weight_decay: Weight decay for AdamW.
        adam_eps: Epsilon for AdamW numerical stability.
        warmup_epochs: Number of linear LR warmup epochs.
        max_epochs: Total training epochs — used to schedule the cosine
            decay from ``lr`` (at end of warmup) down to ``min_lr``.
        min_lr: Floor learning rate at the end of cosine decay.
        latent_stats_ema_decay: EMA decay for the running per-dim mean/std
            of the encoder's latent (``mu`` for VAE, ``z`` otherwise).
            Stored as buffers so downstream code (e.g. a flow trained on
            this latent) can fetch ``ae.latent_mean_ema`` /
            ``ae.latent_std_ema`` straight from the loaded checkpoint and
            train on whitened latents.
    """

    def __init__(
        self,
        data_dim: int,
        latent_dim: int,
        trunk_dim: int = 1024,
        n_encoder_blocks: int = 8,
        n_decoder_blocks: int = 8,
        ffn_mult: float = 8 / 3,
        dropout: float = 0.0,
        training_target: str = "both",
        recon_loss_type: str = "mse",
        berhu_c_fraction: float = 0.2,
        variational: bool = False,
        mse_weight: float = 1.0,
        cosine_weight: float = 1.0,
        kl_weight: float = 1.0,
        kl_warmup_epochs: int = 20,
        free_bits: float = 0.0,
        loss_clamp: Optional[float] = None,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        adam_eps: float = 1e-8,
        warmup_epochs: int = 5,
        max_epochs: int = 250,
        min_lr: float = 1e-7,
        latent_stats_ema_decay: float = 0.99,
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
        self.free_bits = free_bits
        self.loss_clamp = loss_clamp
        self.trunk_dim = trunk_dim
        self.n_encoder_blocks = n_encoder_blocks
        self.n_decoder_blocks = n_decoder_blocks
        self.ffn_mult = ffn_mult
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.adam_eps = adam_eps
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.latent_stats_ema_decay = latent_stats_ema_decay

        # ----- Encoder: stem + constant-width trunk -----
        self.encoder_stem = nn.Linear(data_dim, trunk_dim)
        self.encoder_backbone = nn.Sequential(*[
            SwiGLUResidualBlock(trunk_dim, ffn_mult=ffn_mult, dropout=dropout)
            for _ in range(n_encoder_blocks)
        ])

        # LayerNorm before the latent heads decouples the encoder's content
        # from its scale. Without this, drift in trunk activations leaks into
        # mu/logvar magnitudes (and especially into exp(logvar)) and can
        # destabilize the KL term during warmup.
        self.latent_norm = nn.LayerNorm(trunk_dim)

        if variational:
            self.fc_mu = nn.Linear(trunk_dim, latent_dim)
            self.fc_logvar = nn.Linear(trunk_dim, latent_dim)
            # Zero-init fc_logvar so logvar ≈ 0 (std ≈ 1) at start of training,
            # regardless of which direction `latent_norm` outputs. Without
            # this, exp(logvar) can swing wildly during the first epochs as
            # the trunk blocks wake up and rotate the post-LN direction.
            nn.init.zeros_(self.fc_logvar.weight)
            nn.init.zeros_(self.fc_logvar.bias)
        else:
            self.fc_latent = nn.Linear(trunk_dim, latent_dim)

        # ----- Decoder: head + constant-width trunk + un-projection -----
        self.decoder_head = nn.Linear(latent_dim, trunk_dim)
        self.decoder_trunk = nn.Sequential(*[
            SwiGLUResidualBlock(trunk_dim, ffn_mult=ffn_mult, dropout=dropout)
            for _ in range(n_decoder_blocks)
        ])
        self.decoder_out = nn.Linear(trunk_dim, data_dim)

        # Running per-dim stats of the latent (mu for VAE, z otherwise).
        # Saved with state_dict so a downstream flow can whiten latents
        # using these without a separate fitting pass over the dataset.
        self.register_buffer("latent_mean_ema", torch.zeros(latent_dim))
        self.register_buffer("latent_std_ema", torch.ones(latent_dim))
        self.register_buffer(
            "latent_stats_initialized", torch.tensor(False)
        )

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
        h = self.encoder_backbone(self.encoder_stem(x))
        h = self.latent_norm(h)
        if self.variational:
            # Clamp logvar to keep exp(logvar) in a numerically tame range
            # (≈[4.5e-5, 403]). Bounds the per-dim KL contribution at ~200
            # nats even if fc_logvar overshoots — prevents the runaway-KL
            # spikes seen at the LR warmup boundary. Zero gradient inside
            # the saturated region only.
            logvar = self.fc_logvar(h).clamp(min=-10.0, max=6.0)
            return {"mu": self.fc_mu(h), "logvar": logvar}
        return {"z": self.fc_latent(h)}

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """VAE reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent vector to reconstruction."""
        return self.decoder_out(self.decoder_trunk(self.decoder_head(z)))

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
            # Per-sample, per-dim KL  [B, D]
            kl_per_sample_per_dim = -0.5 * (
                1 + output["logvar"] - output["mu"].pow(2) - output["logvar"].exp()
            )
            # Batch-mean per dim, then sum. Free bits clamps per dim before the
            # sum so dims with KL below the floor contribute a constant (zero
            # gradient) instead of being pushed further down toward collapse.
            kl_per_dim = kl_per_sample_per_dim.mean(dim=0)          # [D]
            kl_raw = kl_per_dim.sum()                                # scalar
            if self.free_bits > 0.0:
                kl_loss = kl_per_dim.clamp(min=self.free_bits).sum()
            else:
                kl_loss = kl_raw

            beta = self._get_beta(self.current_epoch)
            loss = recon_loss + beta * kl_loss

            self.log(f"{prefix}/kl", kl_raw, on_step=False, on_epoch=True, batch_size=bs)
            self.log(f"{prefix}/kl_free", kl_loss, on_step=False, on_epoch=True, batch_size=bs)
            self.log(f"{prefix}/beta", beta, on_step=False, on_epoch=True, batch_size=bs)
            # Diagnostics: active dims (KL > 0.01 nats) and dims pinned at the floor.
            n_active = (kl_per_dim > 0.01).float().sum()
            self.log(f"{prefix}/n_active_dims", n_active, on_step=False, on_epoch=True, batch_size=bs)
            if self.free_bits > 0.0:
                n_at_floor = (kl_per_dim < self.free_bits).float().sum()
                self.log(f"{prefix}/n_at_floor", n_at_floor, on_step=False, on_epoch=True, batch_size=bs)
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
        loss = self._compute_loss(x, output, "train")
        self._update_latent_stats_ema(output)
        return loss

    def _update_latent_stats_ema(self, output: dict) -> None:
        """Update running per-dim mean/std of the latent for downstream whitening.

        For VAE we track ``mu`` (deterministic, what the flow consumes); for
        plain AE we track ``z``. Initial batch seeds the EMA exactly so the
        warm-up isn't dragged down by the zeros-init.
        """
        z = (output["mu"] if self.variational else output["z"]).detach()
        batch_mean = z.mean(dim=0)
        batch_std = z.std(dim=0) + 1e-6
        if not bool(self.latent_stats_initialized):
            self.latent_mean_ema.copy_(batch_mean)
            self.latent_std_ema.copy_(batch_std)
            self.latent_stats_initialized.fill_(True)
        else:
            m = self.latent_stats_ema_decay
            self.latent_mean_ema.mul_(m).add_(batch_mean, alpha=1 - m)
            self.latent_std_ema.mul_(m).add_(batch_std, alpha=1 - m)

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
            eps=self.adam_eps,
        )

        warmup = max(1, self.warmup_epochs)
        total = max(warmup + 1, self.max_epochs)
        min_ratio = self.min_lr / self.lr if self.lr > 0 else 0.0

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup:
                return (epoch + 1) / warmup
            progress = (epoch - warmup) / max(1, total - warmup)
            progress = min(1.0, max(0.0, progress))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
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
