"""
Real NVP V3 Normalizing Flow for Molecular Generation.

This module implements the Real-valued Non-Volume Preserving (Real NVP)
normalizing flow model with semantic masking for HDC embeddings.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import normflows as nf
import pytorch_lightning as pl
import torch
from torch import Tensor


@dataclass
class FlowConfig:
    """Configuration for Real NVP V3 model."""

    # Model architecture
    hv_dim: int = 256
    num_flows: int = 16
    hidden_dim: int = 1792
    num_hidden_layers: int = 3

    # Scale warmup
    smax_initial: float = 2.2
    smax_final: float = 6.5
    smax_warmup_epochs: int = 16

    # Training
    epochs: int = 800
    batch_size: int = 96
    lr: float = 1.91e-4
    weight_decay: float = 3.49e-4

    # Options
    use_act_norm: bool = True
    per_term_standardization: bool = True
    hv_count: int = 2

    seed: int = 42
    device: str = "auto"  # "auto", "cuda", or "cpu"


# Best configs for QM9 and ZINC
QM9_FLOW_CONFIG = FlowConfig(
    hv_dim=256,
    num_flows=16,
    hidden_dim=1792,
    num_hidden_layers=3,
    smax_initial=2.2,
    smax_final=6.5,
    smax_warmup_epochs=16,
    batch_size=96,
    lr=1.91e-4,
    weight_decay=3.49e-4,
    epochs=800,
)

ZINC_FLOW_CONFIG = FlowConfig(
    hv_dim=256,
    num_flows=8,
    hidden_dim=1536,
    num_hidden_layers=2,
    smax_initial=2.5,
    smax_final=7.0,
    smax_warmup_epochs=17,
    batch_size=224,
    lr=5.39e-4,
    weight_decay=1.0e-3,
    epochs=800,
)


class BoundedMLP(torch.nn.Module):
    """MLP with bounded output for numerical stability."""

    def __init__(self, dims: list[int], smax: float = 6.0):
        super().__init__()
        self.net = nf.nets.MLP(dims, init_zeros=True)
        self.smax = float(smax)
        self.last_pre = None

    def forward(self, x: Tensor) -> Tensor:
        pre = self.net(x)
        self.last_pre = pre
        return torch.tanh(pre) * self.smax


def _cast_to_dtype(x: Any, dtype: torch.dtype) -> Any:
    """Recursively cast tensors to target dtype."""
    if torch.is_tensor(x):
        return x.to(dtype) if x.dtype.is_floating_point else x
    if isinstance(x, dict):
        return {k: _cast_to_dtype(v, dtype) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = type(x)
        return t(_cast_to_dtype(v, dtype) for v in x)
    return x


class RealNVPV3Lightning(pl.LightningModule):
    """
    Real NVP V3 normalizing flow with semantic masking.

    Uses separate masks for edge_terms and graph_terms to respect
    the HDC embedding structure during flow transformations.

    Parameters
    ----------
    cfg : FlowConfig
        Model configuration
    """

    def __init__(self, cfg: FlowConfig | dict):
        super().__init__()
        if isinstance(cfg, dict):
            cfg = FlowConfig(**cfg)
        self.save_hyperparameters()
        self.cfg = cfg

        D = int(cfg.hv_dim)
        self.D = D
        self.hv_count = getattr(cfg, "hv_count", 2)
        self.flat_dim = self.hv_count * D

        default = torch.get_default_dtype()

        # Semantic masks: alternate between edge and graph terms
        mask_a = torch.zeros(self.flat_dim, dtype=default)
        mask_a[D:] = 1.0  # 0s for edge_terms, 1s for graph_terms
        self.register_buffer("mask_a", mask_a)

        mask_b = 1.0 - mask_a  # 1s for edge_terms, 0s for graph_terms
        self.register_buffer("mask_b", mask_b)

        # Standardization buffers
        self.register_buffer("mu", torch.zeros(self.flat_dim, dtype=default))
        self.register_buffer("log_sigma", torch.zeros(self.flat_dim, dtype=default))
        self.per_term_standardization = getattr(cfg, "per_term_standardization", True)
        self._per_term_split = D if self.per_term_standardization else None

        # Build flow
        self.s_modules: list[BoundedMLP] = []
        flows: list[nf.flows.Flow] = []

        # ActNorm
        if getattr(cfg, "use_act_norm", True) and hasattr(nf.flows, "ActNorm"):
            flows.append(nf.flows.ActNorm(self.flat_dim))

        # Coupling layers
        hidden_dim = int(getattr(cfg, "hidden_dim", 1024))
        num_hidden_layers = int(getattr(cfg, "num_hidden_layers", 3))
        mlp_layers = [self.flat_dim] + [hidden_dim] * num_hidden_layers + [self.flat_dim]

        for i in range(int(cfg.num_flows)):
            smax = getattr(cfg, "smax_final", 6.0)

            t_net = nf.nets.MLP(mlp_layers.copy(), init_zeros=True)
            s_net = BoundedMLP(mlp_layers.copy(), smax=smax)
            self.s_modules.append(s_net)

            # Alternate semantic mask
            mask = self.mask_a if i % 2 == 0 else self.mask_b
            flows.append(nf.flows.MaskedAffineFlow(mask, t=t_net, s=s_net))

        # Base distribution
        base = nf.distributions.DiagGaussian(self.flat_dim, trainable=False)
        self.flow = nf.NormalizingFlow(q0=base, flows=flows)

    def set_standardization(
        self,
        mu: Tensor,
        sigma: Tensor,
        eps: float = 1e-6,
    ) -> None:
        """Set standardization parameters."""
        tgt_dtype = self.mu.dtype
        mu = torch.as_tensor(mu, dtype=tgt_dtype, device=self.device)
        sigma = torch.as_tensor(sigma, dtype=tgt_dtype, device=self.device)
        self.mu.copy_(mu)
        self.log_sigma.copy_(torch.log(torch.clamp(sigma, min=eps)))

    def set_per_term_standardization(
        self,
        edge_mean: Tensor,
        edge_std: Tensor,
        graph_mean: Tensor,
        graph_std: Tensor,
        eps: float = 1e-6,
    ) -> None:
        """Set per-term standardization from separate edge/graph stats."""
        mu = torch.cat([edge_mean, graph_mean])
        sigma = torch.cat([edge_std, graph_std])
        self.set_standardization(mu, sigma, eps)
        self._per_term_split = self.D

    def _pretransform(self, x: Tensor) -> tuple[Tensor, float]:
        """Standardize input: z = (x - mu) / sigma."""
        if self.per_term_standardization and self._per_term_split is not None:
            split = self._per_term_split
            edge = x[..., :split]
            graph = x[..., split:]

            mu_edge = self.mu[:split]
            mu_graph = self.mu[split:]
            log_sigma_edge = self.log_sigma[:split]
            log_sigma_graph = self.log_sigma[split:]

            z_edge = (edge - mu_edge) * torch.exp(-log_sigma_edge)
            z_graph = (graph - mu_graph) * torch.exp(-log_sigma_graph)
            z = torch.cat([z_edge, z_graph], dim=-1)
        else:
            z = (x - self.mu) * torch.exp(-self.log_sigma)

        return z, float(self.log_sigma.sum().item())

    def _posttransform(self, z: Tensor) -> Tensor:
        """De-standardize output: x = mu + z * sigma."""
        if self.per_term_standardization and self._per_term_split is not None:
            split = self._per_term_split
            z_edge = z[..., :split]
            z_graph = z[..., split:]

            mu_edge = self.mu[:split]
            mu_graph = self.mu[split:]
            log_sigma_edge = self.log_sigma[:split]
            log_sigma_graph = self.log_sigma[split:]

            edge = mu_edge + z_edge * torch.exp(log_sigma_edge)
            graph = mu_graph + z_graph * torch.exp(log_sigma_graph)
            return torch.cat([edge, graph], dim=-1)

        return self.mu + z * torch.exp(self.log_sigma)

    def split(self, flat: Tensor) -> tuple[Tensor, Tensor]:
        """Split flat tensor into edge_terms and graph_terms."""
        D = self.D
        return (
            flat[:, :D].contiguous(),  # edge_terms
            flat[:, D:].contiguous(),  # graph_terms
        )

    def sample(self, num_samples: int) -> tuple[Tensor, Tensor]:
        """Sample from the flow in standardized space."""
        z, logs = self.flow.sample(num_samples)
        return z, logs

    @torch.no_grad()
    def sample_split(self, num_samples: int) -> dict[str, Tensor]:
        """
        Sample molecules and return split embeddings.

        Returns
        -------
        dict
            - edge_terms: [num_samples, D]
            - graph_terms: [num_samples, D]
            - logs: log probabilities
        """
        z, logs = self.sample(num_samples)
        x = self._posttransform(z)
        edge_terms, graph_terms = self.split(x)
        return {
            "edge_terms": edge_terms,
            "graph_terms": graph_terms,
            "logs": logs,
        }

    def decode_from_latent(self, z_std: Tensor) -> Tensor:
        """Decode from standardized latent space (differentiable)."""
        x_std = self.flow.forward(z_std)
        return self._posttransform(x_std)

    def nf_forward_kld(self, flat: Tensor) -> Tensor:
        """Compute negative log-likelihood."""
        z, log_det_corr = self._pretransform(flat)
        nll = -self.flow.log_prob(z) + log_det_corr
        return nll

    def _flat_from_batch(self, batch) -> Tensor:
        """Extract flat tensor from PyG batch."""
        D = self.D
        B = batch.num_graphs
        e = batch.edge_terms.view(B, D)
        g = batch.graph_terms.view(B, D)
        return torch.cat([e, g], dim=-1)

    def training_step(self, batch, batch_idx):
        flat = self._flat_from_batch(batch)
        obj = self.nf_forward_kld(flat)

        # Filter NaN values
        obj = obj[torch.isfinite(obj)]
        if obj.numel() == 0:
            self.log("nan_loss_batches", 1.0, on_step=True, prog_bar=True, batch_size=flat.size(0))
            return None

        loss = obj.mean()
        self.log(
            "train_loss",
            float(loss.detach().cpu().item()),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=flat.size(0),
        )

        # Monitor scale magnitude
        with torch.no_grad():
            s_absmax = 0.0
            for m in self.s_modules:
                if getattr(m, "last_pre", None) is not None and torch.isfinite(m.last_pre).any():
                    s_absmax = max(s_absmax, float(m.last_pre.detach().abs().max().cpu().item()))
        self.log("s_pre_absmax", s_absmax, on_step=True, prog_bar=True, batch_size=flat.size(0))

        return loss

    def validation_step(self, batch, batch_idx):
        flat = self._flat_from_batch(batch)
        obj = self.nf_forward_kld(flat)
        obj = obj[torch.isfinite(obj)]
        val = float("nan") if obj.numel() == 0 else float(obj.mean().detach().cpu().item())
        self.log("val_loss", val, on_epoch=True, prog_bar=True, batch_size=flat.size(0))
        return torch.tensor(val, device=flat.device) if val == val else None

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            foreach=True,
        )

        # 5% warmup then cosine annealing
        steps_per_epoch = max(
            1,
            getattr(self.trainer, "estimated_stepping_batches", 1000) // max(1, self.trainer.max_epochs),
        )
        warmup = int(0.05 * self.trainer.max_epochs) * steps_per_epoch
        total = self.trainer.max_epochs * steps_per_epoch

        sched = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lambda step: min(1.0, step / max(1, warmup))
            * 0.5
            * (1 + math.cos(math.pi * max(0, step - warmup) / max(1, total - warmup))),
        )

        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    def on_fit_start(self):
        with contextlib.suppress(Exception):
            torch.set_float32_matmul_precision("high")

    def on_train_epoch_start(self):
        # Scale warmup for stability
        warm = int(getattr(self.cfg, "smax_warmup_epochs", 15))
        s0 = float(getattr(self.cfg, "smax_initial", 1.0))
        s1 = float(getattr(self.cfg, "smax_final", 6.0))
        if warm > 0:
            t = min(1.0, self.current_epoch / max(1, warm))
            smax = (1 - t) * s0 + t * s1
            for m in self.s_modules:
                m.smax = smax

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return batch.to(device)

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        return _cast_to_dtype(batch, self.dtype)

    @classmethod
    def load_from_checkpoint_file(cls, path: str | Path) -> "RealNVPV3Lightning":
        """
        Load the model from a checkpoint file.

        Handles module path remapping for checkpoints saved with different
        package structures (e.g., 'src' vs. 'graph_hdc').
        """
        from graph_hdc.utils.compat import setup_module_aliases
        setup_module_aliases()

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        cfg_dict = checkpoint["hyper_parameters"].get("cfg", {})
        if isinstance(cfg_dict, FlowConfig):
            cfg = cfg_dict
        else:
            cfg = FlowConfig(**cfg_dict)
        model = cls(cfg)
        model.load_state_dict(checkpoint["state_dict"])
        return model
