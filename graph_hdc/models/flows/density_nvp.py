"""
Real NVP density estimator for HDC vector scoring.

Trains a normalizing flow on the joint distribution of ``[node_terms | graph_terms]``
HDC vectors.  At inference, ``log_prob(x)`` returns an exact log-likelihood that
can be used to rank candidate vectors — higher is more likely under the training
distribution.

Key differences from :class:`RealNVPV3Lightning`:

* **Dequantization** — adds configurable Gaussian noise during training to smooth
  the discrete HDC lattice into a continuous density.
* **Simple save/load** — ``save()`` / ``load()`` use plain ``torch.save`` with
  config + state_dict (no PL checkpoint overhead).
* **Scoring API** — ``log_prob(x)`` is the primary entry-point.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import normflows as nf
import pytorch_lightning as pl
import torch
from torch import Tensor

from graph_hdc.models.flows.real_nvp import BoundedMLP


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DensityConfig:
    """Configuration for the density-estimation Real NVP."""

    # Hypervector dimension (per term — flat_dim = 2 * hv_dim).
    hv_dim: int = 256

    # Number of coupling layers.
    num_flows: int = 16

    # MLP hidden dimension inside each coupling layer.
    hidden_dim: int = 1024

    # Number of hidden layers in each coupling MLP.
    num_hidden_layers: int = 3

    # Maximum scale magnitude (tanh * smax).
    smax: float = 6.0

    # Scale warmup: linearly ramp smax from ``smax_initial`` to ``smax`` over
    # the first ``smax_warmup_epochs`` epochs for training stability.
    smax_initial: float = 2.0
    smax_warmup_epochs: int = 15

    # Whether to prepend an ActNorm layer.
    use_act_norm: bool = True

    # Dequantization noise standard deviation added during *training only*.
    # Smooths the discrete HDC lattice.  0.0 disables dequantization.
    dequant_sigma: float = 0.05

    # Training hyperparameters (stored so they travel with the model).
    lr: float = 2e-4
    weight_decay: float = 1e-4

    seed: int = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class DensityNVP(pl.LightningModule):
    """Real NVP normalizing flow for HDC density estimation.

    Operates on flat ``[node_terms | graph_terms]`` vectors of dimension
    ``2 * hv_dim``.  Coupling layers use **semantic masking** that alternates
    between the node-terms half and the graph-terms half so each coupling
    layer transforms one half conditioned on the other.

    Parameters
    ----------
    cfg : DensityConfig
        Model and training configuration.
    """

    def __init__(self, cfg: DensityConfig | dict):
        super().__init__()
        if isinstance(cfg, dict):
            cfg = DensityConfig(**cfg)
        self.save_hyperparameters()
        self.cfg = cfg

        D = int(cfg.hv_dim)
        self.D = D
        self.flat_dim = 2 * D

        default = torch.get_default_dtype()

        # Semantic masks: alternate between node_terms and graph_terms halves.
        mask_a = torch.zeros(self.flat_dim, dtype=default)
        mask_a[D:] = 1.0  # 0s for node_terms, 1s for graph_terms
        self.register_buffer("mask_a", mask_a)

        mask_b = 1.0 - mask_a  # 1s for node_terms, 0s for graph_terms
        self.register_buffer("mask_b", mask_b)

        # Per-term standardization buffers.
        self.register_buffer("mu", torch.zeros(self.flat_dim, dtype=default))
        self.register_buffer("log_sigma", torch.zeros(self.flat_dim, dtype=default))

        # Build flow -------------------------------------------------------
        self.s_modules: list[BoundedMLP] = []
        flows: list[nf.flows.Flow] = []

        if cfg.use_act_norm and hasattr(nf.flows, "ActNorm"):
            flows.append(nf.flows.ActNorm(self.flat_dim))

        hidden_dim = int(cfg.hidden_dim)
        num_hidden = int(cfg.num_hidden_layers)
        mlp_layers = [self.flat_dim] + [hidden_dim] * num_hidden + [self.flat_dim]

        for i in range(int(cfg.num_flows)):
            t_net = nf.nets.MLP(mlp_layers.copy(), init_zeros=True)
            s_net = BoundedMLP(mlp_layers.copy(), smax=cfg.smax)
            self.s_modules.append(s_net)

            mask = self.mask_a if i % 2 == 0 else self.mask_b
            flows.append(nf.flows.MaskedAffineFlow(mask, t=t_net, s=s_net))

        base = nf.distributions.DiagGaussian(self.flat_dim, trainable=False)
        self.flow = nf.NormalizingFlow(q0=base, flows=flows)

    # ------------------------------------------------------------------
    # Standardization
    # ------------------------------------------------------------------

    def set_standardization(
        self,
        mu: Tensor,
        sigma: Tensor,
        eps: float = 1e-6,
    ) -> None:
        """Set per-dimension standardization parameters.

        Parameters
        ----------
        mu : Tensor
            Mean vector ``(flat_dim,)``.
        sigma : Tensor
            Standard deviation vector ``(flat_dim,)``.
        """
        tgt = self.mu.dtype
        mu = torch.as_tensor(mu, dtype=tgt, device=self.device)
        sigma = torch.as_tensor(sigma, dtype=tgt, device=self.device)
        self.mu.copy_(mu)
        self.log_sigma.copy_(torch.log(torch.clamp(sigma, min=eps)))

    def _standardize(self, x: Tensor) -> tuple[Tensor, float]:
        """Standardize and return the log-det correction for the Jacobian."""
        D = self.D
        node = x[..., :D]
        graph = x[..., D:]

        z_node = (node - self.mu[:D]) * torch.exp(-self.log_sigma[:D])
        z_graph = (graph - self.mu[D:]) * torch.exp(-self.log_sigma[D:])
        z = torch.cat([z_node, z_graph], dim=-1)

        return z, float(self.log_sigma.sum().item())

    def _destandardize(self, z: Tensor) -> Tensor:
        """Invert standardization."""
        D = self.D
        node = self.mu[:D] + z[..., :D] * torch.exp(self.log_sigma[:D])
        graph = self.mu[D:] + z[..., D:] * torch.exp(self.log_sigma[D:])
        return torch.cat([node, graph], dim=-1)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def log_prob(self, x: Tensor) -> Tensor:
        """Compute exact log-likelihood for scoring.

        Parameters
        ----------
        x : Tensor
            ``(batch, 2 * hv_dim)`` flat HDC vectors.

        Returns
        -------
        Tensor
            ``(batch,)`` log-probabilities.  Higher = more likely.
        """
        z, log_det_corr = self._standardize(x)
        # flow.log_prob returns log p(z) + log |det J_flow|
        log_p = self.flow.log_prob(z)
        # Correct for the standardization Jacobian: log |det diag(1/sigma)|
        return log_p - log_det_corr

    def nll(self, x: Tensor) -> Tensor:
        """Negative log-likelihood (for loss computation).

        Returns
        -------
        Tensor
            ``(batch,)`` NLL values.  Lower = more likely.
        """
        return -self.log_prob(x)

    @torch.no_grad()
    def sample(self, num_samples: int) -> Tensor:
        """Sample from the learned distribution.

        Returns
        -------
        Tensor
            ``(num_samples, 2 * hv_dim)`` vectors in data space.
        """
        z, _ = self.flow.sample(num_samples)
        return self._destandardize(z)

    # ------------------------------------------------------------------
    # Batch extraction
    # ------------------------------------------------------------------

    def _flat_from_batch(self, batch) -> Tensor:
        """Extract flat ``[node_terms | graph_terms]`` from a PyG batch.

        Uses ``.as_subclass(Tensor)`` to strip PyG tensor subclass metadata
        which otherwise breaks ``deepcopy`` inside PL's ModelCheckpoint.
        """
        B = batch.num_graphs
        node = batch.node_terms.view(B, self.D).as_subclass(Tensor).float()
        graph = batch.graph_terms.view(B, self.D).as_subclass(Tensor).float()
        return torch.cat([node, graph], dim=-1)

    # ------------------------------------------------------------------
    # Lightning training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        flat = self._flat_from_batch(batch)

        # Dequantization: smooth the discrete lattice.
        if self.cfg.dequant_sigma > 0:
            flat = flat + torch.randn_like(flat) * self.cfg.dequant_sigma

        obj = self.nll(flat)

        # Filter NaN/Inf.
        obj = obj[torch.isfinite(obj)]
        if obj.numel() == 0:
            self.log("nan_loss_batches", 1.0, on_step=True, prog_bar=True, batch_size=flat.size(0))
            return None

        loss = obj.mean()
        self.log(
            "train_loss",
            float(loss.detach().cpu().item()),
            on_step=True, on_epoch=True, prog_bar=True, batch_size=flat.size(0),
        )

        # Monitor scale magnitude.
        with torch.no_grad():
            s_absmax = 0.0
            for m in self.s_modules:
                if getattr(m, "last_pre", None) is not None and torch.isfinite(m.last_pre).any():
                    s_absmax = max(s_absmax, float(m.last_pre.detach().abs().max().cpu().item()))
        self.log("s_pre_absmax", s_absmax, on_step=True, prog_bar=True, batch_size=flat.size(0))

        return loss

    def validation_step(self, batch, batch_idx):
        flat = self._flat_from_batch(batch)
        # No dequantization during validation — evaluate on clean data.
        obj = self.nll(flat)
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

        # Linear warmup (5%) then cosine annealing.
        steps_per_epoch = max(
            1,
            getattr(self.trainer, "estimated_stepping_batches", 1000)
            // max(1, self.trainer.max_epochs),
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
        # Scale warmup for stability.
        warm = int(self.cfg.smax_warmup_epochs)
        s0 = float(self.cfg.smax_initial)
        s1 = float(self.cfg.smax)
        if warm > 0:
            t = min(1.0, self.current_epoch / max(1, warm))
            smax = (1 - t) * s0 + t * s1
            for m in self.s_modules:
                m.smax = smax

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return batch.to(device)

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        return _cast_to_dtype(batch, self.dtype)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save model to a standalone file (config + weights + standardization)."""
        torch.save(
            {
                "config": asdict(self.cfg),
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "DensityNVP":
        """Load a model from a standalone ``.pt`` or a PL checkpoint ``.ckpt``.

        Parameters
        ----------
        path : str | Path
            Path to the ``.pt`` file (from :meth:`save`) **or** a PyTorch
            Lightning checkpoint ``.ckpt`` (from ``ModelCheckpoint``).
        device : str | torch.device
            Device to map tensors to.
        """
        data = torch.load(path, map_location=device, weights_only=False)

        # Standalone format from model.save(): {"config": {...}, "state_dict": {...}}
        if "config" in data:
            cfg = DensityConfig(**data["config"])
            state = data["state_dict"]
        # PL checkpoint format: {"hyper_parameters": {"cfg": {...}}, "state_dict": {...}}
        elif "hyper_parameters" in data:
            hp = data["hyper_parameters"]
            cfg_raw = hp.get("cfg", hp)
            if isinstance(cfg_raw, DensityConfig):
                cfg = cfg_raw
            elif isinstance(cfg_raw, dict):
                cfg = DensityConfig(**cfg_raw)
            else:
                raise ValueError(f"Cannot parse hyper_parameters: {type(cfg_raw)}")
            state = data["state_dict"]
        else:
            raise KeyError(
                f"Unrecognized checkpoint format. Expected 'config' or "
                f"'hyper_parameters' key, found: {list(data.keys())}"
            )

        model = cls(cfg)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model
