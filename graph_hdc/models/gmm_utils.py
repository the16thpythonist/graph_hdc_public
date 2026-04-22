"""
Gaussian Mixture Model utilities for flow matching velocity prediction.

Provides a GMM output head and standalone functions for NLL loss computation,
deterministic mean collapse, and stochastic sampling. All operations work on
flat vectors of shape ``(bs, D)`` and are domain-agnostic.

Reference: Chen et al., "Gaussian Mixture Flow Matching Models" (2025),
https://arxiv.org/abs/2504.05304
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


# =============================================================================
# GMM Output Head
# =============================================================================


class GMMOutputHead(nn.Module):
    """Project backbone features into Gaussian mixture parameters.

    Takes ``(bs, hidden_dim)`` from the velocity network backbone and
    produces K Gaussian components with shared isotropic std.

    Output dict:
        - ``means``: ``(bs, K, data_dim)``
        - ``logweights``: ``(bs, K)``  (log-softmax normalized)
        - ``logstds``: ``(bs,)``  (shared scalar, clamped)

    All output layers are zero-initialized so the model starts with
    uniform weights, near-zero means, and sigma=1.

    Args:
        hidden_dim: Input feature dimension from backbone.
        data_dim: Output data dimension per component.
        num_gaussians: Number of mixture components K.
        logstd_min: Lower clamp for logstd (prevents sigma→0).
        logstd_max: Upper clamp for logstd.
    """

    def __init__(
        self,
        hidden_dim: int,
        data_dim: int,
        num_gaussians: int = 8,
        logstd_min: float = -10.0,
        logstd_max: float = 2.0,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.num_gaussians = num_gaussians
        self.logstd_min = logstd_min
        self.logstd_max = logstd_max

        self.out_means = nn.Linear(hidden_dim, num_gaussians * data_dim)
        self.out_logweights = nn.Linear(hidden_dim, num_gaussians)
        self.out_logstds = nn.Linear(hidden_dim, 1)

        # Small random init for means to break symmetry between components.
        # Kaiming/Xavier are too large at high data_dim (D=2048), causing
        # winner-take-all collapse. std=0.01 gives gentle asymmetry.
        nn.init.normal_(self.out_means.weight, std=0.01)
        nn.init.zeros_(self.out_means.bias)
        nn.init.zeros_(self.out_logweights.weight)
        nn.init.zeros_(self.out_logweights.bias)
        nn.init.zeros_(self.out_logstds.weight)
        nn.init.zeros_(self.out_logstds.bias)

    def forward(self, h: Tensor) -> dict[str, Tensor]:
        """
        Args:
            h: ``(bs, hidden_dim)`` backbone features.

        Returns:
            dict with ``means``, ``logweights``, ``logstds``.
        """
        bs = h.shape[0]

        means = self.out_means(h).view(bs, self.num_gaussians, self.data_dim)
        logweights = self.out_logweights(h).log_softmax(dim=-1)  # (bs, K)
        logstds = self.out_logstds(h).squeeze(-1)  # (bs,)
        logstds = logstds.clamp(self.logstd_min, self.logstd_max)

        return {"means": means, "logweights": logweights, "logstds": logstds}


# =============================================================================
# GMM Loss
# =============================================================================


def gm_nll_loss(gm: dict[str, Tensor], target: Tensor, eps: float = 1e-4) -> Tensor:
    """Gaussian mixture negative log-likelihood loss.

    Computes ``-log p(target | GMM)`` where the GMM has shared isotropic
    covariance across all K components.

    Args:
        gm: dict with:
            - ``means``: ``(bs, K, D)``
            - ``logweights``: ``(bs, K)``
            - ``logstds``: ``(bs,)``
        target: ``(bs, D)`` velocity or x_1 target.
        eps: Numerical stability for inverse std clamping.

    Returns:
        ``(bs,)`` per-sample NLL.
    """
    means = gm["means"]  # (bs, K, D)
    logweights = gm["logweights"]  # (bs, K)
    logstds = gm["logstds"]  # (bs,)

    # Broadcast logstds to (bs, 1, 1) for element-wise ops
    logstds_exp = logstds[:, None, None]  # (bs, 1, 1)

    inverse_stds = torch.exp(-logstds_exp).clamp(max=1.0 / eps)
    # (bs, K, D)
    diff_weighted = (target.unsqueeze(-2) - means) * inverse_stds
    # Per-component log-likelihood summed over D: (bs, K)
    gaussian_ll = (-0.5 * diff_weighted.pow(2) - logstds_exp).sum(dim=-1)
    # Mix with weights via logsumexp: (bs,)
    nll = -torch.logsumexp(gaussian_ll + logweights, dim=-1)

    return nll


# =============================================================================
# GMM Collapse Operations
# =============================================================================


def gm_to_mean(gm: dict[str, Tensor]) -> Tensor:
    """Deterministic collapse: weighted mean of mixture components.

    Args:
        gm: dict with ``means`` (bs, K, D) and ``logweights`` (bs, K).

    Returns:
        ``(bs, D)`` weighted mean.
    """
    weights = gm["logweights"].softmax(dim=-1)  # (bs, K)
    return (weights.unsqueeze(-1) * gm["means"]).sum(dim=-2)  # (bs, D)


@torch.no_grad()
def gm_to_sample(gm: dict[str, Tensor]) -> Tensor:
    """Stochastic sample: pick a component, then draw from it.

    Samples one component index per batch element from
    ``Categorical(softmax(logweights))``, then draws from
    ``N(means_k, exp(2 * logstds))``.

    Args:
        gm: dict with ``means`` (bs, K, D), ``logweights`` (bs, K),
            ``logstds`` (bs,).

    Returns:
        ``(bs, D)`` sampled vectors.
    """
    means = gm["means"]  # (bs, K, D)
    logweights = gm["logweights"]  # (bs, K)
    logstds = gm["logstds"]  # (bs,)

    bs, K, D = means.shape

    # Sample component indices: (bs,)
    probs = logweights.softmax(dim=-1)
    indices = torch.multinomial(probs, 1).squeeze(-1)  # (bs,)

    # Gather selected means: (bs, D)
    selected_means = means[torch.arange(bs, device=means.device), indices]

    # Sample: mu_k + sigma * eps
    stds = logstds.exp()  # (bs,)
    noise = torch.randn_like(selected_means)
    samples = selected_means + stds[:, None] * noise

    return samples


# =============================================================================
# GMM Reverse Transition (for transition loss)
# =============================================================================


def gmm_reverse_transition(
    gm: dict[str, Tensor],
    x_t: Tensor,
    t_from: Tensor,
    t_to: Tensor,
    prediction_type: str = "velocity",
    eps: float = 1e-8,
) -> dict[str, Tensor]:
    """Analytically transition a GMM from ``t_from`` to ``t_to``.

    Given a GMM predicted at the noisier level ``t_from``, computes the
    implied GMM at the cleaner level ``t_to`` using the CondOT transition
    kernel.  This is the core operation of GMFlow's transition loss.

    CondOT convention: ``x_t = t * x_1 + (1-t) * x_0``, so
    ``alpha_t = t`` (signal) and ``sigma_t = 1-t`` (noise).

    Args:
        gm: dict with ``means`` (bs, K, D), ``logweights`` (bs, K),
            ``logstds`` (bs,).
        x_t: ``(bs, D)`` the sample at ``t_from``.
        t_from: ``(bs,)`` noisy timestep (lower t, more noise).
        t_to: ``(bs,)`` cleaner timestep (higher t, less noise).
        prediction_type: ``"velocity"`` or ``"x_prediction"``.
        eps: numerical stability.

    Returns:
        New GMM dict at ``t_to`` with transformed means and logstds.
        Logweights are passed through unchanged.
    """
    means = gm["means"]  # (bs, K, D)
    logstds = gm["logstds"]  # (bs,)

    # Reshape timesteps for broadcasting: (bs, 1, 1)
    t_from = t_from.view(-1, 1, 1)
    t_to = t_to.view(-1, 1, 1)

    alpha_from = t_from.clamp(min=eps)  # signal at noisy level
    alpha_to = t_to  # signal at clean level
    sigma_from = (1 - t_from).clamp(min=eps)  # noise at noisy level
    sigma_to = (1 - t_to).clamp(min=eps)  # noise at clean level

    # Transition kernel coefficients
    sigma_ratio = sigma_to / sigma_from
    alpha_ratio = alpha_from / alpha_to
    beta_sigma_sq = (1 - (sigma_ratio * alpha_ratio) ** 2).clamp(min=0)

    c1 = sigma_ratio ** 2 * alpha_ratio  # weight for x_t
    c2 = beta_sigma_sq * alpha_to  # weight for x_1 estimate
    c3 = (beta_sigma_sq * sigma_to ** 2).clamp(min=eps)  # noise floor

    # Convert GMM means to x_1 estimates
    x_t_exp = x_t.unsqueeze(-2)  # (bs, 1, D)
    if prediction_type == "x_prediction":
        x1_est = means  # means directly predict x_1
        std_scale = c2  # x_1 std maps through c2
    else:
        # velocity: v_i predicts dx_t = x_1 - x_0
        # x_1 = x_t + sigma_from * v_i  (for CondOT: x_1 = x_t + (1-t)*v)
        x1_est = x_t_exp + sigma_from * means
        std_scale = sigma_from * c2  # velocity std maps through sigma*c2

    # Reversed means
    means_clean = c1 * x_t_exp + c2 * x1_est

    # Reversed logstds via logaddexp (numerically stable)
    logstds_exp = logstds.view(-1, 1, 1)  # (bs, 1, 1)
    log_var_pred = 2 * (logstds_exp + torch.log(std_scale.abs().clamp(min=eps)))
    log_var_floor = torch.log(c3)
    logstds_clean = 0.5 * torch.logaddexp(log_var_pred, log_var_floor)
    # Collapse to (bs,) since shared across components and dimensions
    logstds_clean = logstds_clean.view(-1)

    return {
        "means": means_clean,
        "logweights": gm["logweights"],
        "logstds": logstds_clean,
    }
