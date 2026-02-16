"""
Random walk return probability features for graph nodes.

Computes the diagonal of (D^{-1}A)^k where D is the degree matrix and A is
the adjacency matrix, giving the probability that a random walk starting at
node i returns to node i after k steps. These probabilities capture global
structural information (ring membership, centrality, bridge nodes) that local
message passing may miss.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# Precomputed quantile bin boundaries for ZINC molecular graphs.
# Outer key: num_bins.  Inner key: k-value.  Value: sorted boundary list
# of length ``num_bins - 1`` (equal-frequency quantile splits).
# Computed from 5 000 ZINC training molecules (~116 k atoms).
# Note: odd k values are near-degenerate on heavy-atom molecular graphs
# (bipartite-like structure -> odd-step return probabilities ~ 0).
# ---------------------------------------------------------------------------
_ZINC_RW_QUANTILE_BOUNDARIES: dict[int, dict[int, list[float]]] = {
    3: {
        2: [0.416667, 0.444444],
        3: [0.0, 0.0],
        4: [0.261574, 0.333333],
        5: [0.0, 0.0],
        6: [0.211934, 0.268519],
        7: [0.0, 0.0],
        8: [0.182436, 0.236336],
        9: [0.0, 0.000772],
        10: [0.161278, 0.215864],
        11: [0.0, 0.002165],
        12: [0.146152, 0.199449],
        13: [0.0, 0.00397],
        14: [0.134964, 0.18818],
        15: [0.0, 0.005945],
        16: [0.125639, 0.178501],
    },
    4: {
        2: [0.333333, 0.416667, 0.5],
        3: [0.0, 0.0, 0.0],
        4: [0.255208, 0.289352, 0.354167],
        5: [0.0, 0.0, 0.0],
        6: [0.199846, 0.238426, 0.287616],
        7: [0.0, 0.0, 0.003086],
        8: [0.170034, 0.206028, 0.253515],
        9: [0.0, 0.0, 0.007802],
        10: [0.149727, 0.185268, 0.231642],
        11: [0.0, 0.0, 0.012478],
        12: [0.133689, 0.170127, 0.21738],
        13: [0.0, 1e-05, 0.016577],
        14: [0.122111, 0.158143, 0.205156],
        15: [0.0, 4.5e-05, 0.020004],
        16: [0.113759, 0.148607, 0.195082],
    },
    5: {
        2: [0.333333, 0.416667, 0.444444, 0.5],
        3: [0.0, 0.0, 0.0, 0.0],
        4: [0.240741, 0.273148, 0.310185, 0.361111],
        5: [0.0, 0.0, 0.0, 0.0],
        6: [0.189108, 0.220486, 0.252915, 0.302083],
        7: [0.0, 0.0, 0.0, 0.010417],
        8: [0.159722, 0.191422, 0.222312, 0.271235],
        9: [0.0, 0.0, 0.0, 0.01929],
        10: [0.139563, 0.170497, 0.202565, 0.24619],
        11: [0.0, 0.0, 0.000193, 0.026792],
        12: [0.126104, 0.155842, 0.188097, 0.229485],
        13: [0.0, 0.0, 0.000641, 0.031937],
        14: [0.114862, 0.143742, 0.176051, 0.216373],
        15: [0.0, 0.0, 0.001309, 0.036034],
        16: [0.105988, 0.134154, 0.166114, 0.206725],
    },
    6: {
        2: [0.333333, 0.416667, 0.416667, 0.444444, 0.5],
        3: [0.0, 0.0, 0.0, 0.0, 0.0],
        4: [0.222222, 0.261574, 0.289352, 0.333333, 0.375],
        5: [0.0, 0.0, 0.0, 0.0, 0.018519],
        6: [0.179355, 0.211934, 0.238426, 0.268519, 0.317708],
        7: [0.0, 0.0, 0.0, 0.0, 0.028292],
        8: [0.151299, 0.182436, 0.206028, 0.236336, 0.279191],
        9: [0.0, 0.0, 0.0, 0.000772, 0.03628],
        10: [0.131753, 0.161278, 0.185268, 0.215864, 0.25667],
        11: [0.0, 0.0, 0.0, 0.002165, 0.041505],
        12: [0.118311, 0.146152, 0.170127, 0.199449, 0.237798],
        13: [0.0, 0.0, 1e-05, 0.00397, 0.045406],
        14: [0.107878, 0.134964, 0.158143, 0.18818, 0.224247],
        15: [0.0, 0.0, 4.5e-05, 0.005945, 0.048552],
        16: [0.09981, 0.125639, 0.148607, 0.178501, 0.213473],
    },
}

# Backward-compatible alias (4-bin preset)
ZINC_RW_QUANTILE_BOUNDARIES: dict[int, list[float]] = _ZINC_RW_QUANTILE_BOUNDARIES[4]


def get_zinc_rw_boundaries(num_bins: int) -> dict[int, list[float]]:
    """Return precomputed ZINC quantile bin boundaries for a given bin count.

    Available bin counts: 3, 4, 5, 6.  Each returned dict maps
    ``k`` (2..16) to a sorted list of ``num_bins - 1`` boundary values.

    Parameters
    ----------
    num_bins : int
        Number of bins (3, 4, 5, or 6).

    Returns
    -------
    dict[int, list[float]]
        Mapping from k-value to boundary list, suitable for passing as
        ``bin_boundaries`` to :func:`bin_rw_probabilities`.

    Raises
    ------
    ValueError
        If *num_bins* is not one of the precomputed values.
    """
    if num_bins not in _ZINC_RW_QUANTILE_BOUNDARIES:
        available = sorted(_ZINC_RW_QUANTILE_BOUNDARIES.keys())
        raise ValueError(
            f"No precomputed boundaries for num_bins={num_bins}. "
            f"Available: {available}"
        )
    return _ZINC_RW_QUANTILE_BOUNDARIES[num_bins]


def compute_rw_return_probabilities(
    edge_index: Tensor,
    num_nodes: int,
    k_values: tuple[int, ...] = (3, 6),
) -> Tensor:
    """
    Compute random walk return probabilities for each node.

    For each node i and step count k, computes ``[T^k]_{ii}`` where
    ``T = D^{-1} A`` is the row-stochastic transition matrix.

    Parameters
    ----------
    edge_index : Tensor
        Edge index of shape ``[2, E]`` (may be uni- or bi-directional).
    num_nodes : int
        Number of nodes in the graph.
    k_values : tuple of int
        Steps at which to compute return probabilities.

    Returns
    -------
    Tensor
        Shape ``[num_nodes, len(k_values)]`` with return probabilities
        in ``[0, 1]``.
    """
    device = edge_index.device

    # Build dense adjacency matrix
    A = torch.zeros(num_nodes, num_nodes, dtype=torch.float64, device=device)
    row, col = edge_index[0], edge_index[1]
    A[row, col] = 1.0
    # Symmetrise (handles both uni- and bi-directional input)
    A = torch.clamp(A + A.T, max=1.0)

    # Degree and transition matrix
    deg = A.sum(dim=1)
    inv_deg = torch.zeros_like(deg)
    nonzero = deg > 0
    inv_deg[nonzero] = 1.0 / deg[nonzero]

    T = A * inv_deg.unsqueeze(1)  # row i scaled by 1/deg(i)

    # Isolated nodes: self-loop with probability 1
    isolated = ~nonzero
    T[isolated, isolated] = 1.0

    # Compute T^k for each k and extract diagonal
    results = []
    for k in k_values:
        Tk = torch.linalg.matrix_power(T, k)
        results.append(torch.diag(Tk))

    return torch.stack(results, dim=-1).float().cpu()


def bin_rw_probabilities(
    rw_probs: Tensor,
    num_bins: int = 10,
    bin_boundaries: dict[int, list[float]] | None = None,
    k_values: tuple[int, ...] | None = None,
) -> Tensor:
    """
    Discretise RW return probabilities into bin indices.

    When *bin_boundaries* is ``None`` (default), uses uniform bins on
    ``[0, 1]`` for backward compatibility.  When provided, uses
    ``torch.bucketize`` with per-k boundary vectors for data-driven
    (e.g. quantile-based) binning.

    Parameters
    ----------
    rw_probs : Tensor
        Raw return probabilities of shape ``[N, F]`` in ``[0, 1]``.
    num_bins : int
        Number of bins.  For uniform mode this controls the bin width;
        for quantile mode it is only used to clamp the output range.
    bin_boundaries : dict mapping k → list of floats, optional
        Per-step boundary vectors.  Each list has ``num_bins - 1``
        sorted thresholds.  Use :data:`ZINC_RW_QUANTILE_BOUNDARIES`
        for a ready-made preset.
    k_values : tuple of int, optional
        The k-values corresponding to each column of *rw_probs*.
        Required when *bin_boundaries* is not ``None``.

    Returns
    -------
    Tensor
        Integer bin indices (as float, matching ``data.x`` convention)
        of shape ``[N, F]`` with values in ``{0, ..., num_bins - 1}``.
    """
    if bin_boundaries is None:
        # Uniform binning on [0, 1] (original behaviour)
        return (rw_probs * num_bins).long().clamp(0, num_bins - 1).float()

    if k_values is None:
        raise ValueError("k_values is required when bin_boundaries is provided")

    result = torch.empty_like(rw_probs)
    for col_idx, k in enumerate(k_values):
        boundaries = bin_boundaries.get(k)
        if boundaries is None:
            # Fall back to uniform for any k without precomputed boundaries
            result[:, col_idx] = (
                (rw_probs[:, col_idx] * num_bins).long().clamp(0, num_bins - 1).float()
            )
        else:
            edges = torch.tensor(boundaries, dtype=rw_probs.dtype)
            result[:, col_idx] = torch.bucketize(
                rw_probs[:, col_idx].contiguous(), edges, right=True,
            ).clamp(0, num_bins - 1).float()

    return result


def augment_data_with_rw(
    data: Data,
    k_values: tuple[int, ...] = (3, 6),
    num_bins: int = 10,
    bin_boundaries: dict[int, list[float]] | None = None,
) -> Data:
    """
    Augment a PyG Data object with binned RW return probability features.

    Appends ``len(k_values)`` new columns to ``data.x``.

    Parameters
    ----------
    data : Data
        PyG data with ``x`` of shape ``[N, F]`` and ``edge_index`` of
        shape ``[2, E]``.
    k_values : tuple of int
        Random walk steps to compute.
    num_bins : int
        Number of bins per RW feature.
    bin_boundaries : dict mapping k → list of floats, optional
        Per-step quantile boundaries.  See :func:`bin_rw_probabilities`.

    Returns
    -------
    Data
        The same ``data`` object with ``x`` expanded to
        ``[N, F + len(k_values)]``.
    """
    rw_probs = compute_rw_return_probabilities(
        data.edge_index, data.x.size(0), k_values
    )
    rw_binned = bin_rw_probabilities(
        rw_probs, num_bins,
        bin_boundaries=bin_boundaries,
        k_values=k_values,
    )
    data.x = torch.cat([data.x, rw_binned.to(data.x.device)], dim=-1)
    return data
