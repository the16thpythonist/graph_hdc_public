"""
RRWPHyperNet: HyperNet variant with split RRWP encoding.

Uses RRWP-enriched node features only for order-0 (node_terms) readout,
while message passing operates on base features only. This prevents
positional information from interfering with structural binding.
"""

from __future__ import annotations

import math
from collections import Counter, OrderedDict, defaultdict
from dataclasses import replace
from pathlib import Path

import pytorch_lightning as pl
import torch
import torchhd
from torch import Tensor
from torch_geometric.data import Batch, Data

from graph_hdc.hypernet.configs import (
    DSHDCConfig,
    FeatureConfig,
    Features,
    RWConfig,
    _BASE_BINS,
)
from graph_hdc.hypernet.encoder import HyperNet
from graph_hdc.hypernet.feature_encoders import CombinatoricIntegerEncoder
from graph_hdc.hypernet.types import VSAModel
from graph_hdc.utils.helpers import TupleIndexer, cartesian_bind_tensor, scatter_hd


class RRWPHyperNet(HyperNet):
    """HyperNet variant with Random Walk Return Probability (RRWP) features.

    RRWP features capture global structural information — ring membership,
    node centrality, bridge positions — that local message passing may miss.
    Each node is annotated with the probability that a random walk of length
    *k* returns to its starting position, computed for one or more values of
    *k*.  These continuous probabilities are discretised into bin indices and
    appended to the base node feature vector before codebook lookup.  See
    :mod:`graph_hdc.utils.rw_features` for the computation and binning
    details.

    Split-codebook architecture
    ---------------------------
    To prevent positional/structural RRWP information from leaking into the
    message-passing binding operations, this class maintains **two separate
    node codebooks**:

    ``nodes_codebook`` (inherited from :class:`HyperNet`)
        Maps **base features only** (atom type, degree, charge, …) to
        hypervectors.  Used for all message-passing rounds, producing
        ``edge_terms`` and ``graph_embedding``.

    ``nodes_codebook_full``
        Maps the **full feature vector** (base + RRWP bin indices) to
        hypervectors.  Used exclusively for the order-0 ``node_terms``
        readout and for order-0 decoding.

    During :meth:`forward`, message passing runs entirely on the base
    codebook.  Afterwards, ``node_terms`` is recomputed by bundling the
    full-codebook hypervectors (``node_hv_full``), giving the order-0
    embedding access to RRWP-enriched representations without affecting
    edge-level structural binding.

    Data flow
    ---------
    1. RRWP columns are appended to ``data.x`` *before* encoding (typically
       by :func:`~graph_hdc.datasets.utils.encode_dataset` or
       :func:`~graph_hdc.utils.rw_features.augment_data_with_rw`).
    2. :meth:`encode_properties` performs two codebook lookups on the
       augmented ``data.x``:

       - base features (first *F* columns) → ``data.node_hv``
       - full features (first *F + R* columns) → ``data.node_hv_full``

    3. :meth:`forward` delegates message passing to the parent (using
       ``node_hv``), then replaces ``node_terms`` with a bundle of
       ``node_hv_full``.

    Limitations
    -----------
    Edge-level and full-graph decoding are **not supported** because the
    RRWP-enriched codebook is only meaningful at the node level.  Use the
    base :class:`HyperNet` for those tasks.

    Parameters
    ----------
    config : DSHDCConfig
        Configuration with ``rw_config.enabled=True``.  The node feature
        bins must include the RRWP dimensions (as produced by
        ``create_config_with_rw``).
    depth : int, optional
        Override message-passing depth from config.
    observed_node_features : set[tuple], optional
        Full feature tuples (base + RRWP) observed in the dataset.
        Used for codebook pruning.  Base-only tuples are derived
        automatically by stripping the trailing RRWP dimensions.
    """

    def __init__(
        self,
        config: DSHDCConfig,
        depth: int | None = None,
        observed_node_features: set[tuple] | None = None,
    ):
        if not config.rw_config.enabled:
            raise ValueError(
                "RRWPHyperNet requires rw_config.enabled=True. "
                "Use HyperNet for configs without RW features."
            )

        self._num_rw_dims = len(config.rw_config.k_values)

        # Extract full bins from config
        node_fc = next(iter(config.node_feature_configs.values()))
        full_bins = list(node_fc.bins)
        base_bins = full_bins[: -self._num_rw_dims]
        self._full_bins = full_bins
        self._full_feature_dim = len(full_bins)

        # Derive base config (strip RW features)
        base_config = self._derive_base_config(config, base_bins)

        # Derive base observed features (strip RW dims from full tuples)
        base_observed = None
        if observed_node_features is not None:
            base_observed = {t[: -self._num_rw_dims] for t in observed_node_features}

        # Initialize parent with base-only config
        super().__init__(base_config, depth=depth, observed_node_features=base_observed)

        # Restore the original (enabled) RW config
        self.rw_config = config.rw_config

        # Build the full codebook (base + RRWP)
        self._build_full_codebook(
            config=config,
            full_bins=full_bins,
            observed_node_features=observed_node_features,
        )

    @staticmethod
    def _derive_base_config(config: DSHDCConfig, base_bins: list[int]) -> DSHDCConfig:
        """Create a copy of *config* with RRWP features stripped out.

        Replaces the node feature config with one that uses only *base_bins*
        and sets ``rw_config.enabled=False``.  All other config fields
        (hv_dim, vsa, edge features, …) are preserved.

        Parameters
        ----------
        config : DSHDCConfig
            Original config whose node bins include RRWP dimensions.
        base_bins : list[int]
            Bin sizes for the base (non-RRWP) node features only.

        Returns
        -------
        DSHDCConfig
            A shallow copy of *config* suitable for initialising the parent
            :class:`HyperNet` without any RRWP awareness.
        """
        base_node_fc = FeatureConfig(
            count=math.prod(base_bins),
            encoder_cls=CombinatoricIntegerEncoder,
            index_range=(0, len(base_bins)),
            bins=base_bins,
        )
        return replace(
            config,
            node_feature_configs=OrderedDict(
                [(Features.NODE_FEATURES, base_node_fc)]
            ),
            rw_config=RWConfig(enabled=False),
        )

    def _build_full_codebook(
        self,
        config: DSHDCConfig,
        full_bins: list[int],
        observed_node_features: set[tuple] | None,
    ) -> None:
        """Build the full (base + RRWP) node codebook and its indexer.

        Creates a :class:`CombinatoricIntegerEncoder` whose cartesian
        product space spans all *full_bins* dimensions (base features
        followed by RRWP bin indices).  The resulting codebook and
        :class:`TupleIndexer` are stored as ``nodes_codebook_full`` and
        ``nodes_indexer_full``.

        A deterministic but independent random seed (``self.seed + 1000``)
        is used so that the full codebook does not share hypervectors with
        the base codebook.

        If codebook pruning is enabled and *observed_node_features* is
        provided, the codebook is immediately pruned to contain only
        hypervectors for observed feature combinations.

        Parameters
        ----------
        config : DSHDCConfig
            Original (RRWP-enabled) config, used for device selection.
        full_bins : list[int]
            Bin sizes for all feature dimensions (base + RRWP).
        observed_node_features : set[tuple] or None
            Full feature tuples seen in the dataset.  When provided
            together with ``self.prune_codebook=True``, the codebook is
            pruned to these entries only.
        """
        device = torch.device(config.device)

        # Use a derived seed for the full codebook to keep it independent
        # but deterministic
        if self.seed is not None:
            torch.manual_seed(self.seed + 1000)

        full_encoder = CombinatoricIntegerEncoder(
            num_categories=math.prod(full_bins),
            dim=self.hv_dim,
            vsa=self.vsa.value,
            idx_offset=0,
            device=device,
            indexer=TupleIndexer(sizes=full_bins),
            dtype="float64" if self._dtype == torch.float64 else "float32",
        )

        self.nodes_codebook_full = full_encoder.codebook
        self.nodes_indexer_full = full_encoder.indexer

        # Prune full codebook to observed features
        if self.prune_codebook and observed_node_features is not None:
            self._limit_full_codebook(observed_node_features)

    def rebuild_unpruned_codebook(self) -> None:
        """Rebuild both base and full (base + RRWP) codebooks without pruning.

        Re-generates the complete cartesian-product codebooks from scratch
        using the original seeds, discarding any prior pruning.  This is
        useful when the encoder needs to handle feature tuples that were
        not in the original ``observed_node_features`` set (e.g. after
        dataset expansion or during inference on unseen molecules).

        The base codebook is rebuilt by the parent; the full codebook is
        rebuilt here using the same derived seed (``self.seed + 1000``) as
        :meth:`_build_full_codebook` for reproducibility.
        """
        super().rebuild_unpruned_codebook()

        device = self.nodes_codebook_full.device
        dtype_str = "float64" if self._dtype == torch.float64 else "float32"

        # Use the same derived seed as _build_full_codebook
        if self.seed is not None:
            torch.manual_seed(self.seed + 1000)

        full_encoder = CombinatoricIntegerEncoder(
            num_categories=math.prod(self._full_bins),
            dim=self.hv_dim,
            vsa=self.vsa.value,
            idx_offset=0,
            device=device,
            indexer=TupleIndexer(sizes=self._full_bins),
            dtype=dtype_str,
        )

        self.nodes_codebook_full = full_encoder.codebook
        self.nodes_indexer_full = full_encoder.indexer

    def _limit_full_codebook(self, node_features: set[tuple]) -> None:
        """Prune the full codebook to only observed feature tuples.

        Replaces ``nodes_codebook_full`` with a compact version containing
        only entries that correspond to *node_features*, and rebuilds the
        ``nodes_indexer_full`` mapping so that indices 0..N-1 map to the
        sorted observed tuples.  This reduces memory usage and speeds up
        dot-product decoding when the actual feature space is much smaller
        than the full cartesian product.

        Parameters
        ----------
        node_features : set[tuple]
            Full (base + RRWP) feature tuples observed in the dataset.
        """
        sorted_features = sorted(node_features)
        idxs = self.nodes_indexer_full.get_idxs(sorted_features)
        self.nodes_indexer_full.idx_to_tuple = sorted_features
        self.nodes_indexer_full.tuple_to_idx = {
            tup: idx for idx, tup in enumerate(sorted_features)
        }
        self.nodes_codebook_full = self.nodes_codebook_full[idxs].as_subclass(
            type(self.nodes_codebook_full)
        )

    # ─────────────────────────── Encoding ───────────────────────────

    def encode_properties(self, data: Data) -> Data:
        """Encode node properties into both base and full hypervectors.

        Delegates to the parent to produce ``data.node_hv`` from base
        features, then performs a second codebook lookup on the full
        feature vector (base + RRWP columns of ``data.x``) to produce
        ``data.node_hv_full``.

        The full lookup converts each node's feature row to an integer
        tuple, maps it through ``nodes_indexer_full`` to a codebook index,
        and retrieves the corresponding hypervector from
        ``nodes_codebook_full``.

        Parameters
        ----------
        data : Data
            PyG data object whose ``data.x`` has shape ``[N, F+R]`` where
            *F* is the number of base feature columns and *R* is the
            number of RRWP columns (``len(rw_config.k_values)``).

        Returns
        -------
        Data
            The same *data* object, mutated in-place with two new
            attributes:

            - ``node_hv`` — shape ``[N, D]``, base-only hypervectors.
            - ``node_hv_full`` — shape ``[N, D]``, full (base + RRWP)
              hypervectors.
        """
        # Parent encodes base features → data.node_hv
        data = super().encode_properties(data)

        # Encode full features (base + RRWP) via direct codebook lookup
        full_feat = data.x[:, : self._full_feature_dim].long()
        tuples = list(map(tuple, full_feat.tolist()))
        idxs = self.nodes_indexer_full.get_idxs(tuples)
        idxs_tensor = torch.tensor(
            idxs, dtype=torch.long, device=self.nodes_codebook_full.device
        )
        data.node_hv_full = self.nodes_codebook_full[idxs_tensor]

        return data

    # ─────────────────────────── Forward ────────────────────────────

    def forward(
        self,
        data: Data | Batch,
        *,
        bidirectional: bool = False,
        normalize: bool | None = None,
    ) -> dict[str, Tensor]:
        """Encode graphs, using RRWP-enriched hypervectors for order-0 terms.

        Execution proceeds in two stages:

        1. **Parent forward pass** — calls :meth:`HyperNet.forward`, which
           internally invokes :meth:`encode_properties` (setting both
           ``node_hv`` and ``node_hv_full``), runs VSA message passing on
           the *base* hypervectors, and produces ``edge_terms``,
           ``node_terms``, and ``graph_embedding``.
        2. **Order-0 replacement** — the ``node_terms`` entry from step 1
           is discarded and recomputed by bundling ``node_hv_full``
           (RRWP-enriched) per graph via :func:`scatter_hd`.

        This ensures that structural binding (edges, graph embedding) is
        free of positional RRWP information, while the order-0 readout
        benefits from it.

        Parameters
        ----------
        data : Data | Batch
            Single graph or batched graphs.  Must already have RRWP
            columns appended to ``data.x`` (see
            :func:`~graph_hdc.utils.rw_features.augment_data_with_rw`).
        bidirectional : bool, optional
            If ``True``, encode edges in both directions.
        normalize : bool or None, optional
            Override the instance-level normalisation setting.

        Returns
        -------
        dict[str, Tensor]
            Same keys as :meth:`HyperNet.forward`:

            - ``"node_terms"`` — RRWP-enriched order-0 embedding ``[B, D]``
            - ``"edge_terms"`` — structural edge embedding ``[B, D]``
            - ``"graph_embedding"`` — full graph embedding ``[B, D]``
        """
        # Parent forward: message passing with base HVs
        # (calls our encode_properties, which sets both node_hv and node_hv_full)
        result = super().forward(
            data, bidirectional=bidirectional, normalize=normalize
        )

        # Ensure full HVs are on the right device
        if data.node_hv_full.device != data.node_hv.device:
            data.node_hv_full = data.node_hv_full.to(data.node_hv.device)

        # Recompute node_terms using full (RRWP-enriched) HVs
        result["node_terms"] = scatter_hd(
            src=data.node_hv_full, index=data.batch, op="bundle"
        )

        return result

    def __str__(self) -> str:
        lines = [
            f"{type(self).__name__}(",
            f"  vsa={self.vsa.value}, hv_dim={self.hv_dim}, depth={self.depth},",
            f"  dataset={self.base_dataset}, normalize={self.normalize},",
            f"  dtype={self._dtype},",
            f"  nodes_codebook (base): {self.nodes_codebook.shape[0]} entries x {self.nodes_codebook.shape[1]} dim,",
        ]
        if hasattr(self, "nodes_indexer") and hasattr(self.nodes_indexer, "sizes"):
            lines.append(f"  base_feature_bins={self.nodes_indexer.sizes},")
        lines.append(
            f"  nodes_codebook_full (base+RRWP): {self.nodes_codebook_full.shape[0]} entries x {self.nodes_codebook_full.shape[1]} dim,"
        )
        lines.append(f"  full_feature_bins={self._full_bins},")
        rw = self.rw_config
        lines.append(
            f"  rw_config: k_values={rw.k_values}, num_bins={rw.num_bins},"
        )
        if hasattr(self, "edge_feature_codebook") and self.edge_feature_codebook is not None:
            lines.append(
                f"  edge_codebook: {self.edge_feature_codebook.shape[0]} entries x {self.edge_feature_codebook.shape[1]} dim,"
            )
        else:
            lines.append("  edge_codebook: not initialized,")
        lines.append(f"  prune_codebook={self.prune_codebook},")
        lines.append(")")
        return "\n".join(lines)

    @property
    def order_zero_codebook(self):
        """Codebook used for order-0 (node_terms) decoding (full: base + RRWP)."""
        return self.nodes_codebook_full

    # ─────────────────────────── Decoding ───────────────────────────

    def decode_order_zero(self, embedding: torch.Tensor) -> torch.Tensor:
        """Decode node type counts from an order-0 embedding via dot product.

        Computes the dot product of *embedding* against every entry in
        ``nodes_codebook_full``, rounds to the nearest integer, and clamps
        to non-negative values.  Each resulting value approximates how
        many times the corresponding (base + RRWP) feature tuple appears
        in the graph that produced the embedding.

        For FHRR and MAP VSA models the raw dot products are normalised
        by ``hv_dim`` before rounding.

        Parameters
        ----------
        embedding : Tensor
            Order-0 (node_terms) hypervector, shape ``[D]`` or ``[B, D]``.

        Returns
        -------
        Tensor
            Integer counts, shape ``[C]`` or ``[B, C]`` where *C* is the
            size of ``nodes_codebook_full``.
        """
        d = torchhd.dot(embedding, self.nodes_codebook_full)
        if self.vsa in {VSAModel.FHRR, VSAModel.MAP}:
            d = d / self.hv_dim
        return torch.round(d).int().clamp(min=0)

    def decode_order_zero_iterative(
        self, embedding: torch.Tensor, debug: bool = False
    ) -> list[tuple[int, ...]] | tuple[list[tuple[int, ...]], list[float], list[float]]:
        """Decode node types by iteratively subtracting codebook entries.

        Starting from the full order-0 embedding, this method repeatedly:

        1. Finds the ``nodes_codebook_full`` entry with highest cosine
           similarity to the current residual.
        2. Subtracts that entry from the residual.
        3. Records the decoded feature tuple.

        Iteration stops when any of these conditions is met:

        - The residual norm drops below a small threshold (``eps``).
        - Subtracting an entry *increases* the residual norm (the
          contribution is no longer constructive).
        - The dataset-specific maximum node count is reached.

        This is more robust than the dot-product method
        (:meth:`decode_order_zero`) for noisy or generated embeddings,
        since it does not rely on integer-rounded counts.

        Parameters
        ----------
        embedding : Tensor
            A single order-0 hypervector, shape ``[D]``.
        debug : bool, optional
            If ``True``, return additional diagnostics.

        Returns
        -------
        list[tuple[int, ...]]
            Decoded node feature tuples (base + RRWP), one per decoded
            node.  Returned when ``debug=False``.
        tuple[list[tuple], list[float], list[float]]
            ``(decoded_nodes, norms_history, similarities)`` when
            ``debug=True``.  *norms_history* tracks the residual norm
            before each iteration; *similarities* records the peak cosine
            similarity at each step.
        """
        if not hasattr(self, "_min_node_delta_full"):
            norms = [hv.norm().item() for hv in self.nodes_codebook_full]
            self._min_node_delta_full = min(norms) if norms else 1.0

        eps = self._min_node_delta_full * 0.01

        from graph_hdc.hypernet.encoder import (
            MAX_ALLOWED_DECODING_NODES_QM9,
            MAX_ALLOWED_DECODING_NODES_ZINC,
        )

        max_nodes = (
            MAX_ALLOWED_DECODING_NODES_QM9
            if self.base_dataset == "qm9"
            else MAX_ALLOWED_DECODING_NODES_ZINC
        )

        decoded_nodes: list[tuple[int, ...]] = []
        norms_history: list[float] = []
        similarities: list[float] = []
        residual = embedding.clone()
        prev_norm = residual.norm().item()

        while len(decoded_nodes) < max_nodes:
            curr_norm = residual.norm().item()
            norms_history.append(curr_norm)

            if curr_norm <= eps:
                break

            sims = torchhd.cos(residual, self.nodes_codebook_full)
            idx_max = torch.argmax(sims).item()
            max_sim = sims[idx_max].item()
            similarities.append(max_sim)

            hv_found = self.nodes_codebook_full[idx_max]
            residual = residual - hv_found

            new_norm = residual.norm().item()
            if new_norm > prev_norm:
                break
            prev_norm = new_norm

            node_tuple = self.nodes_indexer_full.get_tuple(idx_max)
            decoded_nodes.append(node_tuple)

        if debug:
            return decoded_nodes, norms_history, similarities
        return decoded_nodes

    def decode_order_zero_counter(
        self, embedding: torch.Tensor
    ) -> dict[int, Counter]:
        """Decode node feature counts as a Counter via dot-product decoding.

        Wraps :meth:`decode_order_zero` and converts the integer count
        vector into a ``{batch_idx: Counter}`` mapping, where each
        :class:`~collections.Counter` maps feature tuples to their
        predicted multiplicity.

        Parameters
        ----------
        embedding : Tensor
            Order-0 hypervector(s), shape ``[D]`` or ``[B, D]``.

        Returns
        -------
        dict[int, Counter]
            Mapping from batch index to a Counter of ``(base + RRWP)``
            feature tuples.
        """
        dot_products_rounded = self.decode_order_zero(embedding)
        return self._convert_to_counter(
            similarities=dot_products_rounded, indexer=self.nodes_indexer_full
        )

    def decode_order_zero_counter_iterative(
        self, embedding: torch.Tensor
    ) -> dict[int, Counter]:
        """Decode node feature counts via iterative subtraction.

        Wraps :meth:`decode_order_zero_iterative` for one or more
        embeddings and aggregates the per-node results into Counter
        objects, giving the same output format as
        :meth:`decode_order_zero_counter`.

        Parameters
        ----------
        embedding : Tensor
            Order-0 hypervector(s), shape ``[D]`` or ``[B, D]``.

        Returns
        -------
        dict[int, Counter]
            Mapping from batch index to a Counter of ``(base + RRWP)``
            feature tuples.
        """
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)

        counters: dict[int, Counter] = defaultdict(Counter)
        for b_idx in range(embedding.size(0)):
            decoded_nodes = self.decode_order_zero_iterative(embedding[b_idx])
            for node_tuple in decoded_nodes:
                counters[b_idx][node_tuple] += 1

        return dict(counters)

    def decode_order_one(self, *args, **kwargs):
        raise NotImplementedError(
            "RRWPHyperNet does not support edge decoding. "
            "Use the base HyperNet for edge-level decoding."
        )

    def decode_order_one_no_node_terms(self, *args, **kwargs):
        raise NotImplementedError(
            "RRWPHyperNet does not support edge decoding. "
            "Use the base HyperNet for edge-level decoding."
        )

    def decode_graph(self, *args, **kwargs):
        raise NotImplementedError(
            "RRWPHyperNet does not support graph decoding. "
            "Use the base HyperNet for graph-level decoding."
        )

    def decode_graph_greedy(self, *args, **kwargs):
        raise NotImplementedError(
            "RRWPHyperNet does not support graph decoding. "
            "Use the base HyperNet for graph-level decoding."
        )

    # ─────────────────────────── Device ─────────────────────────────

    def to(self, device, dtype=None):
        """Move RRWPHyperNet to device, including the full codebook."""
        super().to(device, dtype)
        device = torch.device(device)
        self.nodes_codebook_full = self.nodes_codebook_full.to(
            device=device, dtype=self._dtype
        )
        return self

    # ─────────────────────────── Save/Load ──────────────────────────

    def save(self, path: str | Path) -> None:
        """Save the full RRWPHyperNet state to a checkpoint file.

        First delegates to :meth:`HyperNet.save` to write the base state,
        then re-opens the checkpoint and augments it with RRWP-specific
        data: the full codebook, full indexer, RW config, and dimensional
        metadata.  The resulting file can be loaded with :meth:`load`.

        Parameters
        ----------
        path : str | Path
            Destination file path.
        """
        # Let parent save the base state
        super().save(path)

        # Augment with RRWP-specific data
        state = torch.load(path, map_location="cpu", weights_only=False)
        state["type"] = "RRWPHyperNet"
        state["rrwp"] = {
            "num_rw_dims": self._num_rw_dims,
            "full_bins": self._full_bins,
            "full_feature_dim": self._full_feature_dim,
            "full_rw_config": {
                "enabled": self.rw_config.enabled,
                "k_values": self.rw_config.k_values,
                "num_bins": self.rw_config.num_bins,
                "bin_boundaries": self.rw_config.bin_boundaries,
                "clip_range": self.rw_config.clip_range,
            },
            "codebooks": {
                "nodes_full": self.nodes_codebook_full.cpu(),
            },
            "indexers": {
                "nodes_full": self.nodes_indexer_full.__dict__.copy(),
            },
        }
        torch.save(state, path)

    @classmethod
    def load(
        cls, path: str | Path, device: str | torch.device = "cpu"
    ) -> "RRWPHyperNet":
        """Load an RRWPHyperNet from a checkpoint saved with :meth:`save`.

        Loads the base :class:`HyperNet` state first, then upgrades the
        instance to :class:`RRWPHyperNet` by restoring the full codebook,
        full indexer, RW config, and dimensional metadata from the
        ``"rrwp"`` key in the checkpoint.

        Parameters
        ----------
        path : str | Path
            Path to a checkpoint file produced by :meth:`save`.
        device : str | torch.device, optional
            Target device for all tensors.

        Returns
        -------
        RRWPHyperNet

        Raises
        ------
        ValueError
            If the checkpoint does not contain RRWP data (i.e. was saved
            by the base :class:`HyperNet`).
        """
        state = torch.load(path, map_location="cpu", weights_only=False)
        device = torch.device(device)

        if "rrwp" not in state:
            raise ValueError(
                "Checkpoint does not contain RRWP data. "
                "Use HyperNet.load() for base checkpoints."
            )

        # Load base HyperNet state
        base_instance = HyperNet.load(path, device=device)

        # Upgrade to RRWPHyperNet
        base_instance.__class__ = cls

        # Restore RRWP-specific attributes
        rrwp = state["rrwp"]
        base_instance._num_rw_dims = rrwp["num_rw_dims"]
        base_instance._full_bins = rrwp["full_bins"]
        base_instance._full_feature_dim = rrwp["full_feature_dim"]
        base_instance.rw_config = RWConfig(**rrwp["full_rw_config"])
        base_instance.nodes_codebook_full = rrwp["codebooks"]["nodes_full"].to(device)
        base_instance.nodes_indexer_full = TupleIndexer.__new__(TupleIndexer)
        base_instance.nodes_indexer_full.__dict__.update(rrwp["indexers"]["nodes_full"])

        return base_instance
