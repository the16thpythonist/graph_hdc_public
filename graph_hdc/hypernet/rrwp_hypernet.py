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
    """
    HyperNet variant that uses RRWP-augmented node features for order-0
    terms but base-only features for message passing.

    Maintains two node codebooks:

    - ``nodes_codebook`` (inherited): base features only, used for message
      passing, edge_terms, and graph_embedding.
    - ``nodes_codebook_full``: base + RRWP features, used exclusively for
      node_terms (order-0 readout) and order-0 decoding.

    Parameters
    ----------
    config : DSHDCConfig
        Configuration with ``rw_config.enabled=True``. The node feature
        bins must include the RRWP dimensions (as produced by
        ``create_config_with_rw``).
    depth : int, optional
        Override message passing depth from config.
    observed_node_features : set[tuple], optional
        Full feature tuples (base + RRWP) observed in the dataset.
        Used for codebook pruning. Base tuples are derived automatically.
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
        """Create a config with only base features (no RRWP)."""
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
        """Build the full (base + RRWP) codebook and indexer."""
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

    def _limit_full_codebook(self, node_features: set[tuple]) -> None:
        """Prune the full codebook to only observed feature tuples."""
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

        Sets ``data.node_hv`` (base features only, for message passing)
        and ``data.node_hv_full`` (base + RRWP, for order-0 readout).
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
        """
        Encode graphs with RRWP-enriched order-0 terms.

        Message passing (edge_terms, graph_embedding) uses base features
        only. The node_terms output uses the full (base + RRWP) codebook.

        Parameters and return format are identical to
        :meth:`HyperNet.forward`.
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

    @property
    def order_zero_codebook(self):
        """Codebook used for order-0 (node_terms) decoding (full: base + RRWP)."""
        return self.nodes_codebook_full

    # ─────────────────────────── Decoding ───────────────────────────

    def decode_order_zero(self, embedding: torch.Tensor) -> torch.Tensor:
        """Decode node types and counts from embedding using the full codebook."""
        d = torchhd.dot(embedding, self.nodes_codebook_full)
        if self.vsa in {VSAModel.FHRR, VSAModel.MAP}:
            d = d / self.hv_dim
        return torch.round(d).int().clamp(min=0)

    def decode_order_zero_iterative(
        self, embedding: torch.Tensor, debug: bool = False
    ) -> list[tuple[int, ...]] | tuple[list[tuple[int, ...]], list[float], list[float]]:
        """Decode node types via iterative unbinding using the full codebook."""
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
        """Decode node counts as Counter dict using the full codebook."""
        dot_products_rounded = self.decode_order_zero(embedding)
        return self._convert_to_counter(
            similarities=dot_products_rounded, indexer=self.nodes_indexer_full
        )

    def decode_order_zero_counter_iterative(
        self, embedding: torch.Tensor
    ) -> dict[int, Counter]:
        """Decode node counts via iterative unbinding using the full codebook."""
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
        """Save RRWPHyperNet to file (base + full codebook)."""
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
        """Load RRWPHyperNet from file.

        Parameters
        ----------
        path : str | Path
            Path to saved RRWPHyperNet checkpoint.
        device : str | torch.device
            Device to load to.

        Returns
        -------
        RRWPHyperNet
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
