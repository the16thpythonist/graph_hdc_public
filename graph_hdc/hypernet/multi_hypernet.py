"""
MultiHyperNet: Ensemble of HyperNet encoders for richer graph representations.

Wraps multiple independently-initialized HyperNet instances, each with a
different random codebook, and concatenates their Order N (graph_embedding)
outputs. The Order 0 embedding comes from the primary (first) HyperNet only.

Different codebook initializations produce statistically uncorrelated
embeddings, so each sub-HyperNet captures a different "perspective" on the
same molecular graph. Information lost to bundling interference in one
codebook may be preserved in another.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch_geometric.data import Batch, Data

from graph_hdc.hypernet.configs import DSHDCConfig
from graph_hdc.hypernet.encoder import HyperNet


class MultiHyperNet(pl.LightningModule):
    """
    Ensemble of HyperNet encoders providing multiple perspectives
    of the same molecular graph.

    Uses K independently-initialized HyperNets (different random seeds)
    to produce concatenated Order N embeddings. Order 0 and all decoding
    functionality delegate to the primary (first) HyperNet.

    Parameters
    ----------
    hypernets : list[HyperNet]
        List of HyperNet instances with different codebook initializations.
        All must share the same ``base_dataset``. Dimensions may differ
        (mixed-dim ensemble).

    Attributes
    ----------
    hv_dim : int
        Base hypervector dimension (from primary HyperNet).
    ensemble_graph_dim : int
        Total dimension of the concatenated graph embedding (sum of all hv_dims).
    num_hypernets : int
        Number of sub-HyperNets in the ensemble.

    Examples
    --------
    >>> from graph_hdc import get_config
    >>> config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
    >>> multi = MultiHyperNet.from_config(config, seeds=[42, 123, 456])
    >>> out = multi.forward(batch)
    >>> out["graph_embedding"].shape  # [bs, 3 * 256]
    """

    def __init__(self, hypernets: list[HyperNet]):
        super().__init__()

        if not hypernets:
            raise ValueError("Must provide at least one HyperNet.")

        base_dataset = hypernets[0].base_dataset
        for i, hn in enumerate(hypernets[1:], 1):
            if hn.base_dataset != base_dataset:
                raise ValueError(
                    f"HyperNet {i} has base_dataset='{hn.base_dataset}', "
                    f"expected '{base_dataset}' (matching primary)."
                )

        self._hypernets = hypernets
        self._primary = hypernets[0]

    def __str__(self) -> str:
        lines = [
            f"MultiHyperNet(",
            f"  num_hypernets={self.num_hypernets},",
            f"  hv_dim={self.hv_dim}, ensemble_graph_dim={self.ensemble_graph_dim},",
            f"  dataset={self.base_dataset}, depth={self.depth},",
            f"  sub-HyperNet dims={[hn.hv_dim for hn in self._hypernets]},",
            f"  primary: {type(self._primary).__name__},",
        ]
        # Include primary's codebook info
        lines.append(
            f"  nodes_codebook: {self.nodes_codebook.shape[0]} entries x {self.nodes_codebook.shape[1]} dim,"
        )
        if hasattr(self._primary, "nodes_codebook_full"):
            cb_full = self._primary.nodes_codebook_full
            lines.append(
                f"  nodes_codebook_full (RRWP): {cb_full.shape[0]} entries x {cb_full.shape[1]} dim,"
            )
        lines.append(")")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()

    @classmethod
    def from_config(
        cls,
        config: DSHDCConfig,
        seeds: list[int],
    ) -> MultiHyperNet:
        """
        Create an ensemble from a single config with different random seeds.

        All sub-HyperNets share the same ``hv_dim`` and ``hypernet_depth``.

        Parameters
        ----------
        config : DSHDCConfig
            Base configuration (seed field will be overridden per HyperNet).
        seeds : list[int]
            Random seeds for each sub-HyperNet codebook initialization.

        Returns
        -------
        MultiHyperNet
            Ensemble with ``len(seeds)`` sub-HyperNets.
        """
        hypernets = []
        for seed in seeds:
            cfg = replace(config, seed=seed)
            hypernets.append(HyperNet(cfg))
        return cls(hypernets)

    @classmethod
    def from_dim_depth_pairs(
        cls,
        base_config: DSHDCConfig,
        dim_depth_pairs: list[tuple[int, int]],
        base_seed: int = 42,
        observed_node_features: set[tuple] | None = None,
        hypernet_cls: type | None = None,
    ) -> MultiHyperNet:
        """
        Create a mixed-dimension ensemble from (dim, depth) pairs.

        Each sub-HyperNet gets a deterministic seed: ``base_seed + i``.

        Parameters
        ----------
        base_config : DSHDCConfig
            Base configuration (hv_dim, hypernet_depth, and seed will be
            overridden per HyperNet).
        dim_depth_pairs : list[tuple[int, int]]
            List of ``(hv_dim, depth)`` tuples, one per sub-HyperNet.
        base_seed : int
            Starting seed. Sub-HyperNet *i* gets seed ``base_seed + i``.
        observed_node_features : set[tuple] | None
            If provided, passed to each HyperNet for codebook pruning
            (e.g. when using RW-augmented features).
        hypernet_cls : type, optional
            Class to instantiate for each sub-HyperNet. Defaults to
            ``HyperNet``. Pass ``RRWPHyperNet`` to create an ensemble
            with split RRWP encoding.

        Returns
        -------
        MultiHyperNet
            Ensemble with ``len(dim_depth_pairs)`` sub-HyperNets.
        """
        if hypernet_cls is None:
            hypernet_cls = HyperNet
        hypernets = []
        for i, (dim, depth) in enumerate(dim_depth_pairs):
            cfg = replace(
                base_config,
                hv_dim=dim,
                hypernet_depth=depth,
                seed=base_seed + i,
            )
            hypernets.append(hypernet_cls(cfg, observed_node_features=observed_node_features))
        return cls(hypernets)

    # ─────────────────── Properties ───────────────────

    @property
    def hv_dim(self) -> int:
        """Base hypervector dimension (from primary HyperNet)."""
        return self._primary.hv_dim

    @property
    def ensemble_graph_dim(self) -> int:
        """Total dimension of the concatenated graph embedding (K * hv_dim)."""
        return sum(hn.hv_dim for hn in self._hypernets)

    @property
    def num_hypernets(self) -> int:
        """Number of sub-HyperNets in the ensemble."""
        return len(self._hypernets)

    @property
    def primary(self) -> HyperNet:
        """The primary (first) HyperNet, used for Order 0 and decoding."""
        return self._primary

    @property
    def base_dataset(self):
        return self._primary.base_dataset

    @property
    def rw_config(self):
        """RW config from the primary HyperNet (shared across ensemble)."""
        return self._primary.rw_config

    @property
    def seed(self):
        return self._primary.seed

    @property
    def normalize(self):
        return self._primary.normalize

    @property
    def depth(self):
        return self._primary.depth

    @property
    def vsa(self):
        return self._primary.vsa

    # Delegate codebook / indexer access to primary (for decoding)

    @property
    def nodes_codebook(self):
        return self._primary.nodes_codebook

    @property
    def nodes_indexer(self):
        return self._primary.nodes_indexer

    @property
    def edges_codebook(self):
        return self._primary.edges_codebook

    @property
    def edges_indexer(self):
        return self._primary.edges_indexer

    @property
    def edge_feature_codebook(self):
        return self._primary.edge_feature_codebook

    @property
    def edge_feature_indexer(self):
        return self._primary.edge_feature_indexer

    @property
    def node_encoder_map(self):
        return self._primary.node_encoder_map

    @property
    def edge_encoder_map(self):
        return self._primary.edge_encoder_map

    @property
    def order_zero_codebook(self):
        """Codebook used for order-0 (node_terms) decoding (delegates to primary)."""
        return self._primary.order_zero_codebook

    @property
    def dataset_info(self):
        return self._primary.dataset_info

    # ─────────────────── Encoding ───────────────────

    def encode_properties(self, data: Data) -> Data:
        """
        Encode node/edge properties using the primary HyperNet's codebook.

        This sets ``data.node_hv`` for Order 0 computation. Each sub-HyperNet
        re-encodes with its own codebook inside ``forward()``.

        Parameters
        ----------
        data : Data
            PyG Data or Batch object.

        Returns
        -------
        Data
            Same object with ``node_hv`` (and optionally ``edge_hv``) set.
        """
        return self._primary.encode_properties(data)

    def rebuild_unpruned_codebook(self) -> None:
        """Rebuild the full unpruned codebook for every sub-HyperNet."""
        for hn in self._hypernets:
            hn.rebuild_unpruned_codebook()

    def forward(
        self,
        data: Data | Batch,
        *,
        bidirectional: bool = False,
        normalize: bool | None = None,
    ) -> dict[str, Tensor]:
        """
        Encode graphs using all sub-HyperNets and concatenate graph embeddings.

        Each sub-HyperNet independently encodes the input (using its own
        codebook) and produces a graph embedding. The final ``graph_embedding``
        is the concatenation of all K embeddings along the feature dimension.
        ``edge_terms`` comes from the primary HyperNet only.

        Parameters
        ----------
        data : Data | Batch
            PyG Data or Batch object.
        bidirectional : bool
            Whether to duplicate edges for bidirectional processing.
        normalize : bool, optional
            Whether to L2-normalize after each message passing layer.

        Returns
        -------
        dict
            - ``graph_embedding``: ``[batch_size, K * hv_dim]``
            - ``node_terms``: ``[batch_size, hv_dim]`` (from primary only)
            - ``edge_terms``: ``[batch_size, hv_dim]`` (from primary only)
        """
        graph_embeddings = []
        node_terms = None
        edge_terms = None

        for i, hn in enumerate(self._hypernets):
            data_copy = data.clone()
            out = hn.forward(
                data_copy,
                bidirectional=bidirectional,
                normalize=normalize,
            )
            graph_embeddings.append(out["graph_embedding"])
            if i == 0:
                node_terms = out["node_terms"]
                edge_terms = out["edge_terms"]

        return {
            "graph_embedding": torch.cat(graph_embeddings, dim=-1),
            "node_terms": node_terms,
            "edge_terms": edge_terms,
        }

    # ─────────────────── Decoding (delegate to primary) ───────────────────

    def decode_order_zero(self, embedding: Tensor) -> Tensor:
        """Decode node types from Order 0 embedding (delegates to primary)."""
        return self._primary.decode_order_zero(embedding)

    def decode_order_zero_iterative(self, embedding: Tensor, debug: bool = False):
        """Decode node types iteratively (delegates to primary)."""
        return self._primary.decode_order_zero_iterative(embedding, debug=debug)

    def decode_order_zero_counter(self, embedding: Tensor):
        """Decode node counts as Counter dict (delegates to primary)."""
        return self._primary.decode_order_zero_counter(embedding)

    def decode_order_zero_counter_iterative(self, embedding: Tensor):
        """Decode node counts iteratively (delegates to primary)."""
        return self._primary.decode_order_zero_counter_iterative(embedding)

    def decode_order_one(self, edge_term: Tensor, node_counter, debug: bool = False):
        """Decode edges from edge_term (delegates to primary)."""
        return self._primary.decode_order_one(edge_term, node_counter, debug=debug)

    def decode_order_one_no_node_terms(self, edge_term: Tensor, debug: bool = False):
        """Decode edges without node terms (delegates to primary)."""
        return self._primary.decode_order_one_no_node_terms(edge_term, debug=debug)

    def decode_graph(self, edge_term, graph_term, decoder_settings=None, fallback_decoder_settings=None):
        """Full graph decoding pipeline (delegates to primary)."""
        return self._primary.decode_graph(
            edge_term, graph_term,
            decoder_settings=decoder_settings,
            fallback_decoder_settings=fallback_decoder_settings,
        )

    def encode_edge_multiset(self, edge_list):
        """Encode edge multiset (delegates to primary)."""
        return self._primary.encode_edge_multiset(edge_list)

    def ensure_vsa(self, t: Tensor):
        """Ensure tensor is the correct VSA type (delegates to primary)."""
        return self._primary.ensure_vsa(t)

    def limit_nodes_codebook(self, node_features):
        """Limit nodes codebook on all sub-HyperNets."""
        for hn in self._hypernets:
            hn.limit_nodes_codebook(node_features)

    # ─────────────────── Distance ───────────────────

    def calculate_order_n_distance(self, order_n_a: Tensor, order_n_b: Tensor) -> Tensor:
        """
        Core distance metric: per-sub-HyperNet averaged cosine distance.

        Overrides :meth:`HyperNet.calculate_order_n_distance` to handle
        the concatenated ensemble embedding.  The order-N vector is split
        into K chunks (one per sub-HyperNet, possibly with different
        dimensions), cosine distance is computed independently for each
        chunk, and the results are averaged.  This gives each sub-HyperNet
        equal influence regardless of its dimensionality.

        Parameters
        ----------
        order_n_a, order_n_b : Tensor
            Shape ``[batch, ensemble_graph_dim]`` where
            ``ensemble_graph_dim = sum(hn.hv_dim for hn in sub_hypernets)``.
            These are the **already-extracted** order-N portions (no
            order-0 prefix).

        Returns
        -------
        Tensor
            Mean cosine distance across sub-HyperNets, shape ``[batch]``.
        """
        import torch.nn.functional as F

        # Split the concatenated order-N into one chunk per sub-HyperNet.
        chunk_dims = [hn.hv_dim for hn in self._hypernets]
        a_chunks = order_n_a.split(chunk_dims, dim=-1)
        b_chunks = order_n_b.split(chunk_dims, dim=-1)

        # Compute cosine distance independently per sub-HyperNet, then average.
        distances = torch.stack([
            1.0 - F.cosine_similarity(a.float(), b.float(), dim=-1)
            for a, b in zip(a_chunks, b_chunks)
        ], dim=0)

        return distances.mean(dim=0)

    def calculate_distance(self, vec_a: Tensor, vec_b: Tensor) -> Tensor:
        """
        Convenience wrapper: cosine distance from full ``[order_0 | order_N]`` vectors.

        Strips the order-0 prefix (first ``hv_dim`` dims, from the primary
        HyperNet) and delegates to :meth:`calculate_order_n_distance`.

        Parameters
        ----------
        vec_a, vec_b : Tensor
            Shape ``[batch, hv_dim + ensemble_graph_dim]``.

        Returns
        -------
        Tensor
            Mean cosine distance across sub-HyperNets, shape ``[batch]``.
        """
        return self.calculate_order_n_distance(
            vec_a[:, self.hv_dim :], vec_b[:, self.hv_dim :],
        )

    # ─────────────────── Device Management ───────────────────

    def to(self, device, dtype=None):
        """Move all sub-HyperNets to the given device."""
        for hn in self._hypernets:
            hn.to(device, dtype)
        return self

    def eval(self):
        """Set all sub-HyperNets to eval mode."""
        for hn in self._hypernets:
            hn.eval()
        return self

    def train(self, mode: bool = True):
        """Set all sub-HyperNets to train mode."""
        for hn in self._hypernets:
            hn.train(mode)
        return self

    # ─────────────────── Save / Load ───────────────────

    def save(self, path: str | Path) -> None:
        """
        Save MultiHyperNet to file.

        Stores each sub-HyperNet's full state in a single file.

        Parameters
        ----------
        path : str | Path
            Output file path.
        """
        path = Path(path)

        hypernet_states = []
        for hn in self._hypernets:
            state = {
                "config": {
                    "hv_dim": hn.hv_dim,
                    "depth": hn.depth,
                    "vsa": hn.vsa.value,
                    "seed": hn.seed,
                    "normalize": hn.normalize,
                    "base_dataset": hn.base_dataset,
                    "dtype": "float64" if hn._dtype == torch.float64 else "float32",
                    "rw_config": {
                        "enabled": hn.rw_config.enabled,
                        "k_values": hn.rw_config.k_values,
                        "num_bins": hn.rw_config.num_bins,
                    } if hasattr(hn, "rw_config") else None,
                    "prune_codebook": getattr(hn, "prune_codebook", True),
                    "normalize_graph_embedding": getattr(hn, "normalize_graph_embedding", False),
                },
                "encoder_maps": {
                    "node": hn._serialize_encoder_map(hn.node_encoder_map),
                    "edge": hn._serialize_encoder_map(hn.edge_encoder_map),
                    "graph": hn._serialize_encoder_map(hn.graph_encoder_map),
                },
                "codebooks": {
                    "nodes": hn.nodes_codebook.cpu(),
                    "edges": hn._edges_codebook.cpu() if hn._edges_codebook is not None else None,
                    "edge_features": (
                        hn.edge_feature_codebook.cpu()
                        if hn.edge_feature_codebook is not None
                        else None
                    ),
                },
                "indexers": {
                    "nodes": hn.nodes_indexer.__dict__.copy(),
                    "edges": hn._edges_indexer.__dict__.copy() if hn._edges_indexer is not None else None,
                    "edge_features": (
                        hn.edge_feature_indexer.__dict__.copy()
                        if hn.edge_feature_indexer
                        else None
                    ),
                },
            }

            # Save RRWP-specific data for RRWPHyperNet sub-instances
            from graph_hdc.hypernet.rrwp_hypernet import RRWPHyperNet
            if isinstance(hn, RRWPHyperNet):
                state["rrwp"] = {
                    "num_rw_dims": hn._num_rw_dims,
                    "full_bins": hn._full_bins,
                    "full_feature_dim": hn._full_feature_dim,
                    "full_rw_config": {
                        "enabled": hn.rw_config.enabled,
                        "k_values": hn.rw_config.k_values,
                        "num_bins": hn.rw_config.num_bins,
                        "bin_boundaries": hn.rw_config.bin_boundaries,
                        "clip_range": hn.rw_config.clip_range,
                    },
                    "codebooks": {
                        "nodes_full": hn.nodes_codebook_full.cpu(),
                    },
                    "indexers": {
                        "nodes_full": hn.nodes_indexer_full.__dict__.copy(),
                    },
                }

            hypernet_states.append(state)

        torch.save(
            {
                "version": 1,
                "type": "MultiHyperNet",
                "num_hypernets": len(self._hypernets),
                "hypernets": hypernet_states,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> MultiHyperNet:
        """
        Load MultiHyperNet from file.

        Parameters
        ----------
        path : str | Path
            Path to saved MultiHyperNet.
        device : str | torch.device
            Device to load to.

        Returns
        -------
        MultiHyperNet
            Loaded ensemble.
        """
        from graph_hdc.hypernet.feature_encoders import CombinatoricIntegerEncoder
        from graph_hdc.hypernet.types import VSAModel
        from graph_hdc.utils.helpers import TupleIndexer

        path = Path(path)
        data = torch.load(path, map_location="cpu", weights_only=False)
        device = torch.device(device)

        if data.get("type") != "MultiHyperNet":
            raise ValueError(f"File is not a MultiHyperNet checkpoint: {path}")

        hypernets = []
        for state in data["hypernets"]:
            # Reconstruct each HyperNet using the same pattern as HyperNet.load
            hn = HyperNet.__new__(HyperNet)
            pl.LightningModule.__init__(hn)

            cfg = state["config"]
            hn.hv_dim = cfg["hv_dim"]
            hn.depth = cfg["depth"]
            hn.vsa = VSAModel(cfg["vsa"])
            hn.seed = cfg["seed"]
            hn.normalize = cfg["normalize"]
            hn.base_dataset = cfg["base_dataset"]
            hn._dtype = torch.float64 if cfg.get("dtype", "float64") == "float64" else torch.float32
            from graph_hdc.hypernet.configs import RWConfig
            rw_dict = cfg.get("rw_config")
            hn.rw_config = RWConfig(**rw_dict) if rw_dict else RWConfig()
            hn.prune_codebook = cfg.get("prune_codebook", True)
            hn.normalize_graph_embedding = cfg.get("normalize_graph_embedding", False)
            hn._decoding_edge_limit = 50 if hn.base_dataset == "qm9" else 122
            hn._max_step_delta = None

            hn.node_encoder_map = HyperNet._deserialize_encoder_map(
                state["encoder_maps"]["node"], device
            )
            hn.edge_encoder_map = HyperNet._deserialize_encoder_map(
                state["encoder_maps"]["edge"], device
            )
            hn.graph_encoder_map = HyperNet._deserialize_encoder_map(
                state["encoder_maps"]["graph"], device
            )

            hn.nodes_codebook = state["codebooks"]["nodes"].to(device)
            if state["codebooks"]["edges"] is not None:
                hn._edges_codebook = state["codebooks"]["edges"].to(device)
            else:
                hn._edges_codebook = None
            hn.edge_feature_codebook = (
                state["codebooks"]["edge_features"].to(device)
                if state["codebooks"]["edge_features"] is not None
                else None
            )

            hn.nodes_indexer = TupleIndexer.__new__(TupleIndexer)
            hn.nodes_indexer.__dict__.update(state["indexers"]["nodes"])

            if state["indexers"]["edges"] is not None:
                hn._edges_indexer = TupleIndexer.__new__(TupleIndexer)
                hn._edges_indexer.__dict__.update(state["indexers"]["edges"])
            else:
                hn._edges_indexer = None

            if state["indexers"]["edge_features"] is not None:
                hn.edge_feature_indexer = TupleIndexer.__new__(TupleIndexer)
                hn.edge_feature_indexer.__dict__.update(state["indexers"]["edge_features"])
            else:
                hn.edge_feature_indexer = None

            # Restore RRWP-specific data if present
            if "rrwp" in state:
                from graph_hdc.hypernet.rrwp_hypernet import RRWPHyperNet

                hn.__class__ = RRWPHyperNet
                rrwp = state["rrwp"]
                hn._num_rw_dims = rrwp["num_rw_dims"]
                hn._full_bins = rrwp["full_bins"]
                hn._full_feature_dim = rrwp["full_feature_dim"]
                hn.rw_config = RWConfig(**rrwp["full_rw_config"])
                hn.nodes_codebook_full = rrwp["codebooks"]["nodes_full"].to(device)
                hn.nodes_indexer_full = TupleIndexer.__new__(TupleIndexer)
                hn.nodes_indexer_full.__dict__.update(rrwp["indexers"]["nodes_full"])

            hypernets.append(hn)

        return cls(hypernets)

    # ─────────────────── Repr ───────────────────

    def __repr__(self) -> str:
        configs = [
            (hn.hv_dim, hn.depth, hn.seed) for hn in self._hypernets
        ]
        return (
            f"MultiHyperNet("
            f"K={self.num_hypernets}, "
            f"ensemble_graph_dim={self.ensemble_graph_dim}, "
            f"configs(dim,depth,seed)={configs}, "
            f"base_dataset='{self.base_dataset}'"
            f")"
        )
