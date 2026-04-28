"""
HyperNet: Hyperdimensional Graph Encoder.

This module implements the main encoder for molecular graphs using
Vector Symbolic Architectures (VSA) with message passing.
"""

import enum
import heapq
import itertools
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Final

import networkx as nx
import numpy as np
import pytorch_lightning as pl
import torch
import torchhd
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torchhd import VSATensor
from tqdm import tqdm

from graph_hdc.datasets.utils import DatasetInfo, get_dataset_info
from graph_hdc.hypernet.configs import (
    AStarDecoderSettings,
    BaseDataset,
    DecoderSettings,
    DSHDCConfig,
    FallbackDecoderSettings,
    Features,
    IndexRange,
    RWConfig,
)
from graph_hdc.hypernet.correction_utils import (
    CorrectionResult,
    get_corrected_sets,
    get_node_counter,
    target_reached,
)
from graph_hdc.hypernet.decoder import (
    compute_sampling_structure,
    has_valid_ring_structure,
    try_find_isomorphic_graph,
)
from graph_hdc.hypernet.feature_encoders import (
    AbstractFeatureEncoder,
    CombinatoricIntegerEncoder,
)
from graph_hdc.hypernet.types import Feat, VSAModel
from graph_hdc.utils.helpers import (
    DataTransformer,
    TupleIndexer,
    cartesian_bind_tensor,
    scatter_hd,
)
from graph_hdc.utils.nx_utils import (
    add_edge_if_possible,
    add_node_and_connect,
    add_node_with_feat,
    anchors,
    connect_all_if_possible,
    count_ring_mismatches,
    leftover_features,
    order_leftovers_by_degree_distinct,
    powerset,
    residual_degree,
    ring_membership_consistent,
    valence_satisfiable,
)
from graph_hdc.utils.nx_utils import (
    graph_hash as _hash,
)

EncoderMap = dict[Features, tuple[AbstractFeatureEncoder, IndexRange]]
MAX_ALLOWED_DECODING_NODES_QM9: Final[int] = 18
MAX_ALLOWED_DECODING_EDGES_QM9: Final[int] = 50
MAX_ALLOWED_DECODING_NODES_ZINC: Final[int] = 60
MAX_ALLOWED_DECODING_EDGES_ZINC: Final[int] = 122

# PubChem variants: limits scale with max molecule size
_PUBCHEM_DECODING_LIMITS: dict[str, tuple[int, int]] = {
    # (max_nodes, max_edges)
    "pubchem16": (22, 60),
    "pubchem32": (40, 130),
    "pubchem64": (70, 200),
}


class CorrectionLevel(str, enum.Enum):
    """Correction level for decoding."""
    FAIL = "failed to correct"
    ZERO = "not corrected"
    ONE = "edge added/removed"
    TWO = "edge added/removed then re-decoded"
    THREE = "edge added/removed of the level two results"


@dataclass
class AStarSearchStats:
    """Per-decode statistics collected by ``decode_graph_astar``.

    Counters are cumulative across the whole search. Populated on a
    ``DecodingResult`` via the optional ``search_stats`` field — other
    decoders (greedy, main) leave it as ``None``.
    """
    # Loop control
    pops: int = 0                      # total heap pops over the main loop
    early_exit: bool = False           # broke out because sim >= 1 - eps
    budget_exhausted: bool = False     # wall-clock budget fired

    # Pop-level outcomes
    completes: int = 0                 # popped states that were complete
    dead_end_pops: int = 0             # popped but had no valid expansion

    # Child-level counts (accumulated across all pops)
    children_generated: int = 0        # total children enumerated by expansion
    rule_a_rejected: int = 0           # Rule A: non-ring on would-be cycle
    dedup_rejected: int = 0            # WL-hash duplicate
    ring_bc_rejected: int = 0          # Rules B + C
    children_scored: int = 0           # pushed onto heap after all filters

    # Encoder usage
    encoder_forwards: int = 0          # number of self._batched_forward calls
    encoder_graphs_scored: int = 0     # total graphs passed through the encoder

    # Search tree / heap
    seeds_pushed: int = 0              # seeds that made it onto the heap
    unique_states_seen: int = 0        # size of the dedup set at the end
    heap_peak: int = 0                 # max heap size observed
    heap_final: int = 0                # heap size at loop exit

    # Timing (seconds)
    seconds_phase0: float = 0.0        # edge unbinding + node-counter derivation
    seconds_search: float = 0.0        # main best-first loop


@dataclass
class DecodingResult:
    """Result of graph decoding."""
    nx_graphs: list[nx.Graph] = field(default_factory=list)
    final_flags: list[bool] = field(default_factory=lambda: [False])
    target_reached: bool = False
    cos_similarities: list[float] = field(default_factory=lambda: [0.0])
    correction_level: CorrectionLevel = CorrectionLevel.ZERO
    search_stats: AStarSearchStats | None = None


class HyperNet(pl.LightningModule):
    """
    Hyperdimensional Graph Encoder using Vector Symbolic Architectures.

    Encodes molecular graphs into fixed-dimensional hypervectors using:
    - Feature binding: HV_node = bind(HV_atom, HV_degree, ...)
    - Message passing: Aggregate neighbor information across layers
    - Graph readout: Bundle all node embeddings

    Attributes
    ----------
    hv_dim : int
        Hypervector dimensionality
    depth : int
        Number of message passing layers
    vsa : VSAModel
        VSA model (HRR, MAP, or FHRR)
    base_dataset : str
        Base dataset ("qm9" or "zinc")
    """

    __allowed_vsa_models__: ClassVar[set[VSAModel]] = {VSAModel.MAP, VSAModel.FHRR, VSAModel.HRR}

    def __init__(
        self,
        config: DSHDCConfig,
        depth: int | None = None,
        observed_node_features: set[tuple] | None = None,
    ):
        """
        Initialize HyperNet from configuration.

        Parameters
        ----------
        config : DSHDCConfig
            Dataset and encoding configuration
        depth : int, optional
            Override message passing depth from config
        observed_node_features : set[tuple], optional
            Explicit set of observed node feature tuples for codebook
            pruning. When provided, used instead of
            ``get_dataset_info(base_dataset).node_features``. This is
            needed when extra features (e.g. RW bins) have been appended
            to the base node features, since the cached dataset info
            only knows about the base features.
        """
        super().__init__()

        # Core attributes
        self.hv_dim = config.hv_dim
        self.depth = depth if depth is not None else config.hypernet_depth
        self.vsa = self._validate_vsa(config.vsa)
        self.seed = config.seed
        self.normalize = config.normalize
        self.base_dataset: BaseDataset = config.base_dataset
        self._dtype = torch.float64 if config.dtype == "float64" else torch.float32
        self.rw_config: RWConfig = config.rw_config
        self.prune_codebook: bool = config.prune_codebook
        self.normalize_graph_embedding: bool = getattr(config, "normalize_graph_embedding", False)

        # Set seed for reproducible codebook generation
        if self.seed is not None:
            torch.manual_seed(self.seed)

        # Build encoder maps from config
        device = torch.device(config.device)
        self.node_encoder_map: EncoderMap = self._build_encoder_map(
            config.node_feature_configs, config, device
        )
        self.edge_encoder_map: EncoderMap = self._build_encoder_map(
            config.edge_feature_configs or {}, config, device
        )
        self.graph_encoder_map: EncoderMap = self._build_encoder_map(
            config.graph_feature_configs or {}, config, device
        )

        # Build derived codebooks eagerly
        self._build_codebooks(device, observed_node_features=observed_node_features)

        # Decoding parameters (dataset-dependent)
        if self.base_dataset == "qm9":
            self._decoding_edge_limit = MAX_ALLOWED_DECODING_EDGES_QM9
        elif self.base_dataset in _PUBCHEM_DECODING_LIMITS:
            self._decoding_edge_limit = _PUBCHEM_DECODING_LIMITS[self.base_dataset][1]
        else:
            self._decoding_edge_limit = MAX_ALLOWED_DECODING_EDGES_ZINC
        self._max_step_delta: float | None = None
        self._incremental_edges: bool = False

    def __str__(self) -> str:
        lines = [
            f"{type(self).__name__}(",
            f"  vsa={self.vsa.value}, hv_dim={self.hv_dim}, depth={self.depth},",
            f"  dataset={self.base_dataset}, normalize={self.normalize},",
            f"  dtype={self._dtype},",
            f"  nodes_codebook: {self.nodes_codebook.shape[0]} entries x {self.nodes_codebook.shape[1]} dim,",
        ]
        if hasattr(self, "nodes_indexer") and hasattr(self.nodes_indexer, "sizes"):
            lines.append(f"  node_feature_bins={self.nodes_indexer.sizes},")
        if hasattr(self, "edge_feature_codebook") and self.edge_feature_codebook is not None:
            lines.append(
                f"  edge_codebook: {self.edge_feature_codebook.shape[0]} entries x {self.edge_feature_codebook.shape[1]} dim,"
            )
        else:
            lines.append("  edge_codebook: not initialized,")
        lines.append(f"  rw_config: enabled={self.rw_config.enabled},")
        lines.append(f"  prune_codebook={self.prune_codebook},")
        lines.append(")")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def _nx_graphs_to_pyg(graphs: list) -> list:
        """Convert NX graphs to PyG Data, preferring raw 'type' tuples over Feat objects.

        The 'type' attribute (set by the decoder alongside 'feat') stores
        the exact feature tuple without round-tripping through Feat, which
        would booleanize position 4 for datasets without is_in_ring (e.g. QM9+RRWP).
        """
        from torch_geometric.data import Data

        data_list = []
        for g in graphs:
            nodes = sorted(g.nodes)
            idx_of = {n: i for i, n in enumerate(nodes)}
            feats = []
            for n in nodes:
                nd = g.nodes[n]
                if "type" in nd:
                    feats.append(list(nd["type"]))
                elif "feat" in nd:
                    f = nd["feat"]
                    feats.append(list(f.to_tuple()) if hasattr(f, "to_tuple") else list(f))
                else:
                    raise ValueError(f"Node {n} has no 'type' or 'feat' attribute")
            x = torch.tensor(feats, dtype=torch.float)
            src, dst = [], []
            for u, v in g.edges():
                iu, iv = idx_of[u], idx_of[v]
                src.extend([iu, iv])
                dst.extend([iv, iu])
            edge_index = (
                torch.tensor([src, dst], dtype=torch.long)
                if src else torch.empty((2, 0), dtype=torch.long)
            )
            data_list.append(Data(x=x, edge_index=edge_index))
        return data_list

    def _batched_forward(self, graphs: list, batch_size: int = 64) -> dict:
        """Encode a list of NX graphs in mini-batches, returning concatenated outputs."""
        data_list = self._nx_graphs_to_pyg(graphs)
        loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
        outputs = []
        for mini_batch in loader:
            outputs.append(self.forward(mini_batch.to(self.device)))
        keys = outputs[0].keys()
        return {key: torch.cat([o[key] for o in outputs], dim=0) for key in keys}

    def _validate_vsa(self, vsa: VSAModel) -> VSAModel:
        """Validate VSA model is supported."""
        if vsa in self.__allowed_vsa_models__:
            return vsa
        raise ValueError(f"{vsa.value} not supported. Supported: {self.__allowed_vsa_models__}")

    def _build_encoder_map(
        self,
        feature_configs: dict[Features, Any],
        config: DSHDCConfig,
        device: torch.device,
    ) -> EncoderMap:
        """Build encoder map from feature configurations."""
        encoder_map: EncoderMap = {}
        for feat, cfg in feature_configs.items():
            encoder = cfg.encoder_cls(
                num_categories=cfg.count,
                dim=config.hv_dim,
                vsa=config.vsa.value,
                idx_offset=cfg.idx_offset,
                device=device,
                indexer=TupleIndexer(sizes=cfg.bins) if cfg.bins else None,
                dtype=config.dtype,
            )
            encoder_map[feat] = (encoder, cfg.index_range)
        return encoder_map

    def limit_nodes_codebook(self, node_features) -> None:
        # self.populate_nodes_codebooks()
        # self._populate_nodes_indexer()

        # Limit the codebook only to a subset of the nodes.
        sorted_limit_node_set = sorted(node_features)
        idxs = self.nodes_indexer.get_idxs(sorted_limit_node_set)
        self.nodes_indexer.idx_to_tuple = sorted_limit_node_set
        self.nodes_indexer.tuple_to_idx = {tup: idx for idx, tup in enumerate(sorted_limit_node_set)}
        self.nodes_codebook = self.nodes_codebook[idxs].as_subclass(type(self.nodes_codebook))

        for _, (enc, _) in self.node_encoder_map.items():
            enc.codebook = self.nodes_codebook
            enc.indexer = self.nodes_indexer

    def limit_edges_codebook(self, observed_edges: set[tuple[tuple, tuple]]) -> None:
        """Prune edges codebook to only observed (src, dst) node-type pairs.

        Must be called after ``limit_nodes_codebook`` so that
        ``nodes_indexer`` maps the pruned node types.

        Parameters
        ----------
        observed_edges : set of (src_tuple, dst_tuple)
            Edge pairs actually seen in the dataset (directed — include
            both orientations).
        """
        # Build edge HVs only for observed pairs
        edge_pairs = []  # (node_idx_a, node_idx_b) in pruned indexer space
        edge_hvs = []
        seen = set()
        for src_t, dst_t in sorted(observed_edges):
            src_idx = self.nodes_indexer.get_idx(src_t)
            dst_idx = self.nodes_indexer.get_idx(dst_t)
            if src_idx is None or dst_idx is None:
                continue
            key = (src_idx, dst_idx)
            if key in seen:
                continue
            seen.add(key)
            hv = self.nodes_codebook[src_idx].bind(self.nodes_codebook[dst_idx])
            edge_hvs.append(hv)
            edge_pairs.append(key)

        if not edge_hvs:
            return

        self._edges_codebook = torch.stack(edge_hvs).to(self.nodes_codebook.device)
        # Build a lightweight indexer with the same interface as TupleIndexer
        self._edges_indexer = TupleIndexer.__new__(TupleIndexer)
        self._edges_indexer.sizes = []
        self._edges_indexer.idx_to_tuple = edge_pairs
        self._edges_indexer.tuple_to_idx = {t: i for i, t in enumerate(edge_pairs)}

    def enable_incremental_edges(self) -> None:
        """Enable incremental edge codebook mode.

        Initialises an empty edge codebook and indexer that can be grown
        via :meth:`register_edges`.  The ``edges_codebook`` property is
        guarded so it never falls through to the catastrophic full
        Cartesian-product build.
        """
        self._incremental_edges = True
        self._edges_codebook = torch.empty(
            0, self.hv_dim,
            device=self.nodes_codebook.device,
            dtype=self._dtype,
        )
        self._edges_indexer = TupleIndexer.__new__(TupleIndexer)
        self._edges_indexer.sizes = []
        self._edges_indexer.idx_to_tuple = []
        self._edges_indexer.tuple_to_idx = {}
        self._max_step_delta = None

    def register_edges(
        self, edge_pairs: set[tuple[tuple, tuple]]
    ) -> int:
        """Incrementally add new edge pairs to the edge codebook.

        For each ``(src_node_tuple, dst_node_tuple)`` pair, looks up both
        node types in ``nodes_indexer``, computes the bound hypervector,
        and appends it to the edge codebook.  Pairs already registered or
        referencing unknown node types are silently skipped.

        Parameters
        ----------
        edge_pairs : set of (src_tuple, dst_tuple)
            Directed edge pairs expressed as node-feature tuples.

        Returns
        -------
        int
            Number of *newly* registered pairs.
        """
        if self._edges_codebook is None:
            self.enable_incremental_edges()

        new_keys = []
        new_hvs = []
        for src_t, dst_t in edge_pairs:
            src_idx = self.nodes_indexer.get_idx(src_t)
            dst_idx = self.nodes_indexer.get_idx(dst_t)
            if src_idx is None or dst_idx is None:
                continue
            key = (src_idx, dst_idx)
            if key in self._edges_indexer.tuple_to_idx:
                continue
            new_keys.append(key)
            new_hvs.append(
                self.nodes_codebook[src_idx].bind(self.nodes_codebook[dst_idx])
            )

        if not new_hvs:
            return 0

        new_tensor = torch.stack(new_hvs).to(self._edges_codebook.device)
        self._edges_codebook = torch.cat(
            [self._edges_codebook, new_tensor], dim=0
        )
        self._edges_indexer.extend(new_keys)
        # Invalidate cached decoding threshold — it depends on codebook
        # contents and must be recomputed after expansion.
        self._max_step_delta = None
        return len(new_keys)

    def rebuild_unpruned_codebook(self) -> None:
        """Rebuild the full unpruned codebook from the original seed.

        When a HyperNet was saved with ``prune_codebook=True``, the
        encoder codebook and indexer only cover feature tuples observed
        during training.  This method regenerates the full codebook
        (using the stored seed for reproducibility) so that *any* valid
        feature tuple can be encoded.

        The full Cartesian product is built on CPU to avoid GPU OOM,
        then moved to the original device.
        """
        device = self.nodes_codebook.device
        # Use the HyperNet-level dtype (correctly saved/loaded) rather
        # than the encoder's dtype attribute which may be wrong in old
        # checkpoints that didn't serialize it.
        dtype_str = "float64" if self._dtype == torch.float64 else "float32"

        # Regenerate the full codebook with the exact same seed state
        # that was used during __init__: seed is set, then the node
        # encoder codebook is the first random draw.
        torch.manual_seed(self.seed)

        for _feat, (enc, _idx_range) in self.node_encoder_map.items():
            # Fix the encoder's dtype to match the HyperNet (may differ
            # in old checkpoints where dtype wasn't serialized).
            enc.dtype = dtype_str
            # Regenerate original full-size codebook on CPU (uses num_categories
            # which is never modified by pruning)
            enc.codebook = enc.generate_codebook().cpu()
            # Rebuild the full Cartesian-product indexer from the stored
            # bin sizes (also unmodified by pruning)
            enc.indexer = TupleIndexer(enc.indexer.sizes)

        # Rebuild the top-level nodes codebook / indexer on CPU first
        node_codebooks = [e.codebook for e, _ in self.node_encoder_map.values()]
        self.nodes_codebook = cartesian_bind_tensor(node_codebooks).to(device)

        # Move encoder codebooks to target device
        for _, (enc, _) in self.node_encoder_map.items():
            enc.codebook = self.nodes_codebook
            enc.device = device

        enc0 = self.node_encoder_map[Features.NODE_FEATURES][0]
        if isinstance(enc0, CombinatoricIntegerEncoder):
            self.nodes_indexer = enc0.indexer
        else:
            sizes = [e.num_categories for e, _ in self.node_encoder_map.values()]
            self.nodes_indexer = TupleIndexer(sizes)

        # Invalidate cached edges codebook (depends on nodes_codebook)
        self._edges_codebook = None
        self._edges_indexer = None

        self.prune_codebook = False

    def extend_node_codebook(self, missing_tuples: list[tuple]) -> int:
        """Append fresh random hypervectors for feature tuples not yet in the codebook.

        When a HyperNet is loaded with a pruned codebook, encoding a molecule
        that contains a feature tuple absent from the indexer raises. This
        method adds the missing tuples (deduplicated) by sampling fresh random
        hypervectors of the correct VSA type and concatenating them onto the
        nodes codebook, so encode/decode round-trips continue to work for
        previously-unseen feature combinations.
        """
        seen: set[tuple] = set()
        unique: list[tuple] = []
        for t in missing_tuples:
            t = tuple(int(x) for x in t)
            if t in seen or t in self.nodes_indexer.tuple_to_idx:
                continue
            seen.add(t)
            unique.append(t)
        if not unique:
            return 0

        cb = self.nodes_codebook
        new_hvs = torchhd.random(
            len(unique),
            self.hv_dim,
            vsa=self.vsa.value,
            device=cb.device,
            dtype=cb.dtype,
        )
        self.nodes_codebook = torch.cat([cb, new_hvs.to(cb.dtype)], dim=0).as_subclass(type(cb))
        self.nodes_indexer.extend(unique)

        for enc, _ in self.node_encoder_map.values():
            enc.codebook = self.nodes_codebook
            if hasattr(enc, "indexer") and enc.indexer is not None:
                enc.indexer = self.nodes_indexer

        self._edges_codebook = None
        self._edges_indexer = None
        if hasattr(self, "_min_node_delta"):
            del self._min_node_delta

        return len(unique)


    def _build_codebooks(
        self,
        device: torch.device,
        observed_node_features: set[tuple] | None = None,
    ) -> None:
        """Build all derived codebooks from encoder maps.

        The full Cartesian product of node features can be very large
        (millions of entries) when many RRWP bins are used.  To avoid
        GPU OOM, the product is built on CPU first, pruned to the
        observed feature subset, and only then moved to the target device.
        """
        # Nodes codebook: build full Cartesian product on CPU, prune, then move
        node_codebooks = [enc.codebook.cpu() for enc, _ in self.node_encoder_map.values()]
        self.nodes_codebook = cartesian_bind_tensor(node_codebooks)  # CPU

        # Nodes indexer
        enc = self.node_encoder_map[Features.NODE_FEATURES][0]
        if isinstance(enc, CombinatoricIntegerEncoder):
            self.nodes_indexer = enc.indexer
        else:
            sizes = [e.num_categories for e, _ in self.node_encoder_map.values()]
            self.nodes_indexer = TupleIndexer(sizes)

        # Limit codebook based on observed node types (unless pruning is disabled)
        # This happens while still on CPU to avoid allocating the full product on GPU
        if self.prune_codebook:
            if observed_node_features is not None:
                self.limit_nodes_codebook(node_features=observed_node_features)
            else:
                self.limit_nodes_codebook(node_features=get_dataset_info(self.base_dataset).node_features)

        # Now move the (pruned) nodes codebook to the target device
        self.nodes_codebook = self.nodes_codebook.to(device)
        for _, (enc, _) in self.node_encoder_map.items():
            enc.codebook = self.nodes_codebook
            enc.device = device

        # Edge feature codebook (if edge features exist)
        if self.edge_encoder_map:
            edge_codebooks = [enc.codebook for enc, _ in self.edge_encoder_map.values()]
            self.edge_feature_codebook = cartesian_bind_tensor(edge_codebooks).to(device)
            self.edge_feature_indexer = TupleIndexer(
                [e.num_categories for e, _ in self.edge_encoder_map.values()]
            )
        else:
            self.edge_feature_codebook = None
            self.edge_feature_indexer = None

        # Edges codebook: deferred to first access (can be very large)
        self._edges_codebook = None
        self._edges_indexer = None

    @property
    def edges_codebook(self):
        """Lazily build edges codebook on first access (can be very large)."""
        if self._edges_codebook is None:
            if self._incremental_edges:
                raise RuntimeError(
                    "Edge codebook is in incremental mode but has not been "
                    "initialised.  Call enable_incremental_edges() or "
                    "register_edges() before decoding.  The full Cartesian "
                    "product is too large for this encoder "
                    f"({self.nodes_codebook.shape[0]}² nodes)."
                )
            self._edges_codebook = cartesian_bind_tensor(
                [self.nodes_codebook, self.nodes_codebook]
            ).to(self.nodes_codebook.device)
        if (
            self._incremental_edges
            and self._edges_codebook is not None
            and self._edges_codebook.shape[0] == 0
        ):
            raise RuntimeError(
                "Edge codebook is in incremental mode but empty. "
                "Call register_edges() with observed edge pairs before "
                "decoding.  The codebook grows as training data is observed."
            )
        return self._edges_codebook

    @edges_codebook.setter
    def edges_codebook(self, value):
        self._edges_codebook = value

    @property
    def edges_indexer(self):
        """Lazily build edges indexer on first access."""
        if self._edges_indexer is None:
            if self._incremental_edges:
                raise RuntimeError(
                    "Edge indexer is in incremental mode but empty. "
                    "Call register_edges() first."
                )
            self._edges_indexer = TupleIndexer(
                [self.nodes_indexer.size(), self.nodes_indexer.size()]
            )
        return self._edges_indexer

    @edges_indexer.setter
    def edges_indexer(self, value):
        self._edges_indexer = value

    @property
    def order_zero_codebook(self):
        """Codebook used for order-0 (node_terms) decoding."""
        return self.nodes_codebook

    def to(self, device, dtype=None):
        """Move HyperNet to device."""
        device = torch.device(device)
        super().to(device)
        if dtype is not None:
            super().to(dtype)
            self._dtype = dtype

        # Move codebooks
        self.nodes_codebook = self.nodes_codebook.to(device=device, dtype=self._dtype)
        if self._edges_codebook is not None:
            self._edges_codebook = self._edges_codebook.to(device=device, dtype=self._dtype)
        if self.edge_feature_codebook is not None:
            self.edge_feature_codebook = self.edge_feature_codebook.to(device=device, dtype=self._dtype)

        # Move encoder codebooks
        for enc_map in (self.node_encoder_map, self.edge_encoder_map, self.graph_encoder_map):
            for enc, _ in enc_map.values():
                enc.device = device
                enc.codebook = enc.codebook.to(device=device, dtype=self._dtype)

        return self

    # ─────────────────────────── Encoding ───────────────────────────

    def encode_properties(self, data: Data) -> Data:
        """Encode node, edge and graph properties into hypervectors."""
        num_nodes = data.x.size(0)
        data.node_hv = self._slice_encode_bind(self.node_encoder_map, data.x, fallback_count=num_nodes)

        if self.edge_encoder_map and hasattr(data, "edge_attr") and data.edge_attr is not None:
            num_edges = data.edge_index.size(1)
            data.edge_hv = self._slice_encode_bind(self.edge_encoder_map, data.edge_attr, fallback_count=num_edges)

        if self.graph_encoder_map and hasattr(data, "y") and data.y is not None:
            num_graphs = data.y.size(0)
            data.graph_hv = self._slice_encode_bind(self.graph_encoder_map, data.y, fallback_count=num_graphs)

        return data

    def _slice_encode_bind(self, encoder_map: EncoderMap, tensor: Tensor, fallback_count: int) -> Tensor:
        """Slice, encode, and bind features into hypervectors."""
        if tensor is None or not encoder_map:
            return torch.zeros(fallback_count, self.hv_dim, device=self.device, dtype=self._dtype)

        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(-1)

        is_node_map = encoder_map is self.node_encoder_map
        slices = []
        for encoder, (start, end) in encoder_map.values():
            feat = tensor[..., start:end]
            if (
                is_node_map
                and isinstance(encoder, CombinatoricIntegerEncoder)
                and feat.shape[-1] > 1
            ):
                tup_view = feat.squeeze(-1).long() - encoder.idx_offset
                tup_list = list(map(tuple, tup_view.tolist()))
                if any(encoder.indexer.get_idx(t) is None for t in tup_list):
                    missing = [t for t in tup_list if encoder.indexer.get_idx(t) is None]
                    self.extend_node_codebook(missing)
            slices.append(encoder.encode(feat))

        if not slices:
            return torch.zeros(fallback_count, self.hv_dim, device=tensor.device, dtype=tensor.dtype)

        stacked = torch.stack(slices, dim=0)
        to_bind = stacked.transpose(0, 1)
        return torchhd.multibind(to_bind)

    def forward(
        self,
        data: Data | Batch,
        *,
        bidirectional: bool = False,
        normalize: bool | None = None,
    ) -> dict[str, Tensor]:
        """
        Encode graphs into hyperdimensional embeddings.

        Parameters
        ----------
        data : Data | Batch
            PyG Data or Batch object
        bidirectional : bool
            Whether to duplicate edges for bidirectional processing
        normalize : bool, optional
            Whether to L2-normalize after each layer (default: use self.normalize)

        Returns
        -------
        dict
            - graph_embedding: [batch_size, hv_dim] - full graph embeddings
            - node_terms: [batch_size, hv_dim] - level 0 (node features only)
            - edge_terms: [batch_size, hv_dim] - level 1 (node + 1-hop structure)
        """
        if normalize is None:
            normalize = self.normalize

        data = self.encode_properties(data)

        edge_index = torch.cat([data.edge_index, data.edge_index[[1, 0]]], dim=1) if bidirectional else data.edge_index
        srcs, dsts = edge_index

        # Ensure node_hv is on the same device as edge_index (codebooks might be on CPU)
        if data.node_hv.device != edge_index.device:
            data.node_hv = data.node_hv.to(edge_index.device)

        node_dim = data.x.size(0)
        node_hv_stack = data.node_hv.new_zeros(size=(self.depth + 1, node_dim, self.hv_dim))
        node_hv_stack[0] = data.node_hv

        edge_terms = None
        for layer_index in range(self.depth):
            messages = node_hv_stack[layer_index][dsts]
            aggregated = scatter_hd(messages, srcs, dim_size=node_dim, op="bundle")
            prev_hv = node_hv_stack[layer_index].clone()
            hr = torchhd.bind(prev_hv, aggregated)

            if layer_index == 0:
                edge_terms = hr.clone()

            if normalize:
                hr_norm = hr.norm(dim=-1, keepdim=True)
                node_hv_stack[layer_index + 1] = hr / (hr_norm + 1e-8)
            else:
                node_hv_stack[layer_index + 1] = hr

        node_hv_stack = node_hv_stack.transpose(0, 1)
        node_hv = torchhd.multibundle(node_hv_stack)
        node_terms = scatter_hd(src=data.node_hv, index=data.batch, op="bundle")
        graph_embedding = scatter_hd(src=node_hv, index=data.batch, op="bundle")
        if self.normalize_graph_embedding:
            graph_embedding = graph_embedding / (graph_embedding.norm(dim=-1, keepdim=True) + 1e-8)
        edge_terms = scatter_hd(src=edge_terms, index=data.batch, op="bundle")

        return {
            "graph_embedding": graph_embedding,
            "node_terms": node_terms,
            "edge_terms": edge_terms,
        }

    # ─────────────────────────── Decoding ───────────────────────────

    def decode_order_zero(self, embedding: torch.Tensor) -> torch.Tensor:
        """Decode node types and counts from embedding."""
        d = torchhd.dot(embedding, self.nodes_codebook)
        if self.vsa in {VSAModel.FHRR, VSAModel.MAP}:
            d = d / self.hv_dim
        return torch.round(d).int().clamp(min=0)

    def decode_order_zero_iterative(
        self, embedding: torch.Tensor, debug: bool = False
    ) -> list[tuple[int, ...]] | tuple[list[tuple[int, ...]], list[float], list[float]]:
        """
        Decode node types from embedding via iterative unbinding.

        This method uses the same iterative approach as edge decoding:
        1. Compute cosine similarity against nodes codebook
        2. Find best matching node type
        3. Subtract that node's hypervector from the residual
        4. Repeat until residual norm is negligible

        Parameters
        ----------
        embedding : torch.Tensor
            Order-0 graph embedding (bundled node hypervectors)
        debug : bool
            If True, also return norms and similarities for analysis

        Returns
        -------
        list[tuple[int, ...]]
            List of decoded node tuples (one entry per node instance)
        If debug=True, also returns (norms, similarities)
        """
        # Calculate epsilon threshold based on minimum node hypervector norm
        if not hasattr(self, "_min_node_delta"):
            norms = [hv.norm().item() for hv in self.nodes_codebook]
            self._min_node_delta = min(norms) if norms else 1.0

        eps = self._min_node_delta * 0.01

        # Get decoding limit based on dataset
        if self.base_dataset == "qm9":
            max_nodes = MAX_ALLOWED_DECODING_NODES_QM9
        elif self.base_dataset in _PUBCHEM_DECODING_LIMITS:
            max_nodes = _PUBCHEM_DECODING_LIMITS[self.base_dataset][0]
        else:
            max_nodes = MAX_ALLOWED_DECODING_NODES_ZINC

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

            # Compute cosine similarity against all node types
            sims = torchhd.cos(residual, self.nodes_codebook)
            idx_max = torch.argmax(sims).item()
            max_sim = sims[idx_max].item()
            similarities.append(max_sim)

            # Get the best matching node hypervector
            hv_found: VSATensor = self.nodes_codebook[idx_max]

            # Unbind (subtract) the found node from residual
            residual = residual - hv_found

            # Check if norm is still decreasing (stop if corrupted)
            new_norm = residual.norm().item()
            if new_norm > prev_norm:
                break
            prev_norm = new_norm

            # Record the decoded node
            node_tuple = self.nodes_indexer.get_tuple(idx_max)
            decoded_nodes.append(node_tuple)

        if debug:
            return decoded_nodes, norms_history, similarities
        return decoded_nodes

    def decode_order_zero_counter(self, embedding: torch.Tensor) -> dict[int, Counter]:
        """Decode node counts as Counter dict."""
        dot_products_rounded = self.decode_order_zero(embedding)
        return self._convert_to_counter(similarities=dot_products_rounded, indexer=self.nodes_indexer)

    def decode_order_zero_counter_iterative(self, embedding: torch.Tensor) -> dict[int, Counter]:
        """
        Decode node counts as Counter dict using iterative unbinding.

        This is the iterative version of decode_order_zero_counter that uses
        the same approach as edge decoding (cosine similarity + unbinding).

        Parameters
        ----------
        embedding : torch.Tensor
            Order-0 graph embedding(s). Shape [hv_dim] or [batch_size, hv_dim]

        Returns
        -------
        dict[int, Counter]
            Dictionary mapping batch index to Counter of node tuples
        """
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)

        counters: dict[int, Counter] = defaultdict(Counter)

        for b_idx in range(embedding.size(0)):
            decoded_nodes = self.decode_order_zero_iterative(embedding[b_idx])
            for node_tuple in decoded_nodes:
                counters[b_idx][node_tuple] += 1

        return dict(counters)

    @staticmethod
    def _convert_to_counter(similarities: torch.Tensor, indexer: TupleIndexer) -> dict[int, Counter]:
        """Convert similarity tensor to Counter dict."""
        if similarities.dim() == 1:
            similarities = similarities.unsqueeze(0)
        b_indices, t_indices = torch.nonzero(similarities, as_tuple=True)

        counters: dict[int, Counter] = defaultdict(Counter)
        for b, t in zip(b_indices.tolist(), t_indices.tolist(), strict=False):
            tup = indexer.get_tuple(t)
            count = int(similarities[b, t].item())
            counters[b][tup] = count

        return counters

    def encode_edge_multiset(self, edge_list: list[tuple[tuple, tuple]]) -> torch.Tensor:
        """Encode a multiset of edges into a single hypervector (vectorized for 10-100x speedup).
        This is used during greedy decoding with use_modified_graph_embedding to encode
        the leftover edges (edges not yet added to a partial graph) into a hypervector
        that can be subtracted from the target graph_term.

        Parameters
        ----------
        edge_list : list[tuple[tuple, tuple]]
            List of (src_node_tuple, dst_node_tuple) edges, where each node tuple
            contains the node features (atom_type, degree, formal_charge, ...).
            Edges are bidirectional, so typically each edge appears twice.

        Returns
        -------
        torch.Tensor
            Hypervector representation of the edge multiset (shape: [hv_dim]).
            Returns zero vector if edge_list is empty. Returns VSATensor type.
        """
        if not edge_list:
            return torch.zeros(self.hv_dim, device=self.device, dtype=self._dtype)

        # Vectorized computation - batch convert tuples to indices
        src_indices = []
        dst_indices = []
        for src_tuple, dst_tuple in edge_list:
            src_indices.append(self.nodes_indexer.get_idx(src_tuple))
            dst_indices.append(self.nodes_indexer.get_idx(dst_tuple))

        src_indices_tensor = torch.tensor(src_indices, dtype=torch.long, device=self.device)
        dst_indices_tensor = torch.tensor(dst_indices, dtype=torch.long, device=self.device)

        # Batch index nodes_codebook (preserves VSATensor type)
        hv_src = self.nodes_codebook[src_indices_tensor]  # [num_edges, D] VSATensor
        hv_dst = self.nodes_codebook[dst_indices_tensor]  # [num_edges, D] VSATensor

        # Vectorized bind (VSATensor.bind() handles batch operations)
        edge_hvs = hv_src.bind(hv_dst)  # [num_edges, D] VSATensor

        # Sum all edges at once (preserves VSATensor type)
        edge_term = edge_hvs.sum(dim=0)

        return edge_term

    def decode_order_one(
        self,
        edge_term: torch.Tensor,
        node_counter: Counter[tuple[int, ...]],
        debug: bool = False,
    ) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        """
        Returns information about the kind and number of edges (order one information) that were contained in the
        original graph represented by the given ``embedding`` vector.

        **Edge Decoding**

        The aim of this method is to reconstruct the first order information about what kinds of edges existed in
        the original graph based on the given graph embedding vector ``embedding``. The way in which this works is
        that we already get the zero oder constraints (==informations about which nodes are present) passed as an
        argument. Based on that we construct all possible combinations of node pairs (==edges) and calculate the
        corresponding binding of the hypervector representations. Then we can multiply each of these edge hypervectors
        with the final graph embedding to get a projection along that edge type's dimension. The magnitude of this
        projection should be proportional to the number of times that edge type was present in the original graph
        (except for a correction factor).

        Therefore, we iterate over all the possible node pairs and calculate the projection of the graph embedding
        along the direction of the edge hypervector. If the magnitude of this projection is non-zero we can assume
        that this edge type was present in the original graph and we derive the number of times it was present from
        the magnitude of the projection.

        :param edge_term: Graph representation with HDC message passing depth 1.
        :param node_tuples: The list of constraints that represent the zero order information about the
            nodes that were present in the original graph.


        :returns: A list of edges represented as tuples of (u, v) where u and v are node tuples
        """
        all_edges = list(itertools.product(node_counter.keys(), node_counter.keys()))
        num_edges = sum([(k[1] + 1) * n for k, n in node_counter.items()])
        edge_count = num_edges

        # Get all indices at once
        node_tuples_a, node_tuples_b = zip(*all_edges, strict=False)
        idx_a = torch.tensor(self.nodes_indexer.get_idxs(node_tuples_a), dtype=torch.long, device=self.device)
        idx_b = torch.tensor(self.nodes_indexer.get_idxs(node_tuples_b), dtype=torch.long, device=self.device)

        # Gather all node hypervectors at once: [N*N, D]
        hd_a = self.nodes_codebook[idx_a]  # [N*N, D]
        hd_b = self.nodes_codebook[idx_b]  # [N*N, D]

        # Vectorized bind operation
        edges_hdc = hd_a.bind(hd_b)

        norms = []
        similarities = []
        decoded_edges: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        for i in range(int(edge_count // 2)):
            norms.append(edge_term.norm().item())
            sims = torchhd.cos(edge_term, edges_hdc)
            idx_max = torch.argmax(sims).item()
            similarities.append(sims[idx_max].item())
            a_found, b_found = all_edges[idx_max]
            if not a_found or not b_found:
                break
            hd_a_found: VSATensor = self.nodes_codebook[self.nodes_indexer.get_idx(a_found)]
            hd_b_found: VSATensor = self.nodes_codebook[self.nodes_indexer.get_idx(b_found)]
            edge_term -= hd_a_found.bind(hd_b_found)
            edge_term -= hd_b_found.bind(hd_a_found)

            decoded_edges.append((a_found, b_found))
            decoded_edges.append((b_found, a_found))

        if debug:
            return decoded_edges, norms, similarities
        return decoded_edges

    def decode_order_one_no_node_terms(
        self, edge_term: torch.Tensor, debug: bool = False
    ) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        """Extract edge tuples from edge_term via iterative unbinding."""
        # edges_codebook and edges_indexer are already initialized in _build_codebooks()

        if self._max_step_delta is None:
            # Vectorized computation over the edges codebook.
            # For all supported VSAs (HRR, MAP, FHRR), bind is commutative,
            # so bind(a,b) + bind(b,a) = 2*bind(a,b) and
            # norm(delta) = 2 * norm(bind(a,b)).
            self._max_step_delta = (self.edges_codebook.norm(dim=-1) * 2).min().item()

        eps = self._max_step_delta * 0.01

        decoded_edges: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        norms = []
        prev_edge_term = edge_term.clone()
        similarities = []

        while len(decoded_edges) <= self._decoding_edge_limit:
            curr_norm = edge_term.norm().item()
            norms.append(curr_norm)
            if curr_norm <= eps:
                break
            sims = torchhd.cos(edge_term, self.edges_codebook)
            idx_max = torch.argmax(sims).item()
            similarities.append(sims[idx_max].item())
            a_found, b_found = self.edges_indexer.get_tuple(idx_max)
            hd_a_found: VSATensor = self.nodes_codebook[a_found]
            hd_b_found: VSATensor = self.nodes_codebook[b_found]
            edge_term = edge_term - hd_a_found.bind(hd_b_found)
            edge_term = edge_term - hd_b_found.bind(hd_a_found)
            if edge_term.norm().item() > prev_edge_term.norm().item():
                break
            prev_edge_term = edge_term.clone()

            decoded_edges.append((self.nodes_indexer.get_tuple(a_found), self.nodes_indexer.get_tuple(b_found)))
            decoded_edges.append((self.nodes_indexer.get_tuple(b_found), self.nodes_indexer.get_tuple(a_found)))

        if debug:
            return decoded_edges, norms, similarities
        return decoded_edges

    @property
    def dataset_info(self) -> DatasetInfo:
        """Get dataset information (node/edge features, ring info)."""
        if not hasattr(self, "_dataset_info"):
            self._dataset_info = get_dataset_info(self.base_dataset)
        return self._dataset_info

    def _is_feasible_set(self, edge_multiset: list[tuple[tuple, tuple]]) -> bool:
        """
        Check if an edge multiset can form valid graph structures.

        Performs a quick feasibility test by attempting to enumerate a small
        number of graph candidates. If at least one valid graph can be found,
        the edge multiset is considered feasible.

        Parameters
        ----------
        edge_multiset : list[tuple[tuple, tuple]]
            Edge multiset to test

        Returns
        -------
        bool
            True if at least one valid graph can be enumerated
        """

        node_counter = get_node_counter(edge_multiset)
        try:
            matching_components, id_to_type = compute_sampling_structure(
                nodes_multiset=[k for k, v in node_counter.items() for _ in range(v)],
                edges_multiset=edge_multiset,
            )
        except Exception:
            return False

        # Try to find at least one valid graph with a small budget
        decoded_graphs = try_find_isomorphic_graph(
            matching_components=matching_components,
            id_to_type=id_to_type,
            max_samples=100,
        )
        return len(decoded_graphs) > 0

    def _apply_edge_corrections(
        self,
        edge_term: VSATensor,
        initial_decoded_edges: list[tuple[tuple, tuple]],
    ) -> tuple[CorrectionResult, CorrectionLevel]:
        """
        Apply progressive correction strategies to decoded edges.

        When the initially decoded edges don't form a valid graph (node degrees don't match
        edge counts), this method applies correction strategies (currently only Level 1).

        - Level 1: Simple add/remove corrections based on fractional node counts

        Parameters
        ----------
        edge_term : VSATensor
            The edge term hypervector for re-decoding if needed.
        initial_decoded_edges : list[tuple[tuple, tuple]]
            The initially decoded edges that need correction.

        Returns
        -------
        tuple[CorrectionResult, CorrectionLevel]
            - CorrectionResult with add_sets and remove_sets
            - Correction level achieved (ONE or FAIL)
        """

        # Helper function to compute floating-point node counter
        def get_node_counter_fp(edges: list[tuple[tuple, tuple]]) -> dict[tuple, float]:
            node_degree_counter = Counter(u for u, _ in edges)
            return {k: v / (k[1] + 1) for k, v in node_degree_counter.items()}

        # Level 1: Try simple add/remove corrections
        node_counter_fp = get_node_counter_fp(initial_decoded_edges)
        correction_result: CorrectionResult = get_corrected_sets(
            node_counter_fp, initial_decoded_edges, valid_edge_tuples=self.dataset_info.edge_features
        )

        if correction_result.add_sets or correction_result.remove_sets:
            return correction_result, CorrectionLevel.ONE

        # All corrections failed: return initial edges with FAIL level
        return CorrectionResult(add_sets=[initial_decoded_edges]), CorrectionLevel.FAIL

    def _find_top_k_isomorphic_graphs(
        self,
        edge_multiset: list[tuple[tuple, tuple]],
        graph_term: VSATensor,
        iteration_budget: int,
        max_graphs_per_iter: int,
        top_k: int,
        sim_eps: float,
        use_early_stopping: bool,
    ) -> list[tuple[nx.Graph, float]]:
        """
        Find top-k graph candidates through pattern matching and similarity ranking.

        This method enumerates valid graph structures from the given edge multiset,
        re-encodes each candidate to HDC, and ranks them by cosine similarity to
        the target graph_term.

        Parameters
        ----------
        edge_multiset : list[tuple[tuple, tuple]]
            Valid edge multiset from which to generate graph candidates.
        graph_term : VSATensor
            Target graph hypervector for similarity comparison.
        iteration_budget : int
            Number of pattern matching iterations to perform.
        max_graphs_per_iter : int
            Maximum number of graph candidates to generate per iteration.
        top_k : int
            Number of top-scoring graphs to retain from each iteration.
        sim_eps : float
            Early stopping threshold: stop if similarity >= 1.0 - sim_eps.
        use_early_stopping : bool
            Whether to enable early stopping when near-perfect match is found.

        Returns
        -------
        list[tuple[nx.Graph, float]]
            List of (graph, similarity_score) tuples from all iterations.
        """


        top_k_graphs = []
        seen_hashes = set()  # Track unique graphs for deduplication

        for _ in range(iteration_budget):
            # Compute sampling structure from edge multiset
            node_counter = get_node_counter(edge_multiset)
            try:
                matching_components, id_to_type = compute_sampling_structure(
                    nodes_multiset=[k for k, v in node_counter.items() for _ in range(v)],
                    edges_multiset=edge_multiset,
                )
            except Exception:
                continue

            # Enumerate valid graph structures via isomorphism
            decoded_graphs_iter = try_find_isomorphic_graph(
                matching_components=matching_components,
                id_to_type=id_to_type,
                max_samples=max_graphs_per_iter,
            )

            if not decoded_graphs_iter:
                continue

            # Convert graphs to PyG format and add type attributes
            pyg_graphs = []
            for g in decoded_graphs_iter:
                # Add feat attribute from type attribute
                for node_id in g.nodes:
                    node_type = g.nodes[node_id].get("type")
                    if node_type:
                        g.nodes[node_id]["feat"] = Feat.from_tuple(node_type)
                pyg_graph = DataTransformer.nx_to_pyg_with_type_attr(g)
                pyg_graphs.append(pyg_graph)

            # Batch encode candidate graphs back to HDC
            BATCH_SIZE = 128
            loader = DataLoader(pyg_graphs, batch_size=BATCH_SIZE, shuffle=False)

            embedding_list = []
            for mini_batch in loader:
                # Forward pass
                out = self.forward(mini_batch.to(self.device))
                embedding_list.append(out["graph_embedding"].detach())

            graph_hdcs_batch = torch.cat(embedding_list, dim=0)

            # Compute similarities
            sims = torchhd.cos(graph_term, graph_hdcs_batch)

            # Get top k from this iteration
            top_k_sims, top_k_indices = torch.topk(sims, k=min(top_k, len(sims)))
            top_k_sims_cpu = top_k_sims.cpu().numpy()

            # Add results with deduplication using permutation-invariant hash
            for i, sim in enumerate(top_k_sims_cpu):
                graph = decoded_graphs_iter[top_k_indices[i]]
                graph_hash = _hash(graph)

                # Only add if we haven't seen this graph structure before
                if graph_hash not in seen_hashes:
                    seen_hashes.add(graph_hash)
                    top_k_graphs.append((graph, float(sim)))

            # Early stopping: if we found the perfect match, stop iterating
            if use_early_stopping:
                for sim in top_k_sims_cpu:
                    if sim == 1.0:
                        break

        return top_k_graphs

    def decode_graph(
        self,
        edge_term: torch.Tensor,
        graph_term: torch.Tensor,
        decoder_settings: DecoderSettings | None = None,
        fallback_decoder_settings: FallbackDecoderSettings | None = None,
    ) -> DecodingResult:
        """
        Decode NetworkX graphs from HDC embeddings with edge correction and pattern matching.

        This function implements a multi-phase decoding strategy:

        1. **Edge Decoding**: Extract edge multiset from edge_term using cosine similarity
           and iterative unbinding.

        2. **Correction**: If the decoded edges don't form a valid graph (node degrees
           don't match edges), apply progressive correction strategies:
           - Level 0: No correction needed
           - Level 1: Simple add/remove operations on initial decoding
           - Fail: Correction failed (note: Levels 2-3 require decode_order_one not in public repo)

        3. **Pattern Matching**: Enumerate valid graph structures from the corrected edge
           multiset, encode them back to HDC, and rank by cosine similarity to graph_term.

        4. **Fallback**: If all corrections fail, returns empty result (greedy decoder
           not implemented in public repo).

        Parameters
        ----------
        edge_term : torch.Tensor
            Edge-level HDC embedding [hv_dim] (output from forward() at depth=1)
        graph_term : torch.Tensor
            Graph-level HDC embedding [hv_dim] (for similarity ranking)
        decoder_settings : DecoderSettings, optional
            Main decoder settings. If None, uses defaults.
        fallback_decoder_settings : FallbackDecoderSettings, optional
            Settings for greedy fallback decoder (not used in public repo)

        Returns
        -------
        DecodingResult
            Object containing:
            - nx_graphs: list of NetworkX graphs (top-k candidates)
            - final_flags: list of bools (True if successfully decoded)
            - target_reached: bool (True if correction succeeded)
            - cos_similarities: list of float (similarity scores)
            - correction_level: CorrectionLevel enum value

        Notes
        -----
        The public repository implementation has limitations:
        - Only Level 1 corrections are implemented (Level 2-3 require decode_order_one)
        - No greedy beam search fallback decoder
        - Use the full private implementation for complete functionality
        """

        edge_term = self.ensure_vsa(edge_term)
        graph_term = self.ensure_vsa(graph_term)

        if decoder_settings is None:
            decoder_settings = DecoderSettings.get_default_for(base_dataset=self.base_dataset)

        # Validate decoder_settings
        if decoder_settings.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if decoder_settings.iteration_budget < 1:
            raise ValueError("iteration_budget must be >= 1")
        if decoder_settings.max_graphs_per_iter < 1:
            raise ValueError("max_graphs_per_iter must be >= 1")

        # Extract settings from dataclass
        iteration_budget: int = decoder_settings.iteration_budget
        max_graphs_per_iter: int = decoder_settings.max_graphs_per_iter
        top_k: int = decoder_settings.top_k
        sim_eps: float = decoder_settings.sim_eps
        use_early_stopping: bool = decoder_settings.early_stopping
        prefer_smaller_corrective_edits = decoder_settings.prefer_smaller_corrective_edits
        use_correction = decoder_settings.use_correction

        # Phase 1: Decode edge multiset from edge_term using greedy unbinding
        initial_decoded_edges = self.decode_order_one_no_node_terms(edge_term.clone())
        if len(initial_decoded_edges) > self._decoding_edge_limit:
            return DecodingResult()

        # Phase 2: Check if edges form a valid graph (node degrees match edge counts)
        correction_level = CorrectionLevel.ZERO
        decoded_edges = [initial_decoded_edges]

        if not target_reached(initial_decoded_edges):
            if not use_correction:
                return DecodingResult()
            # Edges don't form valid graph → apply progressive correction strategies
            correction_results, correction_level = self._apply_edge_corrections(
                edge_term=edge_term, initial_decoded_edges=initial_decoded_edges
            )

        # Phase 3: If corrections succeeded, enumerate and rank valid graphs via pattern matching
        if correction_level != CorrectionLevel.FAIL:
            top_k_graphs: list[tuple[nx.Graph, float]] = []

            if correction_level == CorrectionLevel.ZERO:
                # We only have the initial set
                decoded_edges = list(filter(self._is_feasible_set, decoded_edges))
            elif prefer_smaller_corrective_edits:
                if correction_results.add_edit_count <= correction_results.remove_edit_count:
                    # Prefer ADD, fallback to REMOVE
                    decoded_edges = list(filter(self._is_feasible_set, correction_results.add_sets))
                    if not decoded_edges:
                        decoded_edges = list(filter(self._is_feasible_set, correction_results.remove_sets))
                else:
                    # Prefer REMOVE, fallback to ADD
                    decoded_edges = list(filter(self._is_feasible_set, correction_results.remove_sets))
                    if not decoded_edges:
                        decoded_edges = list(filter(self._is_feasible_set, correction_results.add_sets))
            else:
                # No preference, use all feasible sets
                feasible_add_sets = list(filter(self._is_feasible_set, correction_results.add_sets))
                feasible_remove_sets = list(filter(self._is_feasible_set, correction_results.remove_sets))
                decoded_edges = feasible_add_sets + feasible_remove_sets

            if len(decoded_edges) == 0:
                # All corrected edge multisets are infeasible
                return DecodingResult()

            # Distribute iteration budget across edge multisets
            iteration_budget = max(1, iteration_budget // len(decoded_edges))

            # For each valid edge multiset, sample and rank graph candidates
            for edge_multiset in decoded_edges:
                graphs_from_multiset = self._find_top_k_isomorphic_graphs(
                    edge_multiset=edge_multiset,
                    graph_term=graph_term,
                    iteration_budget=iteration_budget,
                    max_graphs_per_iter=max_graphs_per_iter,
                    top_k=top_k,
                    sim_eps=sim_eps,
                    use_early_stopping=use_early_stopping,
                )
                top_k_graphs.extend(graphs_from_multiset)

            if len(top_k_graphs) == 0:
                # No valid graphs enumerated even with correction
                return DecodingResult()

            # Sort all candidates by similarity (descending) and take top k
            top_k_graphs = sorted(top_k_graphs, key=lambda x: x[1], reverse=True)[:top_k]
            nx_graphs, cos_sims = [], []
            if top_k_graphs:
                nx_graphs, cos_sims = zip(*top_k_graphs, strict=False)

            return DecodingResult(
                nx_graphs=list(nx_graphs),
                final_flags=[True] * len(top_k_graphs),
                cos_similarities=list(cos_sims),
                target_reached=True,
                correction_level=correction_level,
            )

        # All corrections failed - fallback to greedy decoder
        return self.decode_graph_greedy(
            edge_term=edge_term,
            graph_term=graph_term,
            node_counter=None,
            decoder_settings=fallback_decoder_settings,
        )

    def decode_graph_greedy(
        self,
        edge_term: torch.Tensor,
        graph_term: torch.Tensor,
        node_counter: Counter | None = None,
        decoder_settings: FallbackDecoderSettings | None = None,
    ) -> DecodingResult:
        """
        Greedy beam search decoder for graph reconstruction.

        This is a fallback decoder used when the main decoding strategy fails. It performs
        beam search in the graph construction space, incrementally adding nodes while
        maintaining the highest-scoring partial graphs according to cosine similarity
        with the target graph_term.

        Key improvements:
        - Returns top_k graphs sorted by cosine similarity (consistent with main decoder)
        - Final ranking ensures best candidates are returned first

        Parameters
        ----------
        edge_term : torch.Tensor
            Hypervector representing the edge structure.
        graph_term : torch.Tensor
            Hypervector representing the full graph for similarity comparison.
        node_counter : Counter, optional
            Pre-computed node count information (for 2D/3D vectors with G0 encoded).
        decoder_settings : dict, optional
            Configuration parameters:
            - top_k: int (default: 10)
              Number of top-scoring graphs to return.
            - beam_size: int
              Beam width during search.
            - initial_limit: int (default: 1024)
              Population size limit.
            - pruning_fn: str (default: "cos_sim")
              Similarity function for pruning.
            - use_g3_instead_of_h3: bool (default: False)
              Whether to use combined terms for similarity.

        Returns
        -------
        DecodingResult
            Object containing:
            - nx_graphs: Top-k NetworkX graphs sorted by similarity (descending)
            - final_flags: Completion status for each graph
            - target_reached: Whether valid graph was decoded
            - correction_level: Always CorrectionLevel.FAIL (greedy fallback)
        """
        if decoder_settings is None:
            decoder_settings = FallbackDecoderSettings()
        validate_ring_structure = decoder_settings.validate_ring_structure
        random_sample_ratio = decoder_settings.random_sample_ratio
        use_modified_graph_embedding = decoder_settings.use_modified_graph_embedding
        graph_embedding_attr = decoder_settings.graph_embedding_attr

        # Case 2D/3D vectors with G0 encoded
        if node_counter:
            decoded_edges = self.decode_order_one(edge_term=edge_term, node_counter=node_counter)
            edge_count = sum([(t[1] + 1) * n for t, n in node_counter.items()])
        else:
            decoded_edges = self.decode_order_one_no_node_terms(edge_term=edge_term.clone())
            edge_count = len(decoded_edges) // 2  # bidirectional edges
            node_counter = get_node_counter(decoded_edges, method="ceil")
            if not target_reached(decoded_edges):
                decoded_edges = self.decode_order_one(edge_term=edge_term.clone(), node_counter=node_counter)

        if self.base_dataset == "qm9":
            node_limit = MAX_ALLOWED_DECODING_NODES_QM9
        elif self.base_dataset in _PUBCHEM_DECODING_LIMITS:
            node_limit = _PUBCHEM_DECODING_LIMITS[self.base_dataset][0]
        else:
            node_limit = MAX_ALLOWED_DECODING_NODES_ZINC
        if node_counter.total() > node_limit:
            return DecodingResult(correction_level=CorrectionLevel.FAIL)

        node_count = node_counter.total()
        ## We have the multiset of nodes and the multiset of edges
        # OPTIMIZATION: Convert decoded_edges to Counter for O(1) lookups throughout
        decoded_edges_counter = Counter(decoded_edges)

        # OPTIMIZATION: Store Counter instead of list in population tuples for O(1) operations
        first_pop: list[tuple[nx.Graph, Counter]] = []
        global_seen: set = set()
        for k, (u_t, v_t) in enumerate(decoded_edges):
            G = nx.Graph()
            uid = add_node_with_feat(G, Feat.from_tuple(u_t), raw_type=u_t)
            ok = add_node_and_connect(G, Feat.from_tuple(v_t), connect_to=[uid], total_nodes=node_count, raw_type=v_t) is not None
            if not ok:
                continue
            key = _hash(G)
            if key in global_seen:
                continue
            global_seen.add(key)

            # OPTIMIZATION: Store Counter directly - no conversion to list needed
            remaining_edges_counter = decoded_edges_counter.copy()
            remaining_edges_counter[(u_t, v_t)] -= 1
            remaining_edges_counter[(v_t, u_t)] -= 1

            first_pop.append((G, remaining_edges_counter))

        pruning_fn = decoder_settings.pruning_fn

        def get_similarities(a, b):
            if pruning_fn != "cos_sim":
                diff = a[:, None, :] - b[None, :, :]
                return torch.sum(diff**2, dim=-1)
            return torchhd.cos(a, b)

        initial_limit = decoder_settings.initial_limit
        use_size_aware_pruning = decoder_settings.use_size_aware_pruning
        if decoder_settings.use_one_initial_population:
            # Start with a child both anchors free
            selected = [(G, l) for G, l in first_pop if len(anchors(G)) == 2]
            first_pop = selected[:1] if len(selected) >= 1 else first_pop[:1]
        population = first_pop

        for _ in tqdm(range(2, node_count)):
            children: list[tuple[nx.Graph, Counter]] = []

            # Expand the current population
            for gi, (G, edges_left) in enumerate(population):
                leftovers_ctr = leftover_features(node_counter, G)
                if not leftovers_ctr:
                    continue

                leftover_types = order_leftovers_by_degree_distinct(leftovers_ctr)
                ancrs = anchors(G)
                if not ancrs:
                    continue

                # Choose the first N anchors to expand on
                lowest_degree_ancrs = sorted(ancrs, key=lambda n: residual_degree(G, n))[:1]

                # Try to connect the left over nodes to the lowest degree anchors
                for a, lo_t in list(itertools.product(lowest_degree_ancrs, leftover_types)):
                    a_t = G.nodes[a]["type"]
                    # OPTIMIZATION: Use Counter for O(1) lookup (no set conversion needed)
                    if edges_left[(a_t, lo_t)] == 0:
                        continue

                    # OPTIMIZATION: Use nx.Graph(G) instead of G.copy() for faster copying
                    C = nx.Graph(G)
                    nid = add_node_and_connect(C, Feat.from_tuple(lo_t), connect_to=[a], total_nodes=node_count, raw_type=lo_t)
                    if nid is None:
                        continue
                    if C.number_of_edges() > edge_count:
                        continue

                    keyC = _hash(C)
                    if keyC in global_seen:
                        continue

                    # # Early pruning of bad ring structures
                    if (
                        validate_ring_structure
                        and self.base_dataset in ("zinc", "pubchem16", "pubchem32", "pubchem64")
                        and not has_valid_ring_structure(
                            G=C,
                            processed_histogram=self.dataset_info.ring_histogram,
                            single_ring_atom_types=self.dataset_info.single_ring_features,
                            is_partial=True,  # Graph is still being constructed
                        )
                    ):
                        continue

                    # self._print_and_plot(g=C, graph_terms=graph_term)

                    # OPTIMIZATION: Use Counter arithmetic for O(1) edge removal
                    remaining_edges = edges_left.copy()
                    remaining_edges[(a_t, lo_t)] -= 1
                    remaining_edges[(lo_t, a_t)] -= 1
                    global_seen.add(keyC)
                    children.append((C, remaining_edges))

                    ancrs_rest = [a_ for a_ in ancrs if a_ != a]

                    # OPTIMIZATION: remaining_edges is already a Counter, use it directly
                    nid_t = C.nodes[nid]["type"]

                    for subset in powerset(ancrs_rest):
                        if len(subset) == 0:
                            continue

                        # OPTIMIZATION: Build all_new_connection and validate using Counter
                        all_new_connection = []
                        subset_ts = [C.nodes[s]["type"] for s in subset]
                        should_continue = False
                        for st in subset_ts:
                            ts = (nid_t, st)
                            # Check if edge exists in remaining_edges Counter (O(1))
                            if remaining_edges[ts] == 0:
                                should_continue = True
                                break
                            all_new_connection.append(ts)

                        if should_continue:
                            continue

                        # OPTIMIZATION: Validate edge counts using Counter
                        all_new_counter = Counter(all_new_connection)
                        # if both ends of an edge is the same tuple, it should be considered twice
                        for k, v in all_new_counter.items():
                            if k[0] == k[1]:
                                all_new_counter[k] = 2 * v

                        # Check if we have enough edges in remaining_edges
                        for k, v in all_new_counter.items():
                            if remaining_edges[k] < v:
                                should_continue = True
                                break

                        if should_continue:
                            continue

                        # OPTIMIZATION: Use nx.Graph(C) instead of C.copy()
                        H = nx.Graph(C)
                        new_nid = connect_all_if_possible(H, nid, connect_to=list(subset), total_nodes=node_count)
                        if new_nid is None:
                            continue
                        if H.number_of_edges() > edge_count:
                            continue

                        keyH = _hash(H)
                        if keyH in global_seen:
                            continue

                        # # Early pruning of bad ring structures
                        if (
                            validate_ring_structure
                            and self.base_dataset == "zinc"
                            and not has_valid_ring_structure(
                                G=H,
                                processed_histogram=self.dataset_info.ring_histogram,
                                single_ring_atom_types=self.dataset_info.single_ring_features,
                                is_partial=True,  # Graph is still being constructed
                            )
                        ):
                            continue

                        # OPTIMIZATION: Use Counter arithmetic for O(1) batch edge removal
                        remaining_edges_ = remaining_edges.copy()
                        for a_t, b_t in all_new_connection:
                            remaining_edges_[(a_t, b_t)] -= 1
                            remaining_edges_[(b_t, a_t)] -= 1

                        # self._print_and_plot(g=H, graph_terms=graph_term)

                        global_seen.add(keyH)
                        children.append((H, remaining_edges_))

            ## Collect the children with highest number of edges
            if not children:
                # Extract top_k parameter from decoder_settings
                top_k = decoder_settings.top_k if decoder_settings.top_k is not None else 10

                graphs, edges_left = zip(*population, strict=True)
                # OPTIMIZATION: Use Counter.total() to check if empty
                are_final = [i.total() == 0 for i in edges_left]

                # Compute cosine similarities for all graphs
                enc_out = self._batched_forward(graphs)
                g_terms = enc_out[graph_embedding_attr]
                if decoder_settings.use_g3_instead_of_h3:
                    g_terms = enc_out["node_terms"] + enc_out["edge_terms"] + g_terms

                # Compute similarities and sort
                sims = torchhd.cos(graph_term, g_terms)
                sim_order = torch.argsort(sims, descending=True)

                # Select top_k graphs based on similarity
                top_k_indices = sim_order[: min(top_k, len(graphs))].cpu().numpy()
                top_k_graphs = [graphs[i] for i in top_k_indices]
                top_k_flags = [are_final[i] for i in top_k_indices]
                top_cos_sims = [sims[i].item() for i in top_k_indices]

                # Convert Counter to list for target_reached function
                decoded_edges_list = []
                for edge, count in decoded_edges_counter.items():
                    if count > 0:
                        decoded_edges_list.extend([edge] * count)

                return DecodingResult(
                    nx_graphs=top_k_graphs,
                    final_flags=top_k_flags,
                    cos_similarities=top_cos_sims,
                    target_reached=target_reached(decoded_edges_list),
                    correction_level=CorrectionLevel.FAIL,
                )

            if len(children) > initial_limit:
                initial_limit = decoder_settings.limit
                beam_size = decoder_settings.beam_size

                # Calculate split: top-k by similarity + random from rest
                keep = int((1 - random_sample_ratio) * beam_size)
                random_pick = int(random_sample_ratio * beam_size)

                if use_size_aware_pruning:
                    repo = defaultdict(list)

                    # Prune for each size separately
                    for c, l in children:
                        repo[c.number_of_edges()].append((c, l))

                    res = []
                    for ch in [v for _, v in repo.items()]:
                        # Encode and compute similarity
                        enc_out = self._batched_forward([c for c, _ in ch])

                        g_terms = enc_out[graph_embedding_attr]
                        if decoder_settings.use_g3_instead_of_h3:
                            g_terms = enc_out["node_terms"] + enc_out["edge_terms"] + g_terms

                        if use_modified_graph_embedding:
                            # Modify graph_term for each child based on its leftover edges
                            # This provides fairer comparison by accounting for edges not yet added
                            sims_list = []

                            for idx, (c, leftover_edges_counter) in enumerate(ch):
                                if leftover_edges_counter and leftover_edges_counter.total() > 0:
                                    # Convert Counter to list for encode_edge_multiset
                                    leftover_edges = []
                                    for edge, count in leftover_edges_counter.items():
                                        if count > 0:
                                            leftover_edges.extend([edge] * count)
                                    # Encode the leftover edges into a hypervector
                                    leftover_edge_term = self.encode_edge_multiset(leftover_edges)
                                    # Subtract from target to get modified graph_term
                                    modified_graph_term = graph_term - leftover_edge_term
                                else:
                                    # No leftover edges, use original graph_term
                                    modified_graph_term = graph_term

                                # Compute element-wise similarity for this child
                                child_sim = get_similarities(modified_graph_term, g_terms[idx].unsqueeze(0))
                                sims_list.append(child_sim)

                            # Concatenate all similarities into a single tensor
                            sims = torch.cat(sims_list)
                        else:
                            # Original behavior: compare all to the same graph_term
                            sims = get_similarities(graph_term, g_terms)

                        # Sort by similarity first
                        sim_order = torch.argsort(sims, descending=True)

                        # Take top 'keep' by similarity
                        top_candidates = [ch[i.item()] for i in sim_order[:keep]]
                        res.extend(top_candidates)

                        # Randomly pick 'random_pick' from the rest
                        if random_pick > 0 and len(ch) > keep:
                            remaining_indices = sim_order[keep:].cpu().numpy()
                            if len(remaining_indices) > 0:
                                n_random = min(random_pick, len(remaining_indices))
                                random_indices = np.random.choice(remaining_indices, size=n_random, replace=False)
                                random_candidates = [ch[i] for i in random_indices]
                                res.extend(random_candidates)
                    children = res
                else:
                    # Encode and compute similarity
                    enc_out = self._batched_forward([c for c, _ in children])
                    g_terms = enc_out[graph_embedding_attr]
                    if decoder_settings.use_g3_instead_of_h3:
                        g_terms = enc_out["node_terms"] + enc_out["edge_terms"] + g_terms

                    if use_modified_graph_embedding:
                        # Modify graph_term for each child based on its leftover edges
                        # This provides fairer comparison by accounting for edges not yet added
                        sims_list = []

                        for idx, (c, leftover_edges_counter) in enumerate(children):
                            if leftover_edges_counter and leftover_edges_counter.total() > 0:
                                # Convert Counter to list for encode_edge_multiset
                                leftover_edges = []
                                for edge, count in leftover_edges_counter.items():
                                    if count > 0:
                                        leftover_edges.extend([edge] * count)
                                # Encode the leftover edges into a hypervector
                                leftover_edge_term = self.encode_edge_multiset(leftover_edges)
                                # Subtract from target to get modified graph_term
                                modified_graph_term = graph_term - leftover_edge_term
                            else:
                                # No leftover edges, use original graph_term
                                modified_graph_term = graph_term

                            # Compute element-wise similarity for this child
                            child_sim = get_similarities(modified_graph_term, g_terms[idx].unsqueeze(0))
                            sims_list.append(child_sim)

                        # Concatenate all similarities into a single tensor
                        sims = torch.cat(sims_list)
                    else:
                        # Original behavior: compare all to the same graph_term
                        sims = get_similarities(graph_term, g_terms)

                    # Sort by similarity first
                    sim_order = torch.argsort(sims, descending=True)

                    # Take top 'keep' by similarity
                    top_candidates = [children[i.item()] for i in sim_order[:keep]]

                    # Randomly pick 'random_pick' from the rest
                    result = top_candidates.copy()
                    if random_pick > 0 and len(children) > keep:
                        remaining_indices = sim_order[keep:].cpu().numpy()
                        if len(remaining_indices) > 0:
                            n_random = min(random_pick, len(remaining_indices))
                            random_indices = np.random.choice(remaining_indices, size=n_random, replace=False)
                            random_candidates = [children[i] for i in random_indices]
                            result.extend(random_candidates)

                    children = result

            population = children

        # Extract top_k parameter from decoder_settings (consistent with main decode_graph)
        top_k = decoder_settings.top_k if decoder_settings.top_k is not None else 10

        # Sort the final population by cosine similarity to graph_term
        graphs, edges_left = zip(*population, strict=True)
        # OPTIMIZATION: Use Counter.total() to check if empty
        are_final = [i.total() == 0 for i in edges_left]

        # Compute cosine similarities for all final graphs
        enc_out = self._batched_forward(graphs)
        g_terms = enc_out[graph_embedding_attr]
        if decoder_settings.use_g3_instead_of_h3:
            g_terms = enc_out["node_terms"] + enc_out["edge_terms"] + g_terms

        # Compute similarities and sort
        sims = torchhd.cos(graph_term, g_terms)
        sim_order = torch.argsort(sims, descending=True)

        # Select top_k graphs based on similarity
        top_k_indices = sim_order[: min(top_k, len(graphs))].cpu().numpy()
        top_k_graphs = [graphs[i] for i in top_k_indices]
        top_k_flags = [are_final[i] for i in top_k_indices]
        top_k_sims = [sims[i].item() for i in top_k_indices]

        # Convert Counter to list for target_reached function
        decoded_edges_list = []
        for edge, count in decoded_edges_counter.items():
            if count > 0:
                decoded_edges_list.extend([edge] * count)

        return DecodingResult(
            nx_graphs=top_k_graphs,
            final_flags=top_k_flags,
            cos_similarities=top_k_sims,
            target_reached=target_reached(decoded_edges_list),
            correction_level=CorrectionLevel.FAIL,
        )

    def decode_graph_astar(
        self,
        edge_term: torch.Tensor,
        graph_term: torch.Tensor,
        node_counter: Counter | None = None,
        decoder_settings: AStarDecoderSettings | None = None,
    ) -> DecodingResult:
        """
        A*-style best-first decoder for graph reconstruction.

        Alternative to ``decode_graph_greedy``. Each tree node is a partial
        graph; expansion produces children by either (i) adding an intra-graph
        edge between two existing atoms that still have open valence, or
        (ii) attaching a new atom to an existing anchor via a decoded edge
        type. At every step the highest-cosine-similarity frontier node is
        popped and expanded. Children are scored by encoding each partial
        graph through the HyperNet and computing cosine similarity against
        ``graph_term``. Search stops when a complete graph with
        ``cos_sim >= 1 - similarity_eps`` is reached or when the wall-clock
        ``budget_seconds`` is exhausted — whichever comes first.

        Parameters
        ----------
        edge_term : torch.Tensor
            Hypervector for edge unbinding (same shape as in greedy decoder).
        graph_term : torch.Tensor
            Full-graph hypervector used to score candidates.
        node_counter : Counter, optional
            Pre-decoded node multiset. Derived from edges if omitted.
        decoder_settings : AStarDecoderSettings, optional
            Search configuration (budget, batch cap, etc).

        Returns
        -------
        DecodingResult
            Same fields as ``decode_graph_greedy``. ``correction_level`` is
            always ``CorrectionLevel.FAIL`` (fallback-style output). If any
            complete candidate is found, it is ranked ahead of partials; on
            budget exhaustion with no complete candidate, the best partials
            seen are returned.
        """
        settings = decoder_settings or AStarDecoderSettings()
        eps = settings.similarity_eps
        top_k = settings.top_k
        max_bs = max(1, settings.max_batch_size)
        stats = AStarSearchStats()

        # ── Phase 0: decode edge multiset (mirrors greedy) ──────────
        # Phase 0 is a fixed preprocessing cost; `budget_seconds` applies to
        # the search loop only. t0 is therefore taken AFTER Phase 0.
        t_phase0_start = time.monotonic()
        if node_counter:
            decoded_edges = self.decode_order_one(edge_term=edge_term, node_counter=node_counter)
        else:
            decoded_edges = self.decode_order_one_no_node_terms(edge_term=edge_term.clone())
            if not decoded_edges:
                stats.seconds_phase0 = time.monotonic() - t_phase0_start
                return DecodingResult(correction_level=CorrectionLevel.FAIL, search_stats=stats)
            node_counter = get_node_counter(decoded_edges, method="ceil")
            if not target_reached(decoded_edges):
                decoded_edges = self.decode_order_one(edge_term=edge_term.clone(), node_counter=node_counter)
        stats.seconds_phase0 = time.monotonic() - t_phase0_start

        if self.base_dataset == "qm9":
            node_limit = MAX_ALLOWED_DECODING_NODES_QM9
        elif self.base_dataset in _PUBCHEM_DECODING_LIMITS:
            node_limit = _PUBCHEM_DECODING_LIMITS[self.base_dataset][0]
        else:
            node_limit = MAX_ALLOWED_DECODING_NODES_ZINC
        if node_counter.total() > node_limit:
            return DecodingResult(correction_level=CorrectionLevel.FAIL, search_stats=stats)

        target_nodes = node_counter.total()
        if target_nodes == 0 or not decoded_edges:
            return DecodingResult(correction_level=CorrectionLevel.FAIL, search_stats=stats)

        edge_count = len(decoded_edges) // 2  # decoded_edges holds both directions
        decoded_edges_counter: Counter = Counter(decoded_edges)
        decoded_edges_list = [e for e, c in decoded_edges_counter.items() for _ in range(c)]

        # Search-budget timer starts AFTER Phase 0 preprocessing.
        t0 = time.monotonic()

        # Ring feature index: auto-detect from tuple length when not set.
        # ZINC/PubChem have 5-feature tuples with the in-ring bit at position 4;
        # QM9 has 4-feature tuples with no ring info (ring_idx = -1 → no-op).
        if settings.ring_feature_index is not None:
            ring_idx = settings.ring_feature_index
        else:
            sample_tuple = next(iter(node_counter))
            ring_idx = 4 if len(sample_tuple) >= 5 else -1
        ring_hard_prune = settings.ring_hard_prune and ring_idx >= 0
        ring_penalty = settings.ring_penalty if ring_idx >= 0 else 0.0

        # ── Batched scoring with OOM-safe chunking ──────────────────
        def batch_score(graphs: list[nx.Graph]) -> list[float]:
            sims_out: list[float] = []
            for i in range(0, len(graphs), max_bs):
                chunk = graphs[i : i + max_bs]
                enc_out = self._batched_forward(chunk)
                stats.encoder_forwards += 1
                stats.encoder_graphs_scored += len(chunk)
                gt = enc_out["graph_embedding"]
                if settings.use_g3_instead_of_h3:
                    gt = enc_out["node_terms"] + enc_out["edge_terms"] + gt
                sims = torchhd.cos(graph_term, gt)
                sims_out.extend(sims.cpu().tolist())
            return sims_out

        # ── Phase 1: seed frontier with every distinct decoded edge ─
        seq = itertools.count()
        global_seen: set = set()
        heap: list = []

        # Seed selection — ordered by confidence.
        # `decoded_edges` from decode_order_one* is produced by iterative
        # argmax-unbinding, so entries 0, 2, 4, ... are the distinct edges in
        # DESCENDING unbinding-confidence order (entries 1, 3, 5, ... are the
        # reverse orientations of the same edges). We preserve that order so
        # that the top seed is the highest-confidence edge, and if the caller
        # allows more than one seed we add them in confidence-order.
        ordered_unique: list[tuple[tuple, tuple]] = []
        seen_unordered: set[frozenset] = set()
        for (u_t, v_t) in decoded_edges:
            key = frozenset((u_t, v_t)) if u_t != v_t else frozenset({(u_t, v_t)})
            if key in seen_unordered:
                continue
            seen_unordered.add(key)
            ordered_unique.append((u_t, v_t))

        # `num_seeds <= 0` means "use every distinct edge as a seed"
        seed_limit = len(ordered_unique) if settings.num_seeds <= 0 else settings.num_seeds

        seed_graphs: list[nx.Graph] = []
        seed_remaining: list[Counter] = []
        for (u_t, v_t) in ordered_unique[:seed_limit]:
            G = nx.Graph()
            uid = add_node_with_feat(G, Feat.from_tuple(u_t), raw_type=u_t)
            if add_node_and_connect(
                G, Feat.from_tuple(v_t), connect_to=[uid],
                total_nodes=target_nodes, raw_type=v_t,
            ) is None:
                continue
            h = _hash(G)
            if h in global_seen:
                continue
            global_seen.add(h)
            remaining = decoded_edges_counter.copy()
            remaining[(u_t, v_t)] -= 1
            remaining[(v_t, u_t)] -= 1
            seed_graphs.append(G)
            seed_remaining.append(remaining)

        if not seed_graphs:
            return DecodingResult(correction_level=CorrectionLevel.FAIL, search_stats=stats)

        def priority(sim: float, n_nodes: int) -> float:
            # Lower = popped first. Higher sim and larger n_nodes both lower priority.
            return -(sim + settings.depth_bias * (n_nodes / max(target_nodes, 1)))

        def adjusted_sim(raw_sim: float, G: nx.Graph, remaining: Counter) -> float:
            """Apply a constant valence-feasibility penalty to the raw sim when
            completion is provably impossible from this partial graph."""
            if settings.valence_penalty <= 0.0:
                return raw_sim
            if valence_satisfiable(G, remaining, node_counter):
                return raw_sim
            return raw_sim - settings.valence_penalty

        seed_sims = batch_score(seed_graphs)
        for G, rem, raw_s in zip(seed_graphs, seed_remaining, seed_sims, strict=True):
            # Seeds should also satisfy Rules B+C when ring_hard_prune is on
            # (a 2-atom seed can't violate Rule B, but Rule C could still
            # fire if the decoded edge budget is degenerate).
            if ring_hard_prune and not ring_membership_consistent(
                G, rem, node_counter, ring_idx,
            ):
                stats.ring_bc_rejected += 1
                continue
            s = adjusted_sim(raw_s, G, rem)
            heapq.heappush(heap, (priority(s, G.number_of_nodes()), next(seq), s, G, rem))
            stats.seeds_pushed += 1
        stats.heap_peak = max(stats.heap_peak, len(heap))

        # ── Main best-first loop ─────────────────────────────────────
        completes: list[tuple[float, nx.Graph]] = []
        # best_popped holds only EXPANDABLE pops — dead-ends and completes are
        # excluded so they can't poison the fallback ranking.
        best_popped: list[tuple[float, nx.Graph]] = []
        early_found = False

        while heap:
            _prio, _, sim, G, remaining = heapq.heappop(heap)
            stats.pops += 1

            n_here = G.number_of_nodes()
            all_edges_used = remaining.total() == 0
            at_target_n = n_here == target_nodes

            if at_target_n and all_edges_used:
                # Rule D: soft leaf-time ring-membership mismatch penalty.
                # Counts atoms whose declared in-ring bit disagrees with the
                # actual topology of the completed graph. Rules A/B/C already
                # forbid non-ring atoms on cycles, so in practice this only
                # fires for ring atoms that failed to end up on a cycle.
                final_sim = sim
                if ring_penalty > 0.0:
                    final_sim -= ring_penalty * count_ring_mismatches(G, ring_idx)
                completes.append((final_sim, G))
                stats.completes += 1
                if final_sim >= 1.0 - eps:
                    early_found = True
                    stats.early_exit = True
                    break
                continue

            # Dead-ends we can't expand out of
            if all_edges_used and not at_target_n:
                stats.dead_end_pops += 1
                continue  # no edges left to place remaining atoms

            # Only record genuinely expandable partials for the fallback pool
            best_popped.append((sim, G))

            children: list[tuple[nx.Graph, Counter]] = []

            # (c-i) intra-graph edge additions between existing atoms
            if settings.allow_intra_edges:
                nodes_list = list(G.nodes)
                for i_a in range(len(nodes_list)):
                    u = nodes_list[i_a]
                    if residual_degree(G, u) <= 0:
                        continue
                    u_t = G.nodes[u]["type"]
                    for i_b in range(i_a + 1, len(nodes_list)):
                        v = nodes_list[i_b]
                        if residual_degree(G, v) <= 0:
                            continue
                        if G.has_edge(u, v):
                            continue
                        v_t = G.nodes[v]["type"]
                        if remaining[(u_t, v_t)] <= 0:
                            continue
                        # Rule A: reject intra-edges whose new cycle would
                        # contain a declared non-ring atom. The new cycle's
                        # vertices are exactly the u-v path in G plus {u, v}.
                        if ring_hard_prune:
                            try:
                                path = nx.shortest_path(G, source=u, target=v)
                            except nx.NetworkXNoPath:
                                path = []
                            if any(
                                G.nodes[n]["type"][ring_idx] == 0 for n in path
                            ):
                                stats.rule_a_rejected += 1
                                continue
                        C = nx.Graph(G)
                        if not add_edge_if_possible(C, u, v, strict=True):
                            continue
                        if C.number_of_edges() > edge_count:
                            continue
                        new_rem = remaining.copy()
                        new_rem[(u_t, v_t)] -= 1
                        new_rem[(v_t, u_t)] -= 1
                        children.append((C, new_rem))
                        stats.children_generated += 1

            # (c-ii) new-atom attachments (single-edge, like the greedy inner step)
            if settings.allow_node_additions and not at_target_n:
                leftovers_ctr = leftover_features(node_counter, G)
                if leftovers_ctr:
                    leftover_types = order_leftovers_by_degree_distinct(leftovers_ctr)
                    for a in anchors(G):
                        a_t = G.nodes[a]["type"]
                        for lo_t in leftover_types:
                            if remaining[(a_t, lo_t)] <= 0:
                                continue
                            C = nx.Graph(G)
                            nid = add_node_and_connect(
                                C, Feat.from_tuple(lo_t), connect_to=[a],
                                total_nodes=target_nodes, raw_type=lo_t,
                            )
                            if nid is None:
                                continue
                            if C.number_of_edges() > edge_count:
                                continue
                            new_rem = remaining.copy()
                            new_rem[(a_t, lo_t)] -= 1
                            new_rem[(lo_t, a_t)] -= 1
                            children.append((C, new_rem))
                            stats.children_generated += 1

            if not children:
                continue

            # Dedup by WL-hash across the whole search
            if settings.dedupe:
                fresh: list[tuple[nx.Graph, Counter]] = []
                for C, rem in children:
                    h = _hash(C)
                    if h in global_seen:
                        stats.dedup_rejected += 1
                        continue
                    global_seen.add(h)
                    fresh.append((C, rem))
            else:
                fresh = children

            # Rules B+C: drop children whose ring-membership constraints are
            # provably unsatisfiable (non-ring atom currently on cycle, or
            # pending ring-atoms with no remaining intra-edge budget).
            if ring_hard_prune:
                pre_ring = len(fresh)
                fresh = [
                    (C, rem) for C, rem in fresh
                    if ring_membership_consistent(C, rem, node_counter, ring_idx)
                ]
                stats.ring_bc_rejected += pre_ring - len(fresh)

            if not fresh:
                continue

            fresh_graphs = [c[0] for c in fresh]
            fresh_sims = batch_score(fresh_graphs)

            for (C, rem), raw_s in zip(fresh, fresh_sims, strict=True):
                s = adjusted_sim(raw_s, C, rem)
                heapq.heappush(heap, (priority(s, C.number_of_nodes()), next(seq), s, C, rem))
                stats.children_scored += 1

            if len(heap) > stats.heap_peak:
                stats.heap_peak = len(heap)

            # Optional safety valve: cap frontier size
            if settings.max_frontier_size is not None and len(heap) > settings.max_frontier_size:
                heap = heapq.nsmallest(settings.max_frontier_size, heap)
                heapq.heapify(heap)

            # Budget check at END of loop: guarantees at least one pop+expansion
            # happens per call, even if seeding consumed most of the budget.
            if time.monotonic() - t0 > settings.budget_seconds:
                stats.budget_exhausted = True
                break

        # ── Build result ─────────────────────────────────────────────
        if completes:
            completes.sort(key=lambda x: x[0], reverse=True)
            top = completes[:top_k]
            nx_graphs = [g for _, g in top]
            sims_out = [s for s, _ in top]
            final_flags = [True] * len(top)
        else:
            # No complete candidate reached — fall back to best partials seen.
            # Merge heap-remainder with popped history, dedup by graph hash.
            # Rank by (proximity to target_nodes, sim) so a partial with more
            # atoms outranks a shallower one with a slightly higher sim — this
            # prevents returning a 2-atom seed when we never made it to target.
            pool: dict[Any, tuple[float, nx.Graph]] = {}
            for _prio, _, s, G, remaining in heap:
                h = _hash(G)
                prev = pool.get(h)
                if prev is None or s > prev[0]:
                    pool[h] = (s, G)
            for s, G in best_popped:
                h = _hash(G)
                prev = pool.get(h)
                if prev is None or s > prev[0]:
                    pool[h] = (s, G)

            def _rank_key(item: tuple[float, nx.Graph]) -> tuple[int, float]:
                s, G = item
                # Prefer partials whose node count is closest to target; break
                # ties by higher cosine sim.
                return (-abs(G.number_of_nodes() - target_nodes), s)

            candidates = sorted(pool.values(), key=_rank_key, reverse=True)[:top_k]
            if not candidates:
                stats.seconds_search = time.monotonic() - t0
                stats.heap_final = len(heap)
                stats.unique_states_seen = len(global_seen)
                return DecodingResult(
                    correction_level=CorrectionLevel.FAIL, search_stats=stats,
                )
            nx_graphs = [g for _, g in candidates]
            sims_out = [s for s, _ in candidates]
            # A partial is only "final" if it consumed all decoded edges AND
            # hit target node count (i.e. a complete). Fallback candidates by
            # construction do not meet both; flag accordingly.
            final_flags = [False] * len(candidates)

        _ = early_found  # retained for potential future logging hooks
        stats.seconds_search = time.monotonic() - t0
        stats.heap_final = len(heap)
        stats.unique_states_seen = len(global_seen)
        return DecodingResult(
            nx_graphs=nx_graphs,
            final_flags=final_flags,
            cos_similarities=sims_out,
            target_reached=target_reached(decoded_edges_list),
            correction_level=CorrectionLevel.FAIL,
            search_stats=stats,
        )

    # ─────────────────────────── Distance ───────────────────────────

    def calculate_order_n_distance(self, order_n_a: Tensor, order_n_b: Tensor) -> Tensor:
        """
        Core distance metric: cosine distance between order-N embeddings.

        This is the low-level method that operates on **already-extracted**
        order-N (graph embedding) vectors.  Use this when you have the
        order-N portion in hand (e.g. from ``forward()["graph_embedding"]``).

        For ``MultiHyperNet`` the override splits the concatenated
        embedding into per-sub-HyperNet chunks and averages per-chunk
        distances.  A single ``HyperNet`` computes one cosine distance
        over the full vector.

        Parameters
        ----------
        order_n_a, order_n_b : Tensor
            Shape ``[batch, order_n_dim]``.  For a single HyperNet
            ``order_n_dim == hv_dim``; for a ``MultiHyperNet`` it equals
            ``ensemble_graph_dim`` (sum of all sub-HyperNet hv_dims).

        Returns
        -------
        Tensor
            Cosine distance ``1 - cos_sim``, shape ``[batch]``.
        """
        cos_sim = torch.nn.functional.cosine_similarity(
            order_n_a.float(), order_n_b.float(), dim=-1,
        )
        return 1.0 - cos_sim

    def calculate_distance(self, vec_a: Tensor, vec_b: Tensor) -> Tensor:
        """
        Convenience wrapper: cosine distance from full ``[order_0 | order_N]`` vectors.

        Strips the order-0 prefix (first ``hv_dim`` dims) and delegates to
        :meth:`calculate_order_n_distance`.

        Parameters
        ----------
        vec_a, vec_b : Tensor
            Shape ``[batch, 2 * hv_dim]`` (concatenated order-0 and order-N).

        Returns
        -------
        Tensor
            Cosine distance ``1 - cos_sim``, shape ``[batch]``.
        """
        return self.calculate_order_n_distance(
            vec_a[:, self.hv_dim :], vec_b[:, self.hv_dim :],
        )

    # ─────────────────────────── Save/Load ───────────────────────────

    def save(self, path: str | Path) -> None:
        """
        Save HyperNet to file.

        Parameters
        ----------
        path : str | Path
            Output file path
        """
        state = {
            "version": 2,
            "config": {
                "hv_dim": self.hv_dim,
                "depth": self.depth,
                "vsa": self.vsa.value,
                "seed": self.seed,
                "normalize": self.normalize,
                "base_dataset": self.base_dataset,
                "dtype": "float64" if self._dtype == torch.float64 else "float32",
                "rw_config": {
                    "enabled": self.rw_config.enabled,
                    "k_values": self.rw_config.k_values,
                    "num_bins": self.rw_config.num_bins,
                },
                "prune_codebook": self.prune_codebook,
                "normalize_graph_embedding": self.normalize_graph_embedding,
            },
            "encoder_maps": {
                "node": self._serialize_encoder_map(self.node_encoder_map),
                "edge": self._serialize_encoder_map(self.edge_encoder_map),
                "graph": self._serialize_encoder_map(self.graph_encoder_map),
            },
            "codebooks": {
                "nodes": self.nodes_codebook.cpu(),
                "edges": self._edges_codebook.cpu() if self._edges_codebook is not None else None,
                "edge_features": self.edge_feature_codebook.cpu() if self.edge_feature_codebook is not None else None,
            },
            "indexers": {
                "nodes": self.nodes_indexer.__dict__.copy(),
                "edges": self._edges_indexer.__dict__.copy() if self._edges_indexer is not None else None,
                "edge_features": self.edge_feature_indexer.__dict__.copy() if self.edge_feature_indexer else None,
            },
        }
        torch.save(state, path)

    def _serialize_encoder_map(self, encoder_map: EncoderMap) -> dict[str, Any]:
        """Serialize encoder map to dict."""
        result = {}
        for feat, (encoder, idx_range) in encoder_map.items():
            entry = {
                "encoder_class": encoder.__class__.__name__,
                "init_args": {
                    "dim": encoder.dim,
                    "vsa": encoder.vsa,
                    "num_categories": encoder.num_categories,
                    "idx_offset": encoder.idx_offset,
                    "dtype": encoder.dtype,
                },
                "index_range": idx_range,
                "codebook": encoder.codebook.cpu(),
            }
            if hasattr(encoder, "indexer") and encoder.indexer is not None:
                entry["indexer_state"] = encoder.indexer.__dict__.copy()
            result[feat.name] = entry
        return result

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "HyperNet":
        """
        Load HyperNet from file.

        Parameters
        ----------
        path : str | Path
            Path to saved HyperNet
        device : str | torch.device
            Device to load to

        Returns
        -------
        HyperNet
            Loaded HyperNet instance
        """
        state = torch.load(path, map_location="cpu", weights_only=False)
        device = torch.device(device)

        # Create an instance without calling __init__
        instance = cls.__new__(cls)
        pl.LightningModule.__init__(instance)

        # Restore config attributes
        cfg = state["config"]
        instance.hv_dim = cfg["hv_dim"]
        instance.depth = cfg["depth"]
        instance.vsa = VSAModel(cfg["vsa"])
        instance.seed = cfg["seed"]
        instance.normalize = cfg["normalize"]
        instance.base_dataset = cfg["base_dataset"]
        instance._dtype = torch.float64 if cfg.get("dtype", "float64") == "float64" else torch.float32
        rw_dict = cfg.get("rw_config")
        instance.rw_config = RWConfig(**rw_dict) if rw_dict else RWConfig()
        instance.prune_codebook = cfg.get("prune_codebook", True)
        instance.normalize_graph_embedding = cfg.get("normalize_graph_embedding", False)
        if instance.base_dataset == "qm9":
            instance._decoding_edge_limit = 50
        elif instance.base_dataset in _PUBCHEM_DECODING_LIMITS:
            instance._decoding_edge_limit = _PUBCHEM_DECODING_LIMITS[instance.base_dataset][1]
        else:
            instance._decoding_edge_limit = 122
        instance._max_step_delta = None
        instance._incremental_edges = False

        # Restore encoder maps
        instance.node_encoder_map = cls._deserialize_encoder_map(state["encoder_maps"]["node"], device)
        instance.edge_encoder_map = cls._deserialize_encoder_map(state["encoder_maps"]["edge"], device)
        instance.graph_encoder_map = cls._deserialize_encoder_map(state["encoder_maps"]["graph"], device)

        # Restore codebooks
        instance.nodes_codebook = state["codebooks"]["nodes"].to(device)
        if state["codebooks"]["edges"] is not None:
            instance._edges_codebook = state["codebooks"]["edges"].to(device)
        else:
            instance._edges_codebook = None
        instance.edge_feature_codebook = (
            state["codebooks"]["edge_features"].to(device)
            if state["codebooks"]["edge_features"] is not None
            else None
        )

        # Restore indexers
        instance.nodes_indexer = TupleIndexer.__new__(TupleIndexer)
        instance.nodes_indexer.__dict__.update(state["indexers"]["nodes"])

        if state["indexers"]["edges"] is not None:
            instance._edges_indexer = TupleIndexer.__new__(TupleIndexer)
            instance._edges_indexer.__dict__.update(state["indexers"]["edges"])
        else:
            instance._edges_indexer = None

        if state["indexers"]["edge_features"] is not None:
            instance.edge_feature_indexer = TupleIndexer.__new__(TupleIndexer)
            instance.edge_feature_indexer.__dict__.update(state["indexers"]["edge_features"])
        else:
            instance.edge_feature_indexer = None

        return instance

    @classmethod
    def _deserialize_encoder_map(cls, serialized: dict[str, Any], device: torch.device) -> EncoderMap:
        """Deserialize encoder map from dict."""
        encoder_class_map = {
            "CombinatoricIntegerEncoder": CombinatoricIntegerEncoder,
        }

        result = {}
        for feat_key, entry in serialized.items():
            args = dict(entry["init_args"])
            args["device"] = device

            enc_cls = encoder_class_map[entry["encoder_class"]]

            if "indexer_state" in entry:
                indexer = TupleIndexer.__new__(TupleIndexer)
                indexer.__dict__.update(entry["indexer_state"])
                encoder = enc_cls(**args, indexer=indexer)
            else:
                encoder = enc_cls(**args)

            encoder.codebook = entry["codebook"].to(device)

            feat_enum = Features[feat_key]
            result[feat_enum] = (encoder, tuple(entry["index_range"]))

        return result

    # ─────────────────────────── Utilities ───────────────────────────

    def ensure_vsa(self, t: torch.Tensor) -> VSATensor:
        """Ensure tensor is of the correct VSA type."""
        if isinstance(t, self.vsa.tensor_class):
            return t
        return t.as_subclass(self.vsa.tensor_class)
