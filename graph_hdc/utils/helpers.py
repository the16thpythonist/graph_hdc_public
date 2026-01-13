"""
General helper utilities for Graph HDC.
"""
import itertools
from collections.abc import Iterator, Sequence
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Union

import numpy as np
import torch
import torchhd
from torch import Tensor
from torch_geometric.utils import scatter
from torchhd.tensors.hrr import HRRTensor
from torchhd.tensors.map import MAPTensor

# ========= Paths =========
ROOT = Path(__file__).parent.parent.parent.absolute()
CHECKPOINTS_PATH = ROOT / "experiments" / "checkpoints"
DATA_PATH = ROOT / "data"


# ========= Device Selection =========
def pick_device() -> torch.device:
    """Select the best available device (CUDA > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def pick_device_str() -> str:
    """Return device name as string."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ========= JSON Serialization =========
def jsonable(x: Any) -> Any:
    """Convert Python objects to JSON-serializable types."""
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, (np.generic,)):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, torch.device):
        return str(x)
    if torch.is_tensor(x):
        return x.item() if x.numel() == 1 else x.detach().cpu().tolist()
    if isinstance(x, Enum):
        return x.value
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [jsonable(v) for v in x]
    return str(x)


# ========= TorchHD Utilities =========
ReductionOP = Literal["bind", "bundle"]


def scatter_hd(
    src: Tensor,
    index: Tensor,
    *,
    op: ReductionOP,
    dim_size: int | None = None,
) -> Tensor:
    """
    Scatter-reduce a batch of hypervectors along dim=0 using
    either torchhd.bind or torchhd.bundle.

    Args:
        src: Hypervector batch of shape [N, D]
        index: Bucket indices shape [N] in [0..dim_size)
        op: Either "bind" or "bundle"
        dim_size: Number of output buckets

    Returns:
        Scattered & reduced hypervectors of shape [dim_size, D]
    """
    index = index.to(src.device, dtype=torch.long, non_blocking=True)

    if dim_size is None:
        dim_size = int(index.max().item()) + 1

    # Dispatch on type and op
    reduce = ""
    if isinstance(src, MAPTensor):
        reduce = "sum" if op == "bundle" else "mul"
    elif isinstance(src, HRRTensor) and op == "bundle":
        reduce = "sum"

    if reduce:
        idx_dim = int(index.max().item()) + 1
        result = scatter(src, index, dim=0, dim_size=idx_dim, reduce=reduce)

        if (num_identity_vectors := dim_size - idx_dim) == 0:
            return result

        # Add identity vectors for missing indices
        from graph_hdc.hypernet.types import VSAModel
        vsa = VSAModel.HRR
        if VSAModel.MAP.value in repr(type(src)):
            vsa = VSAModel.MAP
        identities = torchhd.identity(
            num_vectors=num_identity_vectors,
            dimensions=src.shape[-1],
            vsa=vsa.value,
            device=src.device,
        )
        return torch.cat([result, identities])

    # Generic fallback: group rows manually
    buckets = [[] for _ in range(dim_size)]
    for i, b in enumerate(index.tolist()):
        buckets[b].append(src[i])

    op_hd = torchhd.multibind if op == "bind" else torchhd.multibundle
    out = []
    for bucket in buckets:
        if not bucket:
            identity = type(src).identity(1, src.shape[-1], device=src.device).squeeze(0)
            out.append(identity)
        else:
            reduced = op_hd(torch.stack(bucket, dim=0))
            out.append(reduced)
    return torch.stack(out, dim=0)


def cartesian_bind_tensor(tensors: list[torch.Tensor]) -> torch.Tensor:
    """
    Given a list of K hypervector sets, produce the full cross product
    and bind each K-tuple into one hypervector.

    Args:
        tensors: List of hypervector sets, each [Ni, D]

    Returns:
        [N_prod, D] tensor where N_prod = N1 * N2 * ... * NK
    """
    tensors = [t for t in tensors if t is not None]
    if not tensors:
        raise ValueError("Need at least one set")

    if len(tensors) == 1:
        t = tensors[0]
        if t.dim() == 1:
            return t.unsqueeze(-1)
        return t

    sizes = [t.shape[0] for t in tensors]
    idx_grids = torch.cartesian_prod(
        *[torch.arange(n, device=tensors[0].device) for n in sizes]
    )

    hv_list = []
    for k, t in enumerate(tensors):
        idxs = idx_grids[:, k]
        hv = t[idxs] if t.dim() != 1 else t[idxs].unsqueeze(-1)
        hv_list.append(hv)

    stacked = torch.stack(hv_list, dim=1)
    return torchhd.multibind(stacked)


def unbind(composite, factor):
    """
    Recover `other` from composite = bind(H, other), given H.

    Works for any VSA model: MAP, HRR, FHRR, VTB.
    """
    assert type(composite) is type(factor), "Both must be same VSATensor subclass"
    return torchhd.bind(composite, factor.inverse())


# ========= Tuple Indexer =========
class TupleIndexer:
    """Bijection between feature tuples and flat indices."""

    def __init__(self, sizes: Sequence[int]) -> None:
        sizes = [s for s in sizes if s]
        self.sizes = sizes
        self.idx_to_tuple: list[tuple[int, ...]] = (
            list(itertools.product(*(range(N) for N in sizes))) if sizes else []
        )
        self.tuple_to_idx: dict[tuple[int, ...], int] = (
            {t: idx for idx, t in enumerate(self.idx_to_tuple)} if sizes else {}
        )

    def get_tuple(self, idx: int) -> tuple[int, ...]:
        return self.idx_to_tuple[idx]

    def get_tuples(self, idxs: list[int]) -> list[tuple[int, ...]]:
        return [self.idx_to_tuple[idx] for idx in idxs]

    def get_idx(self, tup: Union[tuple[int, ...], int]) -> int | None:
        if isinstance(tup, int):
            return self.tuple_to_idx.get((tup,))
        return self.tuple_to_idx.get(tup)

    def get_idxs(self, tuples: list[Union[tuple[int, ...], int]]) -> list[int]:
        return [self.get_idx(tup) for tup in tuples]

    def size(self) -> int:
        return len(self.idx_to_tuple)

    def get_sizes(self) -> Sequence[int]:
        return self.sizes


# ========= Data Transformation =========
class DataTransformer:
    """Convert between NetworkX and PyTorch Geometric formats."""

    @staticmethod
    def nx_to_pyg(G, node_attr: str = "feat"):
        """
        Convert NetworkX graph to PyG Data.

        Parameters
        ----------
        G : nx.Graph
            NetworkX graph with node features
        node_attr : str
            Node attribute name ("feat" or "type")

        Returns
        -------
        Data
            PyG Data object with x and edge_index
        """
        from torch_geometric.data import Data

        nodes = sorted(G.nodes)
        idx_of = {n: i for i, n in enumerate(nodes)}

        # Auto-detect attribute
        if nodes:
            if "feat" in G.nodes[nodes[0]]:
                node_attr = "feat"
            elif "type" in G.nodes[nodes[0]]:
                node_attr = "type"

        # Extract features
        if node_attr == "feat":
            feats = [list(G.nodes[n][node_attr].to_tuple()) for n in nodes]
        else:
            feats = [list(G.nodes[n][node_attr]) for n in nodes]
        x = torch.tensor(feats, dtype=torch.float)

        # Edges (bidirectional)
        src, dst = [], []
        for u, v in G.edges():
            iu, iv = idx_of[u], idx_of[v]
            src.extend([iu, iv])
            dst.extend([iv, iu])
        edge_index = (
            torch.tensor([src, dst], dtype=torch.long)
            if src
            else torch.empty((2, 0), dtype=torch.long)
        )

        return Data(x=x, edge_index=edge_index)

    @staticmethod
    def nx_to_pyg_with_type_attr(G):
        """
        Convert NetworkX graph to PyG Data using 'type' attribute.

        This method specifically handles graphs where nodes have a 'type'
        attribute containing tuples (instead of Feat objects).

        Parameters
        ----------
        G : nx.Graph
            NetworkX graph with 'type' node attributes

        Returns
        -------
        Data
            PyG Data object with x and edge_index
        """
        from torch_geometric.data import Data

        nodes = sorted(G.nodes)
        idx_of = {n: i for i, n in enumerate(nodes)}

        # Extract features from 'type' attribute
        feats = []
        for n in nodes:
            node_data = G.nodes[n]
            if "feat" in node_data:
                feat = node_data["feat"]
                if hasattr(feat, "to_tuple"):
                    feats.append(list(feat.to_tuple()))
                else:
                    feats.append(list(feat))
            elif "type" in node_data:
                feats.append(list(node_data["type"]))
            else:
                raise ValueError(f"Node {n} has no 'feat' or 'type' attribute")

        x = torch.tensor(feats, dtype=torch.float)

        # Edges (bidirectional)
        src, dst = [], []
        for u, v in G.edges():
            iu, iv = idx_of[u], idx_of[v]
            src.extend([iu, iv])
            dst.extend([iv, iu])
        edge_index = (
            torch.tensor([src, dst], dtype=torch.long)
            if src
            else torch.empty((2, 0), dtype=torch.long)
        )

        return Data(x=x, edge_index=edge_index)

    @staticmethod
    def pyg_to_nx(
        data,
        *,
        strict_undirected: bool = True,
        allow_self_loops: bool = False,
    ):
        """
        Convert a PyG Data (undirected, bidirectional edges) to a mutable NetworkX graph.

        Assumptions
        ----------
        - data.x has shape [N, 4] or [N, 5] with integer-encoded features:
          [atom_type, degree_idx, formal_charge_idx, explicit_hs, (optional) is_in_ring].
        - data.edge_index is bidirectional (both (u,v) and (v,u) present) for undirected graphs.
        - Features are frozen and represent the final target degrees.

        Node attributes
        ---------------
        - feat: Feat instance (constructed from the 4 or 5-tuple).
        - target_degree: feat.target_degree (== degree_idx + 1).

        Parameters
        ----------
        data : Data
            PyG data object.
        strict_undirected : bool, optional
            If True, assert that edge_index is symmetric.
        allow_self_loops : bool, optional
            If False, drop self-loops.

        Returns
        -------
        nx.Graph
            Mutable undirected graph with node attributes.

        Raises
        ------
        ValueError
            If feature dimensionality is not [N, 4] or [N, 5] or edges are not symmetric when required.
        """
        import networkx as nx
        from torch_geometric.data import Data
        from graph_hdc.hypernet.types import Feat

        if data.x is None:
            raise ValueError("data.x is None (expected [N,4] or [N,5] features).")
        if data.edge_index is None:
            raise ValueError("data.edge_index is None.")

        x = data.x
        if x.dim() != 2 or x.size(1) not in [4, 5]:
            raise ValueError(f"Expected data.x shape [N,4] or [N,5], got {tuple(x.size())}.")

        # Ensure integer features
        if not torch.is_floating_point(x):
            x_int = x.to(torch.long)
        else:
            x_int = x.to(torch.long)  # safe cast, encoding is discrete

        N = x_int.size(0)
        ei = data.edge_index
        if ei.dim() != 2 or ei.size(0) != 2:
            raise ValueError("edge_index must be [2, E].")
        src = ei[0].to(torch.long)
        dst = ei[1].to(torch.long)

        # Build undirected edge set (dedup, optional self-loop handling)
        pairs = set()
        for u, v in zip(src.tolist(), dst.tolist(), strict=False):
            if u == v and not allow_self_loops:
                continue
            a, b = (u, v) if u < v else (v, u)
            pairs.add((a, b))

        if strict_undirected:
            # Check that for every (u,v) there is a (v,u) in the original directed list
            dir_pairs = set(zip(src.tolist(), dst.tolist(), strict=False))
            for u, v in list(pairs):
                if (u, v) not in dir_pairs or (v, u) not in dir_pairs:
                    raise ValueError(f"edge_index is not symmetric for undirected edge ({u},{v}).")

        # Construct NX graph
        G = nx.Graph()
        G.add_nodes_from(range(N))

        # Attach features and target degrees
        for n in range(N):
            t = tuple(int(z) for z in x_int[n].tolist())
            f = Feat.from_tuple(t)
            G.nodes[n]["feat"] = f
            G.nodes[n]["target_degree"] = f.target_degree

        # Add edges
        G.add_edges_from(pairs)
        return G


# ========= File Utilities =========
def find_files(
    start_dir: str | Path,
    prefixes: tuple[str, ...] = (),
    skip_substrings: tuple[str, ...] | None = None,
    desired_ending: str | None = ".ckpt",
) -> Iterator[Path]:
    """Yield files from start_dir matching given conditions."""
    for p in Path(start_dir).rglob("*"):
        if not p.is_file():
            continue
        if prefixes and not p.name.startswith(prefixes):
            continue
        if skip_substrings and any(s in str(p) for s in skip_substrings):
            continue
        if desired_ending and not p.name.endswith(desired_ending):
            continue
        yield p
