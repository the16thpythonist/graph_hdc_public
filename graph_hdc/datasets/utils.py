"""
Dataset utilities for graph_hdc.

Provides:
- get_split(): Load QM9 or ZINC dataset splits
- post_compute_encodings(): Compute HDC encodings on-the-fly
- DatasetInfo: Node/edge feature information
"""

from __future__ import annotations

import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from graph_hdc.datasets.qm9_smiles import QM9Smiles
from graph_hdc.datasets.zinc_smiles import ZincSmiles

if TYPE_CHECKING:
    from graph_hdc.hypernet import HyperNet
    from graph_hdc.hypernet.configs import RWConfig


def get_split(
    split: Literal["train", "valid", "test"],
    dataset: Literal["qm9", "zinc"] = "qm9",
) -> InMemoryDataset:
    """
    Load a dataset split.

    Disconnected molecules are automatically filtered during dataset processing.

    Parameters
    ----------
    split : str
        One of {"train", "valid", "test"}
    dataset : str
        One of {"qm9", "zinc"}

    Returns
    -------
    InMemoryDataset
        The loaded dataset
    """
    if dataset == "qm9":
        return QM9Smiles(split=split)
    elif dataset == "zinc":
        return ZincSmiles(split=split)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'qm9' or 'zinc'.")


@dataclass
class DatasetInfo:
    """Aggregated information about a graph dataset."""

    node_features: set[tuple]
    """Set of all node feature tuples in the dataset."""

    edge_features: set[tuple[tuple, tuple]]
    """Set of all edge tuples (sorted by node features)."""

    ring_histogram: dict[tuple, dict[int, int]] | None
    """For ZINC: node feature → ring size → count."""

    single_ring_features: set[tuple] | None
    """For ZINC: node features that appear only in single rings."""


def get_dataset_info(dataset: Literal["qm9", "zinc"]) -> DatasetInfo:
    """
    Get or compute dataset information (node/edge features, ring info).

    Results are cached to the dataset's processed directory.
    """
    if dataset == "qm9":
        dataset_cls = QM9Smiles
    elif dataset == "zinc":
        dataset_cls = ZincSmiles
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    cache_file = Path(dataset_cls(split="test").processed_dir) / "dataset_info.pkl"

    # Try loading from cache
    if cache_file.is_file():
        try:
            with open(cache_file, "rb") as f:
                info_dict = pickle.load(f)
            if isinstance(info_dict, dict) and "node_features" in info_dict:
                return DatasetInfo(**info_dict)
        except (pickle.UnpicklingError, EOFError, TypeError):
            pass

    # Compute from scratch
    print(f"Computing dataset info for {dataset}...")
    node_features: set[tuple] = set()
    edge_features: set[tuple[tuple, tuple]] = set()
    ring_histogram: dict[tuple, Counter] = defaultdict(Counter)
    atom_tuple_total_counts: Counter = Counter()
    never_multiple_rings_counter: Counter = Counter()

    for split in ["train", "valid", "test"]:
        ds = get_split(split=split, dataset=dataset)
        print(f"  Processing {split} ({len(ds)} graphs)...")

        for data in ds:
            # Node features
            current_node_features = {tuple(feat.tolist()) for feat in data.x.int()}
            node_features.update(current_node_features)

            # Edge features
            node_idx_to_tuple = {i: tuple(data.x[i].int().tolist()) for i in range(data.x.size(0))}
            for u, v in data.edge_index.T.tolist():
                feat_u = node_idx_to_tuple[u]
                feat_v = node_idx_to_tuple[v]
                edge_features.add(tuple(sorted((feat_u, feat_v))))

            # Ring info (ZINC only)
            if dataset != "zinc":
                continue

            mol = Chem.MolFromSmiles(data.smiles)
            if mol is None:
                continue

            instance_multiple_rings = set()
            atom_tuples_in_rings = {}

            for atom in mol.GetAtoms():
                if atom.IsInRing():
                    atom_idx = atom.GetIdx()
                    if atom_idx >= len(node_idx_to_tuple):
                        continue

                    atom_tuple = node_idx_to_tuple[atom_idx]
                    atom_tuples_in_rings[atom_idx] = atom_tuple
                    atom_tuple_total_counts[atom_tuple] += 1

                    ring_count = 0
                    for ring_size in range(3, 21):
                        if atom.IsInRingSize(ring_size):
                            ring_histogram[atom_tuple][ring_size] += 1
                            ring_count += 1

                    if ring_count > 1:
                        instance_multiple_rings.add(atom_tuple)

            for atom_tuple in atom_tuples_in_rings.values():
                if atom_tuple not in instance_multiple_rings:
                    never_multiple_rings_counter[atom_tuple] += 1

    # Finalize ring info
    final_ring_histogram: dict[tuple, dict[int, int]] | None = None
    single_ring_features: set[tuple] | None = None

    if dataset == "zinc":
        single_ring_features = set()
        for atom_tuple, total_count in atom_tuple_total_counts.items():
            if never_multiple_rings_counter[atom_tuple] == total_count:
                single_ring_features.add(atom_tuple)
        final_ring_histogram = {k: dict(v) for k, v in ring_histogram.items()}

    # Save to cache
    saved_dict = {
        "node_features": node_features,
        "edge_features": edge_features,
        "ring_histogram": final_ring_histogram,
        "single_ring_features": single_ring_features,
    }
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(saved_dict, f)

    return DatasetInfo(**saved_dict)


@torch.no_grad()
def post_compute_encodings(
    dataset: InMemoryDataset,
    hypernet: "HyperNet",
    *,
    batch_size: int = 1024,
    device: torch.device | None = None,
    normalize_graph: bool = True,
) -> list[Data]:
    """
    Compute HDC encodings for a dataset on-the-fly.

    Parameters
    ----------
    dataset : InMemoryDataset
        The dataset to encode
    hypernet : HyperNet
        The hyperdimensional encoder
    batch_size : int
        Batch size for encoding
    device : torch.device
        Device for computation
    normalize_graph : bool
        Whether to normalize graph embeddings

    Returns
    -------
    list[Data]
        List of Data objects with edge_terms and graph_terms attributes
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    hypernet = hypernet.to(device)
    hypernet.eval()

    augmented: list[Data] = []

    for batch in tqdm(loader, desc="Encoding", unit="batch"):
        batch = batch.to(device)
        out = hypernet.forward(batch, normalize=normalize_graph)

        graph_terms = out["graph_embedding"].detach().cpu()
        node_terms = out["node_terms"].detach().cpu()
        edge_terms = out["edge_terms"].detach().cpu()

        per_graph = batch.to_data_list()
        assert len(per_graph) == graph_terms.size(0)

        for i, d in enumerate(per_graph):
            d = d.clone()
            d.node_terms = node_terms[i]
            d.edge_terms = edge_terms[i]
            d.graph_terms = graph_terms[i]
            augmented.append(d)

    return augmented


def scan_node_features_with_rw(
    dataset_name: Literal["qm9", "zinc"],
    rw_config: "RWConfig",
    max_samples: int | None = None,
) -> set[tuple]:
    """
    Scan a dataset and return observed node feature tuples after RW augmentation.

    This bridges cached datasets (whose ``data.x`` does not include RW columns)
    with the RW-augmented codebook that HyperNet needs.  Each sample is cloned,
    augmented with :func:`augment_data_with_rw`, and the resulting extended
    feature tuples are collected.

    Parameters
    ----------
    dataset_name : {"qm9", "zinc"}
        Dataset to scan.
    rw_config : RWConfig
        Random walk configuration (must have ``enabled=True``).
    max_samples : int, optional
        If given, stop after this many samples (useful for testing).

    Returns
    -------
    set[tuple]
        All unique node feature tuples observed across the dataset
        (with RW columns appended).
    """
    from graph_hdc.utils.rw_features import augment_data_with_rw

    node_features: set[tuple] = set()
    count = 0

    for split in ["train", "valid", "test"]:
        ds = get_split(split=split, dataset=dataset_name)
        for data in tqdm(ds, desc=f"Scanning {split} with RW", unit="mol"):
            d = data.clone()
            d = augment_data_with_rw(d, k_values=rw_config.k_values, num_bins=rw_config.num_bins, bin_boundaries=rw_config.bin_boundaries)
            for row in d.x.int():
                node_features.add(tuple(row.tolist()))
            count += 1
            if max_samples is not None and count >= max_samples:
                return node_features

    return node_features


def compute_standardization_stats(
    encoded_data: list[Data],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute per-term standardization statistics.

    Parameters
    ----------
    encoded_data : list[Data]
        List of Data objects with edge_terms and graph_terms

    Returns
    -------
    tuple
        (edge_mean, edge_std, graph_mean, graph_std)
    """
    edge_terms = torch.stack([d.edge_terms for d in encoded_data])
    graph_terms = torch.stack([d.graph_terms for d in encoded_data])

    edge_mean = edge_terms.mean(dim=0)
    edge_std = edge_terms.std(dim=0).clamp(min=1e-8)
    graph_mean = graph_terms.mean(dim=0)
    graph_std = graph_terms.std(dim=0).clamp(min=1e-8)

    return edge_mean, edge_std, graph_mean, graph_std


