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

from graph_hdc.datasets.pubchem_smiles import PubChemSmiles
from graph_hdc.datasets.qm9_smiles import QM9Smiles
from graph_hdc.datasets.zinc_smiles import ZincSmiles

if TYPE_CHECKING:
    from graph_hdc.hypernet import HyperNet
    from graph_hdc.hypernet.configs import RWConfig


def _strip_stereochemistry(data: Data) -> Data:
    """Remove stereochemistry from a Data object and rebuild features.

    Parses the SMILES, calls ``Chem.RemoveStereochemistry``, then
    regenerates the canonical SMILES and node features so the rest of
    the pipeline never sees ``@``/``@@`` or ``/``/``\\`` notation.
    """
    mol = Chem.MolFromSmiles(data.smiles)
    if mol is None:
        return data

    Chem.RemoveStereochemistry(mol)
    clean_smiles = Chem.MolToSmiles(mol, canonical=True)

    # Rebuild mol from clean SMILES for consistent kekulisation
    mol = Chem.MolFromSmiles(clean_smiles)
    if mol is None:
        return data

    # Rebuild node features in ZINC format:
    # [atom_type, degree-1, formal_charge_idx, total_Hs, is_in_ring]
    from graph_hdc.datasets.zinc_smiles import ZINC_ATOM_TO_IDX

    x = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym not in ZINC_ATOM_TO_IDX:
            return data  # unsupported atom — return unchanged
        x.append([
            float(ZINC_ATOM_TO_IDX[sym]),
            float(max(0, atom.GetDegree() - 1)),
            float(atom.GetFormalCharge() if atom.GetFormalCharge() >= 0 else 2),
            float(atom.GetTotalNumHs()),
            float(atom.IsInRing()),
        ])

    src, dst = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src += [i, j]
        dst += [j, i]

    data = data.clone()
    data.smiles = clean_smiles
    data.x = torch.tensor(x, dtype=torch.float32)
    data.edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros(2, 0, dtype=torch.long)
    return data


def get_split(
    split: Literal["train", "valid", "test"],
    dataset: Literal["qm9", "zinc", "pubchem16", "pubchem32", "pubchem64"] = "qm9",
    strip_stereo: bool = True,
) -> InMemoryDataset:
    """
    Load a dataset split.

    Disconnected molecules are automatically filtered during dataset processing.

    Parameters
    ----------
    split : str
        One of {"train", "valid", "test"}
    dataset : str
        One of {"qm9", "zinc", "pubchem16", "pubchem32", "pubchem64"}
    strip_stereo : bool
        If True (default), remove stereochemistry (chiral centers and E/Z
        bond geometry) from every molecule before returning.

    Returns
    -------
    InMemoryDataset
        The loaded dataset
    """
    if dataset == "qm9":
        ds = QM9Smiles(split=split)
    elif dataset == "zinc":
        ds = ZincSmiles(split=split)
    elif dataset in ("pubchem16", "pubchem32", "pubchem64"):
        ds = PubChemSmiles(variant=dataset, split=split)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset}. "
            f"Use 'qm9', 'zinc', 'pubchem16', 'pubchem32', or 'pubchem64'."
        )

    if strip_stereo:
        ds._data_list = [_strip_stereochemistry(d) for d in ds]
        ds.data, ds.slices = ds.collate(ds._data_list)

    return ds


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


def get_dataset_info(dataset: Literal["qm9", "zinc", "pubchem16", "pubchem32", "pubchem64"]) -> DatasetInfo:
    """
    Get or compute dataset information (node/edge features, ring info).

    Results are cached to the dataset's processed directory.
    """
    if dataset == "qm9":
        ds_test = QM9Smiles(split="test")
    elif dataset == "zinc":
        ds_test = ZincSmiles(split="test")
    elif dataset in ("pubchem16", "pubchem32", "pubchem64"):
        ds_test = PubChemSmiles(variant=dataset, split="test")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    cache_file = Path(ds_test.processed_dir) / "dataset_info.pkl"

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

            # Ring info (ZINC and PubChem-Large)
            if dataset not in ("zinc", "pubchem16", "pubchem32", "pubchem64"):
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

    if dataset in ("zinc", "pubchem16", "pubchem32", "pubchem64"):
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

    # Augment with RW features if the hypernet expects them
    needs_rw = hasattr(hypernet, "rw_config") and hypernet.rw_config.enabled
    if needs_rw:
        from graph_hdc.utils.rw_features import augment_data_with_rw

    augmented: list[Data] = []

    for batch in tqdm(loader, desc="Encoding", unit="batch"):
        if needs_rw:
            data_list = batch.to_data_list()
            data_list = [
                augment_data_with_rw(
                    d,
                    k_values=hypernet.rw_config.k_values,
                    num_bins=hypernet.rw_config.num_bins,
                    bin_boundaries=hypernet.rw_config.bin_boundaries,
                    clip_range=hypernet.rw_config.clip_range,
                )
                for d in data_list
            ]
            from torch_geometric.data import Batch as PyGBatch
            batch = PyGBatch.from_data_list(data_list)

        batch = batch.to(device)
        out = hypernet.forward(batch, normalize=normalize_graph)

        graph_terms = out["graph_embedding"].detach().cpu()
        node_terms = out["node_terms"].detach().cpu()
        edge_terms = out["edge_terms"].detach().cpu()

        per_graph = batch.to_data_list()
        assert len(per_graph) == graph_terms.size(0)

        for i, d in enumerate(per_graph):
            d = d.clone()
            d.node_terms = node_terms[i].clone()
            d.edge_terms = edge_terms[i].clone()
            d.graph_terms = graph_terms[i].clone()
            augmented.append(d)

    return augmented


def scan_node_features_with_rw(
    dataset_name: Literal["qm9", "zinc", "pubchem16", "pubchem32", "pubchem64"],
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
    dataset_name : {"qm9", "zinc", "pubchem16", "pubchem32", "pubchem64"}
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
    nodes, _ = scan_features_with_rw(dataset_name, rw_config, max_samples)
    return nodes


def scan_features_with_rw(
    dataset_name: Literal["qm9", "zinc", "pubchem16", "pubchem32", "pubchem64"],
    rw_config: "RWConfig",
    max_samples: int | None = None,
) -> tuple[set[tuple], set[tuple[tuple, tuple]]]:
    """
    Scan a dataset and return observed node and edge feature tuples after RW
    augmentation.

    Parameters
    ----------
    dataset_name : {"qm9", "zinc", "pubchem16", "pubchem32", "pubchem64"}
        Dataset to scan.
    rw_config : RWConfig
        Random walk configuration (must have ``enabled=True``).
    max_samples : int, optional
        If given, stop after this many samples (useful for testing).

    Returns
    -------
    tuple[set[tuple], set[tuple[tuple, tuple]]]
        (observed_node_tuples, observed_edge_pairs) where each edge pair
        is ``(src_node_tuple, dst_node_tuple)``.
    """
    from graph_hdc.utils.rw_features import augment_data_with_rw

    node_features: set[tuple] = set()
    edge_features: set[tuple[tuple, tuple]] = set()
    count = 0

    for split in ["train", "valid", "test"]:
        ds = get_split(split=split, dataset=dataset_name)
        for data in tqdm(ds, desc=f"Scanning {split} with RW", unit="mol"):
            d = data.clone()
            d = augment_data_with_rw(d, k_values=rw_config.k_values, num_bins=rw_config.num_bins, bin_boundaries=rw_config.bin_boundaries, clip_range=rw_config.clip_range)
            node_tuples = {i: tuple(row.tolist()) for i, row in enumerate(d.x.int())}
            node_features.update(node_tuples.values())
            for u, v in d.edge_index.t().tolist():
                edge_features.add((node_tuples[u], node_tuples[v]))
            count += 1
            if max_samples is not None and count >= max_samples:
                return node_features, edge_features

    return node_features, edge_features


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


