"""
Export golden reference data from the current codebase.

Run ONCE before starting the refactoring:
    python tests/migration/export_golden.py

Requires: RDKit, current graph_hdc package, QM9/ZINC data files.
"""

import hashlib
import json
import math
from pathlib import Path

import torch
from rdkit import Chem

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURES_DIR.mkdir(exist_ok=True)

QM9_SMILES = [
    "C", "CC", "CCO", "C=O", "C#N",
    "c1ccccc1", "CC(=O)O", "c1ccncc1", "C1CC1", "OC(=O)c1ccccc1",
]
ZINC_SMILES = [
    "CC", "c1ccccc1", "c1ccc(cc1)N", "O=C(O)c1ccccc1", "c1ccc(cc1)Cl",
    "c1ccc2c(c1)cccc2", "CC(=O)Nc1ccccc1", "O=S(=O)(O)c1ccccc1",
    "c1cc(ccc1F)Br", "c1ccc(cc1)c2ccccc2",
]


def tensor_sha256(t: torch.Tensor) -> str:
    """Compute SHA256 hash of a tensor's bytes."""
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


def export_codebook(dataset: str, bins: list[int]):
    """Export CombinatoricIntegerEncoder golden data."""
    from graph_hdc.hypernet.feature_encoders import CombinatoricIntegerEncoder
    from graph_hdc.utils.helpers import TupleIndexer

    print(f"Exporting codebook for {dataset}...")
    hv_dim = 256
    seed = 42
    num_categories = math.prod(bins)
    indexer = TupleIndexer(bins)

    encoder = CombinatoricIntegerEncoder(
        dim=hv_dim,
        num_categories=num_categories,
        indexer=indexer,
        vsa="HRR",
        seed=seed,
        dtype="float64",
    )

    codebook = encoder.codebook.cpu()
    codebook_hash = tensor_sha256(codebook)

    # Test tuples: pick 5 diverse ones
    if dataset == "qm9":
        test_tuples = [(0, 0, 0, 0), (2, 1, 0, 3), (3, 4, 2, 4), (1, 2, 1, 2), (0, 3, 0, 1)]
    else:
        test_tuples = [(0, 0, 0, 0, 0), (4, 3, 1, 2, 1), (8, 5, 2, 3, 1), (1, 1, 0, 1, 0), (6, 2, 0, 2, 1)]

    encoded_indices = indexer.get_idxs(test_tuples)

    # Encode test tuples
    test_data = torch.tensor(test_tuples, dtype=torch.float32).unsqueeze(-1)
    # Actually, encode expects the tuples in a specific format
    # Let's use the indexer directly and get hypervectors from codebook
    encoded_hvs = codebook[torch.tensor(encoded_indices, dtype=torch.long)]

    # Roundtrip decode: pass all hvs as a batch so decode gets [N, dim]
    decoded_all = encoder.decode(encoded_hvs)  # [N, num_features]
    decode_results = decoded_all.tolist()

    fixture = {
        "dataset": dataset,
        "bins": bins,
        "hv_dim": hv_dim,
        "seed": seed,
        "num_categories": num_categories,
        "codebook": codebook,
        "codebook_shape": list(codebook.shape),
        "codebook_hash": codebook_hash,
        "test_tuples": [list(t) for t in test_tuples],
        "encoded_indices": encoded_indices,
        "encoded_hvs": encoded_hvs,
        "decode_results": decode_results,
    }

    torch.save(fixture, FIXTURES_DIR / f"codebook_{dataset}.pt")
    print(f"  Saved codebook_{dataset}.pt (shape={codebook.shape}, hash={codebook_hash[:16]}...)")


def export_encoding(dataset: str, smiles_list: list[str]):
    """Export HyperNet.forward() golden data."""
    from graph_hdc.hypernet.configs import get_config
    from graph_hdc.hypernet.encoder import HyperNet

    if dataset == "qm9":
        from graph_hdc.datasets.qm9_smiles import mol_to_data
        config_name = "QM9_SMILES_HRR_256_F64_G1NG3"
    else:
        from graph_hdc.datasets.zinc_smiles import mol_to_data
        config_name = "ZINC_SMILES_HRR_256_F64_5G1NG4"

    print(f"Exporting encoding for {dataset}...")
    config = get_config(config_name)
    config.device = "cpu"
    hypernet = HyperNet(config)

    molecules = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"  WARNING: Could not parse SMILES '{smi}', skipping")
            continue
        data = mol_to_data(mol)
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long)

        with torch.no_grad():
            output = hypernet.forward(data, normalize=True)

        molecules.append({
            "smiles": smi,
            "input_x": data.x.cpu(),
            "input_edge_index": data.edge_index.cpu(),
            "output_graph_embedding": output["graph_embedding"].cpu(),
            "output_node_terms": output["node_terms"].cpu(),
            "output_edge_terms": output["edge_terms"].cpu(),
        })
        print(f"  Encoded '{smi}' ({data.x.size(0)} nodes)")

    # Extract config metadata
    node_cfg = list(config.node_feature_configs.values())[0]
    config_meta = {
        "bins": node_cfg.bins,
        "hv_dim": config.hv_dim,
        "depth": config.hypernet_depth,
        "seed": config.seed,
        "normalize": True,
        "dtype": config.dtype,
        "base_dataset": config.base_dataset,
    }

    fixture = {
        "dataset": dataset,
        "config": config_meta,
        "molecules": molecules,
    }

    torch.save(fixture, FIXTURES_DIR / f"encoder_{dataset}.pt")
    print(f"  Saved encoder_{dataset}.pt ({len(molecules)} molecules)")


def export_node_decoding(dataset: str, smiles_list: list[str]):
    """Export decode_order_zero_iterative golden data."""
    from graph_hdc.hypernet.configs import get_config
    from graph_hdc.hypernet.encoder import HyperNet

    if dataset == "qm9":
        from graph_hdc.datasets.qm9_smiles import mol_to_data
        config_name = "QM9_SMILES_HRR_256_F64_G1NG3"
    else:
        from graph_hdc.datasets.zinc_smiles import mol_to_data
        config_name = "ZINC_SMILES_HRR_256_F64_5G1NG4"

    print(f"Exporting node decoding for {dataset}...")
    config = get_config(config_name)
    config.device = "cpu"
    hypernet = HyperNet(config)

    entries = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        data = mol_to_data(mol)
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long)

        with torch.no_grad():
            output = hypernet.forward(data, normalize=True)

        node_terms = output["node_terms"].squeeze(0)  # [hv_dim]
        decoded_nodes = hypernet.decode_order_zero_iterative(node_terms)

        entries.append({
            "smiles": smi,
            "node_terms": node_terms.cpu(),
            "decoded_node_tuples": [list(t) for t in decoded_nodes],
            "decoded_num_nodes": len(decoded_nodes),
        })
        print(f"  Decoded '{smi}' -> {len(decoded_nodes)} nodes")

    torch.save(entries, FIXTURES_DIR / f"node_decode_{dataset}.pt")
    print(f"  Saved node_decode_{dataset}.pt ({len(entries)} entries)")


def export_rrwp():
    """Export RRWP computation golden data."""
    from graph_hdc.datasets.qm9_smiles import mol_to_data as qm9_mol_to_data
    from graph_hdc.datasets.zinc_smiles import mol_to_data as zinc_mol_to_data
    from graph_hdc.utils.rw_features import (
        bin_rw_probabilities,
        compute_rw_return_probabilities,
        get_zinc_rw_boundaries,
    )

    print("Exporting RRWP...")

    # Use all molecules from each dataset
    qm9_subset = QM9_SMILES
    zinc_subset = ZINC_SMILES

    entries = []
    for smi, mol_to_data_fn, ds_name in [
        *[(s, qm9_mol_to_data, "qm9") for s in qm9_subset],
        *[(s, zinc_mol_to_data, "zinc") for s in zinc_subset],
    ]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        data = mol_to_data_fn(mol)
        edge_index = data.edge_index
        num_nodes = data.x.size(0)

        k_values = (3, 6)
        rw_probs = compute_rw_return_probabilities(edge_index, num_nodes, k_values=k_values)

        # Uniform 10 bins
        binned_uniform_10 = bin_rw_probabilities(rw_probs, num_bins=10)

        # ZINC quantile 4 bins
        zinc_boundaries = get_zinc_rw_boundaries(4)
        binned_zinc_quantile_4 = bin_rw_probabilities(
            rw_probs, num_bins=4,
            bin_boundaries=zinc_boundaries,
            k_values=k_values,
        )

        entries.append({
            "smiles": smi,
            "dataset": ds_name,
            "edge_index": edge_index.tolist(),
            "num_nodes": num_nodes,
            "k_values": list(k_values),
            "rw_probs": rw_probs.tolist(),
            "binned_uniform_10": binned_uniform_10.tolist(),
            "binned_zinc_quantile_4": binned_zinc_quantile_4.tolist(),
        })
        print(f"  RRWP for '{smi}' ({num_nodes} nodes)")

    with open(FIXTURES_DIR / "rrwp.json", "w") as f:
        json.dump(entries, f, indent=2)
    print(f"  Saved rrwp.json ({len(entries)} entries)")


def export_flow_decoder():
    """Export FlowEdgeDecoder forward pass golden data."""
    from graph_hdc.models.flow_edge_decoder import FlowEdgeDecoder

    print("Exporting FlowEdgeDecoder...")

    torch.manual_seed(42)
    model = FlowEdgeDecoder(
        num_node_classes=24,
        num_edge_classes=5,
        hdc_dim=512,
        n_layers=2,
        hidden_dim=64,
        hidden_mlp_dim=128,
        n_heads=4,
        max_nodes=10,
        dropout=0.0,
        noise_type="uniform",
        lr=1e-4,
    )
    model.eval()

    # Create deterministic inputs
    torch.manual_seed(42)
    bs, n, dx, de = 1, 10, 24, 5
    num_real = 5

    # Random one-hot node features
    X_indices = torch.randint(0, dx, (bs, n))
    X = torch.zeros(bs, n, dx)
    X.scatter_(2, X_indices.unsqueeze(-1), 1.0)

    # Random one-hot edge features
    E_indices = torch.randint(0, de, (bs, n, n))
    E = torch.zeros(bs, n, n, de)
    E.scatter_(3, E_indices.unsqueeze(-1), 1.0)

    # Node mask
    node_mask = torch.zeros(bs, n, dtype=torch.bool)
    node_mask[:, :num_real] = True

    # HDC conditioning vector
    hdc = torch.randn(bs, 512)

    # Time step
    t = torch.tensor([[0.5]])

    # Compute conditioning
    with torch.no_grad():
        hdc_cond = model.condition_mlp(hdc)  # (bs, condition_dim)
        t_embed = model.time_mlp(t)  # (bs, time_embed_dim)

    # Build noisy_data and extra_data
    noisy_data = {
        "X_t": X,
        "E_t": E,
        "y_t": hdc_cond,
        "t": t,
        "node_mask": node_mask,
    }

    with torch.no_grad():
        extra_data = model._compute_extra_data(noisy_data)

    # Concatenate time_embed into y_t for the forward pass
    noisy_data_for_forward = {
        "X_t": X,
        "E_t": E,
        "y_t": hdc_cond,
        "t": t,
    }

    with torch.no_grad():
        output = model.forward(noisy_data_for_forward, extra_data, node_mask)

    fixture = {
        "config": {
            "num_node_classes": 24,
            "num_edge_classes": 5,
            "hdc_dim": 512,
            "n_layers": 2,
            "hidden_dim": 64,
            "hidden_mlp_dim": 128,
            "n_heads": 4,
            "max_nodes": 10,
            "dropout": 0.0,
            "noise_type": "uniform",
            "lr": 1e-4,
        },
        "model_state_dict": model.state_dict(),
        "input_X": X,
        "input_E": E,
        "input_node_mask": node_mask,
        "input_hdc": hdc,
        "input_t": t,
        "input_hdc_cond": hdc_cond,
        "input_t_embed": t_embed,
        "extra_data_X": extra_data.X,
        "extra_data_E": extra_data.E,
        "extra_data_y": extra_data.y,
        "output_X": output.X,
        "output_E": output.E,
    }

    torch.save(fixture, FIXTURES_DIR / "flow_decoder.pt")
    print(f"  Saved flow_decoder.pt")


def export_evaluator():
    """Export evaluator metrics golden data."""
    from graph_hdc.utils.chem import canonical_key
    from graph_hdc.utils.evaluator import rdkit_logp, rdkit_qed, rdkit_sa_score

    print("Exporting evaluator...")

    test_smiles = ["CCO", "c1ccccc1", "INVALID_SMILES", "CCO", "CC(=O)O"]

    per_molecule = []
    for smi in test_smiles:
        mol = Chem.MolFromSmiles(smi)
        entry = {"smiles": smi, "valid": mol is not None}
        if mol is not None:
            entry["logp"] = rdkit_logp(mol)
            entry["qed"] = rdkit_qed(mol)
            entry["sa_score"] = rdkit_sa_score(mol)
            entry["canonical_key"] = canonical_key(mol)
        per_molecule.append(entry)
        print(f"  '{smi}' valid={entry['valid']}")

    # Count unique canonical keys among valid molecules
    valid_keys = [e["canonical_key"] for e in per_molecule if e["valid"]]
    unique_count = len(set(valid_keys))

    fixture = {
        "test_smiles": test_smiles,
        "per_molecule": per_molecule,
        "num_valid": sum(1 for e in per_molecule if e["valid"]),
        "num_unique": unique_count,
    }

    with open(FIXTURES_DIR / "evaluator.json", "w") as f:
        json.dump(fixture, f, indent=2)
    print(f"  Saved evaluator.json")


def export_mol_to_data(dataset: str, smiles_list: list[str]):
    """Export mol_to_data() golden data."""
    if dataset == "qm9":
        from graph_hdc.datasets.qm9_smiles import mol_to_data
    else:
        from graph_hdc.datasets.zinc_smiles import mol_to_data

    print(f"Exporting mol_to_data for {dataset}...")

    entries = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"  WARNING: Could not parse SMILES '{smi}', skipping")
            continue
        data = mol_to_data(mol)
        entries.append({
            "smiles": smi,
            "x": data.x.cpu(),
            "edge_index": data.edge_index.cpu(),
            "num_nodes": data.x.size(0),
            "num_edges": data.edge_index.size(1),
            "canonical_smiles": data.smiles,
        })
        print(f"  mol_to_data('{smi}') -> {data.x.size(0)} nodes, {data.edge_index.size(1)} edges")

    torch.save(entries, FIXTURES_DIR / f"mol_to_data_{dataset}.pt")
    print(f"  Saved mol_to_data_{dataset}.pt ({len(entries)} entries)")


def export_edge_decoding(dataset: str, smiles_list: list[str]):
    """Export decode_order_one golden data."""
    from collections import Counter

    from graph_hdc.hypernet.configs import get_config
    from graph_hdc.hypernet.encoder import HyperNet

    if dataset == "qm9":
        from graph_hdc.datasets.qm9_smiles import mol_to_data
        config_name = "QM9_SMILES_HRR_256_F64_G1NG3"
    else:
        from graph_hdc.datasets.zinc_smiles import mol_to_data
        config_name = "ZINC_SMILES_HRR_256_F64_5G1NG4"

    print(f"Exporting edge decoding for {dataset}...")
    config = get_config(config_name)
    config.device = "cpu"
    hypernet = HyperNet(config)

    entries = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        data = mol_to_data(mol)
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long)

        # Get ground truth node tuples from features
        node_tuples = [tuple(int(v) for v in row) for row in data.x.tolist()]
        node_counter = Counter(node_tuples)

        with torch.no_grad():
            output = hypernet.forward(data, normalize=True)

        edge_terms = output["edge_terms"].squeeze(0)  # [hv_dim]
        # Clone before decode since decode_order_one modifies edge_terms in-place
        decoded_edges = hypernet.decode_order_one(edge_terms.clone(), node_counter)

        entries.append({
            "smiles": smi,
            "edge_terms": edge_terms.cpu(),
            "node_counter": dict(node_counter),
            "decoded_edges": decoded_edges,
            "decoded_num_edges": len(decoded_edges),
        })
        print(f"  Edge-decoded '{smi}' -> {len(decoded_edges)} directed edges")

    torch.save(entries, FIXTURES_DIR / f"edge_decode_{dataset}.pt")
    print(f"  Saved edge_decode_{dataset}.pt ({len(entries)} entries)")


if __name__ == "__main__":
    export_codebook("qm9", [4, 5, 3, 5])
    export_codebook("zinc", [9, 6, 3, 4, 2])
    export_encoding("qm9", QM9_SMILES)
    export_encoding("zinc", ZINC_SMILES)
    export_node_decoding("qm9", QM9_SMILES)
    export_node_decoding("zinc", ZINC_SMILES)
    export_mol_to_data("qm9", QM9_SMILES)
    export_mol_to_data("zinc", ZINC_SMILES)
    export_edge_decoding("qm9", QM9_SMILES)
    export_edge_decoding("zinc", ZINC_SMILES)
    export_rrwp()
    export_flow_decoder()
    export_evaluator()
    print("\nAll golden fixtures exported successfully.")
