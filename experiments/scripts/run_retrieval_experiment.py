"""
Retrieval Experiment: Encoding-Decoding Ablation Study

This script evaluates the encoding-decoding performance of HDC graph representations
with ablations over VSA models (HRR), dimensions, and depths.

Metrics:
- Edge decoding accuracy
- Correction percentage
- Final graph accuracy (exact match)
- Average cosine similarity
- Timing breakdown (encoding, edge decoding, graph decoding, total)

Usage:
    python run_retrieval_experiment.py --vsa HRR --hv_dim 1024 --depth 3 --dataset qm9
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torchhd
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch_geometric import seed_everything
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from graph_hdc.datasets.utils import get_split
from graph_hdc.hypernet.configs import DecoderSettings, DSHDCConfig, get_config
from graph_hdc.hypernet.encoder import HyperNet
from graph_hdc.hypernet.types import VSAModel
from graph_hdc.utils.helpers import DataTransformer, pick_device


def create_dynamic_config(base_dataset: str, vsa_model: str, hv_dim: int, depth: int) -> DSHDCConfig:
    """
    Create a dynamic DSHDCConfig for the given parameters.

    Parameters
    ----------
    base_dataset : str
        Either "qm9", "zinc", or "zinc_ring_count"
    vsa_model : str
        VSA model name (currently only "HRR" is supported)
    hv_dim : int
        Hypervector dimension
    depth : int
        Message passing depth

    Returns
    -------
    DSHDCConfig
        Configuration object
    """
    if base_dataset == "qm9":
        # QM9 configuration: 4 features (atom_type, degree, formal_charge, total_num_Hs)
        ds_config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
    else:  # zinc
        ds_config = get_config("ZINC_SMILES_HRR_256_F64_5G1NG4")

    ds_config.vsa = VSAModel(vsa_model)
    ds_config.hv_dim = hv_dim
    ds_config.hypernet_depth = depth
    ds_config.normalize = True
    return ds_config


def plot_hit_rate_by_node_size(detailed_df: pd.DataFrame, output_dir: Path, filename_base: str):
    """
    Generate bar chart showing hit rate (graph accuracy) for each unique node size.

    Parameters
    ----------
    detailed_df : pd.DataFrame
        Detailed results DataFrame with 'num_nodes' and 'graph_accuracy' columns
    output_dir : Path
        Directory to save the plot
    filename_base : str
        Base filename (without extension) for the plot
    """
    # Group by number of nodes and compute hit rate (mean of graph_accuracy)
    hit_rate_by_size = detailed_df.groupby("num_nodes")["graph_accuracy"].agg(["mean", "count"]).reset_index()
    hit_rate_by_size.columns = ["num_nodes", "hit_rate", "count"]
    hit_rate_by_size["hit_rate_pct"] = hit_rate_by_size["hit_rate"] * 100  # Convert to percentage

    # Create bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(
        hit_rate_by_size["num_nodes"],
        hit_rate_by_size["hit_rate_pct"],
        color="steelblue",
        alpha=0.8,
        edgecolor="black",
    )

    # Add labels and title
    plt.xlabel("Number of Nodes (Molecule Size)", fontsize=12)
    plt.ylabel("Hit Rate (%)", fontsize=12)
    plt.title("Hit Rate by Molecular Size", fontsize=14, fontweight="bold")
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    plt.ylim(0, 105)  # 0-100% with some headroom

    # Add count labels on top of bars
    for _, row in hit_rate_by_size.iterrows():
        plt.text(
            row["num_nodes"],
            row["hit_rate_pct"] + 2,
            f"n={int(row['count'])}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    output_path = output_dir / f"{filename_base}_hit_rate_by_node_size.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Hit rate by node size plot saved to: {output_path}")


def graphs_isomorphic(g1: nx.Graph, g2: nx.Graph) -> bool:
    """
    Check if two graphs are isomorphic with node feature matching.

    Handles different node attribute formats:
    - Format 1: {'feat': Feat(...), 'target_degree': ...}
    - Format 2: {'type': (atom_type, degree_idx, formal_charge_idx, explicit_hs)}
    """
    if g1.number_of_nodes() != g2.number_of_nodes() or g1.number_of_edges() != g2.number_of_edges():
        return False

    def normalize_node_features(node_attrs):
        """
        Normalize node attributes to tuple representation.

        Note: Feat.to_tuple() filters out falsy values (0, None), which is incorrect.
        We manually extract the tuple to preserve zeros.
        """
        if "feat" in node_attrs:
            # Format 1: Has Feat object
            feat = node_attrs["feat"]
            # Manual extraction to preserve zeros (to_tuple() filters them out incorrectly)
            if feat.is_in_ring is not None:
                # ZINC dataset (5 features)
                return feat.atom_type, feat.degree_idx, feat.formal_charge_idx, feat.explicit_hs, feat.is_in_ring
            # QM9 dataset (4 features)
            return feat.atom_type, feat.degree_idx, feat.formal_charge_idx, feat.explicit_hs
        if "type" in node_attrs:
            # Format 2: Has type tuple
            return node_attrs["type"]
        # Fallback: return None
        return None

    def node_match(n1, n2):
        """Match nodes by normalized feature tuples."""
        t1 = normalize_node_features(n1)
        t2 = normalize_node_features(n2)
        return t1 == t2 and t1 is not None

    try:
        return nx.is_isomorphic(g1, g2, node_match=node_match)
    except Exception:
        return False


def run_single_experiment(
    vsa_model: str,
    hv_dim: int,
    depth: int,
    dataset_name: str,
    iteration_budget: int,
    n_samples: int = 1000,
    output_dir: Path | None = None,
    early_stopping: bool = False,
    decoder: str = "pattern_matching",
    beam_size: int | None = None,
) -> dict:
    """
    Run a single retrieval experiment.

    Parameters
    ----------
    vsa_model : str
        VSA model name ("HRR")
    hv_dim : int
        Hypervector dimension
    depth : int
        Message passing depth
    dataset_name : str
        Dataset name ("qm9" or "zinc")
    iteration_budget : int
        Number of pattern matching iterations for decoding
    n_samples : int, optional
        Number of samples to evaluate
    output_dir : Path, optional
        Directory to save results
    early_stopping : bool, optional
        Whether to enable early stopping in pattern matching (default: False)

    Returns
    -------
    dict
        Dictionary containing all metrics
    """
    print(f"\n{'=' * 80}")
    print(
        f"Running experiment: VSA={vsa_model}, dim={hv_dim}, depth={depth}, dataset={dataset_name}, iter_budget={iteration_budget}, early_stopping={early_stopping}"
    )
    print(f"{'=' * 80}\n")

    # Create configuration
    config = create_dynamic_config(dataset_name, vsa_model, hv_dim, depth)

    # Load dataset
    dataset = get_split(dataset=config.base_dataset, split="train")

    # Initialize HyperNet
    hypernet = HyperNet(
        config=config,
        depth=depth,
    ).eval()
    hypernet.eval()

    # Stratified sampling based on molecular size (number of atoms)
    dataset_size = len(dataset)
    if n_samples >= dataset_size:
        print(f"Warning: Requested n_samples ({n_samples}) >= dataset size ({dataset_size}). Using full dataset.")
        sample_indices = list(range(dataset_size))
    else:
        # Compute molecule sizes (number of atoms) using PyG's num_nodes property
        print("Computing molecular sizes for stratified sampling...")
        sizes = np.array([dataset[idx].num_nodes for idx in tqdm(range(dataset_size), desc="Analyzing dataset")])

        # Create stratification bins based on quartiles
        bin_labels = pd.qcut(sizes, q=4, labels=False, duplicates="drop")

        # Display stratification statistics
        unique_bins, bin_counts = np.unique(bin_labels, return_counts=True)
        print(f"Stratification: {len(unique_bins)} bins")
        for bin_id, count in zip(unique_bins, bin_counts, strict=False):
            bin_mask = bin_labels == bin_id
            bin_size_range = (sizes[bin_mask].min(), sizes[bin_mask].max())
            print(f"  Bin {bin_id}: {count} molecules, size range: {bin_size_range[0]}-{bin_size_range[1]} atoms")

        # Stratified sampling using sklearn
        all_indices = np.arange(dataset_size)
        sample_indices, _ = train_test_split(all_indices, train_size=n_samples, stratify=bin_labels, random_state=42)
        sample_indices = sample_indices.tolist()

        # Verify stratification
        sampled_bins = bin_labels[sample_indices]
        unique_sampled, sampled_counts = np.unique(sampled_bins, return_counts=True)
        print(f"\nSampled {len(sample_indices)} molecules:")
        for bin_id, count in zip(unique_sampled, sampled_counts, strict=False):
            print(f"  Bin {bin_id}: {count} molecules")

    print(f"\nFinal sample: {len(sample_indices)} molecules from {dataset_size} total")

    # Create subset of dataset with sampled indices
    sampled_dataset = Subset(dataset, sample_indices)

    # Move to GPU if available
    device = pick_device()
    print(f"Using Device: {device}")
    hypernet = hypernet.to(device)

    # Phase 1: Batch Encoding
    # Use DataLoader for efficient batch encoding
    batch_size = 256  # Adjust based on GPU memory
    dataloader = DataLoader(sampled_dataset, batch_size=batch_size, shuffle=False)

    print(f"Encoding {len(sampled_dataset)} samples in batches of {batch_size}...")

    encoded_results = []  # Store (pyg_data, edge_term, graph_term, encoding_time)

    for batch in tqdm(dataloader, desc="Encoding batches"):
        batch = batch.to(device)

        start_time = time.time()
        with torch.no_grad():
            encoding_output = hypernet.forward(batch)
        batch_encoding_time = time.time() - start_time

        # Store individual results from batch
        edge_terms = encoding_output["edge_terms"]
        graph_terms = encoding_output["graph_embedding"]

        # Split batch back to individual samples
        batch_list = batch.to_data_list()

        for i, pyg_data in enumerate(batch_list):
            encoded_results.append(
                {
                    "pyg_data": pyg_data,
                    "edge_term": edge_terms[i],
                    "graph_term": graph_terms[i],
                    "encoding_time": batch_encoding_time / len(batch_list),  # Approximate per-sample time
                    "num_nodes": pyg_data.num_nodes,  # Track molecule size
                }
            )

    print(f"Encoding complete. Processing {len(encoded_results)} samples for decoding...")

    # Metrics
    edge_accuracies = []
    correction_levels = []
    graph_accuracies = []
    cosine_similarities = []

    encoding_times = []
    edge_decoding_times = []
    graph_decoding_times = []
    total_times = []

    # Additional tracking
    num_nodes_list = []
    is_max_cosine_non_hit_list = []
    max_cosine_non_hit_indices = []

    # Phase 2 & 3: Edge Decoding and Graph Decoding
    for result in tqdm(encoded_results, desc="Decoding samples"):
        pyg_data = result["pyg_data"]
        edge_term = result["edge_term"]
        graph_term = result["graph_term"]
        encoding_time = result["encoding_time"]
        num_nodes = result["num_nodes"]

        encoding_times.append(encoding_time)
        num_nodes_list.append(num_nodes)

        # Convert PyG data to NetworkX for ground truth comparison
        nx_graph = DataTransformer.pyg_to_nx(pyg_data)

        # Phase 2: Edge Decoding
        start_time = time.time()
        with torch.no_grad():
            decoded_edges = hypernet.decode_order_one_no_node_terms(edge_term.clone())
        edge_decoding_time = time.time() - start_time
        edge_decoding_times.append(edge_decoding_time)

        # Compute edge accuracy using NetworkX graph
        # Real Data
        node_tuples = [tuple(i) for i in pyg_data.x.int().tolist()]
        original_edges = [(node_tuples[e[0]], node_tuples[e[1]]) for e in pyg_data.edge_index.t().int().cpu().tolist()]
        original_edges_counter = Counter(original_edges)
        decoded_edges_counter = Counter(decoded_edges)

        # Edge accuracy: intersection over union
        intersection = sum((original_edges_counter & decoded_edges_counter).values())
        union = sum((original_edges_counter | decoded_edges_counter).values())
        edge_accuracy = intersection / union if union > 0 else 0.0
        edge_accuracies.append(edge_accuracy)

        # Phase 3: Full Graph Decoding
        start_time = time.time()
        with torch.no_grad():
            decoder_settings = DecoderSettings.get_default_for(base_dataset=config.base_dataset)
            if decoder == "pattern_matching":
                decoding_result = hypernet.decode_graph(
                    edge_term=edge_term,
                    graph_term=graph_term,
                    decoder_settings=decoder_settings,
                )
            else:  # greedy decoder
                # Use provided beam_size or default from DecoderSettings
                # beam_size is only applied to ZINC dataset (requirement)
                if beam_size is not None and dataset_name == "zinc":
                    decoder_settings.fallback_decoder_settings.beam_size = beam_size
                # else: use the default from DecoderSettings.get_default_for()
                #   QM9: 2048, ZINC: 96

                decoding_result = hypernet.decode_graph_greedy(
                    edge_term=edge_term,
                    graph_term=graph_term,
                    decoder_settings=decoder_settings.fallback_decoder_settings,
                )

        graph_decoding_time = time.time() - start_time
        graph_decoding_times.append(graph_decoding_time)

        total_time = encoding_time + edge_decoding_time + graph_decoding_time
        total_times.append(total_time)

        # Record correction level
        correction_levels.append(decoding_result.correction_level.name)

        # Check graph accuracy (exact isomorphism)
        if len(decoding_result.nx_graphs) > 0:
            decoded_graph = decoding_result.nx_graphs[0]
            graph_match = graphs_isomorphic(nx_graph, decoded_graph)
            graph_accuracies.append(1.0 if graph_match else 0.0)

            # Compute cosine similarity by re-encoding the decoded graph
            pyg_decoded = DataTransformer.nx_to_pyg_with_type_attr(decoded_graph)
            batch_decoded = Batch.from_data_list([pyg_decoded]).to(device)
            with torch.no_grad():
                reencoded_output = hypernet.forward(batch_decoded)
            reencoded_graph_term = reencoded_output["graph_embedding"][0]

            cos_sim = torchhd.cos(graph_term, reencoded_graph_term).item()
            cosine_similarities.append(cos_sim)

            # Detect max cosine similarity non-hits
            is_max_cosine_non_hit = (cos_sim == 1.0) and (not graph_match)
            is_max_cosine_non_hit_list.append(is_max_cosine_non_hit)
            if is_max_cosine_non_hit:
                max_cosine_non_hit_indices.append(len(cosine_similarities) - 1)
        else:
            graph_accuracies.append(0.0)
            cosine_similarities.append(0.0)
            is_max_cosine_non_hit_list.append(False)

    # Compute summary statistics
    correction_counter = Counter(correction_levels)
    correction_percentages = {k: v / len(correction_levels) * 100 for k, v in correction_counter.items()}

    # Determine actual beam_size used (for results storage)
    if decoder == "greedy":
        # Get the decoder settings to extract the actual beam_size used
        temp_decoder_settings = DecoderSettings.get_default_for(base_dataset=config.base_dataset)
        if beam_size is not None and dataset_name == "zinc":
            actual_beam_size = beam_size
        else:
            actual_beam_size = temp_decoder_settings.fallback_decoder_settings.beam_size
    else:
        actual_beam_size = None

    results = {
        "vsa_model": vsa_model,
        "hv_dim": hv_dim,
        "depth": depth,
        "dataset": dataset_name,
        "iteration_budget": iteration_budget,
        "early_stopping": early_stopping,
        "n_samples": len(sample_indices),
        "decoder": decoder,
        "beam_size": actual_beam_size,
        # Accuracies
        "edge_accuracy_mean": np.mean(edge_accuracies),
        "edge_accuracy_std": np.std(edge_accuracies),
        "graph_accuracy_mean": np.mean(graph_accuracies),
        "graph_accuracy_std": np.std(graph_accuracies),
        # Correction statistics
        "correction_level_0_pct": correction_percentages.get("ZERO", 0.0),
        "correction_level_1_pct": correction_percentages.get("ONE", 0.0),
        "correction_level_2_pct": correction_percentages.get("TWO", 0.0),
        "correction_level_3_pct": correction_percentages.get("THREE", 0.0),
        "correction_level_fail_pct": correction_percentages.get("FAIL", 0.0),
        # Cosine similarity
        "cosine_similarity_mean": np.mean(cosine_similarities),
        "cosine_similarity_std": np.std(cosine_similarities),
        # Max cosine non-hit tracking
        "max_cosine_non_hit_count": len(max_cosine_non_hit_indices),
        "max_cosine_non_hit_indices": max_cosine_non_hit_indices,
        # Timing
        "encoding_time_mean": np.mean(encoding_times),
        "encoding_time_std": np.std(encoding_times),
        "edge_decoding_time_mean": np.mean(edge_decoding_times),
        "edge_decoding_time_std": np.std(edge_decoding_times),
        "graph_decoding_time_mean": np.mean(graph_decoding_times),
        "graph_decoding_time_std": np.std(graph_decoding_times),
        "total_time_mean": np.mean(total_times),
        "total_time_std": np.std(total_times),
    }

    # Save results
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build filename with beam_size suffix for greedy decoder
        if decoder == "greedy":
            decoder_suffix = f"{decoder}_bs{actual_beam_size}"
        else:
            decoder_suffix = decoder

        # Save summary (include decoder type and beam_size in filename)
        filename = f"{vsa_model}_{dataset_name}_dim{hv_dim}_depth{depth}_iter{iteration_budget}_{decoder_suffix}.json"
        with (output_dir / filename).open(mode="w") as f:
            json.dump(results, f, indent=2)

        # Save detailed results
        detailed_results = pd.DataFrame(
            {
                "decoder": decoder,
                "num_nodes": num_nodes_list,
                "edge_accuracy": edge_accuracies,
                "graph_accuracy": graph_accuracies,
                "correction_level": correction_levels,
                "cosine_similarity": cosine_similarities,
                "is_max_cosine_non_hit": is_max_cosine_non_hit_list,
                "encoding_time": encoding_times,
                "edge_decoding_time": edge_decoding_times,
                "graph_decoding_time": graph_decoding_times,
                "total_time": total_times,
            }
        )
        detailed_filename = (
            f"{vsa_model}_{dataset_name}_dim{hv_dim}_depth{depth}_iter{iteration_budget}_{decoder_suffix}_detailed.csv"
        )
        detailed_results.to_csv(output_dir / detailed_filename, index=False)

        # Generate hit rate by node size plot (include decoder type and beam_size in filename)
        filename_base = f"{vsa_model}_{dataset_name}_dim{hv_dim}_depth{depth}_iter{iteration_budget}_{decoder_suffix}"
        plot_hit_rate_by_node_size(detailed_results, output_dir, filename_base)

    # Print summary
    print("\nResults Summary:")
    print(f"  Edge Accuracy:        {results['edge_accuracy_mean']:.4f} ± {results['edge_accuracy_std']:.4f}")
    print(f"  Graph Accuracy:       {results['graph_accuracy_mean']:.4f} ± {results['graph_accuracy_std']:.4f}")
    print(f"  Cosine Similarity:    {results['cosine_similarity_mean']:.4f} ± {results['cosine_similarity_std']:.4f}")
    print(f"  Correction Level 0:   {results['correction_level_0_pct']:.2f}%")
    print(f"  Correction Level 1:   {results['correction_level_1_pct']:.2f}%")
    print(f"  Correction Level 2:   {results['correction_level_2_pct']:.2f}%")
    print(f"  Correction Level 3:   {results['correction_level_3_pct']:.2f}%")
    print(f"  Correction Level FAIL: {results['correction_level_fail_pct']:.2f}%")
    print(f"  Encoding Time:        {results['encoding_time_mean']:.4f} ± {results['encoding_time_std']:.4f} s")
    print(
        f"  Edge Decoding Time:   {results['edge_decoding_time_mean']:.4f} ± {results['edge_decoding_time_std']:.4f} s"
    )
    print(
        f"  Graph Decoding Time:  {results['graph_decoding_time_mean']:.4f} ± {results['graph_decoding_time_std']:.4f} s"
    )
    print(f"  Total Time:           {results['total_time_mean']:.4f} ± {results['total_time_std']:.4f} s")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run retrieval experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--vsa", type=str, default="HRR", choices=["HRR"], help="VSA model (default: HRR)")
    parser.add_argument("--hv_dim", type=int, default=256, help="Hypervector dimension (default: 256 for QM9)")
    parser.add_argument("--depth", type=int, default=3, help="Message passing depth (default: 3)")
    parser.add_argument(
        "--dataset", type=str, default="zinc", choices=["qm9", "zinc"], help="Dataset name (default: zinc)"
    )
    parser.add_argument("--iter_budget", type=int, default=50, help="Iteration budget for decoding.")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument(
        "--decoder",
        type=str,
        default="greedy",
        choices=["pattern_matching", "greedy"],
        help="Decoder type (default: greedy)",
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="Enable early stopping in pattern matching (default: False)",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=64,
        help="Beam size for greedy decoder (default: 64, ignored for pattern_matching)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Set seed after parsing arguments
    seed_everything(seed=args.seed)

    output_dir = Path(__file__).parent.parent / "results" / "retrieval"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run experiment
    results = run_single_experiment(
        vsa_model=args.vsa,
        hv_dim=args.hv_dim,
        depth=args.depth,
        dataset_name=args.dataset,
        iteration_budget=args.iter_budget,
        n_samples=args.n_samples,
        output_dir=output_dir,
        early_stopping=args.early_stopping,
        decoder=args.decoder,
        beam_size=args.beam_size,
    )

    # Append to summary CSV
    summary_csv = output_dir / "summary.csv"
    df = pd.DataFrame([results])
    if summary_csv.exists():
        df_existing = pd.read_csv(summary_csv)
        df = pd.concat([df_existing, df], ignore_index=True)
    df.to_csv(summary_csv, index=False)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
