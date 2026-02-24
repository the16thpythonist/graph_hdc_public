#!/usr/bin/env python
"""
Two-Stage Flow Matching — End-to-End Generation Evaluation.

Chains two FlowMatchingModels (node stage + graph stage) with a
FlowEdgeDecoder to perform hierarchical molecular graph generation:
  1. Sample node_terms from the node flow model (Stage 1)
  2. Sample graph_terms from the graph flow model conditioned on node_terms (Stage 2)
  3. Concatenate into full HDC vectors [node_terms | graph_terms]
  4. Decode node identities from order-0 embeddings
  5. Generate edges via the FlowEdgeDecoder
  6. Convert to RDKit molecules
  7. Compute comprehensive statistics and visualizations

Usage:
    # Quick test
    python test_flow_matching_two_stages.py --__TESTING__ True

    # Full generation run
    python test_flow_matching_two_stages.py \\
        --HDC_ENCODER_PATH /path/to/encoder.ckpt \\
        --FLOW_DECODER_PATH /path/to/decoder.ckpt \\
        --NODE_FLOW_PATH /path/to/node_flow.ckpt \\
        --GRAPH_FLOW_PATH /path/to/graph_flow.ckpt \\
        --NUM_SAMPLES 100
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw, ImageFont
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from rdkit import Chem
from rdkit.Chem import Crippen, QED, Draw
from torch_geometric.data import Data

from graph_hdc.hypernet import load_hypernet
from graph_hdc.hypernet.encoder import HyperNet
from graph_hdc.models.flow_edge_decoder import (
    FlowEdgeDecoder,
    NODE_FEATURE_DIM,
    get_node_feature_bins,
    node_tuples_to_onehot,
)
from graph_hdc.models.flow_matching import FlowMatchingModel, build_condition
from graph_hdc.utils.experiment_helpers import (
    compute_hdc_distance,
    create_test_dummy_models,
    decode_nodes_from_hdc,
    draw_mol_or_error,
    get_canonical_smiles,
    is_valid_mol,
    pyg_to_mol,
)
from graph_hdc.utils.evaluator import (
    calculate_internal_diversity,
    rdkit_logp,
    rdkit_qed,
    rdkit_sa_score,
)

# Reuse plotting and statistics functions from test_flow_matching
from experiments.generation.test_flow_matching import (
    collect_distribution_data,
    compute_mol_properties,
    plot_generation_summary,
    plot_norm_decay,
    plot_property_distributions,
    plot_structural_distributions,
)


# =============================================================================
# PARAMETERS
# =============================================================================

# -----------------------------------------------------------------------------
# Model Paths
# -----------------------------------------------------------------------------

# :param HDC_ENCODER_PATH:
#     Path to saved HyperNet/MultiHyperNet encoder checkpoint (.ckpt).
HDC_ENCODER_PATH: str = ""

# :param FLOW_DECODER_PATH:
#     Path to saved FlowEdgeDecoder checkpoint (.ckpt).
FLOW_DECODER_PATH: str = ""

# :param NODE_FLOW_PATH:
#     Path to saved Stage 1 FlowMatchingModel checkpoint (node_terms flow).
NODE_FLOW_PATH: str = ""

# :param GRAPH_FLOW_PATH:
#     Path to saved Stage 2 FlowMatchingModel checkpoint (graph_terms flow).
GRAPH_FLOW_PATH: str = ""

# -----------------------------------------------------------------------------
# Sampling Configuration
# -----------------------------------------------------------------------------

# :param NUM_SAMPLES:
#     Number of molecules to generate.
NUM_SAMPLES: int = 10

# :param NUM_REPETITIONS:
#     Best-of-N for FlowEdgeDecoder edge generation (batched).
NUM_REPETITIONS: int = 256

# :param SAMPLE_STEPS:
#     Number of ODE steps for FlowEdgeDecoder discrete flow sampling.
SAMPLE_STEPS: int = 50

# :param ETA:
#     Stochasticity parameter for FlowEdgeDecoder sampling.
ETA: float = 0.0

# :param OMEGA:
#     Target guidance strength for FlowEdgeDecoder sampling.
OMEGA: float = 0.0

# :param SAMPLE_TIME_DISTORTION:
#     Time distortion schedule for FlowEdgeDecoder sampling.
SAMPLE_TIME_DISTORTION: str = "polydec"

# :param FM_SAMPLE_STEPS:
#     Number of ODE steps for FlowMatchingModel sampling (both stages).
FM_SAMPLE_STEPS: int = 1_000

# :param FM_SOLVER_METHOD:
#     ODE solver method for FlowMatchingModel sampling. Options: "euler",
#     "midpoint", "dopri5". None = use whatever was saved in the checkpoint.
FM_SOLVER_METHOD: Optional[str] = "dopri5"

# -----------------------------------------------------------------------------
# Reference Dataset Comparison
# -----------------------------------------------------------------------------

# :param DATASET:
#     Dataset type for atom feature encoding ("zinc" or "qm9").
DATASET: str = "zinc"

# :param COMPARE_DATASET:
#     Whether to load reference dataset for distribution comparison.
COMPARE_DATASET: bool = True

# :param DATASET_SUBSAMPLE:
#     Number of reference molecules to subsample for comparison.
DATASET_SUBSAMPLE: int = 5_000

# -----------------------------------------------------------------------------
# System Configuration
# -----------------------------------------------------------------------------

# :param DEVICE:
#     Device for model inference. "auto" prefers GPU.
DEVICE: str = "cuda"

# :param HDC_DEVICE:
#     Device for the HyperNet HDC encoder.
HDC_DEVICE: str = "cpu"

# :param PRUNE_DECODING_CODEBOOK:
#     Prune the RRWP codebook to only dataset-observed feature tuples.
PRUNE_DECODING_CODEBOOK: bool = True

# :param PLOT_NORM_DECAY:
#     Save per-sample residual norm decay plots during node decoding.
PLOT_NORM_DECAY: bool = True

# :param SEED:
#     Random seed for reproducibility.
SEED: int = 43

# :param __DEBUG__:
#     Debug mode — reuses same output folder during development.
__DEBUG__: bool = True

# :param __TESTING__:
#     Testing mode — runs with dummy models and minimal iterations.
__TESTING__: bool = False


# =============================================================================
# EXPERIMENT
# =============================================================================


@Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)
def experiment(e: Experiment) -> None:
    """Two-Stage Flow Matching — End-to-End Generation Evaluation."""

    e.log("=" * 60)
    e.log("Two-Stage Flow Matching Generation Evaluation")
    e.log("=" * 60)
    e.log_parameters()

    torch.manual_seed(e.SEED)

    # Device setup
    if e.DEVICE == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(e.DEVICE)
    if e.HDC_DEVICE == "auto":
        hdc_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        hdc_device = torch.device(e.HDC_DEVICE)
    e.log(f"Decoder device: {device}")
    e.log(f"HyperNet device: {hdc_device}")

    # =========================================================================
    # Step 1: Load Models
    # =========================================================================

    hypernet, decoder, node_flow, graph_flow, base_hdc_dim = e.apply_hook(
        "load_models", device=device,
    )

    hypernet.to(hdc_device)
    hypernet.eval()
    e.log(str(hypernet))

    # Prune RRWP codebook to observed features for better node decoding
    if e.PRUNE_DECODING_CODEBOOK and hasattr(hypernet, '_limit_full_codebook'):
        from graph_hdc.datasets.utils import scan_node_features_with_rw

        rw_cfg = hypernet.rw_config

        @e.cache.cached(
            name="observed_rw_features",
            scope=lambda _e: (
                "rw_scan",
                _e.DATASET,
                str(rw_cfg.k_values),
                str(rw_cfg.num_bins),
            ),
        )
        def scan_observed_features():
            return scan_node_features_with_rw(
                dataset_name=e.DATASET,
                rw_config=rw_cfg,
            )

        observed = scan_observed_features()
        size_before = hypernet.nodes_codebook_full.shape[0]
        hypernet._limit_full_codebook(observed)
        size_after = hypernet.nodes_codebook_full.shape[0]
        e.log(f"Pruned RRWP codebook: {size_before:,} -> {size_after:,} entries")

    decoder.to(device)
    decoder.eval()
    node_flow.to(device)
    node_flow.eval()
    graph_flow.to(device)
    graph_flow.eval()

    # Override solver method if specified
    if e.FM_SOLVER_METHOD is not None:
        for name, flow in [("node_flow", node_flow), ("graph_flow", graph_flow)]:
            if flow.solver_method != e.FM_SOLVER_METHOD:
                e.log(f"Overriding {name} solver: {flow.solver_method} -> {e.FM_SOLVER_METHOD}")
                flow.solver_method = e.FM_SOLVER_METHOD

    e["model/base_hdc_dim"] = base_hdc_dim
    e["model/concat_hdc_dim"] = 2 * base_hdc_dim

    # =========================================================================
    # Step 2: Two-Stage Sampling
    # =========================================================================

    e.log("\n" + "=" * 60)
    e.log(f"Stage 1: Sampling {e.NUM_SAMPLES} node_terms from node flow...")
    e.log("=" * 60)

    sample_start = time.time()
    with torch.no_grad():
        node_terms = node_flow.sample(
            num_samples=e.NUM_SAMPLES,
            num_steps=e.FM_SAMPLE_STEPS,
            device=device,
        )
    node_time = time.time() - sample_start
    e.log(f"  Sampled node_terms: {node_terms.shape} in {node_time:.2f}s")

    e.log(f"\nStage 2: Sampling {e.NUM_SAMPLES} graph_terms conditioned on node_terms...")
    graph_start = time.time()
    with torch.no_grad():
        graph_terms = graph_flow.sample(
            num_samples=e.NUM_SAMPLES,
            condition=node_terms,
            num_steps=e.FM_SAMPLE_STEPS,
            device=device,
        )
    graph_time = time.time() - graph_start
    e.log(f"  Sampled graph_terms: {graph_terms.shape} in {graph_time:.2f}s")

    # Concatenate into full HDC vectors
    hdc_vectors = torch.cat([node_terms, graph_terms], dim=-1)
    sample_time = node_time + graph_time
    e.log(f"\nCombined HDC vectors: {hdc_vectors.shape} (total {sample_time:.2f}s)")

    # =========================================================================
    # Step 3: Decode Each HDC Vector → Molecular Graph
    # =========================================================================

    e.log("\n" + "=" * 60)
    e.log("Decoding HDC vectors to molecules...")
    e.log("=" * 60)

    molecules_dir = Path(e.path) / "molecules"
    molecules_dir.mkdir(exist_ok=True)

    results = []
    valid_mols = []
    all_smiles = []
    all_norm_histories: List[List[float]] = []
    decode_start = time.time()

    for idx in range(hdc_vectors.shape[0]):
        hdc_vector = hdc_vectors[idx].cpu()

        # 3a. Decode nodes from order_0 part
        node_tuples, num_nodes, norms_history, _ = decode_nodes_from_hdc(
            hypernet, hdc_vector.unsqueeze(0), base_hdc_dim, debug=True
        )
        all_norm_histories.append(norms_history)

        if num_nodes == 0:
            e.log(f"  Sample {idx + 1}/{e.NUM_SAMPLES}: No nodes decoded, skipping")
            results.append({
                "idx": idx,
                "smiles": None,
                "status": "no_nodes",
                "num_nodes": 0,
            })
            continue

        # 3b. Convert node tuples to one-hot features
        feature_bins = get_node_feature_bins(hypernet.rw_config)
        node_features = node_tuples_to_onehot(
            node_tuples, device=device, feature_bins=feature_bins,
        ).unsqueeze(0)
        node_mask = torch.ones(1, num_nodes, dtype=torch.bool, device=device)
        hdc_vec_device = hdc_vector.unsqueeze(0).to(device)

        # 3c. Generate edges
        sample_kwargs = dict(
            sample_steps=e.SAMPLE_STEPS,
            eta=e.ETA,
            omega=e.OMEGA,
            time_distortion=e.SAMPLE_TIME_DISTORTION,
            show_progress=False,
            device=device,
        )

        raw_x = torch.tensor(node_tuples, dtype=torch.float)

        with torch.no_grad():
            if e.NUM_REPETITIONS > 1:
                def score_fn(s):
                    return compute_hdc_distance(
                        s, hdc_vec_device, base_hdc_dim,
                        hypernet, device, dataset=e.DATASET,
                        original_x=raw_x,
                    )

                best_sample, best_distance, _avg_distance = decoder.sample_best_of_n(
                    hdc_vectors=hdc_vec_device,
                    node_features=node_features,
                    node_mask=node_mask,
                    num_repetitions=e.NUM_REPETITIONS,
                    score_fn=score_fn,
                    **sample_kwargs,
                )
                generated_data = best_sample
            else:
                generated_samples = decoder.sample(
                    hdc_vectors=hdc_vec_device,
                    node_features=node_features,
                    node_mask=node_mask,
                    **sample_kwargs,
                )
                generated_data = generated_samples[0]

        # 3d. Convert to RDKit molecule
        mol = pyg_to_mol(generated_data)
        smiles = get_canonical_smiles(mol)
        valid = is_valid_mol(mol)

        # 3e. Compute HDC distance
        hdc_dist = compute_hdc_distance(
            generated_data, hdc_vec_device, base_hdc_dim,
            hypernet, device, dataset=e.DATASET,
            original_x=raw_x,
        )

        # 3f. Save molecule PNG
        mol_img = draw_mol_or_error(mol, size=(300, 300))
        line1 = f"HDC dist: {hdc_dist:.4f}"
        line2 = smiles or "N/A"
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12,
            )
        except OSError:
            font = ImageFont.load_default()
        title_h = 45
        composite = Image.new("RGB", (mol_img.width, mol_img.height + title_h), "white")
        draw_ctx = ImageDraw.Draw(composite)
        draw_ctx.text((5, 3), line1, fill="black", font=font)
        draw_ctx.text((5, 20), line2, fill="black", font=font)
        composite.paste(mol_img, (0, title_h))
        composite.save(molecules_dir / f"molecule_{idx:04d}.png")

        if e.PLOT_NORM_DECAY and norms_history:
            plot_norm_decay([norms_history], molecules_dir / f"norm_decay_{idx:04d}.png")

        # 3g. Collect results
        result = {
            "idx": idx,
            "smiles": smiles,
            "status": "valid" if valid else "invalid",
            "num_nodes": num_nodes,
            "hdc_distance": hdc_dist,
        }

        if valid and mol is not None:
            valid_mols.append(mol)
            all_smiles.append(smiles)
            props = compute_mol_properties(mol)
            result.update(props)
            result["num_atoms"] = mol.GetNumHeavyAtoms()
            result["num_bonds"] = mol.GetNumBonds()

        results.append(result)

        if (idx + 1) % 10 == 0 or idx == 0:
            e.log(f"  Sample {idx + 1}/{e.NUM_SAMPLES}: {smiles or 'N/A'} [{result['status']}]")

    decode_time = time.time() - decode_start
    e.log(f"\nDecoding completed in {decode_time:.2f}s")

    # =========================================================================
    # Step 4: Compute Statistics
    # =========================================================================

    e.log("\n" + "=" * 60)
    e.log("Computing statistics...")
    e.log("=" * 60)

    n_total = len(results)
    n_valid = len(valid_mols)
    n_no_nodes = sum(1 for r in results if r["status"] == "no_nodes")
    n_invalid = sum(1 for r in results if r["status"] == "invalid")

    validity_rate = 100.0 * n_valid / n_total if n_total > 0 else 0.0
    unique_smiles = set(s for s in all_smiles if s is not None)
    uniqueness_rate = 100.0 * len(unique_smiles) / n_valid if n_valid > 0 else 0.0

    training_smiles = set()
    novelty_rate = 0.0

    diversity = 0.0
    if len(valid_mols) >= 2:
        diversity = calculate_internal_diversity(valid_mols)

    e.log(f"  Total samples: {n_total}")
    e.log(f"  Valid: {n_valid} ({validity_rate:.1f}%)")
    e.log(f"  Invalid: {n_invalid}")
    e.log(f"  No nodes decoded: {n_no_nodes}")
    e.log(f"  Unique: {len(unique_smiles)} ({uniqueness_rate:.1f}%)")
    e.log(f"  Internal diversity: {diversity:.1f}%")

    # =========================================================================
    # Step 5: Reference Dataset Comparison (optional)
    # =========================================================================

    ref_dist_data = None
    if e.COMPARE_DATASET and not e.__TESTING__:
        e.log("\n" + "=" * 60)
        e.log("Loading reference dataset for comparison...")
        e.log("=" * 60)

        from graph_hdc.datasets.utils import get_split

        ref_dataset = get_split("train", dataset=e.DATASET)
        n_ref = min(e.DATASET_SUBSAMPLE, len(ref_dataset))

        indices = torch.randperm(len(ref_dataset))[:n_ref].tolist()
        ref_mols = []
        for i in indices:
            data = ref_dataset[i]
            mol = Chem.MolFromSmiles(data.smiles)
            if mol is not None and is_valid_mol(mol):
                ref_mols.append(mol)
                training_smiles.add(get_canonical_smiles(mol))

        e.log(f"  Reference molecules: {len(ref_mols)}")
        ref_dist_data = collect_distribution_data(ref_mols)

        novel_smiles = unique_smiles - training_smiles
        novelty_rate = 100.0 * len(novel_smiles) / n_valid if n_valid > 0 else 0.0
        e.log(f"  Novel: {len(novel_smiles)} ({novelty_rate:.1f}%)")

    # =========================================================================
    # Step 6: Generate Plots
    # =========================================================================

    e.log("\n" + "=" * 60)
    e.log("Generating plots...")
    e.log("=" * 60)

    gen_dist_data = collect_distribution_data(valid_mols)

    structural_path = Path(e.path) / "structural_distributions.png"
    plot_structural_distributions(gen_dist_data, ref_dist_data, structural_path)
    e.log(f"  Structural distributions saved: {structural_path}")

    property_path = Path(e.path) / "property_distributions.png"
    plot_property_distributions(gen_dist_data, ref_dist_data, property_path)
    e.log(f"  Property distributions saved: {property_path}")

    summary_path = Path(e.path) / "generation_summary.png"
    plot_generation_summary(validity_rate, uniqueness_rate, novelty_rate, summary_path)
    e.log(f"  Summary chart saved: {summary_path}")

    if e.PLOT_NORM_DECAY and all_norm_histories:
        norm_decay_path = Path(e.path) / "norm_decay.png"
        plot_norm_decay(all_norm_histories, norm_decay_path)
        e.log(f"  Norm decay plot saved: {norm_decay_path}")

    # =========================================================================
    # Step 7: Save Results
    # =========================================================================

    e.log("\n" + "=" * 60)
    e.log("SUMMARY")
    e.log("=" * 60)

    e["results/validity"] = validity_rate
    e["results/uniqueness"] = uniqueness_rate
    e["results/novelty"] = novelty_rate
    e["results/diversity"] = diversity
    e["results/n_total"] = n_total
    e["results/n_valid"] = n_valid
    e["results/n_invalid"] = n_invalid
    e["results/n_no_nodes"] = n_no_nodes
    e["results/n_unique"] = len(unique_smiles)
    e["results/sample_time_seconds"] = sample_time
    e["results/node_sample_time_seconds"] = node_time
    e["results/graph_sample_time_seconds"] = graph_time
    e["results/decode_time_seconds"] = decode_time

    if gen_dist_data["logp"]:
        e["results/logp_mean"] = float(np.mean(gen_dist_data["logp"]))
        e["results/logp_std"] = float(np.std(gen_dist_data["logp"]))
    if gen_dist_data["qed"]:
        e["results/qed_mean"] = float(np.mean(gen_dist_data["qed"]))
        e["results/qed_std"] = float(np.std(gen_dist_data["qed"]))
    if gen_dist_data["sa_score"]:
        e["results/sa_mean"] = float(np.mean(gen_dist_data["sa_score"]))
        e["results/sa_std"] = float(np.std(gen_dist_data["sa_score"]))

    summary_dict = {
        "config": {
            "hdc_encoder_path": e.HDC_ENCODER_PATH,
            "flow_decoder_path": e.FLOW_DECODER_PATH,
            "node_flow_path": e.NODE_FLOW_PATH,
            "graph_flow_path": e.GRAPH_FLOW_PATH,
            "dataset": e.DATASET,
            "num_samples": e.NUM_SAMPLES,
            "num_repetitions": e.NUM_REPETITIONS,
            "sample_steps": e.SAMPLE_STEPS,
            "eta": e.ETA,
            "omega": e.OMEGA,
            "fm_sample_steps": e.FM_SAMPLE_STEPS,
            "seed": e.SEED,
        },
        "summary": {
            "validity_pct": validity_rate,
            "uniqueness_pct": uniqueness_rate,
            "novelty_pct": novelty_rate,
            "diversity_pct": diversity,
            "n_total": n_total,
            "n_valid": n_valid,
            "n_invalid": n_invalid,
            "n_no_nodes": n_no_nodes,
            "n_unique": len(unique_smiles),
            "sample_time_seconds": sample_time,
            "node_sample_time_seconds": node_time,
            "graph_sample_time_seconds": graph_time,
            "decode_time_seconds": decode_time,
        },
        "properties": {
            "logp_mean": float(np.mean(gen_dist_data["logp"])) if gen_dist_data["logp"] else None,
            "logp_std": float(np.std(gen_dist_data["logp"])) if gen_dist_data["logp"] else None,
            "qed_mean": float(np.mean(gen_dist_data["qed"])) if gen_dist_data["qed"] else None,
            "qed_std": float(np.std(gen_dist_data["qed"])) if gen_dist_data["qed"] else None,
            "sa_mean": float(np.mean(gen_dist_data["sa_score"])) if gen_dist_data["sa_score"] else None,
            "sa_std": float(np.std(gen_dist_data["sa_score"])) if gen_dist_data["sa_score"] else None,
        },
        "results": results,
    }

    e.commit_json("generation_results.json", summary_dict)

    e.log(f"\n  Validity:    {validity_rate:.1f}%")
    e.log(f"  Uniqueness:  {uniqueness_rate:.1f}%")
    e.log(f"  Novelty:     {novelty_rate:.1f}%")
    e.log(f"  Diversity:   {diversity:.1f}%")
    e.log(f"  Sample time: {sample_time:.2f}s (node: {node_time:.2f}s, graph: {graph_time:.2f}s)")
    e.log(f"  Decode time: {decode_time:.2f}s")
    e.log(f"\n  Molecule PNGs saved to: {molecules_dir}")
    e.log("\nExperiment completed!")


# =============================================================================
# HOOKS
# =============================================================================


@experiment.hook("load_models", default=True)
def load_models(
    e: Experiment,
    device: torch.device,
) -> Tuple[HyperNet, FlowEdgeDecoder, FlowMatchingModel, FlowMatchingModel, int]:
    """
    Load HyperNet encoder, FlowEdgeDecoder, and two FlowMatchingModels.

    Returns:
        Tuple of (hypernet, decoder, node_flow, graph_flow, base_hdc_dim).
    """
    e.log("\nLoading models...")

    if e.__TESTING__:
        e.log("TESTING MODE: Creating dummy models...")
        test_device = torch.device("cpu")
        hypernet, decoder, base_hdc_dim = create_test_dummy_models(test_device)

        # Create matching FlowMatchingModels for two-stage
        node_flow = FlowMatchingModel(
            data_dim=base_hdc_dim,
            hidden_dim=64,
            num_blocks=2,
            time_embed_dim=32,
            condition_dim=0,
            default_sample_steps=10,
            vector_part="node_terms",
        )
        graph_flow = FlowMatchingModel(
            data_dim=base_hdc_dim,
            hidden_dim=64,
            num_blocks=2,
            time_embed_dim=32,
            condition_dim=base_hdc_dim,
            condition_embed_dim=32,
            default_sample_steps=10,
            vector_part="graph_terms",
        )

        return hypernet, decoder, node_flow, graph_flow, base_hdc_dim

    # Real model loading
    if not e.HDC_ENCODER_PATH:
        raise ValueError("HDC_ENCODER_PATH is required")
    if not e.FLOW_DECODER_PATH:
        raise ValueError("FLOW_DECODER_PATH is required")
    if not e.NODE_FLOW_PATH:
        raise ValueError("NODE_FLOW_PATH is required")
    if not e.GRAPH_FLOW_PATH:
        raise ValueError("GRAPH_FLOW_PATH is required")

    e.log(f"Loading HyperNet from: {e.HDC_ENCODER_PATH}")
    hypernet = load_hypernet(e.HDC_ENCODER_PATH, device="cpu")
    base_hdc_dim = hypernet.hv_dim
    e.log(f"  HyperNet hv_dim: {base_hdc_dim}")

    e.log(f"Loading FlowEdgeDecoder from: {e.FLOW_DECODER_PATH}")
    decoder = FlowEdgeDecoder.load(e.FLOW_DECODER_PATH, device=device)
    e.log(f"  FlowEdgeDecoder hdc_dim: {decoder.hdc_dim}")

    e.log(f"Loading Node Flow (Stage 1) from: {e.NODE_FLOW_PATH}")
    node_flow = FlowMatchingModel.load_from_checkpoint(
        e.NODE_FLOW_PATH, map_location=device,
    )
    e.log(f"  Node flow data_dim: {node_flow.data_dim}")
    assert node_flow.data_dim == base_hdc_dim, (
        f"Node flow data_dim ({node_flow.data_dim}) != hv_dim ({base_hdc_dim})"
    )

    e.log(f"Loading Graph Flow (Stage 2) from: {e.GRAPH_FLOW_PATH}")
    graph_flow = FlowMatchingModel.load_from_checkpoint(
        e.GRAPH_FLOW_PATH, map_location=device,
    )
    e.log(f"  Graph flow data_dim: {graph_flow.data_dim}, condition_dim: {graph_flow.condition_dim}")
    assert graph_flow.data_dim == base_hdc_dim, (
        f"Graph flow data_dim ({graph_flow.data_dim}) != hv_dim ({base_hdc_dim})"
    )
    assert graph_flow.condition_dim == base_hdc_dim, (
        f"Graph flow condition_dim ({graph_flow.condition_dim}) != hv_dim ({base_hdc_dim})"
    )

    return hypernet, decoder, node_flow, graph_flow, base_hdc_dim


# =============================================================================
# Testing Mode
# =============================================================================


@experiment.testing
def testing(e: Experiment) -> None:
    """Quick test mode with reduced parameters."""
    e.NUM_SAMPLES = 5
    e.NUM_REPETITIONS = 1
    e.SAMPLE_STEPS = 10
    e.FM_SAMPLE_STEPS = 10
    e.COMPARE_DATASET = False
    e.DATASET = "zinc"


# =============================================================================
# Entry Point
# =============================================================================

experiment.run_if_main()
