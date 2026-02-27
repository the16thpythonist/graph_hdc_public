#!/usr/bin/env python
"""
Graph-Stage-Only Generation — Evaluate graph_terms flow with real node_terms.

Conditions the graph flow (Stage 2) on **real** node_terms from dataset molecules
rather than sampled ones. This decouples graph flow quality from node flow quality,
letting us assess how well the graph flow reconstructs topology when given perfect
atom identity information.

Pipeline:
  1. Load reference molecules from the dataset
  2. Encode each with HyperNet to get real node_terms
  3. Sample multiple graph_terms per molecule from the graph flow, conditioned on real node_terms
  4. Concatenate [real_node_terms | sampled_graph_terms] into full HDC vectors
  5. Decode nodes + edges → RDKit molecules
  6. Create per-molecule PNG strips and summary statistics

Usage:
    # Quick test with dummy models
    python test_flow_matching_graph_stage.py --__TESTING__ True

    # Full run
    python test_flow_matching_graph_stage.py \\
        --HDC_ENCODER_PATH /path/to/encoder.ckpt \\
        --FLOW_DECODER_PATH /path/to/decoder.ckpt \\
        --GRAPH_FLOW_PATH /path/to/graph_flow.ckpt \\
        --NUM_MOLECULES 20 --NUM_SAMPLES_PER_MOL 5
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import torch
from PIL import Image, ImageDraw, ImageFont
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from rdkit import Chem
from torch_geometric.data import Data

from graph_hdc.hypernet import load_hypernet
from graph_hdc.hypernet.encoder import HyperNet
from graph_hdc.models.flow_edge_decoder import (
    FlowEdgeDecoder,
    NODE_FEATURE_DIM,
    get_node_feature_bins,
    node_tuples_to_onehot,
)
from graph_hdc.models.flow_matching import FlowMatchingModel
from graph_hdc.datasets.utils import get_split
from graph_hdc.datasets.zinc_smiles import mol_to_data
from graph_hdc.utils.experiment_helpers import (
    compute_hdc_distance,
    create_test_dummy_models,
    decode_nodes_from_hdc,
    draw_mol_or_error,
    get_canonical_smiles,
    is_valid_mol,
    pyg_to_mol,
)


# =============================================================================
# PARAMETERS
# =============================================================================

# -----------------------------------------------------------------------------
# Model Paths
# -----------------------------------------------------------------------------

# :param HDC_ENCODER_PATH:
#     Path to saved HyperNet/MultiHyperNet encoder checkpoint (.ckpt).
HDC_ENCODER_PATH: str = "/media/ssd2/Programming/graph_hdc_public/experiments/decoding/results/train_flow_edge_decoder_streaming/GOOD/hypernet_encoder.ckpt"

# :param FLOW_DECODER_PATH:
#     Path to saved FlowEdgeDecoder checkpoint (.ckpt).
FLOW_DECODER_PATH: str = "/media/ssd2/Programming/graph_hdc_public/experiments/decoding/results/train_flow_edge_decoder_streaming/GOOD/last.ckpt"

# :param GRAPH_FLOW_PATH:
#     Path to saved Stage 2 FlowMatchingModel checkpoint (graph_terms flow).
GRAPH_FLOW_PATH: str = "/media/ssd2/Programming/graph_hdc_public/experiments/generation/results/train_flow_matching__graph/first/last.ckpt"

# -----------------------------------------------------------------------------
# Sampling Configuration
# -----------------------------------------------------------------------------

# :param NUM_MOLECULES:
#     Number of reference molecules to sample from the dataset.
NUM_MOLECULES: int = 20

# :param NUM_SAMPLES_PER_MOL:
#     Number of graph_terms samples to draw per reference molecule.
NUM_SAMPLES_PER_MOL: int = 5

# :param NUM_REPETITIONS:
#     Best-of-N for FlowEdgeDecoder edge generation (batched).
NUM_REPETITIONS: int = 128

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
#     Number of ODE steps for FlowMatchingModel graph_terms sampling.
FM_SAMPLE_STEPS: int = 10_000

# :param FM_SOLVER_METHOD:
#     ODE solver method for FlowMatchingModel sampling. Options: "euler",
#     "midpoint", "dopri5". None = use whatever was saved in the checkpoint.
FM_SOLVER_METHOD: Optional[str] = "midpoint"

# -----------------------------------------------------------------------------
# Dataset & System
# -----------------------------------------------------------------------------

# :param DATASET:
#     Dataset type for atom feature encoding ("zinc" or "qm9").
DATASET: str = "zinc"

# :param DEVICE:
#     Device for model inference.
DEVICE: str = "cuda"

# :param HDC_DEVICE:
#     Device for the HyperNet HDC encoder.
HDC_DEVICE: str = "cpu"

# :param PRUNE_DECODING_CODEBOOK:
#     Prune the RRWP codebook to only dataset-observed feature tuples.
PRUNE_DECODING_CODEBOOK: bool = True

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
    """Graph-Stage-Only Generation Evaluation."""

    e.log("=" * 60)
    e.log("Graph-Stage-Only Generation Evaluation")
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

    e.log("\nStep 1: Loading models...")

    if e.__TESTING__:
        e.log("TESTING MODE: Creating dummy models...")
        test_device = torch.device("cpu")
        device = test_device
        hdc_device = test_device

        hypernet, decoder, base_hdc_dim = create_test_dummy_models(test_device)

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
    else:
        if not e.HDC_ENCODER_PATH:
            raise ValueError("HDC_ENCODER_PATH is required")
        if not e.FLOW_DECODER_PATH:
            raise ValueError("FLOW_DECODER_PATH is required")
        if not e.GRAPH_FLOW_PATH:
            raise ValueError("GRAPH_FLOW_PATH is required")

        e.log(f"  Loading HyperNet from: {e.HDC_ENCODER_PATH}")
        hypernet = load_hypernet(e.HDC_ENCODER_PATH, device="cpu")
        base_hdc_dim = hypernet.hv_dim
        e.log(f"    hv_dim: {base_hdc_dim}")

        e.log(f"  Loading FlowEdgeDecoder from: {e.FLOW_DECODER_PATH}")
        decoder = FlowEdgeDecoder.load(e.FLOW_DECODER_PATH, device=device)
        e.log(f"    hdc_dim: {decoder.hdc_dim}")

        e.log(f"  Loading Graph Flow from: {e.GRAPH_FLOW_PATH}")
        graph_flow = FlowMatchingModel.load_from_checkpoint(
            e.GRAPH_FLOW_PATH, map_location=device,
        )
        e.log(f"    data_dim: {graph_flow.data_dim}, condition_dim: {graph_flow.condition_dim}")
        assert graph_flow.data_dim == base_hdc_dim, (
            f"Graph flow data_dim ({graph_flow.data_dim}) != hv_dim ({base_hdc_dim})"
        )
        assert graph_flow.condition_dim == base_hdc_dim, (
            f"Graph flow condition_dim ({graph_flow.condition_dim}) != hv_dim ({base_hdc_dim})"
        )

    hypernet.to(hdc_device)
    hypernet.eval()

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
        e.log(f"  Pruned RRWP codebook: {size_before:,} -> {size_after:,} entries")

    decoder.to(device)
    decoder.eval()
    graph_flow.to(device)
    graph_flow.eval()

    # Override solver method if specified
    if e.FM_SOLVER_METHOD is not None:
        if graph_flow.solver_method != e.FM_SOLVER_METHOD:
            e.log(f"  Overriding graph_flow solver: {graph_flow.solver_method} -> {e.FM_SOLVER_METHOD}")
            graph_flow.solver_method = e.FM_SOLVER_METHOD

    e.log(f"  base_hdc_dim: {base_hdc_dim}")

    # =========================================================================
    # Step 2: Load & Sample Reference Molecules
    # =========================================================================

    e.log("\nStep 2: Loading reference molecules from dataset...")

    dataset = get_split("train", dataset=e.DATASET)
    indices = torch.randperm(len(dataset))[:e.NUM_MOLECULES].tolist()

    ref_molecules = []  # List of (idx_in_dataset, smiles, rdkit_mol)
    for ds_idx in indices:
        data = dataset[ds_idx]
        smiles = data.smiles
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            ref_molecules.append((ds_idx, smiles, mol))

    e.log(f"  Selected {len(ref_molecules)} valid reference molecules from {len(dataset)} total")

    # =========================================================================
    # Step 3: Encode with HyperNet & Sample Graph Terms
    # =========================================================================

    e.log("\nStep 3-4: Encoding references and sampling graph_terms...")

    molecules_dir = Path(e.path) / "molecules"
    molecules_dir.mkdir(exist_ok=True)

    all_results = []
    total_valid = 0
    total_samples = 0
    all_hdc_distances = []

    # Check if HyperNet needs RRWP augmentation
    _use_rw = hasattr(hypernet, "rw_config") and hypernet.rw_config.enabled
    if _use_rw:
        from graph_hdc.utils.rw_features import augment_data_with_rw
        e.log("  RW augmentation enabled for HyperNet encoding")

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11,
        )
    except OSError:
        font = ImageFont.load_default()

    for mol_idx, (ds_idx, ref_smiles, ref_mol) in enumerate(ref_molecules):
        e.log(f"\n  Molecule {mol_idx + 1}/{len(ref_molecules)}: {ref_smiles}")

        # 3a. Encode with HyperNet to get real node_terms
        pyg_data = mol_to_data(ref_mol)
        # mol_to_data doesn't set batch; HyperNet.forward needs it for scatter
        if pyg_data.batch is None:
            pyg_data.batch = torch.zeros(pyg_data.x.size(0), dtype=torch.long)
        # RRWPHyperNet needs RRWP columns appended to data.x
        if _use_rw:
            pyg_data = augment_data_with_rw(
                pyg_data,
                k_values=hypernet.rw_config.k_values,
                num_bins=hypernet.rw_config.num_bins,
                bin_boundaries=hypernet.rw_config.bin_boundaries,
                clip_range=hypernet.rw_config.clip_range,
            )
        with torch.no_grad():
            output = hypernet.forward(pyg_data, normalize=True)
        node_terms = output["node_terms"].float()  # [1, hv_dim], cast to float32
        node_terms = node_terms.to(device)
        e.log(f"    node_terms shape: {node_terms.shape}")

        # 3b. Sample graph_terms conditioned on real node_terms
        repeated_node_terms = node_terms.repeat(e.NUM_SAMPLES_PER_MOL, 1)  # [N, hv_dim]

        sample_start = time.time()
        with torch.no_grad():
            sampled_graph_terms = graph_flow.sample(
                num_samples=e.NUM_SAMPLES_PER_MOL,
                condition=repeated_node_terms,
                num_steps=e.FM_SAMPLE_STEPS,
                device=device,
            )
        sample_time = time.time() - sample_start
        e.log(f"    Sampled {e.NUM_SAMPLES_PER_MOL} graph_terms in {sample_time:.2f}s")

        # =====================================================================
        # Step 5: Decode each sample → molecule
        # =====================================================================

        mol_results = []
        sample_images = []

        for s_idx in range(e.NUM_SAMPLES_PER_MOL):
            total_samples += 1

            # 5a. Concatenate [real_node_terms | sampled_graph_terms] → full HDC vector
            hdc_vector = torch.cat([
                node_terms[0].cpu(),
                sampled_graph_terms[s_idx].cpu(),
            ], dim=-1)  # [2 * hv_dim]

            # 5b. Decode nodes
            node_tuples, num_nodes = decode_nodes_from_hdc(
                hypernet, hdc_vector.unsqueeze(0), base_hdc_dim, debug=False,
            )

            if num_nodes == 0:
                mol_results.append({
                    "sample_idx": s_idx,
                    "smiles": None,
                    "status": "no_nodes",
                    "num_nodes": 0,
                    "hdc_distance": None,
                })
                sample_images.append((None, "no_nodes", None))
                e.log(f"      Sample {s_idx + 1}: No nodes decoded")
                continue

            # 5c. Convert node tuples to one-hot features
            feature_bins = get_node_feature_bins(hypernet.rw_config)
            node_features = node_tuples_to_onehot(
                node_tuples, device=device, feature_bins=feature_bins,
            ).unsqueeze(0)
            node_mask = torch.ones(1, num_nodes, dtype=torch.bool, device=device)
            hdc_vec_device = hdc_vector.unsqueeze(0).to(device)

            # 5d. Generate edges
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

                    best_sample, best_distance, _ = decoder.sample_best_of_n(
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

            # 5e. Convert to RDKit molecule
            mol = pyg_to_mol(generated_data)
            smiles = get_canonical_smiles(mol)
            valid = is_valid_mol(mol)

            # 5f. Compute HDC distance
            hdc_dist = compute_hdc_distance(
                generated_data, hdc_vec_device, base_hdc_dim,
                hypernet, device, dataset=e.DATASET,
                original_x=raw_x,
            )

            if valid:
                total_valid += 1
            all_hdc_distances.append(hdc_dist)

            status = "valid" if valid else "invalid"
            mol_results.append({
                "sample_idx": s_idx,
                "smiles": smiles,
                "status": status,
                "num_nodes": num_nodes,
                "hdc_distance": hdc_dist,
            })
            sample_images.append((mol, status, hdc_dist))
            e.log(f"      Sample {s_idx + 1}: {smiles or 'N/A'} [{status}] dist={hdc_dist:.4f}")

        # =====================================================================
        # Step 6: Create per-molecule PNG strip
        # =====================================================================

        cell_w, cell_h = 300, 300
        annotation_h = 50
        n_cells = 1 + len(sample_images)  # original + samples
        strip_w = n_cells * cell_w
        strip_h = cell_h + annotation_h
        strip = Image.new("RGB", (strip_w, strip_h), "white")
        draw_ctx = ImageDraw.Draw(strip)

        # Draw original molecule
        orig_img = draw_mol_or_error(ref_mol, size=(cell_w, cell_h))
        strip.paste(orig_img, (0, 0))
        draw_ctx.text((5, cell_h + 3), "ORIGINAL", fill="blue", font=font)
        # Truncate SMILES if too long for the cell
        display_smiles = ref_smiles if len(ref_smiles) <= 40 else ref_smiles[:37] + "..."
        draw_ctx.text((5, cell_h + 18), display_smiles, fill="black", font=font)

        # Draw each sample
        for s_idx, (s_mol, s_status, s_dist) in enumerate(sample_images):
            x_offset = (s_idx + 1) * cell_w
            s_img = draw_mol_or_error(s_mol, size=(cell_w, cell_h))
            strip.paste(s_img, (x_offset, 0))

            if s_status == "no_nodes":
                draw_ctx.text((x_offset + 5, cell_h + 3), "no nodes", fill="red", font=font)
            else:
                s_smiles = mol_results[s_idx].get("smiles") or "N/A"
                display_s = s_smiles if len(s_smiles) <= 40 else s_smiles[:37] + "..."
                color = "green" if s_status == "valid" else "red"
                draw_ctx.text((x_offset + 5, cell_h + 3), display_s, fill="black", font=font)
                dist_str = f"dist={s_dist:.4f}" if s_dist is not None else ""
                draw_ctx.text(
                    (x_offset + 5, cell_h + 18),
                    f"{s_status}  {dist_str}",
                    fill=color,
                    font=font,
                )

        strip.save(molecules_dir / f"mol_{mol_idx:04d}.png")

        all_results.append({
            "mol_idx": mol_idx,
            "dataset_idx": ds_idx,
            "ref_smiles": ref_smiles,
            "samples": mol_results,
        })

    # =========================================================================
    # Step 7: Summary Statistics
    # =========================================================================

    e.log("\n" + "=" * 60)
    e.log("SUMMARY")
    e.log("=" * 60)

    validity_rate = 100.0 * total_valid / total_samples if total_samples > 0 else 0.0
    hdc_dist_mean = float(np.mean(all_hdc_distances)) if all_hdc_distances else 0.0
    hdc_dist_std = float(np.std(all_hdc_distances)) if all_hdc_distances else 0.0

    # Per-molecule validity
    per_mol_validity = []
    for entry in all_results:
        n_valid = sum(1 for s in entry["samples"] if s["status"] == "valid")
        n_total = len(entry["samples"])
        rate = 100.0 * n_valid / n_total if n_total > 0 else 0.0
        per_mol_validity.append(rate)

    e.log(f"  Total samples:     {total_samples}")
    e.log(f"  Valid molecules:   {total_valid} ({validity_rate:.1f}%)")
    e.log(f"  HDC distance:      {hdc_dist_mean:.4f} +/- {hdc_dist_std:.4f}")
    e.log(f"  Per-mol validity:  {np.mean(per_mol_validity):.1f}% +/- {np.std(per_mol_validity):.1f}%")
    e.log(f"  Molecule PNGs:     {molecules_dir}")

    summary_dict = {
        "config": {
            "hdc_encoder_path": e.HDC_ENCODER_PATH,
            "flow_decoder_path": e.FLOW_DECODER_PATH,
            "graph_flow_path": e.GRAPH_FLOW_PATH,
            "dataset": e.DATASET,
            "num_molecules": e.NUM_MOLECULES,
            "num_samples_per_mol": e.NUM_SAMPLES_PER_MOL,
            "num_repetitions": e.NUM_REPETITIONS,
            "sample_steps": e.SAMPLE_STEPS,
            "fm_sample_steps": e.FM_SAMPLE_STEPS,
            "fm_solver_method": e.FM_SOLVER_METHOD,
            "eta": e.ETA,
            "omega": e.OMEGA,
            "seed": e.SEED,
        },
        "summary": {
            "total_samples": total_samples,
            "total_valid": total_valid,
            "validity_pct": validity_rate,
            "hdc_distance_mean": hdc_dist_mean,
            "hdc_distance_std": hdc_dist_std,
            "per_mol_validity_mean": float(np.mean(per_mol_validity)),
            "per_mol_validity_std": float(np.std(per_mol_validity)),
        },
        "molecules": all_results,
    }

    e.commit_json("generation_results.json", summary_dict)
    e.log("\nExperiment completed!")


# =============================================================================
# Testing Mode
# =============================================================================


@experiment.testing
def testing(e: Experiment) -> None:
    """Quick test mode with reduced parameters."""
    e.NUM_MOLECULES = 2
    e.NUM_SAMPLES_PER_MOL = 2
    e.NUM_REPETITIONS = 1
    e.SAMPLE_STEPS = 10
    e.FM_SAMPLE_STEPS = 10
    e.DATASET = "zinc"


# =============================================================================
# Entry Point
# =============================================================================

experiment.run_if_main()
