#!/usr/bin/env python
"""
HDC Interpolation - Linearly interpolate between molecules in HDC space.

For each pair of SMILES strings:
1. Encode both molecules to HDC vectors (concatenated [order_0 | order_N])
2. Linearly interpolate between them in HDC space
3. Reconstruct molecules at each interpolation point (decode nodes + generate edges)
4. Create a visualization showing the interpolation path

Output per pair: a 2-row plot where
  Row 1: original molecule A (left) and B (right), empty cells in between
  Row 2: reconstruction of A, N interpolation steps, reconstruction of B

Subplot titles show decoded SMILES and HDC cosine distance to the
interpolated vector at that step.

Usage:
    # Quick test
    python hdc_interpolation.py --__TESTING__ True

    # Full run
    python hdc_interpolation.py \
        --HDC_ENCODER_PATH /path/to/encoder.ckpt \
        --FLOW_DECODER_PATH /path/to/decoder.ckpt

    # Custom interpolation steps
    python hdc_interpolation.py \
        --HDC_ENCODER_PATH /path/to/encoder.ckpt \
        --FLOW_DECODER_PATH /path/to/decoder.ckpt \
        --NUM_INTERPOLATION_STEPS 10
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.data import Data

from graph_hdc.hypernet import load_hypernet
from graph_hdc.hypernet.encoder import HyperNet
from graph_hdc.models.flow_edge_decoder import (
    FlowEdgeDecoder,
    node_tuples_to_onehot,
    onehot_to_raw_features,
)
from graph_hdc.utils.experiment_helpers import (
    create_test_dummy_models,
    decode_nodes_from_hdc,
    get_canonical_smiles,
    pyg_to_mol,
    smiles_to_pyg_data,
)
from graph_hdc.utils.helpers import scatter_hd


# =============================================================================
# PARAMETERS
# =============================================================================

# -----------------------------------------------------------------------------
# Model Paths
# -----------------------------------------------------------------------------

# :param HDC_ENCODER_PATH:
#     Path to saved HyperNet encoder checkpoint (.ckpt). Required unless
#     running in __TESTING__ mode.
HDC_ENCODER_PATH: str = "/media/ssd2/Programming/graph_hdc_public/experiments/decoding/results/train_flow_edge_decoder_streaming/GOOD/hypernet_encoder.ckpt"

# :param FLOW_DECODER_PATH:
#     Path to saved FlowEdgeDecoder checkpoint (.ckpt). Required unless
#     running in __TESTING__ mode.
FLOW_DECODER_PATH: str = "/media/ssd2/Programming/graph_hdc_public/experiments/decoding/results/train_flow_edge_decoder_streaming/GOOD/last.ckpt"

# -----------------------------------------------------------------------------
# Dataset Configuration
# -----------------------------------------------------------------------------

# :param DATASET:
#     Dataset type for atom feature encoding. Options: "zinc", "qm9".
DATASET: str = "zinc"

# -----------------------------------------------------------------------------
# Interpolation Configuration
# -----------------------------------------------------------------------------

# :param INTERPOLATION_SAMPLES:
#     List of (SMILES_A, SMILES_B) tuples to interpolate between.
INTERPOLATION_SAMPLES: List[Tuple[str, str]] = [
    ("Nc1ccccc1SCCCOc1ccccc1", "C1C(C(C#N)N2C(C3=CC=CC=C3)CC(C)=N2)=COC=1"),
    ("C1C(C(C#N)N2C(C3=CC=CC=C3)CC(C)=N2)=COC=1", "CCN(Cc1ccc(OC)c(OC)c1)C(=O)c2cscc2"),
    ("CCO", "CC=O"),
]

# :param NUM_INTERPOLATION_STEPS:
#     Number of intermediate interpolation steps between the two endpoints.
#     Total columns in the plot = NUM_INTERPOLATION_STEPS + 2.
NUM_INTERPOLATION_STEPS: int = 10

# :param NUM_REPETITIONS:
#     Number of independent edge generation attempts per interpolation point.
#     When > 1, each attempt is scored by HDC cosine distance to the
#     interpolated vector, and the best result is kept.
NUM_REPETITIONS: int = 64

# -----------------------------------------------------------------------------
# Sampling Configuration
# -----------------------------------------------------------------------------

# :param SAMPLE_STEPS:
#     Number of denoising steps during discrete flow matching sampling.
SAMPLE_STEPS: int = 100

# :param ETA:
#     Stochasticity parameter for sampling. 0.0 = deterministic CTMC.
ETA: float = 0.0

# :param OMEGA:
#     Target guidance strength parameter for sampling.
OMEGA: float = 0.0

# :param SAMPLE_TIME_DISTORTION:
#     Time distortion schedule for sampling. Options: "identity", "polydec".
SAMPLE_TIME_DISTORTION: str = "polydec"

# :param NOISE_TYPE_OVERRIDE:
#     Override the noise type used during sampling. Options: "uniform",
#     "marginal", or None (use the type the model was trained with).
NOISE_TYPE_OVERRIDE: Optional[str] = None

# :param DETERMINISTIC:
#     If True, use argmax instead of sampling for deterministic trajectories.
DETERMINISTIC: bool = False

# -----------------------------------------------------------------------------
# System Configuration
# -----------------------------------------------------------------------------

# :param SEED:
#     Random seed for reproducibility.
SEED: int = 42

# :param DEVICE:
#     Device for the FlowEdgeDecoder inference. Options: "auto", "cpu", "cuda".
DEVICE: str = "cuda"

# -----------------------------------------------------------------------------
# Debug/Testing Modes
# -----------------------------------------------------------------------------

# :param __DEBUG__:
#     Debug mode - reuses same output folder during development.
__DEBUG__: bool = True

# :param __TESTING__:
#     Testing mode - runs with minimal iterations for validation.
__TESTING__: bool = False


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def encode_smiles(
    smiles: str,
    hypernet: HyperNet,
    dataset: str,
    hdc_device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Encode a SMILES string to a concatenated [order_0 | order_N] HDC vector.

    Returns:
        HDC vector of shape (concat_dim,) or None on failure.
    """
    data = smiles_to_pyg_data(smiles, dataset)
    if data is None:
        return None

    data = data.to(hdc_device)
    data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=hdc_device)

    with torch.no_grad():
        data = hypernet.encode_properties(data)
        if data.node_hv.device != hdc_device:
            data.node_hv = data.node_hv.to(hdc_device)

        order_zero = scatter_hd(src=data.node_hv, index=data.batch, op="bundle")
        encoder_output = hypernet.forward(data, normalize=True)
        order_n = encoder_output["graph_embedding"]
        hdc_vector = torch.cat([order_zero, order_n], dim=-1).squeeze(0)

    return hdc_vector


def compute_hdc_distance(
    generated_data: Data,
    target_hdc_vector: torch.Tensor,
    hypernet: HyperNet,
    dataset: str = "zinc",
) -> float:
    """
    Compute HDC cosine distance between a generated graph and a target HDC vector.

    Encodes the generated graph directly using its ``edge_index`` and node
    features (reversed from one-hot via ``onehot_to_raw_features``),
    bypassing the lossy SMILES round-trip.

    Returns:
        Cosine distance (float). Lower is better. Returns inf on failure.
    """
    try:
        raw_x = onehot_to_raw_features(generated_data.x)
        hdc_device = hypernet.nodes_codebook.device

        gen_data = Data(
            x=raw_x.to(hdc_device),
            edge_index=generated_data.edge_index.to(hdc_device),
        )
        gen_data.batch = torch.zeros(
            gen_data.x.size(0), dtype=torch.long, device=hdc_device
        )

        with torch.no_grad():
            gen_data = hypernet.encode_properties(gen_data)
            if gen_data.node_hv.device != hdc_device:
                gen_data.node_hv = gen_data.node_hv.to(hdc_device)

            order_zero = scatter_hd(src=gen_data.node_hv, index=gen_data.batch, op="bundle")
            output = hypernet.forward(gen_data, normalize=True)
            order_n = output["graph_embedding"]
            gen_hdc = torch.cat([order_zero, order_n], dim=-1).squeeze(0)

        return hypernet.calculate_distance(
            target_hdc_vector.unsqueeze(0).to(hdc_device),
            gen_hdc.unsqueeze(0),
        ).item()
    except Exception:
        return float("inf")


def reconstruct_from_hdc(
    hdc_vector: torch.Tensor,
    hypernet: HyperNet,
    decoder: FlowEdgeDecoder,
    base_hdc_dim: int,
    decoder_device: torch.device,
    sample_steps: int,
    eta: float,
    omega: float,
    time_distortion: str,
    noise_type_override: Optional[str],
    deterministic: bool,
    num_repetitions: int,
    dataset: str,
) -> Tuple[Optional[Data], Optional[Chem.Mol], Optional[str], float]:
    """
    Reconstruct a molecule from an HDC vector.

    Steps:
    1. Decode nodes from HDC
    2. Generate edges with FlowEdgeDecoder
    3. Convert to RDKit molecule
    4. Compute HDC distance to the input vector

    Returns:
        Tuple of (pyg_data, rdkit_mol, smiles, hdc_distance).
        Any field may be None if reconstruction fails.
    """
    node_tuples, num_nodes = decode_nodes_from_hdc(
        hypernet, hdc_vector.unsqueeze(0), base_hdc_dim
    )

    if num_nodes == 0:
        return None, None, None, float("inf")

    node_features = node_tuples_to_onehot(node_tuples, device=decoder_device).unsqueeze(0)
    node_mask = torch.ones(1, num_nodes, dtype=torch.bool, device=decoder_device)
    hdc_vectors_batch = hdc_vector.unsqueeze(0).to(decoder_device)

    if num_repetitions > 1:
        # Batched best-of-N
        batch_hdc = hdc_vectors_batch.expand(num_repetitions, -1)
        batch_nodes = node_features.expand(num_repetitions, -1, -1)
        batch_mask = node_mask.expand(num_repetitions, -1)

        with torch.no_grad():
            all_samples = decoder.sample(
                hdc_vectors=batch_hdc,
                node_features=batch_nodes,
                node_mask=batch_mask,
                sample_steps=sample_steps,
                eta=eta,
                omega=omega,
                time_distortion=time_distortion,
                noise_type_override=noise_type_override,
                show_progress=False,
                deterministic=deterministic,
                device=decoder_device,
            )

        # Select best by HDC distance
        best_sample = None
        best_dist = float("inf")
        for sample in all_samples:
            d = compute_hdc_distance(sample, hdc_vector, hypernet, dataset)
            if d < best_dist:
                best_dist = d
                best_sample = sample

        generated_data = best_sample
    else:
        with torch.no_grad():
            samples = decoder.sample(
                hdc_vectors=hdc_vectors_batch,
                node_features=node_features,
                node_mask=node_mask,
                sample_steps=sample_steps,
                eta=eta,
                omega=omega,
                time_distortion=time_distortion,
                noise_type_override=noise_type_override,
                show_progress=False,
                deterministic=deterministic,
                device=decoder_device,
            )
        generated_data = samples[0]

    if generated_data is None:
        return None, None, None, float("inf")

    mol = pyg_to_mol(generated_data)
    smiles = get_canonical_smiles(mol)
    dist = compute_hdc_distance(generated_data, hdc_vector, hypernet, dataset)

    return generated_data, mol, smiles, dist


def create_interpolation_plot(
    original_mol_a: Chem.Mol,
    original_mol_b: Chem.Mol,
    original_smiles_a: str,
    original_smiles_b: str,
    reconstructed_mols: List[Optional[Chem.Mol]],
    reconstructed_smiles: List[Optional[str]],
    hdc_distances: List[float],
    t_values: List[float],
    save_path: Path,
) -> None:
    """
    Create the interpolation visualization plot.

    Row 1: Original A (col 0), empty middle cells, Original B (last col)
    Row 2: All reconstructions/interpolations with SMILES + HDC distance titles
    """
    n_cols = len(reconstructed_mols)
    mol_img_size = (300, 300)

    fig, axes = plt.subplots(
        2, n_cols,
        figsize=(3.5 * n_cols, 7.5),
        dpi=150,
    )

    # Ensure axes is 2D even for n_cols=1
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    # --- Row 1: Originals ---
    for col in range(n_cols):
        ax = axes[0, col]
        if col == 0:
            # Original A
            try:
                img = Draw.MolToImage(original_mol_a, size=mol_img_size)
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, "Render\nFailed", ha="center", va="center",
                        fontsize=20, transform=ax.transAxes)
            smiles_display = original_smiles_a
            if len(smiles_display) > 30:
                smiles_display = smiles_display[:27] + "..."
            ax.set_title(f"Original A\n{smiles_display}", fontsize=14, pad=8)
        elif col == n_cols - 1:
            # Original B
            try:
                img = Draw.MolToImage(original_mol_b, size=mol_img_size)
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, "Render\nFailed", ha="center", va="center",
                        fontsize=20, transform=ax.transAxes)
            smiles_display = original_smiles_b
            if len(smiles_display) > 30:
                smiles_display = smiles_display[:27] + "..."
            ax.set_title(f"Original B\n{smiles_display}", fontsize=14, pad=8)
        else:
            ax.set_visible(False)
        ax.axis("off")

    # --- Row 2: Reconstructions / Interpolations ---
    for col in range(n_cols):
        ax = axes[1, col]
        mol = reconstructed_mols[col]
        smi = reconstructed_smiles[col]
        dist = hdc_distances[col]
        t = t_values[col]

        if mol is not None:
            try:
                img = Draw.MolToImage(mol, size=mol_img_size)
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, "Render\nFailed", ha="center", va="center",
                        fontsize=20, transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "Failed", ha="center", va="center",
                    fontsize=24, color="red", transform=ax.transAxes)

        # Title: SMILES + distance
        smi_display = smi if smi else "N/A"
        if len(smi_display) > 25:
            smi_display = smi_display[:22] + "..."
        dist_str = f"d={dist:.4f}" if dist != float("inf") else "d=inf"

        if col == 0:
            label = f"Recon A (t={t:.2f})"
        elif col == n_cols - 1:
            label = f"Recon B (t={t:.2f})"
        else:
            label = f"t={t:.2f}"

        ax.set_title(f"{label}\n{smi_display}\n{dist_str}", fontsize=12, pad=8)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# =============================================================================
# EXPERIMENT
# =============================================================================


@Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)
def experiment(e: Experiment) -> None:
    """HDC interpolation between molecule pairs."""

    e.log("=" * 60)
    e.log("HDC Interpolation Experiment")
    e.log("=" * 60)
    e.log_parameters()

    # Device setup
    if e.DEVICE == "auto":
        decoder_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        decoder_device = torch.device(e.DEVICE)
    hdc_device = torch.device("cpu")
    e.log(f"Decoder device: {decoder_device}")
    e.log(f"HyperNet device: {hdc_device} (always CPU)")

    # =========================================================================
    # Load Models
    # =========================================================================

    hypernet, decoder, base_hdc_dim = e.apply_hook("load_models", device=decoder_device)

    hypernet.to(hdc_device)
    hypernet.eval()
    decoder.to(decoder_device)
    decoder.eval()

    e["config/base_hdc_dim"] = base_hdc_dim
    e["config/device"] = str(decoder_device)

    # =========================================================================
    # Load SMILES Pairs
    # =========================================================================

    pairs = e.apply_hook("load_smiles_pairs")
    e.log(f"Number of SMILES pairs: {len(pairs)}")
    e["data/num_pairs"] = len(pairs)

    # Store config
    e["config/hdc_encoder_path"] = e.HDC_ENCODER_PATH
    e["config/flow_decoder_path"] = e.FLOW_DECODER_PATH
    e["config/dataset"] = e.DATASET
    e["config/num_interpolation_steps"] = e.NUM_INTERPOLATION_STEPS
    e["config/num_repetitions"] = e.NUM_REPETITIONS
    e["config/sample_steps"] = e.SAMPLE_STEPS

    # Output directory
    plots_dir = Path(e.path) / "interpolation_plots"
    plots_dir.mkdir(exist_ok=True)

    # =========================================================================
    # Process Each Pair
    # =========================================================================

    e.log("\n" + "=" * 60)
    e.log("Processing SMILES Pairs")
    e.log("=" * 60)

    all_results = []
    start_time = time.time()

    n_steps = e.NUM_INTERPOLATION_STEPS
    t_values = torch.linspace(0.0, 1.0, n_steps + 2).tolist()

    for pair_idx, (smiles_a, smiles_b) in enumerate(pairs):
        e.log(f"\nPair {pair_idx + 1}/{len(pairs)}: {smiles_a} <-> {smiles_b}")

        # Parse originals
        mol_a = Chem.MolFromSmiles(smiles_a)
        mol_b = Chem.MolFromSmiles(smiles_b)
        if mol_a is None or mol_b is None:
            e.log("  WARNING: Failed to parse one or both SMILES, skipping pair")
            all_results.append({
                "pair_idx": pair_idx,
                "smiles_a": smiles_a,
                "smiles_b": smiles_b,
                "status": "skipped",
                "error": "Failed to parse SMILES",
            })
            continue

        # Encode both molecules
        hdc_a = encode_smiles(smiles_a, hypernet, e.DATASET, hdc_device)
        hdc_b = encode_smiles(smiles_b, hypernet, e.DATASET, hdc_device)

        if hdc_a is None or hdc_b is None:
            e.log("  WARNING: Failed to encode one or both molecules, skipping pair")
            all_results.append({
                "pair_idx": pair_idx,
                "smiles_a": smiles_a,
                "smiles_b": smiles_b,
                "status": "skipped",
                "error": "Failed to encode to HDC",
            })
            continue

        e.log(f"  HDC vectors computed (dim={hdc_a.shape[0]})")

        # Compute endpoint HDC distance
        endpoint_dist = hypernet.calculate_distance(
            hdc_a.unsqueeze(0), hdc_b.unsqueeze(0)
        ).item()
        e.log(f"  Endpoint HDC distance: {endpoint_dist:.6f}")

        # Interpolate and reconstruct
        reconstructed_mols = []
        reconstructed_smiles = []
        hdc_distances = []
        pair_details = []

        for step_idx, t in enumerate(t_values):
            # Linear interpolation
            hdc_t = (1.0 - t) * hdc_a + t * hdc_b

            step_prefix = f"  Step {step_idx + 1}/{len(t_values)} (t={t:.3f}): "

            gen_data, gen_mol, gen_smiles, dist = reconstruct_from_hdc(
                hdc_vector=hdc_t,
                hypernet=hypernet,
                decoder=decoder,
                base_hdc_dim=base_hdc_dim,
                decoder_device=decoder_device,
                sample_steps=e.SAMPLE_STEPS,
                eta=e.ETA,
                omega=e.OMEGA,
                time_distortion=e.SAMPLE_TIME_DISTORTION,
                noise_type_override=e.NOISE_TYPE_OVERRIDE,
                deterministic=e.DETERMINISTIC,
                num_repetitions=e.NUM_REPETITIONS,
                dataset=e.DATASET,
            )

            reconstructed_mols.append(gen_mol)
            reconstructed_smiles.append(gen_smiles)
            hdc_distances.append(dist)

            e.log(f"{step_prefix}{gen_smiles or 'N/A'} (d={dist:.4f})")

            pair_details.append({
                "t": t,
                "smiles": gen_smiles,
                "hdc_distance": dist if dist != float("inf") else None,
                "is_valid": gen_mol is not None,
            })

        # Create plot
        plot_path = plots_dir / f"interpolation_{pair_idx + 1:03d}.png"
        create_interpolation_plot(
            original_mol_a=mol_a,
            original_mol_b=mol_b,
            original_smiles_a=smiles_a,
            original_smiles_b=smiles_b,
            reconstructed_mols=reconstructed_mols,
            reconstructed_smiles=reconstructed_smiles,
            hdc_distances=hdc_distances,
            t_values=t_values,
            save_path=plot_path,
        )
        e.log(f"  Plot saved: {plot_path.name}")

        all_results.append({
            "pair_idx": pair_idx,
            "smiles_a": smiles_a,
            "smiles_b": smiles_b,
            "endpoint_hdc_distance": endpoint_dist,
            "status": "completed",
            "steps": pair_details,
        })

    # =========================================================================
    # Summary
    # =========================================================================

    total_time = time.time() - start_time
    completed = sum(1 for r in all_results if r["status"] == "completed")
    skipped = sum(1 for r in all_results if r["status"] == "skipped")

    e.log("\n" + "=" * 60)
    e.log("SUMMARY")
    e.log("=" * 60)
    e.log(f"Total pairs: {len(pairs)}")
    e.log(f"Completed: {completed}")
    e.log(f"Skipped: {skipped}")
    e.log(f"Total time: {total_time:.2f} seconds")
    e.log("=" * 60)

    e["results/total_pairs"] = len(pairs)
    e["results/completed"] = completed
    e["results/skipped"] = skipped
    e["results/total_time_seconds"] = total_time
    e["results/details"] = all_results

    e.commit_json("interpolation_results.json", {
        "config": {
            "num_interpolation_steps": e.NUM_INTERPOLATION_STEPS,
            "num_repetitions": e.NUM_REPETITIONS,
            "sample_steps": e.SAMPLE_STEPS,
            "dataset": e.DATASET,
        },
        "summary": {
            "total_pairs": len(pairs),
            "completed": completed,
            "skipped": skipped,
            "total_time_seconds": total_time,
        },
        "results": all_results,
    })

    e.log("\nExperiment completed!")
    e.log(f"Plots saved to: {plots_dir}")


# =============================================================================
# HOOKS
# =============================================================================


@experiment.hook("load_models", default=True)
def load_models(
    e: Experiment,
    device: torch.device,
) -> Tuple[HyperNet, FlowEdgeDecoder, int]:
    """Load HyperNet encoder and FlowEdgeDecoder models."""
    e.log("\nLoading models...")

    if e.__TESTING__:
        e.log("TESTING MODE: Creating dummy models...")
        device = torch.device("cpu")
        hypernet, decoder, base_hdc_dim = create_test_dummy_models(device)
    else:
        if not e.HDC_ENCODER_PATH:
            raise ValueError("HDC_ENCODER_PATH is required")
        if not e.FLOW_DECODER_PATH:
            raise ValueError("FLOW_DECODER_PATH is required")

        e.log(f"Loading HyperNet from: {e.HDC_ENCODER_PATH}")
        hypernet = load_hypernet(e.HDC_ENCODER_PATH, device="cpu")
        base_hdc_dim = hypernet.hv_dim
        e.log(f"  HyperNet hv_dim: {base_hdc_dim}")

        e.log(f"Loading FlowEdgeDecoder from: {e.FLOW_DECODER_PATH}")
        decoder = FlowEdgeDecoder.load(e.FLOW_DECODER_PATH, device=device)
        e.log(f"  FlowEdgeDecoder hdc_dim: {decoder.hdc_dim}")

    return hypernet, decoder, base_hdc_dim


@experiment.hook("load_smiles_pairs", default=True)
def load_smiles_pairs(e: Experiment) -> List[Tuple[str, str]]:
    """Load SMILES pairs for interpolation."""
    return e.INTERPOLATION_SAMPLES


# =============================================================================
# Testing Mode
# =============================================================================


@experiment.testing
def testing(e: Experiment) -> None:
    """Quick test mode with reduced parameters."""
    e.SAMPLE_STEPS = 10
    e.NUM_INTERPOLATION_STEPS = 2
    e.NUM_REPETITIONS = 1
    e.INTERPOLATION_SAMPLES = [("CCO", "CC=O")]
    e.DATASET = "zinc"


# =============================================================================
# Entry Point
# =============================================================================

experiment.run_if_main()
