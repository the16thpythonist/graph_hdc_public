#!/usr/bin/env python
"""
Property Optimization: QED Maximization and LogP Targeting.

Uses gradient-based optimization in the latent space to generate
molecules with desired properties.

Usage:
    # QED Maximization
    python run_optimization.py --mode qed_max --flow_ckpt path/to/flow.ckpt --regressor_ckpt path/to/qed_regressor.ckpt

    # LogP Targeting
    python run_optimization.py --mode logp_target --target 2.0 --flow_ckpt path/to/flow.ckpt --regressor_ckpt path/to/logp_regressor.ckpt
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from tqdm.auto import tqdm

from graph_hdc import (
    HyperNet,
    get_config,
    DecoderSettings,
    FallbackDecoderSettings,
    GenerationEvaluator,
    rdkit_logp,
    rdkit_qed,
    calculate_internal_diversity,
)
from graph_hdc.models.flows import RealNVPV3Lightning
from graph_hdc.models.regressors import PropertyRegressor
from graph_hdc.utils.chem import reconstruct_for_eval


def compute_top_k_property(values: list[float], k: int) -> float | None:
    """Compute average of top-k property values."""
    if not values or len(values) < k:
        return None
    sorted_vals = sorted(values, reverse=True)
    return float(np.mean(sorted_vals[:k]))


def compute_guacamol_score(
    validity: float,
    uniqueness: float,
    novelty: float,
    top_k_prop: float | None,
    int_div: float,
) -> float | None:
    """
    Compute GuacaMol-style composite score.

    The score is the geometric mean of the component metrics.
    For QED maximization, this includes validity, uniqueness, novelty,
    internal diversity, and the normalized QED score.
    """
    if top_k_prop is None:
        return None

    # Normalize property to [0, 1] (QED already in this range)
    prop_normalized = top_k_prop

    # Compute geometric mean of 5 components
    components = [validity, uniqueness, novelty, int_div, prop_normalized]

    # Handle zeros
    if any(c <= 0 for c in components):
        return 0.0

    return float(np.exp(np.mean(np.log(components))))


def optimize_latent(
    flow_model: RealNVPV3Lightning,
    regressor: PropertyRegressor,
    n_samples: int,
    target: float | None,
    mode: str,
    lr: float = 5e-4,
    steps: int = 500,
    lambda_prior: float = 0.01,
    grad_clip: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Optimize latent codes to achieve target property.

    Parameters
    ----------
    flow_model : RealNVPV3Lightning
        Trained flow model
    regressor : PropertyRegressor
        Property regressor
    n_samples : int
        Number of molecules to optimize
    target : float
        Target property value (for targeting mode)
    mode : str
        "qed_max" for QED maximization, "logp_target" for LogP targeting
    lr : float
        Learning rate
    steps : int
        Number of optimization steps
    lambda_prior : float
        Prior regularization weight
    grad_clip : float
        Gradient clipping value
    device : torch.device
        Computation device

    Returns
    -------
    tuple
        (edge_terms, graph_terms) optimized embeddings
    """
    flow_model.eval()
    regressor.eval()

    # Sample initial latent codes from prior
    z = torch.randn(n_samples, flow_model.flat_dim, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([z], lr=lr)

    pbar = tqdm(range(steps), desc="Optimizing")
    for step in pbar:
        optimizer.zero_grad()

        # Decode latent to data space
        x = flow_model.decode_from_latent(z)
        edge_terms, graph_terms = flow_model.split(x)

        # Predict property from full HDC embedding (edge_terms + graph_terms)
        # The regressor was trained on the concatenated representation
        prop_pred = regressor(x)

        # Compute loss
        if mode == "qed_max":
            # Maximize QED (minimize negative QED)
            prop_loss = -prop_pred.mean()
        else:
            # Target specific value
            prop_loss = ((prop_pred - target) ** 2).mean()

        # Prior regularization (stay close to standard Gaussian)
        prior_loss = (z ** 2).sum(dim=-1).mean()

        loss = prop_loss + lambda_prior * prior_loss

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_([z], grad_clip)

        optimizer.step()

        if step % 50 == 0:
            with torch.no_grad():
                mean_prop = prop_pred.mean().item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "prop": f"{mean_prop:.4f}"})

    # Get final embeddings
    with torch.no_grad():
        x_final = flow_model.decode_from_latent(z)
        edge_terms, graph_terms = flow_model.split(x_final)

    return edge_terms, graph_terms


def decode_and_evaluate(
    edge_terms: torch.Tensor,
    graph_terms: torch.Tensor,
    hypernet: HyperNet,
    evaluator: GenerationEvaluator,
    decoder_settings: DecoderSettings,
    mode: str,
    target: float | None,
    dataset: str = "qm9",
) -> dict:
    """Decode embeddings and evaluate results with comprehensive metrics."""
    n_samples = edge_terms.shape[0]

    results = {
        "n_samples": n_samples,
        "n_valid": 0,
        "valid_smiles": [],
        "properties": {"qed": [], "logp": []},
    }

    valid_mols = []

    print("\nDecoding optimized molecules...")
    for i in tqdm(range(n_samples)):
        try:
            result = hypernet.decode_graph(
                edge_term=edge_terms[i],
                graph_term=graph_terms[i],
                decoder_settings=decoder_settings,
            )

            if result.nx_graphs:
                nx_graph = result.nx_graphs[0]
                try:
                    mol = reconstruct_for_eval(nx_graph, dataset=dataset)
                    smiles = Chem.MolToSmiles(mol, canonical=True)
                except Exception:
                    mol, smiles = None, None

                if mol is not None and smiles:
                    results["n_valid"] += 1
                    results["valid_smiles"].append(smiles)
                    valid_mols.append(mol)

                    # Use central evaluator's property functions
                    qed_val = rdkit_qed(mol)
                    logp_val = rdkit_logp(mol)

                    if qed_val is not None:
                        results["properties"]["qed"].append(qed_val)
                    if logp_val is not None:
                        results["properties"]["logp"].append(logp_val)

        except Exception as e:
            warnings.warn(f"Decoding failed for sample {i}: {e}", RuntimeWarning, stacklevel=2)

    # Core metrics using evaluator's training set
    n_valid = results["n_valid"]
    valid_smiles = results["valid_smiles"]
    unique_smiles = list(set(valid_smiles))
    n_unique = len(unique_smiles)
    n_novel = len(set(valid_smiles) - evaluator.T)

    validity = n_valid / n_samples if n_samples > 0 else 0.0
    uniqueness = n_unique / n_valid if n_valid > 0 else 0.0
    novelty = n_novel / n_unique if n_unique > 0 else 0.0

    results["validity"] = validity
    results["n_unique"] = n_unique
    results["n_novel"] = n_novel
    results["uniqueness"] = uniqueness
    results["novelty"] = novelty

    # Diversity metrics using central evaluator's function
    print("Computing diversity metrics...")
    int_div1 = calculate_internal_diversity(valid_mols, radius=2) / 100.0 if len(valid_mols) >= 2 else 0.0
    int_div2 = calculate_internal_diversity(valid_mols, radius=3) / 100.0 if len(valid_mols) >= 2 else 0.0
    results["int_div1"] = int_div1
    results["int_div2"] = int_div2

    # Property statistics
    for prop_name, values in results["properties"].items():
        if values:
            results[f"{prop_name}_mean"] = float(np.mean(values))
            results[f"{prop_name}_std"] = float(np.std(values))
            results[f"{prop_name}_max"] = float(np.max(values))
            results[f"{prop_name}_min"] = float(np.min(values))

    # Mode-specific metrics
    if mode == "qed_max" and results["properties"]["qed"]:
        qed_values = results["properties"]["qed"]

        # Top-k QED scores (k = 1, 2, 3, 10, 100)
        for k in [1, 2, 3, 10, 100]:
            top_k = compute_top_k_property(qed_values, k)
            if top_k is not None:
                results[f"top_{k}_qed"] = top_k

        # GuacaMol-style composite scores for top-1, top-10, top-100
        print("Computing GuacaMol composite scores...")
        for k in [1, 10, 100]:
            top_k_qed = compute_top_k_property(qed_values, k)
            guacamol = compute_guacamol_score(
                validity=validity,
                uniqueness=uniqueness,
                novelty=novelty,
                top_k_prop=top_k_qed,
                int_div=int_div1,
            )
            if guacamol is not None:
                results[f"guacamol_top_{k}"] = guacamol

    if mode == "logp_target" and results["properties"]["logp"]:
        errors = [abs(v - target) for v in results["properties"]["logp"]]
        results["mae"] = float(np.mean(errors))
        results["within_0.5"] = sum(1 for e in errors if e < 0.5) / len(errors)
        results["within_1.0"] = sum(1 for e in errors if e < 1.0) / len(errors)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Property-guided molecular optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", type=str, required=True, choices=["qed_max", "logp_target"])
    parser.add_argument("--flow_ckpt", type=str, required=True, help="Path to flow model checkpoint")
    parser.add_argument("--regressor_ckpt", type=str, required=True, help="Path to regressor checkpoint")
    parser.add_argument("--dataset", type=str, default="qm9", choices=["qm9", "zinc"])
    parser.add_argument("--target", type=float, default=2.0, help="Target value for logp_target mode")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lambda_prior", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    print(f"Mode: {args.mode}")
    if args.mode == "logp_target":
        print(f"Target LogP: {args.target}")

    # Get dataset config
    if args.dataset == "qm9":
        ds_config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
    else:
        ds_config = get_config("ZINC_SMILES_HRR_256_F64_5G1NG4")

    # Load flow model (use custom loader for module path remapping)
    print(f"\nLoading flow model from {args.flow_ckpt}...")
    flow_model = RealNVPV3Lightning.load_from_checkpoint_file(args.flow_ckpt)
    flow_model.eval()
    flow_model.to(device)

    # Load regressor
    print(f"Loading regressor from {args.regressor_ckpt}...")
    regressor = PropertyRegressor.load_from_checkpoint_file(args.regressor_ckpt)
    regressor.eval()
    regressor.to(device)

    # Load HyperNet
    print("Initializing HyperNet...")
    ds_config.device = str(device)
    ds_config.dtype = "float32"
    hypernet = HyperNet(ds_config)
    hypernet.eval()

    # Initialize central evaluator for training set and consistent metrics
    print("Initializing GenerationEvaluator...")
    evaluator = GenerationEvaluator(base_dataset=args.dataset, device=device)
    print(f"Training set size: {len(evaluator.T)}")

    # Optimize
    print(f"\nOptimizing {args.n_samples} molecules for {args.steps} steps...")
    edge_terms, graph_terms = optimize_latent(
        flow_model=flow_model,
        regressor=regressor,
        n_samples=args.n_samples,
        target=args.target if args.mode == "logp_target" else None,
        mode=args.mode,
        lr=args.lr,
        steps=args.steps,
        lambda_prior=args.lambda_prior,
        grad_clip=args.grad_clip,
        device=device,
    )

    # Decode and evaluate
    fallback_settings = FallbackDecoderSettings(
        beam_size=32,
        top_k=1,
    )
    decoder_settings = DecoderSettings(
        iteration_budget=1 if args.dataset == "qm9" else 10,
        max_graphs_per_iter=1024,
        early_stopping=True,
        fallback_decoder_settings=fallback_settings,
    )
    decoder_settings.top_k = 1

    results = decode_and_evaluate(
        edge_terms=edge_terms,
        graph_terms=graph_terms,
        hypernet=hypernet,
        evaluator=evaluator,
        decoder_settings=decoder_settings,
        mode=args.mode,
        target=args.target if args.mode == "logp_target" else None,
        dataset=args.dataset,
    )

    # Print results
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)

    # Core metrics
    print("\n--- Core Metrics ---")
    print(f"Validity:    {results['validity']:.4f} ({results['n_valid']}/{results['n_samples']})")
    print(f"Uniqueness:  {results['uniqueness']:.4f} ({results['n_unique']}/{results['n_valid']})")
    print(f"Novelty:     {results['novelty']:.4f} ({results['n_novel']}/{results['n_unique']})")

    # Diversity metrics
    print("\n--- Diversity Metrics ---")
    print(f"IntDiv1:     {results['int_div1']:.4f}")
    print(f"IntDiv2:     {results['int_div2']:.4f}")

    if args.mode == "qed_max":
        # Top-k QED scores
        print("\n--- Top-k QED Scores ---")
        for k in [1, 2, 3, 10, 100]:
            key = f"top_{k}_qed"
            if key in results:
                print(f"Top-{k:3d} QED: {results[key]:.4f}")

        # GuacaMol composite scores
        print("\n--- GuacaMol Composite Scores ---")
        for k in [1, 10, 100]:
            key = f"guacamol_top_{k}"
            if key in results:
                print(f"GuacaMol (Top-{k:3d}): {results[key]:.4f}")

        # Property statistics
        print("\n--- QED Statistics ---")
        print(f"Mean QED:    {results.get('qed_mean', 'N/A')}")
        print(f"Std QED:     {results.get('qed_std', 'N/A')}")
        print(f"Max QED:     {results.get('qed_max', 'N/A')}")
        print(f"Min QED:     {results.get('qed_min', 'N/A')}")
    else:
        # LogP targeting metrics
        print("\n--- LogP Targeting Metrics ---")
        print(f"Target LogP:  {args.target}")
        print(f"MAE:          {results.get('mae', 'N/A')}")
        print(f"Within 0.5:   {results.get('within_0.5', 'N/A')}")
        print(f"Within 1.0:   {results.get('within_1.0', 'N/A')}")
        print(f"Mean LogP:    {results.get('logp_mean', 'N/A')}")
        print(f"Std LogP:     {results.get('logp_std', 'N/A')}")

    print("="*60)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(args.flow_ckpt).parent / f"optimization_{args.mode}.json"

    # Don't save full SMILES list in JSON (just first 20)
    results["valid_smiles"] = results["valid_smiles"][:20]

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
