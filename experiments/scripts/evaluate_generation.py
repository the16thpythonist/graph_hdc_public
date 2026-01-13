#!/usr/bin/env python
"""
Evaluate Unconditional Molecular Generation.

Computes:
- Validity, Uniqueness, Novelty (via GenerationEvaluator)
- Internal Diversity (IntDiv1, IntDiv2)
- Property distributions (LogP, QED, SA Score)
- FCD (Fréchet ChemNet Distance)
- KL Divergence on property distributions

Usage:
    python evaluate_generation.py --checkpoint path/to/flow.ckpt --dataset qm9 --n_samples 1000
"""

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from scipy import stats
from tqdm.auto import tqdm

# FCD import (optional)
try:
    from fcd import canonical_smiles, get_fcd, load_ref_model
    FCD_AVAILABLE = True
except ImportError:
    FCD_AVAILABLE = False

from graph_hdc import (
    DecoderSettings,
    FallbackDecoderSettings,
    GenerationEvaluator,
    HyperNet,
    get_config,
)
from graph_hdc.models.flows import RealNVPV3Lightning


def compute_fcd(gen_smiles: list[str], ref_smiles: list[str]) -> float | None:
    """
    Compute Fréchet ChemNet Distance between generated and reference molecules.

    Lower FCD means the generated distribution is closer to the reference.
    """
    if not FCD_AVAILABLE:
        return None

    if len(gen_smiles) < 2 or len(ref_smiles) < 2:
        return None

    try:
        # Canonicalize SMILES
        gen_canonical = [canonical_smiles(s) for s in gen_smiles if s]
        ref_canonical = [canonical_smiles(s) for s in ref_smiles if s]

        # Filter None values
        gen_canonical = [s for s in gen_canonical if s is not None]
        ref_canonical = [s for s in ref_canonical if s is not None]

        if len(gen_canonical) < 2:
            return None

        # Load model and compute FCD
        model = load_ref_model()
        fcd_score = get_fcd(gen_canonical, ref_canonical, model)
        return float(fcd_score)
    except Exception:
        return None


def compute_kl_divergence(
    gen_values: list[float],
    ref_values: list[float],
    n_bins: int = 50,
) -> float | None:
    """
    Compute KL divergence between generated and reference property distributions.

    Uses histogram-based estimation with smoothing to avoid division by zero.
    """
    if len(gen_values) < 10 or len(ref_values) < 10:
        return None

    try:
        # Determine bin edges from combined data
        all_values = gen_values + ref_values
        min_val, max_val = min(all_values), max(all_values)

        # Add small epsilon to avoid edge issues
        eps = (max_val - min_val) * 0.01
        bins = np.linspace(min_val - eps, max_val + eps, n_bins + 1)

        # Compute histograms
        gen_hist, _ = np.histogram(gen_values, bins=bins, density=True)
        ref_hist, _ = np.histogram(ref_values, bins=bins, density=True)

        # Add smoothing to avoid log(0)
        smooth = 1e-10
        gen_hist = gen_hist + smooth
        ref_hist = ref_hist + smooth

        # Normalize
        gen_hist = gen_hist / gen_hist.sum()
        ref_hist = ref_hist / ref_hist.sum()

        # Compute KL divergence: KL(gen || ref)
        kl = stats.entropy(gen_hist, ref_hist)
        return float(kl)
    except Exception:
        return None


def evaluate_generation(
    flow_model: RealNVPV3Lightning,
    hypernet: HyperNet,
    evaluator: GenerationEvaluator,
    n_samples: int,
    decoder_settings: DecoderSettings,
    device: torch.device,
    dataset: str = "qm9",
) -> dict:
    """
    Evaluate unconditional generation using central GenerationEvaluator.

    Returns metrics: validity, uniqueness, novelty, diversity, property stats,
    FCD score, and KL divergence for property distributions.
    """
    flow_model.eval()
    flow_model.to(device)

    # Sample from flow
    print(f"\nSampling {n_samples} molecules...")
    samples = flow_model.sample_split(n_samples)
    edge_terms = samples["edge_terms"].to(device)
    graph_terms = samples["graph_terms"].to(device)

    # Decode
    nx_graphs = []
    final_flags = []
    sims = []
    correction_levels = []

    print("Decoding molecules...")
    for i in tqdm(range(n_samples)):
        try:
            result = hypernet.decode_graph(
                edge_term=edge_terms[i],
                graph_term=graph_terms[i],
                decoder_settings=decoder_settings,
            )

            if result.nx_graphs:
                nx_graphs.append(result.nx_graphs[0])
                final_flags.append(True)
                sims.append(result.best_similarity)
                correction_levels.append(result.correction_level)
            else:
                # Add empty graph as placeholder
                import networkx as nx
                nx_graphs.append(nx.Graph())
                final_flags.append(False)
                sims.append(0.0)
                correction_levels.append(result.correction_level)

        except Exception as e:
            warnings.warn(f"Decoding failed for sample {i}: {e}", RuntimeWarning, stacklevel=2)
            import networkx as nx
            nx_graphs.append(nx.Graph())
            final_flags.append(False)
            sims.append(0.0)
            from graph_hdc import CorrectionLevel
            correction_levels.append(CorrectionLevel.FAIL)

    # Use central evaluator for core metrics
    eval_results = evaluator.evaluate(
        n_samples=n_samples,
        samples=nx_graphs,
        final_flags=final_flags,
        sims=sims,
        correction_levels=correction_levels,
    )

    # Get valid molecules and SMILES for FCD/KL computation
    mols, valid_flags, _, _ = evaluator.get_mols_valid_flags_sims_and_correction_levels()
    valid_smiles = []
    gen_properties = {"logp": [], "qed": []}

    for mol, valid in zip(mols, valid_flags, strict=False):
        if valid and mol is not None:
            try:
                smiles = Chem.MolToSmiles(mol, canonical=True)
                valid_smiles.append(smiles)

                # Get properties from mol for KL divergence
                from graph_hdc import rdkit_logp, rdkit_qed
                logp = rdkit_logp(mol)
                qed = rdkit_qed(mol)
                gen_properties["logp"].append(logp)
                gen_properties["qed"].append(qed)
            except Exception:
                pass

    # Compute FCD (Fréchet ChemNet Distance)
    print("Computing FCD...")
    fcd_score = compute_fcd(valid_smiles, evaluator.train_smiles_list)

    # Compute KL divergence for each property
    print("Computing KL divergence for property distributions...")
    kl_divergences = {}
    for prop_name in ["logp", "qed"]:
        gen_values = gen_properties.get(prop_name, [])
        ref_values = evaluator.train_properties.get(prop_name, [])
        kl = compute_kl_divergence(gen_values, ref_values)
        if kl is not None:
            kl_divergences[prop_name] = kl

    # Correction level distribution
    correction_counts = Counter([cl.value if hasattr(cl, 'value') else cl for cl in correction_levels])

    # Build results
    results = {
        "n_samples": n_samples,
        "n_valid": sum(valid_flags),
        "n_unique": int(eval_results["uniqueness"] * sum(valid_flags) / 100) if sum(valid_flags) > 0 else 0,
        "validity": eval_results["validity"] / 100.0,  # Convert to fraction
        "uniqueness": eval_results["uniqueness"] / 100.0,
        "novelty": eval_results["novelty"] / 100.0,
        "nuv": eval_results["nuv"] / 100.0,
        "int_div1": eval_results["internal_diversity_p1"] / 100.0,
        "int_div2": eval_results["internal_diversity_p2"] / 100.0,
        "correction_distribution": {str(k): v for k, v in correction_counts.items()},
        "property_stats": {
            "logp": {
                "mean": eval_results.get("logp_mean", float("nan")),
                "std": eval_results.get("logp_std", float("nan")),
            },
            "qed": {
                "mean": eval_results.get("qed_mean", float("nan")),
                "std": eval_results.get("qed_std", float("nan")),
            },
            "sa_score": {
                "mean": eval_results.get("sa_score_mean", float("nan")),
                "std": eval_results.get("sa_score_std", float("nan")),
            },
        },
        "fcd": fcd_score,
        "kl_divergence": kl_divergences,
        "valid_smiles": valid_smiles[:100],  # Save first 100 for inspection
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate unconditional molecular generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to flow model checkpoint")
    parser.add_argument("--dataset", type=str, default="qm9", choices=["qm9", "zinc"])
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--beam_size", type=int, default=None, help="Beam size for fallback decoder (default: 2048 for QM9, 64 for ZINC)")
    parser.add_argument("--iteration_budget", type=int, default=None, help="Iteration budget (default: 3 for QM9, 25 for ZINC)")
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

    # Get dataset config
    if args.dataset == "qm9":
        ds_config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
    else:
        ds_config = get_config("ZINC_SMILES_HRR_256_F64_5G1NG4")

    # Load flow model (use custom loader for module path remapping)
    print(f"\nLoading flow model from {args.checkpoint}...")
    flow_model = RealNVPV3Lightning.load_from_checkpoint_file(args.checkpoint)
    flow_model.eval()

    # Load HyperNet
    print("Initializing HyperNet...")
    ds_config.device = str(device)
    ds_config.dtype = "float32"
    hypernet = HyperNet(ds_config)
    hypernet.eval()

    # Initialize central evaluator
    print("Initializing GenerationEvaluator...")
    evaluator = GenerationEvaluator(base_dataset=args.dataset, device=device)
    print(f"Training set size: {len(evaluator.T)}")

    if not FCD_AVAILABLE:
        print("Warning: FCD package not available. Install with: pixi add fcd")

    # Decoder settings - defaults from original research
    if args.dataset == "qm9":
        beam_size = args.beam_size if args.beam_size is not None else 2048
        iteration_budget = args.iteration_budget if args.iteration_budget is not None else 3
    else:  # zinc
        beam_size = args.beam_size if args.beam_size is not None else 64
        iteration_budget = args.iteration_budget if args.iteration_budget is not None else 25

    fallback_settings = FallbackDecoderSettings(
        beam_size=beam_size,
        top_k=1,
    )
    decoder_settings = DecoderSettings(
        iteration_budget=iteration_budget,
        max_graphs_per_iter=1024,
        early_stopping=True,
        fallback_decoder_settings=fallback_settings,
    )
    decoder_settings.top_k = 1
    print(f"Decoder settings: beam_size={beam_size}, iteration_budget={iteration_budget}")

    # Evaluate
    results = evaluate_generation(
        flow_model=flow_model,
        hypernet=hypernet,
        evaluator=evaluator,
        n_samples=args.n_samples,
        decoder_settings=decoder_settings,
        device=device,
        dataset=args.dataset,
    )

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Validity:    {results['validity']:.4f} ({results['n_valid']}/{results['n_samples']})")
    print(f"Uniqueness:  {results['uniqueness']:.4f}")
    print(f"Novelty:     {results['novelty']:.4f}")
    print(f"NUV:         {results['nuv']:.4f}")
    print(f"IntDiv1:     {results['int_div1']:.4f}")
    print(f"IntDiv2:     {results['int_div2']:.4f}")

    # FCD score
    fcd = results.get("fcd")
    if fcd is not None:
        print(f"FCD:         {fcd:.4f}")
    else:
        print("FCD:         N/A (package not available or too few samples)")

    # KL divergence
    kl_div = results.get("kl_divergence", {})
    if kl_div:
        print("\nKL Divergence (gen || ref):")
        for prop, kl in kl_div.items():
            print(f"  {prop}: {kl:.4f}")

    print("\nProperty Statistics:")
    for prop, prop_stats in results["property_stats"].items():
        mean = prop_stats.get("mean", float("nan"))
        std = prop_stats.get("std", float("nan"))
        if not (np.isnan(mean) and np.isnan(std)):
            print(f"  {prop}: mean={mean:.3f}, std={std:.3f}")
    print("="*60)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(args.checkpoint).parent / f"eval_{args.n_samples}_samples.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
