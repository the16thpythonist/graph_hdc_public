#!/usr/bin/env python
"""
Generate molecules from a trained Flow Matching checkpoint.

Loads a FlowMatchingModel, samples HDC vectors, then decodes them to
molecular graphs using the HyperNet greedy decoder.

Usage:
    python experiments/scripts/generate_from_fm.py \
        --checkpoint /path/to/flow_matching.ckpt \
        --n_samples 10 \
        --device cpu
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import time
from pathlib import Path

import psutil
import torch
from rdkit import Chem
from tqdm.auto import tqdm

from graph_hdc.datasets.utils import get_split
from graph_hdc.hypernet import HyperNet
from graph_hdc.hypernet.configs import (
    DecoderSettings,
    FallbackDecoderSettings,
    create_config_with_rw,
)
from graph_hdc.models.flow_matching import FlowMatchingModel
from graph_hdc.utils.chem import reconstruct_for_eval


# =============================================================================
# Memory tracking
# =============================================================================

def mem_mb() -> float:
    """Current RSS of this process in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


def gpu_mb() -> str:
    """Current GPU memory usage string, or '' if no CUDA."""
    if not torch.cuda.is_available():
        return ""
    alloc = torch.cuda.memory_allocated() / 1e6
    reserved = torch.cuda.memory_reserved() / 1e6
    return f"  GPU: {alloc:.0f}MB alloc / {reserved:.0f}MB reserved"


def log_mem(label: str) -> None:
    """Print a memory checkpoint line."""
    print(f"[MEM] {label}: {mem_mb():.0f} MB RSS{gpu_mb()}")


# =============================================================================
# Dataset scanning for codebook pruning
# =============================================================================

def scan_observed_features(
    dataset_name: str,
    num_base_features: int = 5,
    max_samples: int | None = None,
) -> tuple[set[tuple], set[tuple[tuple, tuple]]]:
    """Scan dataset to collect observed node types and edge pairs.

    Only uses the base node features (no RRWP), so it works with a
    plain HyperNet.

    Returns
    -------
    (observed_nodes, observed_edges)
        observed_nodes: set of node feature tuples
        observed_edges: set of (src_tuple, dst_tuple) edge pairs
    """
    node_features: set[tuple] = set()
    edge_pairs: set[tuple[tuple, tuple]] = set()
    count = 0

    for split in ["train", "valid", "test"]:
        ds = get_split(split=split, dataset=dataset_name)
        for data in tqdm(ds, desc=f"Scanning {split}", unit="mol"):
            x = data.x[:, :num_base_features].int()
            node_tuples = {i: tuple(row.tolist()) for i, row in enumerate(x)}
            node_features.update(node_tuples.values())
            for u, v in data.edge_index.t().tolist():
                edge_pairs.add((node_tuples[u], node_tuples[v]))
            count += 1
            if max_samples is not None and count >= max_samples:
                return node_features, edge_pairs

    return node_features, edge_pairs


# =============================================================================
# Helpers
# =============================================================================

def create_hypernet(hv_dim: int, base_dataset: str, device: str) -> HyperNet:
    """Create a base HyperNet encoder/decoder for the given config."""
    config = create_config_with_rw(
        base_dataset=base_dataset,
        hv_dim=hv_dim,
    )
    config.device = device
    config.dtype = "float64"
    hypernet = HyperNet(config)
    hypernet.eval()
    return hypernet


def decode_molecule(
    hypernet: HyperNet,
    node_terms: torch.Tensor,
    graph_terms: torch.Tensor,
    decoder_settings: DecoderSettings,
    verbose: bool = False,
) -> dict:
    """Decode a single molecule from HDC vectors."""
    edge_term_approx = graph_terms - node_terms

    if verbose:
        # Phase 1: Edge decoding timing
        t0 = time.time()
        edge_vsa = hypernet.ensure_vsa(edge_term_approx.clone())
        decoded_edges = hypernet.decode_order_one_no_node_terms(edge_vsa)
        t_edge_decode = time.time() - t0
        n_edges = len(decoded_edges)
        print(f"    [diag] decode_order_one_no_node_terms: {t_edge_decode:.3f}s, {n_edges} edges decoded")

        # Phase 2: Full decode_graph (includes correction + greedy fallback)
        t1 = time.time()

    result = hypernet.decode_graph(
        edge_term=edge_term_approx,
        graph_term=graph_terms,
        decoder_settings=decoder_settings,
    )

    if verbose:
        t_full = time.time() - t1
        print(f"    [diag] decode_graph total: {t_full:.3f}s, "
              f"correction={result.correction_level}, "
              f"graphs={len(result.nx_graphs)}")

    if result.nx_graphs:
        G = result.nx_graphs[0]
        try:
            mol = reconstruct_for_eval(G, dataset=hypernet.base_dataset)
        except Exception:
            mol = None
        if mol is not None:
            try:
                smiles = Chem.MolToSmiles(mol, canonical=True)
                return {
                    "smiles": smiles,
                    "valid": True,
                    "similarity": result.best_similarity,
                    "correction_level": str(result.correction_level),
                }
            except Exception:
                pass

    return {
        "smiles": None,
        "valid": False,
        "similarity": 0.0,
        "correction_level": str(result.correction_level),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate molecules from a Flow Matching checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to FlowMatchingModel checkpoint (.ckpt)",
    )
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--sample_steps", type=int, default=100)
    parser.add_argument("--solver", type=str, default=None,
                        help="ODE solver override (euler, midpoint, dopri5)")
    parser.add_argument("--dataset", type=str, default="zinc",
                        choices=["qm9", "zinc"])
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu recommended to avoid GPU OOM)")
    parser.add_argument("--beam_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("Flow Matching Molecule Generation (memory-safe)")
    print("=" * 60)

    log_mem("startup (after imports)")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Step 1: Load checkpoint to CPU, then move ────────────────────
    log_mem("before checkpoint load")
    print(f"Loading FlowMatchingModel from {args.checkpoint}...")

    model = FlowMatchingModel.load_from_checkpoint(
        args.checkpoint, map_location="cpu",
    )
    log_mem("after load_from_checkpoint (CPU)")

    model.eval()
    model.to(device)
    gc.collect()
    log_mem("after model.to(device) + gc")

    if args.solver is not None:
        print(f"Overriding solver: {model.solver_method} -> {args.solver}")
        model.solver_method = args.solver

    hv_dim = model.data_dim // 2
    print(f"data_dim={model.data_dim}, hv_dim={hv_dim}")
    print(f"solver={model.solver_method}, steps={args.sample_steps}")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {num_params:,} ({num_params * 4 / 1e6:.1f} MB in float32)")

    # ── Step 2: Sample HDC vectors ───────────────────────────────────
    log_mem("before ODE sampling")
    print(f"\nSampling {args.n_samples} HDC vectors ({args.sample_steps} ODE steps)...")
    t0 = time.time()
    with torch.no_grad():
        vectors = model.sample(
            num_samples=args.n_samples,
            num_steps=args.sample_steps,
            device=device,
        )
    sample_time = time.time() - t0
    log_mem("after ODE sampling")
    print(f"Sampled {vectors.shape} in {sample_time:.2f}s")

    # Free flow model — no longer needed
    vectors_cpu = vectors.cpu().double()
    del vectors, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log_mem("after freeing flow model")

    node_terms_all = vectors_cpu[:, :hv_dim]
    graph_terms_all = vectors_cpu[:, hv_dim:]

    # ── Step 3: Create HyperNet with pruned codebooks ────────────────
    log_mem("before dataset scan")
    print(f"\nScanning {args.dataset} dataset for observed node/edge types...")
    observed_nodes, observed_edges = scan_observed_features(args.dataset)
    print(f"  Observed node types: {len(observed_nodes)} (of 1296 possible)")
    print(f"  Observed edge pairs: {len(observed_edges)} (of 1,679,616 possible)")
    log_mem("after dataset scan")

    print(f"Creating HyperNet (hv_dim={hv_dim}, dataset={args.dataset})...")
    hypernet = create_hypernet(hv_dim, args.dataset, device="cpu")
    log_mem("after HyperNet creation (unpruned)")

    # Prune codebooks to observed features — this is the critical step
    # that prevents the 34GB edge codebook explosion
    print("Pruning codebooks to observed features...")
    hypernet.limit_nodes_codebook(observed_nodes)
    print(f"  Nodes codebook: {hypernet.nodes_codebook.shape[0]} entries")
    log_mem("after limit_nodes_codebook")

    hypernet.limit_edges_codebook(observed_edges)
    edge_cb_size = hypernet._edges_codebook.shape[0] if hypernet._edges_codebook is not None else 0
    edge_cb_mb = edge_cb_size * hv_dim * 8 / 1e6  # float64
    print(f"  Edges codebook: {edge_cb_size} entries ({edge_cb_mb:.1f} MB)")
    log_mem("after limit_edges_codebook")

    # Decoder settings
    fallback = FallbackDecoderSettings(beam_size=args.beam_size, top_k=1)
    decoder_settings = DecoderSettings(
        iteration_budget=25,
        max_graphs_per_iter=1024,
        early_stopping=True,
        fallback_decoder_settings=fallback,
    )
    decoder_settings.top_k = 1

    # ── Step 4: Decode molecules ─────────────────────────────────────
    print(f"\nDecoding {args.n_samples} molecules...")
    results = []
    valid_count = 0
    decode_start = time.time()

    decode_times = []
    for i in range(args.n_samples):
        t_dec = time.time()
        r = decode_molecule(
            hypernet,
            node_terms_all[i],
            graph_terms_all[i],
            decoder_settings,
            verbose=True,
        )
        dt = time.time() - t_dec
        decode_times.append(dt)
        r["idx"] = i
        r["decode_time_sec"] = dt
        results.append(r)

        if r["valid"]:
            valid_count += 1

        status = r["smiles"] if r["smiles"] else "(failed to decode)"
        print(f"  [{i+1}/{args.n_samples}] {dt:.2f}s  {status}")

    decode_time = time.time() - decode_start
    log_mem("after all decoding")

    # ── Summary ──────────────────────────────────────────────────────
    valid_smiles = [r["smiles"] for r in results if r["valid"]]
    unique_smiles = set(valid_smiles)
    validity = 100.0 * valid_count / args.n_samples if args.n_samples > 0 else 0
    uniqueness = 100.0 * len(unique_smiles) / valid_count if valid_count > 0 else 0

    print(f"\n{'='*60}")
    print("GENERATION RESULTS")
    print(f"{'='*60}")
    print(f"  Total samples:  {args.n_samples}")
    print(f"  Valid:          {valid_count} ({validity:.1f}%)")
    print(f"  Unique:         {len(unique_smiles)} ({uniqueness:.1f}%)")
    print(f"  Sample time:    {sample_time:.2f}s")
    print(f"  Decode time:    {decode_time:.2f}s (avg {sum(decode_times)/len(decode_times):.2f}s/mol)")
    print(f"  Per-mol times:  {', '.join(f'{t:.1f}s' for t in decode_times)}")
    print(f"  Peak RSS:       {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")
    print(f"{'='*60}")

    if valid_smiles:
        print("\nGenerated SMILES:")
        for i, smi in enumerate(valid_smiles[:50]):
            print(f"  {i+1}. {smi}")

    # Save results
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(args.checkpoint).parent / "generated_molecules.json"

    summary = {
        "config": {
            "checkpoint": str(args.checkpoint),
            "n_samples": args.n_samples,
            "sample_steps": args.sample_steps,
            "dataset": args.dataset,
            "hv_dim": hv_dim,
            "beam_size": args.beam_size,
            "seed": args.seed,
        },
        "summary": {
            "validity_pct": validity,
            "uniqueness_pct": uniqueness,
            "n_valid": valid_count,
            "n_unique": len(unique_smiles),
            "sample_time_sec": sample_time,
            "decode_time_sec": decode_time,
        },
        "codebook_stats": {
            "observed_node_types": len(observed_nodes),
            "observed_edge_pairs": len(observed_edges),
            "edges_codebook_entries": edge_cb_size,
            "edges_codebook_mb": edge_cb_mb,
        },
        "molecules": results,
        "valid_smiles": list(unique_smiles),
    }

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
