#!/usr/bin/env python
"""
Encode → decode demo for two ZINC molecules.

Loads the HyperNet from `experiments/encoders/zinc_d1024_depth3_k6_10_14_b8.ckpt`,
encodes each SMILES into HDC `edge_terms` + `graph_embedding`, then reconstructs
the graph using either `decode_graph_greedy` (beam search) or
`decode_graph_astar` (best-first search). Saves a side-by-side PNG of original
vs. reconstructed and prints detailed per-molecule diagnostics to stdout.

Usage:
    python experiments/generation/encode_decode_demo.py                    # default: greedy
    python experiments/generation/encode_decode_demo.py --decoder greedy
    python experiments/generation/encode_decode_demo.py --decoder astar
    python experiments/generation/encode_decode_demo.py --decoder astar --budget-seconds 60
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from rdkit import Chem
from rdkit.Chem import Draw

from graph_hdc.datasets.zinc_smiles import mol_to_data
from graph_hdc.hypernet import load_hypernet
from graph_hdc.hypernet.configs import AStarDecoderSettings, FallbackDecoderSettings
from graph_hdc.utils.chem import reconstruct_for_eval

REPO = Path(__file__).resolve().parent.parent.parent
ENCODER_PATH = REPO / "experiments/encoders/zinc_d1024_depth3_k6_10_14_b10.ckpt"

SMILES = [
    "CC(=O)Nc1ccc(Nc2nccc(OCc3ccccc3)n2)cc1",
    "Cc1cccc2[nH]c(C(=O)NC(c3cc(=O)[nH]c(-c4ccccn4)n3)C(C)C)cc12",
]

# Greedy defaults
BEAM_SIZE = 1028 * 2
POPULATION_LIMIT = 4096

# A* defaults
ASTAR_BUDGET_SECONDS = 30.0
ASTAR_MAX_BATCH_SIZE = 512
ASTAR_MAX_FRONTIER_SIZE: int | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--decoder", choices=("greedy", "astar"), default="greedy",
        help="Which decoder to use (default: greedy).",
    )
    p.add_argument(
        "--budget-seconds", type=float, default=ASTAR_BUDGET_SECONDS,
        help="[astar only] Wall-clock budget per molecule in seconds.",
    )
    p.add_argument(
        "--max-batch-size", type=int, default=ASTAR_MAX_BATCH_SIZE,
        help="[astar only] Max graphs per encoder forward pass (OOM cap).",
    )
    p.add_argument(
        "--max-frontier-size", type=int, default=ASTAR_MAX_FRONTIER_SIZE,
        help="[astar only] Optional cap on frontier size.",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Output PNG path. Defaults to encode_decode_demo_<decoder>.png next to this script.",
    )
    return p.parse_args()


def encode_smiles(hypernet, smi: str):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smi}")
    data = mol_to_data(mol)
    data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
    out = hypernet.forward(data)
    return out["edge_terms"][0], out["graph_embedding"][0], data


def edge_multiset_from_data(data) -> Counter:
    """Ground-truth bidirectional edge multiset from a PyG Data object."""
    x = data.x.long().tolist()
    src, dst = data.edge_index.tolist()
    return Counter((tuple(x[s]), tuple(x[d])) for s, d in zip(src, dst, strict=True))


def decode_one(hypernet, edge_term, graph_term, vsa_cls, method, settings):
    edge_term = edge_term.as_subclass(vsa_cls)
    graph_term = graph_term.as_subclass(vsa_cls)
    if method == "greedy":
        return hypernet.decode_graph_greedy(
            edge_term=edge_term,
            graph_term=graph_term,
            decoder_settings=settings,
        )
    if method == "astar":
        return hypernet.decode_graph_astar(
            edge_term=edge_term,
            graph_term=graph_term,
            decoder_settings=settings,
        )
    raise ValueError(f"Unknown decoder method: {method!r}")


def canonical(smi: str | None) -> str:
    if not smi:
        return "INVALID"
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol, canonical=True) if mol is not None else "INVALID"


def main():
    args = parse_args()
    device = torch.device("cpu")

    print(f"Loading HyperNet from {ENCODER_PATH}")
    hypernet = load_hypernet(str(ENCODER_PATH), device=device)
    hypernet.eval()
    vsa_cls = hypernet.vsa.tensor_class
    print(f"  hv_dim={hypernet.hv_dim}, depth={hypernet.depth}, "
          f"vsa={hypernet.vsa.value}, base_dataset={hypernet.base_dataset}")

    print(f"Decoder: {args.decoder}")
    if args.decoder == "greedy":
        settings = FallbackDecoderSettings(
            beam_size=BEAM_SIZE,
            initial_limit=POPULATION_LIMIT,
            limit=POPULATION_LIMIT,
            top_k=1,
        )
        print(
            f"Greedy settings: beam_size={settings.beam_size}, "
            f"initial_limit={settings.initial_limit}, limit={settings.limit}, "
            f"top_k={settings.top_k}"
        )
    else:
        settings = AStarDecoderSettings(
            budget_seconds=args.budget_seconds,
            max_batch_size=args.max_batch_size,
            max_frontier_size=args.max_frontier_size,
            top_k=1,
        )
        print(
            f"A* settings: budget_seconds={settings.budget_seconds}, "
            f"max_batch_size={settings.max_batch_size}, "
            f"max_frontier_size={settings.max_frontier_size}, "
            f"top_k={settings.top_k}"
        )

    output_path = args.output or (
        Path(__file__).resolve().parent / f"encode_decode_demo_{args.decoder}.png"
    )

    originals: list[Chem.Mol] = []
    reconstructed: list[Chem.Mol | None] = []
    recon_smiles: list[str] = []

    for i, smi in enumerate(SMILES):
        print(f"\n── Molecule {i + 1} ──────────────────────────────")
        print(f"  Original SMILES : {smi}")
        canon_orig = canonical(smi)
        print(f"  Canonical       : {canon_orig}")

        mol = Chem.MolFromSmiles(smi)
        originals.append(mol)
        n_atoms = mol.GetNumAtoms()
        n_bonds = mol.GetNumBonds()
        print(f"  Atoms / Bonds   : {n_atoms} / {n_bonds}")

        with torch.no_grad():
            edge_term, graph_term, data = encode_smiles(hypernet, smi)
        print(f"  edge_term shape : {tuple(edge_term.shape)}")
        print(f"  graph_term shape: {tuple(graph_term.shape)}")

        # Compare decoded edge multiset against the ground-truth multiset
        gt_edges = edge_multiset_from_data(data)
        with torch.no_grad():
            decoded_edges = hypernet.decode_order_one_no_node_terms(
                edge_term=edge_term.as_subclass(vsa_cls).clone()
            )
        decoded_edges_ctr = Counter(decoded_edges)
        edges_match = decoded_edges_ctr == gt_edges
        gt_total = sum(gt_edges.values())
        dec_total = sum(decoded_edges_ctr.values())
        intersection = sum((decoded_edges_ctr & gt_edges).values())
        missing_ctr = gt_edges - decoded_edges_ctr
        extra_ctr = decoded_edges_ctr - gt_edges
        print(f"  edge multiset   : decoded={dec_total} | gt={gt_total} | "
              f"correct={intersection} | missing={sum(missing_ctr.values())} | "
              f"extra={sum(extra_ctr.values())}")
        print(f"  edge multiset OK: {edges_match}")
        if not edges_match:
            if missing_ctr:
                top_missing = missing_ctr.most_common(3)
                print(f"    e.g. missing  : {top_missing}")
            if extra_ctr:
                top_extra = extra_ctr.most_common(3)
                print(f"    e.g. extra    : {top_extra}")

        with torch.no_grad():
            result = decode_one(
                hypernet, edge_term, graph_term, vsa_cls, args.decoder, settings,
            )

        print(f"  target_reached  : {result.target_reached}")
        print(f"  correction_level: {result.correction_level}")
        print(f"  candidates      : {len(result.nx_graphs)}")
        if result.cos_similarities:
            top_sim = float(result.cos_similarities[0])
            print(f"  top cos-sim     : {top_sim:.4f}")
        if result.final_flags:
            print(f"  final_flags     : {result.final_flags}")

        # A* search statistics (only present when decoder == 'astar')
        if result.search_stats is not None:
            s = result.search_stats
            print(f"  ── A* search stats ──")
            print(f"    pops              : {s.pops}   (completes={s.completes}  dead_ends={s.dead_end_pops})")
            print(f"    early_exit        : {s.early_exit}   budget_exhausted: {s.budget_exhausted}")
            print(f"    seeds pushed      : {s.seeds_pushed}")
            print(f"    children generated: {s.children_generated}")
            print(f"      ruleA (nonring on new cycle): {s.rule_a_rejected}")
            print(f"      dedup duplicates             : {s.dedup_rejected}")
            print(f"      rulesBC (ring membership)    : {s.ring_bc_rejected}")
            print(f"      passed  → scored & pushed    : {s.children_scored}")
            print(f"    encoder forwards  : {s.encoder_forwards}  (graphs scored: {s.encoder_graphs_scored})")
            print(f"    unique states seen: {s.unique_states_seen}")
            print(f"    heap peak / final : {s.heap_peak} / {s.heap_final}")
            print(f"    time phase0/search: {s.seconds_phase0:.2f}s / {s.seconds_search:.2f}s")

        recon_mol: Chem.Mol | None = None
        if result.nx_graphs:
            try:
                recon_mol = reconstruct_for_eval(result.nx_graphs[0], dataset="zinc")
            except Exception as exc:
                print(f"  reconstruct_for_eval failed: {exc}")

        recon_smi = canonical(Chem.MolToSmiles(recon_mol)) if recon_mol is not None else "INVALID"
        match = recon_smi == canon_orig
        print(f"  Reconstructed   : {recon_smi}")
        print(f"  Exact match     : {match}")

        reconstructed.append(recon_mol)
        recon_smiles.append(recon_smi)

    # ── Side-by-side figure ───────────────────────────────────────────
    n = len(SMILES)
    fig, axes = plt.subplots(n, 2, figsize=(7, 3.5 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    for i, (orig_mol, recon_mol, smi, recon_smi) in enumerate(
        zip(originals, reconstructed, SMILES, recon_smiles, strict=True)
    ):
        for col, (mol, title) in enumerate([
            (orig_mol, f"Original\n{smi}"),
            (recon_mol, f"Reconstructed\n{recon_smi}"),
        ]):
            ax = axes[i, col]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=8)
            if mol is not None:
                ax.imshow(Draw.MolToImage(mol, size=(350, 350)))
            else:
                ax.text(0.5, 0.5, "INVALID", ha="center", va="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    print(f"\nSaved figure to {output_path}")


if __name__ == "__main__":
    main()
