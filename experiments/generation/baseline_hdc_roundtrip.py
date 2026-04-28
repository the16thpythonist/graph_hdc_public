"""One-off: pure HDC encode -> decode roundtrip baseline (no autoencoder).

Mirrors the per-epoch ``ReconstructionEvalCallback`` in
``train_autoencoder.py`` but skips the AE entirely: each molecule is
encoded with the HyperNet, then decoded directly from its own
``edge_terms`` / ``graph_terms`` (no AE reconstruction step). The
exact-match rate over the validation subset is the upper bound the
autoencoder can ever achieve given this encoder + decoder configuration.

Usage::

    python experiments/generation/baseline_hdc_roundtrip.py \\
        --encoder /media/ssd2/Programming/_branch/graph_hdc_public/experiments/encoders/zinc_d1024_depth3_k6_10_14_b8.ckpt \\
        --dataset zinc \\
        --n-eval 50 \\
        --beam-size 8 \\
        --limit 1024
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw

from graph_hdc.datasets.utils import (
    get_dataset_info,
    get_split,
    post_compute_encodings,
)
from graph_hdc.hypernet import load_hypernet
from graph_hdc.hypernet.configs import FallbackDecoderSettings
from graph_hdc.utils.chem import reconstruct_for_eval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--encoder",
        type=str,
        default=(
            "/media/ssd2/Programming/_branch/graph_hdc_public/experiments/"
            "encoders/zinc_d1024_depth3_k6_10_14_b8.ckpt"
        ),
        help="Path to HyperNet encoder .ckpt (same one used in training).",
    )
    parser.add_argument("--dataset", type=str, default="zinc")
    parser.add_argument(
        "--n-eval",
        type=int,
        default=50,
        help="Number of validation molecules to evaluate (matches RECON_EVAL_N_SAMPLES).",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=8,
        help="Decoder beam size (matches the per-epoch eval).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1024,
        help="Decoder population limit (matches the per-epoch eval).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default=str(
            Path(__file__).parent / "baseline_hdc_roundtrip.png"
        ),
        help="Where to save the original-vs-reconstructed grid plot.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"=== HDC roundtrip baseline (no autoencoder) ===")
    print(f"Encoder:   {args.encoder}")
    print(f"Dataset:   {args.dataset}")
    print(f"N eval:    {args.n_eval}")
    print(f"Beam size: {args.beam_size}")
    print(f"Limit:     {args.limit}")
    print(f"Device:    {device}")

    # -- 1. Load the same encoder used in training -----------------------
    if not Path(args.encoder).is_file():
        raise FileNotFoundError(f"Encoder checkpoint not found: {args.encoder}")
    hypernet = load_hypernet(args.encoder, device=str(device))

    # Prune edge codebook to the dataset's observed edges, mirroring
    # train_autoencoder.py's load_encoder hook.
    needs_rw = hasattr(hypernet, "rw_config") and hypernet.rw_config.enabled
    if needs_rw:
        from graph_hdc.datasets.utils import scan_features_with_rw

        print("Scanning observed edges (RW-augmented)...")
        _, observed_edges = scan_features_with_rw(
            args.dataset, hypernet.rw_config, max_samples=None,
        )
    else:
        observed_edges = get_dataset_info(args.dataset).edge_features
    hypernet.limit_edges_codebook(observed_edges)
    print(f"Pruned edge codebook to {len(observed_edges)} observed pairs.")

    # -- 2. Load + encode the validation set -----------------------------
    print("Loading validation split...")
    valid_ds = get_split("valid", dataset=args.dataset)
    print(f"Total valid molecules: {len(valid_ds)}")

    n_eval = min(args.n_eval, len(valid_ds))
    subset = valid_ds[:n_eval]
    print(f"Encoding {n_eval} molecules...")
    encoded = post_compute_encodings(
        subset, hypernet, device=device, batch_size=min(256, n_eval),
    )

    # -- 3. Pure roundtrip: decode straight from the original HDC vectors -
    decoder_settings = FallbackDecoderSettings(
        beam_size=args.beam_size,
        limit=args.limit,
        top_k=1,
    )
    decode_device = hypernet.nodes_codebook.device
    vsa_cls = hypernet.vsa.tensor_class

    n_total = len(encoded)
    n_valid = 0
    n_exact = 0
    tanimoto_sims: list[float] = []
    mol_pairs: list[tuple[Chem.Mol | None, Chem.Mol | None, bool, float]] = []
    t0 = time.time()

    for i, d in enumerate(encoded):
        edge_term = d.edge_terms.view(-1).to(decode_device).as_subclass(vsa_cls)
        graph_term = d.graph_terms.view(-1).to(decode_device).as_subclass(vsa_cls)

        orig_mol = Chem.MolFromSmiles(d.smiles)
        recon_mol = None

        try:
            result = hypernet.decode_graph_greedy(
                edge_term=edge_term,
                graph_term=graph_term,
                decoder_settings=decoder_settings,
            )
            if result.nx_graphs:
                g = result.nx_graphs[0]
                recon_mol = reconstruct_for_eval(g, dataset=args.dataset)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}] decode failed: {exc}")

        is_exact = False
        tanimoto = 0.0
        if recon_mol is not None and orig_mol is not None:
            n_valid += 1

            # Match the eval's stereochemistry-stripped canonical comparison.
            orig_nosmi = Chem.RWMol(orig_mol)
            Chem.RemoveStereochemistry(orig_nosmi)
            recon_nosmi = Chem.RWMol(recon_mol)
            Chem.RemoveStereochemistry(recon_nosmi)
            orig_smi = Chem.MolToSmiles(orig_nosmi, canonical=True)
            recon_smi = Chem.MolToSmiles(recon_nosmi, canonical=True)
            is_exact = recon_smi == orig_smi
            if is_exact:
                n_exact += 1

            fp_orig = AllChem.GetMorganFingerprintAsBitVect(orig_mol, 2, nBits=2048)
            fp_recon = AllChem.GetMorganFingerprintAsBitVect(recon_mol, 2, nBits=2048)
            tanimoto = DataStructs.TanimotoSimilarity(fp_orig, fp_recon)
            tanimoto_sims.append(tanimoto)

        mol_pairs.append((orig_mol, recon_mol, is_exact, tanimoto))

        if (i + 1) % 10 == 0 or (i + 1) == n_total:
            print(
                f"  [{i + 1:>4d}/{n_total}] "
                f"exact={n_exact}/{i + 1} ({100.0 * n_exact / (i + 1):.1f}%), "
                f"valid={n_valid}/{i + 1} ({100.0 * n_valid / (i + 1):.1f}%)"
            )

    dt = time.time() - t0
    validity = 100.0 * n_valid / max(1, n_total)
    exact_match = 100.0 * n_exact / max(1, n_total)
    mean_tanimoto = (
        sum(tanimoto_sims) / len(tanimoto_sims) if tanimoto_sims else 0.0
    )

    print()
    print("=" * 60)
    print("Pure HDC roundtrip baseline (encoder + decoder only, no AE)")
    print("=" * 60)
    print(f"N total:       {n_total}")
    print(f"N valid:       {n_valid}  ({validity:.2f}%)")
    print(f"N exact:       {n_exact}  ({exact_match:.2f}%)")
    print(f"Mean Tanimoto: {mean_tanimoto:.4f}  (over valid reconstructions)")
    print(f"Decode time:   {dt:.1f}s ({dt / max(1, n_total):.2f}s/mol)")
    print("=" * 60)

    # -- 4. Side-by-side grid plot ---------------------------------------
    plot_path = Path(args.plot_path)
    print(f"\nRendering {n_total} side-by-side pairs to {plot_path} ...")

    n_pairs = len(mol_pairs)
    fig, axes = plt.subplots(n_pairs, 2, figsize=(8, 3 * n_pairs))
    if n_pairs == 1:
        axes = axes.reshape(1, 2)

    fig.suptitle(
        f"Pure HDC roundtrip — exact={exact_match:.1f}%, "
        f"valid={validity:.1f}%, tanimoto={mean_tanimoto:.3f} "
        f"(beam={args.beam_size}, limit={args.limit})",
        fontsize=14, fontweight="bold",
    )

    for i, (orig_mol, recon_mol, is_exact, tanimoto) in enumerate(mol_pairs):
        ax_orig = axes[i, 0]
        ax_recon = axes[i, 1]

        for ax in (ax_orig, ax_recon):
            ax.set_xticks([])
            ax.set_yticks([])

        if orig_mol is not None:
            try:
                AllChem.Compute2DCoords(orig_mol)
                img = Draw.MolToImage(orig_mol, size=(300, 300))
                ax_orig.imshow(img)
            except Exception:
                ax_orig.text(0.5, 0.5, "render failed", ha="center", va="center")
        else:
            ax_orig.text(0.5, 0.5, "(no mol)", ha="center", va="center")

        if recon_mol is not None:
            try:
                AllChem.Compute2DCoords(recon_mol)
                img = Draw.MolToImage(recon_mol, size=(300, 300))
                ax_recon.imshow(img)
            except Exception:
                ax_recon.text(0.5, 0.5, "render failed", ha="center", va="center")
        else:
            ax_recon.text(0.5, 0.5, "(decode failed)", ha="center", va="center")

        ax_orig.set_ylabel(f"#{i}", fontsize=10)
        if i == 0:
            ax_orig.set_title("Original", fontsize=12, fontweight="bold")
            ax_recon.set_title("Reconstructed", fontsize=12, fontweight="bold")

        recon_color = "green" if is_exact else "red"
        recon_label = "EXACT" if is_exact else f"tani={tanimoto:.2f}"
        ax_recon.set_xlabel(recon_label, color=recon_color, fontsize=10, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.995])
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
