#!/usr/bin/env python
"""
Latent space interpolation between two ZINC molecules.

Picks two random molecules from the ZINC validation set, encodes them through
the HyperNet (HDC) and HDCAutoencoder, interpolates in latent space, then
decodes each interpolation step back to a molecule and plots the results.

Usage:
    python experiments/generation/latent_interpolation.py
"""
from __future__ import annotations

import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from rdkit import Chem
from rdkit.Chem import Draw

from graph_hdc.datasets.utils import get_split, post_compute_encodings
from graph_hdc.hypernet import load_hypernet
from graph_hdc.hypernet.configs import FallbackDecoderSettings
from graph_hdc.models.autoencoder import HDCAutoencoder
from graph_hdc.utils.chem import reconstruct_for_eval

# ── Paths ────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent.parent
ENCODER_PATH = REPO / "experiments/encoders/zinc_d1024_depth3_k6_10_14_b8.ckpt"
AE_CHECKPOINT = (
    REPO
    / "experiments/generation/results/train_autoencoder/24_04_2026__15_32__9tte/autoencoder.ckpt"
)
OUTPUT_PATH = REPO / "experiments/generation/results/latent_interpolation_lerp16.png"

N_STEPS = 16  # number of interpolation steps (including endpoints)
SEED = None  # set to an int for reproducibility


def main():
    device = torch.device("cpu")

    if SEED is not None:
        random.seed(SEED)
        torch.manual_seed(SEED)

    # ── Load models ──────────────────────────────────────────────────
    print("Loading HyperNet encoder...")
    hypernet = load_hypernet(str(ENCODER_PATH), device=device)
    hypernet.eval()
    hv_dim = hypernet.hv_dim

    print("Loading HDCAutoencoder...")
    model = HDCAutoencoder.load(str(AE_CHECKPOINT), map_location=device)
    model.eval()
    print(
        f"  data_dim={model.hparams.data_dim}, latent_dim={model.hparams.latent_dim}"
    )

    # ── Encode a small subset of ZINC validation ─────────────────────
    print("Loading ZINC validation split...")
    val_ds = get_split("valid", dataset="zinc")
    # Encode a small chunk (we only need 2 molecules, but encoding is batched)
    n_encode = min(200, len(val_ds))
    subset = val_ds[:n_encode]
    print(f"Encoding {n_encode} molecules...")
    encoded = post_compute_encodings(subset, hypernet, batch_size=64, device=device)
    # Swap edge_terms → node_terms slot (same as training script)
    for d in encoded:
        d.node_terms = d.edge_terms

    # ── Pick two random molecules ────────────────────────────────────
    idx_a, idx_b = random.sample(range(len(encoded)), 2)
    data_a, data_b = encoded[idx_a], encoded[idx_b]
    smi_a = data_a.smiles if hasattr(data_a, "smiles") else "mol_A"
    smi_b = data_b.smiles if hasattr(data_b, "smiles") else "mol_B"
    print(f"Molecule A (idx {idx_a}): {smi_a}")
    print(f"Molecule B (idx {idx_b}): {smi_b}")

    # Build HDC vectors [edge_terms | graph_terms]
    def extract_vec(d):
        node = d.node_terms.view(-1)
        graph = d.graph_terms.view(-1)
        return torch.cat([node, graph]).float()

    vec_a = extract_vec(data_a)
    vec_b = extract_vec(data_b)

    # ── Encode into latent space ─────────────────────────────────────
    with torch.no_grad():
        enc_a = model.encode(vec_a.unsqueeze(0))
        enc_b = model.encode(vec_b.unsqueeze(0))
        z_a = enc_a["mu"] if model.variational else enc_a["z"]  # [1, latent_dim]
        z_b = enc_b["mu"] if model.variational else enc_b["z"]

    # ── Linear interpolation (lerp) ────────────────────────────────
    alphas = torch.linspace(0.0, 1.0, N_STEPS)
    za, zb = z_a.squeeze(0), z_b.squeeze(0)
    z_interp = torch.stack([(1 - a) * za + a * zb for a in alphas])

    # Decode latent → HDC space
    with torch.no_grad():
        recon_vectors = model.decode(z_interp)  # [N_STEPS, data_dim]

    # ── Decode HDC vectors → molecules ───────────────────────────────
    vsa_cls = hypernet.vsa.tensor_class
    decoder_settings = FallbackDecoderSettings(
        beam_size=8,
        limit=1024,
        top_k=1,
    )

    mols: list[Chem.Mol | None] = []
    smiles_list: list[str] = []

    for i in range(N_STEPS):
        edge_term = recon_vectors[i, :hv_dim].as_subclass(vsa_cls)
        graph_term = recon_vectors[i, hv_dim:].as_subclass(vsa_cls)

        try:
            result = hypernet.decode_graph_greedy(
                edge_term=edge_term,
                graph_term=graph_term,
                decoder_settings=decoder_settings,
            )
            if result.nx_graphs:
                mol = reconstruct_for_eval(result.nx_graphs[0], dataset="zinc")
            else:
                mol = None
        except Exception as e:
            print(f"  Step {i} (alpha={alphas[i]:.2f}): decode failed – {e}")
            mol = None

        smi = Chem.MolToSmiles(mol) if mol is not None else "INVALID"
        smiles_list.append(smi)
        mols.append(mol)
        print(f"  alpha={alphas[i]:.2f}: {smi}")

    # ── Plot ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, N_STEPS, figsize=(3 * N_STEPS, 3.5))
    if N_STEPS == 1:
        axes = [axes]

    for i, (ax, mol, smi) in enumerate(zip(axes, mols, smiles_list)):
        ax.set_xticks([])
        ax.set_yticks([])
        label = f"α={alphas[i]:.2f}\n{smi}"
        if i == 0:
            label = f"A: {smi_a}\n(original)"
        elif i == N_STEPS - 1:
            label = f"B: {smi_b}\n(original)"

        if mol is not None:
            img = Draw.MolToImage(mol, size=(300, 300))
            ax.imshow(img)
        else:
            ax.text(
                0.5, 0.5, "INVALID", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="red",
            )
        ax.set_title(label, fontsize=8, wrap=True)

    fig.suptitle("Latent Space Interpolation (HDCAutoencoder)", fontsize=13)
    fig.tight_layout()
    fig.savefig(str(OUTPUT_PATH), dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {OUTPUT_PATH}")
    import os
    print(f"File exists right after save: {os.path.exists(OUTPUT_PATH)}")
    print(f"File size: {os.path.getsize(OUTPUT_PATH)} bytes")
    import time; time.sleep(2)
    print(f"File exists after 2s: {os.path.exists(OUTPUT_PATH)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
