#!/usr/bin/env python
"""
Aggregate-posterior sampling from an HDC-VAE's latent space.

Instead of sampling from the prior ``N(0, I)`` (which for VAEs trained with
small KL weight is mostly out-of-distribution for the decoder), this script:

  1. Encodes a batch of training molecules through HyperNet + VAE encoder,
  2. Collects the per-dimension ``μ`` values,
  3. Fits a diagonal Gaussian ``N(μ_mean, (T · μ_std)²)`` to them,
  4. Samples ``N_SAMPLES`` latent vectors from that fitted Gaussian,
  5. Decodes each back to HDC space and reconstructs molecules.

``TEMPERATURE`` scales the fitted std; ``T=1.0`` reproduces the empirical
aggregate posterior, ``T<1`` stays closer to the mean.

Usage:
    python experiments/generation/latent_sampling.py
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
OUTPUT_PATH = REPO / "experiments/generation/results/latent_sampling.png"

N_SAMPLES = 10            # plotted in a 2x5 grid
N_FIT_MOLECULES = 500     # training molecules used to fit the aggregate posterior
TEMPERATURE = 1.0         # scales fitted per-dim std; <1 stays closer to the mean
SEED = None               # set to an int for reproducibility


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
    assert model.variational, (
        f"Checkpoint at {AE_CHECKPOINT} is not a VAE (model.variational=False); "
        "prior sampling is only meaningful for variational models."
    )
    latent_dim = model.hparams.latent_dim
    print(
        f"  data_dim={model.hparams.data_dim}, latent_dim={latent_dim}, "
        f"variational=True"
    )

    # ── Fit aggregate posterior on a batch of training molecules ─────
    print(f"Loading ZINC train split and encoding {N_FIT_MOLECULES} molecules...")
    train_ds = get_split("train", dataset="zinc")
    n_fit = min(N_FIT_MOLECULES, len(train_ds))
    subset = train_ds[:n_fit]
    encoded = post_compute_encodings(subset, hypernet, batch_size=64, device=device)
    # Same slot swap as the interpolation script
    for d in encoded:
        d.node_terms = d.edge_terms

    hv_vectors = torch.stack(
        [
            torch.cat([d.node_terms.view(-1), d.graph_terms.view(-1)]).float()
            for d in encoded
        ]
    )  # [n_fit, data_dim]

    with torch.no_grad():
        enc_out = model.encode(hv_vectors)
        mus = enc_out["mu"]  # [n_fit, latent_dim]

    mu_mean = mus.mean(dim=0)                        # [latent_dim]
    mu_std = mus.std(dim=0, unbiased=False).clamp_min(1e-6)
    print(
        f"  fitted aggregate posterior: "
        f"||μ_mean||={mu_mean.norm().item():.3f}, "
        f"mean(std)={mu_std.mean().item():.3f}, "
        f"max(std)={mu_std.max().item():.3f}"
    )

    # ── Sample from fitted aggregate posterior ───────────────────────
    print(f"Sampling {N_SAMPLES} latents with temperature={TEMPERATURE}...")
    eps = torch.randn(N_SAMPLES, latent_dim, device=device)
    z = mu_mean + TEMPERATURE * mu_std * eps

    with torch.no_grad():
        recon_vectors = model.decode(z)  # [N_SAMPLES, data_dim]

    # ── Decode HDC vectors → molecules ───────────────────────────────
    vsa_cls = hypernet.vsa.tensor_class
    decoder_settings = FallbackDecoderSettings(
        beam_size=8,
        limit=1024,
        top_k=1,
    )

    mols: list[Chem.Mol | None] = []
    smiles_list: list[str] = []

    for i in range(N_SAMPLES):
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
            print(f"  Sample {i}: decode failed – {e}")
            mol = None

        smi = Chem.MolToSmiles(mol) if mol is not None else "INVALID"
        smiles_list.append(smi)
        mols.append(mol)
        print(f"  sample {i:2d}: {smi}")

    # ── Plot (2x5 grid) ──────────────────────────────────────────────
    n_rows, n_cols = 2, 5
    assert n_rows * n_cols == N_SAMPLES, "grid size must match N_SAMPLES"

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()

    for i, (ax, mol, smi) in enumerate(zip(axes, mols, smiles_list)):
        ax.set_xticks([])
        ax.set_yticks([])
        if mol is not None:
            img = Draw.MolToImage(mol, size=(300, 300))
            ax.imshow(img)
        else:
            ax.text(
                0.5, 0.5, "INVALID", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="red",
            )
        ax.set_title(f"#{i}\n{smi}", fontsize=8, wrap=True)

    fig.suptitle(
        f"HDC-VAE Aggregate-Posterior Samples "
        f"(T={TEMPERATURE}, n_fit={n_fit}, latent_dim={latent_dim})",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(str(OUTPUT_PATH), dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
