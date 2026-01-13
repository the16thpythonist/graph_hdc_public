#!/usr/bin/env python
"""
Train Real NVP V3 Normalizing Flow for Molecular Generation.

Usage:
    python train_flow.py --dataset qm9
    python train_flow.py --dataset zinc --epochs 500

    # With mixed precision (2-3x speedup on A100/H100):
    python train_flow.py --dataset qm9 --precision bf16-mixed
"""
import argparse
import datetime
import os
import random
import string
import tempfile
from pathlib import Path

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger
from torch_geometric.loader import DataLoader

from graph_hdc.datasets.utils import get_split, post_compute_encodings
from graph_hdc.hypernet.configs import get_config
from graph_hdc.hypernet.encoder import HyperNet
from graph_hdc.models.flows.real_nvp import (
    QM9_FLOW_CONFIG,
    ZINC_FLOW_CONFIG,
    FlowConfig,
    RealNVPV3Lightning,
)

# Fix for PyTorch Lightning's _atomic_save which uses tmpfs (limited quota).
# Set TMPDIR to the current working directory to use disk storage instead.
_CUSTOM_TMPDIR = Path.cwd() / ".tmp_checkpoints"
_CUSTOM_TMPDIR.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(_CUSTOM_TMPDIR)
tempfile.tempdir = str(_CUSTOM_TMPDIR)  # Also update tempfile module directly


# Setup
torch.set_default_dtype(torch.float32)


def setup_experiment(name: str) -> dict[str, Path]:
    """Create experiment directory structure."""
    base_dir = Path(__file__).parent.parent / "results" / "train_flow"
    base_dir.mkdir(parents=True, exist_ok=True)

    slug = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{''.join(random.choices(string.ascii_lowercase, k=4))}"
    exp_dir = base_dir / f"{name}_{slug}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    dirs = {
        "exp_dir": exp_dir,
        "models_dir": exp_dir / "models",
        "logs_dir": exp_dir / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    return dirs


@torch.no_grad()
def fit_standardization(model: RealNVPV3Lightning, loader: DataLoader, device: torch.device):
    """Fit per-feature standardization on training data."""
    D = model.D
    hv_count = model.hv_count
    flat_dim = hv_count * D

    cnt = 0
    sum_vec = torch.zeros(flat_dim, device=device)
    sumsq_vec = torch.zeros(flat_dim, device=device)

    for batch in loader:
        batch = batch.to(device)
        x = model._flat_from_batch(batch)
        cnt += x.shape[0]
        sum_vec += x.sum(dim=0)
        sumsq_vec += (x * x).sum(dim=0)

    mu = sum_vec / cnt
    var = (sumsq_vec / cnt - mu**2).clamp_min_(0)
    sigma = var.sqrt().clamp_min_(1e-6)

    model.set_standardization(mu, sigma)
    model._per_term_split = D  # Enable per-term standardization


def main():
    parser = argparse.ArgumentParser(
        description="Train Real NVP V3 flow model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset selection
    parser.add_argument("--dataset", type=str, default="qm9", choices=["qm9", "zinc"],
                        help="Dataset to train on")

    # Model architecture (defaults from best configs when None)
    parser.add_argument("--num_flows", type=int, default=None,
                        help="Number of coupling layers (QM9: 16, ZINC: 8)")
    parser.add_argument("--hidden_dim", type=int, default=None,
                        help="Hidden dimension for coupling networks (QM9: 1792, ZINC: 1536)")
    parser.add_argument("--num_hidden_layers", type=int, default=None,
                        help="Number of hidden layers in coupling networks (QM9: 3, ZINC: 2)")

    # Scale warmup
    parser.add_argument("--smax_initial", type=float, default=None,
                        help="Initial scale bound (QM9: 2.2, ZINC: 2.5)")
    parser.add_argument("--smax_final", type=float, default=None,
                        help="Final scale bound (QM9: 6.5, ZINC: 7.0)")
    parser.add_argument("--smax_warmup_epochs", type=int, default=None,
                        help="Epochs for scale warmup (QM9: 16, ZINC: 17)")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=None,
                        help="Training epochs (default: 800)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size (QM9: 96, ZINC: 224)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (QM9: 1.91e-4, ZINC: 5.39e-4)")
    parser.add_argument("--weight_decay", type=float, default=None,
                        help="Weight decay (QM9: 3.49e-4, ZINC: 1e-3)")

    # Other options
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--precision", type=str, default="32", choices=["32", "64", "bf16-mixed"],
                        help="Training precision")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cpu, cuda)")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    args = parser.parse_args()

    # Set seed
    seed_everything(args.seed, workers=True)

    # Select base config (copy to avoid modifying global)
    if args.dataset == "qm9":
        base_cfg = QM9_FLOW_CONFIG
        ds_config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
    else:
        base_cfg = ZINC_FLOW_CONFIG
        ds_config = get_config("ZINC_SMILES_HRR_256_F64_5G1NG4")

    # Create a new config with CLI overrides
    cfg = FlowConfig(
        hv_dim=base_cfg.hv_dim,
        num_flows=args.num_flows if args.num_flows is not None else base_cfg.num_flows,
        hidden_dim=args.hidden_dim if args.hidden_dim is not None else base_cfg.hidden_dim,
        num_hidden_layers=args.num_hidden_layers if args.num_hidden_layers is not None else base_cfg.num_hidden_layers,
        smax_initial=args.smax_initial if args.smax_initial is not None else base_cfg.smax_initial,
        smax_final=args.smax_final if args.smax_final is not None else base_cfg.smax_final,
        smax_warmup_epochs=args.smax_warmup_epochs if args.smax_warmup_epochs is not None else base_cfg.smax_warmup_epochs,
        epochs=args.epochs if args.epochs is not None else base_cfg.epochs,
        batch_size=args.batch_size if args.batch_size is not None else base_cfg.batch_size,
        lr=args.lr if args.lr is not None else base_cfg.lr,
        weight_decay=args.weight_decay if args.weight_decay is not None else base_cfg.weight_decay,
        use_act_norm=base_cfg.use_act_norm,
        per_term_standardization=base_cfg.per_term_standardization,
        seed=args.seed,
    )

    print(f"\n{'='*60}")
    print(f"Training Real NVP V3 on {args.dataset.upper()}")
    print(f"{'='*60}")
    print(f"Config: {cfg}")
    print(f"{'='*60}\n")

    # Setup experiment
    dirs = setup_experiment(f"{args.dataset}_nvp")

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load datasets
    print("Loading datasets...")
    train_ds = get_split("train", dataset=args.dataset)
    valid_ds = get_split("valid", dataset=args.dataset)

    print(f"Train: {len(train_ds)} samples")
    print(f"Valid: {len(valid_ds)} samples")

    # Create or load HyperNet
    print("\nInitializing HyperNet encoder...")
    ds_config.device = str(device)
    ds_config.dtype = "float32"  # Match flow model precision
    hypernet = HyperNet(ds_config)
    hypernet.eval()

    # Compute HDC encodings
    print("Computing HDC encodings...")
    train_encoded = post_compute_encodings(train_ds, hypernet, device=device)
    valid_encoded = post_compute_encodings(valid_ds, hypernet, device=device)

    # Create data loaders from encoded data
    train_loader = DataLoader(train_encoded, batch_size=cfg.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_encoded, batch_size=cfg.batch_size, shuffle=False, num_workers=args.num_workers)

    # Create model
    print("\nCreating Real NVP V3 model...")
    model = RealNVPV3Lightning(cfg)

    # Fit standardization
    print("Fitting standardization...")
    fit_standardization(model, train_loader, device)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=dirs["models_dir"],
            filename="best-{epoch:03d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        EarlyStopping(monitor="val_loss", patience=50, mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # Logger
    logger = CSVLogger(dirs["logs_dir"], name="train")

    # Trainer
    trainer = Trainer(
        max_epochs=cfg.epochs,
        accelerator="auto" if args.device == "auto" else args.device.split(":")[0],
        devices=1,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        enable_progress_bar=True,
    )

    # Train
    print("\nStarting training...")
    trainer.fit(model, train_loader, valid_loader)

    # Save final checkpoint
    final_path = dirs["models_dir"] / "final.ckpt"
    trainer.save_checkpoint(final_path)
    print(f"\nTraining complete! Model saved to: {final_path}")

    # Verification step: reload best model and generate samples
    print("\n" + "=" * 60)
    print("VERIFICATION: Loading best model and generating samples...")
    print("=" * 60)

    # Find the best checkpoint
    best_ckpt = list(dirs["models_dir"].glob("best-*.ckpt"))
    if best_ckpt:
        best_path = best_ckpt[0]
        print(f"Loading best checkpoint: {best_path}")

        # Load using the PyTorch Lightning method
        loaded_model = RealNVPV3Lightning.load_from_checkpoint(str(best_path))
        loaded_model.eval()
        loaded_model.to(device)

        # Generate 10 samples
        with torch.no_grad():
            samples = loaded_model.sample_split(10)
            edge_terms = samples["edge_terms"]
            graph_terms = samples["graph_terms"]

            # Check for NaN/Inf
            edge_ok = torch.isfinite(edge_terms).all()
            graph_ok = torch.isfinite(graph_terms).all()

            print("Generated 10 samples:")
            print(f"  - edge_terms shape: {edge_terms.shape}")
            print(f"  - graph_terms shape: {graph_terms.shape}")
            print(f"  - edge_terms finite: {edge_ok}")
            print(f"  - graph_terms finite: {graph_ok}")

            if edge_ok and graph_ok:
                print("VERIFICATION PASSED: Model loads and generates valid samples.")
            else:
                print("WARNING: Generated samples contain NaN/Inf values!")
    else:
        print("WARNING: No best checkpoint found for verification.")

    print("=" * 60)


if __name__ == "__main__":
    main()
