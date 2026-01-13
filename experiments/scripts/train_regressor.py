#!/usr/bin/env python
"""
Train Property Regressor for Molecular Properties.

Usage:
    python train_regressor.py --dataset qm9 --property logp
    python train_regressor.py --dataset zinc --property qed
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
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch_geometric.loader import DataLoader

from graph_hdc import (
    HyperNet,
    get_config,
    get_split,
    post_compute_encodings,
)
from graph_hdc.models.regressors import (PropertyRegressor, QM9_LOGP_CONFIG, QM9_QED_CONFIG, RegressorConfig,
                                         ZINC_LOGP_CONFIG, ZINC_QED_CONFIG)

# Fix for PyTorch Lightning's _atomic_save which uses tmpfs (limited quota).
# Set TMPDIR to the current working directory to use disk storage instead.
_CUSTOM_TMPDIR = Path.cwd() / ".tmp_checkpoints"
_CUSTOM_TMPDIR.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(_CUSTOM_TMPDIR)
tempfile.tempdir = str(_CUSTOM_TMPDIR)

# Setup
torch.set_default_dtype(torch.float32)


# Config mapping
CONFIGS = {
    ("qm9", "logp"): QM9_LOGP_CONFIG,
    ("qm9", "qed"): QM9_QED_CONFIG,
    ("zinc", "logp"): ZINC_LOGP_CONFIG,
    ("zinc", "qed"): ZINC_QED_CONFIG,
}


def setup_experiment(name: str) -> dict[str, Path]:
    """Create experiment directory structure."""
    base_dir = Path(__file__).parent.parent / "results" / "train_regressor"
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


def main():
    parser = argparse.ArgumentParser(
        description="Train property regressor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset and property selection
    parser.add_argument("--dataset", type=str, default="qm9", choices=["qm9", "zinc"],
                        help="Dataset to train on")
    parser.add_argument("--property", type=str, default="logp", choices=["logp", "qed"],
                        help="Target property to predict")

    # Model architecture (defaults from best configs when None)
    parser.add_argument("--hidden_dims", type=str, default=None,
                        help="Hidden layer dimensions as comma-separated values (e.g., '256,256')")
    parser.add_argument("--activation", type=str, default=None,
                        choices=["relu", "gelu", "silu", "tanh", "leaky_relu"],
                        help="Activation function")
    parser.add_argument("--norm", type=str, default=None,
                        choices=["none", "batch_norm", "lay_norm"],
                        help="Normalization type")
    parser.add_argument("--dropout", type=float, default=None,
                        help="Dropout rate")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=None,
                        help="Training epochs (default: 200)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=None,
                        help="Weight decay")

    # Other options
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--precision", type=str, default="32", choices=["32", "64", "bf16-mixed"],
                        help="Training precision")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cpu, cuda)")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    args = parser.parse_args()

    # Set seed
    seed_everything(args.seed, workers=True)

    # Get base config (best hyperparameters)
    cfg_key = (args.dataset, args.property)
    if cfg_key in CONFIGS:
        base_cfg = CONFIGS[cfg_key]
    else:
        # Default config
        base_cfg = RegressorConfig(
            input_dim=256,
            hidden_dims=(256, 256),
            target_property=args.property,
        )

    # Parse hidden_dims from CLI if provided
    if args.hidden_dims is not None:
        hidden_dims = tuple(int(x.strip()) for x in args.hidden_dims.split(","))
    else:
        hidden_dims = base_cfg.hidden_dims

    # Create new config with CLI overrides
    cfg = RegressorConfig(
        input_dim=base_cfg.input_dim,
        hidden_dims=hidden_dims,
        activation=args.activation if args.activation is not None else base_cfg.activation,
        norm=args.norm if args.norm is not None else base_cfg.norm,
        dropout=args.dropout if args.dropout is not None else base_cfg.dropout,
        lr=args.lr if args.lr is not None else base_cfg.lr,
        weight_decay=args.weight_decay if args.weight_decay is not None else base_cfg.weight_decay,
        batch_size=args.batch_size if args.batch_size is not None else base_cfg.batch_size,
        epochs=args.epochs if args.epochs is not None else base_cfg.epochs,
        target_property=args.property,
    )

    # Get dataset config
    if args.dataset == "qm9":
        ds_config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
    else:
        ds_config = get_config("ZINC_SMILES_HRR_256_F64_5G1NG4")

    print(f"\n{'='*60}")
    print(f"Training {args.property.upper()} Regressor on {args.dataset.upper()}")
    print(f"{'='*60}")
    print(f"Config: {cfg}")
    print(f"{'='*60}\n")

    # Setup experiment
    dirs = setup_experiment(f"{args.dataset}_{args.property}_regressor")

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load datasets
    print("Loading datasets...")
    train_ds = get_split("train", dataset=args.dataset)
    valid_ds = get_split("valid", dataset=args.dataset)
    test_ds = get_split("test", dataset=args.dataset)

    print(f"Train: {len(train_ds)} samples")
    print(f"Valid: {len(valid_ds)} samples")
    print(f"Test: {len(test_ds)} samples")

    # Create or load HyperNet
    print("\nInitializing HyperNet encoder...")
    ds_config.device = str(device)
    ds_config.dtype = "float32"  # Match regressor precision
    hypernet = HyperNet(ds_config)
    hypernet.eval()

    # Compute HDC encodings
    print("Computing HDC encodings...")
    train_encoded = post_compute_encodings(train_ds, hypernet, device=device)
    valid_encoded = post_compute_encodings(valid_ds, hypernet, device=device)
    test_encoded = post_compute_encodings(test_ds, hypernet, device=device)

    # Create data loaders using config batch size
    train_loader = DataLoader(train_encoded, batch_size=cfg.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_encoded, batch_size=cfg.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_encoded, batch_size=cfg.batch_size, shuffle=False, num_workers=args.num_workers)

    # Create model
    print("\nCreating PropertyRegressor model...")
    model = PropertyRegressor(
        input_dim=cfg.input_dim,
        hidden_dims=cfg.hidden_dims,
        activation=cfg.activation,
        norm=cfg.norm,
        dropout=cfg.dropout,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        target_property=cfg.target_property,
    )

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
        EarlyStopping(monitor="val_loss", patience=30, mode="min"),
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
        enable_progress_bar=True,
    )

    # Train
    print("\nStarting training...")
    trainer.fit(model, train_loader, valid_loader)

    # Test
    print("\nEvaluating on test set...")
    trainer.test(model, test_loader)

    # Save final checkpoint
    final_path = dirs["models_dir"] / "final.ckpt"
    trainer.save_checkpoint(final_path)
    print(f"\nTraining complete! Model saved to: {final_path}")


if __name__ == "__main__":
    main()
