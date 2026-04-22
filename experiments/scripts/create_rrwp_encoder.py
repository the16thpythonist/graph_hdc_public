#!/usr/bin/env python
"""
Create and export a HyperNet encoder with RRWP features.

Scans the dataset for observed node features (with RW augmentation),
builds the HyperNet, and saves it as a checkpoint that can be loaded
by ``load_hypernet()``.

Usage:
    # Default: ZINC, dim=1024, depth=3, k=(6,10,14), bins=8
    python experiments/scripts/create_rrwp_encoder.py

    # Custom config
    python experiments/scripts/create_rrwp_encoder.py \
        --dataset zinc --hv_dim 512 --depth 3 \
        --k_values 4 8 12 --num_bins 8 \
        --output encoders/my_encoder.ckpt
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from graph_hdc.datasets.utils import scan_node_features_with_rw
from graph_hdc.hypernet import HyperNet, RWConfig, create_config_with_rw


def main():
    parser = argparse.ArgumentParser(
        description="Create and export a HyperNet encoder with RRWP features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, default="zinc",
                        choices=["qm9", "zinc", "pubchem16", "pubchem32", "pubchem64"])
    parser.add_argument("--hv_dim", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--k_values", type=int, nargs="+", default=[6, 10, 14])
    parser.add_argument("--num_bins", type=int, default=8)
    parser.add_argument("--no_prune", action="store_true",
                        help="Disable codebook pruning")
    parser.add_argument("--output", type=str, default=None,
                        help="Output checkpoint path (default: encoders/<auto>.ckpt)")
    parser.add_argument("--feature_cache", type=str, default=None,
                        help="Path to cache observed features pickle")
    args = parser.parse_args()

    rw_config = RWConfig(
        enabled=True,
        k_values=tuple(args.k_values),
        num_bins=args.num_bins,
    )

    k_str = "_".join(str(k) for k in args.k_values)
    name = f"{args.dataset}_d{args.hv_dim}_depth{args.depth}_k{k_str}_b{args.num_bins}"

    # Output path
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path("encoders") / f"{name}.ckpt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Feature cache path
    if args.feature_cache:
        cache_path = Path(args.feature_cache)
    else:
        cache_path = out_path.parent / f"{name}_features.pkl"

    # Scan observed features (or load from cache)
    if cache_path.exists():
        print(f"Loading cached features from {cache_path}")
        with open(cache_path, "rb") as f:
            observed_features = pickle.load(f)
    else:
        print(f"Scanning {args.dataset} for observed node features with RW augmentation...")
        print(f"  k_values={rw_config.k_values}, num_bins={rw_config.num_bins}")
        observed_features = scan_node_features_with_rw(args.dataset, rw_config)
        with open(cache_path, "wb") as f:
            pickle.dump(observed_features, f)
        print(f"  Cached {len(observed_features)} feature tuples to {cache_path}")

    print(f"Observed {len(observed_features)} unique feature tuples")

    # Create config and HyperNet
    print(f"Creating HyperNet: dim={args.hv_dim}, depth={args.depth}, "
          f"k={args.k_values}, bins={args.num_bins}, prune={not args.no_prune}")
    config = create_config_with_rw(
        base_dataset=args.dataset,
        hv_dim=args.hv_dim,
        rw_config=rw_config,
        hypernet_depth=args.depth,
        prune_codebook=not args.no_prune,
    )
    config.device = "cpu"
    config.dtype = "float32"

    hypernet = HyperNet(config, observed_node_features=observed_features)
    hypernet.eval()

    # Save
    hypernet.save(out_path)
    print(f"Saved encoder to {out_path}")
    print(f"  hv_dim={hypernet.hv_dim}, depth={hypernet.depth}")
    print(f"  codebook shape: {hypernet.nodes_codebook.shape}")


if __name__ == "__main__":
    main()
