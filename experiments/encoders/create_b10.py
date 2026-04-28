"""
One-off: create the b=10 sibling of zinc_d1024_depth3_k6_10_14_b8.zip.

Matches the exact settings of create_rrwp_encoder.py's ZINC defaults, with
only num_bins bumped from 8 to 10. Uniform binning on [0, 1] — same as
the existing b=8 export. Swap in get_zinc_rw_boundaries(10) below if you
want quantile binning.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from graph_hdc.datasets.utils import scan_node_features_with_rw
from graph_hdc.hypernet import HyperNet, RWConfig, create_config_with_rw


DATASET = "zinc"
HV_DIM = 1024
DEPTH = 3
K_VALUES = (6, 10, 14)
NUM_BINS = 10
PRUNE = True

OUT_DIR = Path(__file__).resolve().parent
NAME = f"{DATASET}_d{HV_DIM}_depth{DEPTH}_k{'_'.join(map(str, K_VALUES))}_b{NUM_BINS}"
CKPT_PATH = OUT_DIR / f"{NAME}.zip"
CACHE_PATH = OUT_DIR / f"{NAME}_features.pkl"


def main() -> None:
    rw_config = RWConfig(
        enabled=True,
        k_values=K_VALUES,
        num_bins=NUM_BINS,
    )

    if CACHE_PATH.exists():
        print(f"Loading cached features from {CACHE_PATH}")
        with open(CACHE_PATH, "rb") as f:
            observed_features = pickle.load(f)
    else:
        print(f"Scanning {DATASET} for observed node features "
              f"(k={K_VALUES}, num_bins={NUM_BINS})...")
        observed_features = scan_node_features_with_rw(DATASET, rw_config)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(observed_features, f)
        print(f"  Cached {len(observed_features)} feature tuples to {CACHE_PATH}")

    print(f"Observed {len(observed_features)} unique (base+RRWP) feature tuples")

    config = create_config_with_rw(
        base_dataset=DATASET,
        hv_dim=HV_DIM,
        rw_config=rw_config,
        hypernet_depth=DEPTH,
        prune_codebook=PRUNE,
    )
    config.device = "cpu"
    config.dtype = "float32"

    hypernet = HyperNet(config, observed_node_features=observed_features)
    hypernet.eval()

    hypernet.save(CKPT_PATH)
    print(f"Saved encoder to {CKPT_PATH}")
    print(f"  hv_dim={hypernet.hv_dim}, depth={hypernet.depth}")
    print(f"  codebook shape: {tuple(hypernet.nodes_codebook.shape)}")


if __name__ == "__main__":
    main()
