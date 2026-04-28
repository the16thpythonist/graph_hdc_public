#!/usr/bin/env python
"""
Train Flow Matching on autoencoder latent space.

Child experiment of ``train_flow_matching.py`` that trains a flow matching
model in the latent space of a pre-trained :class:`HDCAutoencoder` instead
of directly on HDC vectors.

**Pipeline**:

1. Encode molecules through HyperNet → ``[edge_terms | graph_terms]``
2. Encode HDC vectors through pre-trained AE → latent ``z``
3. Train flow matching on ``z`` (typically 256-dim vs 2048-dim HDC)
4. At generation time: sample ``z`` → AE decode → HDC → HyperNet decode → molecule

The lower-dimensional, smoother latent space should be easier for flow
matching to learn compared to the peaked, high-dimensional HDC distribution.

Usage:
    python experiments/generation/train_flow_matching__ae.py \\
        --ENCODER_PATH encoders/zinc_d1024_depth3_k6_10_14_b8.ckpt \\
        --AE_PATH results/train_autoencoder/debug/last.ckpt
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import torch
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from graph_hdc.hypernet import load_hypernet
from graph_hdc.hypernet.encoder import HyperNet
from graph_hdc.models.autoencoder import HDCAutoencoder
from graph_hdc.models.flow_matching import MultiCondition

# Module-level storage for non-serializable objects shared between hooks.
# PyComex's e["key"] data storage serializes to JSON, so PyTorch models
# cannot be stored there.
_shared: dict = {}


def _encoder_config_hash_from_ckpt(encoder_path: str) -> str:
    """Compute a short deterministic hash from an encoder checkpoint's config."""
    state = torch.load(encoder_path, map_location="cpu", weights_only=False)
    config = state["config"]
    raw = json.dumps(config, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

# =============================================================================
# PARAMETER OVERRIDES
# =============================================================================

# :param AE_PATH:
#       Path to a pre-trained HDCAutoencoder checkpoint (saved via
#       ``HDCAutoencoder.save()``).  The AE must have been trained with
#       ``training_target="both"`` so it reconstructs full
#       ``[edge_terms | graph_terms]`` vectors.
AE_PATH: str = "/media/ssd2/Programming/_branch/graph_hdc_public/experiments/generation/results/train_autoencoder/23_04_2026__04_29__tyCM/autoencoder.ckpt"

# :param ENCODER_PATH:
#       Path to HyperNet encoder checkpoint (same as base).
ENCODER_PATH: str = "/media/ssd2/Programming/_branch/graph_hdc_public/experiments/encoders/zinc_d1024_depth3_k6_10_14_b8.ckpt"

# Same as edge_graph: operate on the full [edge_terms | graph_terms] space,
# but compressed through the AE.  Latent vectors are split across
# node_terms / graph_terms so _extract_vectors(VECTOR_PART="both") restores
# the full latent via concatenation.
VECTOR_PART: str = "both"

# AE latents are continuous (not discrete HDC vectors), so target noise
# annealing is not needed by default.
TARGET_NOISE_SIGMA: float = 0.0

# Enable periodic molecule generation during training.  The
# ``transform_flow_samples`` hook decodes latent samples back to HDC space
# before the callback's standard HyperNet decoding.
GEN_EVAL_EVERY_N_EPOCHS: int = 10

# Architecture — appropriate for ~256-dim latent (vs 2048-dim HDC).
HIDDEN_DIM: int = 1024
NUM_BLOCKS: int = 5
BATCH_SIZE: int = 256

# Standard MSE loss with velocity prediction.  Velocity fits the smooth,
# approximately-Gaussian AE latent better than x_prediction: x_prediction +
# MSE regresses to the conditional mean near t=0 (shrinkage), producing
# under-dispersed samples and drifting per-dim means — the symptoms we
# observed on earlier runs.
LOSS_TYPE: str = "gm_nll"
PREDICTION_TYPE: str = "velocity"
DROPOUT: float = 0.02

# =============================================================================
# EXPERIMENT
# =============================================================================

experiment = Experiment.extend(
    "train_flow_matching.py",
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


# =============================================================================
# HOOKS
# =============================================================================


@experiment.hook("encoder_config_hash", default=True)
def encoder_config_hash(e: Experiment):
    """Return a short hash of the encoder config for cache scoping."""
    if e.ENCODER_PATH and Path(e.ENCODER_PATH).exists():
        return _encoder_config_hash_from_ckpt(e.ENCODER_PATH)
    return "testing"


@experiment.hook("scan_observed_edges", default=True)
@experiment.cache.cached(
    name="observed_edges",
    scope=lambda _e: (
        "observed_edges",
        _e.DATASET.lower(),
        _e.apply_hook("encoder_config_hash"),
    ),
)
def scan_observed_edges(e: Experiment, hypernet: HyperNet):
    """Scan dataset for observed edge pairs (cached)."""
    needs_rw = hasattr(hypernet, "rw_config") and hypernet.rw_config.enabled
    if needs_rw:
        from graph_hdc.datasets.utils import scan_features_with_rw

        e.log("Scanning dataset for observed edge pairs (with RW augmentation)...")
        max_samples = 200 if e.__TESTING__ else None
        _, edges = scan_features_with_rw(
            e.DATASET.lower(), hypernet.rw_config, max_samples=max_samples,
        )
        return edges
    else:
        from graph_hdc.datasets.utils import get_dataset_info

        e.log("Loading observed edge pairs from dataset info...")
        return get_dataset_info(e.DATASET.lower()).edge_features


@experiment.hook("load_encoder", default=False)
def load_encoder(e: Experiment, device: torch.device):
    """Load HyperNet encoder and pre-trained autoencoder.

    The HyperNet is loaded and pruned using cached edge observations.
    The AE is loaded and stored on ``_shared["ae"]`` for use by downstream hooks.
    """
    if not e.AE_PATH or not Path(e.AE_PATH).exists():
        if e.__TESTING__:
            # Create a tiny AE for testing
            ae = HDCAutoencoder(
                data_dim=512,  # 2 * test hv_dim (256)
                latent_dim=32,
                trunk_dim=128,
                n_encoder_blocks=1,
                n_decoder_blocks=1,
                training_target="both",
            )
            ae.eval()
            _shared["ae"] = ae

            from graph_hdc.hypernet.configs import get_config

            config = get_config("ZINC_SMILES_HRR_256_F64_5G1NG4")
            config.device = str(device)
            config.dtype = "float32"
            hypernet = HyperNet(config)
            hypernet.eval()
            _shared["hypernet"] = hypernet
            return hypernet
        else:
            raise ValueError(
                "AE_PATH must be set to a valid HDCAutoencoder checkpoint. "
                "Use --__TESTING__ True for quick tests without a checkpoint."
            )

    # Load HyperNet
    if e.ENCODER_PATH and Path(e.ENCODER_PATH).exists():
        hypernet = load_hypernet(e.ENCODER_PATH, device=str(device))
    elif e.__TESTING__:
        from graph_hdc.hypernet.configs import get_config

        config = get_config("ZINC_SMILES_HRR_256_F64_5G1NG4")
        config.device = str(device)
        config.dtype = "float32"
        hypernet = HyperNet(config)
    else:
        raise ValueError(
            "ENCODER_PATH must be set to a valid HyperNet checkpoint path."
        )

    # Prune edges codebook (cached)
    enc_hash = e.apply_hook("encoder_config_hash")
    e.log(f"Encoder config hash: {enc_hash}")

    observed_edges = e.apply_hook("scan_observed_edges", hypernet=hypernet)
    hypernet.limit_edges_codebook(observed_edges)
    e.log(f"Pruned edges codebook: {hypernet.edges_codebook.shape[0]} entries")
    hypernet.eval()

    # Load pre-trained AE
    ae = HDCAutoencoder.load(str(e.AE_PATH), map_location=str(device))
    ae.eval()
    assert ae.training_target == "both", (
        f"AE training_target must be 'both', got {ae.training_target!r}. "
        f"The flow model needs full [edge_terms | graph_terms] reconstruction."
    )
    e.log(
        f"Loaded AE: latent_dim={ae.latent_dim}, data_dim={ae.data_dim}, "
        f"variational={ae.variational}"
    )

    _shared["ae"] = ae
    _shared["hypernet"] = hypernet
    return hypernet


@experiment.hook("override_data_dim", default=True)
def override_data_dim(e: Experiment, data_dim: int, hv_dim: int) -> int:
    """Set data_dim to the AE's latent dimensionality."""
    latent_dim = _shared["ae"].latent_dim
    e.log(
        f"Overriding data_dim: {data_dim} -> {latent_dim} (AE latent space)"
    )
    return latent_dim


@experiment.hook("load_and_encode_data", default=False)
@experiment.cache.cached(
    name="ae_encoded_data",
    scope=lambda _e: (
        "ae_encoded_data",
        _e.DATASET.lower(),
        _e.apply_hook("encoder_config_hash"),
        hashlib.sha256(_e.AE_PATH.encode()).hexdigest()[:16],
        f"n_{_e.NUM_SUBSAMPLE or 'all'}",
    ),
)
def load_and_encode_data(
    e: Experiment,
    hypernet: HyperNet,
    device: torch.device,
    conditioning: Optional[MultiCondition] = None,
):
    """Load data, encode through HyperNet, then project to AE latent space.

    Cached via ``@experiment.cache.cached`` to skip expensive recomputation
    on subsequent runs with the same encoder, AE, and dataset.

    After HyperNet encoding:
    1. Swap ``edge_terms`` into ``node_terms`` slot (same as __edge_graph).
    2. Concatenate ``[edge_terms | graph_terms]`` → AE encode → latent z.
    3. Split latent z across ``node_terms`` / ``graph_terms`` so that
       ``_extract_vectors(vector_part="both")`` restores the full latent
       via concatenation.
    """
    from graph_hdc.datasets.utils import get_split, post_compute_encodings

    e.log("Loading dataset...")
    train_ds = get_split("train", dataset=e.DATASET.lower())
    valid_ds = get_split("valid", dataset=e.DATASET.lower())

    if e.__TESTING__:
        train_ds = train_ds[:64]
        valid_ds = valid_ds[:16]
    elif e.NUM_SUBSAMPLE is not None:
        n = e.NUM_SUBSAMPLE
        n_val = max(1, n // 5)
        train_ds = train_ds[:n]
        valid_ds = valid_ds[:n_val]

    e.log(f"Train: {len(train_ds)}, Valid: {len(valid_ds)}")

    e.log("Computing HDC encodings...")
    train_encoded = post_compute_encodings(
        train_ds, hypernet, device=device, batch_size=e.ENCODER_BATCH_SIZE,
    )
    torch.cuda.empty_cache()
    valid_encoded = post_compute_encodings(
        valid_ds, hypernet, device=device, batch_size=e.ENCODER_BATCH_SIZE,
    )
    torch.cuda.empty_cache()
    e.log(f"Encoded: {len(train_encoded)} train, {len(valid_encoded)} valid")

    # Swap edge_terms -> node_terms (same as __edge_graph)
    for d in train_encoded + valid_encoded:
        d.node_terms = d.edge_terms

    # Encode all vectors through the pre-trained AE
    ae = _shared["ae"]
    ae_device = next(ae.parameters()).device
    e.log(f"Encoding to AE latent space (latent_dim={ae.latent_dim})...")

    batch_size = 256
    all_data = train_encoded + valid_encoded
    for start in range(0, len(all_data), batch_size):
        batch = all_data[start : start + batch_size]
        # Build [edge_terms | graph_terms] vectors
        vecs = torch.stack([
            torch.cat([d.node_terms.view(-1), d.graph_terms.view(-1)])
            for d in batch
        ]).float().to(ae_device)

        with torch.no_grad():
            enc_out = ae.encode(vecs)
            z = enc_out["mu"] if ae.variational else enc_out["z"]

        z = z.cpu()
        half = z.shape[-1] // 2
        for i, d in enumerate(batch):
            # Split latent across node_terms / graph_terms so that
            # _extract_vectors("both") = cat(node_terms, graph_terms) = z.
            # .clone() is required: plain slices share storage with the batch
            # tensor, and pickle does not dedup storage across view objects —
            # each view would serialize the full batch storage, blowing the
            # cache file up by ~batch_size.
            d.node_terms = z[i, :half].clone()
            d.graph_terms = z[i, half:].clone()

    latent_dim = all_data[0].node_terms.shape[-1] + all_data[0].graph_terms.shape[-1]
    e.log(
        f"AE encoding complete. Latent dim={latent_dim} "
        f"(split: node_terms={all_data[0].node_terms.shape[-1]}, "
        f"graph_terms={all_data[0].graph_terms.shape[-1]})"
    )

    return train_encoded, valid_encoded


@experiment.hook("transform_flow_samples", default=True)
def transform_flow_samples(e: Experiment, samples: torch.Tensor) -> torch.Tensor:
    """Decode flow-generated latent vectors back to HDC space via the AE."""
    ae = _shared["ae"]
    ae_device = next(ae.parameters()).device
    with torch.no_grad():
        hdc_vectors = ae.decode(samples.to(ae_device))
    return hdc_vectors.cpu()




experiment.run_if_main()
