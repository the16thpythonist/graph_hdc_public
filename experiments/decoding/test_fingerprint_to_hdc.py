#!/usr/bin/env python
"""
Evaluate a pre-trained Fingerprint-to-HDC TranslatorMLP.

Loads a trained TranslatorMLP checkpoint and evaluates end-to-end molecular
reconstruction on a held-out test set (exported by train_fingerprint_to_hdc.py
as ``test_smiles.csv``).

Pipeline per molecule:
    SMILES → ECFP4 → TranslatorMLP → predicted HDC
        → decode_nodes_from_hdc → FlowEdgeDecoder.sample_best_of_n() → graph
        → compare with original molecule

Usage:
    # Quick smoke test
    python test_fingerprint_to_hdc.py --__TESTING__ True

    # Full evaluation
    python test_fingerprint_to_hdc.py \\
        --TEST_CSV_PATH /path/to/test_smiles.csv \\
        --TRANSLATOR_CHECKPOINT_PATH /path/to/best.ckpt \\
        --HYPERNET_PATH /path/to/encoder.ckpt \\
        --FLOW_EDGE_DECODER_PATH /path/to/decoder.ckpt
"""

from __future__ import annotations

import csv
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from torch_geometric.data import Data

from graph_hdc.datasets.zinc_smiles import mol_to_data as mol_to_zinc_data
from graph_hdc.hypernet import load_hypernet
from graph_hdc.hypernet.multi_hypernet import MultiHyperNet
from graph_hdc.models.flow_edge_decoder import (
    FlowEdgeDecoder,
    get_node_feature_bins,
    node_tuples_to_onehot,
    preprocess_for_flow_edge_decoder,
)
from graph_hdc.utils.experiment_helpers import (
    compute_hdc_distance,
    compute_tanimoto_similarity,
    create_reconstruction_plot,
    decode_nodes_from_hdc,
    get_canonical_smiles,
    is_valid_mol,
    pyg_to_mol,
)

# =============================================================================
# PARAMETERS
# =============================================================================

# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------

# :param TEST_CSV_PATH:
#     Path to a CSV file with a ``smiles`` column containing cleaned test
#     SMILES.  Typically exported by train_fingerprint_to_hdc.py as
#     ``test_smiles.csv`` in the training experiment archive.
TEST_CSV_PATH: str = ""

# :param TRANSLATOR_CHECKPOINT_PATH:
#     Path to a trained TranslatorMLP Lightning checkpoint (.ckpt).
TRANSLATOR_CHECKPOINT_PATH: str = ""

# :param HYPERNET_PATH:
#     Path to the HyperNet / MultiHyperNet / RRWPHyperNet checkpoint used
#     during training.  Required for computing ground-truth HDC vectors and
#     for node decoding.
HYPERNET_PATH: str = ""

# :param FLOW_EDGE_DECODER_PATH:
#     Path to a pre-trained FlowEdgeDecoder checkpoint (.ckpt).
FLOW_EDGE_DECODER_PATH: str = ""

# -----------------------------------------------------------------------------
# Fingerprint
# -----------------------------------------------------------------------------

# :param FP_RADIUS:
#     Morgan fingerprint radius.  Must match the training configuration.
FP_RADIUS: int = 2

# :param FP_NBITS:
#     Number of bits in the Morgan fingerprint.  Must match training.
FP_NBITS: int = 2048

# -----------------------------------------------------------------------------
# Sampling
# -----------------------------------------------------------------------------

# :param SAMPLE_STEPS:
#     Number of denoising steps for FlowEdgeDecoder sampling.
SAMPLE_STEPS: int = 50

# :param ETA:
#     Stochasticity parameter for FlowEdgeDecoder sampling.
ETA: float = 0.0

# :param SAMPLE_TIME_DISTORTION:
#     Time distortion during sampling.  Options: "identity", "polydec".
SAMPLE_TIME_DISTORTION: str = "polydec"

# :param NUM_REPETITIONS:
#     Best-of-N repetitions per molecule during FlowEdgeDecoder sampling.
NUM_REPETITIONS: int = 128

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

# :param NUM_TEST_SAMPLES:
#     Maximum number of test molecules to evaluate.  Set to 0 to use all.
NUM_TEST_SAMPLES: int = 0

# -----------------------------------------------------------------------------
# System
# -----------------------------------------------------------------------------

# :param DEVICE:
#     Device for all models.  Options: "cpu", "cuda", "cuda:0", etc.
DEVICE: str = "cuda"

# :param SEED:
#     Random seed for reproducibility.
SEED: int = 42

# :param __DEBUG__:
#     Debug mode — reuses the same output folder during development.
__DEBUG__: bool = True

# :param __TESTING__:
#     Testing mode — runs with minimal iterations for quick validation.
__TESTING__: bool = False


# =============================================================================
# HELPERS
# =============================================================================


def clean_mol(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """
    Clean a molecule: remove stereo, radicals, charges, explicit Hs.

    Mirrors the cleaning in train_fingerprint_to_hdc.py.
    """
    try:
        mol = Chem.RWMol(mol)
        Chem.RemoveStereochemistry(mol)

        for atom in mol.GetAtoms():
            if atom.GetNumRadicalElectrons() > 0:
                atom.SetNumRadicalElectrons(0)
                atom.SetNoImplicit(False)

        for atom in mol.GetAtoms():
            charge = atom.GetFormalCharge()
            if charge > 0:
                hs = atom.GetNumExplicitHs()
                remove = min(charge, hs)
                atom.SetNumExplicitHs(hs - remove)
                atom.SetFormalCharge(charge - remove)
            elif charge < 0:
                atom.SetNumExplicitHs(atom.GetNumExplicitHs() + abs(charge))
                atom.SetFormalCharge(0)

        mol = mol.GetMol()
        mol = Chem.RemoveHs(mol)
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def smiles_to_fingerprint(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> Optional[np.ndarray]:
    """Convert a SMILES string to a Morgan fingerprint numpy array."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp = gen.GetFingerprint(mol)
        arr = np.zeros(n_bits, dtype=np.float32)
        from rdkit import DataStructs
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return None


# =============================================================================
# EXPERIMENT
# =============================================================================


@Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)
def experiment(e: Experiment) -> None:
    """Evaluate a trained fingerprint-to-HDC translator on a test set."""

    # ── tmpdir fix for Lightning ──
    custom_tmpdir = Path(e.path) / ".tmp_checkpoints"
    custom_tmpdir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(custom_tmpdir)
    tempfile.tempdir = str(custom_tmpdir)

    pl.seed_everything(e.SEED)

    device = torch.device(e.DEVICE)

    e.log("=" * 60)
    e.log("Fingerprint → HDC — Test Evaluation")
    e.log("=" * 60)
    e.log(f"Test CSV:       {e.TEST_CSV_PATH}")
    e.log(f"Translator:     {e.TRANSLATOR_CHECKPOINT_PATH}")
    e.log(f"HyperNet:       {e.HYPERNET_PATH}")
    e.log(f"Decoder:        {e.FLOW_EDGE_DECODER_PATH}")
    e.log(f"Device:         {e.DEVICE}")
    e.log(f"Best-of-N:      {e.NUM_REPETITIONS} reps, {e.SAMPLE_STEPS} steps")
    e.log("=" * 60)

    # ── store config ──
    e["config/test_csv_path"] = e.TEST_CSV_PATH
    e["config/translator_checkpoint"] = e.TRANSLATOR_CHECKPOINT_PATH
    e["config/hypernet_path"] = e.HYPERNET_PATH
    e["config/decoder_path"] = e.FLOW_EDGE_DECODER_PATH
    e["config/device"] = e.DEVICE
    e["config/fp_radius"] = e.FP_RADIUS
    e["config/fp_nbits"] = e.FP_NBITS
    e["config/sample_steps"] = e.SAMPLE_STEPS
    e["config/num_repetitions"] = e.NUM_REPETITIONS

    # =====================================================================
    # Load Models
    # =====================================================================

    e.log("\nLoading HyperNet encoder...")
    hypernet = load_hypernet(e.HYPERNET_PATH, device="cpu")
    hypernet.eval()

    if isinstance(hypernet, MultiHyperNet):
        actual_hdc_dim = hypernet.hv_dim
        ensemble_graph_dim = hypernet.ensemble_graph_dim
    else:
        actual_hdc_dim = hypernet.hv_dim
        ensemble_graph_dim = actual_hdc_dim

    concat_hdc_dim = actual_hdc_dim + ensemble_graph_dim
    e.log(f"HyperNet loaded: hdc_dim={actual_hdc_dim}, concat_dim={concat_hdc_dim}")

    e.log("Loading TranslatorMLP...")
    model = e.apply_hook(
        "load_translator",
        checkpoint_path=e.TRANSLATOR_CHECKPOINT_PATH,
        device=device,
    )

    e.log("Loading FlowEdgeDecoder...")
    decoder = e.apply_hook(
        "load_decoder",
        checkpoint_path=e.FLOW_EDGE_DECODER_PATH,
        device=device,
    )

    # =====================================================================
    # Load Test Data (hook)
    # =====================================================================

    test_data = e.apply_hook(
        "load_test_data",
        hypernet=hypernet,
    )

    e["data/test_size"] = len(test_data)

    # =====================================================================
    # Evaluate (hook)
    # =====================================================================

    metrics = e.apply_hook(
        "evaluate",
        model=model,
        hypernet=hypernet,
        decoder=decoder,
        test_data=test_data,
        device=device,
        actual_hdc_dim=actual_hdc_dim,
    )

    e.log("\n" + "=" * 60)
    e.log("Evaluation completed!")
    e.log("=" * 60)


# =============================================================================
# HOOKS
# =============================================================================


@experiment.hook("load_translator", default=True)
def load_translator(
    e: Experiment,
    checkpoint_path: str,
    device: torch.device,
) -> pl.LightningModule:
    """
    Load a trained TranslatorMLP from a Lightning checkpoint.

    Override this hook to load a different model architecture.
    """
    if not checkpoint_path:
        raise ValueError("TRANSLATOR_CHECKPOINT_PATH is required.")

    from graph_hdc.models.translator_mlp import TranslatorMLP

    model = TranslatorMLP.load_from_checkpoint(checkpoint_path, map_location=device)
    model.to(device)
    model.eval()
    e.log(f"TranslatorMLP loaded ({sum(p.numel() for p in model.parameters()):,} params)")
    return model


@experiment.hook("load_decoder", default=True)
def load_decoder(
    e: Experiment,
    checkpoint_path: str,
    device: torch.device,
) -> FlowEdgeDecoder:
    """
    Load a pre-trained FlowEdgeDecoder.

    Override this hook to load a different decoder.
    """
    if not checkpoint_path:
        raise ValueError("FLOW_EDGE_DECODER_PATH is required.")

    decoder = FlowEdgeDecoder.load(checkpoint_path, device=device)
    decoder.eval()
    e.log("FlowEdgeDecoder loaded.")
    return decoder


@experiment.hook("load_test_data", default=True)
def load_test_data(
    e: Experiment,
    hypernet,
) -> List[Dict[str, Any]]:
    """
    Load test SMILES from CSV, compute fingerprints and HDC vectors.

    The CSV should contain already-cleaned SMILES (as exported by
    train_fingerprint_to_hdc.py).  No additional cleaning is applied.

    Override this hook to change data loading behaviour.

    Returns
    -------
    list[dict]
        Each dict has keys: smiles, fingerprint (Tensor), hdc_vector (Tensor)
    """
    csv_path = e.TEST_CSV_PATH
    if not csv_path:
        raise ValueError("TEST_CSV_PATH is required.")

    e.log(f"\nLoading test SMILES from {csv_path}...")
    smiles_list: list[str] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smi = row["smiles"].strip()
            if smi:
                smiles_list.append(smi)

    e.log(f"Read {len(smiles_list)} test SMILES")

    if e.__TESTING__:
        smiles_list = smiles_list[:10]
        e.log(f"TESTING mode: using {len(smiles_list)} molecules")

    # ── compute fingerprints + HDC vectors ──
    e.log("Computing fingerprints and HDC vectors...")
    records: list[dict] = []
    skipped = 0
    cpu_device = torch.device("cpu")

    for i, smi in enumerate(smiles_list):
        if (i + 1) % 1000 == 0:
            e.log(f"  processed {i + 1}/{len(smiles_list)}...")

        fp_arr = smiles_to_fingerprint(smi, radius=e.FP_RADIUS, n_bits=e.FP_NBITS)
        if fp_arr is None:
            skipped += 1
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            skipped += 1
            continue

        try:
            pyg_data = mol_to_zinc_data(mol)
        except Exception:
            skipped += 1
            continue

        processed = preprocess_for_flow_edge_decoder(pyg_data, hypernet, device=cpu_device)
        if processed is None:
            skipped += 1
            continue

        hdc_vec = processed.hdc_vector.squeeze(0)

        records.append({
            "smiles": Chem.MolToSmiles(mol, canonical=True),
            "fingerprint": torch.from_numpy(fp_arr),
            "hdc_vector": hdc_vec,
        })

    e.log(f"Successfully processed {len(records)} molecules (skipped {skipped})")

    if len(records) == 0:
        raise ValueError("No valid molecules after preprocessing!")

    return records


@experiment.hook("evaluate", default=True)
def evaluate(
    e: Experiment,
    model: pl.LightningModule,
    hypernet,
    decoder: FlowEdgeDecoder,
    test_data: List[Dict[str, Any]],
    device: torch.device,
    actual_hdc_dim: int,
) -> Dict[str, Any]:
    """
    End-to-end reconstruction evaluation.

    For each test molecule:
    1. Fingerprint → TranslatorMLP → predicted HDC
    2. decode_nodes_from_hdc → node tuples
    3. FlowEdgeDecoder.sample_best_of_n() → generated graph
    4. Compare with original molecule

    Override this hook to change evaluation behaviour.
    """
    e.log("\n" + "=" * 60)
    e.log("Reconstruction Evaluation")
    e.log("=" * 60)

    rw_config = getattr(hypernet, "rw_config", None)
    feature_bins = get_node_feature_bins(rw_config)

    num_samples = len(test_data)
    if e.NUM_TEST_SAMPLES > 0:
        num_samples = min(e.NUM_TEST_SAMPLES, num_samples)

    e.log(f"Evaluating {num_samples} test molecules...")

    model.to(device)
    model.eval()

    # ── phase 1: predict HDC + decode nodes ──
    e.log("Phase 1: predicting HDC vectors and decoding nodes...")
    sampling_start = time.time()
    prepared: list[dict] = []
    results: list[dict] = []
    node_decode_correct = 0

    for idx in range(num_samples):
        if (idx + 1) % 10 == 0 or idx == 0:
            e.log(f"  decoding nodes {idx + 1}/{num_samples}...")
        item = test_data[idx]
        original_smiles = item["smiles"]
        fp_tensor = item["fingerprint"].unsqueeze(0).to(device)

        original_mol = Chem.MolFromSmiles(original_smiles)
        if original_mol is None:
            results.append({
                "idx": idx,
                "original_smiles": original_smiles,
                "error": "Could not parse original SMILES",
            })
            continue

        with torch.no_grad():
            pred_hdc = model(fp_tensor).squeeze(0).cpu()

        try:
            node_tuples, num_nodes = decode_nodes_from_hdc(
                hypernet, pred_hdc.unsqueeze(0), actual_hdc_dim
            )
        except Exception as ex:
            results.append({
                "idx": idx,
                "original_smiles": original_smiles,
                "error": f"Node decode failed: {ex}",
            })
            continue

        if num_nodes == 0:
            results.append({
                "idx": idx,
                "original_smiles": original_smiles,
                "error": "No nodes decoded",
            })
            continue

        original_num_atoms = original_mol.GetNumAtoms()
        if num_nodes == original_num_atoms:
            node_decode_correct += 1

        node_features = node_tuples_to_onehot(
            node_tuples, device=device, feature_bins=feature_bins
        )

        prepared.append({
            "idx": idx,
            "original_smiles": original_smiles,
            "original_mol": original_mol,
            "pred_hdc": pred_hdc,
            "node_features": node_features,
            "num_nodes": num_nodes,
        })

    # ── phase 2: generate, evaluate, and plot per molecule ──
    num_reps = e.NUM_REPETITIONS
    e.log(f"Phase 2: generating edges for {len(prepared)} molecules "
          f"(best-of-{num_reps}, {e.SAMPLE_STEPS} steps)...")

    sample_kwargs = dict(
        sample_steps=e.SAMPLE_STEPS,
        eta=e.ETA,
        time_distortion=e.SAMPLE_TIME_DISTORTION,
        show_progress=False,
        device=device,
    )

    recon_dir = Path(e.path) / "reconstructions"
    recon_dir.mkdir(parents=True, exist_ok=True)

    valid_count = 0
    match_count = 0
    plot_count = 0
    tanimoto_scores: list[float] = []

    for i, item in enumerate(prepared):
        idx = item["idx"]
        original_smiles = item["original_smiles"]
        original_mol = item["original_mol"]

        hdc_vec = item["pred_hdc"].unsqueeze(0).to(device)
        n = item["num_nodes"]
        nf = item["node_features"].unsqueeze(0).to(device)
        mask = torch.ones(1, n, dtype=torch.bool, device=device)

        def score_fn(s, _orig=hdc_vec, _dim=actual_hdc_dim):
            return compute_hdc_distance(
                s, _orig, _dim, hypernet, device,
            )

        with torch.no_grad():
            best_sample, best_dist, avg_dist = decoder.sample_best_of_n(
                hdc_vectors=hdc_vec,
                node_features=nf,
                node_mask=mask,
                num_repetitions=num_reps,
                score_fn=score_fn,
                **sample_kwargs,
            )

        generated_mol = pyg_to_mol(best_sample)
        generated_smiles = get_canonical_smiles(generated_mol)
        is_valid = is_valid_mol(generated_mol)

        original_canonical = None
        if original_mol is not None:
            try:
                original_mol_no_h = Chem.RemoveAllHs(original_mol)
                original_canonical = Chem.MolToSmiles(original_mol_no_h, canonical=True)
            except Exception:
                original_canonical = get_canonical_smiles(original_mol)

        is_match = (
            is_valid
            and generated_smiles is not None
            and original_canonical is not None
            and generated_smiles == original_canonical
        )

        tanimoto = compute_tanimoto_similarity(original_mol, generated_mol)

        if is_valid:
            valid_count += 1
        if is_match:
            match_count += 1
        tanimoto_scores.append(tanimoto)

        status = "MATCH" if is_match else ("Valid" if is_valid else "Invalid")
        e.log(f"  * molecule {i + 1}/{len(prepared)} - {status} - "
              f"dist: {best_dist:.4f} - tanimoto: {tanimoto:.3f} - "
              f"{original_smiles} -> {generated_smiles or 'N/A'}")

        results.append({
            "idx": idx,
            "original_smiles": original_smiles,
            "generated_smiles": generated_smiles,
            "is_valid": is_valid,
            "is_match": is_match,
            "tanimoto": tanimoto,
            "hdc_distance": best_dist,
        })

        # ── plot ──
        plot_path = recon_dir / f"reconstruction_{plot_count + 1:03d}.png"
        create_reconstruction_plot(
            original_mol=original_mol,
            generated_mol=generated_mol,
            original_smiles=original_smiles,
            generated_smiles=generated_smiles or "N/A",
            is_valid=is_valid,
            is_match=is_match,
            sample_idx=idx,
            save_path=plot_path,
        )
        plot_count += 1

    if plot_count > 0:
        e.log(f"Saved {plot_count} reconstruction plots to {recon_dir}")

    # ── summary ──
    total_sampling_time = time.time() - sampling_start
    mean_tanimoto = float(np.mean(tanimoto_scores)) if tanimoto_scores else 0.0
    median_tanimoto = float(np.median(tanimoto_scores)) if tanimoto_scores else 0.0

    e.log("\n" + "-" * 40)
    e.log("Reconstruction Summary:")
    e.log(f"  Total samples:          {num_samples}")
    e.log(f"  Prepared (nodes ok):    {len(prepared)}")
    e.log(f"  Valid molecules:        {valid_count} ({100 * valid_count / max(num_samples, 1):.1f}%)")
    e.log(f"  Exact matches:          {match_count} ({100 * match_count / max(num_samples, 1):.1f}%)")
    e.log(f"  Node count accuracy:    {node_decode_correct}/{num_samples} "
          f"({100 * node_decode_correct / max(num_samples, 1):.1f}%)")
    e.log(f"  Mean Tanimoto:          {mean_tanimoto:.4f}")
    e.log(f"  Median Tanimoto:        {median_tanimoto:.4f}")
    e.log(f"  Sampling time:          {total_sampling_time:.2f}s")
    e.log("-" * 40)

    metrics = {
        "num_samples": num_samples,
        "num_prepared": len(prepared),
        "valid_count": valid_count,
        "match_count": match_count,
        "valid_rate": valid_count / max(num_samples, 1),
        "match_rate": match_count / max(num_samples, 1),
        "node_decode_correct": node_decode_correct,
        "node_decode_accuracy": node_decode_correct / max(num_samples, 1),
        "mean_tanimoto": mean_tanimoto,
        "median_tanimoto": median_tanimoto,
        "total_sampling_time_seconds": total_sampling_time,
        "results": results,
    }

    e["evaluation/num_samples"] = num_samples
    e["evaluation/valid_count"] = valid_count
    e["evaluation/match_count"] = match_count
    e["evaluation/valid_rate"] = metrics["valid_rate"]
    e["evaluation/match_rate"] = metrics["match_rate"]
    e["evaluation/node_decode_accuracy"] = metrics["node_decode_accuracy"]
    e["evaluation/mean_tanimoto"] = mean_tanimoto
    e["evaluation/median_tanimoto"] = median_tanimoto
    e["evaluation/total_sampling_time_seconds"] = total_sampling_time

    e.commit_json("evaluation_results.json", metrics)

    return metrics


# =============================================================================
# Entry Point
# =============================================================================

experiment.run_if_main()
