"""
Central generation evaluation module.

Provides consistent evaluation metrics for molecular generation,
analogous to src/generation/evaluator.py in the main repository.
"""

import math
from collections.abc import Callable

import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import QED, AllChem, Crippen, DataStructs
from rdkit.Contrib.SA_Score import sascorer

from graph_hdc.datasets import QM9Smiles, ZincSmiles
from graph_hdc.hypernet import CorrectionLevel
from graph_hdc.utils.chem import canonical_key, is_valid_molecule, reconstruct_for_eval


def rdkit_logp(m: Chem.Mol) -> float:
    """Calculate LogP."""
    return Crippen.MolLogP(m)


def rdkit_qed(m: Chem.Mol) -> float:
    """Calculate QED."""
    return QED.qed(m)


def rdkit_sa_score(m: Chem.Mol) -> float:
    """
    Calculate Synthetic Accessibility Score.

    Lower values (1-10 scale) indicate easier synthesis.
    """
    return sascorer.calculateScore(m)


def rdkit_max_ring_size(m: Chem.Mol) -> int:
    """Calculate maximum ring size in molecule."""
    ring_info = m.GetRingInfo()
    if not ring_info.NumRings():
        return 0
    return max(len(ring) for ring in ring_info.AtomRings())


def calculate_internal_diversity(mols: list[Chem.Mol], radius: int = 2, nbits: int = 2048) -> float:
    """
    Calculate internal diversity as average pairwise Tanimoto distance.

    Returns percentage (0-100).
    """
    if len(mols) < 2:
        return 0.0

    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nbits) for m in mols]

    similarities = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            similarities.append(sim)

    if not similarities:
        return 0.0

    avg_similarity = sum(similarities) / len(similarities)
    # Internal diversity = 1 - average similarity
    return 100.0 * (1.0 - avg_similarity)


class GenerationEvaluator:
    """Central evaluator for molecular generation metrics."""

    def __init__(self, base_dataset: str, device=None):
        """
        Initialize evaluator with training set for novelty checking.

        Args:
            base_dataset: "qm9" or "zinc"
            device: Computation device (optional)
        """
        self.device = device
        self.base_dataset = base_dataset

        # Load training and validation sets for novelty calculation
        # (matches original implementation which uses smiles + eval_smiles)
        if base_dataset == "zinc":
            train_dataset = ZincSmiles(split="train")
            valid_dataset = ZincSmiles(split="valid")
        else:
            train_dataset = QM9Smiles(split="train")
            valid_dataset = QM9Smiles(split="valid")

        # Combine training and validation smiles for novelty checking
        self.T = {d.smiles for d in train_dataset} | {d.smiles for d in valid_dataset}

        # Store training smiles as list for FCD (uses train only)
        self.train_smiles_list = [d.smiles for d in train_dataset]

        # Pre-compute training set properties for KL divergence
        self.train_properties = {"logp": [], "qed": []}
        for d in train_dataset:
            if hasattr(d, "logp") and d.logp is not None:
                val = d.logp.item() if hasattr(d.logp, "item") else float(d.logp)
                self.train_properties["logp"].append(val)
            if hasattr(d, "qed") and d.qed is not None:
                val = d.qed.item() if hasattr(d.qed, "item") else float(d.qed)
                self.train_properties["qed"].append(val)

        # State for current evaluation
        self.mols: list[Chem.Mol | None] | None = None
        self.valid_flags: list[bool] | None = None
        self.sims: list[float] | None = None
        self.correction_levels: list[CorrectionLevel] | None = None

    def to_mols_and_validate(self, samples: list[nx.Graph]) -> tuple[list[Chem.Mol | None], list[bool]]:
        """Convert NetworkX graphs to RDKit molecules and validate."""
        mols: list[Chem.Mol | None] = []
        for g in samples:
            try:
                mols.append(reconstruct_for_eval(g, dataset=self.base_dataset))
            except Exception as e:
                print(f"nx_to_mol error: {e}")
                mols.append(None)
        valid_flags = [(m is not None and is_valid_molecule(m)) for m in mols]
        return mols, valid_flags

    def evaluate(
        self,
        n_samples: int,
        samples: list[nx.Graph],
        final_flags: list[bool],
        sims: list[float],
        correction_levels: list[CorrectionLevel],
    ) -> dict[str, float]:
        """
        Evaluate unconditional generation metrics.

        Args:
            n_samples: Number of samples attempted
            samples: List of NetworkX graphs
            final_flags: Whether each sample reached final state
            sims: Cosine similarities to target
            correction_levels: Correction level for each sample

        Returns:
            Dictionary of evaluation metrics
        """
        def sim_stats(values: list[float], prefix: str) -> dict[str, float]:
            if not values:
                return {f"{prefix}_sim_mean": 0.0, f"{prefix}_sim_min": 0.0, f"{prefix}_sim_max": 0.0}
            v = np.array(values)
            return {
                f"{prefix}_sim_mean": float(v.mean()),
                f"{prefix}_sim_std": float(v.std()),
            }

        # Split sims by final vs nonfinal
        final_sims, non_final_sims = [], []
        for flag, s in zip(final_flags, sims, strict=False):
            best = s
            (final_sims if flag else non_final_sims).append(best)

        sims_eval = {}
        sims_eval.update(sim_stats(final_sims, "final"))
        sims_eval.update(sim_stats(non_final_sims, "nonfinal"))

        # Convert to molecules and validate
        mols, valid_flags = self.to_mols_and_validate(samples)
        self.mols = mols
        self.valid_flags = valid_flags
        self.sims = sims
        self.correction_levels = correction_levels

        n_valid = sum(valid_flags)
        validity = 100.0 * n_valid / n_samples if n_samples else 0.0

        # Uniqueness / novelty
        valid_canon = [canonical_key(m) for m, f in zip(mols, valid_flags, strict=False) if f]
        valid_canon = [c for c in valid_canon if c is not None]
        unique_valid = set(valid_canon)
        uniqueness = 100.0 * len(unique_valid) / n_valid if n_valid else 0.0

        novel_set = unique_valid - self.T
        novelty = 100.0 * len(novel_set) / n_valid if n_valid else 0.0
        nuv = 100.0 * len(novel_set) / n_samples if n_samples else 0.0

        # Calculate internal diversity metrics (p=1 with radius=2, p=2 with radius=3)
        valid_mols = [m for m, v in zip(mols, valid_flags, strict=False) if v]
        internal_div_p1 = calculate_internal_diversity(valid_mols, radius=2) if len(valid_mols) >= 2 else 0.0
        internal_div_p2 = calculate_internal_diversity(valid_mols, radius=3) if len(valid_mols) >= 2 else 0.0

        # Calculate property statistics for valid molecules
        prop_stats = {}
        if valid_mols:
            try:
                logp_vals = [rdkit_logp(m) for m in valid_mols]
                prop_stats["logp_mean"] = float(np.mean(logp_vals))
                prop_stats["logp_std"] = float(np.std(logp_vals))
            except Exception:
                prop_stats["logp_mean"] = float("nan")
                prop_stats["logp_std"] = float("nan")

            try:
                qed_vals = [rdkit_qed(m) for m in valid_mols]
                prop_stats["qed_mean"] = float(np.mean(qed_vals))
                prop_stats["qed_std"] = float(np.std(qed_vals))
            except Exception:
                prop_stats["qed_mean"] = float("nan")
                prop_stats["qed_std"] = float("nan")

            try:
                sa_vals = [rdkit_sa_score(m) for m in valid_mols]
                prop_stats["sa_score_mean"] = float(np.mean(sa_vals))
                prop_stats["sa_score_std"] = float(np.std(sa_vals))
            except Exception:
                prop_stats["sa_score_mean"] = float("nan")
                prop_stats["sa_score_std"] = float("nan")

            try:
                ring_vals = [rdkit_max_ring_size(m) for m in valid_mols]
                prop_stats["max_ring_size_mean"] = float(np.mean(ring_vals))
                prop_stats["max_ring_size_std"] = float(np.std(ring_vals))
            except Exception:
                prop_stats["max_ring_size_mean"] = float("nan")
                prop_stats["max_ring_size_std"] = float("nan")

        return {
            "dataset": self.base_dataset,
            "final_flags": 100.0 * sum(final_flags) / n_samples if n_samples else 0.0,
            "validity": validity,
            "uniqueness": uniqueness,
            "novelty": novelty,
            "nuv": nuv,
            "internal_diversity_p1": internal_div_p1,
            "internal_diversity_p2": internal_div_p2,
            "cos_sim": sims_eval,
            **prop_stats,
        }

    def evaluate_conditional(
        self,
        samples: list[nx.Graph],
        target: float,
        final_flags: list[bool],
        sims: list[list[float]],
        prop_fn: Callable = rdkit_logp,
        eps: float = 0.2,
        compute_diversity: bool = True,
        total_samples: int = 100,
    ) -> dict[str, dict[str, float]]:
        """
        Evaluate conditional generation (property targeting).

        Returns a dict with stable sections:
          - meta: dataset and configuration
          - total: metrics normalized by total_samples
          - valid: metrics over valid, non-NaN property samples
          - hits: metrics over the hit subset (|prop-target| <= eps among valid)
        """

        out = {
            "meta": {
                "dataset": self.base_dataset,
                "n_samples": len(samples),
                "total_samples": int(total_samples),
                "target": float(target),
                "epsilon": float(eps),
            },
            "total": {
                "validity_pct": 0.0,
                "final_pct": 100.0 * sum(final_flags) / total_samples if total_samples else 0.0,
            },
            "valid": {
                "n_valid": 0,
                "n_valid_non_nan": 0,
                "mae_to_target": float("nan"),
                "rmse_to_target": float("nan"),
                "success_at_eps_pct": 0.0,
                "final_success_at_eps_pct": 0.0,
                "uniqueness_pct": 0.0,
                "novelty_pct": 0.0,
            },
            "hits": {
                "n_hits": 0,
                "uniqueness_hits_pct": 0.0,
                "novelty_hits_pct": 0.0,
                "diversity_hits_pct": 0.0,
            },
        }

        # Convert to molecules and validate
        mols, valid = self.to_mols_and_validate(samples)
        self.mols = mols
        self.valid_flags = valid
        self.sims = [max(s) for s in sims]

        n_valid = int(sum(valid))
        out["valid"]["n_valid"] = n_valid
        out["total"]["validity_pct"] = 100.0 * n_valid / total_samples if total_samples else 0.0

        if n_valid == 0:
            return out

        # Compute property on valid only
        props_triplets = []
        tgt = float(target)
        v_idx = -1
        for i, (m, v, f) in enumerate(zip(mols, valid, final_flags, strict=False)):
            if not v:
                continue
            v_idx += 1
            try:
                p = float(prop_fn(m))
            except Exception:
                p = float("nan")
            props_triplets.append((v_idx, p, bool(f)))

        # Filter non-NaN props
        paired = [(vidx, p, tgt, f) for (vidx, p, f) in props_triplets if not math.isnan(p)]
        den = len(paired)
        out["valid"]["n_valid_non_nan"] = den
        if den == 0:
            return out

        # Absolute errors & success
        abs_err = [abs(p - t) for (_, p, t, _) in paired]
        finals = [f for (_, _, _, f) in paired]

        out["valid"]["mae_to_target"] = sum(abs_err) / den
        out["valid"]["rmse_to_target"] = math.sqrt(sum(e * e for e in abs_err) / den)
        out["valid"]["success_at_eps_pct"] = 100.0 * sum(e <= eps for e in abs_err) / den
        out["valid"]["final_success_at_eps_pct"] = (
            100.0 * sum((e <= eps) and f for e, f in zip(abs_err, finals, strict=False)) / den
        )

        # Uniqueness / novelty over valid
        valid_canon = [canonical_key(m) for m, v in zip(mols, valid, strict=False) if v]
        valid_canon = [c for c in valid_canon if c is not None]
        if valid_canon:
            uniq = 100.0 * len(set(valid_canon)) / len(valid_canon)
            novel = 100.0 * len(set(valid_canon) - self.T) / len(valid_canon)
            out["valid"]["uniqueness_pct"] = uniq
            out["valid"]["novelty_pct"] = novel

        # Hit subset (|prop-target| <= eps)
        hit_paired = [(vidx, p) for (vidx, p, _, _) in paired if abs(p - tgt) <= eps]
        out["hits"]["n_hits"] = len(hit_paired)
        if not hit_paired:
            return out

        # Map hit valid indices back to canonical keys
        hit_valid_idx = [vidx for (vidx, _) in hit_paired]
        hit_keys = []
        for vidx in hit_valid_idx:
            if 0 <= vidx < len(valid_canon):
                k = valid_canon[vidx]
                if k is not None:
                    hit_keys.append(k)

        if hit_keys:
            n_hits = len(hit_paired)
            out["hits"]["uniqueness_hits_pct"] = 100.0 * len(set(hit_keys)) / n_hits
            out["hits"]["novelty_hits_pct"] = 100.0 * len(set(hit_keys) - self.T) / n_hits

        if compute_diversity and len(hit_valid_idx) >= 2:
            # Rebuild list of valid mol indices
            valid_orig_idx = [i for i, v in enumerate(valid) if v]
            hit_orig_idx = [valid_orig_idx[vidx] for vidx in hit_valid_idx]
            hit_mols = [mols[i] for i in hit_orig_idx]
            fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in hit_mols]
            sims_list = [
                DataStructs.TanimotoSimilarity(fps[i], fps[j]) for i in range(len(fps)) for j in range(i + 1, len(fps))
            ]
            if sims_list:
                out["hits"]["diversity_hits_pct"] = 100.0 * (1.0 - (sum(sims_list) / len(sims_list)))

        return out

    def get_mols_valid_flags_sims_and_correction_levels(self):
        """Get cached evaluation state."""
        return self.mols, self.valid_flags, self.sims, self.correction_levels
