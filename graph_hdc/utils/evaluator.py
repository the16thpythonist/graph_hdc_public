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
from rdkit.Chem import QED, AllChem, Crippen, DataStructs, Descriptors
from rdkit.Contrib.SA_Score import sascorer

from graph_hdc.datasets import QM9Smiles, ZincSmiles
from graph_hdc.hypernet import CorrectionLevel
from graph_hdc.utils.chem import canonical_key, is_valid_molecule, reconstruct_for_eval


# 3x3 grid property configuration: (key, display label, histogram range, RDKit fn).
# Order defines both the output dict keys and the panel layout (row-major).
PROPERTY_GRID_CONFIG: list[tuple[str, str, tuple[float, float], Callable]] = [
    ("logp",                "logP",               (-5.0, 10.0),  lambda m: Crippen.MolLogP(m)),
    ("qed",                 "QED",                (0.0, 1.0),    lambda m: QED.qed(m)),
    ("sa_score",            "SA Score",           (1.0, 10.0),   lambda m: sascorer.calculateScore(m)),
    ("heavy_atoms",         "Heavy atoms",        (0.0, 50.0),   lambda m: float(m.GetNumHeavyAtoms())),
    ("max_ring_size",       "Max ring size",      (0.0, 20.0),   lambda m: float(rdkit_max_ring_size(m))),
    ("fraction_sp3",        "Fraction sp3",       (0.0, 1.0),    lambda m: Descriptors.FractionCSP3(m)),
    ("num_rings",           "# Rings",            (0.0, 10.0),   lambda m: float(Descriptors.RingCount(m))),
    ("num_aromatic_rings",  "# Aromatic rings",   (0.0, 8.0),    lambda m: float(Descriptors.NumAromaticRings(m))),
    ("num_rotatable_bonds", "# Rotatable bonds",  (0.0, 15.0),   lambda m: float(Descriptors.NumRotatableBonds(m))),
]


def compute_property_dict(mols: list[Chem.Mol]) -> dict[str, list[float]]:
    """Compute all 9 grid properties for a list of valid molecules."""
    out: dict[str, list[float]] = {key: [] for key, _, _, _ in PROPERTY_GRID_CONFIG}
    for m in mols:
        if m is None:
            continue
        for key, _, _, fn in PROPERTY_GRID_CONFIG:
            try:
                out[key].append(float(fn(m)))
            except Exception:
                pass
    return out


def histogram_kl_divergence(
    gen: list[float] | np.ndarray,
    ref: list[float] | np.ndarray,
    bins: int = 100,
    value_range: tuple[float, float] | None = None,
    eps: float = 1e-10,
) -> float:
    """KL(gen || ref) between histograms of two scalar distributions."""
    gen_arr = np.asarray(gen, dtype=float)
    ref_arr = np.asarray(ref, dtype=float)
    if gen_arr.size == 0 or ref_arr.size == 0:
        return float("nan")
    if value_range is None:
        lo = min(float(gen_arr.min()), float(ref_arr.min()))
        hi = max(float(gen_arr.max()), float(ref_arr.max()))
        if hi == lo:
            return 0.0
        value_range = (lo, hi)
    gen_hist, _ = np.histogram(gen_arr, bins=bins, range=value_range, density=False)
    ref_hist, _ = np.histogram(ref_arr, bins=bins, range=value_range, density=False)
    p = gen_hist.astype(float) + eps
    q = ref_hist.astype(float) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))


def compute_fcd_score(
    gen_smiles: list[str],
    ref_smiles: list[str],
    device: str = "cpu",
) -> float:
    """Fréchet ChemNet Distance between two SMILES lists."""
    try:
        from fcd_torch import FCD
    except ImportError:
        return float("nan")
    try:
        fcd = FCD(device=device, n_jobs=1, canonize=True)
        return float(fcd(ref=ref_smiles, gen=gen_smiles))
    except Exception:
        return float("nan")


def plot_property_grid(
    gen_props: dict[str, list[float]],
    ref_props: dict[str, list[float]],
    kl_values: dict[str, float] | None = None,
    title: str | None = None,
):
    """Render a 3x3 grid of overlaid histograms (gen vs reference)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    bins = 40
    for ax, (key, label, vrange, _) in zip(axes.flat, PROPERTY_GRID_CONFIG, strict=False):
        gen_vals = gen_props.get(key, [])
        ref_vals = ref_props.get(key, [])
        if ref_vals:
            ax.hist(
                ref_vals, bins=bins, range=vrange, density=True,
                color="tab:gray", alpha=0.55, label=f"dataset (n={len(ref_vals)})",
            )
        if gen_vals:
            ax.hist(
                gen_vals, bins=bins, range=vrange, density=True,
                color="tab:red", alpha=0.55, label=f"generated (n={len(gen_vals)})",
            )
        title_str = label
        if kl_values is not None and key in kl_values and np.isfinite(kl_values[key]):
            title_str += f"   KL={kl_values[key]:.3f}"
        ax.set_title(title_str)
        ax.set_xlabel(label)
        ax.set_ylabel("density")
        ax.legend(loc="best", fontsize=8)
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


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

        # Optional injected reference property distributions (used for KL).
        # Populated via set_reference_properties(); not computed eagerly.
        self._reference_properties: dict[str, list[float]] | None = None

    def set_reference_properties(self, props: dict[str, list[float]]) -> None:
        """Inject pre-computed reference (training-set) property distributions.

        Expected keys match ``PROPERTY_GRID_CONFIG``.  Used by ``evaluate()``
        when ``compute_kl=True`` to compute per-property KL divergences.
        """
        self._reference_properties = dict(props)

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
        *,
        compute_kl: bool = False,
        compute_fcd: bool = False,
        fcd_device: str = "cpu",
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

        # Optional: per-property KL divergence against injected reference.
        # Also attaches ``_gen_properties`` (the full 9-property dict) so
        # callers can render the distribution grid without recomputing.
        kl_block: dict[str, float] = {}
        gen_props: dict[str, list[float]] | None = None
        if compute_kl and valid_mols:
            gen_props = compute_property_dict(valid_mols)
            ref = self._reference_properties
            if ref is not None:
                for key, _, vrange, _ in PROPERTY_GRID_CONFIG:
                    kl_block[f"kl_{key}"] = histogram_kl_divergence(
                        gen_props.get(key, []),
                        ref.get(key, []),
                        bins=100,
                        value_range=vrange,
                    )

        # Optional: Fréchet ChemNet Distance against the training set.
        fcd_block: dict[str, float] = {}
        if compute_fcd and valid_mols:
            gen_smiles = [canonical_key(m) for m in valid_mols]
            gen_smiles = [s for s in gen_smiles if s]
            if gen_smiles and self.train_smiles_list:
                fcd_block["fcd"] = compute_fcd_score(
                    gen_smiles, self.train_smiles_list, device=fcd_device,
                )

        result: dict = {
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
            **kl_block,
            **fcd_block,
        }
        if gen_props is not None:
            # Under a leading underscore so pycomex-style plain dict logging
            # doesn't try to serialize the whole distribution.
            result["_gen_properties"] = gen_props
        return result

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
