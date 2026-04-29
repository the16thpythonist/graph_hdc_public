#!/usr/bin/env python
"""
Optuna HPO for the HDC Autoencoder on ZINC.

Each trial runs ``experiments/generation/train_autoencoder.py`` as a PyComex
experiment via subprocess, with hyperparameters overridden through PyComex's
``--PARAM value`` CLI interface.  After the child process finishes, this
script reads ``experiment_data.json`` from the trial's PyComex output folder
and uses ``results/best_recon_exact_match`` (the best graph-matching accuracy
recorded over the entire training run by ``ReconstructionEvalCallback``) as
the Optuna objective.  Direction is **maximize**.

Storage backend
---------------
Optuna's default SQLite storage is officially documented as **unstable
under concurrent writes from multiple processes** (see Optuna FAQ:
"How can I solve the error that occurs when performing parallel
optimization with SQLite?"). For multi-process / multi-machine workers
contributing to the same study, the canonical alternatives are:

  1. ``JournalStorage(JournalFileBackend(...))``  ← used here
       Append-only journal log on a single file, guarded by OS file
       locks.  Designed exactly for this use-case (parallel workers
       writing to a shared study), works on local FS and NFS, no
       server to set up.  Recommended in Optuna's "Easy Distributed
       Optimization" docs whenever SQLite isn't safe enough.
  2. ``JournalStorage(JournalRedisBackend(...))``
       Same journal model but backed by Redis.  Faster than the file
       backend at high parallelism, requires a running Redis server.
  3. PostgreSQL / MySQL via SQLAlchemy URL
       (``storage="postgresql+psycopg2://user:pw@host/db"``).
       Most robust at high write rates, but needs a real RDBMS.

For local + small-cluster parallel HPO, the file-journal backend is the
simplest reliable choice and is what we use below.  A portable CSV
mirror is also exported on every worker exit so the study can be
shipped between machines (the journal log itself is also portable —
just copy the file).

Usage
-----
    # Single worker, 50 trials
    python hpo_train_autoencoder.py --n_trials 50

    # Multiple workers (run on each machine / GPU concurrently — they
    # share the same journal file so trials don't collide):
    python hpo_train_autoencoder.py --n_trials 25 &
    python hpo_train_autoencoder.py --n_trials 25 &

    # Quick smoke test: short training, 2 trials
    python hpo_train_autoencoder.py --n_trials 2 --epochs 5

    # Override the default journal location
    python hpo_train_autoencoder.py --journal /shared/nfs/path/zinc_ae.log
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import optuna
import pandas as pd

# JournalStorage moved between optuna 3.x and 4.x — support both.
try:
    # optuna >= 4
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend  # type: ignore
    _JOURNAL_FACTORY = lambda path: JournalStorage(JournalFileBackend(str(path)))
except ImportError:
    # optuna 3.x
    from optuna.storages import JournalFileStorage, JournalStorage  # type: ignore
    _JOURNAL_FACTORY = lambda path: JournalStorage(JournalFileStorage(str(path)))


# ── Paths ────────────────────────────────────────────────────────────

THIS_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = THIS_DIR / "train_autoencoder.py"

# Where train_autoencoder.py drops its PyComex result folders. Matches
# `base_path=folder_path(__file__)` + `namespace=file_namespace(__file__)`.
TRAIN_RESULTS_DIR = THIS_DIR / "results" / "train_autoencoder"

# HPO bookkeeping folder (study journal, CSV mirror, logs)
HPO_DIR = THIS_DIR / "results" / "hpo_train_autoencoder"
HPO_DIR.mkdir(parents=True, exist_ok=True)

STUDY_NAME = "ae_zinc_match_acc"
DEFAULT_JOURNAL = HPO_DIR / "study.journal.log"
DEFAULT_CSV = HPO_DIR / "trials.csv"

# Default ZINC encoder.  Can be overridden via --encoder.
DEFAULT_ENCODER = (
    THIS_DIR.parent / "encoders" / "zinc_d1024_depth5_k4_8_12_18_b8.zip"
)


# ── Logging ──────────────────────────────────────────────────────────


def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Optuna search space ──────────────────────────────────────────────


# ── Fixed (non-tuned) train_autoencoder parameters ──────────────────
# These are passed to every trial as constant overrides.
FIXED_OVERRIDES = {
    "VARIATIONAL": False,
    "USE_EMA": False,
    "WARMUP_EPOCHS": 5,
    "COSINE_WEIGHT": 0.0,
    "RECON_LOSS_TYPE": "berhu",
    "FFN_MULT": 8 / 3,  # canonical SwiGLU multiplier (LLaMA/PaLM)
    # Early stopping on val/loss — kills plateaued trials early to save
    # compute. Patience = 25 epochs is conservative w.r.t. the cosine LR
    # schedule (which keeps decaying through EPOCHS).
    "EARLY_STOPPING": True,
    "EARLY_STOPPING_PATIENCE": 25,
    "EARLY_STOPPING_MIN_DELTA": 1e-4,
}


def get_search_space() -> dict[str, optuna.distributions.BaseDistribution]:
    """Distributions for ``optuna.trial.create_trial`` when rebuilding
    the study from a CSV mirror.  Keep in sync with ``sample_params``."""
    return {
        "latent_dim": optuna.distributions.CategoricalDistribution([128, 256]),
        "trunk_dim": optuna.distributions.CategoricalDistribution([512, 1024, 1536, 2048]),
        "n_blocks": optuna.distributions.IntDistribution(4, 12),
        "dropout": optuna.distributions.FloatDistribution(0.0, 0.15),
        "learning_rate": optuna.distributions.FloatDistribution(5e-5, 5e-3, log=True),
        "weight_decay": optuna.distributions.FloatDistribution(1e-7, 1e-3, log=True),
        "batch_size": optuna.distributions.IntDistribution(64, 512),
    }


def sample_params(trial: optuna.Trial) -> dict:
    """Sample one trial's hyperparameters from the search space.

    ``n_blocks`` controls both encoder and decoder block counts (kept tied
    for symmetry — the autoencoder is a mirrored stack)."""
    return {
        "latent_dim": trial.suggest_categorical("latent_dim", [128, 256]),
        "trunk_dim": trial.suggest_categorical("trunk_dim", [512, 1024, 1536, 2048]),
        "n_blocks": trial.suggest_int("n_blocks", 4, 12),
        "dropout": trial.suggest_float("dropout", 0.0, 0.15),
        "learning_rate": trial.suggest_float("learning_rate", 5e-5, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True),
        "batch_size": trial.suggest_int("batch_size", 64, 512),
    }


# ── Subprocess invocation ────────────────────────────────────────────


def trial_prefix(trial_number: int) -> str:
    """PyComex __PREFIX__ used to locate the trial's result folder."""
    return f"hpo_t{trial_number:04d}__"


def build_command(
    params: dict,
    *,
    encoder_path: Path,
    epochs: int,
    seed: int,
    device: str,
    hdc_device: str,
    prefix: str,
    extra_overrides: dict | None = None,
) -> list[str]:
    """Build the `python train_autoencoder.py --PARAM value ...` command
    for one trial.  PyComex's CLI parses values via ``eval()``, so booleans
    and numbers are passed as Python literals."""
    cmd: list[str] = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--DATASET", repr("zinc"),
        "--ENCODER_PATH", repr(str(encoder_path)),
        "--__DEBUG__", "False",
        "--__TESTING__", "False",
        "--__PREFIX__", repr(prefix),
        "--SEED", str(seed),
        "--EPOCHS", str(epochs),
        "--DEVICE", repr(device),
        "--HDC_DEVICE", repr(hdc_device),
        "--VERBOSE", "False",
        # Sampled params → train_autoencoder.py PARAMETERS
        "--LATENT_DIM", str(params["latent_dim"]),
        "--TRUNK_DIM", str(params["trunk_dim"]),
        "--N_ENCODER_BLOCKS", str(params["n_blocks"]),
        "--N_DECODER_BLOCKS", str(params["n_blocks"]),
        "--DROPOUT", str(params["dropout"]),
        "--LEARNING_RATE", str(params["learning_rate"]),
        "--WEIGHT_DECAY", str(params["weight_decay"]),
        "--BATCH_SIZE", str(params["batch_size"]),
    ]
    # Apply fixed (non-tuned) overrides
    for k, v in FIXED_OVERRIDES.items():
        cmd += [f"--{k}", repr(v) if isinstance(v, str) else str(v)]
    if extra_overrides:
        for k, v in extra_overrides.items():
            cmd += [f"--{k}", repr(v) if isinstance(v, str) else str(v)]
    return cmd


def find_trial_result_dir(prefix: str, started_after: float) -> Path | None:
    """Locate the PyComex output folder created by this trial.

    PyComex names folders ``{__PREFIX__}{timestamp}__{rand}``.  We pick
    the newest folder under ``results/train_autoencoder/`` that
    starts with ``prefix`` and was created after the subprocess kicked
    off (mtime guard against stale leftovers).
    """
    if not TRAIN_RESULTS_DIR.is_dir():
        return None
    candidates = [
        p for p in TRAIN_RESULTS_DIR.iterdir()
        if p.is_dir() and p.name.startswith(prefix) and p.stat().st_mtime >= started_after - 60
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def read_objective(result_dir: Path) -> tuple[float, dict]:
    """Read the objective (best graph matching accuracy, %) from a
    completed PyComex run.  Falls back to ``max(recon_exact_match)``
    if the explicit ``results/best_recon_exact_match`` key is absent
    (older runs)."""
    data_path = result_dir / "experiment_data.json"
    if not data_path.is_file():
        raise FileNotFoundError(f"No experiment_data.json in {result_dir}")

    with data_path.open() as f:
        data = json.load(f)

    extras: dict = {}

    # Preferred: the result key written by ReconstructionEvalCallback when
    # it sees a new best (post-commit b5922d6).
    results = data.get("results") or {}
    best = results.get("best_recon_exact_match")
    if best is not None:
        extras["best_recon_epoch"] = results.get("best_recon_epoch")
        extras["final_recon_exact_match"] = results.get("final_recon_exact_match")
        extras["final_recon_validity"] = results.get("final_recon_validity")
        extras["final_recon_tanimoto"] = results.get("final_recon_tanimoto")
        extras["final_val_mse"] = results.get("final_val_mse")
        extras["final_val_cos_sim"] = results.get("final_val_cos_sim")
        return float(best), extras

    # Fallback: peak of the per-epoch tracked series.
    series = data.get("recon_exact_match")
    if not series:
        raise KeyError(
            f"Neither results/best_recon_exact_match nor recon_exact_match "
            f"present in {data_path}"
        )
    extras["fallback_used"] = "max(recon_exact_match)"
    return float(max(series)), extras


# ── Trial driver ─────────────────────────────────────────────────────


def run_trial(
    trial: optuna.Trial,
    *,
    encoder_path: Path,
    epochs: int,
    seed: int,
    device: str,
    hdc_device: str,
    timeout: int | None,
) -> float:
    params = sample_params(trial)
    prefix = trial_prefix(trial.number)
    log(f"Trial {trial.number}: {params}")

    cmd = build_command(
        params,
        encoder_path=encoder_path,
        epochs=epochs,
        seed=seed,
        device=device,
        hdc_device=hdc_device,
        prefix=prefix,
    )
    trial.set_user_attr("prefix", prefix)
    trial.set_user_attr("cmd", " ".join(cmd))

    log_path = HPO_DIR / "trial_logs" / f"trial_{trial.number:04d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    with log_path.open("w") as logf:
        logf.write(f"# cmd: {' '.join(cmd)}\n")
        logf.flush()
        try:
            proc = subprocess.run(
                cmd,
                stdout=logf,
                stderr=subprocess.STDOUT,
                cwd=str(THIS_DIR),
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            log(f"Trial {trial.number}: TIMEOUT after {timeout}s")
            trial.set_user_attr("failure_reason", f"timeout({timeout}s)")
            return float("-inf")

    elapsed = time.time() - started
    trial.set_user_attr("training_time_min", round(elapsed / 60, 2))

    if proc.returncode != 0:
        log(f"Trial {trial.number}: subprocess exit={proc.returncode} (log={log_path})")
        trial.set_user_attr("failure_reason", f"exit_{proc.returncode}")
        return float("-inf")

    result_dir = find_trial_result_dir(prefix, started_after=started)
    if result_dir is None:
        log(f"Trial {trial.number}: result dir not found (prefix={prefix})")
        trial.set_user_attr("failure_reason", "result_dir_missing")
        return float("-inf")

    trial.set_user_attr("result_dir", str(result_dir))

    try:
        value, extras = read_objective(result_dir)
    except (FileNotFoundError, KeyError) as e:
        log(f"Trial {trial.number}: {e}")
        trial.set_user_attr("failure_reason", f"read_objective: {e}")
        return float("-inf")

    for k, v in extras.items():
        if v is not None:
            trial.set_user_attr(k, v)

    log(
        f"Trial {trial.number}: best_recon_exact_match={value:.2f}% "
        f"(took {elapsed/60:.1f} min, dir={result_dir.name})"
    )
    return value


# ── Study management ─────────────────────────────────────────────────


def load_study(journal_path: Path) -> optuna.Study:
    """Open (or create) the parallel-safe journal-backed study."""
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    storage = _JOURNAL_FACTORY(journal_path)
    return optuna.create_study(
        study_name=STUDY_NAME,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
    )


def export_trials(study: optuna.Study, csv_path: Path) -> None:
    space = get_search_space()
    rows = []
    for t in study.get_trials(deepcopy=False):
        row: dict = {
            "number": t.number,
            "value": t.value,
            "state": t.state.name if hasattr(t.state, "name") else str(t.state),
        }
        for k in space:
            row[k] = t.params.get(k)
        for attr_name, attr_value in t.user_attrs.items():
            row[attr_name] = attr_value
        rows.append(row)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    log(f"Exported {len(rows)} trials to {csv_path}")


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Optuna HPO for HDC autoencoder on ZINC. "
            "Drives experiments/generation/train_autoencoder.py via subprocess. "
            "Objective: maximize results/best_recon_exact_match."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n_trials", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--encoder", type=str, default=str(DEFAULT_ENCODER),
        help="Path to ZINC HyperNet encoder checkpoint (.zip)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Training device passed to train_autoencoder (cuda/cpu/auto)",
    )
    parser.add_argument(
        "--hdc_device", type=str, default="cpu",
        help="HDC encoding device passed to train_autoencoder",
    )
    parser.add_argument(
        "--journal", type=str, default=str(DEFAULT_JOURNAL),
        help="Path to JournalFileBackend log (parallel-safe shared storage)",
    )
    parser.add_argument(
        "--csv", type=str, default=str(DEFAULT_CSV),
        help="Portable CSV mirror of the study (re-exported after each trial)",
    )
    parser.add_argument(
        "--timeout_min", type=int, default=0,
        help="Per-trial wallclock budget in minutes (0 = no limit)",
    )
    args = parser.parse_args()

    encoder_path = Path(args.encoder).expanduser().resolve()
    if not encoder_path.is_file():
        sys.exit(f"Encoder checkpoint not found: {encoder_path}")
    if not TRAIN_SCRIPT.is_file():
        sys.exit(f"Train script not found: {TRAIN_SCRIPT}")

    journal_path = Path(args.journal).expanduser().resolve()
    csv_path = Path(args.csv).expanduser().resolve()
    timeout = args.timeout_min * 60 if args.timeout_min > 0 else None

    study = load_study(journal_path)

    log(f"Study:        {STUDY_NAME}")
    log(f"Backend:      JournalStorage (parallel-safe) @ {journal_path}")
    log(f"CSV mirror:   {csv_path}")
    log(f"Encoder:      {encoder_path}")
    log(f"Train script: {TRAIN_SCRIPT}")
    log(f"Existing trials in study: {len(study.trials)}")
    log(f"New trials this worker:   {args.n_trials}")
    log(f"Epochs/trial: {args.epochs}, per-trial timeout: {timeout}s")
    print()

    def objective(trial: optuna.Trial) -> float:
        try:
            return run_trial(
                trial,
                encoder_path=encoder_path,
                epochs=args.epochs,
                seed=args.seed,
                device=args.device,
                hdc_device=args.hdc_device,
                timeout=timeout,
            )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            log(f"Trial {trial.number} unexpected error: {type(e).__name__}: {e}")
            trial.set_user_attr("failure_reason", f"{type(e).__name__}: {str(e)[:200]}")
            return float("-inf")

    try:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            callbacks=[lambda s, t: export_trials(s, csv_path)],
            gc_after_trial=True,
        )
    except KeyboardInterrupt:
        log("Interrupted by user (Ctrl+C)")
    finally:
        export_trials(study, csv_path)

    # ── Summary ──
    print()
    print("=" * 70)
    print("HPO SUMMARY — autoencoder/zinc")
    print("=" * 70)
    print(f"Total trials in study: {len(study.trials)}")

    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.value is not None and math.isfinite(t.value)
    ]
    if study.best_trial and math.isfinite(study.best_value):
        bt = study.best_trial
        print(f"\nBest trial: #{bt.number}")
        print(f"  best_recon_exact_match: {study.best_value:.2f}%")
        for k in ("training_time_min", "result_dir", "final_recon_exact_match",
                  "final_recon_validity", "final_recon_tanimoto"):
            if k in bt.user_attrs:
                print(f"  {k}: {bt.user_attrs[k]}")
        print("\n  Hyperparameters:")
        for k, v in bt.params.items():
            print(f"    {k}: {v}")

    if len(completed) > 1:
        sorted_trials = sorted(completed, key=lambda t: t.value, reverse=True)
        print(f"\nTop 5 trials (by best_recon_exact_match):")
        for i, t in enumerate(sorted_trials[:5], 1):
            print(f"  {i}. #{t.number}: {t.value:.2f}%  ({t.user_attrs.get('result_dir', '')})")

    print("=" * 70)
    print(f"Journal:    {journal_path}")
    print(f"CSV:        {csv_path}")
    print(f"Trial dirs: {TRAIN_RESULTS_DIR}")


if __name__ == "__main__":
    main()
