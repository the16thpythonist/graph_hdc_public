# PYCOMEX.md

## Overview

**PyComex (Python Computational Experiments)** is the experiment management framework used in this project. It handles:

- Automatic folder organization for experiment outputs
- Parameter tracking and metadata as JSON
- Artifact management (figures, checkpoints, logs)
- Experiment inheritance for creating variations without code duplication
- Command-line parameter overrides

## Experiment File Structure

### Minimal Template

```python
"""Experiment description - saved as metadata."""
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

# --- PARAMETERS (uppercase = auto-detected) ---
LEARNING_RATE: float = 0.001
EPOCHS: int = 100
BATCH_SIZE: int = 32

# --- SPECIAL PARAMETERS ---
__DEBUG__: bool = True        # Reuses same folder during development
__TESTING__: bool = False     # Quick run mode for validation

# --- EXPERIMENT DECORATOR ---
@Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)
def experiment(e: Experiment) -> None:
    """Main experiment logic."""

    # Store configuration
    e["config/learning_rate"] = LEARNING_RATE

    # Log messages (writes to stdout AND log file)
    e.log("Starting experiment...")

    # Training loop
    for epoch in range(EPOCHS):
        loss = train_one_epoch()
        e.track("metrics/loss", loss)  # Time-series tracking

    # Save artifacts
    e["results/final_loss"] = loss
    e.commit_json("results.json", {"final_loss": loss})
    e.commit_fig("training_curve.png", fig)

# --- REQUIRED: Run when executed directly ---
experiment.run_if_main()
```

### Required Components

1. **Decorator**: `@Experiment(base_path, namespace, glob)`
2. **Function signature**: `def experiment(e: Experiment) -> None:`
3. **Execution call**: `experiment.run_if_main()` at end of file

## Key APIs

### Data Storage

```python
# Slash notation creates nested structure automatically
e["config/learning_rate"] = 0.001
e["metrics/train/loss"] = 0.5
e["results/final_accuracy"] = 0.95

# Retrieve data
value = e["config/learning_rate"]
```

### Logging

```python
e.log("Starting training...")           # Log to stdout + file
e.log(f"Epoch {i}: loss = {loss:.4f}")
e.log_parameters()                       # Print all parameters
e.log_parameters(["LEARNING_RATE"])      # Print specific parameters
```

### Tracking (Time-Series)

```python
# Appends to a list automatically
for epoch in range(EPOCHS):
    e.track("metrics/loss", loss)
    e.track("metrics/accuracy", acc)

# Track multiple at once
e.track_many({"loss": loss, "accuracy": acc})
```

### Saving Artifacts

```python
e.commit_json("config.json", config_dict)    # Save JSON
e.commit_fig("plot.png", matplotlib_fig)     # Save figure
e.commit_raw("summary.txt", "text content")  # Save raw text

# File handle for custom formats
with e.open("custom.pkl", "wb") as f:
    pickle.dump(data, f)
```

### Parameter Access

```python
# Direct attribute access
lr = e.LEARNING_RATE
epochs = e.EPOCHS

# Via parameters dict
lr = e.parameters["LEARNING_RATE"]
```

## Special Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `__DEBUG__` | `False` | Reuses same output folder (for development) |
| `__TESTING__` | `False` | Quick run mode for validation |
| `__REPRODUCIBLE__` | `False` | Captures dependencies for reproducibility |
| `__CACHING__` | `True` | Enable/disable computation caching |

## Optional Decorators

### Testing Mode

```python
@experiment.testing
def testing(e: Experiment):
    """Runs when __TESTING__ = True. Reduce iterations for quick tests."""
    e.EPOCHS = 2
    e.BATCH_SIZE = 4
```

### Analysis

```python
@experiment.analysis
def analysis(e: Experiment):
    """Post-experiment analysis - runs after main experiment."""
    min_loss = min(e["metrics/loss"])
    e.log(f"Minimum loss achieved: {min_loss}")
    e.commit_json("analysis.json", {"min_loss": min_loss})
```

### Hooks

```python
@experiment.hook("before_training")
def setup(e: Experiment):
    e.log("Setting up...")

@experiment.hook("after_epoch")
def log_epoch(e: Experiment, epoch, loss):
    e.log(f"Epoch {epoch}: {loss:.4f}")

# Execute hooks in main experiment
def experiment(e: Experiment):
    e.apply_hook("before_training")
    for epoch in range(EPOCHS):
        loss = train()
        e.apply_hook("after_epoch", epoch=epoch, loss=loss)
```

## Experiment Inheritance

Create variations without duplicating code:

```python
# child_experiment.py
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

experiment = Experiment.extend(
    "parent_experiment.py",
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

# Override parameters (just redefine them)
LEARNING_RATE = 0.01  # Different from parent

# Extend via hooks defined in parent
@experiment.hook("process_data")
def custom_processing(e, data):
    return modified_data

experiment.run_if_main()
```

## Caching

PyComex provides a caching system for expensive computations that don't need to be repeated across runs. **Use with care** - cached data can become stale if preprocessing logic changes but the cache scope remains the same.

### Basic Usage

```python
@experiment
def experiment(e: Experiment):
    # The @e.cache.cached decorator caches function results to disk
    @e.cache.cached(name="heavy_computation", scope=("preprocessing",))
    def heavy_computation(data):
        # Expensive operation - only runs once, then loads from cache
        return processed_data

    result = heavy_computation(my_data)  # First call: computes and caches
    result = heavy_computation(my_data)  # Subsequent calls: loads from cache
```

### Cache Scope

The `scope` parameter defines the folder structure for cached files. Use it to organize cache by parameters that affect the computation:

```python
# Static scope
@e.cache.cached("model", scope=("models", "transformer"))

# Dynamic scope based on experiment parameters
def get_scope(e):
    return ("data", e.DATASET, f"dim_{e.HDC_DIM}")

@e.cache.cached("embeddings", scope=get_scope)
def compute_embeddings():
    ...
```

### Cache Backends

```python
from pycomex.functional.cache import CacheBackend

# Pickle (default) - general Python objects
@e.cache.cached("data", backend=CacheBackend.PICKLE)

# Joblib - optimized for NumPy arrays and large data
@e.cache.cached("arrays", backend=CacheBackend.JOBLIB)

# JSON - human-readable, limited to JSON-serializable types
@e.cache.cached("config", backend=CacheBackend.JSON)
```

### Disabling Cache

```python
# Disable for a fresh run (no loading or saving)
e.cache.set_enabled(False)

# Or use __CACHING__ parameter
__CACHING__: bool = False  # Disables caching when set
```

### Direct Save/Load API

```python
# Save directly (without decorator)
e.cache.save(data, name="my_data", scope=("preprocessing",))

# Load directly
data = e.cache.load(name="my_data", scope=("preprocessing",))
```

### Cache Location

Cached files are stored in `<experiment_path>/cache/` with folder structure matching the scope:

```
results/my_experiment/debug/
└── cache/
    └── preprocessing/
        └── heavy_computation.pkl.gz
```

### Caution

- **Stale cache**: If you change preprocessing logic but keep the same scope, the cache will return outdated results. Either change the scope or delete the cache folder.
- **Large objects**: Caching large datasets (e.g., processed PyG Data lists) can consume significant disk space.
- **Reproducibility**: Cache bypasses computation, which may hide bugs introduced in later code changes.

## Running Experiments

```bash
# Run directly
python experiments/my_experiment.py

# Override parameters via CLI
python experiments/my_experiment.py --LEARNING_RATE 0.01 --EPOCHS 50

# Run in testing mode
python experiments/my_experiment.py --__TESTING__ True
```

## Output Structure

Each run creates a timestamped folder:

```
results/experiment_name/
└── 2024-01-15__14-30-00__abc123/
    ├── experiment_log.txt    # All e.log() output
    ├── experiment_meta.json  # Parameters, timing, metadata
    ├── experiment_data.json  # All e["key"] = value data
    ├── code/                 # Snapshot of experiment code
    ├── results.json          # Custom artifacts
    └── plot.png
```

With `__DEBUG__ = True`, output goes to a reusable `debug/` folder instead.

## Best Practices

1. **Put parameters at the top** - All uppercase globals are auto-detected
2. **Use `__DEBUG__ = True`** during development to avoid folder proliferation
3. **Use `e.log()` not `print()`** to ensure messages go to log files
4. **Use slash notation** for nested data organization: `e["metrics/train/loss"]`
5. **Track metrics with `e.track()`** for automatic time-series storage
6. **Always call `experiment.run_if_main()`** at the end of the file

## Reference

- PyComex repository: `/tmp/pycomex`
- Example files: `/tmp/pycomex/pycomex/examples/`
- Full documentation: `/tmp/pycomex/docs/`
