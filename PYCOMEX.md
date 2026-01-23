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
