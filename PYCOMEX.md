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

    # Training loop (use underscores in track keys, not slashes)
    for epoch in range(EPOCHS):
        loss = train_one_epoch()
        e.track("loss_train", loss)  # Time-series tracking

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
e.log("Starting training...")             # Log to stdout + file
e.log(f"Epoch {i}: loss = {loss:.4f}")
e.log_parameters()                         # Print all parameters
e.log_parameters(["LEARNING_RATE"])        # Print specific parameters
```

### Tracking (Time-Series)

**Important**: `e.track()` **appends** to a list each call (time-series), while `e["key"] = value` **overwrites** (single value).

```python
# Appends to a list automatically
for epoch in range(EPOCHS):
    e.track("loss_train", loss)    # Creates [0.27, 0.29, 0.21, ...]
    e.track("loss_val", val_loss)

# Track multiple at once
e.track_many({"loss": loss, "accuracy": acc})
```

#### Tracking vs Storage Comparison

| Aspect | `e.track()` | `e["key"] = value` |
|--------|------------|-------------------|
| Behavior | **Appends** to list | **Overwrites** value |
| Use case | Training loss per epoch | Model config, final results |
| Data structure | Always a list | Any type |
| Auto-visualization | Yes (plugins generate plots) | No |

#### Known Bug: Use Underscores, Not Slashes

**Important**: Use underscores in tracked key names, not slashes:

```python
# CORRECT - use underscores
e.track("loss_train", loss)
e.track("loss_val", val_loss)

# INCORRECT - slash-separated keys may get overwritten instead of appended
e.track("loss/train", loss)  # Bug: may not append correctly
```

#### Accessing Tracked Data After Experiment

```python
# Load archived experiment
loaded = Experiment.load("/path/to/experiment/folder")

# Access tracked time-series
train_losses = loaded["loss_train"]  # [0.27, 0.29, 0.21, ...]

# Find all tracked quantities via metadata
tracked_keys = loaded.metadata["__track__"]  # ["loss_train", "loss_val", ...]
```

#### Automatic Visualization

The `PlotTrackedElementsPlugin` automatically generates:
- **Line plots** for numeric tracking (saved as `{key}.png`)
- **Videos** for image tracking (saved as `{key}.mp4`)

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

## Parameter Documentation

### Auto-Detection Convention

Any global variable with an **entirely UPPERCASE name** is automatically detected as a parameter:

```python
# All of these are auto-detected as parameters:
LEARNING_RATE: float = 0.001
BATCH_SIZE: int = 32
NUM_EPOCHS: int = 100
DATASET: str = "QM9"
```

### Adding Type Hints

Type hints are captured in metadata and displayed in help output:

```python
from typing import List, Dict, Optional

LEARNING_RATE: float = 0.001           # Type captured: "float"
BATCH_SIZE: int = 32                   # Type captured: "int"
MODELS: List[str] = ["bert", "gpt"]    # Type captured: "list[str]"
CONFIG: Dict[str, int] = {}            # Type captured: "dict[str, int]"
```

### Documenting with Comments

Use `:param` syntax in **comments** (not docstrings) directly above the parameter:

```python
# :param LEARNING_RATE:
#     Initial learning rate for Adam optimizer. Should be between 0.0001 and 0.1.
#     Can span multiple lines as long as they're indented with # and spaces.
LEARNING_RATE: float = 0.001

# :param BATCH_SIZE:
#     Number of samples per batch. Higher values improve stability but use more memory.
BATCH_SIZE: int = 32

# :param DATASET:
#     Which dataset to train on: "QM9" or "ZINC"
DATASET: str = "QM9"
```

**Note**: PyComex parses `# :param NAME:` from module comments, not from the docstring. The description must be on subsequent lines, indented with `#` followed by spaces.

### Accessing Parameters

Two equivalent methods:

```python
def experiment(e: Experiment):
    # Method 1: Direct attribute access (recommended)
    lr = e.LEARNING_RATE
    bs = e.BATCH_SIZE

    # Method 2: Via parameters dictionary
    lr = e.parameters["LEARNING_RATE"]
```

### CLI Parameter Override

Parameters can be overridden via command line:

```bash
python experiment.py --LEARNING_RATE 0.01 --BATCH_SIZE 64 --__TESTING__ True
```

The parser uses `eval()` to interpret values, so Python literals work:

```bash
python experiment.py --MODELS '["bert", "gpt2"]' --CONFIG '{"a": 1, "b": 2}'
```

## Special Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `__DEBUG__` | `False` | Reuses same output folder (for development) |
| `__TESTING__` | `False` | Quick run mode for validation |
| `__REPRODUCIBLE__` | `False` | Captures dependencies for reproducibility |
| `__CACHING__` | `True` | Enable/disable computation caching |
| `__PREFIX__` | `""` | Prefix for experiment name |
| `__OPTUNA__` | `False` | Enable Optuna hyperparameter optimization |
| `__INCLUDE__` | `None` | Mixin files to include (str or list[str]) |

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
    train_losses = e["loss_train"]
    min_loss = min(train_losses)
    min_epoch = train_losses.index(min_loss)
    e.log(f"Minimum loss: {min_loss:.4f} at epoch {min_epoch}")
    e.commit_json("analysis.json", {"min_loss": min_loss, "min_epoch": min_epoch})
```

### Hooks

PyComex has a **two-tier hook system**:
- **Plugin-level hooks** (global): Use `@hook(name, priority)` decorator, sorted by priority
- **Experiment-level hooks** (local): Use `@experiment.hook(name, replace, default)` decorator

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

#### Hook Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `replace` | `True` | If `True`, replaces all previous hooks for that name. If `False`, appends to existing hooks. |
| `default` | `True` | If `True`, hook only registers if none exists yet (fallback behavior). |

#### Hook Patterns

**Filter hooks** (transform data and return modified value):
```python
# In base experiment
WORDS = e.apply_hook("filter_words", words=WORDS, default=WORDS)

# In child experiment
@experiment.hook("filter_words")
def remove_random_words(e, words):
    return modified_words  # Return value used by parent
```

**Action hooks** (perform side effects, no return value):
```python
@experiment.hook("after_run", replace=False)
def extra_logging(e):
    e.log("Extra logging from child")
```

#### Built-in Hook Points

- `before_run` - Right before main experiment code executes
- `after_run` - Right after main experiment code completes
- `before_testing` - Before testing mode adjustments
- Custom hooks via `e.apply_hook("custom_name", ...)`

#### Hook Best Practices

1. **Use `replace=False` for composable behavior** - Multiple hooks can coexist
2. **Use `replace=True` only when truly replacing** parent behavior
3. **Use `default=True` in base classes** for fallback implementations
4. **Document hook purpose** with docstrings including parameters and return values
5. **Filter hooks must return values**; action hooks typically don't

## Experiment Inheritance

Create variations without duplicating code. The `Experiment.extend()` method enables inheritance.

### Basic Inheritance

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

### How Inheritance Works

1. **Parent experiment is loaded** with all its parameters and hooks
2. **Child's globals update parent's globals** (child parameters override)
3. **`update_parameters()` discovers** all uppercase variables from merged globals
4. **Hooks are inherited** with control via `replace` and `default` flags

### Hook Inheritance

```python
# Parent experiment
@experiment.hook("filter_words", default=True)
def default_filter(e, words):
    return words  # Fallback implementation

# Child experiment - overrides parent hook
@experiment.hook("filter_words")
def custom_filter(e, words):
    return words[:100]  # Child's implementation replaces parent

# Child experiment - appends to parent hooks
@experiment.hook("after_run", replace=False)
def child_cleanup(e):
    e.log("Child cleanup")  # Runs after parent's after_run hooks
```

### Mixins (Reusable Hook Containers)

Mixins provide reusable hooks without full experiment inheritance:

```python
# logging_mixin.py
from pycomex.functional.mixin import ExperimentMixin

mixin = ExperimentMixin(glob=globals())

@mixin.hook("before_run", replace=False)
def log_start(e):
    e.log("Experiment starting...")

@mixin.hook("after_run", replace=False)
def log_end(e):
    e.log("Experiment finished!")
```

```python
# experiment.py
experiment = Experiment.extend(...)
experiment.include('logging_mixin.py')  # Include after extend
```

### Inheritance Gotchas

1. **Hooks are prepended, not appended** when `replace=False` - This maintains intuitive execution order (parent → child) but registration order is reversed internally.

2. **Nested parameter updates may not work** - Deep nested structures won't merge correctly. Keep parameter structures flat or manually merge nested dicts.

3. **Parameters must be in module scope** - They're discovered at initialization time via uppercase global scanning, not computed dynamically.

4. **Always use relative paths in `extend()`**:
   ```python
   experiment = Experiment.extend(
       "base_experiment.py",  # Relative to current file
       ...
   )
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

### General

1. **Put parameters at the top** - All uppercase globals are auto-detected
2. **Add type hints** to parameters for metadata capture: `LEARNING_RATE: float = 0.001`
3. **Document parameters with `:param`** syntax in module docstrings
4. **Use `__DEBUG__ = True`** during development to avoid folder proliferation
5. **Use `e.log()` not `print()`** to ensure messages go to log files
6. **Always call `experiment.run_if_main()`** at the end of the file

### Data Storage

7. **Use slash notation for `e["key"]`** storage: `e["metrics/train/loss"]`
8. **Use underscores for `e.track()`** keys: `e.track("loss_train", loss)` (not `loss/train`)
9. **Use `e.track()` for time-series**, `e["key"] = value` for single values

### Hooks and Inheritance

10. **Use `replace=False`** for composable hooks that append to parent behavior
11. **Use `default=True`** in base experiments for fallback hook implementations
12. **Include mixins after `extend()`**: `experiment.include('mixin.py')`
13. **Always use relative paths** in `Experiment.extend()`

### Logging Parameters

14. **Log parameters early** for debugging: `e.log_parameters()` at experiment start

## Reference

- PyComex repository: `/tmp/pycomex`
- Example files: `/tmp/pycomex/pycomex/examples/`
- Full documentation: `/tmp/pycomex/docs/`
