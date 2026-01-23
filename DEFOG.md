# DEFOG.md

## Overview

**DeFoG (Discrete Flow Matching for Graph Generation)** is a generative model for discrete graph-structured data. Its key innovation is **decoupling training from sampling** - parameters like `eta`, `omega`, and `time_distortion` can be adjusted at inference time without retraining.

The `src/core` module in `/tmp/DeFoG` provides a standalone, clean API.

## Quick Start

```python
from src.core import DeFoGModel
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl

# Create model from dataloader (auto-infers dimensions & marginals)
loader = DataLoader(dataset, batch_size=32)
model = DeFoGModel.from_dataloader(
    loader,
    n_layers=6,
    hidden_dim=256,
    noise_type="marginal",
)

# Train with PyTorch Lightning
trainer = pl.Trainer(max_epochs=100)
trainer.fit(model, train_dataloaders=loader)

# Generate samples
samples = model.sample(num_samples=100)  # Returns List[Data]

# Save/Load
model.save("checkpoint")
model = DeFoGModel.load("checkpoint", device="cuda")
```

## Constructor Parameters

```python
model = DeFoGModel(
    # Required: data dimensions
    num_node_classes=4,
    num_edge_classes=2,

    # Architecture
    n_layers=6,                      # Transformer depth (2-12)
    hidden_dim=256,                  # Hidden dimension (64-512)
    hidden_mlp_dim=512,              # MLP hidden dimension
    n_heads=8,                       # Attention heads
    dropout=0.1,

    # Noise distribution
    noise_type="marginal",           # "uniform", "marginal", "absorbing"
    node_marginals=tensor,           # Empirical node class frequencies
    edge_marginals=tensor,           # Empirical edge class frequencies

    # Graph sizes
    node_counts=tensor,              # Distribution of graph sizes
    max_nodes=100,

    # Extra features
    extra_features_type="rrwp",      # "none", "rrwp", "cycles"
    rrwp_steps=10,

    # Training
    lr=1e-4,
    weight_decay=1e-5,
    lambda_edge=1.0,                 # Edge loss weight vs node loss
    train_time_distortion="identity",

    # Sampling defaults (adjustable at inference)
    sample_steps=100,
    eta=0.0,
    omega=0.0,
    sample_time_distortion="identity",
)
```

## Sampling API

### Basic Sampling

```python
# Default sampling (uses constructor defaults)
samples = model.sample(num_samples=100)

# Override parameters at inference time
samples = model.sample(
    num_samples=100,
    eta=50.0,                    # Stochasticity
    omega=0.1,                   # Target guidance
    sample_steps=1000,           # Denoising iterations
    time_distortion="polydec",   # Step size distribution
    device="cuda",
    show_progress=True,
)

# Fixed graph size
samples = model.sample(num_samples=100, num_nodes=10)

# Variable sizes per sample
samples = model.sample(num_samples=100, num_nodes=torch.tensor([5, 8, 10, ...]))
```

### Key Sampling Parameters

| Parameter | Range | Effect |
|-----------|-------|--------|
| `eta` | 0-100 | Stochasticity for error correction. 0=deterministic, higher=more exploration |
| `omega` | 0-0.5 | Target guidance strength. Higher=closer to training distribution |
| `sample_steps` | 10-1000 | Denoising iterations. More=higher quality, slower |
| `time_distortion` | see below | Step size distribution. `polydec` critical for graphs |

### Time Distortion Options

| Type | Effect |
|------|--------|
| `identity` | Uniform steps |
| `polydec` | Smaller steps near t=1, preserves global structure (recommended for graphs) |
| `polyinc` | Smaller steps near t=0 |
| `cos` | Emphasize boundaries |

## Data Format

### Input (PyTorch Geometric)

```python
from torch_geometric.data import Data

data = Data(
    x=torch.Tensor(n_nodes, num_node_classes),    # One-hot node features
    edge_index=torch.LongTensor(2, num_edges),    # Edge indices
    edge_attr=torch.Tensor(num_edges, num_edge_classes),  # One-hot edge features
)
```

### Output from `sample()`

Returns `List[Data]` with the same format as input.

### Dense Conversion Utilities

```python
from src.core import to_dense, dense_to_pyg

# PyG batch to dense tensors
dense_data, node_mask = to_dense(
    batch.x, batch.edge_index, batch.edge_attr, batch.batch
)
# dense_data.X: (batch_size, max_nodes, dx)
# dense_data.E: (batch_size, max_nodes, max_nodes, de)
# node_mask: (batch_size, max_nodes) boolean

# Dense back to PyG
pyg_list = dense_to_pyg(dense_data.X, dense_data.E, dense_data.y, node_mask, n_nodes)
```

## Model Persistence

```python
# Save (auto-appends .ckpt)
path = model.save("my_model")

# Load
model = DeFoGModel.load("my_model", device="cuda")
model = DeFoGModel.load("my_model.ckpt", device="cpu")
```

## Noise Types

```python
from src.core import LimitDistribution

# Uniform: equal probability for all classes
limit_dist = LimitDistribution(
    noise_type="uniform",
    num_node_classes=4,
    num_edge_classes=2,
)

# Marginal: matches training data distribution (recommended)
limit_dist = LimitDistribution(
    noise_type="marginal",
    num_node_classes=4,
    num_edge_classes=2,
    node_marginals=torch.tensor([0.5, 0.2, 0.2, 0.1]),
    edge_marginals=torch.tensor([0.9, 0.1]),
)

# Absorbing: virtual absorbing state
limit_dist = LimitDistribution(
    noise_type="absorbing",
    num_node_classes=4,
    num_edge_classes=2,
)
```

## Dataset Statistics

```python
from src.core import compute_dataset_statistics

stats = compute_dataset_statistics(dataloader)
# Returns: num_node_classes, num_edge_classes, max_nodes,
#          node_marginals, edge_marginals, node_counts
```

## How It Works

### Training

1. Add noise via linear interpolation: `x_t = t*x_1 + (1-t)*noise`
2. GraphTransformer predicts clean data: `p_theta(x_1 | x_t)`
3. Cross-entropy loss against ground truth

### Sampling

1. Sample initial noise from `p_0`
2. Denoising loop (t=0 to T):
   - Predict `p_theta(x_1 | x_t)`
   - Compute rate matrix: `R_t = R*_t + eta*R^DB_t + omega*R^TG_t`
   - CTMC transition to next state
3. Return clean graphs as PyG Data objects

### Rate Matrix Components

- `R*_t`: Base flow matching rate (from predicted marginals)
- `R^DB_t`: Detailed balance term (controlled by `eta`, adds reversibility)
- `R^TG_t`: Target guidance term (controlled by `omega`, amplifies clean transitions)

## Public API

```python
from src.core import (
    # Main model
    DeFoGModel,

    # Data utilities
    PlaceHolder, to_dense, dense_to_pyg,
    compute_dataset_statistics, DistributionNodes,
    encode_no_edge, symmetrize_edges,

    # Noise distribution
    LimitDistribution, sample_noise, sample_from_probs,

    # Rate matrix
    RateMatrixDesigner,

    # Time distortion
    TimeDistorter,

    # Neural network
    GraphTransformer, XEyTransformerLayer, timestep_embedding,

    # Features
    ExtraFeatures, RRWPFeatures,

    # Loss
    TrainLoss, compute_loss_components,
)
```

## Reference

- DeFoG repository: `/tmp/DeFoG`
- Core module: `/tmp/DeFoG/src/core/`
- Documentation: `/tmp/DeFoG/src/core/README.md`
- Usage example: `/tmp/DeFoG/examples/core_usage.py`
