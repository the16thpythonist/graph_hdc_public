# CLAUDE.md

## Project Overview

**GraphHDC** is a Master's thesis implementation combining Hyperdimensional Computing (HDC) with normalizing flows for probabilistic molecular graph generation. The key innovation is using training-free Vector Symbolic Architectures (VSA) with Holographic Reduced Representations (HRR) to encode molecular graphs deterministically, then learning distributions over these representations with normalizing flows.

## Repository Structure

```
graph_hdc/
├── hypernet/           # HDC Encoder/Decoder (training-free)
│   ├── encoder.py      # HyperNet - main encoding engine
│   ├── decoder.py      # Graph reconstruction algorithms
│   ├── configs.py      # Dataset-specific HDC configurations
│   └── feature_encoders.py  # Codebook generation
├── models/
│   ├── flows/          # Real NVP normalizing flow
│   └── regressors/     # Property prediction (LogP, QED)
├── datasets/           # PyG dataset implementations
│   ├── qm9_smiles.py   # QM9 dataset (133K molecules, ≤9 atoms)
│   ├── zinc_smiles.py  # ZINC dataset (249K molecules, avg 23 atoms)
│   └── utils.py        # Dataset utilities
└── utils/              # Helpers, evaluation metrics

experiments/
├── scripts/            # Training & evaluation entry points
│   ├── train_flow.py
│   ├── train_regressor.py
│   ├── evaluate_generation.py
│   └── run_optimization.py
└── configs/            # YAML hyperparameter configs

data/                   # Raw SMILES datasets
tests/                  # Unit tests
```

## Key Commands

```bash
# Installation (uses Pixi package manager)
pixi install
pixi shell -e local      # CPU
pixi shell -e local-gpu  # GPU

# Training
python experiments/scripts/train_flow.py --dataset qm9
python experiments/scripts/train_regressor.py --dataset qm9 --property logp

# Evaluation
python experiments/scripts/evaluate_generation.py --checkpoint <path> --dataset qm9 --n_samples 10000
python experiments/scripts/run_optimization.py --mode qed_max --flow_ckpt <path> --regressor_ckpt <path>

# Tests
pytest tests/
```

## Quick Usage

```python
from rdkit import Chem
from graph_hdc import HyperNet, get_config
from graph_hdc.datasets.qm9_smiles import mol_to_data

config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
hypernet = HyperNet(config)
mol = Chem.MolFromSmiles("CCO")
output = hypernet.forward(mol_to_data(mol), normalize=True)
graph_embedding = output["graph_embedding"]  # [1, 256]
```

## Core Concepts

- **HyperNet**: Training-free encoder using VSA message passing. Outputs `edge_terms` and `graph_terms` hypervectors.
- **DSHDCConfig**: Configuration dataclass defining feature encodings, hypervector dimensions, and message passing depth.
- **CombinatoricIntegerEncoder**: Maps discrete node features (atom type, degree, charge, hydrogens) to hypervector codebooks.
- **Real NVP Flow**: Learns distribution over hypervector space for generation.

## Adding Custom Datasets

1. Create dataset class in `graph_hdc/datasets/` inheriting from `InMemoryDataset`
2. Define `mol_to_data()` function producing PyG Data with `x`, `edge_index`, `smiles`
3. Add configuration in `configs.py` with feature bins matching your node features
4. Register in `get_split()` in `datasets/utils.py`

## Dependencies

Core: PyTorch, PyTorch Geometric, RDKit, TorchHD, Normflows, PyTorch Lightning

## Experiment Framework

This project uses **PyComex** for experiment management. See [PYCOMEX.md](PYCOMEX.md) for detailed usage including:
- Experiment file structure and templates
- Parameter tracking and data storage APIs
- Artifact management (figures, checkpoints, logs)
- Experiment inheritance for creating variations

## Metadata

- **Author**: Arvand Kaveh (KIT, Institute of Theoretical Informatics)
- **License**: MIT
