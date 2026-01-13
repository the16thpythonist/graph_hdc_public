# GraphHDC: Hyperdimensional Representations for Probabilistic Graph Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

This repository provides the reproducible implementation for the Master's thesis on Hyperdimensional Computing (HDC) for molecular graph encoding, generation, and property-guided optimization.

---

## Thesis Metadata

| Field | Details |
|-------|---------|
| **Title** | Hyperdimensional Representations for Probabilistic Graph Generation |
| **Author** | M.Sc. Arvand Kaveh |
| **Institution** | Karlsruhe Institute of Technology (KIT) |
| **Institute** | Institute of Theoretical Informatics |
| **Degree** | Master's Thesis |
| **First Reviewer** | Prof. Dr. Pascal Friederich |
| **Second Reviewer** | Prof. Dr. Gerhard Neumann |
| **Advisor** | M.Sc. Jonas Teufel |
| **Period** | June 21, 2025 – December 21, 2025 |

---

## Abstract

Generating molecular structures with targeted properties is fundamental to drug discovery and materials science. Deep generative models offer data-driven molecular design, but bridging discrete graph representations to continuous latent spaces requires sophisticated architectural adaptations—dequantization schemes, permutation-equivariant layers, or specialized coupling mechanisms. Whether training-free encoders can bypass these learned architectures while preserving sufficient structure for probabilistic generation remains unexplored.

Here we present GraphHDC, a framework combining Hyperdimensional Computing with normalizing flows for molecular graph generation. We demonstrate that training-free HDC encoders—operating through deterministic algebraic operations on random hypervectors—produce semantically meaningful representations where standard flow architectures learn molecular distributions and property regressors enable gradient-based conditional generation. Evaluated on QM9 and ZINC benchmarks, the framework achieves generation validity of 98.0% on QM9 and 99.6% on ZINC with 100% uniqueness. On QM9, distributional fidelity is competitive with flow-based baselines (FCD 0.70, KL 0.86); on ZINC, validity is high but distributional fidelity is limited (FCD 0.014), indicating that encoding design choices—particularly feature selection—directly determine distributional quality. For property-conditioned generation, QED maximization reaches the 0.948 benchmark ceiling with 100% novelty; LogP targeting achieves Top-100 MAD of 0.015.

These results establish Hyperdimensional Computing as a promising approach for molecular generation, with encoding schema design emerging as a key research frontier for improving distributional fidelity. By demonstrating that hyperdimensional encodings support both distribution learning and property optimization, this work opens new research directions at the intersection of symbolic computing and molecular design. Encoding schema design emerges as the primary lever for achieving full distributional fidelity—aromaticity and ring topology encoding offer concrete extension paths.

---

## Key Contributions

1. **First HDC-based probabilistic graph generator** — GraphHDC represents the first application of Hyperdimensional Computing to probabilistic molecular graph generation.

2. **Training-free encoder-decoder pipeline** — Novel greedy beam search and stochastic pattern matching algorithms enable graph reconstruction from hypervectors without learned encoder/decoder parameters.

3. **Standard flow architectures on HDC space** — Demonstrates that Real NVP can effectively learn distributions over HDC hypervectors without specialized graph-aware modifications.

4. **Property-conditioned generation** — Gradient-based latent space optimization enables QED maximization and LogP targeting on the HDC-learned manifold.

---

## Installation

This project uses [Pixi](https://pixi.sh/) for package management:

```bash
# Clone the repository
git clone https://github.com/Vandool/graph_hdc_public.git
cd graph_hdc_public

# Install with Pixi
pixi install

# Activate the environment (CPU)
pixi shell -e local

# Or activate GPU environment
pixi shell -e local-gpu
pixi run setup-gpu  # Required once for GPU setup
```

### Verify Installation

```bash
# Run tests
pytest tests/ -v

# Verify GPU (if using local-gpu)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Quick Start

### Encode a Molecule

```python
import torch
from rdkit import Chem
from graph_hdc import HyperNet, get_config
from graph_hdc.datasets.qm9_smiles import mol_to_data

# Load configuration
config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")

# Create encoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hypernet = HyperNet(config).to(device)

# Encode a molecule
mol = Chem.MolFromSmiles("CCO")  # Ethanol
data = mol_to_data(mol)
data.batch = torch.zeros(data.x.size(0), dtype=torch.long)

output = hypernet.forward(data, normalize=True)
graph_embedding = output["graph_embedding"]  # [1, 256]
edge_terms = output["edge_terms"]  # [1, 256]
```

### Train a Flow Model

```bash
# Uses research hyperparameters by default (800 epochs, etc.)
python experiments/scripts/train_flow.py --dataset qm9
```

### Generate Molecules

```bash
python experiments/scripts/evaluate_generation.py \
    --checkpoint experiments/results/train_flow/<run_dir>/models/best-*.ckpt \
    --dataset qm9 \
    --n_samples 1000
```

### Property Optimization

```bash
# QED Maximization
python experiments/scripts/run_optimization.py \
    --mode qed_max \
    --flow_ckpt <path_to_flow_checkpoint> \
    --regressor_ckpt <path_to_qed_regressor_checkpoint> \
    --dataset zinc

# LogP Targeting
python experiments/scripts/run_optimization.py \
    --mode logp_target \
    --target 2.0 \
    --flow_ckpt <path_to_flow_checkpoint> \
    --regressor_ckpt <path_to_logp_regressor_checkpoint> \
    --dataset qm9
```

---

## Data

The repository includes pre-processed SMILES datasets in `data/`. Raw SMILES files are provided; PyG-format tensors are generated automatically on first use.

| Dataset | Molecules | Atoms (avg) | Raw Data | Processing Module |
|---------|-----------|-------------|----------|-------------------|
| **QM9** | 133,885 | ≤9 | [`data/QM9Smiles/raw/`](data/QM9Smiles/raw/) | [`graph_hdc/datasets/qm9_smiles.py`](graph_hdc/datasets/qm9_smiles.py) |
| **ZINC** | 249,455 | 23.2 | [`data/ZincSmiles/raw/`](data/ZincSmiles/raw/) | [`graph_hdc/datasets/zinc_smiles.py`](graph_hdc/datasets/zinc_smiles.py) |

Each dataset contains `train_smile.txt`, `valid_smile.txt`, and `test_smile.txt` files. The processing modules convert SMILES strings to PyG `Data` objects with node features (atom type, degree, formal charge, total hydrogens, and ring membership for ZINC) and compute LogP/QED properties.

---

## Reproducing Results

**Training is required.** This repository does not provide pre-trained checkpoints.

**All scripts use research hyperparameters by default** when only `--dataset` is specified. The exact configurations are documented in [`experiments/configs/`](experiments/configs/).

### Step 1: Train Flow Models

```bash
# QM9 (~4 hours on GPU, 800 epochs)
python experiments/scripts/train_flow.py --dataset qm9

# ZINC (~6 hours on GPU, 800 epochs)
python experiments/scripts/train_flow.py --dataset zinc
```

Checkpoints are saved to `experiments/results/train_flow/<run_name>/models/`.

### Step 2: Train Property Regressors

```bash
# QM9 LogP regressor (~30 min, 200 epochs)
python experiments/scripts/train_regressor.py --dataset qm9 --property logp

# QM9 QED regressor (~30 min, 200 epochs)
python experiments/scripts/train_regressor.py --dataset qm9 --property qed

# ZINC QED regressor (~45 min, 200 epochs)
python experiments/scripts/train_regressor.py --dataset zinc --property qed

# ZINC LogP regressor (~45 min, 200 epochs)
python experiments/scripts/train_regressor.py --dataset zinc --property logp
```

Checkpoints are saved to `experiments/results/train_regressor/<run_name>/models/`.

### Step 3: Evaluate Generation

```bash
python experiments/scripts/evaluate_generation.py \
    --checkpoint experiments/results/train_flow/<run_name>/models/best-*.ckpt \
    --dataset qm9 \
    --n_samples 10000
```

### Step 4: Run Property Optimization

```bash
# QED Maximization on ZINC
python experiments/scripts/run_optimization.py \
    --mode qed_max \
    --flow_ckpt experiments/results/train_flow/<zinc_run>/models/best-*.ckpt \
    --regressor_ckpt experiments/results/train_regressor/<zinc_qed_run>/models/best-*.ckpt \
    --dataset zinc

# LogP Targeting on QM9
python experiments/scripts/run_optimization.py \
    --mode logp_target \
    --target 2.0 \
    --flow_ckpt experiments/results/train_flow/<qm9_run>/models/best-*.ckpt \
    --regressor_ckpt experiments/results/train_regressor/<qm9_logp_run>/models/best-*.ckpt \
    --dataset qm9
```

### Step 5: Run Retrieval Experiment

```bash
python experiments/scripts/run_retrieval_experiment.py \
    --dataset qm9 \
    --n_samples 1000 \
    --decoder greedy \
    --beam_size 64
```

---

## Experiments

### Experiment 1: Reconstruction Accuracy

**Objective:** Establish encoder-decoder reconstruction accuracy as theoretical validity ceiling.

**Protocol:**
- Sample 1,000 molecules stratified by node count from training sets
- Test dimensions D ∈ {256, 512, 1024} and message passing depths K ∈ {2, 3, 4, 5}
- Two decoder algorithms: greedy beam search, stochastic pattern matching
- Success measured by graph isomorphism testing

**Results:**

| Dataset | Algorithm | Best Accuracy | Configuration |
|---------|-----------|---------------|---------------|
| QM9 | Pattern Matching | **99.6%** | D=256, K=3 |
| QM9 | Greedy Beam Search | **99.3%** | D=256, K=4 |
| ZINC | Pattern Matching (I=50) | **61.1%** | D=256, K=4 |
| ZINC | Greedy Beam Search (B=96) | **50.4%** | D=256, K=4 |

**Key Finding:** Near-perfect reconstruction on small molecules; scalability challenges on larger molecules bounded by decoder algorithms, not encoding capacity.

---

### Experiment 2: Unconditional Generation

**Objective:** Validate that normalizing flows can learn HDC hypervector manifold for de novo generation.

**Protocol:**
- Train RealNVP normalizing flow on encoded training molecules
- Generate 10,000 valid molecules with 3 random seeds
- Compare against VAE, flow, and diffusion baselines

**Results on QM9:**

| Model | Validity | Uniqueness | Novelty | FCD↑ | KL↑ |
|-------|----------|------------|---------|------|-----|
| MoFlow [[1]](#references) | 100% | 99.2% | 98.0% | 0.40 | — |
| DiGress [[2]](#references) | 99.0% | 96.2% | 33.4% | 0.98 | — |
| **GraphHDC** | **98.0%** | **100%** | **80.9%** | **0.70** | **0.86** |

**Results on ZINC:**

| Model | Validity | Uniqueness | Novelty | FCD↑ | KL↑ |
|-------|----------|------------|---------|------|-----|
| MoFlow [[1]](#references) | 100% | 99.99% | 100% | — | — |
| DiGress [[2]](#references) | 85.2% | 100% | 99.9% | — | — |
| **GraphHDC** | **99.6%** | **100%** | **100%** | **0.014** | **0.617** |

**Key Finding:** High validity achieved on both datasets. Distributional fidelity competitive on QM9; limited on ZINC due to ring topology shift (6-membered rings: 90.4% → 8.3%) and aromaticity loss (93.4% → 9.5%).

---

### Experiment 3: QED Maximization

**Objective:** Test property optimization through latent space navigation.

**Protocol:**
- Bayesian hyperparameter optimization: 100 trials × 100 samples
- Generate 10,000 molecules with optimized hyperparameters
- Target: maximize Quantitative Estimate of Drug-likeness (QED)

**Results:**

| Model | Top-1 QED | Top-10 Mean | Top-100 Mean | Novelty | Diversity |
|-------|-----------|-------------|--------------|---------|-----------|
| Graph-GA [[3]](#references) | **0.948** | — | — | — | — |
| MoFlow [[1]](#references) | **0.948** | — | — | — | — |
| **GraphHDC** | **0.948** | **0.940** | **0.918** | **100%** | **0.913** |

**Key Finding:** Reaches 0.948 benchmark ceiling—the empirical QED maximum in ZINC. Uniquely achieves 100% novelty with high diversity, indicating exploration of diverse high-QED scaffolds rather than training set retrieval.

---

### Experiment 4: LogP Targeting

**Objective:** Demonstrate precise property control for specific target values.

**Protocol:**
- Target three LogP values: -1.0 (hydrophilic), 0.5 (mean), 2.0 (lipophilic)
- Bayesian HPO: 100 trials × 100 samples per target
- Generate 10,000 samples per target with ε-filter (ε = 0.193)

**Results:**

| Target LogP | Pass Rate | Full MAD | Top-100 MAD | Validity |
|-------------|-----------|----------|-------------|----------|
| -1.0 | 26.7% | 0.684 | 0.024 | 100% |
| 0.5 | 87.7% | 0.294 | **0.010** | 100% |
| 2.0 | 77.6% | 0.429 | 0.011 | 100% |
| **Aggregate** | **63.9%** | **0.469** | **0.015** | **100%** |

**Key Finding:** Top-100 MAD of 0.015 demonstrates molecules can be generated within 0.015 LogP units of any target—precision suitable for lead optimization workflows.

---

### Experiment 5: Ablation Studies

**Objective:** Validate design decisions and identify performance-critical hyperparameters.

**Key Findings:**

1. **Hypervector Dimension:**
   - Capacity saturation at D ≥ 256 (QM9: D ≥ 196)
   - Higher dimensions degrade generation quality (inverse correlation)
   - Recommendation: D=256 for both encoding fidelity and flow learning

2. **Message Passing Depth:**
   - Optimal: K=3 for QM9, K=4 for ZINC
   - K=5 causes catastrophic over-smoothing (82.9–83.8% on QM9)

3. **Flow Architecture:**
   - Flow Matching achieves best FCD/KL but incompatible with regressor-based conditioning
   - Neural Spline achieves highest novelty (97.0%) but 4–6× training cost
   - RealNVP selected for balance of quality, efficiency, and conditioning compatibility

---

## Highlighted Findings

### What Works

- **High validity on both datasets:** 98.0% (QM9), 99.6% (ZINC)
- **Perfect uniqueness and high novelty:** No mode collapse
- **Competitive distributional fidelity on small molecules:** FCD 0.70, KL 0.86 on QM9
- **Benchmark-ceiling property optimization:** QED 0.948 with 100% novelty
- **Precise property targeting:** Top-100 MAD of 0.015 for LogP
- **Training-free encoder-decoder:** Deterministic algebraic operations, no learned parameters

### Limitations

- **Distributional fidelity on larger molecules:** FCD 0.014 on ZINC stems from ring topology shift (46% contribution: 6-membered rings collapse from 90.4% to 8.3%, macrocycles explode from 0.02% to 47.9%) and aromaticity loss (40% contribution: 93.4% → 9.5%). Generated molecules exhibit near-uniform ring size distribution (uniformity ratio 0.89) versus training's concentration at 6-membered rings.
- **Decoder computational cost:** 15.58s per molecule for high-accuracy ZINC reconstruction
- **Position-dependent targeting precision:** Tail targets harder than near-mean targets

### Future Directions

- Aromaticity and ring topology-aware encoding extensions
- Ring size distribution priors in decoder algorithms
- Energy-based decoder refinement with HDC similarity
- Flow Matching with classifier-free guidance for conditioning

---

## Research Questions Answered

| RQ | Question | Answer |
|----|----------|--------|
| **RQ1** | Can HDC representations enable reconstruction with sufficient fidelity? | Conditionally yes: 99.6% on QM9, 61.1% on ZINC (bounded by decoder algorithms) |
| **RQ2** | Can decoders produce valid molecules from noisy hypervectors? | Yes: 99.6% validity on ZINC despite 61.1% reconstruction accuracy |
| **RQ3** | Is the HDC manifold learnable for de novo generation? | Yes: Standard flows learn distributions; fidelity depends on encoding design |
| **RQ4** | Does HDC space exhibit semantic structure for property conditioning? | Yes: QED ceiling reached, LogP MAD 0.015, Tanimoto-cosine correlation confirmed |

---

## Project Structure

```
graph_hdc/
├── graph_hdc/                  # Main package
│   ├── datasets/               # QM9 and ZINC SMILES datasets
│   ├── hypernet/               # HDC encoder/decoder
│   │   ├── encoder.py          # HyperNet class
│   │   ├── decoder.py          # Graph decoding
│   │   ├── configs.py          # Dataset configurations
│   │   └── feature_encoders.py # Feature codebooks
│   ├── models/
│   │   ├── flows/              # Real NVP V3 normalizing flow
│   │   └── regressors/         # Property regressors
│   └── utils/                  # Utilities
│
├── experiments/
│   ├── scripts/                # Training and evaluation scripts
│   ├── configs/                # Research hyperparameters (YAML)
│   └── results/                # Training outputs (checkpoints, logs)
│
└── tests/                      # Unit tests
```

---

## Key Components

### HDC Encoding

The encoder uses three main operations:
- **Bind (⊗)**: Circular convolution combines features (node attributes with position)
- **Bundle (+)**: Addition aggregates multiple hypervectors
- **Unbind (⊘)**: Circular correlation retrieves constituents

### Node Features

**QM9 (4 features, bins=[4,5,3,5]):**
- Atom type: C, N, O, F
- Degree: 0-4
- Formal charge: 0, +1, -1
- Total hydrogens: 0-4

**ZINC (5 features, bins=[9,6,3,4,2]):**
- Atom type: Br, C, Cl, F, I, N, O, P, S
- Degree: 0-5
- Formal charge: 0, +1, -1
- Total hydrogens: 0-3
- Is in ring: 0, 1

### Flow Model

Real NVP V3 with:
- Semantic masking (respects edge_terms/graph_terms boundary)
- Per-term standardization
- ActNorm layers
- Scale warmup for training stability

---

## Research Configurations

All hyperparameters were optimized via Bayesian HPO and are stored in [`experiments/configs/`](experiments/configs/). These are the **default values** used by the training scripts.

### Flow Models

| Parameter | QM9 | ZINC | Config File |
|-----------|-----|------|-------------|
| Epochs | 800 | 800 | |
| Batch size | 96 | 224 | |
| Learning rate | 1.91e-4 | 5.39e-4 | |
| Weight decay | 3.49e-4 | 1e-3 | |
| Num flows | 16 | 8 | |
| Hidden dim | 1792 | 1536 | |
| Hidden layers | 3 | 2 | |
| HV dimension | 256 | 256 | |
| **Full config** | | | [`qm9_flow.yaml`](experiments/configs/qm9_flow.yaml), [`zinc_flow.yaml`](experiments/configs/zinc_flow.yaml) |

### Property Regressors

| Parameter | QM9 LogP | QM9 QED | ZINC LogP | ZINC QED |
|-----------|----------|---------|-----------|----------|
| Epochs | 200 | 200 | 200 | 200 |
| Batch size | 192 | 64 | 352 | 224 |
| Learning rate | 8.63e-4 | 6.35e-4 | 8.64e-5 | 5.36e-4 |
| Hidden dims | [256, 256] | [512, 256, 128, 32] | [1536, 896, 128] | [1536, 768, 128] |
| Activation | GELU | SiLU | SiLU | SiLU |
| Dropout | 0.061 | 0.049 | 0.127 | 0.178 |

**Config files:** [`qm9_logp_regressor.yaml`](experiments/configs/qm9_logp_regressor.yaml) | [`qm9_qed_regressor.yaml`](experiments/configs/qm9_qed_regressor.yaml) | [`zinc_logp_regressor.yaml`](experiments/configs/zinc_logp_regressor.yaml) | [`zinc_qed_regressor.yaml`](experiments/configs/zinc_qed_regressor.yaml)

**Note:** To override defaults, pass CLI arguments (e.g., `--epochs 100 --lr 1e-4`). See `--help` for all options.

---

## GPU Support

All scripts automatically use CUDA if available. Override with:

```bash
# Force CPU
python experiments/scripts/train_flow.py --device cpu

# Force specific GPU
python experiments/scripts/train_flow.py --device cuda:0
```

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- PyTorch Geometric
- RDKit
- TorchHD
- Normflows

See `pyproject.toml` for full dependencies.

---

## References

[1] C. Zang and F. Wang, "MoFlow: An Invertible Flow Model for Generating Molecular Graphs," *KDD*, 2020.

[2] C. Vignac et al., "DiGress: Discrete Denoising Diffusion for Graph Generation," *ICLR*, 2023.

[3] J. H. Jensen, "A Graph-Based Genetic Algorithm and Generative Model/Monte Carlo Tree Search for the Exploration of Chemical Space," *Chemical Science*, 2019.

---

## Citation

If you use this code, please cite:

```bibtex
@mastersthesis{kaveh2025graphhdc,
  author = {Kaveh, Arvand},
  title = {Hyperdimensional Representations for Probabilistic Graph Generation},
  school = {Karlsruhe Institute of Technology},
  year = {2025},
  type = {Master's Thesis}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Arvand Kaveh, Jonas Teufel

---

## Acknowledgments

This work was conducted at the Institute of Theoretical Informatics, Karlsruhe Institute of Technology (KIT), under the supervision of Prof. Dr. Pascal Friederich and Prof. Dr. Gerhard Neumann, with advising from M.Sc. Jonas Teufel.