# Experiment Configuration Files

This directory contains YAML configuration files documenting the **research hyperparameters** used in the thesis experiments.

## Important Note

These YAML files are **reference documentation only** and are **not loaded by the training scripts**. The scripts use hardcoded Python dataclass configurations (`QM9_FLOW_CONFIG`, `ZINC_FLOW_CONFIG`, etc.) defined in the source code.

## Purpose

These files serve as:
1. **Human-readable documentation** of the exact hyperparameters used in thesis experiments
2. **Research artifacts** for reproducibility verification
3. **Reference** for users who want to understand the default configurations

## Files

| File | Description |
|------|-------------|
| `qm9_flow.yaml` | QM9 Real NVP flow model hyperparameters |
| `zinc_flow.yaml` | ZINC Real NVP flow model hyperparameters |
| `qm9_logp_regressor.yaml` | QM9 LogP property regressor hyperparameters |
| `qm9_qed_regressor.yaml` | QM9 QED property regressor hyperparameters |
| `zinc_logp_regressor.yaml` | ZINC LogP property regressor hyperparameters |
| `zinc_qed_regressor.yaml` | ZINC QED property regressor hyperparameters |

## Overriding Defaults

To use different hyperparameters, pass CLI arguments to the training scripts:

```bash
# Override learning rate and batch size
python experiments/scripts/train_flow.py --dataset qm9 --lr 1e-4 --batch_size 128

# See all available options
python experiments/scripts/train_flow.py --help
```

## Synchronization

The values in these YAML files are synchronized with the Python configurations in:
- `graph_hdc/models/flows/real_nvp.py` (FlowConfig)
- `graph_hdc/models/regressors/property_regressor.py` (RegressorConfig)

If you notice any discrepancies, the Python code is authoritative.
