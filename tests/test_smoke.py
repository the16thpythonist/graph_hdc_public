"""
Smoke tests for verifying experiment scripts and imports.

These tests verify the scripts can be imported and basic functionality works.
They do NOT verify result correctness - only that the pipeline executes.
"""

import pytest
import torch

# force_cpu fixture is now in conftest.py (autouse=True)


def test_torchhd_patch_applied():
    """Verify TorchHD patch is applied on package import."""
    import inspect

    import graph_hdc  # noqa: F401
    from torchhd import HRRTensor

    # Check that multibind is patched (should have squeeze for identity short-circuit)
    source = inspect.getsource(HRRTensor.multibind)
    assert "squeeze" in source, "TorchHD patch not applied - multibind not patched"


def test_core_imports():
    """Test all core public imports work."""
    from graph_hdc import (
        CorrectionLevel,
        DecoderSettings,
        DecodingResult,
        DSHDCConfig,
        FallbackDecoderSettings,
        GenerationEvaluator,
        HyperNet,
        calculate_internal_diversity,
        get_config,
        get_split,
        post_compute_encodings,
        rdkit_logp,
        rdkit_qed,
        rdkit_sa_score,
    )

    assert HyperNet is not None
    assert get_config is not None
    assert GenerationEvaluator is not None
    assert CorrectionLevel is not None
    assert DecodingResult is not None
    assert DSHDCConfig is not None
    assert DecoderSettings is not None
    assert FallbackDecoderSettings is not None
    assert get_split is not None
    assert post_compute_encodings is not None
    assert rdkit_logp is not None
    assert rdkit_qed is not None
    assert rdkit_sa_score is not None
    assert calculate_internal_diversity is not None


def test_model_imports():
    """Test model imports work."""
    from graph_hdc.models.flows import (
        FlowConfig,
        QM9_FLOW_CONFIG,
        RealNVPV3Lightning,
        ZINC_FLOW_CONFIG,
    )
    from graph_hdc.models.regressors import (
        MolecularProperty,
        PropertyRegressor,
        QM9_LOGP_CONFIG,
        QM9_QED_CONFIG,
        RegressorConfig,
        ZINC_LOGP_CONFIG,
        ZINC_QED_CONFIG,
    )

    assert RealNVPV3Lightning is not None
    assert FlowConfig is not None
    assert QM9_FLOW_CONFIG is not None
    assert ZINC_FLOW_CONFIG is not None
    assert PropertyRegressor is not None
    assert RegressorConfig is not None
    assert MolecularProperty is not None
    assert QM9_LOGP_CONFIG is not None
    assert QM9_QED_CONFIG is not None
    assert ZINC_LOGP_CONFIG is not None
    assert ZINC_QED_CONFIG is not None


def test_dataset_imports():
    """Test dataset imports work."""
    from graph_hdc.datasets import (
        DatasetInfo,
        QM9Smiles,
        ZincSmiles,
        get_dataset_info,
        get_split,
        post_compute_encodings,
    )

    assert QM9Smiles is not None
    assert ZincSmiles is not None
    assert get_split is not None
    assert get_dataset_info is not None
    assert DatasetInfo is not None
    assert post_compute_encodings is not None


def test_hypernet_imports():
    """Test hypernet imports work."""
    from graph_hdc.hypernet import (
        CorrectionLevel,
        DecoderSettings,
        DecodingResult,
        DSHDCConfig,
        FallbackDecoderSettings,
        Feat,
        Features,
        HyperNet,
        VSAModel,
        get_config,
    )

    assert HyperNet is not None
    assert CorrectionLevel is not None
    assert DecodingResult is not None
    assert get_config is not None
    assert DSHDCConfig is not None
    assert DecoderSettings is not None
    assert FallbackDecoderSettings is not None
    assert Features is not None
    assert Feat is not None
    assert VSAModel is not None


def test_utils_imports():
    """Test utils imports work."""
    from graph_hdc.utils import (
        DataTransformer,
        GenerationEvaluator,
        TupleIndexer,
        calculate_internal_diversity,
        canonical_key,
        is_valid_molecule,
        pick_device,
        rdkit_logp,
        rdkit_qed,
        rdkit_sa_score,
        reconstruct_for_eval,
    )

    assert GenerationEvaluator is not None
    assert rdkit_logp is not None
    assert rdkit_qed is not None
    assert rdkit_sa_score is not None
    assert calculate_internal_diversity is not None
    assert DataTransformer is not None
    assert TupleIndexer is not None
    assert pick_device is not None
    assert is_valid_molecule is not None
    assert canonical_key is not None
    assert reconstruct_for_eval is not None


def test_config_loading():
    """Test config loading for both datasets."""
    from graph_hdc import get_config

    qm9_config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
    assert qm9_config.base_dataset == "qm9"
    assert qm9_config.hv_dim == 256

    zinc_config = get_config("ZINC_SMILES_HRR_256_F64_5G1NG4")
    assert zinc_config.base_dataset == "zinc"
    assert zinc_config.hv_dim == 256


def test_hypernet_creation():
    """Test HyperNet can be created for QM9 dataset."""
    from graph_hdc import HyperNet, get_config

    qm9_config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
    qm9_config.device = "cpu"
    qm9_hypernet = HyperNet(qm9_config)
    assert qm9_hypernet is not None
    assert qm9_hypernet.hv_dim == 256


def test_flow_model_creation():
    """Test flow model can be created."""
    from graph_hdc.models.flows import FlowConfig, RealNVPV3Lightning

    config = FlowConfig(hv_dim=256, num_flows=2, hidden_dim=64, num_hidden_layers=1)
    model = RealNVPV3Lightning(config)
    assert model is not None
    assert model.D == 256


def test_regressor_model_creation():
    """Test regressor model can be created."""
    from graph_hdc.models.regressors import PropertyRegressor

    model = PropertyRegressor(
        input_dim=256,
        hidden_dims=(64, 32),
        dropout=0.0,
        target_property="logp",
    )
    assert model is not None

    # Test forward pass
    x = torch.randn(4, 256)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (4,) or output.shape == (4, 1)


def test_evaluator_creation():
    """Test evaluator can be created for QM9 dataset."""
    from graph_hdc import GenerationEvaluator

    qm9_eval = GenerationEvaluator(base_dataset="qm9")
    assert qm9_eval is not None
    assert len(qm9_eval.T) > 0  # Should have training smiles


@pytest.mark.slow
def test_flow_sample():
    """Test flow model can sample (slow test)."""
    from graph_hdc.models.flows import FlowConfig, RealNVPV3Lightning

    config = FlowConfig(hv_dim=64, num_flows=2, hidden_dim=32, num_hidden_layers=1)
    model = RealNVPV3Lightning(config)
    model.eval()

    # Set standardization (required for sampling)
    model.set_standardization(
        torch.zeros(config.hv_dim * 2),
        torch.ones(config.hv_dim * 2),
    )

    with torch.no_grad():
        result = model.sample_split(num_samples=2)

    assert "edge_terms" in result
    assert "graph_terms" in result
    assert result["edge_terms"].shape == (2, 64)
    assert result["graph_terms"].shape == (2, 64)
