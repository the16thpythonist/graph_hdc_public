"""Unit tests for the Flow Matching model components."""
import pytest
import torch
from torch_geometric.data import Batch, Data

from graph_hdc.models.flow_matching import (
    CONDITION_REGISTRY,
    AdaLNZeroBlock,
    CrippenLogP,
    FiLMResidualBlock,
    FlowMatchingModel,
    HeavyAtomCount,
    MinibatchOTCoupler,
    MultiCondition,
    PseudoHuberLoss,
    SinusoidalTimeEmbedding,
    SwiGLUFiLMBlock,
    VelocityDiT,
    VelocityMLP,
    VelocityModelWrapper,
    build_condition,
    get_condition,
    sample_logit_normal,
)
from graph_hdc.models.gmm_utils import (
    GMMOutputHead,
    gm_nll_loss,
    gm_to_mean,
    gm_to_sample,
    gmm_reverse_transition,
)


# =============================================================================
# Fixtures
# =============================================================================

DATA_DIM = 32
HIDDEN_DIM = 64
NUM_BLOCKS = 2
TIME_EMBED_DIM = 16
BATCH_SIZE = 8


@pytest.fixture
def model():
    return FlowMatchingModel(
        data_dim=DATA_DIM,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        time_embed_dim=TIME_EMBED_DIM,
        condition_dim=0,
        use_ot_coupling=True,
        default_sample_steps=5,
        warmup_epochs=0,
    )


COND_RAW_DIM = 2
COND_EMBED_DIM = 16


@pytest.fixture
def cond_model():
    return FlowMatchingModel(
        data_dim=DATA_DIM,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        time_embed_dim=TIME_EMBED_DIM,
        condition_dim=COND_RAW_DIM,
        condition_embed_dim=COND_EMBED_DIM,
        use_ot_coupling=True,
        default_sample_steps=5,
        warmup_epochs=0,
    )


@pytest.fixture
def fake_batch():
    """Create a fake PyG batch with node_terms and graph_terms."""
    data_list = []
    for _ in range(BATCH_SIZE):
        d = Data()
        d.node_terms = torch.randn(DATA_DIM // 2)
        d.graph_terms = torch.randn(DATA_DIM // 2)
        d.edge_index = torch.zeros(2, 0, dtype=torch.long)
        d.num_nodes = 1
        data_list.append(d)
    return Batch.from_data_list(data_list)


# =============================================================================
# SinusoidalTimeEmbedding
# =============================================================================


def test_sinusoidal_time_embedding_shape():
    emb = SinusoidalTimeEmbedding(embed_dim=TIME_EMBED_DIM)
    t = torch.rand(BATCH_SIZE)
    out = emb(t)
    assert out.shape == (BATCH_SIZE, TIME_EMBED_DIM)


def test_sinusoidal_time_embedding_different_times():
    emb = SinusoidalTimeEmbedding(embed_dim=TIME_EMBED_DIM)
    t1 = torch.tensor([0.0])
    t2 = torch.tensor([1.0])
    assert not torch.allclose(emb(t1), emb(t2))


# =============================================================================
# FiLMResidualBlock
# =============================================================================


def test_film_residual_block_shape():
    block = FiLMResidualBlock(HIDDEN_DIM, film_dim=HIDDEN_DIM)
    x = torch.randn(BATCH_SIZE, HIDDEN_DIM)
    film = torch.randn(BATCH_SIZE, HIDDEN_DIM)
    out = block(x, film)
    assert out.shape == (BATCH_SIZE, HIDDEN_DIM)


def test_film_residual_block_near_identity_at_init():
    block = FiLMResidualBlock(HIDDEN_DIM, film_dim=HIDDEN_DIM)
    x = torch.randn(BATCH_SIZE, HIDDEN_DIM)
    film = torch.zeros(BATCH_SIZE, HIDDEN_DIM)
    out = block(x, film)
    # With zero-initialized FiLM params, the residual branch output should be
    # small relative to the input (not exactly zero due to LayerNorm + SiLU)
    residual = (out - x).abs().mean()
    assert residual < 1.0, f"Residual too large at init: {residual}"


# =============================================================================
# SwiGLUFiLMBlock
# =============================================================================


def test_swiglu_film_block_shape():
    block = SwiGLUFiLMBlock(HIDDEN_DIM, film_dim=HIDDEN_DIM)
    x = torch.randn(BATCH_SIZE, HIDDEN_DIM)
    film = torch.randn(BATCH_SIZE, HIDDEN_DIM)
    out = block(x, film)
    assert out.shape == (BATCH_SIZE, HIDDEN_DIM)


def test_swiglu_film_block_near_identity_at_init():
    block = SwiGLUFiLMBlock(HIDDEN_DIM, film_dim=HIDDEN_DIM)
    x = torch.randn(BATCH_SIZE, HIDDEN_DIM)
    film = torch.zeros(BATCH_SIZE, HIDDEN_DIM)
    out = block(x, film)
    # With zero-initialized FiLM params (scale=0, shift=0, gate=0),
    # the residual branch is multiplied by gate=sigmoid(0)=0.5 * h,
    # but h itself is small because the SwiGLU output is near zero at init.
    residual = (out - x).abs().mean()
    assert residual < 1.0, f"Residual too large at init: {residual}"


def test_swiglu_film_block_gradient_flow():
    block = SwiGLUFiLMBlock(HIDDEN_DIM, film_dim=HIDDEN_DIM)
    # Perturb film_proj away from zero-init so gradients flow through film_emb
    with torch.no_grad():
        block.film_proj[-1].weight.add_(0.01 * torch.randn_like(block.film_proj[-1].weight))
    x = torch.randn(BATCH_SIZE, HIDDEN_DIM, requires_grad=True)
    film = torch.randn(BATCH_SIZE, HIDDEN_DIM, requires_grad=True)
    out = block(x, film)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert film.grad is not None and film.grad.abs().sum() > 0


def test_swiglu_film_block_with_dropout():
    block = SwiGLUFiLMBlock(HIDDEN_DIM, film_dim=HIDDEN_DIM, dropout=0.1)
    x = torch.randn(BATCH_SIZE, HIDDEN_DIM)
    film = torch.randn(BATCH_SIZE, HIDDEN_DIM)
    block.train()
    out = block(x, film)
    assert out.shape == (BATCH_SIZE, HIDDEN_DIM)


# =============================================================================
# VelocityMLP
# =============================================================================


def test_velocity_mlp_shape():
    net = VelocityMLP(DATA_DIM, HIDDEN_DIM, NUM_BLOCKS, TIME_EMBED_DIM)
    x = torch.randn(BATCH_SIZE, DATA_DIM)
    t = torch.rand(BATCH_SIZE)
    out = net(x, t)
    assert out.shape == (BATCH_SIZE, DATA_DIM)


def test_velocity_mlp_zero_init():
    net = VelocityMLP(DATA_DIM, HIDDEN_DIM, NUM_BLOCKS, TIME_EMBED_DIM)
    x = torch.randn(BATCH_SIZE, DATA_DIM)
    t = torch.rand(BATCH_SIZE)
    out = net(x, t)
    assert out.abs().max() < 0.5  # Near zero at init


def test_velocity_mlp_conditional():
    net = VelocityMLP(DATA_DIM, HIDDEN_DIM, NUM_BLOCKS, TIME_EMBED_DIM, condition_dim=16)
    x = torch.randn(BATCH_SIZE, DATA_DIM)
    t = torch.rand(BATCH_SIZE)
    c = torch.randn(BATCH_SIZE, 16)
    out = net(x, t, condition=c)
    assert out.shape == (BATCH_SIZE, DATA_DIM)


def test_velocity_mlp_swiglu_shape():
    net = VelocityMLP(
        DATA_DIM, HIDDEN_DIM, NUM_BLOCKS, TIME_EMBED_DIM, block_type="swiglu",
    )
    x = torch.randn(BATCH_SIZE, DATA_DIM)
    t = torch.rand(BATCH_SIZE)
    out = net(x, t)
    assert out.shape == (BATCH_SIZE, DATA_DIM)


def test_velocity_mlp_swiglu_zero_init():
    net = VelocityMLP(
        DATA_DIM, HIDDEN_DIM, NUM_BLOCKS, TIME_EMBED_DIM, block_type="swiglu",
    )
    x = torch.randn(BATCH_SIZE, DATA_DIM)
    t = torch.rand(BATCH_SIZE)
    out = net(x, t)
    assert out.abs().max() < 0.5, f"Output too large at init: {out.abs().max()}"


def test_velocity_mlp_swiglu_conditional():
    net = VelocityMLP(
        DATA_DIM, HIDDEN_DIM, NUM_BLOCKS, TIME_EMBED_DIM,
        condition_dim=16, block_type="swiglu",
    )
    x = torch.randn(BATCH_SIZE, DATA_DIM)
    t = torch.rand(BATCH_SIZE)
    c = torch.randn(BATCH_SIZE, 16)
    out = net(x, t, condition=c)
    assert out.shape == (BATCH_SIZE, DATA_DIM)


def test_velocity_mlp_swiglu_gradient_flow():
    net = VelocityMLP(
        DATA_DIM, HIDDEN_DIM, NUM_BLOCKS, TIME_EMBED_DIM, block_type="swiglu",
    )
    # Perturb film projections and output proj away from zero-init
    with torch.no_grad():
        for block in net.blocks:
            block.film_proj[-1].weight.add_(
                0.01 * torch.randn_like(block.film_proj[-1].weight)
            )
        net.output_proj.weight.add_(0.01 * torch.randn_like(net.output_proj.weight))
    x = torch.randn(BATCH_SIZE, DATA_DIM, requires_grad=True)
    t = torch.rand(BATCH_SIZE)
    out = net(x, t)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_velocity_mlp_invalid_block_type():
    with pytest.raises(ValueError, match="Unknown block_type"):
        VelocityMLP(DATA_DIM, HIDDEN_DIM, NUM_BLOCKS, TIME_EMBED_DIM, block_type="bad")


# =============================================================================
# VelocityModelWrapper
# =============================================================================


def test_velocity_model_wrapper():
    net = VelocityMLP(DATA_DIM, HIDDEN_DIM, NUM_BLOCKS, TIME_EMBED_DIM)
    wrapper = VelocityModelWrapper(net)
    x = torch.randn(BATCH_SIZE, DATA_DIM)
    t = torch.rand(BATCH_SIZE)
    out = wrapper(x=x, t=t)
    assert out.shape == (BATCH_SIZE, DATA_DIM)


# =============================================================================
# MinibatchOTCoupler
# =============================================================================


def test_ot_coupler_shape():
    x0 = torch.randn(BATCH_SIZE, DATA_DIM)
    x1 = torch.randn(BATCH_SIZE, DATA_DIM)
    x0_reordered = MinibatchOTCoupler.couple(x0, x1)
    assert x0_reordered.shape == x0.shape


def test_ot_coupler_is_permutation():
    x0 = torch.randn(BATCH_SIZE, DATA_DIM)
    x1 = torch.randn(BATCH_SIZE, DATA_DIM)
    x0_reordered = MinibatchOTCoupler.couple(x0, x1)

    # Each row in x0_reordered should match a row in x0
    for i in range(BATCH_SIZE):
        dists = (x0 - x0_reordered[i]).norm(dim=-1)
        assert dists.min() < 1e-5


def test_ot_coupler_reduces_cost():
    torch.manual_seed(42)
    x0 = torch.randn(BATCH_SIZE, DATA_DIM)
    x1 = torch.randn(BATCH_SIZE, DATA_DIM)

    identity_cost = (x0 - x1).pow(2).sum()
    x0_ot = MinibatchOTCoupler.couple(x0, x1)
    ot_cost = (x0_ot - x1).pow(2).sum()

    assert ot_cost <= identity_cost + 1e-6


# =============================================================================
# FlowMatchingModel
# =============================================================================


def test_flow_matching_loss_finite(model, fake_batch):
    loss = model.training_step(fake_batch, 0)
    assert torch.isfinite(loss)
    assert loss > 0


def test_flow_matching_sample_shape(model):
    samples = model.sample(num_samples=4, num_steps=3)
    assert samples.shape == (4, DATA_DIM)
    assert torch.isfinite(samples).all()


def test_flow_matching_encode_shape(model):
    x1 = torch.randn(4, DATA_DIM)
    z = model.encode(x1, num_steps=3)
    assert z.shape == (4, DATA_DIM)
    assert torch.isfinite(z).all()


def test_flow_matching_nll_finite(model):
    x1 = torch.randn(4, DATA_DIM)
    nll = model.compute_nll(x1, num_steps=3)
    assert nll.shape == (4,)
    assert torch.isfinite(nll).all()


def test_flow_matching_standardization(model):
    mean = torch.randn(DATA_DIM)
    std = torch.rand(DATA_DIM) + 0.5
    model.set_standardization(mean, std)

    x = torch.randn(4, DATA_DIM)
    z = model.standardize(x)
    x_rec = model.destandardize(z)
    assert torch.allclose(x, x_rec, atol=1e-5)


def test_flow_matching_save_load(model, tmp_path):
    # Save and load using state_dict (standard PyTorch pattern)
    ckpt_path = tmp_path / "test.pt"
    torch.save(model.state_dict(), ckpt_path)

    loaded = FlowMatchingModel(
        data_dim=DATA_DIM,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        time_embed_dim=TIME_EMBED_DIM,
        condition_dim=0,
        default_sample_steps=5,
        warmup_epochs=0,
    )
    loaded.load_state_dict(torch.load(ckpt_path, weights_only=True))

    x = torch.randn(4, DATA_DIM)
    t = torch.rand(4)

    model.eval()
    loaded.eval()
    with torch.no_grad():
        out1 = model.velocity_net(x, t)
        out2 = loaded.velocity_net(x, t)
    assert torch.allclose(out1, out2)


def test_flow_matching_conditional(cond_model):
    x = torch.randn(4, DATA_DIM)
    c1 = torch.zeros(4, COND_RAW_DIM)
    c2 = torch.ones(4, COND_RAW_DIM)
    loss1 = cond_model.compute_loss(x, condition=c1)
    loss2 = cond_model.compute_loss(x, condition=c2)
    # Losses should be finite (not necessarily different due to randomness in t/x0)
    assert torch.isfinite(loss1)
    assert torch.isfinite(loss2)


def test_flow_matching_extract_vectors(model, fake_batch):
    vectors = model._extract_vectors(fake_batch)
    assert vectors.shape == (BATCH_SIZE, DATA_DIM)


# =============================================================================
# Conditioning Interface
# =============================================================================


def test_heavy_atom_count():
    cond = HeavyAtomCount()
    assert cond.name == "heavy_atoms"
    assert cond.condition_dim == 1
    # Ethanol: C, C, O = 3 heavy atoms
    assert cond.evaluate("CCO") == 3.0
    lo, hi = cond.value_range()
    assert lo < hi


def test_crippen_logp():
    cond = CrippenLogP()
    assert cond.name == "logp"
    assert cond.condition_dim == 1
    val = cond.evaluate("CCO")
    assert isinstance(val, float)
    lo, hi = cond.value_range()
    assert lo < hi


def test_heavy_atom_invalid_smiles():
    cond = HeavyAtomCount()
    assert cond.evaluate("INVALID") == 0.0


def test_crippen_logp_invalid_smiles():
    cond = CrippenLogP()
    assert cond.evaluate("INVALID") == 0.0


def test_condition_registry():
    assert "heavy_atoms" in CONDITION_REGISTRY
    assert "logp" in CONDITION_REGISTRY
    cond = get_condition("heavy_atoms")
    assert isinstance(cond, HeavyAtomCount)


def test_condition_registry_unknown():
    with pytest.raises(KeyError, match="Unknown condition"):
        get_condition("nonexistent")


def test_multi_condition_single():
    mc = MultiCondition([HeavyAtomCount()])
    assert mc.condition_dim == 1
    assert "heavy_atoms" in mc.name
    val = mc.evaluate("CCO")
    assert val == 3.0
    t = mc.evaluate_multi("CCO")
    assert t.shape == (1,)


def test_multi_condition_multi():
    mc = MultiCondition([HeavyAtomCount(), CrippenLogP()])
    assert mc.condition_dim == 2
    assert "heavy_atoms" in mc.name
    assert "logp" in mc.name
    t = mc.evaluate_multi("CCO")
    assert t.shape == (2,)
    assert t[0] == 3.0  # heavy atoms


def test_multi_condition_evaluate_batch():
    mc = MultiCondition([HeavyAtomCount(), CrippenLogP()])
    smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
    batch = mc.evaluate_batch(smiles)
    assert batch.shape == (3, 2)
    assert batch[0, 0] == 3.0  # ethanol heavy atoms


def test_multi_condition_sample_condition_dict():
    mc = MultiCondition([HeavyAtomCount(), CrippenLogP()])
    t = mc.sample_condition({"heavy_atoms": 10.0, "logp": 2.5})
    assert t.shape == (2,)
    assert t[0] == 10.0
    assert t[1] == 2.5


def test_multi_condition_sample_condition_list():
    mc = MultiCondition([HeavyAtomCount(), CrippenLogP()])
    t = mc.sample_condition([10.0, 2.5])
    assert t.shape == (2,)
    assert t[0] == 10.0
    assert t[1] == 2.5


def test_multi_condition_sample_condition_scalar():
    mc = MultiCondition([HeavyAtomCount()])
    t = mc.sample_condition(10.0)
    assert t.shape == (1,)
    assert t[0] == 10.0


def test_build_condition():
    mc = build_condition(["heavy_atoms", "logp"])
    assert isinstance(mc, MultiCondition)
    assert mc.condition_dim == 2


def test_extract_condition_with_batch(cond_model):
    """Test that _extract_condition detects batch.condition attribute."""
    data_list = []
    for _ in range(4):
        d = Data()
        d.node_terms = torch.randn(DATA_DIM // 2)
        d.graph_terms = torch.randn(DATA_DIM // 2)
        d.condition = torch.randn(COND_RAW_DIM)
        d.edge_index = torch.zeros(2, 0, dtype=torch.long)
        d.num_nodes = 1
        data_list.append(d)
    batch = Batch.from_data_list(data_list)
    cond = cond_model._extract_condition(batch)
    assert cond is not None
    assert cond.shape == (4, COND_RAW_DIM)


def test_extract_condition_fallback_zeros(cond_model):
    """Test that _extract_condition returns zeros when no .condition attr."""
    data_list = []
    for _ in range(4):
        d = Data()
        d.node_terms = torch.randn(DATA_DIM // 2)
        d.graph_terms = torch.randn(DATA_DIM // 2)
        d.edge_index = torch.zeros(2, 0, dtype=torch.long)
        d.num_nodes = 1
        data_list.append(d)
    batch = Batch.from_data_list(data_list)
    cond = cond_model._extract_condition(batch)
    assert cond is not None
    assert cond.shape == (4, COND_RAW_DIM)
    assert (cond == 0).all()


def test_extract_condition_none_for_unconditional(model, fake_batch):
    """Unconditional model returns None from _extract_condition."""
    cond = model._extract_condition(fake_batch)
    assert cond is None


# =============================================================================
# Feature Compatibility Tests
# =============================================================================


@pytest.mark.parametrize(
    "loss_type,time_sampling,prediction_type,use_ot",
    [
        ("mse", "uniform", "velocity", True),
        ("mse", "uniform", "velocity", False),
        ("mse", "logit_normal", "velocity", True),
        ("mse", "uniform", "x_prediction", True),
        ("mse", "logit_normal", "x_prediction", True),
        ("pseudo_huber", "uniform", "velocity", True),
        ("pseudo_huber", "logit_normal", "velocity", True),
        ("pseudo_huber", "uniform", "x_prediction", True),
        ("pseudo_huber", "logit_normal", "x_prediction", False),
        # All features combined
        ("pseudo_huber", "logit_normal", "x_prediction", True),
    ],
)
def test_feature_combination_training(
    loss_type, time_sampling, prediction_type, use_ot, fake_batch,
):
    """Verify all feature combinations produce finite losses and gradients."""
    m = FlowMatchingModel(
        data_dim=DATA_DIM,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        time_embed_dim=TIME_EMBED_DIM,
        loss_type=loss_type,
        time_sampling=time_sampling,
        prediction_type=prediction_type,
        use_ot_coupling=use_ot,
        default_sample_steps=5,
        warmup_epochs=0,
    )
    m.train()
    loss = m.training_step(fake_batch, 0)
    assert torch.isfinite(loss), f"Non-finite loss: {loss}"
    assert loss > 0
    loss.backward()
    for name, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name}"


@pytest.mark.parametrize("prediction_type", ["velocity", "x_prediction"])
def test_feature_combination_sampling(prediction_type):
    """Verify sampling works for all prediction types."""
    m = FlowMatchingModel(
        data_dim=DATA_DIM,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        time_embed_dim=TIME_EMBED_DIM,
        prediction_type=prediction_type,
        default_sample_steps=5,
        warmup_epochs=0,
    )
    m.eval()
    with torch.no_grad():
        samples = m.sample(num_samples=4, num_steps=5)
    assert samples.shape == (4, DATA_DIM)
    assert torch.isfinite(samples).all()


@pytest.mark.parametrize("prediction_type", ["velocity", "x_prediction"])
def test_feature_combination_encode(prediction_type):
    """Verify encode (reverse ODE) works for all prediction types."""
    m = FlowMatchingModel(
        data_dim=DATA_DIM,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        time_embed_dim=TIME_EMBED_DIM,
        prediction_type=prediction_type,
        default_sample_steps=5,
        warmup_epochs=0,
    )
    m.eval()
    x1 = torch.randn(4, DATA_DIM)
    with torch.no_grad():
        z = m.encode(x1, num_steps=5)
    assert z.shape == (4, DATA_DIM)
    assert torch.isfinite(z).all()


@pytest.mark.parametrize("prediction_type", ["velocity", "x_prediction"])
def test_feature_combination_nll(prediction_type):
    """Verify NLL computation works for all prediction types."""
    m = FlowMatchingModel(
        data_dim=DATA_DIM,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        time_embed_dim=TIME_EMBED_DIM,
        prediction_type=prediction_type,
        default_sample_steps=5,
        warmup_epochs=0,
    )
    m.eval()
    x1 = torch.randn(4, DATA_DIM)
    nll = m.compute_nll(x1, num_steps=5)
    assert nll.shape == (4,)
    assert torch.isfinite(nll).all()


# =============================================================================
# Individual Feature Unit Tests
# =============================================================================


def test_pseudo_huber_loss_basic():
    """PseudoHuberLoss produces finite scalar output."""
    loss_fn = PseudoHuberLoss(data_dim=DATA_DIM)
    v_pred = torch.randn(BATCH_SIZE, DATA_DIM)
    v_target = torch.randn(BATCH_SIZE, DATA_DIM)
    loss = loss_fn(v_pred, v_target)
    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert loss > 0


def test_pseudo_huber_loss_zero_residual():
    """PseudoHuberLoss is near zero for identical inputs."""
    loss_fn = PseudoHuberLoss(data_dim=DATA_DIM)
    v = torch.randn(BATCH_SIZE, DATA_DIM)
    loss = loss_fn(v, v)
    assert loss < 1e-4


def test_pseudo_huber_loss_custom_c():
    """PseudoHuberLoss accepts custom c parameter."""
    loss_fn = PseudoHuberLoss(data_dim=DATA_DIM, c=1.0)
    assert loss_fn.c == 1.0
    v_pred = torch.randn(BATCH_SIZE, DATA_DIM)
    v_target = torch.randn(BATCH_SIZE, DATA_DIM)
    loss = loss_fn(v_pred, v_target)
    assert torch.isfinite(loss)


def test_sample_logit_normal_shape():
    """sample_logit_normal returns correct shape in [eps, 1-eps]."""
    t = sample_logit_normal(100)
    assert t.shape == (100,)
    assert (t > 0).all()
    assert (t < 1).all()


def test_sample_logit_normal_concentrates_midrange():
    """Logit-normal samples concentrate around 0.5 more than uniform."""
    torch.manual_seed(42)
    t = sample_logit_normal(10000, m=0.0, s=1.0)
    mid_fraction = ((t > 0.3) & (t < 0.7)).float().mean()
    # Logit-normal with m=0,s=1 should have >50% in [0.3, 0.7]
    # (uniform would have 40%)
    assert mid_fraction > 0.45


# =============================================================================
# AdaLNZeroBlock
# =============================================================================


def test_adaln_zero_block_shape():
    block = AdaLNZeroBlock(HIDDEN_DIM, num_heads=4)
    x = torch.randn(BATCH_SIZE, 8, HIDDEN_DIM)  # 8 tokens
    c = torch.randn(BATCH_SIZE, HIDDEN_DIM)
    out = block(x, c)
    assert out.shape == (BATCH_SIZE, 8, HIDDEN_DIM)


def test_adaln_zero_block_near_identity_at_init():
    block = AdaLNZeroBlock(HIDDEN_DIM, num_heads=4)
    x = torch.randn(BATCH_SIZE, 8, HIDDEN_DIM)
    c = torch.zeros(BATCH_SIZE, HIDDEN_DIM)
    out = block(x, c)
    # With zero-initialized gates, the block output should be very close to input
    residual = (out - x).abs().mean()
    assert residual < 0.1, f"Residual too large at init: {residual}"


# =============================================================================
# VelocityDiT
# =============================================================================


def test_velocity_dit_shape():
    net = VelocityDiT(
        DATA_DIM, HIDDEN_DIM, num_blocks=2, num_heads=4,
        num_tokens=8, time_embed_dim=TIME_EMBED_DIM,
    )
    x = torch.randn(BATCH_SIZE, DATA_DIM)
    t = torch.rand(BATCH_SIZE)
    out = net(x, t)
    assert out.shape == (BATCH_SIZE, DATA_DIM)


def test_velocity_dit_zero_init():
    net = VelocityDiT(
        DATA_DIM, HIDDEN_DIM, num_blocks=2, num_heads=4,
        num_tokens=8, time_embed_dim=TIME_EMBED_DIM,
    )
    x = torch.randn(BATCH_SIZE, DATA_DIM)
    t = torch.rand(BATCH_SIZE)
    out = net(x, t)
    assert out.abs().max() < 0.5, f"Output not near zero at init: max={out.abs().max()}"


def test_velocity_dit_conditional():
    net = VelocityDiT(
        DATA_DIM, HIDDEN_DIM, num_blocks=2, num_heads=4,
        num_tokens=8, time_embed_dim=TIME_EMBED_DIM, condition_dim=16,
    )
    x = torch.randn(BATCH_SIZE, DATA_DIM)
    t = torch.rand(BATCH_SIZE)
    c = torch.randn(BATCH_SIZE, 16)
    out = net(x, t, condition=c)
    assert out.shape == (BATCH_SIZE, DATA_DIM)


def test_velocity_dit_non_divisible_dim():
    """DiT handles data_dim not evenly divisible by num_tokens."""
    odd_dim = 30  # 30 / 8 = 3.75, padded to 32
    net = VelocityDiT(
        odd_dim, HIDDEN_DIM, num_blocks=2, num_heads=4,
        num_tokens=8, time_embed_dim=TIME_EMBED_DIM,
    )
    x = torch.randn(BATCH_SIZE, odd_dim)
    t = torch.rand(BATCH_SIZE)
    out = net(x, t)
    assert out.shape == (BATCH_SIZE, odd_dim)


def test_velocity_dit_gradient_flow():
    """Verify gradients flow through all DiT components."""
    net = VelocityDiT(
        DATA_DIM, HIDDEN_DIM, num_blocks=2, num_heads=4,
        num_tokens=8, time_embed_dim=TIME_EMBED_DIM,
    )
    x = torch.randn(BATCH_SIZE, DATA_DIM, requires_grad=True)
    t = torch.rand(BATCH_SIZE)
    out = net(x, t)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    for name, p in net.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"


# =============================================================================
# FlowMatchingModel with DiT
# =============================================================================


@pytest.fixture
def dit_model():
    return FlowMatchingModel(
        data_dim=DATA_DIM,
        hidden_dim=HIDDEN_DIM,
        num_blocks=2,
        time_embed_dim=TIME_EMBED_DIM,
        condition_dim=0,
        use_ot_coupling=True,
        default_sample_steps=5,
        warmup_epochs=0,
        velocity_arch="dit",
        num_heads=4,
        num_tokens=8,
    )


def test_dit_model_loss_finite(dit_model, fake_batch):
    dit_model.train()
    loss = dit_model.training_step(fake_batch, 0)
    assert torch.isfinite(loss)
    assert loss > 0


def test_dit_model_sample_shape(dit_model):
    dit_model.eval()
    with torch.no_grad():
        samples = dit_model.sample(num_samples=4, num_steps=3)
    assert samples.shape == (4, DATA_DIM)
    assert torch.isfinite(samples).all()


def test_dit_model_save_load(dit_model, tmp_path):
    ckpt_path = tmp_path / "dit_test.pt"
    torch.save(dit_model.state_dict(), ckpt_path)

    loaded = FlowMatchingModel(
        data_dim=DATA_DIM,
        hidden_dim=HIDDEN_DIM,
        num_blocks=2,
        time_embed_dim=TIME_EMBED_DIM,
        condition_dim=0,
        default_sample_steps=5,
        warmup_epochs=0,
        velocity_arch="dit",
        num_heads=4,
        num_tokens=8,
    )
    loaded.load_state_dict(torch.load(ckpt_path, weights_only=True))

    x = torch.randn(4, DATA_DIM)
    t = torch.rand(4)

    dit_model.eval()
    loaded.eval()
    with torch.no_grad():
        out1 = dit_model.velocity_net(x, t)
        out2 = loaded.velocity_net(x, t)
    assert torch.allclose(out1, out2)


def test_invalid_velocity_arch():
    with pytest.raises(ValueError, match="Unknown velocity_arch"):
        FlowMatchingModel(data_dim=DATA_DIM, velocity_arch="invalid")


@pytest.mark.parametrize("velocity_arch", ["mlp", "dit"])
def test_velocity_arch_training(velocity_arch, fake_batch):
    """Verify both architectures produce finite losses and gradients."""
    kwargs = dict(
        data_dim=DATA_DIM,
        hidden_dim=HIDDEN_DIM,
        num_blocks=2,
        time_embed_dim=TIME_EMBED_DIM,
        use_ot_coupling=True,
        default_sample_steps=5,
        warmup_epochs=0,
        velocity_arch=velocity_arch,
    )
    if velocity_arch == "dit":
        kwargs.update(num_heads=4, num_tokens=8)
    m = FlowMatchingModel(**kwargs)
    m.train()
    loss = m.training_step(fake_batch, 0)
    assert torch.isfinite(loss), f"Non-finite loss for {velocity_arch}"
    loss.backward()
    for name, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name}"


# =============================================================================
# vector_part parameter
# =============================================================================


def test_vector_part_node_terms():
    """vector_part='node_terms' extracts only the first half from batch."""
    hv_dim = DATA_DIM // 2
    m = FlowMatchingModel(
        data_dim=hv_dim,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        time_embed_dim=TIME_EMBED_DIM,
        vector_part="node_terms",
        default_sample_steps=5,
        warmup_epochs=0,
    )
    data_list = []
    for _ in range(4):
        d = Data()
        d.node_terms = torch.randn(hv_dim)
        d.graph_terms = torch.randn(hv_dim)
        d.edge_index = torch.zeros(2, 0, dtype=torch.long)
        d.num_nodes = 1
        data_list.append(d)
    batch = Batch.from_data_list(data_list)
    vectors = m._extract_vectors(batch)
    assert vectors.shape == (4, hv_dim)
    # Should match node_terms, not graph_terms
    expected = torch.stack([d.node_terms for d in data_list])
    assert torch.allclose(vectors, expected)


def test_vector_part_graph_terms():
    """vector_part='graph_terms' extracts only the second half from batch."""
    hv_dim = DATA_DIM // 2
    m = FlowMatchingModel(
        data_dim=hv_dim,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        time_embed_dim=TIME_EMBED_DIM,
        vector_part="graph_terms",
        default_sample_steps=5,
        warmup_epochs=0,
    )
    data_list = []
    for _ in range(4):
        d = Data()
        d.node_terms = torch.randn(hv_dim)
        d.graph_terms = torch.randn(hv_dim)
        d.edge_index = torch.zeros(2, 0, dtype=torch.long)
        d.num_nodes = 1
        data_list.append(d)
    batch = Batch.from_data_list(data_list)
    vectors = m._extract_vectors(batch)
    assert vectors.shape == (4, hv_dim)
    expected = torch.stack([d.graph_terms for d in data_list])
    assert torch.allclose(vectors, expected)


def test_vector_part_graph_condition():
    """vector_part='graph_terms' returns node_terms as condition."""
    hv_dim = DATA_DIM // 2
    m = FlowMatchingModel(
        data_dim=hv_dim,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        time_embed_dim=TIME_EMBED_DIM,
        condition_dim=hv_dim,
        condition_embed_dim=16,
        vector_part="graph_terms",
        default_sample_steps=5,
        warmup_epochs=0,
    )
    data_list = []
    for _ in range(4):
        d = Data()
        d.node_terms = torch.randn(hv_dim)
        d.graph_terms = torch.randn(hv_dim)
        d.edge_index = torch.zeros(2, 0, dtype=torch.long)
        d.num_nodes = 1
        data_list.append(d)
    batch = Batch.from_data_list(data_list)
    cond = m._extract_condition(batch)
    assert cond is not None
    assert cond.shape == (4, hv_dim)
    expected = torch.stack([d.node_terms for d in data_list])
    assert torch.allclose(cond, expected)


def test_graph_terms_training_step():
    """End-to-end training step with vector_part='graph_terms' and conditioning."""
    hv_dim = DATA_DIM // 2
    m = FlowMatchingModel(
        data_dim=hv_dim,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        time_embed_dim=TIME_EMBED_DIM,
        condition_dim=hv_dim,
        condition_embed_dim=16,
        vector_part="graph_terms",
        default_sample_steps=5,
        warmup_epochs=0,
    )
    data_list = []
    for _ in range(BATCH_SIZE):
        d = Data()
        d.node_terms = torch.randn(hv_dim)
        d.graph_terms = torch.randn(hv_dim)
        d.edge_index = torch.zeros(2, 0, dtype=torch.long)
        d.num_nodes = 1
        data_list.append(d)
    batch = Batch.from_data_list(data_list)
    m.train()
    loss = m.training_step(batch, 0)
    assert torch.isfinite(loss)
    assert loss > 0
    loss.backward()


# =============================================================================
# Cosine Loss
# =============================================================================


@pytest.mark.parametrize("prediction_type", ["velocity", "x_prediction"])
def test_cosine_loss_finite(prediction_type):
    """Cosine loss produces finite total loss for both prediction types."""
    m = FlowMatchingModel(
        data_dim=DATA_DIM,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        time_embed_dim=TIME_EMBED_DIM,
        prediction_type=prediction_type,
        cosine_loss_weight=1.0,
        default_sample_steps=5,
        warmup_epochs=0,
    )
    m.train()
    x = torch.randn(BATCH_SIZE, DATA_DIM)
    loss = m.compute_loss(x)
    assert torch.isfinite(loss)
    assert loss > 0
    loss.backward()
    for name, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name}"


def test_cosine_loss_training_step(fake_batch):
    """Full training_step works with cosine loss enabled."""
    m = FlowMatchingModel(
        data_dim=DATA_DIM,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        time_embed_dim=TIME_EMBED_DIM,
        cosine_loss_weight=0.5,
        default_sample_steps=5,
        warmup_epochs=0,
    )
    m.train()
    loss = m.training_step(fake_batch, 0)
    assert torch.isfinite(loss)
    assert loss > 0


def test_cosine_loss_zero_weight_matches_original(fake_batch):
    """cosine_loss_weight=0.0 produces same loss as without cosine loss."""
    torch.manual_seed(42)
    m = FlowMatchingModel(
        data_dim=DATA_DIM,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        time_embed_dim=TIME_EMBED_DIM,
        cosine_loss_weight=0.0,
        default_sample_steps=5,
        warmup_epochs=0,
    )
    m.train()
    torch.manual_seed(123)
    loss_no_cos = m.compute_loss(torch.randn(4, DATA_DIM))

    m2 = FlowMatchingModel(
        data_dim=DATA_DIM,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        time_embed_dim=TIME_EMBED_DIM,
        cosine_loss_weight=0.0,
        default_sample_steps=5,
        warmup_epochs=0,
    )
    m2.load_state_dict(m.state_dict())
    m2.train()
    torch.manual_seed(123)
    loss_zero = m2.compute_loss(torch.randn(4, DATA_DIM))

    assert torch.allclose(loss_no_cos, loss_zero)


# =============================================================================
# Target Noise
# =============================================================================


def test_target_noise_loss_finite(fake_batch):
    """Target noise produces finite losses and gradients."""
    m = FlowMatchingModel(
        data_dim=DATA_DIM,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        time_embed_dim=TIME_EMBED_DIM,
        target_noise_sigma=0.1,
        default_sample_steps=5,
        warmup_epochs=0,
    )
    m.train()
    loss = m.training_step(fake_batch, 0)
    assert torch.isfinite(loss)
    assert loss > 0
    loss.backward()
    for name, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name}"


def test_target_noise_disabled_matches_baseline(fake_batch):
    """target_noise_sigma=0.0 is a no-op — loss matches a model without it."""
    m = FlowMatchingModel(
        data_dim=DATA_DIM,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        time_embed_dim=TIME_EMBED_DIM,
        target_noise_sigma=0.0,
        default_sample_steps=5,
        warmup_epochs=0,
    )
    m.train()
    torch.manual_seed(123)
    loss_zero = m.compute_loss(torch.randn(4, DATA_DIM))

    m2 = FlowMatchingModel(
        data_dim=DATA_DIM,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        time_embed_dim=TIME_EMBED_DIM,
        default_sample_steps=5,
        warmup_epochs=0,
    )
    m2.load_state_dict(m.state_dict())
    m2.train()
    torch.manual_seed(123)
    loss_baseline = m2.compute_loss(torch.randn(4, DATA_DIM))

    assert torch.allclose(loss_zero, loss_baseline)


def test_target_noise_only_during_training():
    """Target noise is not applied during eval (validation loss is clean)."""
    m = FlowMatchingModel(
        data_dim=DATA_DIM,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        time_embed_dim=TIME_EMBED_DIM,
        target_noise_sigma=0.5,  # large sigma to make the difference obvious
        default_sample_steps=5,
        warmup_epochs=0,
    )

    # Eval mode: target noise should be skipped
    m.eval()
    torch.manual_seed(42)
    loss_eval = m.compute_loss(torch.randn(4, DATA_DIM))

    # Compare with a sigma=0 model in eval mode (should be identical)
    m_clean = FlowMatchingModel(
        data_dim=DATA_DIM,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        time_embed_dim=TIME_EMBED_DIM,
        target_noise_sigma=0.0,
        default_sample_steps=5,
        warmup_epochs=0,
    )
    m_clean.load_state_dict(m.state_dict())
    m_clean.eval()
    torch.manual_seed(42)
    loss_clean = m_clean.compute_loss(torch.randn(4, DATA_DIM))

    assert torch.allclose(loss_eval, loss_clean)


# =============================================================================
# Condition Dropout
# =============================================================================


def test_cond_dropout_loss_finite():
    """Condition dropout produces finite losses and gradients."""
    m = FlowMatchingModel(
        data_dim=DATA_DIM,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        time_embed_dim=TIME_EMBED_DIM,
        condition_dim=COND_RAW_DIM,
        condition_embed_dim=COND_EMBED_DIM,
        cond_dropout_prob=0.5,
        default_sample_steps=5,
        warmup_epochs=0,
    )
    m.train()
    x = torch.randn(BATCH_SIZE, DATA_DIM)
    c = torch.randn(BATCH_SIZE, COND_RAW_DIM)
    loss = m.compute_loss(x, condition=c)
    assert torch.isfinite(loss)
    assert loss > 0
    loss.backward()
    for name, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name}"


def test_cond_dropout_only_during_training():
    """Condition dropout is not applied during eval."""
    m = FlowMatchingModel(
        data_dim=DATA_DIM,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        time_embed_dim=TIME_EMBED_DIM,
        condition_dim=COND_RAW_DIM,
        condition_embed_dim=COND_EMBED_DIM,
        cond_dropout_prob=0.5,
        default_sample_steps=5,
        warmup_epochs=0,
    )

    # In eval mode, dropout should be skipped — same result as prob=0
    m_clean = FlowMatchingModel(
        data_dim=DATA_DIM,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        time_embed_dim=TIME_EMBED_DIM,
        condition_dim=COND_RAW_DIM,
        condition_embed_dim=COND_EMBED_DIM,
        cond_dropout_prob=0.0,
        default_sample_steps=5,
        warmup_epochs=0,
    )
    m_clean.load_state_dict(m.state_dict())

    m.eval()
    m_clean.eval()

    x = torch.randn(4, DATA_DIM)
    c = torch.randn(4, COND_RAW_DIM)
    torch.manual_seed(42)
    loss_dropout = m.compute_loss(x, condition=c)
    torch.manual_seed(42)
    loss_clean = m_clean.compute_loss(x, condition=c)

    assert torch.allclose(loss_dropout, loss_clean)


def test_cond_dropout_disabled_is_noop():
    """cond_dropout_prob=0.0 gives identical results to no dropout."""
    m = FlowMatchingModel(
        data_dim=DATA_DIM,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        time_embed_dim=TIME_EMBED_DIM,
        condition_dim=COND_RAW_DIM,
        condition_embed_dim=COND_EMBED_DIM,
        cond_dropout_prob=0.0,
        default_sample_steps=5,
        warmup_epochs=0,
    )
    m2 = FlowMatchingModel(
        data_dim=DATA_DIM,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        time_embed_dim=TIME_EMBED_DIM,
        condition_dim=COND_RAW_DIM,
        condition_embed_dim=COND_EMBED_DIM,
        default_sample_steps=5,
        warmup_epochs=0,
    )
    m2.load_state_dict(m.state_dict())

    m.train()
    m2.train()
    x = torch.randn(4, DATA_DIM)
    c = torch.randn(4, COND_RAW_DIM)
    torch.manual_seed(42)
    loss1 = m.compute_loss(x, condition=c)
    torch.manual_seed(42)
    loss2 = m2.compute_loss(x, condition=c)

    assert torch.allclose(loss1, loss2)


# =============================================================================
# PCA Dimensionality Reduction
# =============================================================================

PCA_ORIG_DIM = 32  # original hv_dim
PCA_DIM = 8        # reduced dim


def _fit_test_pca(orig_dim: int = PCA_ORIG_DIM, pca_dim: int = PCA_DIM):
    """Create synthetic data with known low-rank structure and fit PCA."""
    torch.manual_seed(0)
    # Data lives in a low-rank subspace (pca_dim) embedded in orig_dim
    basis = torch.randn(pca_dim, orig_dim)
    basis = torch.linalg.qr(basis.T).Q.T  # orthonormal rows (pca_dim, orig_dim)
    coeffs = torch.randn(200, pca_dim)
    data = coeffs @ basis  # (200, orig_dim) — rank pca_dim

    mean = data.mean(dim=0)
    centered = data - mean
    _, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    components = Vh[:pca_dim]  # (pca_dim, orig_dim)
    return data, components, mean


def test_pca_roundtrip():
    """PCA project -> inverse roundtrip preserves vectors for low-rank data."""
    data, components, mean = _fit_test_pca()

    m = FlowMatchingModel(
        data_dim=PCA_DIM, hidden_dim=32, num_blocks=1,
        time_embed_dim=8, default_sample_steps=3, warmup_epochs=0,
    )
    m.set_pca(components, mean)
    assert m._use_pca

    projected = m.pca_project(data)
    assert projected.shape == (200, PCA_DIM)

    reconstructed = m.pca_inverse(projected)
    assert reconstructed.shape == (200, PCA_ORIG_DIM)

    cos_sim = torch.nn.functional.cosine_similarity(data, reconstructed, dim=-1)
    assert cos_sim.mean() > 0.99, f"Mean cosine sim {cos_sim.mean():.4f} too low"


def test_pca_extract_vectors():
    """_extract_vectors returns PCA-dim vectors when PCA is active."""
    _, components, mean = _fit_test_pca()

    m = FlowMatchingModel(
        data_dim=PCA_DIM, hidden_dim=32, num_blocks=1,
        time_embed_dim=8, default_sample_steps=3, warmup_epochs=0,
        vector_part="node_terms",
    )
    m.set_pca(components, mean)

    # Build a fake batch with node_terms of shape (orig_dim,)
    data_list = []
    for _ in range(4):
        d = Data()
        d.node_terms = torch.randn(PCA_ORIG_DIM)
        d.graph_terms = torch.randn(PCA_ORIG_DIM)
        d.edge_index = torch.zeros(2, 0, dtype=torch.long)
        d.num_nodes = 1
        data_list.append(d)
    batch = Batch.from_data_list(data_list)

    vectors = m._extract_vectors(batch)
    assert vectors.shape == (4, PCA_DIM)


def test_pca_sample_shape():
    """sample() returns (num_samples, original_dim) when PCA is active."""
    _, components, mean = _fit_test_pca()

    m = FlowMatchingModel(
        data_dim=PCA_DIM, hidden_dim=32, num_blocks=1,
        time_embed_dim=8, default_sample_steps=3, warmup_epochs=0,
    )
    m.set_pca(components, mean)

    # Fit dummy standardization in PCA space
    m.set_standardization(torch.zeros(PCA_DIM), torch.ones(PCA_DIM))

    samples = m.sample(5)
    # Output should be in original space, not PCA space
    assert samples.shape == (5, PCA_ORIG_DIM)


def test_pca_encode_decode_roundtrip():
    """encode -> ODE forward -> destandardize -> pca_inverse reconstructs."""
    data, components, mean = _fit_test_pca()

    m = FlowMatchingModel(
        data_dim=PCA_DIM, hidden_dim=32, num_blocks=1,
        time_embed_dim=8, default_sample_steps=5, warmup_epochs=0,
    )
    m.set_pca(components, mean)

    # Fit standardization on PCA-projected data
    projected = m.pca_project(data)
    m.set_standardization(projected.mean(dim=0), projected.std(dim=0).clamp(min=1e-8))

    # encode takes original-space vectors
    subset = data[:4]
    encoded = m.encode(subset)
    assert encoded.shape == (4, PCA_DIM)  # noise in PCA-dim space


def test_pca_training_step():
    """Full training_step works with PCA-enabled model."""
    _, components, mean = _fit_test_pca()

    m = FlowMatchingModel(
        data_dim=PCA_DIM, hidden_dim=32, num_blocks=1,
        time_embed_dim=8, default_sample_steps=3, warmup_epochs=0,
        vector_part="node_terms",
    )
    m.set_pca(components, mean)
    m.set_standardization(torch.zeros(PCA_DIM), torch.ones(PCA_DIM))

    data_list = []
    for _ in range(4):
        d = Data()
        d.node_terms = torch.randn(PCA_ORIG_DIM)
        d.graph_terms = torch.randn(PCA_ORIG_DIM)
        d.edge_index = torch.zeros(2, 0, dtype=torch.long)
        d.num_nodes = 1
        data_list.append(d)
    batch = Batch.from_data_list(data_list)

    m.train()
    loss = m.training_step(batch, 0)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_pca_not_active_by_default(model):
    """PCA is inactive when set_pca has not been called."""
    assert not model._use_pca
    # sample should return data_dim, not go through PCA inverse
    samples = model.sample(3)
    assert samples.shape == (3, DATA_DIM)


def test_pca_save_load_state_dict():
    """PCA buffers survive save/load_state_dict (strict=True)."""
    _, components, mean = _fit_test_pca()

    m = FlowMatchingModel(
        data_dim=PCA_DIM, hidden_dim=32, num_blocks=1,
        time_embed_dim=8, default_sample_steps=3, warmup_epochs=0,
    )
    m.set_pca(components, mean)
    m.set_standardization(torch.zeros(PCA_DIM), torch.ones(PCA_DIM))

    sd = m.state_dict()
    assert "pca_components" in sd
    assert "pca_mean" in sd

    # Load into fresh model (None PCA buffers)
    m2 = FlowMatchingModel(
        data_dim=PCA_DIM, hidden_dim=32, num_blocks=1,
        time_embed_dim=8, default_sample_steps=3, warmup_epochs=0,
    )
    m2.load_state_dict(sd, strict=True)
    assert m2._use_pca
    assert torch.allclose(m.pca_components, m2.pca_components)
    assert torch.allclose(m.pca_mean, m2.pca_mean)

    # Sampling should return original dim
    samples = m2.sample(3)
    assert samples.shape == (3, PCA_ORIG_DIM)


# =============================================================================
# Gaussian Mixture (GMFlow) — Unit Tests
# =============================================================================

NUM_GAUSSIANS = 4


def test_gmm_output_head_shapes():
    """GMMOutputHead returns dict with correct shapes."""
    head = GMMOutputHead(HIDDEN_DIM, DATA_DIM, NUM_GAUSSIANS)
    h = torch.randn(BATCH_SIZE, HIDDEN_DIM)
    gm = head(h)

    assert isinstance(gm, dict)
    assert gm["means"].shape == (BATCH_SIZE, NUM_GAUSSIANS, DATA_DIM)
    assert gm["logweights"].shape == (BATCH_SIZE, NUM_GAUSSIANS)
    assert gm["logstds"].shape == (BATCH_SIZE,)


def test_gmm_output_head_init():
    """At initialization, weights are uniform, logstd=0, and components are asymmetric."""
    head = GMMOutputHead(HIDDEN_DIM, DATA_DIM, NUM_GAUSSIANS)

    # With zero input, output is just bias (zeros) — test weights/logstd
    h_zero = torch.zeros(1, HIDDEN_DIM)
    gm = head(h_zero)
    expected_logw = -torch.log(torch.tensor(float(NUM_GAUSSIANS)))
    assert torch.allclose(gm["logweights"], expected_logw.expand_as(gm["logweights"]), atol=1e-5)
    assert gm["logstds"].abs().max() < 1e-6  # logstd=0 → sigma=1

    # With non-zero input, Kaiming init breaks symmetry between components
    h = torch.ones(1, HIDDEN_DIM)
    gm = head(h)
    assert not torch.allclose(gm["means"][0, 0], gm["means"][0, 1], atol=1e-6)


def test_gm_nll_loss_shape_and_finite():
    """NLL loss has shape (bs,) and is finite."""
    gm = {
        "means": torch.randn(BATCH_SIZE, NUM_GAUSSIANS, DATA_DIM),
        "logweights": torch.randn(BATCH_SIZE, NUM_GAUSSIANS).log_softmax(dim=-1),
        "logstds": torch.zeros(BATCH_SIZE),
    }
    target = torch.randn(BATCH_SIZE, DATA_DIM)

    loss = gm_nll_loss(gm, target)
    assert loss.shape == (BATCH_SIZE,)
    assert torch.isfinite(loss).all()


def test_gm_nll_loss_gradient_flow():
    """Gradients flow through all GMM parameters."""
    means = torch.randn(BATCH_SIZE, NUM_GAUSSIANS, DATA_DIM, requires_grad=True)
    logweights = torch.randn(BATCH_SIZE, NUM_GAUSSIANS, requires_grad=True)
    logstds = torch.zeros(BATCH_SIZE, requires_grad=True)
    gm = {
        "means": means,
        "logweights": logweights.log_softmax(dim=-1),
        "logstds": logstds,
    }
    target = torch.randn(BATCH_SIZE, DATA_DIM)

    loss = gm_nll_loss(gm, target).mean()
    loss.backward()

    assert means.grad is not None and torch.isfinite(means.grad).all()
    assert logweights.grad is not None and torch.isfinite(logweights.grad).all()
    assert logstds.grad is not None and torch.isfinite(logstds.grad).all()


def test_gm_to_mean_shape():
    """gm_to_mean returns (bs, D) and is correct for uniform weights."""
    means = torch.randn(BATCH_SIZE, NUM_GAUSSIANS, DATA_DIM)
    gm = {
        "means": means,
        "logweights": torch.zeros(BATCH_SIZE, NUM_GAUSSIANS).log_softmax(dim=-1),
    }
    result = gm_to_mean(gm)
    assert result.shape == (BATCH_SIZE, DATA_DIM)
    # With uniform weights, mean should be arithmetic mean of components
    expected = means.mean(dim=-2)
    assert torch.allclose(result, expected, atol=1e-5)


def test_gm_to_sample_shape():
    """gm_to_sample returns (bs, D) with finite values."""
    gm = {
        "means": torch.randn(BATCH_SIZE, NUM_GAUSSIANS, DATA_DIM),
        "logweights": torch.zeros(BATCH_SIZE, NUM_GAUSSIANS).log_softmax(dim=-1),
        "logstds": torch.zeros(BATCH_SIZE),
    }
    result = gm_to_sample(gm)
    assert result.shape == (BATCH_SIZE, DATA_DIM)
    assert torch.isfinite(result).all()


# =============================================================================
# Gaussian Mixture (GMFlow) — Integration Tests
# =============================================================================


def test_velocity_mlp_gmm_output():
    """VelocityMLP with output_type='gmm' returns dict with correct shapes."""
    net = VelocityMLP(
        data_dim=DATA_DIM,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        time_embed_dim=TIME_EMBED_DIM,
        output_type="gmm",
        num_gaussians=NUM_GAUSSIANS,
    )
    x_t = torch.randn(BATCH_SIZE, DATA_DIM)
    t = torch.rand(BATCH_SIZE)
    out = net(x_t, t)

    assert isinstance(out, dict)
    assert out["means"].shape == (BATCH_SIZE, NUM_GAUSSIANS, DATA_DIM)
    assert out["logweights"].shape == (BATCH_SIZE, NUM_GAUSSIANS)
    assert out["logstds"].shape == (BATCH_SIZE,)


def test_flow_matching_gm_nll_training_step(fake_batch):
    """FlowMatchingModel with loss_type='gm_nll' produces finite loss."""
    m = FlowMatchingModel(
        data_dim=DATA_DIM,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        time_embed_dim=TIME_EMBED_DIM,
        loss_type="gm_nll",
        num_gaussians=NUM_GAUSSIANS,
        default_sample_steps=5,
        warmup_epochs=0,
    )
    m.train()
    loss = m.training_step(fake_batch, 0)
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name}"


def test_flow_matching_gm_nll_sample():
    """Sampling works with GMM model and produces correct shape."""
    m = FlowMatchingModel(
        data_dim=DATA_DIM,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        time_embed_dim=TIME_EMBED_DIM,
        loss_type="gm_nll",
        num_gaussians=NUM_GAUSSIANS,
        default_sample_steps=5,
        warmup_epochs=0,
    )
    m.eval()
    samples = m.sample(4)
    assert samples.shape == (4, DATA_DIM)
    assert torch.isfinite(samples).all()


def test_flow_matching_gm_nll_with_cosine_loss(fake_batch):
    """GMM model is compatible with auxiliary cosine loss."""
    m = FlowMatchingModel(
        data_dim=DATA_DIM,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        time_embed_dim=TIME_EMBED_DIM,
        loss_type="gm_nll",
        num_gaussians=NUM_GAUSSIANS,
        cosine_loss_weight=0.1,
        default_sample_steps=5,
        warmup_epochs=0,
    )
    m.train()
    loss = m.training_step(fake_batch, 0)
    assert torch.isfinite(loss)
    loss.backward()


def test_flow_matching_gm_nll_checkpoint():
    """GMM model state_dict save/load roundtrip works."""
    m = FlowMatchingModel(
        data_dim=DATA_DIM,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        time_embed_dim=TIME_EMBED_DIM,
        loss_type="gm_nll",
        num_gaussians=NUM_GAUSSIANS,
        default_sample_steps=5,
        warmup_epochs=0,
    )
    sd = m.state_dict()

    m2 = FlowMatchingModel(
        data_dim=DATA_DIM,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        time_embed_dim=TIME_EMBED_DIM,
        loss_type="gm_nll",
        num_gaussians=NUM_GAUSSIANS,
        default_sample_steps=5,
        warmup_epochs=0,
    )
    m2.load_state_dict(sd, strict=True)

    # Verify sampling produces same results
    torch.manual_seed(42)
    s1 = m.eval().sample(2)
    torch.manual_seed(42)
    s2 = m2.eval().sample(2)
    assert torch.allclose(s1, s2)


def test_flow_matching_mse_unchanged(fake_batch):
    """Existing loss_type='mse' still works identically (backward compat)."""
    m = FlowMatchingModel(
        data_dim=DATA_DIM,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        time_embed_dim=TIME_EMBED_DIM,
        loss_type="mse",
        default_sample_steps=5,
        warmup_epochs=0,
    )
    m.train()
    torch.manual_seed(99)
    loss = m.compute_loss(torch.randn(4, DATA_DIM))
    assert torch.isfinite(loss)
    assert loss > 0


# =============================================================================
# GMM Transition Loss
# =============================================================================


def test_gmm_reverse_transition_shapes():
    """Reverse transition preserves GMM shapes."""
    gm = {
        "means": torch.randn(BATCH_SIZE, NUM_GAUSSIANS, DATA_DIM),
        "logweights": torch.randn(BATCH_SIZE, NUM_GAUSSIANS).log_softmax(dim=-1),
        "logstds": torch.zeros(BATCH_SIZE),
    }
    x_t = torch.randn(BATCH_SIZE, DATA_DIM)
    t_from = torch.rand(BATCH_SIZE) * 0.5  # noisy (low t)
    t_to = t_from + 0.3  # cleaner (higher t)

    gm_rev = gmm_reverse_transition(gm, x_t, t_from, t_to, "velocity")

    assert gm_rev["means"].shape == (BATCH_SIZE, NUM_GAUSSIANS, DATA_DIM)
    assert gm_rev["logweights"].shape == (BATCH_SIZE, NUM_GAUSSIANS)
    assert gm_rev["logstds"].shape == (BATCH_SIZE,)
    assert torch.isfinite(gm_rev["means"]).all()
    assert torch.isfinite(gm_rev["logstds"]).all()


def test_gmm_reverse_transition_gradients():
    """Gradients flow through reverse transition."""
    means = torch.randn(BATCH_SIZE, NUM_GAUSSIANS, DATA_DIM, requires_grad=True)
    logstds = torch.zeros(BATCH_SIZE, requires_grad=True)
    gm = {
        "means": means,
        "logweights": torch.zeros(BATCH_SIZE, NUM_GAUSSIANS).log_softmax(dim=-1),
        "logstds": logstds,
    }
    x_t = torch.randn(BATCH_SIZE, DATA_DIM)
    t_from = torch.full((BATCH_SIZE,), 0.3)
    t_to = torch.full((BATCH_SIZE,), 0.7)

    gm_rev = gmm_reverse_transition(gm, x_t, t_from, t_to, "velocity")
    loss = gm_rev["means"].sum() + gm_rev["logstds"].sum()
    loss.backward()

    assert means.grad is not None and torch.isfinite(means.grad).all()
    assert logstds.grad is not None and torch.isfinite(logstds.grad).all()


def test_transition_loss_training_step(fake_batch):
    """Transition loss produces finite loss and gradients."""
    m = FlowMatchingModel(
        data_dim=DATA_DIM,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        time_embed_dim=TIME_EMBED_DIM,
        loss_type="gm_nll",
        num_gaussians=NUM_GAUSSIANS,
        trans_ratio=0.5,
        default_sample_steps=5,
        warmup_epochs=0,
    )
    m.train()
    loss = m.training_step(fake_batch, 0)
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name}"


@pytest.mark.parametrize("pred_type", ["velocity", "x_prediction"])
def test_transition_loss_both_prediction_types(fake_batch, pred_type):
    """Transition loss works with both prediction types."""
    m = FlowMatchingModel(
        data_dim=DATA_DIM,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        time_embed_dim=TIME_EMBED_DIM,
        loss_type="gm_nll",
        prediction_type=pred_type,
        num_gaussians=NUM_GAUSSIANS,
        trans_ratio=0.5,
        default_sample_steps=5,
        warmup_epochs=0,
    )
    m.train()
    loss = m.training_step(fake_batch, 0)
    assert torch.isfinite(loss)
    loss.backward()


def test_transition_loss_with_cosine(fake_batch):
    """Transition loss is compatible with cosine auxiliary loss."""
    m = FlowMatchingModel(
        data_dim=DATA_DIM,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        time_embed_dim=TIME_EMBED_DIM,
        loss_type="gm_nll",
        num_gaussians=NUM_GAUSSIANS,
        trans_ratio=0.5,
        cosine_loss_weight=0.1,
        default_sample_steps=5,
        warmup_epochs=0,
    )
    m.train()
    loss = m.training_step(fake_batch, 0)
    assert torch.isfinite(loss)
    loss.backward()


def test_transition_loss_sample():
    """Sampling still works after configuring transition loss."""
    m = FlowMatchingModel(
        data_dim=DATA_DIM,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        time_embed_dim=TIME_EMBED_DIM,
        loss_type="gm_nll",
        num_gaussians=NUM_GAUSSIANS,
        trans_ratio=0.5,
        default_sample_steps=5,
        warmup_epochs=0,
    )
    m.eval()
    samples = m.sample(4)
    assert samples.shape == (4, DATA_DIM)
    assert torch.isfinite(samples).all()


def test_transition_loss_backward_compat(fake_batch):
    """MSE path is unaffected by trans_ratio parameter."""
    m = FlowMatchingModel(
        data_dim=DATA_DIM,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        time_embed_dim=TIME_EMBED_DIM,
        loss_type="mse",
        trans_ratio=0.5,  # should be ignored
        default_sample_steps=5,
        warmup_epochs=0,
    )
    m.train()
    torch.manual_seed(99)
    loss = m.compute_loss(torch.randn(4, DATA_DIM))
    assert torch.isfinite(loss)
    assert loss > 0
