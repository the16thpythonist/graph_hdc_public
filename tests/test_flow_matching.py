"""Unit tests for the Flow Matching model components."""
import pytest
import torch
from torch_geometric.data import Batch, Data

from graph_hdc.models.flow_matching import (
    CONDITION_REGISTRY,
    CrippenLogP,
    FiLMResidualBlock,
    FlowMatchingModel,
    HeavyAtomCount,
    MinibatchOTCoupler,
    MultiCondition,
    SinusoidalTimeEmbedding,
    VelocityMLP,
    VelocityModelWrapper,
    build_condition,
    get_condition,
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


