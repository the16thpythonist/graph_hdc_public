"""
GPU functionality tests.

These tests verify that GPU operations work correctly when CUDA is available.
They are skipped automatically on systems without a compatible GPU.
"""
import pytest
import torch
from torch_geometric.data import Data


def gpu_available_and_compatible() -> bool:
    """Check if CUDA is available and compatible with current GPU."""
    if not torch.cuda.is_available():
        return False

    # Try a simple GPU operation to check compatibility
    try:
        x = torch.randn(2, 2, device="cuda")
        _ = x @ x  # Simple matmul
        return True
    except RuntimeError as e:
        # Catch "CUDA error: no kernel image is available" or similar
        if "not compatible" in str(e) or "no kernel image" in str(e):
            return False
        raise


requires_gpu = pytest.mark.skipif(
    not gpu_available_and_compatible(),
    reason="GPU not available or not compatible with installed PyTorch"
)


@requires_gpu
class TestGPUBasics:
    """Basic GPU operation tests."""

    def test_cuda_available(self):
        """CUDA should be available."""
        assert torch.cuda.is_available()

    def test_tensor_to_cuda(self):
        """Tensors can be moved to GPU."""
        x = torch.randn(10, 10)
        x_cuda = x.cuda()
        assert x_cuda.device.type == "cuda"

    def test_matmul_on_gpu(self):
        """Matrix multiplication works on GPU."""
        a = torch.randn(100, 100, device="cuda")
        b = torch.randn(100, 100, device="cuda")
        c = a @ b
        assert c.device.type == "cuda"
        assert c.shape == (100, 100)


@requires_gpu
class TestPyGOnGPU:
    """PyTorch Geometric GPU tests."""

    def test_data_to_cuda(self):
        """PyG Data objects can be moved to GPU."""
        x = torch.randn(4, 16)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)

        data = data.to("cuda")

        assert data.x.device.type == "cuda"
        assert data.edge_index.device.type == "cuda"

    def test_scatter_on_gpu(self):
        """PyG scatter operations work on GPU."""
        from torch_geometric.utils import scatter

        x = torch.randn(10, 16, device="cuda")
        batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2], device="cuda")

        out = scatter(x, batch, dim=0, reduce="mean")

        assert out.device.type == "cuda"
        assert out.shape == (3, 16)


@requires_gpu
class TestHyperNetOnGPU:
    """HyperNet GPU tests."""

    def test_hypernet_creation_on_gpu(self):
        """HyperNet can be created on GPU."""
        from graph_hdc.hypernet.configs import get_config
        from graph_hdc.hypernet.encoder import HyperNet

        config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
        config.device = "cuda"
        hypernet = HyperNet(config)

        # Core codebooks should be on CUDA
        assert hypernet.nodes_codebook.device.type == "cuda"
        assert hypernet.edges_codebook.device.type == "cuda"

        # Node encoder codebooks should also be on CUDA
        for enc, _ in hypernet.node_encoder_map.values():
            assert enc.codebook.device.type == "cuda"

    def test_encoding_on_gpu(self):
        """Molecule encoding works on GPU."""
        from graph_hdc.hypernet.configs import get_config
        from graph_hdc.hypernet.encoder import HyperNet

        config = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
        config.device = "cuda"
        hypernet = HyperNet(config)

        # Ethane: C-C
        x = torch.tensor([
            [0.0, 0.0, 0.0, 3.0],
            [0.0, 0.0, 0.0, 3.0],
        ], device="cuda")
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device="cuda")

        data = Data(x=x, edge_index=edge_index)
        data.batch = torch.zeros(2, dtype=torch.long, device="cuda")

        with torch.no_grad():
            output = hypernet.forward(data, normalize=True)

        assert output["graph_embedding"].device.type == "cuda"
        assert output["edge_terms"].device.type == "cuda"
        assert output["graph_embedding"].shape == (1, 256)

    def test_encoding_cpu_vs_gpu_consistency(self):
        """CPU and GPU encoding produce same results."""
        from graph_hdc.hypernet.configs import get_config
        from graph_hdc.hypernet.encoder import HyperNet

        # Create identical configs with same seed
        config_cpu = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
        config_cpu.device = "cpu"
        config_cpu.seed = 42

        config_gpu = get_config("QM9_SMILES_HRR_256_F64_G1NG3")
        config_gpu.device = "cuda"
        config_gpu.seed = 42

        hypernet_cpu = HyperNet(config_cpu)
        hypernet_gpu = HyperNet(config_gpu)

        # Same molecule on both
        x = torch.tensor([
            [0.0, 0.0, 0.0, 3.0],
            [0.0, 0.0, 0.0, 3.0],
        ])
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

        data_cpu = Data(x=x.clone(), edge_index=edge_index.clone())
        data_cpu.batch = torch.zeros(2, dtype=torch.long)

        data_gpu = Data(x=x.clone().cuda(), edge_index=edge_index.clone().cuda())
        data_gpu.batch = torch.zeros(2, dtype=torch.long, device="cuda")

        with torch.no_grad():
            out_cpu = hypernet_cpu.forward(data_cpu, normalize=True)
            out_gpu = hypernet_gpu.forward(data_gpu, normalize=True)

        # Compare on CPU
        assert torch.allclose(
            out_cpu["graph_embedding"],
            out_gpu["graph_embedding"].cpu(),
            atol=1e-5
        ), "CPU and GPU graph_embedding should match"

        assert torch.allclose(
            out_cpu["edge_terms"],
            out_gpu["edge_terms"].cpu(),
            atol=1e-5
        ), "CPU and GPU edge_terms should match"


@requires_gpu
class TestFlowOnGPU:
    """Flow model GPU tests."""

    def test_flow_forward_on_gpu(self):
        """Flow model forward pass works on GPU."""
        from graph_hdc.models.flows.real_nvp import FlowConfig, RealNVPV3Lightning

        cfg = FlowConfig(hv_dim=256, num_flows=4, hidden_dim=256, num_hidden_layers=2)
        model = RealNVPV3Lightning(cfg).cuda()

        # Random hypervector batch (flat_dim = hv_count * hv_dim = 2 * 256 = 512)
        x = torch.randn(8, 512, device="cuda")

        # Forward pass (normflows forward returns single tensor)
        z = model.flow.forward(x)

        assert z.device.type == "cuda"
        assert z.shape == x.shape

    def test_flow_inverse_on_gpu(self):
        """Flow model inverse (sampling) works on GPU."""
        from graph_hdc.models.flows.real_nvp import FlowConfig, RealNVPV3Lightning

        cfg = FlowConfig(hv_dim=256, num_flows=4, hidden_dim=256, num_hidden_layers=2)
        model = RealNVPV3Lightning(cfg).cuda()

        # Sample from prior (flat_dim = hv_count * hv_dim = 2 * 256 = 512)
        z = torch.randn(8, 512, device="cuda")

        # Inverse pass (normflows inverse returns single tensor)
        x = model.flow.inverse(z)

        assert x.device.type == "cuda"
        assert x.shape == z.shape

    def test_flow_sample_on_gpu(self):
        """Flow model sampling works on GPU."""
        from graph_hdc.models.flows.real_nvp import FlowConfig, RealNVPV3Lightning

        cfg = FlowConfig(hv_dim=256, num_flows=4, hidden_dim=256, num_hidden_layers=2)
        model = RealNVPV3Lightning(cfg).cuda()

        # Sample from the model (returns z and log probs)
        z, log_probs = model.sample(8)

        assert z.device.type == "cuda"
        assert z.shape == (8, 512)
        assert log_probs.device.type == "cuda"
        assert log_probs.shape == (8,)


if __name__ == "__main__":
    # Run with verbose output
    pytest.main([__file__, "-v", "-s"])
