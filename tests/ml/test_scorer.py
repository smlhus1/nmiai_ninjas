"""Tests for ScorerMLP."""
import pytest
import torch

from ml.scorer import ScorerMLP


class TestScorerMLP:
    def test_param_count(self):
        model = ScorerMLP()
        n_params = sum(p.numel() for p in model.parameters())
        assert 100_000 <= n_params <= 150_000, f"Param count {n_params} outside 100K-150K range"

    def test_forward_shape(self):
        model = ScorerMLP()
        x = torch.randn(1200, 48)
        y = model(x)
        assert y.shape == (1200, 1)

    def test_output_range(self):
        model = ScorerMLP()
        x = torch.randn(500, 48)
        y = model(x)
        assert (y >= 0.0).all(), f"Output below 0: min={y.min()}"
        assert (y <= 1.0).all(), f"Output above 1: max={y.max()}"

    def test_single_sample(self):
        model = ScorerMLP()
        x = torch.randn(1, 48)
        y = model(x)
        assert y.shape == (1, 1)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    def test_cuda(self):
        model = ScorerMLP().cuda()
        x = torch.randn(100, 48, device="cuda")
        y = model(x)
        assert y.shape == (100, 1)
        assert y.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    def test_cpu_gpu_equivalence(self):
        model = ScorerMLP()
        model.eval()
        x = torch.randn(50, 48)

        with torch.no_grad():
            cpu_out = model(x)

        model_gpu = ScorerMLP()
        model_gpu.load_state_dict(model.state_dict())
        model_gpu = model_gpu.cuda().eval()

        with torch.no_grad():
            gpu_out = model_gpu(x.cuda())

        diff = (cpu_out - gpu_out.cpu()).abs().max()
        assert diff < 1e-5, f"CPU/GPU diff {diff} exceeds 1e-5"
