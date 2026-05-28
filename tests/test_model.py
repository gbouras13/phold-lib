"""Unit tests for pholdlib.prostt5.model.

CNN and toCPU are pure PyTorch and do not use transformers at all; they just
happen to live in the same file as get_T5_model / load_predictor.  The
conftest mocks the `transformers` package so model.py can be imported and the
CNN/toCPU logic tested without a working HuggingFace install.

get_T5_model and load_predictor integration-level behaviour is covered in
test_integration.py.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

# conftest.py has already set up the transformers mock if needed
from pholdlib.prostt5.model import CNN, load_predictor, toCPU


# ===========================================================================
# CNN architecture
# ===========================================================================

class TestCNN:
    def test_is_nn_module(self):
        assert isinstance(CNN(), nn.Module)

    def test_forward_output_shape(self):
        """(B, L, 1024) embeddings → (B, 20, L) logits."""
        B, L, F = 2, 15, 1024
        cnn = CNN()
        x = torch.randn(B, L, F)
        out = cnn(x)
        assert out.shape == (B, 20, L)

    def test_forward_single_residue(self):
        """Sequence of length 1 should not cause a padding/kernel error."""
        cnn = CNN()
        x = torch.randn(1, 1, 1024)
        out = cnn(x)
        assert out.shape == (1, 20, 1)

    def test_forward_long_sequence(self):
        """300-residue batch should run without OOM on CPU."""
        cnn = CNN()
        x = torch.randn(1, 300, 1024)
        out = cnn(x)
        assert out.shape == (1, 20, 300)

    def test_output_is_float_tensor(self):
        cnn = CNN()
        out = cnn(torch.randn(1, 5, 1024))
        assert out.dtype == torch.float32

    def test_eval_mode_no_gradient(self):
        cnn = CNN().eval()
        x = torch.randn(1, 5, 1024)
        with torch.no_grad():
            out = cnn(x)
        assert out.shape == (1, 20, 5)

    def test_classifier_has_two_conv_layers(self):
        cnn = CNN()
        convs = [m for m in cnn.classifier if isinstance(m, nn.Conv2d)]
        assert len(convs) == 2

    def test_first_conv_shape(self):
        cnn = CNN()
        conv = next(m for m in cnn.classifier if isinstance(m, nn.Conv2d))
        assert conv.in_channels == 1024
        assert conv.out_channels == 32

    def test_last_conv_outputs_20_classes(self):
        cnn = CNN()
        convs = [m for m in cnn.classifier if isinstance(m, nn.Conv2d)]
        assert convs[-1].out_channels == 20

    def test_half_precision_forward(self):
        """CNN should work in FP16 (relevant for GPU inference)."""
        cnn = CNN().half()
        x = torch.randn(1, 5, 1024).half()
        out = cnn(x)
        assert out.dtype == torch.float16


# ===========================================================================
# toCPU
# ===========================================================================

class TestToCPU:
    def test_1d_tensor_returns_numpy(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        result = toCPU(t)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_2d_tensor_squeezes_trailing_dim_of_1(self):
        # shape (5, 1) → squeeze(dim=-1) → (5,)
        t = torch.ones(5, 1)
        result = toCPU(t)
        assert result.shape == (5,)

    def test_2d_tensor_no_squeeze_when_last_dim_gt_1(self):
        # shape (5, 3) → squeeze(dim=-1) is a no-op
        t = torch.ones(5, 3)
        result = toCPU(t)
        assert result.shape == (5, 3)

    def test_3d_tensor(self):
        # shape (1, 5, 1) → squeeze(dim=-1) → (1, 5)
        t = torch.ones(1, 5, 1)
        result = toCPU(t)
        assert result.shape == (1, 5)

    def test_result_is_numpy_array(self):
        result = toCPU(torch.randn(3, 3))
        assert isinstance(result, np.ndarray)

    def test_detaches_from_graph(self):
        t = torch.randn(3, requires_grad=True)
        result = toCPU(t)
        # Should not raise – tensor is detached before numpy conversion
        assert isinstance(result, np.ndarray)


# ===========================================================================
# load_predictor
# ===========================================================================

class TestLoadPredictor:
    def _save_state_dict(self, path: Path):
        """Save a randomly initialised CNN state dict to path."""
        cnn = CNN()
        torch.save(cnn.state_dict(), path)

    def _save_checkpoint(self, path: Path):
        """Save a checkpoint with the 'state_dict' key (phold style)."""
        cnn = CNN()
        torch.save({"state_dict": cnn.state_dict(), "epoch": 1}, path)

    def test_loads_state_dict_file(self, tmp_path):
        ckpt = tmp_path / "predictor.pth"
        self._save_state_dict(ckpt)
        model = load_predictor(ckpt, torch.device("cpu"))
        assert isinstance(model, CNN)

    def test_loads_pt_checkpoint(self, tmp_path):
        ckpt = tmp_path / "predictor.pt"
        self._save_checkpoint(ckpt)
        model = load_predictor(ckpt, torch.device("cpu"))
        assert isinstance(model, CNN)

    def test_loaded_model_is_in_eval_mode(self, tmp_path):
        ckpt = tmp_path / "predictor.pth"
        self._save_state_dict(ckpt)
        model = load_predictor(ckpt, torch.device("cpu"))
        assert not model.training

    def test_loaded_model_produces_correct_output_shape(self, tmp_path):
        ckpt = tmp_path / "predictor.pth"
        self._save_state_dict(ckpt)
        model = load_predictor(ckpt, torch.device("cpu"))
        out = model(torch.randn(1, 10, 1024))
        assert out.shape == (1, 20, 10)
