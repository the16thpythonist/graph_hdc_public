"""Tests for HDCAutoencoder save/load and loss functions."""
from pathlib import Path

import pytest
import torch

from graph_hdc.models.autoencoder import (
    BerHuReconstructionLoss,
    HDCAutoencoder,
    MAEReconstructionLoss,
    MSEReconstructionLoss,
)

DATA_DIM = 64
LATENT_DIM = 16
ENCODER_HIDDEN = [32]


@pytest.fixture
def model():
    return HDCAutoencoder(
        data_dim=DATA_DIM,
        latent_dim=LATENT_DIM,
        encoder_hidden_dims=ENCODER_HIDDEN,
    )


@pytest.fixture
def sample_input():
    return torch.randn(4, DATA_DIM)


class TestSaveLoad:

    def test_save_creates_file(self, model, tmp_path):
        path = tmp_path / "model.ckpt"
        result = model.save(path)
        assert path.is_file()
        assert isinstance(result, Path)

    def test_load_restores_hparams(self, model, tmp_path):
        path = tmp_path / "model.ckpt"
        model.save(path)
        loaded = HDCAutoencoder.load(path)
        assert loaded.hparams["data_dim"] == DATA_DIM
        assert loaded.hparams["latent_dim"] == LATENT_DIM
        assert loaded.hparams["encoder_hidden_dims"] == ENCODER_HIDDEN

    def test_save_load_roundtrip(self, model, sample_input, tmp_path):
        path = tmp_path / "model.ckpt"
        model.eval()
        with torch.no_grad():
            original_out = model(sample_input)["x_hat"]

        model.save(path)
        loaded = HDCAutoencoder.load(path)
        with torch.no_grad():
            loaded_out = loaded(sample_input)["x_hat"]

        assert torch.allclose(original_out, loaded_out, atol=1e-6)

    @pytest.mark.parametrize("loss_type", ["mse", "mae", "berhu"])
    def test_save_load_each_loss_type(self, loss_type, sample_input, tmp_path):
        model = HDCAutoencoder(
            data_dim=DATA_DIM,
            latent_dim=LATENT_DIM,
            encoder_hidden_dims=ENCODER_HIDDEN,
            recon_loss_type=loss_type,
        )
        model.eval()
        path = tmp_path / "model.ckpt"

        with torch.no_grad():
            original_out = model(sample_input)["x_hat"]

        model.save(path)
        loaded = HDCAutoencoder.load(path)
        assert loaded.recon_loss_type == loss_type

        with torch.no_grad():
            loaded_out = loaded(sample_input)["x_hat"]

        assert torch.allclose(original_out, loaded_out, atol=1e-6)

    def test_save_load_vae(self, sample_input, tmp_path):
        model = HDCAutoencoder(
            data_dim=DATA_DIM,
            latent_dim=LATENT_DIM,
            encoder_hidden_dims=ENCODER_HIDDEN,
            variational=True,
        )
        model.eval()
        path = tmp_path / "model.ckpt"
        model.save(path)
        loaded = HDCAutoencoder.load(path)
        assert loaded.variational is True
        assert loaded.hparams["variational"] is True


class TestLossFunctions:

    def test_mse_shape(self, sample_input):
        target = torch.randn_like(sample_input)
        loss = MSEReconstructionLoss()(sample_input, target)
        assert loss.shape == (sample_input.shape[0],)

    def test_mae_shape(self, sample_input):
        target = torch.randn_like(sample_input)
        loss = MAEReconstructionLoss()(sample_input, target)
        assert loss.shape == (sample_input.shape[0],)

    def test_berhu_shape(self, sample_input):
        target = torch.randn_like(sample_input)
        loss = BerHuReconstructionLoss()(sample_input, target)
        assert loss.shape == (sample_input.shape[0],)

    def test_mse_zero_on_identical(self):
        x = torch.randn(4, 32)
        loss = MSEReconstructionLoss()(x, x)
        assert torch.allclose(loss, torch.zeros(4))

    def test_mae_zero_on_identical(self):
        x = torch.randn(4, 32)
        loss = MAEReconstructionLoss()(x, x)
        assert torch.allclose(loss, torch.zeros(4))

    def test_berhu_zero_on_identical(self):
        x = torch.randn(4, 32)
        loss = BerHuReconstructionLoss()(x, x)
        assert torch.allclose(loss, torch.zeros(4))

    def test_berhu_all_below_threshold_matches_mae(self):
        """With c_fraction=1.0, all errors are <= c, so BerHu equals MAE."""
        x = torch.randn(4, 32)
        target = torch.randn(4, 32)
        berhu = BerHuReconstructionLoss(c_fraction=1.0)(x, target)
        mae = MAEReconstructionLoss()(x, target)
        assert torch.allclose(berhu, mae, atol=1e-6)
