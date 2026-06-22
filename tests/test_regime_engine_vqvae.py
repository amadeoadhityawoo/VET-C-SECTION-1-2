import torch

from vetc.regime.regime_engine_vqvae import RegimeEngineVQVAE
from vetc.regime.vector_quantizer import VectorQuantizer


def test_vector_quantizer_outputs_expected_shapes_and_finite_values():
    quantizer = VectorQuantizer()
    x = torch.randn(2, 96)

    output = quantizer(x)

    assert output["quantized"].shape == (2, 96)
    assert output["loss"].shape == ()
    assert output["code_indices"].shape == (2,)
    assert torch.isfinite(output["quantized"]).all().item()
    assert torch.isfinite(output["loss"]).all().item()


def test_regime_engine_vqvae_outputs_expected_shapes_and_finite_values():
    engine = RegimeEngineVQVAE()
    h = torch.randn(2, 512)
    surprise = torch.randn(2, 1)

    output = engine(h, surprise)

    assert output["regime_token"].shape == (2, 96)
    assert output["commit_loss"].shape == ()
    assert output["code_indices"].shape == (2,)
    assert output["quantized"].shape == (2, 96)
    assert torch.isfinite(output["regime_token"]).all().item()
    assert torch.isfinite(output["commit_loss"]).all().item()
    assert torch.isfinite(output["quantized"]).all().item()


def test_regime_engine_vqvae_import_path():
    from vetc.regime.regime_engine_vqvae import RegimeEngineVQVAE

    assert RegimeEngineVQVAE.__name__ == "RegimeEngineVQVAE"

