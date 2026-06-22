import torch

from vetc.core.types import TerrainOutput
from vetc.terrain.fourier_features import FourierFeatures
from vetc.terrain.vet_terrain_generator import VETTerrainGenerator


def test_fourier_features_preserve_batch_and_are_finite():
    features = FourierFeatures()
    x = torch.randn(2, 16)

    output = features(x)

    assert output.shape[0] == 2
    assert output.shape[1] > 16
    assert torch.isfinite(output).all().item()


def test_vet_terrain_generator_outputs_expected_shape_and_finite_values():
    generator = VETTerrainGenerator()
    h = torch.randn(2, 512)
    regime_token = torch.randn(2, 96)
    surprise = torch.randn(2, 1)

    output = generator(h, regime_token, surprise)

    assert isinstance(output, TerrainOutput)
    assert output.psi_field.shape == (2, 5)
    assert torch.isfinite(output.psi_field).all().item()


def test_vet_terrain_generator_output_ranges_are_stable():
    generator = VETTerrainGenerator()
    h = torch.randn(2, 512)
    regime_token = torch.randn(2, 96)
    surprise = torch.randn(2, 1)

    psi_field = generator(h, regime_token, surprise).psi_field

    assert torch.all((psi_field[:, 0] >= -1.0) & (psi_field[:, 0] <= 1.0))
    assert torch.all((psi_field[:, 1] >= 0.0) & (psi_field[:, 1] <= 1.0))
    assert torch.all((psi_field[:, 2] >= 0.0) & (psi_field[:, 2] <= 1.0))
    assert torch.all(psi_field[:, 3] >= 0.0)
    assert torch.isfinite(psi_field[:, 3]).all().item()
    assert torch.all((psi_field[:, 4] >= 0.0) & (psi_field[:, 4] <= 1.0))


def test_vet_terrain_generator_import_path():
    from vetc.terrain.vet_terrain_generator import VETTerrainGenerator

    assert VETTerrainGenerator.__name__ == "VETTerrainGenerator"

