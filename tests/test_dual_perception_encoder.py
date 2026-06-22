import torch

from vetc.core.types import EncoderOutput
from vetc.encoder.dual_perception_encoder import DualPerceptionEncoder


def test_dual_perception_encoder_outputs_expected_shapes_and_finite_tensors():
    encoder = DualPerceptionEncoder()
    ext_obs = torch.randn(2, 6, 64, 64)
    int_obs = torch.randn(2, 32, 11)

    output = encoder(ext_obs, int_obs)

    assert isinstance(output, EncoderOutput)
    assert output.z_ext.shape == (2, 128, 16, 16)
    assert output.z_int.shape == (2, 128)
    assert torch.isfinite(output.z_ext).all().item()
    assert torch.isfinite(output.z_int).all().item()

