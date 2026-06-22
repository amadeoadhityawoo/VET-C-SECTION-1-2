import torch

from vetc.runtime.vetc_system import VETCSystem


def _dummy_inputs(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "rgb": torch.randn(batch_size, 3, 64, 64),
        "depth": torch.randn(batch_size, 1, 64, 64),
        "imu": torch.randn(batch_size, 32, 6),
        "battery": torch.randn(batch_size, 32, 1),
        "motor_state": torch.randn(batch_size, 32, 4),
    }


def test_vetc_system_forward_outputs_expected_shapes_and_finite_tensors():
    system = VETCSystem()
    system.eval()

    with torch.no_grad():
        output = system(**_dummy_inputs())

    expected_shapes = {
        "action": (2, 4),
        "ext_obs": (2, 6, 64, 64),
        "int_obs": (2, 32, 11),
        "z_ext": (2, 128, 16, 16),
        "z_int": (2, 128),
        "fused": (2, 256),
        "h": (2, 512),
        "z_prior": (2, 64),
        "z_post": (2, 64),
        "surprise": (2, 1),
        "kl_map": (2, 64),
        "safe_sigma": (2, 64),
        "regime_token": (2, 96),
        "psi_field": (2, 5),
    }
    for key, shape in expected_shapes.items():
        assert output[key].shape == shape
        assert torch.isfinite(output[key]).all().item()
    assert output["commit_loss"].shape == ()
    assert torch.isfinite(output["commit_loss"]).all().item()

