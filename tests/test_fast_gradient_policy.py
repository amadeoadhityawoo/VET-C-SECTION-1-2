import torch

from vetc.core.types import ActionCommand
from vetc.policy.fast_gradient_policy import FastGradientPolicy


def test_fast_gradient_policy_outputs_expected_shape_and_finite_action():
    policy = FastGradientPolicy()
    psi_field = torch.randn(2, 5)

    output = policy(psi_field)

    assert isinstance(output, ActionCommand)
    assert output.action.shape == (2, 4)
    assert torch.isfinite(output.action).all().item()


def test_fast_gradient_policy_action_range_is_normalized():
    policy = FastGradientPolicy()
    psi_field = torch.randn(2, 5) * 10.0

    action = policy(psi_field).action

    assert torch.all(action >= -1.0)
    assert torch.all(action <= 1.0)


def test_fast_gradient_policy_import_path():
    from vetc.policy.fast_gradient_policy import FastGradientPolicy

    assert FastGradientPolicy.__name__ == "FastGradientPolicy"

