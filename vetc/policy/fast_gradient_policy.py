"""Fast gradient policy for VET-C RFC-1014."""

from __future__ import annotations

import torch
from torch import nn

from vetc.core import constants as C
from vetc.core.interfaces import BaseVETCModule
from vetc.core.tensor_checks import check_finite, check_shape
from vetc.core.types import ActionCommand


class FastGradientPolicy(BaseVETCModule):
    """Convert psi-field terrain signals into normalized drone actions.

    Input:
        psi_field: [B, 5]

    Output:
        action: [B, 4]
            0: thrust
            1: roll
            2: pitch
            3: yaw
    """

    def __init__(self, *, validate_finite: bool = True) -> None:
        super().__init__()
        self.validate_finite = validate_finite
        self.policy = nn.Sequential(
            nn.Linear(C.PSI_FIELD_DIM, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, C.ACTION_DIM),
        )

    def forward(self, psi_field: torch.Tensor) -> ActionCommand:
        """Return normalized action commands in range [-1, 1]."""

        # psi_field: [B, 5]
        check_shape(psi_field, C.PSI_FIELD_SHAPE, "psi_field")
        if self.validate_finite:
            check_finite(psi_field, "psi_field")

        action = torch.tanh(self.policy(psi_field))

        # action: [B, 4]
        check_shape(action, C.ACTION_SHAPE, "action")
        if self.validate_finite:
            check_finite(action, "action")

        return ActionCommand(action=action)


__all__ = ["FastGradientPolicy"]

