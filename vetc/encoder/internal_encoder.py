"""Internal observation encoder for VET-C RFC-1006."""

from __future__ import annotations

import torch
from torch import nn

from vetc.core import constants as C
from vetc.core.interfaces import BaseVETCModule
from vetc.core.tensor_checks import check_dtype, check_finite, check_shape
from vetc.core.types import ObservationTensor


class InternalEncoder(BaseVETCModule):
    """Encode internal sensor history into a compact latent vector.

    Input:
        observations.int_obs: [B, 32, 11]

    Output:
        z_int: [B, 128]
    """

    def __init__(
        self,
        *,
        expected_dtype: torch.dtype = torch.float32,
        validate_finite: bool = True,
    ) -> None:
        super().__init__()
        self.expected_dtype = expected_dtype
        self.validate_finite = validate_finite
        self.encoder = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(C.SENSOR_HISTORY_STEPS * C.INT_OBS_DIM, 256),
            nn.GELU(),
            nn.Linear(256, C.Z_INT_DIM),
        )

    def forward(self, observations: ObservationTensor) -> torch.Tensor:
        """Return ``z_int`` with shape [B, 128]."""

        check_shape(observations.int_obs, C.INT_OBS_SHAPE, "int_obs")
        check_dtype(observations.int_obs, self.expected_dtype, "int_obs")
        if self.validate_finite:
            check_finite(observations.int_obs, "int_obs")

        z_int = self.encoder(observations.int_obs)
        check_shape(z_int, C.Z_INT_SHAPE, "z_int")
        if self.validate_finite:
            check_finite(z_int, "z_int")

        return z_int


__all__ = ["InternalEncoder"]

