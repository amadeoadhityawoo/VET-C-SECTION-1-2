"""External observation encoder for VET-C RFC-1005."""

from __future__ import annotations

import torch
from torch import nn

from vetc.core import constants as C
from vetc.core.interfaces import BaseVETCModule
from vetc.core.tensor_checks import check_dtype, check_finite, check_shape
from vetc.core.types import ObservationTensor


class ExternalEncoder(BaseVETCModule):
    """Encode external observations into a spatial latent feature map.

    Input:
        observations.ext_obs: [B, 6, 64, 64]

    Output:
        z_ext: [B, 128, 16, 16]
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
            nn.Conv2d(C.EXT_OBS_CHANNELS, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, C.Z_EXT_CHANNELS, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )

    def forward(self, observations: ObservationTensor) -> torch.Tensor:
        """Return ``z_ext`` with shape [B, 128, 16, 16]."""

        check_shape(observations.ext_obs, C.EXT_OBS_SHAPE, "ext_obs")
        check_dtype(observations.ext_obs, self.expected_dtype, "ext_obs")
        if self.validate_finite:
            check_finite(observations.ext_obs, "ext_obs")

        z_ext = self.encoder(observations.ext_obs)
        check_shape(z_ext, C.Z_EXT_SHAPE, "z_ext")
        if self.validate_finite:
            check_finite(z_ext, "z_ext")

        return z_ext


__all__ = ["ExternalEncoder"]

