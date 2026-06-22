"""Dual perception encoder for VET-C RFC-1007."""

from __future__ import annotations

import torch

from vetc.core import constants as C
from vetc.core.interfaces import BaseVETCModule
from vetc.core.tensor_checks import check_batch_match, check_shape
from vetc.core.types import EncoderOutput, ObservationTensor
from vetc.encoder.external_encoder import ExternalEncoder
from vetc.encoder.internal_encoder import InternalEncoder


class DualPerceptionEncoder(BaseVETCModule):
    """Run external and internal encoders side by side.

    Inputs:
        ext_obs: [B, 6, 64, 64]
        int_obs: [B, 32, 11]

    Outputs:
        z_ext: [B, 128, 16, 16]
        z_int: [B, 128]

    This module intentionally does not fuse the two latent streams.
    """

    def __init__(
        self,
        *,
        expected_dtype: torch.dtype = torch.float32,
        validate_finite: bool = True,
    ) -> None:
        super().__init__()
        self.external_encoder = ExternalEncoder(
            expected_dtype=expected_dtype,
            validate_finite=validate_finite,
        )
        self.internal_encoder = InternalEncoder(
            expected_dtype=expected_dtype,
            validate_finite=validate_finite,
        )

    def forward(self, ext_obs: torch.Tensor, int_obs: torch.Tensor) -> EncoderOutput:
        """Encode external and internal observations without fusion."""

        # ext_obs: [B, 6, 64, 64]; int_obs: [B, 32, 11]
        check_shape(ext_obs, C.EXT_OBS_SHAPE, "ext_obs")
        check_shape(int_obs, C.INT_OBS_SHAPE, "int_obs")
        check_batch_match(ext_obs, int_obs)

        observations = ObservationTensor(ext_obs=ext_obs, int_obs=int_obs)
        z_ext = self.external_encoder(observations)
        z_int = self.internal_encoder(observations)

        # z_ext: [B, 128, 16, 16]; z_int: [B, 128]
        check_shape(z_ext, C.Z_EXT_SHAPE, "z_ext")
        check_shape(z_int, C.Z_INT_SHAPE, "z_int")
        check_batch_match(z_ext, z_int)

        return EncoderOutput(z_ext=z_ext, z_int=z_int)


__all__ = ["DualPerceptionEncoder"]

