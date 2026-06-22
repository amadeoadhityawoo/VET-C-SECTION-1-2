"""Regime-conditioned fusion for VET-C RFC-1008."""

from __future__ import annotations

import torch
from torch import nn

from vetc.core import constants as C
from vetc.core.interfaces import BaseVETCModule
from vetc.core.tensor_checks import (
    check_batch_match,
    check_finite,
    check_shape,
)
from vetc.fusion.regime_film import RegimeFiLM
from vetc.fusion.spatial_pooler import SpatialPooler


class RegimeConditionedFusion(BaseVETCModule):
    """Fuse external, internal, and regime features into [B, 256].

    Inputs:
        z_ext: [B, 128, 16, 16]
        z_int: [B, 128]
        regime_token: [B, 96] or None

    Output:
        fused: [B, 256]
    """

    def __init__(self, *, validate_finite: bool = True) -> None:
        super().__init__()
        self.validate_finite = validate_finite
        self.spatial_pooler = SpatialPooler(validate_finite=validate_finite)
        self.regime_film = RegimeFiLM(validate_finite=validate_finite)
        self.mlp = nn.Sequential(
            nn.Linear(C.FUSED_DIM, C.FUSED_DIM),
            nn.GELU(),
            nn.Linear(C.FUSED_DIM, C.FUSED_DIM),
        )

    def forward(
        self,
        z_ext: torch.Tensor,
        z_int: torch.Tensor,
        regime_token: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a regime-conditioned fused representation."""

        # z_ext: [B, 128, 16, 16]; z_int: [B, 128]
        check_shape(z_ext, C.Z_EXT_SHAPE, "z_ext")
        check_shape(z_int, C.Z_INT_SHAPE, "z_int")
        check_batch_match(z_ext, z_int)
        if self.validate_finite:
            check_finite(z_ext, "z_ext")
            check_finite(z_int, "z_int")

        if regime_token is None:
            regime_token = z_int.new_zeros((z_int.shape[0], C.REGIME_TOKEN_DIM))
        else:
            check_shape(regime_token, C.REGIME_TOKEN_SHAPE, "regime_token")
            check_batch_match(z_int, regime_token)
            if self.validate_finite:
                check_finite(regime_token, "regime_token")

        pooled_ext = self.spatial_pooler(z_ext)

        # pooled_ext: [B, 128]; z_int: [B, 128]; features: [B, 256]
        features = torch.cat((pooled_ext, z_int), dim=1)
        check_shape(features, C.FUSED_SHAPE, "features")

        conditioned = self.regime_film(features, regime_token)
        fused = self.mlp(conditioned)

        # fused: [B, 256]
        check_shape(fused, C.FUSED_SHAPE, "fused")
        if self.validate_finite:
            check_finite(fused, "fused")

        return fused


__all__ = ["RegimeConditionedFusion"]

