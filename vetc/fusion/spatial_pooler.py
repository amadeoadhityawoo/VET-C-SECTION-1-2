"""Spatial pooling for external latent maps."""

from __future__ import annotations

import torch
from torch import nn

from vetc.core import constants as C
from vetc.core.interfaces import BaseVETCModule
from vetc.core.tensor_checks import check_finite, check_shape


class SpatialPooler(BaseVETCModule):
    """Pool ``z_ext`` from [B, 128, 16, 16] to [B, 128]."""

    def __init__(self, *, validate_finite: bool = True) -> None:
        super().__init__()
        self.validate_finite = validate_finite
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, z_ext: torch.Tensor) -> torch.Tensor:
        """Return pooled external features with shape [B, 128]."""

        # z_ext: [B, 128, 16, 16]
        check_shape(z_ext, C.Z_EXT_SHAPE, "z_ext")
        if self.validate_finite:
            check_finite(z_ext, "z_ext")

        pooled_ext = self.flatten(self.pool(z_ext))

        # pooled_ext: [B, 128]
        check_shape(pooled_ext, (None, C.Z_EXT_CHANNELS), "pooled_ext")
        if self.validate_finite:
            check_finite(pooled_ext, "pooled_ext")

        return pooled_ext


__all__ = ["SpatialPooler"]

