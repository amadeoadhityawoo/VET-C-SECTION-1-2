"""Lightweight Fourier feature expansion for terrain generation."""

from __future__ import annotations

import torch

from vetc.core.interfaces import BaseVETCModule
from vetc.core.tensor_checks import check_finite


class FourierFeatures(BaseVETCModule):
    """Append sinusoidal features to a rank-2 input tensor.

    Input:
        x: [B, D]

    Output:
        features: [B, D * (1 + 2 * num_bands)]
    """

    def __init__(
        self,
        *,
        num_bands: int = 4,
        max_frequency: float = 4.0,
        validate_finite: bool = True,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.validate_finite = validate_finite
        frequencies = torch.linspace(1.0, max_frequency, steps=num_bands)
        self.register_buffer("frequencies", frequencies)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x`` with sine/cosine frequency bands appended."""

        if x.ndim != 2:
            raise ValueError(f"x must have rank 2 [B, D], got shape {tuple(x.shape)}.")
        if self.validate_finite:
            check_finite(x, "x")

        scaled = x.unsqueeze(-1) * self.frequencies
        sin_features = torch.sin(scaled).flatten(start_dim=1)
        cos_features = torch.cos(scaled).flatten(start_dim=1)
        features = torch.cat((x, sin_features, cos_features), dim=1)

        if self.validate_finite:
            check_finite(features, "fourier_features")
        return features


__all__ = ["FourierFeatures"]

