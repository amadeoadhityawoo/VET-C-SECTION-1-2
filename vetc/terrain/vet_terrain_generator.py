"""VET terrain generator for RFC-1013."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from vetc.core import constants as C
from vetc.core.interfaces import BaseVETCModule
from vetc.core.tensor_checks import check_batch_match, check_finite, check_shape
from vetc.core.types import TerrainOutput
from vetc.terrain.fourier_features import FourierFeatures


class VETTerrainGenerator(BaseVETCModule):
    """Generate a five-element psi field from state, regime, and surprise.

    Inputs:
        h: [B, 512]
        regime_token: [B, 96]
        surprise: [B, 1]

    Output:
        psi_field: [B, 5]
            0: value in [-1, 1]
            1: entropy in [0, 1]
            2: confidence in [0, 1]
            3: prediction_precision >= 0
            4: homeostatic_weight in [0, 1]
    """

    def __init__(
        self,
        *,
        projection_dim: int = 128,
        num_frequency_bands: int = 4,
        validate_finite: bool = True,
    ) -> None:
        super().__init__()
        self.projection_dim = projection_dim
        self.validate_finite = validate_finite
        input_dim = C.RSSM_HIDDEN_DIM + C.REGIME_TOKEN_DIM + C.SURPRISE_DIM
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.GELU(),
        )
        self.fourier_features = FourierFeatures(
            num_bands=num_frequency_bands,
            validate_finite=validate_finite,
        )
        fourier_dim = projection_dim * (1 + 2 * num_frequency_bands)
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, C.PSI_FIELD_DIM),
        )

    def forward(
        self,
        h: torch.Tensor,
        regime_token: torch.Tensor,
        surprise: torch.Tensor,
    ) -> TerrainOutput:
        """Return stabilized terrain field values."""

        # h: [B, 512]; regime_token: [B, 96]; surprise: [B, 1]
        check_shape(h, C.RSSM_H_SHAPE, "h")
        check_shape(regime_token, C.REGIME_TOKEN_SHAPE, "regime_token")
        check_shape(surprise, C.SURPRISE_SHAPE, "surprise")
        check_batch_match(h, regime_token, surprise)
        if self.validate_finite:
            check_finite(h, "h")
            check_finite(regime_token, "regime_token")
            check_finite(surprise, "surprise")

        fused_inputs = torch.cat((h, regime_token, surprise), dim=1)
        projected = self.input_projection(fused_inputs)
        encoded = self.fourier_features(projected)
        raw_psi = self.mlp(encoded)

        # raw_psi: [B, 5] -> psi_field: [B, 5] with stable element ranges.
        value = torch.tanh(raw_psi[:, 0:1])
        entropy = torch.sigmoid(raw_psi[:, 1:2])
        confidence = torch.sigmoid(raw_psi[:, 2:3])
        prediction_precision = F.softplus(raw_psi[:, 3:4])
        homeostatic_weight = torch.sigmoid(raw_psi[:, 4:5])
        psi_field = torch.cat(
            (
                value,
                entropy,
                confidence,
                prediction_precision,
                homeostatic_weight,
            ),
            dim=1,
        )

        check_shape(psi_field, C.PSI_FIELD_SHAPE, "psi_field")
        if self.validate_finite:
            check_finite(psi_field, "psi_field")

        return TerrainOutput(psi_field=psi_field)


__all__ = ["VETTerrainGenerator"]

