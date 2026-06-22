"""Regime-conditioned FiLM modulation."""

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


class RegimeFiLM(BaseVETCModule):
    """Apply FiLM conditioning to fused features using a regime token.

    Inputs:
        features: [B, 256]
        regime_token: [B, 96]

    Output:
        conditioned: [B, 256]
    """

    def __init__(self, *, validate_finite: bool = True) -> None:
        super().__init__()
        self.validate_finite = validate_finite
        self.gamma = nn.Linear(C.REGIME_TOKEN_DIM, C.FUSED_DIM)
        self.beta = nn.Linear(C.REGIME_TOKEN_DIM, C.FUSED_DIM)
        self._reset_parameters()

    def forward(self, features: torch.Tensor, regime_token: torch.Tensor) -> torch.Tensor:
        """Return FiLM-conditioned features with shape [B, 256]."""

        # features: [B, 256]; regime_token: [B, 96]
        check_shape(features, C.FUSED_SHAPE, "features")
        check_shape(regime_token, C.REGIME_TOKEN_SHAPE, "regime_token")
        check_batch_match(features, regime_token)
        if self.validate_finite:
            check_finite(features, "features")
            check_finite(regime_token, "regime_token")

        gamma = self.gamma(regime_token)
        beta = self.beta(regime_token)
        conditioned = features * (1.0 + gamma) + beta

        # conditioned: [B, 256]
        check_shape(conditioned, C.FUSED_SHAPE, "conditioned")
        if self.validate_finite:
            check_finite(conditioned, "conditioned")

        return conditioned

    def _reset_parameters(self) -> None:
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)


__all__ = ["RegimeFiLM"]

