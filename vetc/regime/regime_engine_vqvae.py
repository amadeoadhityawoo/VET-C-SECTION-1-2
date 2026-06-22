"""Regime engine with VQ-VAE-style quantization for VET-C RFC-1012."""

from __future__ import annotations

import torch
from torch import nn

from vetc.core import constants as C
from vetc.core.interfaces import BaseVETCModule
from vetc.core.tensor_checks import check_batch_match, check_finite, check_shape
from vetc.regime.dead_code_manager import DeadCodeManager
from vetc.regime.vector_quantizer import VectorQuantizer


RegimeEngineOutput = dict[str, torch.Tensor]


class RegimeEngineVQVAE(BaseVETCModule):
    """Convert hidden state and surprise into a quantized regime token.

    Inputs:
        h: [B, 512]
        surprise: [B, 1]

    Outputs:
        regime_token: [B, 96]
        commit_loss: scalar tensor
        code_indices: [B]
        quantized: [B, 96]
    """

    def __init__(
        self,
        *,
        num_codes: int = 32,
        code_dim: int = C.REGIME_TOKEN_DIM,
        beta: float = 0.25,
        validate_finite: bool = True,
    ) -> None:
        super().__init__()
        self.code_dim = code_dim
        self.validate_finite = validate_finite
        self.projector = nn.Sequential(
            nn.Linear(C.RSSM_HIDDEN_DIM + C.SURPRISE_DIM, 256),
            nn.GELU(),
            nn.Linear(256, code_dim),
        )
        self.quantizer = VectorQuantizer(
            num_codes=num_codes,
            code_dim=code_dim,
            beta=beta,
            validate_finite=validate_finite,
        )
        self.dead_code_manager = DeadCodeManager(num_codes=num_codes)

    def forward(self, h: torch.Tensor, surprise: torch.Tensor) -> RegimeEngineOutput:
        """Return quantized regime information."""

        # h: [B, 512]; surprise: [B, 1]
        check_shape(h, C.RSSM_H_SHAPE, "h")
        check_shape(surprise, C.SURPRISE_SHAPE, "surprise")
        check_batch_match(h, surprise)
        if self.validate_finite:
            check_finite(h, "h")
            check_finite(surprise, "surprise")

        projected = self.projector(torch.cat((h, surprise), dim=1))
        check_shape(projected, (None, self.code_dim), "projected")
        if self.validate_finite:
            check_finite(projected, "projected")

        quantizer_output = self.quantizer(projected)
        regime_token = quantizer_output["quantized"]
        commit_loss = quantizer_output["loss"]
        code_indices = quantizer_output["code_indices"]
        self.dead_code_manager.update_usage(code_indices)

        output = {
            "regime_token": regime_token,
            "commit_loss": commit_loss,
            "code_indices": code_indices,
            "quantized": regime_token,
        }
        self._validate_output(output)
        return output

    def _validate_output(self, output: RegimeEngineOutput) -> None:
        check_shape(output["regime_token"], C.REGIME_TOKEN_SHAPE, "regime_token")
        check_shape(output["commit_loss"], (), "commit_loss")
        check_shape(output["code_indices"], (None,), "code_indices")
        check_shape(output["quantized"], C.REGIME_TOKEN_SHAPE, "quantized")
        if self.validate_finite:
            check_finite(output["regime_token"], "regime_token")
            check_finite(output["commit_loss"], "commit_loss")
            check_finite(output["quantized"], "quantized")


__all__ = ["RegimeEngineOutput", "RegimeEngineVQVAE"]

