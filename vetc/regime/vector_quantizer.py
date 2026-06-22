"""Vector quantizer for VET-C RFC-1012."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from vetc.core import constants as C
from vetc.core.interfaces import BaseVETCModule
from vetc.core.tensor_checks import check_finite, check_shape


VectorQuantizerOutput = dict[str, torch.Tensor]


class VectorQuantizer(BaseVETCModule):
    """Nearest-code vector quantizer with a straight-through estimator.

    Input:
        x: [B, 96]

    Outputs:
        quantized: [B, 96]
        loss: scalar tensor
        code_indices: [B]
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
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.beta = beta
        self.validate_finite = validate_finite
        self.codebook = nn.Embedding(num_codes, code_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)

    def forward(self, x: torch.Tensor) -> VectorQuantizerOutput:
        """Quantize ``x`` to the nearest codebook vectors."""

        # x: [B, 96]
        check_shape(x, (None, self.code_dim), "x")
        if self.validate_finite:
            check_finite(x, "x")

        codebook = self.codebook.weight
        distances = (
            x.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * x @ codebook.t()
            + codebook.pow(2).sum(dim=1)
        )
        code_indices = torch.argmin(distances, dim=1)
        quantized = self.codebook(code_indices)
        quantized_st = x + (quantized - x).detach()

        codebook_loss = F.mse_loss(quantized, x.detach())
        commitment_loss = self.beta * F.mse_loss(x, quantized.detach())
        total_vq_loss = codebook_loss + commitment_loss

        output = {
            "quantized": quantized_st,
            "loss": total_vq_loss,
            "code_indices": code_indices,
        }
        self._validate_output(output)
        return output

    def _validate_output(self, output: VectorQuantizerOutput) -> None:
        check_shape(output["quantized"], (None, self.code_dim), "quantized")
        check_shape(output["loss"], (), "loss")
        check_shape(output["code_indices"], (None,), "code_indices")
        if self.validate_finite:
            check_finite(output["quantized"], "quantized")
            check_finite(output["loss"], "loss")


__all__ = ["VectorQuantizer", "VectorQuantizerOutput"]

