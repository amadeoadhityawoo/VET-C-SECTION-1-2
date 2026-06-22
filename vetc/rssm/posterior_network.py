"""Posterior network for VET-C RFC-1009."""

from __future__ import annotations

import torch
from torch import nn

from vetc.core import constants as C
from vetc.core.interfaces import BaseVETCModule
from vetc.core.tensor_checks import check_batch_match, check_finite, check_shape


class PosteriorNetwork(BaseVETCModule):
    """Predict posterior latent distribution parameters from state and fusion.

    Inputs:
        h: [B, 512]
        fused: [B, 256]

    Outputs:
        post_mu: [B, 64]
        post_logvar: [B, 64]
    """

    def __init__(
        self,
        *,
        fused_dim: int = C.FUSED_DIM,
        hidden_dim: int = C.RSSM_HIDDEN_DIM,
        latent_dim: int = C.RSSM_LATENT_DIM,
        logvar_min: float = -10.0,
        logvar_max: float = 2.0,
        validate_finite: bool = True,
    ) -> None:
        super().__init__()
        self.fused_dim = fused_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.validate_finite = validate_finite
        self.network = nn.Sequential(
            nn.Linear(hidden_dim + fused_dim, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim * 2),
        )

    def forward(
        self,
        h: torch.Tensor,
        fused: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return posterior ``mu`` and clamped ``logvar`` tensors."""

        # h: [B, 512]; fused: [B, 256]
        check_shape(h, (None, self.hidden_dim), "h")
        check_shape(fused, (None, self.fused_dim), "fused")
        check_batch_match(h, fused)
        if self.validate_finite:
            check_finite(h, "h")
            check_finite(fused, "fused")

        params = self.network(torch.cat((h, fused), dim=1))
        mu, logvar = params.chunk(2, dim=1)
        logvar = logvar.clamp(self.logvar_min, self.logvar_max)

        # mu/logvar: [B, 64]
        check_shape(mu, (None, self.latent_dim), "post_mu")
        check_shape(logvar, (None, self.latent_dim), "post_logvar")
        if self.validate_finite:
            check_finite(mu, "post_mu")
            check_finite(logvar, "post_logvar")

        return mu, logvar


__all__ = ["PosteriorNetwork"]
