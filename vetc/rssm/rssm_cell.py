"""RSSM cell for VET-C RFC-1009."""

from __future__ import annotations

import torch
from torch import nn

from vetc.core import constants as C
from vetc.core.interfaces import BaseVETCModule
from vetc.core.tensor_checks import check_batch_match, check_finite, check_shape
from vetc.rssm.posterior_network import PosteriorNetwork
from vetc.rssm.prior_network import PriorNetwork


RSSMCellOutput = dict[str, torch.Tensor]


class RSSMCell(BaseVETCModule):
    """Single RSSM transition/update cell.

    Inputs:
        fused: [B, 256]
        h_prev: [B, 512]
        z_prev: [B, 64]

    Outputs:
        h: [B, 512]
        z_prior: [B, 64]
        z_post: [B, 64]
        prior_mu/prior_logvar: [B, 64]
        post_mu/post_logvar: [B, 64]
    """

    def __init__(
        self,
        *,
        fused_dim: int = C.FUSED_DIM,
        hidden_dim: int = C.RSSM_HIDDEN_DIM,
        latent_dim: int = C.RSSM_LATENT_DIM,
        validate_finite: bool = True,
    ) -> None:
        super().__init__()
        self.fused_dim = fused_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.validate_finite = validate_finite
        self.gru = nn.GRUCell(
            input_size=fused_dim + latent_dim,
            hidden_size=hidden_dim,
        )
        self.prior_network = PriorNetwork(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            validate_finite=validate_finite,
        )
        self.posterior_network = PosteriorNetwork(
            fused_dim=fused_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            validate_finite=validate_finite,
        )

    def forward(
        self,
        fused: torch.Tensor,
        h_prev: torch.Tensor,
        z_prev: torch.Tensor,
    ) -> RSSMCellOutput:
        """Run one RSSM cell step."""

        # fused: [B, 256]; h_prev: [B, 512]; z_prev: [B, 64]
        check_shape(fused, (None, self.fused_dim), "fused")
        check_shape(h_prev, (None, self.hidden_dim), "h_prev")
        check_shape(z_prev, (None, self.latent_dim), "z_prev")
        check_batch_match(fused, h_prev, z_prev)
        if self.validate_finite:
            check_finite(fused, "fused")
            check_finite(h_prev, "h_prev")
            check_finite(z_prev, "z_prev")

        gru_input = torch.cat((fused, z_prev), dim=1)
        h = self.gru(gru_input, h_prev)

        prior_mu, prior_logvar = self.prior_network(h)
        post_mu, post_logvar = self.posterior_network(h, fused)
        z_prior = self._sample(prior_mu, prior_logvar)
        z_post = self._sample(post_mu, post_logvar)

        output = {
            "h": h,
            "z_prior": z_prior,
            "z_post": z_post,
            "prior_mu": prior_mu,
            "prior_logvar": prior_logvar,
            "post_mu": post_mu,
            "post_logvar": post_logvar,
        }
        self._validate_output(output)
        return output

    def _sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _validate_output(self, output: RSSMCellOutput) -> None:
        expected_shapes = {
            "h": (None, self.hidden_dim),
            "z_prior": (None, self.latent_dim),
            "z_post": (None, self.latent_dim),
            "prior_mu": (None, self.latent_dim),
            "prior_logvar": (None, self.latent_dim),
            "post_mu": (None, self.latent_dim),
            "post_logvar": (None, self.latent_dim),
        }
        for name, shape in expected_shapes.items():
            check_shape(output[name], shape, name)
            if self.validate_finite:
                check_finite(output[name], name)


__all__ = ["RSSMCell", "RSSMCellOutput"]
