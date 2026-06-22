"""Prediction error evaluation for VET-C RFC-1011."""

from __future__ import annotations

import torch

from vetc.core import constants as C
from vetc.core.interfaces import BaseVETCModule
from vetc.core.tensor_checks import check_batch_match, check_finite, check_shape


PredictionErrorOutput = dict[str, torch.Tensor]


class PredictionErrorEvaluator(BaseVETCModule):
    """Compare posterior and prior latent states.

    Inputs:
        z_post: [B, 64]
        z_prior: [B, 64]
        prior_mu/prior_logvar/post_mu/post_logvar: optional [B, 64]

    Outputs:
        surprise: [B, 1]
        kl_map: [B, 64]
        safe_sigma: [B, 64]
    """

    def __init__(
        self,
        *,
        eps: float = 1e-6,
        logvar_min: float = -10.0,
        logvar_max: float = 2.0,
        validate_finite: bool = True,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.validate_finite = validate_finite

    def forward(
        self,
        z_post: torch.Tensor,
        z_prior: torch.Tensor,
        prior_mu: torch.Tensor | None = None,
        prior_logvar: torch.Tensor | None = None,
        post_mu: torch.Tensor | None = None,
        post_logvar: torch.Tensor | None = None,
    ) -> PredictionErrorOutput:
        """Return surprise, KL/error map, and stable sigma estimate."""

        # z_post/z_prior: [B, 64]
        self._check_latent(z_post, "z_post")
        self._check_latent(z_prior, "z_prior")
        check_batch_match(z_post, z_prior)

        gaussian_params = (prior_mu, prior_logvar, post_mu, post_logvar)
        for name, tensor in zip(
            ("prior_mu", "prior_logvar", "post_mu", "post_logvar"),
            gaussian_params,
        ):
            if tensor is not None:
                self._check_latent(tensor, name)
                check_batch_match(z_post, tensor)

        if all(tensor is not None for tensor in gaussian_params):
            kl_map = self._gaussian_kl_map(
                prior_mu=prior_mu,
                prior_logvar=prior_logvar,
                post_mu=post_mu,
                post_logvar=post_logvar,
            )
        else:
            kl_map = (z_post - z_prior) ** 2

        # surprise: [B, 1]; kl_map/safe_sigma: [B, 64]
        kl_map = torch.clamp(kl_map, min=0.0)
        surprise = kl_map.mean(dim=-1, keepdim=True)
        safe_sigma = self._safe_sigma(prior_logvar, kl_map)

        output = {
            "surprise": surprise,
            "kl_map": kl_map,
            "safe_sigma": safe_sigma,
        }
        self._validate_output(output)
        return output

    def _gaussian_kl_map(
        self,
        *,
        prior_mu: torch.Tensor,
        prior_logvar: torch.Tensor,
        post_mu: torch.Tensor,
        post_logvar: torch.Tensor,
    ) -> torch.Tensor:
        prior_logvar = prior_logvar.clamp(self.logvar_min, self.logvar_max)
        post_logvar = post_logvar.clamp(self.logvar_min, self.logvar_max)
        prior_var = torch.exp(prior_logvar)
        post_var = torch.exp(post_logvar)
        mean_delta_sq = (post_mu - prior_mu) ** 2
        return 0.5 * (
            prior_logvar
            - post_logvar
            + (post_var + mean_delta_sq) / prior_var
            - 1.0
        )

    def _safe_sigma(
        self,
        prior_logvar: torch.Tensor | None,
        kl_map: torch.Tensor,
    ) -> torch.Tensor:
        if prior_logvar is not None:
            logvar = prior_logvar.clamp(self.logvar_min, self.logvar_max)
            return torch.sqrt(torch.exp(logvar) + self.eps)

        return torch.sqrt(kl_map + self.eps)

    def _check_latent(self, tensor: torch.Tensor, name: str) -> None:
        check_shape(tensor, (None, C.RSSM_LATENT_DIM), name)
        if self.validate_finite:
            check_finite(tensor, name)

    def _validate_output(self, output: PredictionErrorOutput) -> None:
        expected_shapes = {
            "surprise": C.SURPRISE_SHAPE,
            "kl_map": C.KL_MAP_SHAPE,
            "safe_sigma": (None, C.RSSM_LATENT_DIM),
        }
        for name, shape in expected_shapes.items():
            check_shape(output[name], shape, name)
            if self.validate_finite:
                check_finite(output[name], name)


__all__ = ["PredictionErrorEvaluator", "PredictionErrorOutput"]

