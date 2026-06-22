"""Unified RSSM wrapper for VET-C RFC-1010."""

from __future__ import annotations

import torch

from vetc.core import constants as C
from vetc.core.interfaces import BaseVETCModule
from vetc.core.tensor_checks import check_batch_match, check_finite, check_shape
from vetc.rssm.rssm_cell import RSSMCell, RSSMCellOutput


class UnifiedRSSM(BaseVETCModule):
    """Stateful-friendly wrapper around ``RSSMCell``.

    Inputs:
        fused: [B, 256]
        h_prev: [B, 512] or None
        z_prev: [B, 64] or None

    Outputs preserve ``RSSMCell``'s dictionary return format.
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
        self.cell = RSSMCell(
            fused_dim=fused_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            validate_finite=validate_finite,
        )

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero ``h_prev`` and ``z_prev`` tensors."""

        dtype = dtype or torch.float32
        h_prev = torch.zeros(
            batch_size,
            self.hidden_dim,
            device=device,
            dtype=dtype,
        )
        z_prev = torch.zeros(
            batch_size,
            self.latent_dim,
            device=device,
            dtype=dtype,
        )
        return h_prev, z_prev

    def forward(
        self,
        fused: torch.Tensor,
        h_prev: torch.Tensor | None = None,
        z_prev: torch.Tensor | None = None,
    ) -> RSSMCellOutput:
        """Run one RSSM step, creating zero previous state when omitted."""

        # fused: [B, 256]
        check_shape(fused, (None, self.fused_dim), "fused")
        if self.validate_finite:
            check_finite(fused, "fused")

        batch_size = int(fused.shape[0])
        initial_h, initial_z = self.initial_state(
            batch_size,
            device=fused.device,
            dtype=fused.dtype,
        )
        h_prev = initial_h if h_prev is None else h_prev
        z_prev = initial_z if z_prev is None else z_prev

        # h_prev: [B, 512]; z_prev: [B, 64]
        check_shape(h_prev, (None, self.hidden_dim), "h_prev")
        check_shape(z_prev, (None, self.latent_dim), "z_prev")
        check_batch_match(fused, h_prev, z_prev)
        if self.validate_finite:
            check_finite(h_prev, "h_prev")
            check_finite(z_prev, "z_prev")

        output = self.cell(fused, h_prev, z_prev)
        self._validate_output(output)
        return output

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


__all__ = ["UnifiedRSSM"]

