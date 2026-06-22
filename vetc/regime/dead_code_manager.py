"""Simple codebook usage tracker for VET-C RFC-1012."""

from __future__ import annotations

import torch
from torch import nn


class DeadCodeManager(nn.Module):
    """Track vector-quantizer code usage counts.

    TODO: Add EMA dead-code replacement when the codebook training loop exists.
    """

    def __init__(self, *, num_codes: int = 32) -> None:
        super().__init__()
        self.num_codes = num_codes
        self.register_buffer("usage_counts", torch.zeros(num_codes, dtype=torch.long))

    @torch.no_grad()
    def update_usage(self, code_indices: torch.Tensor) -> None:
        """Accumulate usage counts for selected code indices."""

        indices = code_indices.detach().to(
            device=self.usage_counts.device,
            dtype=torch.long,
        )
        counts = torch.bincount(indices.reshape(-1), minlength=self.num_codes)
        self.usage_counts.add_(counts[: self.num_codes])

    def get_usage(self) -> torch.Tensor:
        """Return a copy of current code usage counts."""

        return self.usage_counts.clone()


__all__ = ["DeadCodeManager"]

