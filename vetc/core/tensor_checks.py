"""Tensor validation helpers for VET-C modules."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from vetc.core.errors import NonFiniteTensorError, TensorDTypeError, TensorShapeError


def check_shape(
    tensor: torch.Tensor,
    expected_shape: Sequence[int | None],
    name: str,
) -> torch.Tensor:
    """Validate a tensor shape, allowing ``None`` as a wildcard dimension."""

    actual_shape = tuple(tensor.shape)
    expected = tuple(expected_shape)
    if len(actual_shape) != len(expected):
        raise TensorShapeError(
            f"{name} expected rank {len(expected)} with shape {expected}, "
            f"got rank {len(actual_shape)} with shape {actual_shape}."
        )

    mismatches = [
        (index, actual, expected_dim)
        for index, (actual, expected_dim) in enumerate(zip(actual_shape, expected))
        if expected_dim is not None and actual != expected_dim
    ]
    if mismatches:
        details = ", ".join(
            f"dim {index}: expected {expected_dim}, got {actual}"
            for index, actual, expected_dim in mismatches
        )
        raise TensorShapeError(
            f"{name} shape mismatch: expected {expected}, got {actual_shape} ({details})."
        )

    return tensor


def check_dtype(
    tensor: torch.Tensor,
    expected_dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    """Validate that a tensor has the expected dtype."""

    if tensor.dtype != expected_dtype:
        raise TensorDTypeError(
            f"{name} expected dtype {expected_dtype}, got {tensor.dtype}."
        )
    return tensor


def check_batch_match(*tensors: torch.Tensor) -> int | None:
    """Validate that all provided tensors share the same leading batch size."""

    if not tensors:
        return None

    batch_sizes = []
    for index, tensor in enumerate(tensors):
        if tensor.ndim == 0:
            raise TensorShapeError(
                f"tensor {index} must have a batch dimension; got scalar shape ()."
            )
        batch_sizes.append(int(tensor.shape[0]))

    expected = batch_sizes[0]
    for index, batch_size in enumerate(batch_sizes[1:], start=1):
        if batch_size != expected:
            raise TensorShapeError(
                f"batch size mismatch: tensor 0 has batch {expected}, "
                f"tensor {index} has batch {batch_size}."
            )

    return expected


def check_finite(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Validate that a tensor contains only finite values."""

    if not torch.isfinite(tensor).all().item():
        raise NonFiniteTensorError(f"{name} contains NaN or infinite values.")
    return tensor


__all__ = [
    "check_batch_match",
    "check_dtype",
    "check_finite",
    "check_shape",
]

