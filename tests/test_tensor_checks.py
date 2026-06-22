import pytest
import torch

from vetc.core.errors import NonFiniteTensorError, TensorDTypeError, TensorShapeError
from vetc.core.tensor_checks import (
    check_batch_match,
    check_dtype,
    check_finite,
    check_shape,
)


def test_check_shape_accepts_matching_shape_with_wildcard_batch():
    tensor = torch.zeros(2, 3, 64, 64)

    assert check_shape(tensor, (None, 3, 64, 64), "rgb") is tensor


def test_check_shape_rejects_wrong_shape():
    tensor = torch.zeros(2, 1, 64, 64)

    with pytest.raises(TensorShapeError):
        check_shape(tensor, (None, 3, 64, 64), "rgb")


def test_check_dtype_accepts_matching_dtype():
    tensor = torch.zeros(2, 3, dtype=torch.float32)

    assert check_dtype(tensor, torch.float32, "features") is tensor


def test_check_dtype_rejects_wrong_dtype():
    tensor = torch.zeros(2, 3, dtype=torch.float64)

    with pytest.raises(TensorDTypeError):
        check_dtype(tensor, torch.float32, "features")


def test_check_batch_match_accepts_equal_batches():
    first = torch.zeros(2, 3)
    second = torch.zeros(2, 4, 5)

    assert check_batch_match(first, second) == 2


def test_check_batch_match_rejects_mismatched_batches():
    first = torch.zeros(2, 3)
    second = torch.zeros(3, 4)

    with pytest.raises(TensorShapeError):
        check_batch_match(first, second)


def test_check_finite_rejects_nan_tensor():
    tensor = torch.tensor([0.0, float("nan")])

    with pytest.raises(NonFiniteTensorError):
        check_finite(tensor, "diagnostics")


def test_check_finite_accepts_finite_tensor():
    tensor = torch.tensor([0.0, 1.0])

    assert check_finite(tensor, "diagnostics") is tensor

