import pytest
import torch

from vetc.core.errors import NonFiniteTensorError, TensorDTypeError, TensorShapeError
from vetc.core.types import ObservationTensor
from vetc.encoder.internal_encoder import InternalEncoder


def _observations(batch_size: int = 2) -> ObservationTensor:
    return ObservationTensor(
        ext_obs=torch.randn(batch_size, 6, 64, 64),
        int_obs=torch.randn(batch_size, 32, 11),
    )


def test_internal_encoder_outputs_expected_shape():
    encoder = InternalEncoder()

    z_int = encoder(_observations())

    assert z_int.shape == (2, 128)
    assert z_int.dtype == torch.float32


def test_internal_encoder_rejects_wrong_shape():
    encoder = InternalEncoder()
    observations = _observations()
    observations.int_obs = torch.zeros(2, 31, 11)

    with pytest.raises(TensorShapeError):
        encoder(observations)


def test_internal_encoder_rejects_wrong_dtype():
    encoder = InternalEncoder()
    observations = _observations()
    observations.int_obs = observations.int_obs.to(torch.float64)

    with pytest.raises(TensorDTypeError):
        encoder(observations)


def test_internal_encoder_rejects_non_finite_input():
    encoder = InternalEncoder()
    observations = _observations()
    observations.int_obs[:, 0, 0] = float("inf")

    with pytest.raises(NonFiniteTensorError):
        encoder(observations)

