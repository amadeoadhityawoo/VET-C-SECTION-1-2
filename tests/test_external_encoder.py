import pytest
import torch

from vetc.core.errors import NonFiniteTensorError, TensorDTypeError, TensorShapeError
from vetc.core.types import ObservationTensor
from vetc.encoder.external_encoder import ExternalEncoder


def _observations(batch_size: int = 2) -> ObservationTensor:
    return ObservationTensor(
        ext_obs=torch.randn(batch_size, 6, 64, 64),
        int_obs=torch.randn(batch_size, 32, 11),
    )


def test_external_encoder_outputs_expected_shape():
    encoder = ExternalEncoder()

    z_ext = encoder(_observations())

    assert z_ext.shape == (2, 128, 16, 16)
    assert z_ext.dtype == torch.float32


def test_external_encoder_rejects_wrong_shape():
    encoder = ExternalEncoder()
    observations = _observations()
    observations.ext_obs = torch.zeros(2, 5, 64, 64)

    with pytest.raises(TensorShapeError):
        encoder(observations)


def test_external_encoder_rejects_wrong_dtype():
    encoder = ExternalEncoder()
    observations = _observations()
    observations.ext_obs = observations.ext_obs.to(torch.float64)

    with pytest.raises(TensorDTypeError):
        encoder(observations)


def test_external_encoder_rejects_non_finite_input():
    encoder = ExternalEncoder()
    observations = _observations()
    observations.ext_obs[:, :, 0, 0] = float("nan")

    with pytest.raises(NonFiniteTensorError):
        encoder(observations)

