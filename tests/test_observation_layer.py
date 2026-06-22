import pytest
import torch

from vetc.core.errors import NonFiniteTensorError, TensorDTypeError, TensorShapeError
from vetc.core.types import ObservationPacket, ObservationTensor
from vetc.observation.observation_layer import ObservationLayer


def _packet(batch_size: int = 2) -> ObservationPacket:
    return ObservationPacket(
        rgb=torch.full((batch_size, 3, 64, 64), 1.0),
        depth=torch.full((batch_size, 1, 64, 64), 2.0),
        flow=torch.full((batch_size, 2, 64, 64), 3.0),
        gps=torch.zeros(batch_size, 3),
        imu=torch.full((batch_size, 32, 6), 4.0),
        battery=torch.full((batch_size, 32, 1), 5.0),
        motor_state=torch.full((batch_size, 32, 4), 6.0),
    )


def test_observation_layer_builds_canonical_observation_tensor():
    layer = ObservationLayer()

    output = layer(_packet())

    assert isinstance(output, ObservationTensor)
    assert output.ext_obs.shape == (2, 6, 64, 64)
    assert output.int_obs.shape == (2, 32, 11)
    assert torch.equal(output.ext_obs[:, :3], torch.full((2, 3, 64, 64), 1.0))
    assert torch.equal(output.ext_obs[:, 3:4], torch.full((2, 1, 64, 64), 2.0))
    assert torch.equal(output.ext_obs[:, 4:6], torch.full((2, 2, 64, 64), 3.0))
    assert torch.equal(output.int_obs[:, :, :6], torch.full((2, 32, 6), 4.0))
    assert torch.equal(output.int_obs[:, :, 6:7], torch.full((2, 32, 1), 5.0))
    assert torch.equal(output.int_obs[:, :, 7:11], torch.full((2, 32, 4), 6.0))


def test_observation_layer_rejects_wrong_input_shape():
    layer = ObservationLayer()
    packet = _packet()
    packet.flow = torch.zeros(2, 3, 64, 64)

    with pytest.raises(TensorShapeError):
        layer(packet)


def test_observation_layer_rejects_batch_mismatch():
    layer = ObservationLayer()
    packet = _packet()
    packet.gps = torch.zeros(3, 3)

    with pytest.raises(TensorShapeError):
        layer(packet)


def test_observation_layer_rejects_wrong_dtype():
    layer = ObservationLayer()
    packet = _packet()
    packet.rgb = packet.rgb.to(torch.float64)

    with pytest.raises(TensorDTypeError):
        layer(packet)


def test_observation_layer_rejects_non_finite_inputs():
    layer = ObservationLayer()
    packet = _packet()
    packet.depth[:, :, 0, 0] = float("nan")

    with pytest.raises(NonFiniteTensorError):
        layer(packet)


def test_observation_layer_can_skip_finite_validation():
    layer = ObservationLayer(validate_finite=False)
    packet = _packet()
    packet.depth[:, :, 0, 0] = float("nan")

    output = layer(packet)

    assert torch.isnan(output.ext_obs[:, 3:4, 0, 0]).all()
