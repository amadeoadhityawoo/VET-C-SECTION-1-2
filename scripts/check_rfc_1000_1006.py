"""Verification smoke test for VET-C RFC-1000 through RFC-1006."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vetc.core.types import ObservationPacket  # noqa: E402
from vetc.encoder.external_encoder import ExternalEncoder  # noqa: E402
from vetc.encoder.internal_encoder import InternalEncoder  # noqa: E402
from vetc.observation.observation_layer import ObservationLayer  # noqa: E402


def _assert_shape(tensor: torch.Tensor, expected: tuple[int, ...], name: str) -> None:
    actual = tuple(tensor.shape)
    if actual != expected:
        raise AssertionError(f"{name} expected shape {expected}, got {actual}.")


def main() -> None:
    batch_size = 2

    packet = ObservationPacket(
        rgb=torch.randn(batch_size, 3, 64, 64),
        depth=torch.randn(batch_size, 1, 64, 64),
        flow=torch.randn(batch_size, 2, 64, 64),
        gps=torch.randn(batch_size, 3),
        imu=torch.randn(batch_size, 32, 6),
        battery=torch.randn(batch_size, 32, 1),
        motor_state=torch.randn(batch_size, 32, 4),
    )

    observation_layer = ObservationLayer()
    external_encoder = ExternalEncoder().eval()
    internal_encoder = InternalEncoder().eval()

    observations = observation_layer(packet)
    _assert_shape(observations.ext_obs, (batch_size, 6, 64, 64), "ext_obs")
    _assert_shape(observations.int_obs, (batch_size, 32, 11), "int_obs")

    with torch.no_grad():
        z_ext = external_encoder(observations)
        z_int = internal_encoder(observations)

    _assert_shape(z_ext, (batch_size, 128, 16, 16), "z_ext")
    _assert_shape(z_int, (batch_size, 128), "z_int")

    print("rfc 1000-1006 ok")


if __name__ == "__main__":
    main()

