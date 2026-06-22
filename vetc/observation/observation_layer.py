"""Observation preprocessing for VET-C RFC-1004."""

from __future__ import annotations

import torch

from vetc.core import constants as C
from vetc.core.interfaces import BaseVETCModule
from vetc.core.tensor_checks import (
    check_batch_match,
    check_dtype,
    check_finite,
    check_shape,
)
from vetc.core.types import ObservationPacket, ObservationTensor


class ObservationLayer(BaseVETCModule):
    """Convert raw sensor packets into canonical observation tensors.

    Input shapes:
        packet.rgb: [B, 3, 64, 64]
        packet.depth: [B, 1, 64, 64]
        packet.flow: [B, 2, 64, 64]
        packet.gps: [B, 3]
        packet.imu: [B, 32, 6]
        packet.battery: [B, 32, 1]
        packet.motor_state: [B, 32, 4]

    Output shapes:
        ext_obs: [B, 6, 64, 64], concatenating RGB, depth, and flow channels
        int_obs: [B, 32, 11], concatenating IMU, battery, and motor-state features

    GPS is validated as part of the raw packet contract, but it is not folded
    into RFC-1004's canonical observation tensors.
    """

    def __init__(
        self,
        *,
        expected_dtype: torch.dtype = torch.float32,
        validate_finite: bool = True,
    ) -> None:
        super().__init__()
        self.expected_dtype = expected_dtype
        self.validate_finite = validate_finite

    def forward(self, packet: ObservationPacket) -> ObservationTensor:
        """Build canonical external and internal observation tensors."""

        self._validate_packet(packet)

        ext_obs = torch.cat((packet.rgb, packet.depth, packet.flow), dim=1)
        int_obs = torch.cat((packet.imu, packet.battery, packet.motor_state), dim=-1)

        check_shape(ext_obs, C.EXT_OBS_SHAPE, "ext_obs")
        check_shape(int_obs, C.INT_OBS_SHAPE, "int_obs")
        check_batch_match(ext_obs, int_obs)

        if self.validate_finite:
            check_finite(ext_obs, "ext_obs")
            check_finite(int_obs, "int_obs")

        return ObservationTensor(ext_obs=ext_obs, int_obs=int_obs)

    def _validate_packet(self, packet: ObservationPacket) -> None:
        fields = (
            ("rgb", packet.rgb, C.RGB_SHAPE),
            ("depth", packet.depth, C.DEPTH_SHAPE),
            ("flow", packet.flow, C.FLOW_SHAPE),
            ("gps", packet.gps, C.GPS_SHAPE),
            ("imu", packet.imu, C.IMU_SHAPE),
            ("battery", packet.battery, C.BATTERY_SHAPE),
            ("motor_state", packet.motor_state, C.MOTOR_STATE_SHAPE),
        )

        for name, tensor, shape in fields:
            check_shape(tensor, shape, name)
            check_dtype(tensor, self.expected_dtype, name)
            if self.validate_finite:
                check_finite(tensor, name)

        check_batch_match(*(tensor for _, tensor, _ in fields))


__all__ = ["ObservationLayer"]

