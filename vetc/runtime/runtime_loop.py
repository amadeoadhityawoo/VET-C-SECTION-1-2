"""Simple dummy runtime loop for VET-C RFC-1015."""

from __future__ import annotations

import time

import torch

from vetc.core import constants as C
from vetc.runtime.vetc_system import VETCSystem


class RuntimeLoop:
    """Run repeated dummy inference cycles through ``VETCSystem``."""

    def __init__(
        self,
        system: VETCSystem | None = None,
        *,
        batch_size: int = 2,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.system = system if system is not None else VETCSystem()
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.system.to(device=device, dtype=dtype)
        self.system.eval()

    def run(self, cycles: int = 100) -> dict[str, object]:
        """Run dummy sensor inputs through the model repeatedly."""

        h_prev: torch.Tensor | None = None
        z_prev: torch.Tensor | None = None
        regime_token: torch.Tensor | None = None
        final_action: torch.Tensor | None = None
        nan_detected = False
        total_step_time = 0.0

        with torch.no_grad():
            for _ in range(cycles):
                inputs = self._dummy_inputs()
                start = time.perf_counter()
                output = self.system(
                    **inputs,
                    h_prev=h_prev,
                    z_prev=z_prev,
                    regime_token=regime_token,
                )
                total_step_time += time.perf_counter() - start

                final_action = output["action"]
                h_prev = output["h"]
                z_prev = output["z_post"]
                regime_token = output["regime_token"]
                if not self._all_finite(output):
                    nan_detected = True

        avg_step_time_ms = (total_step_time / max(cycles, 1)) * 1000.0
        final_shape = tuple(final_action.shape) if final_action is not None else ()
        return {
            "cycles": cycles,
            "final_action_shape": final_shape,
            "nan_detected": nan_detected,
            "avg_step_time_ms": avg_step_time_ms,
        }

    def _dummy_inputs(self) -> dict[str, torch.Tensor]:
        return {
            "rgb": torch.randn(
                self.batch_size,
                C.RGB_CHANNELS,
                C.IMAGE_HEIGHT,
                C.IMAGE_WIDTH,
                device=self.device,
                dtype=self.dtype,
            ),
            "depth": torch.randn(
                self.batch_size,
                C.DEPTH_CHANNELS,
                C.IMAGE_HEIGHT,
                C.IMAGE_WIDTH,
                device=self.device,
                dtype=self.dtype,
            ),
            "imu": torch.randn(
                self.batch_size,
                C.SENSOR_HISTORY_STEPS,
                C.IMU_DIM,
                device=self.device,
                dtype=self.dtype,
            ),
            "battery": torch.randn(
                self.batch_size,
                C.SENSOR_HISTORY_STEPS,
                C.BATTERY_DIM,
                device=self.device,
                dtype=self.dtype,
            ),
            "motor_state": torch.randn(
                self.batch_size,
                C.SENSOR_HISTORY_STEPS,
                C.MOTOR_STATE_DIM,
                device=self.device,
                dtype=self.dtype,
            ),
        }

    @staticmethod
    def _all_finite(output: dict[str, torch.Tensor]) -> bool:
        return all(torch.isfinite(tensor).all().item() for tensor in output.values())


__all__ = ["RuntimeLoop"]

