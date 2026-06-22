"""Run one dummy VET-C inference pass."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vetc.core import constants as C  # noqa: E402
from vetc.runtime.vetc_system import VETCSystem  # noqa: E402


def main() -> None:
    system = VETCSystem().eval()
    batch_size = 2
    inputs = {
        "rgb": torch.randn(batch_size, C.RGB_CHANNELS, C.IMAGE_HEIGHT, C.IMAGE_WIDTH),
        "depth": torch.randn(
            batch_size,
            C.DEPTH_CHANNELS,
            C.IMAGE_HEIGHT,
            C.IMAGE_WIDTH,
        ),
        "imu": torch.randn(batch_size, C.SENSOR_HISTORY_STEPS, C.IMU_DIM),
        "battery": torch.randn(batch_size, C.SENSOR_HISTORY_STEPS, C.BATTERY_DIM),
        "motor_state": torch.randn(
            batch_size,
            C.SENSOR_HISTORY_STEPS,
            C.MOTOR_STATE_DIM,
        ),
    }
    with torch.no_grad():
        output = system(**inputs)

    print("action:")
    print(output["action"])
    print("output shapes:")
    for key in (
        "ext_obs",
        "int_obs",
        "z_ext",
        "z_int",
        "fused",
        "h",
        "z_prior",
        "z_post",
        "surprise",
        "kl_map",
        "safe_sigma",
        "regime_token",
        "psi_field",
        "action",
    ):
        print(f"{key}: {tuple(output[key].shape)}")


if __name__ == "__main__":
    main()

