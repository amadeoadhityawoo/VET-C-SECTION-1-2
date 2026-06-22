"""End-to-end VET-C v0.1 runtime system for RFC-1015."""

from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any

import torch
from torch import nn

from vetc.core import constants as C
from vetc.core.tensor_checks import check_batch_match, check_finite, check_shape
from vetc.core.types import ObservationPacket
from vetc.encoder.dual_perception_encoder import DualPerceptionEncoder
from vetc.fusion.regime_conditioned_fusion import RegimeConditionedFusion
from vetc.observation.observation_layer import ObservationLayer
from vetc.policy.fast_gradient_policy import FastGradientPolicy
from vetc.prediction.prediction_error_evaluator import PredictionErrorEvaluator
from vetc.regime.regime_engine_vqvae import RegimeEngineVQVAE
from vetc.rssm.unified_rssm import UnifiedRSSM
from vetc.terrain.vet_terrain_generator import VETTerrainGenerator


class VETCSystem(nn.Module):
    """Connect completed RFC-1004 through RFC-1014 modules end to end."""

    def __init__(self, *, validate_finite: bool = True) -> None:
        super().__init__()
        self.validate_finite = validate_finite
        self.observation_layer = ObservationLayer(validate_finite=validate_finite)
        self.encoder = DualPerceptionEncoder(validate_finite=validate_finite)
        self.fusion = RegimeConditionedFusion(validate_finite=validate_finite)
        self.rssm = UnifiedRSSM(validate_finite=validate_finite)
        self.prediction_error = PredictionErrorEvaluator(
            validate_finite=validate_finite,
        )
        self.regime_engine = RegimeEngineVQVAE(validate_finite=validate_finite)
        self.terrain_generator = VETTerrainGenerator(validate_finite=validate_finite)
        self.policy = FastGradientPolicy(validate_finite=validate_finite)

    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        imu: torch.Tensor,
        battery: torch.Tensor,
        motor_state: torch.Tensor,
        h_prev: torch.Tensor | None = None,
        z_prev: torch.Tensor | None = None,
        regime_token: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run one VET-C system step."""

        self._validate_sensor_inputs(rgb, depth, imu, battery, motor_state)
        packet = self._make_observation_packet(
            rgb=rgb,
            depth=depth,
            imu=imu,
            battery=battery,
            motor_state=motor_state,
        )

        observation_output = self.observation_layer(packet)
        ext_obs = _extract(observation_output, "ext_obs")
        int_obs = _extract(observation_output, "int_obs")

        encoder_output = self.encoder(ext_obs, int_obs)
        z_ext = _extract(encoder_output, "z_ext")
        z_int = _extract(encoder_output, "z_int")

        fused = self.fusion(z_ext, z_int, regime_token)
        rssm_output = self.rssm(fused, h_prev=h_prev, z_prev=z_prev)
        h = _extract(rssm_output, "h")
        z_prior = _extract(rssm_output, "z_prior")
        z_post = _extract(rssm_output, "z_post")

        prediction_output = self.prediction_error(
            z_post,
            z_prior,
            prior_mu=_optional_extract(rssm_output, "prior_mu"),
            prior_logvar=_optional_extract(rssm_output, "prior_logvar"),
            post_mu=_optional_extract(rssm_output, "post_mu"),
            post_logvar=_optional_extract(rssm_output, "post_logvar"),
        )
        surprise = _extract(prediction_output, "surprise")
        kl_map = _extract(prediction_output, "kl_map")
        safe_sigma = _extract(prediction_output, "safe_sigma")

        regime_output = self.regime_engine(h, surprise)
        new_regime_token = _extract(regime_output, "regime_token")
        commit_loss = _extract(regime_output, "commit_loss")

        terrain_output = self.terrain_generator(h, new_regime_token, surprise)
        psi_field = _extract(terrain_output, "psi_field")

        policy_output = self.policy(psi_field)
        action = _extract(policy_output, "action")

        output = {
            "action": action,
            "ext_obs": ext_obs,
            "int_obs": int_obs,
            "z_ext": z_ext,
            "z_int": z_int,
            "fused": fused,
            "h": h,
            "z_prior": z_prior,
            "z_post": z_post,
            "surprise": surprise,
            "kl_map": kl_map,
            "safe_sigma": safe_sigma,
            "regime_token": new_regime_token,
            "commit_loss": commit_loss,
            "psi_field": psi_field,
        }
        self._validate_outputs(output)
        return output

    def _make_observation_packet(
        self,
        *,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        imu: torch.Tensor,
        battery: torch.Tensor,
        motor_state: torch.Tensor,
    ) -> ObservationPacket:
        batch_size = rgb.shape[0]
        flow = rgb.new_zeros(
            batch_size,
            C.FLOW_CHANNELS,
            C.IMAGE_HEIGHT,
            C.IMAGE_WIDTH,
        )
        gps = rgb.new_zeros(batch_size, C.GPS_DIM)
        return ObservationPacket(
            rgb=rgb,
            depth=depth,
            flow=flow,
            gps=gps,
            imu=imu,
            battery=battery,
            motor_state=motor_state,
        )

    def _validate_sensor_inputs(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        imu: torch.Tensor,
        battery: torch.Tensor,
        motor_state: torch.Tensor,
    ) -> None:
        sensors = {
            "rgb": (rgb, C.RGB_SHAPE),
            "depth": (depth, C.DEPTH_SHAPE),
            "imu": (imu, C.IMU_SHAPE),
            "battery": (battery, C.BATTERY_SHAPE),
            "motor_state": (motor_state, C.MOTOR_STATE_SHAPE),
        }
        for name, (tensor, shape) in sensors.items():
            check_shape(tensor, shape, name)
            if self.validate_finite:
                check_finite(tensor, name)
        check_batch_match(*(tensor for tensor, _ in sensors.values()))

    def _validate_outputs(self, output: dict[str, torch.Tensor]) -> None:
        expected_shapes = {
            "action": C.ACTION_SHAPE,
            "ext_obs": C.EXT_OBS_SHAPE,
            "int_obs": C.INT_OBS_SHAPE,
            "z_ext": C.Z_EXT_SHAPE,
            "z_int": C.Z_INT_SHAPE,
            "fused": C.FUSED_SHAPE,
            "h": C.RSSM_H_SHAPE,
            "z_prior": C.Z_PRIOR_SHAPE,
            "z_post": C.Z_POST_SHAPE,
            "surprise": C.SURPRISE_SHAPE,
            "kl_map": C.KL_MAP_SHAPE,
            "safe_sigma": C.KL_MAP_SHAPE,
            "regime_token": C.REGIME_TOKEN_SHAPE,
            "commit_loss": (),
            "psi_field": C.PSI_FIELD_SHAPE,
        }
        for name, shape in expected_shapes.items():
            check_shape(output[name], shape, name)
            if self.validate_finite:
                check_finite(output[name], name)


def _extract(container: Any, field: str) -> torch.Tensor:
    value = _optional_extract(container, field)
    if value is None:
        raise KeyError(f"Missing field {field!r} in {type(container).__name__}.")
    return value


def _optional_extract(container: Any, field: str) -> torch.Tensor | None:
    if isinstance(container, dict):
        return container.get(field)
    if is_dataclass(container) and hasattr(container, field):
        return getattr(container, field)
    if hasattr(container, field):
        return getattr(container, field)
    return None


__all__ = ["VETCSystem"]

