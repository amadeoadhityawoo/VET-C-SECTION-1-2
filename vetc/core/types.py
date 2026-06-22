"""Dataclasses that define the VET-C core tensor contracts.

The default spatial resolution is H=64 and W=64. Batch size B and trajectory
horizon T are intentionally dynamic unless a specific module chooses otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class ObservationPacket:
    """Raw sensor packet.

    Shapes:
        rgb: [B, 3, 64, 64]
        depth: [B, 1, 64, 64]
        flow: [B, 2, 64, 64]
        gps: [B, 3]
        imu: [B, 32, 6]
        battery: [B, 32, 1]
        motor_state: [B, 32, 4]
    """

    rgb: torch.Tensor
    depth: torch.Tensor
    flow: torch.Tensor
    gps: torch.Tensor
    imu: torch.Tensor
    battery: torch.Tensor
    motor_state: torch.Tensor


@dataclass(slots=True)
class ObservationTensor:
    """Canonical observation tensors for downstream modules.

    Shapes:
        ext_obs: [B, 6, 64, 64]
        int_obs: [B, 32, 11]
    """

    ext_obs: torch.Tensor
    int_obs: torch.Tensor


@dataclass(slots=True)
class EncoderOutput:
    """Outputs from future external/internal encoders.

    Shapes:
        z_ext: [B, 128, 16, 16]
        z_int: [B, 128]
    """

    z_ext: torch.Tensor
    z_int: torch.Tensor


@dataclass(slots=True)
class FusionOutput:
    """Fused representation passed into dynamics and control modules.

    Shapes:
        fused: [B, 256]
        fusion_error: [B, 1]
    """

    fused: torch.Tensor
    fusion_error: torch.Tensor


@dataclass(slots=True)
class LatentState:
    """RSSM-style latent state placeholder.

    Shapes:
        h: [B, 512]
        z_prior: [B, 64]
        z_post: [B, 64]
    """

    h: torch.Tensor
    z_prior: torch.Tensor
    z_post: torch.Tensor


@dataclass(slots=True)
class PredictionOutput:
    """Prediction diagnostics emitted by future dynamics modules.

    Shapes:
        surprise: [B, 1]
        kl_map: [B, 64]
        fusion_error: [B, 1]
    """

    surprise: torch.Tensor
    kl_map: torch.Tensor
    fusion_error: torch.Tensor


@dataclass(slots=True)
class RegimeTensor:
    """Compact regime representation.

    Shapes:
        regime_token: [B, 96]
    """

    regime_token: torch.Tensor


@dataclass(slots=True)
class TerrainOutput:
    """Terrain-field representation.

    Shapes:
        psi_field: [B, 5]
    """

    psi_field: torch.Tensor


@dataclass(slots=True)
class MemoryOutput:
    """Memory tokens for future retrieval or attention mechanisms.

    Shapes:
        memory_tokens: [B, 128]
    """

    memory_tokens: torch.Tensor


@dataclass(slots=True)
class FutureTrajectory:
    """Future trajectory distribution placeholder.

    Shapes:
        trajectory: [B, T, 64]
        density: [B, T]
    """

    trajectory: torch.Tensor
    density: torch.Tensor


@dataclass(slots=True)
class ActionCommand:
    """Low-level action command for the drone controller.

    Shapes:
        action: [B, 4]
    """

    action: torch.Tensor


@dataclass(slots=True)
class VETCStepOutput:
    """Aggregated output of one future VET-C runtime/model step.

    Fields:
        action: ActionCommand with action [B, 4]
        latent_state: LatentState with h [B, 512], z_prior/z_post [B, 64]
        prediction: PredictionOutput with surprise [B, 1], kl_map [B, 64]
        regime: RegimeTensor with regime_token [B, 96]
        terrain: TerrainOutput with psi_field [B, 5]
        memory: MemoryOutput with memory_tokens [B, 128]
        future: FutureTrajectory with trajectory [B, T, 64], density [B, T]
    """

    action: ActionCommand
    latent_state: LatentState
    prediction: PredictionOutput
    regime: RegimeTensor
    terrain: TerrainOutput
    memory: MemoryOutput
    future: FutureTrajectory


@dataclass(slots=True)
class TrainingBatch:
    """Training batch container for future learning loops.

    Shapes:
        observations.ext_obs: [B, 6, 64, 64]
        observations.int_obs: [B, 32, 11]
        actions: [B, 4]
        target_trajectory: [B, T, 64]
        target_density: [B, T]
    """

    observations: ObservationTensor
    actions: torch.Tensor
    target_trajectory: torch.Tensor
    target_density: torch.Tensor


@dataclass(slots=True)
class LossOutput:
    """Loss values emitted by future training modules.

    Shapes:
        total: scalar tensor []
        reconstruction: scalar tensor []
        kl: scalar tensor []
        policy: scalar tensor []
        terrain: scalar tensor []
    """

    total: torch.Tensor
    reconstruction: torch.Tensor
    kl: torch.Tensor
    policy: torch.Tensor
    terrain: torch.Tensor


@dataclass(slots=True)
class RuntimeState:
    """Mutable runtime state snapshot for edge deployment.

    Shapes:
        step_index: scalar tensor []
        latent_state.h: [B, 512]
        latent_state.z_prior: [B, 64]
        latent_state.z_post: [B, 64]
        memory.memory_tokens: [B, 128]
    """

    step_index: torch.Tensor
    latent_state: LatentState
    memory: MemoryOutput


__all__ = [
    "ActionCommand",
    "EncoderOutput",
    "FusionOutput",
    "FutureTrajectory",
    "LatentState",
    "LossOutput",
    "MemoryOutput",
    "ObservationPacket",
    "ObservationTensor",
    "PredictionOutput",
    "RegimeTensor",
    "RuntimeState",
    "TerrainOutput",
    "TrainingBatch",
    "VETCStepOutput",
]

