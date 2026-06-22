import torch

from vetc.core.types import (
    ActionCommand,
    EncoderOutput,
    FusionOutput,
    FutureTrajectory,
    LatentState,
    LossOutput,
    MemoryOutput,
    ObservationPacket,
    ObservationTensor,
    PredictionOutput,
    RegimeTensor,
    RuntimeState,
    TerrainOutput,
    TrainingBatch,
    VETCStepOutput,
)


def test_core_dataclasses_can_be_instantiated_with_dummy_tensors():
    batch_size = 2
    horizon = 3

    packet = ObservationPacket(
        rgb=torch.zeros(batch_size, 3, 64, 64),
        depth=torch.zeros(batch_size, 1, 64, 64),
        flow=torch.zeros(batch_size, 2, 64, 64),
        gps=torch.zeros(batch_size, 3),
        imu=torch.zeros(batch_size, 32, 6),
        battery=torch.zeros(batch_size, 32, 1),
        motor_state=torch.zeros(batch_size, 32, 4),
    )
    observations = ObservationTensor(
        ext_obs=torch.zeros(batch_size, 6, 64, 64),
        int_obs=torch.zeros(batch_size, 32, 11),
    )
    encoder_output = EncoderOutput(
        z_ext=torch.zeros(batch_size, 128, 16, 16),
        z_int=torch.zeros(batch_size, 128),
    )
    fusion_output = FusionOutput(
        fused=torch.zeros(batch_size, 256),
        fusion_error=torch.zeros(batch_size, 1),
    )
    latent_state = LatentState(
        h=torch.zeros(batch_size, 512),
        z_prior=torch.zeros(batch_size, 64),
        z_post=torch.zeros(batch_size, 64),
    )
    prediction = PredictionOutput(
        surprise=torch.zeros(batch_size, 1),
        kl_map=torch.zeros(batch_size, 64),
        fusion_error=torch.zeros(batch_size, 1),
    )
    regime = RegimeTensor(regime_token=torch.zeros(batch_size, 96))
    terrain = TerrainOutput(psi_field=torch.zeros(batch_size, 5))
    memory = MemoryOutput(memory_tokens=torch.zeros(batch_size, 128))
    future = FutureTrajectory(
        trajectory=torch.zeros(batch_size, horizon, 64),
        density=torch.zeros(batch_size, horizon),
    )
    action = ActionCommand(action=torch.zeros(batch_size, 4))
    step_output = VETCStepOutput(
        action=action,
        latent_state=latent_state,
        prediction=prediction,
        regime=regime,
        terrain=terrain,
        memory=memory,
        future=future,
    )
    training_batch = TrainingBatch(
        observations=observations,
        actions=torch.zeros(batch_size, 4),
        target_trajectory=torch.zeros(batch_size, horizon, 64),
        target_density=torch.zeros(batch_size, horizon),
    )
    loss_output = LossOutput(
        total=torch.tensor(0.0),
        reconstruction=torch.tensor(0.0),
        kl=torch.tensor(0.0),
        policy=torch.tensor(0.0),
        terrain=torch.tensor(0.0),
    )
    runtime_state = RuntimeState(
        step_index=torch.tensor(0),
        latent_state=latent_state,
        memory=memory,
    )

    assert packet.rgb.shape == (batch_size, 3, 64, 64)
    assert observations.ext_obs.shape == (batch_size, 6, 64, 64)
    assert encoder_output.z_ext.shape == (batch_size, 128, 16, 16)
    assert fusion_output.fused.shape == (batch_size, 256)
    assert step_output.action is action
    assert training_batch.observations is observations
    assert loss_output.total.ndim == 0
    assert runtime_state.latent_state is latent_state

