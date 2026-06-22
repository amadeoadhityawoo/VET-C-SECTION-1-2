# RFC-1001 - Core Types

Status: MVP Implemented
Section: Implementation RFC
Version: v0.1

## Purpose

Defines core dataclasses for VET-C tensor contracts.

## Scope

Includes observation, encoder, fusion, latent, prediction, regime, terrain, memory, trajectory, action, training, loss, and runtime containers.

## Current MVP Status

Implemented in this repository.

## Key Tensor Contracts

- `ObservationTensor.ext_obs`: `[B, 6, 64, 64]`
- `ObservationTensor.int_obs`: `[B, 32, 11]`
- `EncoderOutput.z_ext`: `[B, 128, 16, 16]`
- `EncoderOutput.z_int`: `[B, 128]`
- `ActionCommand.action`: `[B, 4]`

## Implementation Notes

Implemented in `vetc/core/types.py`.

## Limitations

Dataclasses are lightweight containers and do not enforce shapes by themselves.

## Future Work

Add richer typed containers if training and deployment systems require them.

