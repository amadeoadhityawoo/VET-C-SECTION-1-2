# RFC-1006 - Internal Encoder

Status: MVP Implemented
Section: Implementation RFC
Version: v0.1

## Purpose

Encodes internal sensor history into a compact latent vector.

## Scope

Includes a lightweight MLP over IMU, battery, and motor-state history.

## Current MVP Status

Implemented in this repository.

## Key Tensor Contracts

- `int_obs`: `[B, 32, 11]`
- `z_int`: `[B, 128]`

## Implementation Notes

Implemented in `vetc/encoder/internal_encoder.py`.

## Limitations

The encoder is untrained and uses a simple flattened history representation.

## Future Work

Evaluate recurrent or attention-based internal encoders.

