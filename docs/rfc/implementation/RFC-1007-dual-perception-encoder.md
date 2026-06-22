# RFC-1007 - Dual Perception Encoder

Status: MVP Implemented
Section: Implementation RFC
Version: v0.1

## Purpose

Runs external and internal encoders together.

## Scope

Includes a wrapper that returns both `z_ext` and `z_int` without fusing them.

## Current MVP Status

Implemented in this repository.

## Key Tensor Contracts

- `ext_obs`: `[B, 6, 64, 64]`
- `int_obs`: `[B, 32, 11]`
- `z_ext`: `[B, 128, 16, 16]`
- `z_int`: `[B, 128]`

## Implementation Notes

Implemented in `vetc/encoder/dual_perception_encoder.py`.

## Limitations

No cross-modal attention or fusion occurs in this module.

## Future Work

Keep fusion concerns in dedicated fusion modules.

