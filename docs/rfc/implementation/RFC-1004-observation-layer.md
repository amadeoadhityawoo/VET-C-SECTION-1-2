# RFC-1004 - Observation Layer

Status: MVP Implemented
Section: Implementation RFC
Version: v0.1

## Purpose

Converts raw sensor packets into canonical observation tensors.

## Scope

Includes RGB, depth, flow, GPS validation, and internal sensor-history concatenation.

## Current MVP Status

Implemented in this repository.

## Key Tensor Contracts

- `rgb`: `[B, 3, 64, 64]`
- `depth`: `[B, 1, 64, 64]`
- `flow`: `[B, 2, 64, 64]`
- `ext_obs`: `[B, 6, 64, 64]`
- `int_obs`: `[B, 32, 11]`

## Implementation Notes

Implemented in `vetc/observation/observation_layer.py`.

## Limitations

RFC-1015 runtime uses zero placeholders for flow and GPS in dummy input mode.

## Future Work

Add real sensor adapters and normalization policy.

