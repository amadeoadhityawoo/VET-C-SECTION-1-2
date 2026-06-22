# RFC-0005 - Runtime Inference

Status: Draft
Section: Theory RFC
Version: v0.1

## Purpose

Defines the intended runtime inference behavior for VET-C.

## Scope

Includes sensor input handling, recurrent state updates, regime token carryover, action output, and runtime health checks.

## Current MVP Status

This is partially reflected in the Section 2 MVP through a dummy runtime loop using synthetic inputs.

## Key Tensor Contracts

- `h_prev`: `[B, 512]`
- `z_prev`: `[B, 64]`
- `regime_token`: `[B, 96]`
- `action`: `[B, 4]`

## Implementation Notes

Current MVP files are `vetc/runtime/vetc_system.py`, `vetc/runtime/runtime_loop.py`, `scripts/run_dummy_infer.py`, and `scripts/run_runtime_loop.py`.

## Limitations

The runtime loop uses dummy tensors, not real sensors or flight-controller I/O.

## Future Work

Add hardware adapters, safety checks, scheduling, logging, deployment profiling, and device-specific runtime integration.

