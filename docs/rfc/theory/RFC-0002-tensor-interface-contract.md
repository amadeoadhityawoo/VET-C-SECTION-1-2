# RFC-0002 - Tensor Interface Contract

Status: Draft
Section: Theory RFC
Version: v0.1

## Purpose

Defines canonical tensor shapes and names used across VET-C.

## Scope

Includes observation tensors, encoder outputs, fusion tensors, RSSM states, prediction error tensors, regime tokens, terrain fields, and action commands.

## Current MVP Status

This theory contract is partially implemented in the Section 2 MVP through constants, dataclasses, and tests.

## Key Tensor Contracts

- `ext_obs`: `[B, 6, 64, 64]`
- `int_obs`: `[B, 32, 11]`
- `z_ext`: `[B, 128, 16, 16]`
- `z_int`: `[B, 128]`
- `fused`: `[B, 256]`
- `h`: `[B, 512]`
- `regime_token`: `[B, 96]`
- `psi_field`: `[B, 5]`
- `action`: `[B, 4]`

## Implementation Notes

Implemented primarily in `vetc/core/constants.py`, `vetc/core/types.py`, and `vetc/core/tensor_checks.py`.

## Limitations

The contract covers the runnable MVP path only.

## Future Work

Add contracts for training batches, datasets, memory, rollout, and deployment interfaces.

