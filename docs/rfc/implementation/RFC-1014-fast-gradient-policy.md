# RFC-1014 - Fast Gradient Policy

Status: MVP Implemented
Section: Implementation RFC
Version: v0.1

## Purpose

Maps terrain psi fields to normalized action commands.

## Scope

Includes a lightweight MLP and tanh-normalized action output.

## Current MVP Status

Implemented in this repository.

## Key Tensor Contracts

- `psi_field`: `[B, 5]`
- `action`: `[B, 4]`

## Implementation Notes

Implemented in `vetc/policy/fast_gradient_policy.py`.

## Limitations

The policy is untrained and not safe for real drone control.

## Future Work

Train policy behavior, add safety constraints, and integrate with actuator interfaces.

