# RFC-1011 - Prediction Error

Status: MVP Implemented
Section: Implementation RFC
Version: v0.1

## Purpose

Computes prediction error and surprise from posterior and prior latent states.

## Scope

Includes squared-error fallback, optional diagonal Gaussian KL map, surprise, and safe sigma.

## Current MVP Status

Implemented in this repository.

## Key Tensor Contracts

- `z_post`: `[B, 64]`
- `z_prior`: `[B, 64]`
- `surprise`: `[B, 1]`
- `kl_map`: `[B, 64]`
- `safe_sigma`: `[B, 64]`

## Implementation Notes

Implemented in `vetc/prediction/prediction_error_evaluator.py`.

## Limitations

The metric is used for MVP connectivity, not calibrated uncertainty.

## Future Work

Calibrate surprise and evaluate it against real prediction errors.

