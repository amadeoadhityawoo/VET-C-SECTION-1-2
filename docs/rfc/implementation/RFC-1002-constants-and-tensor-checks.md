# RFC-1002 - Constants and Tensor Checks

Status: MVP Implemented
Section: Implementation RFC
Version: v0.1

## Purpose

Defines shared tensor dimensions, validation helpers, and custom errors.

## Scope

Includes tensor-shape templates, dtype checks, batch matching, finite checks, and error classes.

## Current MVP Status

Implemented in this repository and used throughout the MVP modules.

## Key Tensor Contracts

Core constants define shapes for observations, latents, fused state, RSSM state, regime token, psi field, and action.

## Implementation Notes

Implemented in `vetc/core/constants.py`, `vetc/core/tensor_checks.py`, and `vetc/core/errors.py`.

## Limitations

Validation is runtime shape checking, not a static type system.

## Future Work

Add optional stricter validation modes for training and deployment.

