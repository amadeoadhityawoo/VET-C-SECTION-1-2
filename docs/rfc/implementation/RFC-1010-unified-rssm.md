# RFC-1010 - Unified RSSM

Status: MVP Implemented
Section: Implementation RFC
Version: v0.1

## Purpose

Wraps the RSSM cell with state initialization and a simpler forward API.

## Scope

Includes zero initial state creation and optional previous-state handling.

## Current MVP Status

Implemented in this repository.

## Key Tensor Contracts

- `fused`: `[B, 256]`
- `h_prev`: `[B, 512]` or `None`
- `z_prev`: `[B, 64]` or `None`
- `h`: `[B, 512]`
- `z_post`: `[B, 64]`

## Implementation Notes

Implemented in `vetc/rssm/unified_rssm.py`.

## Limitations

Only one-step inference is covered by the MVP wrapper.

## Future Work

Add sequence APIs and training-time rollout helpers.

