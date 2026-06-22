# RFC-1008 - Regime Conditioned Fusion

Status: MVP Implemented
Section: Implementation RFC
Version: v0.1

## Purpose

Combines external and internal latents into a fused representation conditioned by a regime token.

## Scope

Includes spatial pooling, FiLM conditioning, and a compact MLP.

## Current MVP Status

Implemented in this repository.

## Key Tensor Contracts

- `z_ext`: `[B, 128, 16, 16]`
- `z_int`: `[B, 128]`
- `regime_token`: `[B, 96]`
- `fused`: `[B, 256]`

## Implementation Notes

Implemented in `vetc/fusion/spatial_pooler.py`, `vetc/fusion/regime_film.py`, and `vetc/fusion/regime_conditioned_fusion.py`.

## Limitations

FiLM starts stable and simple; no learned regime semantics exist before training.

## Future Work

Train and evaluate conditioning behavior.

