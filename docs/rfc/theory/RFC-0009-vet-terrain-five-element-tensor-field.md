# RFC-0009 - VET Terrain Five Element Tensor Field

Status: Draft
Section: Theory RFC
Version: v0.1

## Purpose

Defines the five-element VET terrain field used by policy modules.

## Scope

Includes value, entropy, confidence, prediction precision, and homeostatic weight.

## Current MVP Status

This is partially implemented in the Section 2 MVP through `VETTerrainGenerator`.

## Key Tensor Contracts

- `h`: `[B, 512]`
- `regime_token`: `[B, 96]`
- `surprise`: `[B, 1]`
- `psi_field`: `[B, 5]`

## Implementation Notes

Implemented in `vetc/terrain/fourier_features.py` and `vetc/terrain/vet_terrain_generator.py`.

## Limitations

The MVP generator is untrained and uses a compact MLP with stabilized outputs.

## Future Work

Add trained terrain semantics, richer diagnostics, and integration with future planning systems.

