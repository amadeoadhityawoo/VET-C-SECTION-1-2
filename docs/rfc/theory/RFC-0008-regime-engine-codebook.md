# RFC-0008 - Regime Engine Codebook

Status: Draft
Section: Theory RFC
Version: v0.1

## Purpose

Defines the theory of regime tokens and codebook-based regime inference.

## Scope

Includes regime-token dimensionality, vector quantization, code usage, and dead-code management.

## Current MVP Status

This is partially implemented in the Section 2 MVP through a simple VQ-VAE-style regime engine.

## Key Tensor Contracts

- `h`: `[B, 512]`
- `surprise`: `[B, 1]`
- `regime_token`: `[B, 96]`
- `code_indices`: `[B]`

## Implementation Notes

Implemented in `vetc/regime/vector_quantizer.py`, `vetc/regime/dead_code_manager.py`, and `vetc/regime/regime_engine_vqvae.py`.

## Limitations

Dead-code management is a simple usage counter; EMA replacement is not implemented.

## Future Work

Add robust codebook training, EMA updates, dead-code replacement, diagnostics, and visualization.

