# RFC-1012 - Regime VQ-VAE

Status: MVP Implemented
Section: Implementation RFC
Version: v0.1

## Purpose

Infers a discrete-codebook-conditioned regime token from hidden state and surprise.

## Scope

Includes vector quantization, commitment loss, code indices, and simple code usage tracking.

## Current MVP Status

Implemented in this repository.

## Key Tensor Contracts

- `h`: `[B, 512]`
- `surprise`: `[B, 1]`
- `regime_token`: `[B, 96]`
- `code_indices`: `[B]`

## Implementation Notes

Implemented in `vetc/regime/vector_quantizer.py`, `vetc/regime/dead_code_manager.py`, and `vetc/regime/regime_engine_vqvae.py`.

## Limitations

No EMA replacement, trained codebook semantics, or codebook diagnostics exist yet.

## Future Work

Add robust codebook management and regime interpretation tools.

