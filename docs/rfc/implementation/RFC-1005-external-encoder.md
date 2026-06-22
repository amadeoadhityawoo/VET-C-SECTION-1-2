# RFC-1005 - External Encoder

Status: MVP Implemented
Section: Implementation RFC
Version: v0.1

## Purpose

Encodes external visual observations into spatial latent features.

## Scope

Includes a compact convolutional encoder for `ext_obs`.

## Current MVP Status

Implemented in this repository.

## Key Tensor Contracts

- `ext_obs`: `[B, 6, 64, 64]`
- `z_ext`: `[B, 128, 16, 16]`

## Implementation Notes

Implemented in `vetc/encoder/external_encoder.py`.

## Limitations

The encoder is untrained and minimal.

## Future Work

Train and profile the encoder for Jetson-class deployment.

