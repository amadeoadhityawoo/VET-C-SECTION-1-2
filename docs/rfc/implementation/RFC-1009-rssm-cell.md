# RFC-1009 - RSSM Cell

Status: MVP Implemented
Section: Implementation RFC
Version: v0.1

## Purpose

Defines one recurrent state-space model cell for VET-C.

## Scope

Includes GRU hidden update, prior network, posterior network, and reparameterized latent sampling.

## Current MVP Status

Implemented in this repository.

## Key Tensor Contracts

- `fused`: `[B, 256]`
- `h_prev`: `[B, 512]`
- `z_prev`: `[B, 64]`
- `h`: `[B, 512]`
- `z_prior`: `[B, 64]`
- `z_post`: `[B, 64]`

## Implementation Notes

Implemented in `vetc/rssm/prior_network.py`, `vetc/rssm/posterior_network.py`, and `vetc/rssm/rssm_cell.py`.

## Limitations

The cell is untrained and has no sequence training loop yet.

## Future Work

Add sequence rollout training, KL scheduling, and world-model evaluation.

