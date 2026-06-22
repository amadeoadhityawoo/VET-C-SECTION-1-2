# RFC-0007 - Loss Function

Status: Future Work
Section: Theory RFC
Version: v0.1

## Purpose

Defines intended VET-C training losses.

## Scope

Includes reconstruction, KL, commitment, policy, terrain, prediction, and possible auxiliary losses.

## Current MVP Status

This is currently a theory/specification placeholder. The MVP computes some loss-like tensors, such as VQ commitment loss, but no training loss module is implemented.

## Implementation Notes

Future implementation is expected under `losses/` and training code under `trainer/`.

## Limitations

No optimizer-facing aggregate loss is implemented yet.

## Future Work

Define and implement full training losses, weighting schedules, and tests.

