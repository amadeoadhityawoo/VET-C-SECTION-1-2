# RFC-1003 - Core Interfaces

Status: MVP Implemented
Section: Implementation RFC
Version: v0.1

## Purpose

Defines base interfaces for future VET-C modules.

## Scope

Includes abstract base classes for modules, encoders, RSSM, regime, terrain, policy, loss, and runtime components.

## Current MVP Status

Implemented in this repository.

## Implementation Notes

Implemented in `vetc/core/interfaces.py`.

## Limitations

Some MVP modules use direct concrete signatures where the abstract interface is broader than the implementation.

## Future Work

Refine interfaces as Section 3+ training and deployment modules are added.

