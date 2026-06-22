# VET-C

VET-C is an edge AI drone control model foundation targeting NVIDIA Jetson Orin Nano deployments.

## Current Scope

This repository currently implements only the VET-C v0.1 foundation sprint:

- RFC-1000: repository skeleton
- RFC-1001: core dataclasses
- RFC-1002: constants, tensor validation helpers, and custom exceptions
- RFC-1003: base interfaces/classes for future modules

Neural network modules are intentionally out of scope for this sprint. ObservationLayer, encoders, RSSM, VQ-VAE, terrain, policy, and runtime loop implementations have not been added yet.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Tests

```bash
pytest
```

