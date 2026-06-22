# VET-C

VET-C is an edge AI drone control model architecture targeting NVIDIA Jetson Orin Nano-class deployment. The current repository contains a PyTorch MVP that proves the end-to-end tensor contracts for perception, latent dynamics, regime inference, terrain-field generation, policy output, and a dummy runtime loop.

This repository is a runnable PyTorch MVP architecture skeleton. It validates tensor contracts and end-to-end module connectivity using dummy inputs, but it is not yet a trained or deployable drone controller.

## Sections

Section 1 in the PDF refers to the Theory RFC series, RFC-0000 through RFC-0016. It defines the system architecture, tensor interface contract, repository architecture, training pipeline specification, runtime inference specification, loss function specification, regime engine and codebook, VET terrain and five-element tensor field, memory, dreaming, density flow, higher-level theory, and MVP roadmap.

Section 2 in the PDF refers to the Implementation RFC series, RFC-1000 through RFC-1015. It defines the actual Codex implementation sprints: repository initialization, core types, constants and tensor checks, interfaces, observation layer, external/internal encoders, dual perception encoder, fusion, RSSM, prediction error, regime VQ-VAE, terrain generator, fast policy, and runtime system.

This repository currently implements the Section 2 MVP scope, RFC-1000 to RFC-1015. It does not yet fully implement all Section 1 theory documents as markdown files. It does not yet implement Section 3+ training, dataset, simulation, memory, dream, or deployment systems.

## Implemented RFCs

- RFC-1000: repository skeleton
- RFC-1001: core dataclasses
- RFC-1002: constants, tensor checks, and custom exceptions
- RFC-1003: base interfaces/classes
- RFC-1004: `ObservationLayer`
- RFC-1005: `ExternalEncoder`
- RFC-1006: `InternalEncoder`
- RFC-1007: `DualPerceptionEncoder`
- RFC-1008: `RegimeConditionedFusion`
- RFC-1009: `RSSMCell`
- RFC-1010: `UnifiedRSSM`
- RFC-1011: `PredictionErrorEvaluator`
- RFC-1012: `RegimeEngineVQVAE`
- RFC-1013: `VETTerrainGenerator`
- RFC-1014: `FastGradientPolicy`
- RFC-1015: `VETCSystem` and `RuntimeLoop`

## Repository Structure

```text
vetc/
  core/          Core dataclasses, constants, tensor checks, interfaces, errors
  observation/   Sensor packet to canonical observation tensors
  encoder/       External, internal, and dual perception encoders
  fusion/        Spatial pooling and regime-conditioned fusion
  rssm/          RSSM cell and unified RSSM wrapper
  prediction/    Prediction error and surprise evaluation
  regime/        VQ-VAE-style regime token engine
  terrain/       Fourier features and psi-field generator
  policy/        Fast gradient policy for action output
  runtime/       End-to-end VET-C system and dummy runtime loop

tests/           Unit and integration tests
scripts/         Dummy inference and runtime loop entry points
configs/         Default configuration
docs/            RFC documentation folders
dataset/         Future data loading and dataset interfaces placeholder
trainer/         Future training loop and experiment orchestration placeholder
losses/          Future loss modules placeholder
simulator/       Future simulation adapters placeholder
deploy/          Future deployment assets, Docker files, and deploy docs placeholder
```

## Repository Structure Note

The original PDF roadmap describes a larger future VET-C repository containing dataset, trainer, losses, simulator, deploy, memory, and rollout systems. This repository currently implements the Section 2 MVP scope, RFC-1000 to RFC-1015. Therefore, only the modules required for the runnable v0.1 pipeline are implemented. Future folders are included as placeholders and will be filled when Section 3+ systems are implemented.

RFC placeholder documents have been added under `docs/rfc/theory` and `docs/rfc/implementation` to align the repository with the PDF roadmap.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Tests

```bash
python3 -m pytest -q
```

Expected current result:

```text
48 passed
```

## Run Dummy Inference

```bash
python3 scripts/run_dummy_infer.py
```

This runs one dummy forward pass through `VETCSystem` and prints the action tensor plus important output shapes.

## Run Runtime Loop

```bash
python3 scripts/run_runtime_loop.py --cycles 100
```

Longer dummy runs are also supported:

```bash
python3 scripts/run_runtime_loop.py --cycles 10000
```

The script prints cycle count, final action shape, whether NaN/Inf was detected, and average step time in milliseconds.

## Expected Output Shapes

| Tensor | Shape |
| --- | --- |
| `ext_obs` | `[2, 6, 64, 64]` |
| `int_obs` | `[2, 32, 11]` |
| `z_ext` | `[2, 128, 16, 16]` |
| `z_int` | `[2, 128]` |
| `fused` | `[2, 256]` |
| `h` | `[2, 512]` |
| `z_prior` | `[2, 64]` |
| `z_post` | `[2, 64]` |
| `surprise` | `[2, 1]` |
| `kl_map` | `[2, 64]` |
| `safe_sigma` | `[2, 64]` |
| `regime_token` | `[2, 96]` |
| `psi_field` | `[2, 5]` |
| `action` | `[2, 4]` |

## Current Verification

```text
python3 -m pytest -q
48 passed
```

```text
python3 scripts/run_dummy_infer.py
action shape: [2, 4]
```

```text
python3 scripts/run_runtime_loop.py --cycles 100
cycles: 100
final_action_shape: (2, 4)
nan_detected: False
avg_step_time_ms: approximately 4.1 ms
```

## Known Limitations

- The model is untrained and uses random initial weights.
- Dummy scripts use synthetic random sensor inputs.
- This proves end-to-end pipeline wiring and tensor contracts only.
- It is not a trained or deployable drone controller.
- No real sensor ingestion, flight controller integration, safety envelope, calibration, or Jetson-specific optimization is included yet.
- Memory, dreaming, density flow, MCTS, H+ Grasp, training loops, datasets, and deployment runtime are future work.
