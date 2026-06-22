# VET-C Section 2 MVP Report

## Implementation Summary

VET-C is an edge AI drone control model architecture targeting Jetson Orin Nano-class systems. This repository now contains the Section 1 foundation and Section 2 MVP pipeline from RFC-1000 through RFC-1015.

The implemented MVP connects dummy sensor inputs to a normalized action command through observation processing, perception encoding, regime-conditioned fusion, RSSM dynamics, prediction error evaluation, VQ-VAE-style regime inference, terrain-field generation, policy output, and a simple runtime loop.

This is an untrained PyTorch architecture skeleton. It validates tensor contracts and end-to-end execution, but it is not a trained or deployable drone controller.

## Completed RFCs

| RFC | Component |
| --- | --- |
| RFC-1000 | Repository skeleton |
| RFC-1001 | Core dataclasses |
| RFC-1002 | Constants, tensor checks, custom exceptions |
| RFC-1003 | Base interfaces/classes |
| RFC-1004 | ObservationLayer |
| RFC-1005 | ExternalEncoder |
| RFC-1006 | InternalEncoder |
| RFC-1007 | DualPerceptionEncoder |
| RFC-1008 | RegimeConditionedFusion |
| RFC-1009 | RSSMCell |
| RFC-1010 | UnifiedRSSM |
| RFC-1011 | PredictionErrorEvaluator |
| RFC-1012 | RegimeEngineVQVAE |
| RFC-1013 | VETTerrainGenerator |
| RFC-1014 | FastGradientPolicy |
| RFC-1015 | VETCSystem and RuntimeLoop |

## Module Explanation

`vetc.core` defines shared tensor dimensions, dataclasses, exceptions, tensor validation helpers, and base interfaces.

`vetc.observation` converts raw sensor tensors into canonical external and internal observation tensors. The runtime system supplies zero placeholders for flow and GPS when using the simplified RFC-1015 dummy input signature.

`vetc.encoder` contains the external visual encoder, internal sensor-history encoder, and dual perception wrapper.

`vetc.fusion` pools spatial visual latents, concatenates them with internal latents, and applies regime-conditioned FiLM before producing a fused `[B, 256]` representation.

`vetc.rssm` implements the RSSM cell and unified wrapper for recurrent hidden state and latent prior/posterior tensors.

`vetc.prediction` computes surprise, KL/error maps, and safe sigma estimates from posterior and prior latent states.

`vetc.regime` projects hidden state plus surprise into a VQ-VAE-style quantized regime token and tracks simple code usage.

`vetc.terrain` converts hidden state, regime token, and surprise into a five-element psi field: value, entropy, confidence, prediction precision, and homeostatic weight.

`vetc.policy` converts the psi field into a normalized `[B, 4]` action command: thrust, roll, pitch, and yaw.

`vetc.runtime` wires all implemented modules into `VETCSystem` and provides a dummy `RuntimeLoop`.

## Repository Structure Note

The original PDF roadmap describes a larger future VET-C repository containing dataset, trainer, losses, simulator, deploy, memory, and rollout systems. This repository currently implements the Section 2 MVP scope, RFC-1000 to RFC-1015. Therefore, only the modules required for the runnable v0.1 pipeline are implemented. Future folders are included as placeholders and will be filled when Section 3+ systems are implemented.

## End-to-End Pipeline

```text
rgb/depth/imu/battery/motor_state
  -> ObservationLayer
  -> ext_obs, int_obs
  -> DualPerceptionEncoder
  -> z_ext, z_int
  -> RegimeConditionedFusion
  -> fused
  -> UnifiedRSSM
  -> h, z_prior, z_post, prior/posterior Gaussian params
  -> PredictionErrorEvaluator
  -> surprise, kl_map, safe_sigma
  -> RegimeEngineVQVAE
  -> regime_token, commit_loss
  -> VETTerrainGenerator
  -> psi_field
  -> FastGradientPolicy
  -> action
```

## Tensor Shape Table

| Tensor | Shape |
| --- | --- |
| `rgb` | `[B, 3, 64, 64]` |
| `depth` | `[B, 1, 64, 64]` |
| `imu` | `[B, 32, 6]` |
| `battery` | `[B, 32, 1]` |
| `motor_state` | `[B, 32, 4]` |
| `ext_obs` | `[B, 6, 64, 64]` |
| `int_obs` | `[B, 32, 11]` |
| `z_ext` | `[B, 128, 16, 16]` |
| `z_int` | `[B, 128]` |
| `fused` | `[B, 256]` |
| `h` | `[B, 512]` |
| `z_prior` | `[B, 64]` |
| `z_post` | `[B, 64]` |
| `surprise` | `[B, 1]` |
| `kl_map` | `[B, 64]` |
| `safe_sigma` | `[B, 64]` |
| `regime_token` | `[B, 96]` |
| `psi_field` | `[B, 5]` |
| `action` | `[B, 4]` |

For current dummy verification, `B = 2`.

## Test Results

Command:

```bash
python3 -m pytest -q
```

Current result:

```text
48 passed
```

## Runtime Result

Dummy inference:

```bash
python3 scripts/run_dummy_infer.py
```

Result summary:

```text
action shape: [2, 4]
```

Runtime loop:

```bash
python3 scripts/run_runtime_loop.py --cycles 100
```

Result summary:

```text
cycles: 100
final_action_shape: (2, 4)
nan_detected: False
avg_step_time_ms: approximately 4.1 ms
```

## Current Limitations

- The model is untrained and randomly initialized.
- Runtime scripts use synthetic dummy sensor tensors.
- The MVP validates pipeline execution and tensor contracts, not control quality.
- It is not deployable as a real drone controller.
- No dataset, training loop, checkpointing, hardware I/O, safety layer, calibration, or Jetson deployment optimization is implemented.
- Future modules such as memory, dreaming, density flow, MCTS, and H+ Grasp are not included.

## Recommended Next Steps

Future work should add training data interfaces, supervised/self-supervised training loops, checkpoint support, deterministic evaluation harnesses, hardware input adapters, actuator output adapters, safety constraints, profiling on Jetson Orin Nano, and deployment packaging.
