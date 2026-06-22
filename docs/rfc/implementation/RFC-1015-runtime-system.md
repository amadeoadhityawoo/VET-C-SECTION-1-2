# RFC-1015 - Runtime System

Status: MVP Implemented
Section: Implementation RFC
Version: v0.1

## Purpose

Connects all Section 2 MVP modules into one runnable VET-C system.

## Scope

Includes end-to-end dummy inference, recurrent state carryover, regime-token carryover, finite checks, and runtime timing summary.

## Current MVP Status

Implemented in this repository.

## Key Tensor Contracts

- `rgb`: `[B, 3, 64, 64]`
- `depth`: `[B, 1, 64, 64]`
- `h`: `[B, 512]`
- `z_post`: `[B, 64]`
- `regime_token`: `[B, 96]`
- `action`: `[B, 4]`

## Implementation Notes

Implemented in `vetc/runtime/vetc_system.py`, `vetc/runtime/runtime_loop.py`, `scripts/run_dummy_infer.py`, and `scripts/run_runtime_loop.py`.

## Validation Result

Pytest command:

```bash
python3 -m pytest -q
```

Pytest result:

```text
48 passed
```

Runtime command:

```bash
python3 scripts/run_runtime_loop.py --cycles 10000
```

Runtime result:

```text
cycles: 10000
final_action_shape: (2, 4)
nan_detected: False
avg_step_time_ms: 4.3522
```

## Limitations

Runtime inputs are dummy tensors and do not connect to real hardware.

## Future Work

Add real sensor I/O, safety gates, logging, hardware deployment, and performance profiling.
