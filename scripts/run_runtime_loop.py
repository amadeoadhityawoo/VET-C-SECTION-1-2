"""Run the VET-C dummy runtime loop."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vetc.runtime.runtime_loop import RuntimeLoop  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VET-C dummy runtime loop.")
    parser.add_argument("--cycles", type=int, default=100)
    args = parser.parse_args()

    summary = RuntimeLoop().run(cycles=args.cycles)
    print(f"cycles: {summary['cycles']}")
    print(f"final_action_shape: {summary['final_action_shape']}")
    print(f"nan_detected: {summary['nan_detected']}")
    print(f"avg_step_time_ms: {summary['avg_step_time_ms']:.4f}")


if __name__ == "__main__":
    main()

