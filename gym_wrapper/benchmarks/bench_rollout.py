#!/usr/bin/env python3
"""
Benchmark steps/sec for VangersEnv.

Example:
  VANGERS_ENGINE_PATH=/x/Code/Vangers/build/gym_wrapper \
  VANGERS_DATA_PATH=/x/Assets/Vangers/original/data \
  CLUNK_ROOT=/x/Code/Vangers/external/clunk/install \
  LD_LIBRARY_PATH=/x/Code/Vangers/external/clunk/install/lib:$LD_LIBRARY_PATH \
  python3.12 gym_wrapper/benchmarks/bench_rollout.py --steps 2000 --render-mode none
"""

import argparse
import os
import sys
import time

import numpy as np

DEFAULT_MECHOS_NAME = "OxidizeMonk"

# Ensure repo root is on sys.path so "import gym_wrapper" works when run from source.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(REPO_ROOT)

from gym_wrapper.vangers_env import VangersEnv


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark VangersEnv throughput")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--render-mode", choices=["none", "rgb_array"], default="none")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    args = parser.parse_args()

    render_mode = None if args.render_mode == "none" else "rgb_array"

    env = VangersEnv(
        width=args.width,
        height=args.height,
        render_mode=render_mode,
        headless=args.headless,
        mechos_name=DEFAULT_MECHOS_NAME,
    )
    env.reset()

    steps = int(args.steps)
    t0 = time.time()
    for _ in range(steps):
        action = env.action_space.sample()
        env.step(action)
        if render_mode == "rgb_array":
            _ = env.render()
    t1 = time.time()

    elapsed = max(1e-6, t1 - t0)
    print(f"steps={steps} elapsed={elapsed:.3f}s steps/sec={steps/elapsed:.1f}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
