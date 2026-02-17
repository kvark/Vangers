#!/usr/bin/env python3
"""
Gymnasium dummy-policy rollout example.

Example:
  VANGERS_ENGINE_PATH=/x/Code/Vangers/build/gym_wrapper \
  VANGERS_DATA_PATH=/x/Assets/Vangers/original/data \
  CLUNK_ROOT=/x/Code/Vangers/external/clunk/install \
  LD_LIBRARY_PATH=/x/Code/Vangers/external/clunk/install/lib:$LD_LIBRARY_PATH \
  python3.12 gym_wrapper/examples/gym_dummy_rollout.py --steps 200
"""

import argparse
import os
import time

import gymnasium as gym

import sys
# Ensure repo root is on sys.path so "import gym_wrapper" works when run from source.
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(repo_root)

from gym_wrapper import register_gym_env


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a dummy-policy rollout via Gymnasium")
    parser.add_argument("--env-id", default="Vangers-v0")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    args = parser.parse_args()

    if args.render and args.headless:
        args.headless = False

    register_gym_env(args.env_id, width=args.width, height=args.height, headless=args.headless)

    render_mode = "human" if args.render else None
    env = gym.make(args.env_id, render_mode=render_mode)
    obs, info = env.reset(seed=int(time.time()))
    print(f"reset: obs keys={list(obs.keys())} info keys={list(info.keys()) if isinstance(info, dict) else info}")

    total_reward = 0.0
    for step in range(args.steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        if render_mode is not None:
            _ = env.render()
            time.sleep(1.0 / max(1, args.fps))
        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    print(f"done: steps={args.steps} total_reward={total_reward:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
