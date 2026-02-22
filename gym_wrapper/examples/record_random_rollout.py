#!/usr/bin/env python3
"""
Record a random-policy rollout to MP4.

Example:
  VANGERS_ENGINE_PATH=/x/Code/Vangers/build/gym_wrapper \
  VANGERS_DATA_PATH=/x/Assets/Vangers/original/data \
  CLUNK_ROOT=/x/Code/Vangers/external/clunk/install \
  LD_LIBRARY_PATH=/x/Code/Vangers/external/clunk/install/lib:$LD_LIBRARY_PATH \
  python3.12 gym_wrapper/examples/record_random_rollout.py --seconds 10 --out vangers_random_rollout.mp4
"""

import argparse
import os
import time

import cv2

DEFAULT_MECHOS_NAME = "OxidizeMonk"

# Ensure we can import the vangers_env module from the parent directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from vangers_env import VangersEnv
except ImportError:
    # Fallback if running from a different location or package not installed
    try:
        sys.path.append(os.path.join(os.getcwd(), 'gym_wrapper'))
        from vangers_env import VangersEnv
    except ImportError:
        print("Error: Could not import vangers_env. Run from the examples directory or install the package.")
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Record a Vangers random-policy rollout to MP4")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=int, default=10)
    parser.add_argument("--out", default="vangers_random_rollout.mp4")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--data-path", default=None, help="Overrides VANGERS_DATA_PATH")
    parser.add_argument("--engine-path", default=None, help="Overrides VANGERS_ENGINE_PATH")
    args = parser.parse_args()

    if args.data_path:
        os.environ["VANGERS_DATA_PATH"] = args.data_path
    if args.engine_path:
        os.environ["VANGERS_ENGINE_PATH"] = args.engine_path

    out_path = os.path.abspath(args.out)
    print(f"Recording to {out_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, float(args.fps), (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError("Failed to open video writer")

    env = VangersEnv(
        width=args.width,
        height=args.height,
        render_mode="rgb_array",
        headless=args.headless,
        mechos_name=DEFAULT_MECHOS_NAME,
    )
    try:
        env.reset(seed=int(time.time()))
        start = time.time()
        while time.time() - start < args.seconds:
            frame = env.render()
            if frame is None:
                raise RuntimeError("render() returned None")
            if frame.shape[1] != args.width or frame.shape[0] != args.height:
                frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            env.step(env.action_space.sample())
            time.sleep(1.0 / args.fps)
    finally:
        writer.release()
        try:
            env.close()
        except Exception:
            pass
        try:
            env.engine_lib.cleanup()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
