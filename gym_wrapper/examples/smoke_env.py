#!/usr/bin/env python3
"""
Smoke test for Vangers Gym wrapper.

Runs a single reset/step with a specified data path to verify that
engine init and asset loading succeed.
"""

import argparse
import faulthandler
import os
import sys
import time

DEFAULT_MECHOS_NAME = "OxidizeMonk"

# Ensure we can import the vangers_env module from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from vangers_env import VangersEnv
except ImportError:
    print("Error: Could not import vangers_env. Run from the examples directory or install the package.")
    sys.exit(1)


def run_smoke(data_path: str, width: int = 160, height: int = 120, timeout: int = 30, headless: bool = True) -> int:
    os.environ["VANGERS_DATA_PATH"] = data_path
    print(f"[smoke] Using VANGERS_DATA_PATH={data_path}", flush=True)

    if timeout > 0:
        faulthandler.dump_traceback_later(timeout, repeat=False)

    try:
        print("[smoke] Creating env...", flush=True)
        env = VangersEnv(
            width=width,
            height=height,
            render_mode="rgb_array",
            headless=headless,
            mechos_name=DEFAULT_MECHOS_NAME,
        )
        print("[smoke] Env created", flush=True)
    except Exception as exc:
        print(f"[smoke] Env init failed: {exc}", flush=True)
        return 1

    try:
        print("[smoke] Calling reset...", flush=True)
        obs, info = env.reset(seed=int(time.time()))
        print("[smoke] Reset OK", flush=True)
        action = env.action_space.sample()
        print("[smoke] Calling step...", flush=True)
        obs, reward, terminated, truncated, info = env.step(action)
        print("[smoke] reset/step OK", flush=True)
        print(f"[smoke] obs keys: {list(obs.keys())}", flush=True)
        print(f"[smoke] reward: {reward:.4f} terminated={terminated} truncated={truncated}", flush=True)
        return 0
    except Exception as exc:
        print(f"[smoke] reset/step failed: {exc}", flush=True)
        return 2
    finally:
        try:
            faulthandler.cancel_dump_traceback_later()
        except Exception:
            pass
        try:
            env.close()
        except Exception:
            pass
        try:
            env.engine_lib.cleanup()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vangers Gym wrapper smoke test")
    parser.add_argument("--data-path", required=True, help="Path to Vangers data root")
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--timeout", type=int, default=30, help="Seconds before dumping traceback (0 to disable)")
    parser.add_argument("--headless", action="store_true", default=True, help="Enable SDL dummy driver")
    parser.add_argument("--no-headless", dest="headless", action="store_false", help="Disable SDL dummy driver")

    args = parser.parse_args()
    sys.exit(run_smoke(args.data_path, width=args.width, height=args.height, timeout=args.timeout, headless=args.headless))
