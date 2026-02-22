#!/usr/bin/env python3
"""
Record rollouts to a dataset file.

This records observations (state vector), actions, rewards, and info JSON.
Optional video capture can be enabled for short recordings.

Example:
  VANGERS_ENGINE_PATH=/x/Code/Vangers/build/gym_wrapper \
  VANGERS_DATA_PATH=/x/Assets/Vangers/original/data \
  CLUNK_ROOT=/x/Code/Vangers/external/clunk/install \
  LD_LIBRARY_PATH=/x/Code/Vangers/external/clunk/install/lib:$LD_LIBRARY_PATH \
  python3.12 gym_wrapper/benchmarks/record_trajectories.py --steps 2000 --out /x/Code/Vangers/rollout.npz
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# Default to the requested starting mechos unless overridden.
DEFAULT_MECHOS_NAME = "OxidizeMonk"

try:
    import cv2
except Exception:
    cv2 = None

# Ensure repo root is on sys.path so "import gym_wrapper" works when run from source.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(REPO_ROOT)

from gym_wrapper.vangers_env import VangersEnv


class KeyState:
    def __init__(self):
        self._pressed = set()
        self._listener = None
        self._use_pynput = False
        self._quit_requested = False

        try:
            from pynput import keyboard
            self._use_pynput = True

            def on_press(key):
                if key == keyboard.Key.esc:
                    self._quit_requested = True
                    return
                try:
                    k = key.char.lower()
                except AttributeError:
                    k = str(key)
                if k in self._pressed:
                    return
                self._pressed.add(k)

            def on_release(key):
                try:
                    self._pressed.discard(key.char.lower())
                except AttributeError:
                    self._pressed.discard(str(key))

            self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            self._listener.start()
        except Exception:
            self._use_pynput = False

    def available(self):
        return self._use_pynput

    def pressed(self):
        return set(self._pressed)

    def quit_requested(self):
        return self._quit_requested


def action_from_keys(pressed):
    # Default Vangers keys (from src/iscreen/controls.cpp):
    # arrows: move/turn, W: activate kid, Q: handbrake, E/LCTRL: fire
    forward = "Key.up" in pressed or "up" in pressed
    backward = "Key.down" in pressed or "down" in pressed
    left = "Key.left" in pressed or "left" in pressed
    right = "Key.right" in pressed or "right" in pressed
    fire = "e" in pressed or "Key.ctrl_l" in pressed or "Key.ctrl_r" in pressed
    special1 = "w" in pressed
    special2 = "q" in pressed

    move = 1 if forward and not backward else 2 if backward and not forward else 0
    turn = 1 if left and not right else 2 if right and not left else 0
    fire_val = 1 if fire else 0
    sp1 = 1 if special1 else 0
    sp2 = 1 if special2 else 0
    return np.array([move, turn, fire_val, sp1, sp2], dtype=np.int64)


def main() -> int:
    parser = argparse.ArgumentParser(description="Record Vangers rollouts to dataset")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--out", required=True, help="Output .npz file")
    parser.add_argument("--data-path", default=None, help="Overrides VANGERS_DATA_PATH")
    parser.add_argument("--engine-path", default=None, help="Overrides VANGERS_ENGINE_PATH")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--policy", choices=["random", "zeros", "human"], default="random")
    parser.add_argument("--record-video", action="store_true", default=False)
    parser.add_argument("--video-out", default=None, help="Optional MP4 output path")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--realtime", action="store_true", default=True)
    parser.add_argument("--no-realtime", dest="realtime", action="store_false")
    args = parser.parse_args()

    if args.data_path:
        os.environ["VANGERS_DATA_PATH"] = args.data_path
    if args.engine_path:
        os.environ["VANGERS_ENGINE_PATH"] = args.engine_path
    if not os.environ.get("VANGERS_DATA_PATH") and not os.environ.get("VANGERS_RESOURCE_PATH"):
        fallback_data = "/x/Assets/Vangers/original/data"
        if os.path.isdir(fallback_data):
            os.environ["VANGERS_DATA_PATH"] = fallback_data

    render_mode = "rgb_array" if args.record_video else None
    if args.policy == "human":
        if cv2 is None:
            raise RuntimeError("cv2 is required for --policy human (window display)")
        render_mode = "rgb_array"
        args.headless = True
    env = VangersEnv(
        width=args.width,
        height=args.height,
        render_mode=render_mode,
        resource_path=os.environ.get("VANGERS_DATA_PATH") or os.environ.get("VANGERS_RESOURCE_PATH"),
        mechos_name=DEFAULT_MECHOS_NAME,
        headless=args.headless,
    )
    try:
        env.engine_lib.set_mechos_name(DEFAULT_MECHOS_NAME)
    except Exception:
        pass

    obs, info = env.reset(seed=int(time.time()))

    actions = []
    rewards = []
    states = []
    terminations = []
    truncations = []

    info_path = os.path.splitext(args.out)[0] + ".jsonl"
    info_f = open(info_path, "w", encoding="utf-8")

    writer = None
    if args.record_video:
        if cv2 is None:
            raise RuntimeError("cv2 not available; install opencv-python or disable --record-video")
        out_path = args.video_out or (os.path.splitext(args.out)[0] + ".mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, float(args.fps), (args.width, args.height))
        if not writer.isOpened():
            raise RuntimeError("Failed to open video writer")

    key_state = KeyState() if args.policy == "human" else None
    if args.policy == "human" and not key_state.available():
        raise RuntimeError("pynput is required for --policy human (keyboard input)")
    window_name = "VangersEnv"

    target_dt = 1.0 / max(1, args.fps)
    try:
        for _ in range(int(args.steps)):
            frame_start = time.time()
            if args.policy == "zeros":
                action = np.zeros(5, dtype=int)
            elif args.policy == "human":
                action = action_from_keys(key_state.pressed())
                if key_state.quit_requested():
                    break
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)

            actions.append(np.array(action, dtype=np.int64))
            rewards.append(float(reward))
            states.append(np.array(obs.get("state"), dtype=np.float32))
            terminations.append(bool(terminated))
            truncations.append(bool(truncated))
            info_f.write(json.dumps(info) + "\n")

            if render_mode is not None:
                frame = env.render()
                if frame is not None:
                    if frame.shape[0] != args.height or frame.shape[1] != args.width:
                        frame = cv2.resize(frame, (args.width, args.height))
                    if args.policy == "human":
                        cv2.imshow(window_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)
                    if writer is not None:
                        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            if args.policy == "human" and args.realtime:
                elapsed = time.time() - frame_start
                if elapsed < target_dt:
                    time.sleep(target_dt - elapsed)

            if terminated or truncated:
                obs, info = env.reset()

    finally:
        if writer is not None:
            writer.release()
        if args.policy == "human" and cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        info_f.close()
        env.close()

    np.savez_compressed(
        args.out,
        actions=np.stack(actions, axis=0),
        rewards=np.array(rewards, dtype=np.float32),
        states=np.stack(states, axis=0),
        terminations=np.array(terminations, dtype=np.bool_),
        truncations=np.array(truncations, dtype=np.bool_),
    )

    print(f"Saved dataset: {args.out}")
    print(f"Saved info log: {info_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
