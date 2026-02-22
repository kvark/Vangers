#!/usr/bin/env python3
"""
Train an agent on the Vangers gym wrapper while showing a live preview window
of the current in-memory policy in action.

Changes from previous example:
- Uses an in-memory shared model for rendering (no disk loading).
- Uses a custom CNN+MLP combined feature extractor for image+vector observations.
- Renderer uses a lock to call model.predict safely while training continues.
- Adds debug overlay with step/reward/player stats.
- CLI flag `--in-memory-render` to toggle in-memory renderer (default: enabled).
"""

from __future__ import annotations

import argparse
import os
import threading
import time
import signal
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Optional

DEFAULT_MECHOS_NAME = "OxidizeMonk"
# NOTE: This script intentionally does not auto-select or re-exec into a virtualenv.
# Please run it explicitly with your virtualenv's Python interpreter, for example:
#   ~/Venv/bin/python gym_wrapper/examples/train_and_render.py [args]
# This keeps execution explicit and avoids modifying sys.path at runtime.

import numpy as np
from copy import deepcopy

# This example expects you to run it with the desired virtualenv activated.
# Example:
#   ~/Venv/bin/python gym_wrapper/examples/train_and_render.py --timesteps 50000
# Import the vangers wrapper (assumed available in your runtime environment).
try:
    from vangers_env import VangersEnv
except Exception:
    import sys
    # Fallback: add the package directory to sys.path when running from the repo
    pkg_dir = Path(__file__).resolve().parents[1]  # gym_wrapper directory
    if str(pkg_dir) not in sys.path:
        sys.path.insert(0, str(pkg_dir))
    from vangers_env import VangersEnv

# Third-party dependencies (assumed installed in the virtualenv). Fail fast if missing.
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn

# OpenCV is required for rendering in this example and is assumed present in the virtualenv.
import cv2
try:
    import gymnasium as gym  # noqa: F401
    print(f"Using Gymnasium {getattr(gym, '__version__', '')}")
except Exception:
    pass


class PeriodicSaveCallback(BaseCallback):
    """
    Callback that saves the model to a designated path every `save_freq` steps.
    Additionally updates shared training statistics (steps/sec) if a shared_model
    container is provided. This callback accepts an optional `shared_model`
    dictionary (the same dict used by the renderer and snapshotter) and updates
    `shared_model['stats']` with a small stats dict.
    """

    def __init__(self, save_dir: str, save_freq: int = 5000, verbose: int = 0, shared_model: Optional[dict] = None, timeout_seconds: Optional[float] = None):
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_freq = int(save_freq)
        self._last_saved = 0
        self._start_time = time.time()
        self.shared_model = shared_model
        self.timeout_seconds = float(timeout_seconds) if timeout_seconds is not None else None

    def _on_step(self) -> bool:
        # Update simple training throughput stats on every callback invocation
        try:
            now = time.time()
            elapsed = max(1e-6, now - self._start_time)
            steps_per_sec = float(self.num_timesteps) / elapsed
            if self.shared_model is not None:
                try:
                    # best-effort update; shared model may be used concurrently
                    self.shared_model.setdefault('stats', {})['train_steps_per_sec'] = steps_per_sec
                    self.shared_model['stats']['timesteps'] = int(self.num_timesteps)
                    self.shared_model['stats']['last_update'] = now
                except Exception:
                    pass
        except Exception:
            pass

        # This method is called frequently; save only on intervals
        try:
            num = int(self.num_timesteps)
            if num - self._last_saved >= self.save_freq:
                fname = self.save_dir / f"model_{num}.zip"
                # save is thread-safe for SB3 (best-effort). If it fails, ignore.
                try:
                    self.model.save(str(fname))
                    latest = self.save_dir / "model_latest.zip"
                    # Update latest symlink (or copy on Windows)
                    try:
                        if latest.exists():
                            latest.unlink()
                        latest.symlink_to(fname.name)
                    except Exception:
                        # fallback to copying the file to model_latest.zip
                        shutil.copyfile(str(fname), str(self.save_dir / "model_latest.zip"))
                    if self.verbose:
                        print(f"[callback] Saved checkpoint: {fname}")
                except Exception as e:
                    if self.verbose:
                        print(f"[callback] Failed to save checkpoint: {e}")
                self._last_saved = num
        except Exception:
            pass
        # Timeout check (wall-clock)
        ts = getattr(self, "timeout_seconds", None)
        if ts is not None:
            if (time.time() - self._start_time) >= float(ts):
                if self.verbose:
                    print("[callback] Timeout reached, stopping training.")
                return False
        return True


def make_vec_env_from_vangers(width: int, height: int, engine_lib_path: Optional[str] = None):
    """
    Create a DummyVecEnv wrapping a single VangersEnv instance. Using a vectorized
    wrapper makes it easy to feed the env into SB3.
    The returned VecEnv is wrapped with VecTransposeImage so policies see
    channel-first images (C, H, W) as expected by SB3's PyTorch policies.
    """
    def _thunk():
        return VangersEnv(
            width=width,
            height=height,
            render_mode=None,
            engine_lib_path=engine_lib_path,
            mechos_name=DEFAULT_MECHOS_NAME,
        )
    vec = DummyVecEnv([_thunk])
    # Wrap with VecTransposeImage to transpose HWC->CHW for the policy.
    try:
        vec = VecTransposeImage(vec)
    except Exception:
        # If VecTransposeImage is unavailable or fails, fall back to the original vec.
        pass
    return vec


# ---------------------------------------------------------------------------
# Custom feature extractor: CNN for 'screen' + MLP for 'state' -> concatenated
# ---------------------------------------------------------------------------
if BaseFeaturesExtractor is not object and th is not None and nn is not None:
    class CombinedExtractor(BaseFeaturesExtractor):
        """
        Combined extractor for dict observations with keys:
          - 'screen': expected channel-first image (C x H x W)
          - 'state' : vector of floats

        This simplified extractor assumes the policy-visible observation space uses
        channel-first ordering (C, H, W) as is typical after SB3's VecTransposeImage.
        It performs an explicit check at construction time and raises a helpful
        error if the observation space does not match this expectation.
        """
        def __init__(self, observation_space, cnn_output_dim: int = 128):
            # observation_space is a gym.spaces.Dict
            super().__init__(observation_space, features_dim=1)
            assert 'screen' in observation_space.spaces and 'state' in observation_space.spaces, \
                "CombinedExtractor expects a dict observation with 'screen' and 'state'"

            screen_space = observation_space.spaces['screen']
            # Expect channels-first (C, H, W) for the policy-visible observation space.
            if len(screen_space.shape) == 3:
                C, H, W = map(int, screen_space.shape)
            elif len(screen_space.shape) == 2:
                # grayscale (H, W) -> treat as (1, H, W)
                C = 1
                H, W = map(int, screen_space.shape)
            else:
                raise RuntimeError(
                    "Unsupported 'screen' observation shape in policy-visible observation_space: "
                    f"{screen_space.shape}. Expected (C, H, W) (channel-first)."
                )

            # Sanity check for channels: expect common image channels
            if C not in (1, 3, 4):
                raise RuntimeError(
                    "CombinedExtractor expects channel-first images with C in {1,3,4}. "
                    "If your environment provides HxWxC (channels-last), SB3 may have added a "
                    "transpose wrapper; ensure the policy sees channel-first images (VecTransposeImage) "
                    "or adapt the extractor accordingly."
                )

            self.cnn = nn.Sequential(
                nn.Conv2d(C, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            # compute flattened size using the inferred H,W
            with th.no_grad():
                dummy = th.zeros(1, C, H, W)
                cnn_out = self.cnn(dummy)
                cnn_flat = int(cnn_out.shape[1])

            self.fc_screen = nn.Sequential(
                nn.Linear(cnn_flat, cnn_output_dim),
                nn.ReLU()
            )

            state_dim = int(observation_space.spaces['state'].shape[0])
            self.fc_state = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU()
            )

            self._features_dim = cnn_output_dim + 64

        def forward(self, observations):
            # observations: dict of tensors or numpy arrays
            screen = observations['screen']
            state = observations['state']

            # Accept numpy or torch tensors. Expect channel-first ordering (C,H,W) or batched (B,C,H,W)
            if isinstance(screen, np.ndarray):
                x = th.as_tensor(screen, dtype=th.float32)
                if x.ndim == 3:
                    # single (C,H,W) -> (1,C,H,W)
                    x = x.unsqueeze(0)
                elif x.ndim == 4:
                    # Batched: ensure it's (B,C,H,W)
                    # If it's (B,H,W,C), the extractor will see incorrect shape; we keep the explicit
                    # assumption that the policy-visible space is channel-first.
                    pass
                else:
                    raise ValueError(f"Unsupported screen ndarray ndim: {x.ndim}")
            else:
                x = screen
                if x.ndim == 3:
                    x = x.unsqueeze(0)
                elif x.ndim != 4:
                    raise ValueError(f"Unsupported screen tensor ndim: {x.ndim}")

            # normalize to [0,1]
            x = x / 255.0
            x = self.cnn(x)
            x = self.fc_screen(x)

            # State
            if isinstance(state, np.ndarray):
                s = th.as_tensor(state, dtype=th.float32)
                if s.ndim == 1:
                    s = s.unsqueeze(0)
            else:
                s = state
                if s.ndim == 1:
                    s = s.unsqueeze(0)
            s = self.fc_state(s)

            features = th.cat([x, s], dim=1)
            return features
else:
    CombinedExtractor = None  # type: ignore


def renderer_in_memory_loop(
    render_width: int,
    render_height: int,
    shared_model_container: dict,
    model_lock: threading.Lock,
    stop_event: threading.Event,
    play_speed: float = 20.0,
    preview_fps: int = 20,
    debug_overlay: bool = True,
    train_input_size: Optional[tuple] = None,
    record_path: Optional[str] = None,
):
    """
    Renderer that uses the in-memory shared model (or its snapshot) for live preview.

    Notes:
      - Prefers `shared_model_container['snapshot']` (created by the snapshotter) to
        avoid contention with training. Falls back to `shared_model_container['model']`.
      - Ensures the 'screen' array is transposed from HxWxC -> CxHxW before calling
        `model.predict`, matching the observation format the policy saw during training
        (VecTransposeImage was used for the training env).
    """
    if cv2 is None:
        print("OpenCV (cv2) not installed. In-memory renderer disabled.")
        return

    print("[renderer] Starting in-memory preview renderer.")
    preview_env = VangersEnv(
        width=render_width,
        height=render_height,
        render_mode="rgb_array",
        mechos_name=DEFAULT_MECHOS_NAME,
    )
    try:
        preview_env.engine_lib.set_render_mode(preview_env.instance.instance_ptr, 1)
    except Exception:
        pass
    # Try to set fast-forward time scale
    try:
        preview_env.engine_lib.set_time_scale(preview_env.instance.instance_ptr, float(play_speed))
        print(f"[renderer] Set engine time scale to {play_speed}x for preview.")
    except Exception:
        print("[renderer] Could not set engine time scale; preview will run at engine default speed.")

    window_name = "Vangers - Live (in-memory policy)"
    window_ok = False
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, render_width, render_height)
        window_ok = True
    except Exception as e:
        print(f"[renderer] Headless mode detected (cv2 window unavailable): {e}. Rendering will be disabled.")

    frame_delay_ms = max(1, int(1000 / preview_fps))
    writer = None
    if record_path:
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(record_path, fourcc, float(preview_fps), (int(render_width), int(render_height)))
            if not writer or not writer.isOpened():
                print(f"[renderer] Failed to open video writer at {record_path}")
                writer = None
            else:
                print(f"[renderer] Recording preview to: {record_path}")
        except Exception as e:
            print(f"[renderer] Could not initialize video writer: {e}")
            writer = None

    try:
        while not stop_event.is_set():
            # Prefer using a lightweight snapshot (if available) so rendering doesn't
            # block training. Fallback to the live model otherwise.
            model = shared_model_container.get('snapshot') or shared_model_container.get('model')
            if model is None:
                # show waiting frame
                waiting = np.zeros((render_height, render_width, 3), dtype=np.uint8)
                cv2.putText(waiting, "Waiting for model...", (10, render_height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow(window_name, waiting)
                if cv2.waitKey(frame_delay_ms) & 0xFF == ord('q'):
                    stop_event.set()
                    break
                time.sleep(0.2)
                continue

            # Run a preview episode using the live/snapshot model
            obs, info = preview_env.reset()
            done = False
            episode_reward = 0.0
            steps = 0

            while not done and not stop_event.is_set():
                # Build observation matching training shape if needed
                obs_for_model = obs
                if train_input_size is not None:
                    # resize screen component to the model's expected training size
                    try:
                        screen = obs['screen']  # H x W x C
                        resized = cv2.resize(screen, (train_input_size[0], train_input_size[1]), interpolation=cv2.INTER_LINEAR)
                        # transpose HWC -> CHW
                        if resized.ndim == 3:
                            transposed = np.transpose(resized, (2, 0, 1)).copy()
                        elif resized.ndim == 2:
                            transposed = resized[np.newaxis, ...].copy()
                        else:
                            transposed = np.transpose(resized, (2, 0, 1)).copy()
                        obs_for_model = {'screen': transposed, 'state': obs['state']}
                    except Exception:
                        # Best-effort: try a safe transpose of current screen if resize failed.
                        try:
                            screen = obs['screen']
                            if screen.ndim == 3:
                                transposed = np.transpose(screen, (2, 0, 1)).copy()
                                obs_for_model = {'screen': transposed, 'state': obs['state']}
                        except Exception:
                            obs_for_model = obs

                # Predict with lock (snapshot should be safe but we keep a short lock for consistency)
                try:
                    with model_lock:
                        # Ensure batch dimension and correct shapes for dict observation
                        if isinstance(obs_for_model, dict):
                            scr = obs_for_model.get('screen')
                            st = obs_for_model.get('state')
                            if isinstance(scr, np.ndarray) and scr.ndim == 3:
                                scr = scr[None, ...]
                            if isinstance(st, np.ndarray) and st.ndim == 1:
                                st = st[None, ...]
                            obs_batched = {'screen': scr, 'state': st}
                        else:
                            obs_batched = obs_for_model
                        act, _ = model.predict(obs_batched, deterministic=True)
                        action = np.asarray(act)
                        if action.ndim == 2 and action.shape[0] == 1:
                            action = action[0]
                        action = action.reshape(-1)
                        if action.size < 5:
                            padded = np.zeros(5, dtype=np.int64)
                            padded[:action.size] = action
                            action = padded
                        else:
                            action = action[:5]
                        nvec = np.array([3, 3, 2, 2, 2], dtype=np.int64)
                        action = np.mod(action.astype(np.int64), nvec)
                except Exception:
                    # fallback action
                    action = np.zeros(5, dtype=int)

                result = preview_env.step(action)
                if len(result) == 5:
                    obs, reward, terminated, truncated, info = result
                    done = bool(terminated or truncated)
                else:
                    obs, reward, done, info = result

                episode_reward += float(reward)
                steps += 1

                # Get the frame to render (RGB)
                frame = None
                try:
                    frame = preview_env.render(mode="rgb_array")
                except Exception:
                    try:
                        frame = preview_env.instance._get_observation().get("screen")
                    except Exception:
                        frame = None

                if frame is None:
                    frame = np.zeros((render_height, render_width, 3), dtype=np.uint8)

                # Debug overlay
                if debug_overlay:
                    try:
                        # draw semi-transparent background
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (0,0), (render_width, 36), (0,0,0), -1)
                        alpha = 0.5
                        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                        info_text = f"Steps: {steps}  Reward: {episode_reward:.2f}"
                        # player stats from info if available
                        ppos = info.get('player_position') or info.get('player_position', (0.0, 0.0, 0.0))
                        armor = info.get('player_armor', 0)
                        energy = info.get('player_energy', 0)
                        info_text2 = f"Pos: {ppos[0]:.1f},{ppos[1]:.1f}  Armor: {armor}  Energy: {energy}"
                        cv2.putText(frame, info_text, (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                        cv2.putText(frame, info_text2, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1, cv2.LINE_AA)
                    except Exception:
                        pass

                # Convert RGB -> BGR for OpenCV
                try:
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                except Exception:
                    bgr = frame

                if writer is not None:
                    try:
                        writer.write(bgr)
                    except Exception:
                        pass
                if window_ok:
                    try:
                        cv2.imshow(window_name, bgr)
                        if cv2.waitKey(frame_delay_ms) & 0xFF == ord('q'):
                            stop_event.set()
                            break
                    except Exception:
                        # If window display fails mid-run (e.g., display lost), switch to headless mode
                        window_ok = False
                else:
                    # Headless: just throttle to approximate preview FPS
                    time.sleep(frame_delay_ms / 1000.0)

            # small pause between preview episodes
            time.sleep(0.05)

    finally:
        try:
            preview_env.close()
        except Exception:
            pass
        try:
            if writer is not None:
                writer.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("[renderer] In-memory renderer shutdown complete.")


def renderer_loop(
    render_width: int,
    render_height: int,
    checkpoint_dir: str,
    stop_event: threading.Event,
    play_speed: float = 20.0,
    preview_fps: int = 20,
    debug_overlay: bool = True,
    train_input_size: Optional[tuple] = None,
    record_path: Optional[str] = None,
):
    """
    Disk-based renderer: periodically loads the latest saved model from `checkpoint_dir`
    and runs preview episodes using that model. This is a fallback when in-memory snapshots
    are not desired or the preview is intended to use saved checkpoints.
    """
    if cv2 is None:
        print("OpenCV (cv2) not installed. Disk-based renderer disabled.")
        return

    from pathlib import Path
    ckpt_dir = Path(checkpoint_dir)
    print("[renderer] Starting disk-based preview renderer; monitoring:", ckpt_dir)

    preview_env = VangersEnv(
        width=render_width,
        height=render_height,
        render_mode="rgb_array",
        mechos_name=DEFAULT_MECHOS_NAME,
    )
    try:
        preview_env.engine_lib.set_render_mode(preview_env.instance.instance_ptr, 1)
    except Exception:
        pass
    try:
        preview_env.engine_lib.set_time_scale(preview_env.instance.instance_ptr, float(play_speed))
    except Exception:
        pass

    window_name = "Vangers - Live (disk policy)"
    window_ok = False
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, render_width, render_height)
        window_ok = True
    except Exception as e:
        print(f"[renderer] Headless mode detected (cv2 window unavailable): {e}. Rendering will be disabled.")

    frame_delay_ms = max(1, int(1000 / preview_fps))
    writer = None
    if record_path:
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(record_path, fourcc, float(preview_fps), (int(render_width), int(render_height)))
            if not writer or not writer.isOpened():
                print(f"[renderer] Failed to open video writer at {record_path}")
                writer = None
            else:
                print(f"[renderer] Recording preview to: {record_path}")
        except Exception as e:
            print(f"[renderer] Could not initialize video writer: {e}")
            writer = None

    loaded_model = None
    last_mtime = 0.0

    try:
        while not stop_event.is_set():
            # Find the newest model file (prefer model_latest.zip, else newest model_*.zip)
            candidate_latest = ckpt_dir / "model_latest.zip"
            candidate = None
            try:
                if candidate_latest.exists():
                    candidate = candidate_latest
                else:
                    files = sorted(ckpt_dir.glob("model_*.zip"))
                    if files:
                        candidate = files[-1]
            except Exception:
                candidate = None

            # Load model if changed
            try:
                if candidate is not None:
                    mtime = candidate.stat().st_mtime
                    if loaded_model is None or mtime != last_mtime:
                        # Load on CPU for rendering
                        from stable_baselines3 import PPO as _PPO
                        # If candidate is a symlink to a versioned file, resolve it
                        try:
                            path_to_load = str(candidate.resolve())
                        except Exception:
                            path_to_load = str(candidate)
                        loaded_model = _PPO.load(path_to_load, device="cpu")
                        last_mtime = mtime
                        print(f"[renderer] Loaded model for preview: {candidate.name}")
            except Exception as e:
                # Loading failed; keep using previously loaded model (if any)
                print(f"[renderer] Failed to load model {candidate}: {e}")

            if loaded_model is None:
                waiting = np.zeros((render_height, render_width, 3), dtype=np.uint8)
                cv2.putText(waiting, "Waiting for checkpoints...", (10, render_height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow(window_name, waiting)
                if cv2.waitKey(frame_delay_ms) & 0xFF == ord('q'):
                    stop_event.set()
                    break
                time.sleep(0.5)
                continue

            # Run a preview episode using loaded_model
            obs, info = preview_env.reset()
            done = False
            episode_reward = 0.0
            steps = 0

            while not done and not stop_event.is_set():
                obs_for_model = obs
                if train_input_size is not None:
                    try:
                        screen = obs['screen']
                        resized = cv2.resize(screen, (train_input_size[0], train_input_size[1]), interpolation=cv2.INTER_LINEAR)
                        if resized.ndim == 3:
                            transposed = np.transpose(resized, (2, 0, 1)).copy()
                        elif resized.ndim == 2:
                            transposed = resized[np.newaxis, ...].copy()
                        else:
                            transposed = np.transpose(resized, (2, 0, 1)).copy()
                        obs_for_model = {'screen': transposed, 'state': obs['state']}
                    except Exception:
                        try:
                            screen = obs['screen']
                            if screen.ndim == 3:
                                transposed = np.transpose(screen, (2, 0, 1)).copy()
                                obs_for_model = {'screen': transposed, 'state': obs['state']}
                        except Exception:
                            obs_for_model = obs

                try:
                    # Ensure batch dimension and correct shapes for dict observation
                    if isinstance(obs_for_model, dict):
                        scr = obs_for_model.get('screen')
                        st = obs_for_model.get('state')
                        if isinstance(scr, np.ndarray) and scr.ndim == 3:
                            scr = scr[None, ...]
                        if isinstance(st, np.ndarray) and st.ndim == 1:
                            st = st[None, ...]
                        obs_batched = {'screen': scr, 'state': st}
                    else:
                        obs_batched = obs_for_model
                    act, _ = loaded_model.predict(obs_batched, deterministic=True)
                    action = np.asarray(act)
                    if action.ndim == 2 and action.shape[0] == 1:
                        action = action[0]
                    action = action.reshape(-1)
                    if action.size < 5:
                        padded = np.zeros(5, dtype=np.int64)
                        padded[:action.size] = action
                        action = padded
                    else:
                        action = action[:5]
                    nvec = np.array([3, 3, 2, 2, 2], dtype=np.int64)
                    action = np.mod(action.astype(np.int64), nvec)
                except Exception:
                    action = np.zeros(5, dtype=int)

                result = preview_env.step(action)
                if len(result) == 5:
                    obs, reward, terminated, truncated, info = result
                    done = bool(terminated or truncated)
                else:
                    obs, reward, done, info = result

                episode_reward += float(reward)
                steps += 1

                # Render frame
                frame = None
                try:
                    frame = preview_env.render(mode="rgb_array")
                except Exception:
                    try:
                        frame = preview_env.instance._get_observation().get("screen")
                    except Exception:
                        frame = None

                if frame is None:
                    frame = np.zeros((render_height, render_width, 3), dtype=np.uint8)

                if debug_overlay:
                    try:
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (0,0), (render_width, 36), (0,0,0), -1)
                        alpha = 0.5
                        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                        info_text = f"Steps: {steps}  Reward: {episode_reward:.2f}"
                        ppos = info.get('player_position') or info.get('player_position', (0.0, 0.0, 0.0))
                        armor = info.get('player_armor', 0)
                        energy = info.get('player_energy', 0)
                        info_text2 = f"Pos: {ppos[0]:.1f},{ppos[1]:.1f}  Armor: {armor}  Energy: {energy}"
                        cv2.putText(frame, info_text, (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                        cv2.putText(frame, info_text2, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1, cv2.LINE_AA)
                    except Exception:
                        pass

                try:
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                except Exception:
                    bgr = frame

                if writer is not None:
                    try:
                        writer.write(bgr)
                    except Exception:
                        pass
                if window_ok:
                    try:
                        cv2.imshow(window_name, bgr)
                        if cv2.waitKey(frame_delay_ms) & 0xFF == ord('q'):
                            stop_event.set()
                            break
                    except Exception:
                        window_ok = False
                else:
                    time.sleep(frame_delay_ms / 1000.0)

            # small pause between preview episodes
            time.sleep(0.05)

    finally:
        try:
            preview_env.close()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("[renderer] Disk-based renderer shutdown complete.")


def main():
    parser = argparse.ArgumentParser(description="Train a PPO agent on Vangers while rendering current in-memory policy")
    parser.add_argument("--timesteps", type=int, default=50000, help="Total training timesteps")
    parser.add_argument("--save-interval", type=int, default=5000, help="Save checkpoint every N timesteps")
    parser.add_argument("--width", type=int, default=160, help="Training environment width")
    parser.add_argument("--height", type=int, default=120, help="Training environment height")
    parser.add_argument("--render-width", type=int, default=320, help="Preview window width")
    parser.add_argument("--render-height", type=int, default=240, help="Preview window height")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory to store checkpoints (default: temp dir)")
    parser.add_argument("--play-speed", type=float, default=20.0, help="Preview engine time scale (20.0 => 20x)")
    parser.add_argument("--preview-fps", type=int, default=20, help="FPS for preview window updates")
    parser.add_argument("--no-render", action="store_true", help="Disable preview renderer")
    parser.add_argument("--in-memory-render", action="store_true", default=True, help="Use in-memory renderer (default: enabled)")
    parser.add_argument("--debug-overlay", action="store_true", default=True, help="Show debug overlay in preview")
    # Allow overriding PPO's rollout length for fast local tests (smaller -> more frequent updates)
    parser.add_argument("--ppo-n-steps", type=int, default=2048, help="PPO n_steps (rollout length). Use small values like 64 for quick tests")
    parser.add_argument("--timeout-seconds", type=int, default=None, help="Stop training after N seconds (wall-clock timeout)")
    parser.add_argument("--record-preview", type=str, default=None, help="Write preview recording to this MP4 file (headless-friendly)")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto", help="Training device selection")
    parser.add_argument("--train-subprocess", action="store_true", help="Run training in a subprocess and render from disk checkpoints")
    args = parser.parse_args()

    # Dependencies are assumed to be present in the configured virtualenv (~/Venv/venv).
    # Fail fast if something is missing — let the import error explain what is absent.

    # Prepare checkpoint directory (we still save periodically but renderer uses in-memory model)
    if args.checkpoint_dir:
        ckpt_dir = Path(args.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    else:
        ckpt_dir = Path(tempfile.mkdtemp(prefix="vangers_ckpt_"))

    print(f"Checkpoints will be saved to: {ckpt_dir}")

    # Verify the expected resource path exists (the engine expects game data here)
    resource_path = Path("/x/Work/VangersData")
    if not resource_path.exists():
        print("Warning: expected Vangers resource path not found at /x/Work/VangersData.")
        print("The engine will likely fail to initialize without the full game data.")
        print("If your data is located elsewhere, either create a symlink at /x/Work/VangersData")
        print("or set the environment variable VANGERS_RESOURCE_PATH and modify the loader accordingly.")
    else:
        print("Found Vangers resource path at /x/Work/VangersData")

    # If the user hasn't set VANGERS_ENGINE_LIB, attempt to auto-discover a built shared library
    if not os.environ.get("VANGERS_ENGINE_LIB"):
        candidate_paths = []
        repo_root = Path(__file__).resolve().parents[2]
        candidate_paths.extend([
            repo_root / "gym_wrapper" / "libvangers_engine.so",
            repo_root / "build" / "libvangers_engine.so",
            repo_root / "build" / "vangers_engine" / "libvangers_engine.so",
            repo_root / "lib" / "libvangers_engine.so",
            Path("/usr/local/lib/libvangers_engine.so"),
            Path("/usr/lib/libvangers_engine.so"),
        ])

        found_lib = None
        for p in candidate_paths:
            try:
                if p.exists():
                    found_lib = p
                    break
            except Exception:
                continue

        if found_lib:
            os.environ["VANGERS_ENGINE_LIB"] = str(found_lib)
            print(f"Auto-detected vangers engine shared library at: {found_lib}")
            print("Setting VANGERS_ENGINE_LIB so the Python loader will use the built shared library.")
        else:
            print("VANGERS_ENGINE_LIB not set and no built shared library was found in common locations.")
            print("If you haven't built the shared engine, build it with:")
            print("  mkdir build && cd build && cmake -DBUILD_SHARED_ENGINE=ON .. && make -j")
            print("After building, either set VANGERS_ENGINE_LIB to the built library path,")
            print("or copy the built shared library into one of the locations printed above.")

    # Create the training environment and pass engine lib override (if any)
    print("Creating training environment...")
    train_env = make_vec_env_from_vangers(args.width, args.height, engine_lib_path=os.environ.get("VANGERS_ENGINE_LIB"))

    # Build the PPO model with custom extractor (CNN for screen + MLP for state)
    print("Creating PPO model with combined CNN+MLP feature extractor...")
    policy_kwargs = {
        "features_extractor_class": CombinedExtractor,
        "features_extractor_kwargs": {"cnn_output_dim": 128}
    }
    # Pass through n_steps from CLI so tests can use short rollouts (faster feedback)
    device = ("cuda" if (args.device == "cuda" or (args.device == "auto" and th.cuda.is_available())) else "cpu")
    model = PPO(policy="MultiInputPolicy", env=train_env, verbose=1, policy_kwargs=policy_kwargs, n_steps=args.ppo_n_steps, device=device)

    # Setup callback to periodically save the model
    save_callback = PeriodicSaveCallback(str(ckpt_dir), save_freq=args.save_interval, verbose=1, timeout_seconds=args.timeout_seconds)

    # Shared model container and lock for in-memory rendering
    shared_model = {"model": model, "snapshot": None, "stats": {}}
    model_lock = threading.Lock()

    # Snapshotter: periodically creates a light-weight copy of the model for non-blocking inference
    def snapshotter_loop(shared_model: dict, model_lock: threading.Lock, stop_evt: threading.Event, interval: float = 1.0):
        from copy import deepcopy as _deepcopy
        while not stop_evt.is_set():
            try:
                with model_lock:
                    live_model = shared_model.get("model")
                    if live_model is not None:
                        try:
                            # Deepcopy the model for inference use in renderer (best-effort).
                            snapshot = _deepcopy(live_model)
                            try:
                                # Ensure CPU snapshot for cross-device safety in renderer
                                if hasattr(snapshot, "policy") and hasattr(snapshot.policy, "to"):
                                    snapshot.policy.to("cpu")
                            except Exception:
                                pass
                            shared_model["snapshot"] = snapshot
                        except Exception:
                            # Fallback: try saving/loading via temporary file if deepcopy fails
                            try:
                                tmpf = Path(tempfile.mkdtemp()) / "tmp_model.zip"
                                live_model.save(str(tmpf))
                                from stable_baselines3 import PPO as _PPO
                                snapshot = _PPO.load(str(tmpf), device="cpu")
                                shared_model["snapshot"] = snapshot
                            except Exception:
                                pass
            except Exception:
                pass
            time.sleep(interval)

    # Start renderer thread (optional)
    stop_event = threading.Event()
    renderer_thread = None
    snapshot_thread = None

    # Hard wall-clock timeout (enforced via SIGINT) to prevent hangs
    if args.timeout_seconds is not None and args.timeout_seconds > 0:
        def _hard_timeout():
            try:
                time.sleep(float(args.timeout_seconds))
                print("[timeout] Hard timeout reached; stopping training and renderer.")
                stop_event.set()
                try:
                    os.kill(os.getpid(), signal.SIGINT)
                except Exception:
                    pass
            except Exception:
                pass
        threading.Thread(target=_hard_timeout, daemon=True).start()

    if not args.no_render and cv2 is not None and args.in_memory_render and not args.train_subprocess:
        # start snapshotter so renderer uses snapshot copies (reduces contention)
        snapshot_thread = threading.Thread(target=snapshotter_loop, args=(shared_model, model_lock, stop_event), daemon=True)
        snapshot_thread.start()

        renderer_thread = threading.Thread(
            target=renderer_in_memory_loop,
            args=(
                args.render_width,
                args.render_height,
                shared_model,
                model_lock,
                stop_event,
            ),
            kwargs={'play_speed': args.play_speed, 'preview_fps': args.preview_fps, 'debug_overlay': args.debug_overlay, 'train_input_size': (args.width, args.height), 'record_path': args.record_preview},
            daemon=True,
        )
        renderer_thread.start()
    elif not args.no_render and cv2 is not None and not args.in_memory_render:
        # Fallback to disk-based renderer: keep compatibility but still run snapshotter
        snapshot_thread = threading.Thread(target=snapshotter_loop, args=(shared_model, model_lock, stop_event), daemon=True)
        snapshot_thread.start()
        renderer_thread = threading.Thread(
            target=renderer_loop,
            args=(args.render_width, args.render_height, str(ckpt_dir), stop_event, args.play_speed, args.preview_fps),
            kwargs={'debug_overlay': args.debug_overlay, 'train_input_size': (args.width, args.height), 'record_path': args.record_preview},
            daemon=True,
        )
        renderer_thread.start()

    # If training in subprocess, launch a child process that runs training only and render from disk checkpoints here
    if args.train_subprocess:
        child_cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--timesteps", str(args.timesteps),
            "--width", str(args.width),
            "--height", str(args.height),
            "--checkpoint-dir", str(ckpt_dir),
            "--ppo-n-steps", str(args.ppo_n_steps),
            "--device", args.device,
            "--no-render",
        ]
        print(f"[subprocess] launching training: {' '.join(child_cmd)}")
        # Start disk-based renderer so the preview is visible while the child process trains
        if not args.no_render and cv2 is not None:
            renderer_thread = threading.Thread(
                target=renderer_loop,
                args=(args.render_width, args.render_height, str(ckpt_dir), stop_event, args.play_speed, args.preview_fps),
                kwargs={'debug_overlay': args.debug_overlay, 'train_input_size': (args.width, args.height), 'record_path': args.record_preview},
                daemon=True,
            )
            renderer_thread.start()
        proc = subprocess.Popen(child_cmd, env=os.environ.copy(), preexec_fn=os.setsid)
        try:
            if args.timeout_seconds:
                try:
                    proc.wait(timeout=float(args.timeout_seconds))
                except subprocess.TimeoutExpired:
                    print("[timeout] Killing training subprocess due to timeout")
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                        time.sleep(1.0)
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except Exception:
                        proc.kill()
            else:
                proc.wait()
        finally:
            # stop renderer and snapshotter if any
            stop_event.set()
            if renderer_thread is not None:
                try:
                    renderer_thread.join(timeout=5.0)
                except Exception:
                    pass
            if snapshot_thread is not None:
                try:
                    snapshot_thread.join(timeout=5.0)
                except Exception:
                    pass
            try:
                train_env.close()
            except Exception:
                pass
            print("Done.")
            if args.checkpoint_dir is None:
                print(f"Temporary checkpoint directory retained at: {ckpt_dir}")
            return
    # Train the model in a separate thread so we can enforce a hard timeout
    train_done = threading.Event()
    train_exc = []

    def _train():
        try:
            print(f"Starting training for {args.timesteps} timesteps...")
            start = time.time()
            model.learn(total_timesteps=args.timesteps, callback=save_callback)
            duration = time.time() - start
            print(f"Training completed in {duration:.1f}s")
        except KeyboardInterrupt:
            print("Training interrupted by user.")
        except Exception as e:
            print(f"[train] Exception during training: {e}")
            train_exc.append(e)
        finally:
            train_done.set()

    t = threading.Thread(target=_train, daemon=True)
    t.start()

    # If a timeout was requested, wait for it and then forcefully stop if needed
    if args.timeout_seconds is not None and args.timeout_seconds > 0:
        t.join(float(args.timeout_seconds))
        if not train_done.is_set():
            print("[timeout] Timeout reached; attempting graceful shutdown...")
            stop_event.set()
            try:
                train_env.close()
            except Exception:
                pass
            if renderer_thread is not None:
                try:
                    renderer_thread.join(timeout=2.0)
                except Exception:
                    pass
            if snapshot_thread is not None:
                try:
                    snapshot_thread.join(timeout=2.0)
                except Exception:
                    pass
            print("[timeout] Exiting process to avoid hang.")
            sys.exit(130)
    else:
        t.join()

    # Save final model (best effort)
    try:
        final_path = ckpt_dir / "model_final.zip"
        model.save(str(final_path))
        latest = ckpt_dir / "model_latest.zip"
        try:
            if latest.exists():
                latest.unlink()
            latest.symlink_to(final_path.name)
        except Exception:
            shutil.copyfile(str(final_path), str(ckpt_dir / "model_latest.zip"))
        print(f"Final model saved to: {final_path}")
    except Exception as e:
        print(f"Failed to save final model: {e}")

    # Signal renderer and snapshotter to stop and wait for threads
    if renderer_thread is not None or snapshot_thread is not None:
        print("[main] Signaling renderer and snapshotter to shut down...")
        stop_event.set()
        if renderer_thread is not None:
            try:
                renderer_thread.join(timeout=5.0)
            except Exception:
                pass
        if snapshot_thread is not None:
            try:
                snapshot_thread.join(timeout=5.0)
            except Exception:
                pass

    # Cleanup
    try:
        train_env.close()
    except Exception:
        pass

    print("Done.")

    # If we created a temporary checkpoint directory, leave it for inspection but notify user
    if args.checkpoint_dir is None:
        print(f"Temporary checkpoint directory retained at: {ckpt_dir}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[signal] Interrupted by user (Ctrl+C). Exiting cleanly.")
