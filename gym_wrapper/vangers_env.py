#!/usr/bin/env python3
"""
Gym wrapper for the Vangers C++ game engine.

This version requires the compiled shared library. On initialization the loader
will call `vangers_engine_init_with_path("/x/Work/VangersData")` (preferred) or
fall back to `vangers_engine_init()` if the former symbol is not present.

If the engine shared library cannot be found the loader raises FileNotFoundError.
To run without building the engine you must build the shared library and either:

 - set environment variable VANGERS_ENGINE_LIB to the full path of the built library
 - set VANGERS_ENGINE_PATH to a directory containing `libvangers_engine.*`
 - place the built library into one of the standard locations probed by this loader.

The rest of this module exposes:
 - ctypes mappings for the C API structures
 - `VangersEngineLib` wrapper that loads the shared object and calls init with the
   canonical resource path "/x/Work/VangersData"
 - `VangersInstance` wrapper for a single deterministic instance
 - `VangersVectorizedEnv` and `VangersEnv` gym wrappers
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Union

import ctypes
import numpy as np

import gymnasium as gym  # type: ignore
from gymnasium import spaces  # type: ignore
_EnvBase = gym.Env




# ---------------------------------------------------------------------------
# ctypes structure definitions
# ---------------------------------------------------------------------------

class Vector3(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float)
    ]


class PlayerState(ctypes.Structure):
    _fields_ = [
        ("position", Vector3),
        ("velocity", Vector3),
        ("angle", ctypes.c_float),
        ("pitch", ctypes.c_float),
        ("roll", ctypes.c_float),
        ("armor", ctypes.c_int),
        ("energy", ctypes.c_int),
        ("speed", ctypes.c_int),
        ("max_speed", ctypes.c_int),
        ("volume", ctypes.c_int),
        ("max_volume", ctypes.c_int),
        ("money", ctypes.c_int),
        ("alive", ctypes.c_bool),
        ("on_ground", ctypes.c_bool),
        ("in_water", ctypes.c_bool),
    ]


class GameEvent(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("value", ctypes.c_float),
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("timestamp", ctypes.c_uint64),
    ]


# ---------------------------------------------------------------------------
# Library loader (strict: shared library required)
# ---------------------------------------------------------------------------

class VangersEngineLib:
    """
    Loader for the compiled vangers engine shared library.

    This loader:
      - Discovers the shared library using env vars or common install paths.
      - Loads the library with ctypes.CDLL.
      - Binds the C API function signatures used by the Python wrapper.
      - Calls engine initialization using the fixed resource path:
          "/x/Work/VangersData"

    If the shared library is not found an exception is raised. This module does
    not provide a Python fallback engine.
    """

    DEFAULT_RESOURCE_PATH = "/x/Work/VangersData"

    def __init__(self, lib_path: Optional[str] = None):
        self.lib_path = lib_path or self._find_engine_library()
        # Attempt to load the library
        try:
            self._lib: ctypes.CDLL = ctypes.CDLL(self.lib_path)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load Vangers engine shared library at {self.lib_path}: {e}")
        self._bind_functions()
        self._init_engine()

    def _find_engine_library(self) -> str:
        """
        Discover the engine shared library.

        Order:
          1. VANGERS_ENGINE_LIB (file)
          2. VANGERS_ENGINE_PATH / VANGERS_GYM_LIB (dir or file)
          3. ctypes.util.find_library("vangers_engine")
          4. Several common locations relative to this file and system library dirs
        """
        # 1) environment override file
        env_file = os.environ.get("VANGERS_ENGINE_LIB")
        env_path = os.environ.get("VANGERS_ENGINE_PATH") or os.environ.get("VANGERS_GYM_LIB")

        if env_file:
            p = Path(env_file)
            if p.is_file():
                return str(p)
            raise FileNotFoundError(f"VANGERS_ENGINE_LIB set but not found: {env_file}")

        # If user pointed at a directory or file via env path
        if env_path:
            p = Path(env_path)
            if p.is_file():
                return str(p)
            if p.is_dir():
                for pattern in ("libvangers_engine.*", "vangers_engine.*"):
                    matches = sorted(p.glob(pattern))
                    if matches:
                        return str(matches[0])

        # 2) try ctypes.util.find_library
        try:
            from ctypes.util import find_library
            soname = find_library("vangers_engine")
            if soname:
                return soname
        except Exception:
            pass

        # 3) common candidate directories (prefer repository build locations first)
        # Prefer the repository-level build and lib directories (e.g. Vangers/build)
        # since the CMake build places the shared library there by default.
        repo_root = Path(__file__).resolve().parents[1]
        candidate_dirs = [
            repo_root / "build",
            repo_root / "lib",
            Path(__file__).parent / "build",
            Path(__file__).parent / "lib",
            Path.cwd(),
            Path("/usr/local/lib"),
            Path("/usr/lib"),
        ]

        candidate_basenames = ("libvangers_engine", "vangers_engine")
        exts = (".so", ".so.*") if sys.platform != "darwin" else (".dylib",)

        for d in candidate_dirs:
            if not d:
                continue
            for base in candidate_basenames:
                for ext in exts:
                    if ext.endswith("*"):
                        pattern = f"{base}{ext[:-1]}*"
                        for match in sorted(d.glob(pattern)):
                            return str(match)
                    else:
                        p = d / f"{base}{ext}"
                        if p.exists():
                            return str(p)
            # broad glob
            for pattern in ("libvangers_engine.*", "vangers_engine.*"):
                for match in sorted(d.glob(pattern)):
                    return str(match)

        # 4) fallback hardcoded (also prefer repository build locations)
        possible_paths = [
            str(repo_root / "build" / "libvangers_engine.so"),
            str(repo_root / "lib" / "libvangers_engine.so"),
            "./libvangers_engine.so",
            "./build/libvangers_engine.so",
            "./lib/libvangers_engine.so",
            str(Path(__file__).parent / "libvangers_engine.so"),
            str(Path(__file__).parent / "build" / "libvangers_engine.so"),
            "/usr/local/lib/libvangers_engine.so",
            "/usr/lib/libvangers_engine.so",
        ]

        for p in possible_paths:
            if os.path.isfile(p):
                return p

        # Collect diagnostic information about locations and methods we attempted so
        # users get a helpful error message instead of a vague "not found".
        attempted = []
        try:
            if env_file:
                attempted.append(f"VANGERS_ENGINE_LIB={env_file}")
        except Exception:
            pass
        try:
            if env_path:
                attempted.append(f"VANGERS_ENGINE_PATH/VANGERS_GYM_LIB={env_path}")
        except Exception:
            pass

        try:
            from ctypes.util import find_library
            soname = find_library("vangers_engine")
            if soname:
                attempted.append(f"ctypes.util.find_library -> {soname}")
        except Exception:
            soname = None

        # Candidate directories we probed
        try:
            for d in candidate_dirs:
                attempted.append(str(d))
        except Exception:
            pass

        # Candidate file paths we checked
        try:
            for p in possible_paths:
                attempted.append(str(p))
        except Exception:
            pass

        # Remove empties and deduplicate while preserving order
        seen = set()
        cleaned = []
        for item in attempted:
            if not item:
                continue
            if item in seen:
                continue
            seen.add(item)
            cleaned.append(item)

        tried_list = "\n  - ".join(cleaned) if cleaned else " (no locations were recorded)"

        raise FileNotFoundError(
            "Vangers engine library not found. The loader attempted the following locations/methods:\n\n"
            f"  - {tried_list}\n\n"
            "Please build the shared library and either:\n"
            "  - set VANGERS_ENGINE_LIB to the full path of the built library, or\n"
            "  - place the built shared library into one of the locations above, or\n"
            "  - add the directory containing the library to VANGERS_ENGINE_PATH / VANGERS_GYM_LIB.\n"
            "Build instructions (example):\n"
            "  mkdir build && cd build && cmake -DBUILD_SHARED_ENGINE=ON .. && make -j\n"
        )

    def _bind_functions(self) -> None:
        """Bind the C API functions we use and set arg/return types."""
        lib = self._lib

        # Initialization
        if hasattr(lib, "vangers_engine_init"):
            lib.vangers_engine_init.restype = ctypes.c_int
            lib.vangers_engine_init.argtypes = []
        if hasattr(lib, "vangers_engine_init_with_path"):
            lib.vangers_engine_init_with_path.restype = ctypes.c_int
            lib.vangers_engine_init_with_path.argtypes = [ctypes.c_char_p]

        if hasattr(lib, "vangers_engine_cleanup"):
            lib.vangers_engine_cleanup.restype = None
            lib.vangers_engine_cleanup.argtypes = []

        # Version
        if hasattr(lib, "vangers_get_version"):
            lib.vangers_get_version.restype = ctypes.c_char_p
            lib.vangers_get_version.argtypes = []

        # Instance management
        lib.vangers_create_instance.restype = ctypes.c_void_p
        lib.vangers_create_instance.argtypes = [ctypes.c_int, ctypes.c_int]

        lib.vangers_destroy_instance.restype = None
        lib.vangers_destroy_instance.argtypes = [ctypes.c_void_p]

        lib.vangers_reset_instance.restype = ctypes.c_int
        lib.vangers_reset_instance.argtypes = [ctypes.c_void_p]

        # Simulation control
        lib.vangers_step_simulation.restype = ctypes.c_int
        lib.vangers_step_simulation.argtypes = [ctypes.c_void_p, ctypes.c_int]

        # Optional time scaling / pause
        if hasattr(lib, "vangers_set_time_scale"):
            lib.vangers_set_time_scale.restype = None
            lib.vangers_set_time_scale.argtypes = [ctypes.c_void_p, ctypes.c_float]

        if hasattr(lib, "vangers_pause_simulation"):
            lib.vangers_pause_simulation.restype = None
            lib.vangers_pause_simulation.argtypes = [ctypes.c_void_p, ctypes.c_bool]

        # Action interface
        lib.vangers_set_action.restype = None
        lib.vangers_set_action.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]

        # State query
        lib.vangers_get_player_state.restype = ctypes.c_int
        lib.vangers_get_player_state.argtypes = [ctypes.c_void_p, ctypes.POINTER(PlayerState)]

        lib.vangers_get_frame_buffer.restype = ctypes.c_int
        lib.vangers_get_frame_buffer.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int]

        # Events
        if hasattr(lib, "vangers_get_events"):
            lib.vangers_get_events.restype = ctypes.c_int
            lib.vangers_get_events.argtypes = [ctypes.c_void_p, ctypes.POINTER(GameEvent), ctypes.c_int]

        if hasattr(lib, "vangers_clear_events"):
            lib.vangers_clear_events.restype = None
            lib.vangers_clear_events.argtypes = [ctypes.c_void_p]

        # Environment configuration
        if hasattr(lib, "vangers_set_map"):
            lib.vangers_set_map.restype = None
            lib.vangers_set_map.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        if hasattr(lib, "vangers_set_render_mode"):
            lib.vangers_set_render_mode.restype = None
            lib.vangers_set_render_mode.argtypes = [ctypes.c_void_p, ctypes.c_int]
        if hasattr(lib, "vangers_set_physics_substeps"):
            lib.vangers_set_physics_substeps.restype = None
            lib.vangers_set_physics_substeps.argtypes = [ctypes.c_void_p, ctypes.c_int]

    def _init_engine(self) -> None:
        """
        Initialize the engine using the canonical resource path.
        Prefer `vangers_engine_init_with_path` when available so the engine
        can find its data at /x/Work/VangersData.
        """
        lib = self._lib
        resource_path = self.DEFAULT_RESOURCE_PATH.encode("utf-8")

        if hasattr(lib, "vangers_engine_init_with_path"):
            res = lib.vangers_engine_init_with_path(ctypes.c_char_p(resource_path))
            if int(res) != 1:
                raise RuntimeError(f"vangers_engine_init_with_path failed with code {res}")
        elif hasattr(lib, "vangers_engine_init"):
            res = lib.vangers_engine_init()
            if int(res) != 1:
                raise RuntimeError(f"vangers_engine_init failed with code {res}")
        else:
            raise RuntimeError("Loaded library does not expose an initialization function")

    # Thin wrapper methods
    def create_instance(self, width: int, height: int):
        return self._lib.vangers_create_instance(int(width), int(height))

    def destroy_instance(self, instance_ptr):
        if instance_ptr is None:
            return
        try:
            self._lib.vangers_destroy_instance(instance_ptr)
        except Exception:
            pass

    def reset_instance(self, instance_ptr) -> int:
        return int(self._lib.vangers_reset_instance(instance_ptr))

    def step_simulation(self, instance_ptr, num_steps: int) -> int:
        return int(self._lib.vangers_step_simulation(instance_ptr, int(num_steps)))

    def set_time_scale(self, instance_ptr, scale: float):
        if hasattr(self._lib, "vangers_set_time_scale"):
            try:
                self._lib.vangers_set_time_scale(instance_ptr, ctypes.c_float(scale))
            except Exception:
                pass

    def pause_simulation(self, instance_ptr, paused: bool):
        if hasattr(self._lib, "vangers_pause_simulation"):
            try:
                self._lib.vangers_pause_simulation(instance_ptr, ctypes.c_bool(paused))
            except Exception:
                pass

    def set_action(self, instance_ptr, movement: int, steering: int, fire: int, special1: int, special2: int):
        self._lib.vangers_set_action(instance_ptr, int(movement), int(steering), int(fire), int(special1), int(special2))

    def get_player_state(self, instance_ptr, state_ptr) -> int:
        return int(self._lib.vangers_get_player_state(instance_ptr, state_ptr))

    def get_frame_buffer(self, instance_ptr, buffer_ptr, buffer_size: int) -> int:
        return int(self._lib.vangers_get_frame_buffer(instance_ptr, buffer_ptr, int(buffer_size)))

    def get_events(self, instance_ptr, events_buffer, max_events: int) -> int:
        if hasattr(self._lib, "vangers_get_events"):
            return int(self._lib.vangers_get_events(instance_ptr, events_buffer, int(max_events)))
        return 0

    def clear_events(self, instance_ptr):
        if hasattr(self._lib, "vangers_clear_events"):
            try:
                self._lib.vangers_clear_events(instance_ptr)
            except Exception:
                pass

    def set_map(self, instance_ptr, map_name: str):
        if hasattr(self._lib, "vangers_set_map"):
            try:
                self._lib.vangers_set_map(instance_ptr, ctypes.c_char_p(map_name.encode('utf-8')))
            except Exception:
                pass

    def set_render_mode(self, instance_ptr, mode: int):
        if hasattr(self._lib, "vangers_set_render_mode"):
            try:
                self._lib.vangers_set_render_mode(instance_ptr, int(mode))
            except Exception:
                pass

    def set_physics_substeps(self, instance_ptr, substeps: int):
        if hasattr(self._lib, "vangers_set_physics_substeps"):
            try:
                self._lib.vangers_set_physics_substeps(instance_ptr, int(substeps))
            except Exception:
                pass

    def get_version(self) -> str:
        if hasattr(self._lib, "vangers_get_version"):
            try:
                v = self._lib.vangers_get_version()
                if v:
                    return v.decode('utf-8') if isinstance(v, (bytes, bytearray)) else str(v)
            except Exception:
                pass
        return "unknown"

    def cleanup(self):
        if hasattr(self._lib, "vangers_engine_cleanup"):
            try:
                self._lib.vangers_engine_cleanup()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# VangersInstance: a single deterministic game instance
# ---------------------------------------------------------------------------

class VangersInstance:
    """
    Wraps a single engine instance.
    """

    def __init__(self, engine_lib: VangersEngineLib, width: int = 640, height: int = 480):
        self.engine_lib = engine_lib
        self.width = int(width)
        self.height = int(height)

        # Create instance in engine
        self.instance_ptr = self.engine_lib.create_instance(self.width, self.height)
        if not self.instance_ptr:
            raise RuntimeError("Failed to create Vangers engine instance")

        # Buffers
        self.player_state = PlayerState()
        self.frame_buffer = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.events_buffer = (GameEvent * 64)()

        # Episode
        self.step_count = 0
        self.total_reward = 0.0
        self.done = False

    def __del__(self):
        try:
            if getattr(self, "instance_ptr", None):
                self.engine_lib.destroy_instance(self.instance_ptr)
        except Exception:
            pass

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        res = self.engine_lib.reset_instance(self.instance_ptr)
        if res != 1:
            # Attempt to recover by recreating the engine instance
            try:
                if getattr(self, "instance_ptr", None):
                    self.engine_lib.destroy_instance(self.instance_ptr)
            except Exception:
                pass
            # Recreate instance
            new_ptr = self.engine_lib.create_instance(self.width, self.height)
            if not new_ptr:
                raise RuntimeError("Failed to recreate Vangers engine instance during reset")
            self.instance_ptr = new_ptr
            # Try resetting again
            res2 = self.engine_lib.reset_instance(self.instance_ptr)
            if res2 != 1:
                raise RuntimeError(f"Failed to reset instance after recreation: {res2}")

        self.step_count = 0
        self.total_reward = 0.0
        self.done = False

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray, num_substeps: int = 1) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("Episode terminated; call reset() before stepping")

        # Ensure action length
        if isinstance(action, (list, tuple)):
            action = np.array(action)
        if action.shape[0] < 5:
            raise ValueError("Action must have length >= 5")

        # send action
        self.engine_lib.set_action(self.instance_ptr,
                                   int(action[0]), int(action[1]), int(action[2]), int(action[3]), int(action[4]))

        total_reward = 0.0
        for _ in range(num_substeps):
            ok = self.engine_lib.step_simulation(self.instance_ptr, 1)
            if ok != 1:
                # treat as termination
                self.done = True
                break

            # Update player state (direct wrapper call)
            self.engine_lib.get_player_state(self.instance_ptr, ctypes.byref(self.player_state))

            # Process events for reward
            total_reward += self._process_events()

        self.step_count += 1
        self.total_reward += total_reward

        observation = self._get_observation()

        terminated = self.done or not bool(self.player_state.alive)
        truncated = self.step_count >= 10000

        info = self._get_info()
        info['events'] = self._get_recent_events()

        return observation, total_reward, terminated, truncated, info

    def _get_observation(self) -> Dict[str, np.ndarray]:
        # Refresh state
        try:
            self.engine_lib.get_player_state(self.instance_ptr, ctypes.byref(self.player_state))
        except Exception:
            pass

        # Frame buffer
        try:
            ptr = self.frame_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
            buffer_size = int(self.frame_buffer.size)
            self.engine_lib.get_frame_buffer(self.instance_ptr, ptr, buffer_size)
        except Exception:
            pass

        # compact state vector
        st = np.array([
            self.player_state.position.x / 1000.0,
            self.player_state.position.y / 1000.0,
            self.player_state.position.z / 1000.0,
            self.player_state.velocity.x / 100.0,
            self.player_state.velocity.y / 100.0,
            self.player_state.velocity.z / 100.0,
            (self.player_state.angle / (2 * np.pi)) if self.player_state.angle is not None else 0.0,
            (self.player_state.pitch / (np.pi / 2)) if self.player_state.pitch is not None else 0.0,
            (self.player_state.roll / (np.pi / 2)) if self.player_state.roll is not None else 0.0,
            self.player_state.armor / 1000.0,
            self.player_state.energy / 1000.0,
            self.player_state.speed / 200.0,
            self.player_state.max_speed / 200.0,
            self.player_state.volume / 100.0,
            self.player_state.max_volume / 100.0,
            self.player_state.money / 10000.0,
            float(self.player_state.alive),
            float(self.player_state.on_ground),
            float(self.player_state.in_water),
            float(self.step_count) / 10000.0,
        ], dtype=np.float32)

        return {
            "screen": self.frame_buffer.copy(),
            "state": st
        }

    def _process_events(self) -> float:
        reward = 0.0

        # If engine provides events
        try:
            num = self.engine_lib.get_events(self.instance_ptr, self.events_buffer, len(self.events_buffer))
        except Exception:
            num = 0

        if num <= 0:
            # small alive + movement bonus
            reward += 0.1
            reward += float(self.player_state.speed) * 0.001
            return reward

        for i in range(num):
            ev = self.events_buffer[i]
            if ev.type == 0:
                reward -= 5.0
            elif ev.type == 1:
                reward += 10.0
            elif ev.type == 2:
                reward += 100.0
                self.done = True
            elif ev.type == 3:
                reward -= float(ev.value) * 0.01

        try:
            self.engine_lib.clear_events(self.instance_ptr)
        except Exception:
            pass

        reward += 0.1
        reward += float(self.player_state.speed) * 0.001
        return reward

    def _get_info(self) -> Dict[str, Any]:
        return {
            "step_count": int(self.step_count),
            "total_reward": float(self.total_reward),
            "player_alive": bool(self.player_state.alive),
            "player_position": (
                float(self.player_state.position.x),
                float(self.player_state.position.y),
                float(self.player_state.position.z),
            ),
            "player_armor": int(self.player_state.armor),
            "player_energy": int(self.player_state.energy),
        }

    def _get_recent_events(self) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        try:
            num = self.engine_lib.get_events(self.instance_ptr, self.events_buffer, len(self.events_buffer))
        except Exception:
            num = 0

        for i in range(num):
            ev = self.events_buffer[i]
            events.append({
                "type": int(ev.type),
                "value": float(ev.value),
                "position": (float(ev.x), float(ev.y)),
                "timestamp": int(ev.timestamp),
            })
        return events


# ---------------------------------------------------------------------------
# Vectorized environment and single env wrappers
# ---------------------------------------------------------------------------

class VangersVectorizedEnv(object):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        num_envs: int = 4,
        screen_width: int = 640,
        screen_height: int = 480,
        frame_skip: int = 4,
        max_episode_steps: int = 10000,
        engine_lib_path: Optional[str] = None,
        render_mode: Optional[str] = None,
        reward_scale: float = 1.0,
    ):
        self.num_envs = int(num_envs)
        self.screen_width = int(screen_width)
        self.screen_height = int(screen_height)
        self.frame_skip = int(frame_skip)
        self.max_episode_steps = int(max_episode_steps)
        self.render_mode = render_mode
        self.reward_scale = float(reward_scale)

        self.engine_lib = VangersEngineLib(engine_lib_path)
        self.instances = [
            VangersInstance(self.engine_lib, screen_width, screen_height)
            for _ in range(self.num_envs)
        ]

        if spaces is not None:
            self.action_space = spaces.MultiDiscrete([3, 3, 2, 2, 2])
            self.observation_space = spaces.Dict({
                "screen": spaces.Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8),
                "state": spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
            })
        else:
            self.action_space = None
            self.observation_space = None

        self.episode_rewards = np.zeros(self.num_envs, dtype=float)
        self.episode_lengths = np.zeros(self.num_envs, dtype=int)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            np.random.seed(seed)
        observations = []
        infos = []
        for instance in self.instances:
            obs, info = instance.reset()
            observations.append(obs)
            infos.append(info)
        self.episode_rewards[:] = 0.0
        self.episode_lengths[:] = 0
        return self._stack_observations(observations), infos

    def step(self, actions: Union[np.ndarray, List[np.ndarray]]):
        if isinstance(actions, np.ndarray):
            if actions.ndim == 1:
                actions = [actions.copy() for _ in range(self.num_envs)]
            elif actions.ndim == 2:
                actions = [actions[i] for i in range(self.num_envs)]
            else:
                raise ValueError("Unsupported actions shape")
        elif isinstance(actions, (list, tuple)):
            if len(actions) != self.num_envs:
                raise ValueError("Number of actions must equal num_envs")
        else:
            raise ValueError("Unsupported action type")

        observations = []
        rewards = []
        terminations = []
        truncations = []
        infos = []

        for i, (instance, action) in enumerate(zip(self.instances, actions)):
            obs, reward, terminated, truncated, info = instance.step(action, self.frame_skip)
            observations.append(obs)
            rewards.append(reward * self.reward_scale)
            terminations.append(bool(terminated))
            truncations.append(bool(truncated))
            infos.append(info)

            self.episode_rewards[i] += reward
            self.episode_lengths[i] += 1

            if terminated or truncated:
                info["episode"] = {
                    "r": float(self.episode_rewards[i]),
                    "l": int(self.episode_lengths[i])
                }
                reset_obs, reset_info = instance.reset()
                observations[i] = reset_obs
                infos[i].update(reset_info)
                self.episode_rewards[i] = 0.0
                self.episode_lengths[i] = 0

        return self._stack_observations(observations), np.array(rewards, dtype=float), np.array(terminations, dtype=bool), np.array(truncations, dtype=bool), infos

    def _stack_observations(self, observations: List[Dict[str, np.ndarray]]):
        screens = np.stack([obs["screen"] for obs in observations])
        states = np.stack([obs["state"] for obs in observations])
        return {"screen": screens, "state": states}

    def render(self, mode: Optional[str] = None):
        rm = mode or self.render_mode
        if rm == "rgb_array":
            return self.instances[0].frame_buffer
        elif rm == "human":
            try:
                import cv2
                cv2.imshow("Vangers Vectorized Env", self.instances[0].frame_buffer)
                cv2.waitKey(1)
            except Exception:
                print("OpenCV not available for rendering")
        return None

    def close(self):
        for instance in self.instances:
            try:
                if hasattr(instance, "instance_ptr") and instance.instance_ptr:
                    self.engine_lib.destroy_instance(instance.instance_ptr)
                    instance.instance_ptr = None
            except Exception:
                pass
        # Do not call global engine cleanup here to avoid affecting other in-process environments.

    def get_episode_rewards(self):
        return self.episode_rewards.copy()

    def get_episode_lengths(self):
        return self.episode_lengths.copy()

    def set_time_scale(self, scale: float):
        for instance in self.instances:
            try:
                self.engine_lib.set_time_scale(instance.instance_ptr, float(scale))
            except Exception:
                pass


class VangersEnv(_EnvBase):
    def __init__(self, width: int = 640, height: int = 480, render_mode: Optional[str] = None, engine_lib_path: Optional[str] = None):
        self.width = int(width)
        self.height = int(height)
        self.render_mode = render_mode

        self.engine_lib = VangersEngineLib(engine_lib_path)
        self.instance = VangersInstance(self.engine_lib, width, height)

        if spaces is not None:
            self.action_space = spaces.MultiDiscrete([3, 3, 2, 2, 2])
            self.observation_space = spaces.Dict({
                "screen": spaces.Box(low=0, high=255, shape=(height, width, 3), dtype=np.uint8),
                "state": spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
            })
        else:
            self.action_space = None
            self.observation_space = None

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            np.random.seed(seed)
        obs, info = self.instance.reset()
        return obs, info

    def step(self, action):
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        return self.instance.step(action)

    def render(self, mode: Optional[str] = None):
        rm = mode or self.render_mode
        if rm == "rgb_array":
            obs = self.instance._get_observation()
            return obs.get("screen")
        elif rm == "human":
            try:
                import cv2
                cv2.imshow("VangersEnv", self.instance.frame_buffer)
                cv2.waitKey(1)
            except Exception:
                return self.instance.frame_buffer
        return None

    def close(self):
        try:
            if hasattr(self, "instance"):
                del self.instance
        except Exception:
            pass
        # Avoid calling engine_lib.cleanup() here; it is a process-global teardown and may break other envs.
        try:
            if hasattr(self, "engine_lib") and self.engine_lib:
                del self.engine_lib
        except Exception:
            pass

    def __del__(self):
        self.close()


def make_vangers_env(**kwargs) -> VangersVectorizedEnv:
    kwargs = dict(kwargs)
    kwargs["num_envs"] = 1
    return VangersVectorizedEnv(**kwargs)


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing Vangers Vectorized Environment (real engine).")
    try:
        env = VangersVectorizedEnv(num_envs=1, screen_width=320, screen_height=240, engine_lib_path=None)
        obs, infos = env.reset()
        print(f"Observation shapes: screen={obs['screen'].shape}, state={obs['state'].shape}")
        print("Engine version (if available):", env.engine_lib.get_version())

        for step in range(5):
            # sample random action
            action_space = getattr(env, "action_space", None)
            if action_space is not None and hasattr(action_space, "sample"):
                actions = [action_space.sample() for _ in range(env.num_envs)]
            else:
                actions = [np.zeros(5, dtype=int) for _ in range(env.num_envs)]
            obs, rewards, terms, truncs, infos = env.step(actions)
            print(f"Step {step}: rewards={rewards}, terminations={terms}, truncations={truncs}")
        env.close()
        print("Test completed successfully.")
    except Exception as e:
        print("Test failed:", e)
        print("Ensure the engine shared library is built and available and that the resource path")
        print(f"is present at: {VangersEngineLib.DEFAULT_RESOURCE_PATH}")
