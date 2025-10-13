#!/usr/bin/env python3
"""
Gym wrapper for the Vangers C++ game engine.

This file exposes:
- ctypes structure definitions that mirror the C API.
- `VangersEngineLib` — a thin ctypes loader for the shared library.
- `VangersInstance` — a single deterministic game instance wrapper.
- `VangersVectorizedEnv` — a simple vectorized gymnasium-compatible environment.
- `VangersEnv` — a convenience single-environment wrapper.

Note: The ctypes structures are declared before the library loader so they can
be referenced when defining C function arg/return types.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Union

import ctypes
import glob
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:
    # The code will still work with a fallback defined in VangersEnv.__init__
    gym = None
    spaces = None


# ---------------------------------------------------------------------------
# ctypes structure definitions (must come before the library loader)
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
        ("type", ctypes.c_int),      # 0=collision, 1=item_collected, 2=objective_complete, etc.
        ("value", ctypes.c_float),   # Event-specific value (damage, reward, etc.)
        ("x", ctypes.c_float),       # Event position x
        ("y", ctypes.c_float),       # Event position y
        ("timestamp", ctypes.c_uint64),
    ]


# ---------------------------------------------------------------------------
# Library loader
# ---------------------------------------------------------------------------

class VangersEngineLib:
    """
    Direct interface to Vangers game engine compiled as a shared library.
    Provides deterministic, in-process execution with full time control.
    """

    def __init__(self, lib_path: Optional[str] = None):
        self.lib_path = lib_path or self._find_engine_library()
        self._lib = None
        self._load_library()

    def _find_engine_library(self) -> str:
        """Find the Vangers engine shared library on disk.

        Discovery order (best-effort):
        1. Environment variables: VANGERS_ENGINE_LIB, VANGERS_ENGINE_PATH, VANGERS_GYM_LIB
           - If a file path is provided, it is used directly.
           - If a directory is provided, it is searched for matching library files.
        2. ctypes.util.find_library("vangers_engine") (platform-aware)
        3. Common build/install locations relative to this python file and cwd.
        4. Fallback list used previously.

        This function supports Linux (.so), macOS (.dylib) and Windows (.dll) names.
        """
        # 1) Environment overrides (allow direct file or directory)
        env_vars = ("VANGERS_ENGINE_LIB", "VANGERS_ENGINE_PATH", "VANGERS_GYM_LIB")
        for ev in env_vars:
            val = os.environ.get(ev)
            if not val:
                continue
            p = Path(val)
            if p.is_file():
                return str(p)
            if p.is_dir():
                # Search for expected library names in the provided directory
                for pattern in ("libvangers_engine.*", "vangers_engine.*"):
                    matches = sorted(p.glob(pattern))
                    if matches:
                        return str(matches[0])

        # 2) Try ctypes.util.find_library (lets the OS linker help)
        try:
            from ctypes.util import find_library
            found = find_library("vangers_engine")
            if found:
                # find_library may return a soname (e.g. libvangers_engine.so.1) or a full path
                return found
        except Exception:
            pass

        # Determine platform-specific extension names
        if sys.platform == "darwin":
            exts = (".dylib",)
        elif sys.platform == "win32":
            exts = (".dll",)
        else:
            exts = (".so", ".so.*")

        # 3) Common search locations (expand relative paths)
        candidate_dirs = [
            Path.cwd(),
            Path(__file__).parent,
            Path(__file__).parent / "build",
            Path(__file__).parent / "build",
            Path(__file__).parent / "lib",
            Path(__file__).parent / ".." / "build" / "gym_wrapper",
            Path("/usr/local/lib"),
            Path("/usr/lib"),
        ]

        candidate_basenames = ("libvangers_engine", "vangers_engine")

        for d in candidate_dirs:
            if not d:
                continue
            for base in candidate_basenames:
                # Try exact names first
                for ext in exts:
                    # handle glob-like ext like ".so.*"
                    if ext.endswith("*"):
                        pattern = f"{base}{ext[:-1]}*"
                        for match in sorted(d.glob(pattern)):
                            return str(match)
                    else:
                        p = d / f"{base}{ext}"
                        if p.exists():
                            return str(p)

            # Broad glob search for any matching files in the directory
            for pattern in ("libvangers_engine.*", "vangers_engine.*"):
                for match in sorted(d.glob(pattern)):
                    return str(match)

        # 4) Fallback to previously used hardcoded list for backward compatibility
        possible_paths = [
            "./libvangers_engine.so",
            "./build/libvangers_engine.so",
            "./lib/libvangers_engine.so",
            "../build/libvangers_engine.so",
            "/usr/local/lib/libvangers_engine.so",
            str(Path(__file__).parent / "libvangers_engine.so"),
            str(Path(__file__).parent / "build" / "libvangers_engine.so"),
        ]

        for path in possible_paths:
            if os.path.isfile(path):
                return path

        # If nothing found, provide actionable error with hints
        raise FileNotFoundError(
            "Vangers engine library not found.\n"
            "Hints:\n"
            " - Build it with: cmake -DBUILD_SHARED_ENGINE=ON && make\n"
            " - Or set environment variable VANGERS_ENGINE_LIB to the full path of the library,\n"
            "   or VANGERS_ENGINE_PATH to a directory containing the built library.\n"
        )

    def _load_library(self):
        """Load the shared library and define the C interface via ctypes."""
        # Load
        self._lib = ctypes.CDLL(self.lib_path)

        # Engine initialization / cleanup
        self._lib.vangers_engine_init.restype = ctypes.c_int
        self._lib.vangers_engine_init.argtypes = []

        self._lib.vangers_engine_cleanup.restype = None
        self._lib.vangers_engine_cleanup.argtypes = []

        # Optional debug/version helpers (if present)
        if hasattr(self._lib, "vangers_get_version"):
            self._lib.vangers_get_version.restype = ctypes.c_char_p
            self._lib.vangers_get_version.argtypes = []

        # Instance management
        self._lib.vangers_create_instance.restype = ctypes.c_void_p
        self._lib.vangers_create_instance.argtypes = [ctypes.c_int, ctypes.c_int]

        self._lib.vangers_destroy_instance.restype = None
        self._lib.vangers_destroy_instance.argtypes = [ctypes.c_void_p]

        self._lib.vangers_reset_instance.restype = ctypes.c_int
        self._lib.vangers_reset_instance.argtypes = [ctypes.c_void_p]

        # Simulation control
        self._lib.vangers_step_simulation.restype = ctypes.c_int
        self._lib.vangers_step_simulation.argtypes = [ctypes.c_void_p, ctypes.c_int]

        # Optional time scaling / pause (guard if not present)
        if hasattr(self._lib, "vangers_set_time_scale"):
            self._lib.vangers_set_time_scale.restype = None
            self._lib.vangers_set_time_scale.argtypes = [ctypes.c_void_p, ctypes.c_float]

        if hasattr(self._lib, "vangers_pause_simulation"):
            self._lib.vangers_pause_simulation.restype = None
            self._lib.vangers_pause_simulation.argtypes = [ctypes.c_void_p, ctypes.c_bool]

        # Action interface
        self._lib.vangers_set_action.restype = None
        self._lib.vangers_set_action.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]

        # State query
        self._lib.vangers_get_player_state.restype = ctypes.c_int
        self._lib.vangers_get_player_state.argtypes = [ctypes.c_void_p, ctypes.POINTER(PlayerState)]

        self._lib.vangers_get_frame_buffer.restype = ctypes.c_int
        self._lib.vangers_get_frame_buffer.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int
        ]

        # Event system
        if hasattr(self._lib, "vangers_get_events"):
            self._lib.vangers_get_events.restype = ctypes.c_int
            self._lib.vangers_get_events.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(GameEvent), ctypes.c_int
            ]

        if hasattr(self._lib, "vangers_clear_events"):
            self._lib.vangers_clear_events.restype = None
            self._lib.vangers_clear_events.argtypes = [ctypes.c_void_p]

        # Environment configuration (optional)
        if hasattr(self._lib, "vangers_set_map"):
            self._lib.vangers_set_map.restype = None
            self._lib.vangers_set_map.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        if hasattr(self._lib, "vangers_set_render_mode"):
            self._lib.vangers_set_render_mode.restype = None
            self._lib.vangers_set_render_mode.argtypes = [ctypes.c_void_p, ctypes.c_int]
        if hasattr(self._lib, "vangers_set_physics_substeps"):
            self._lib.vangers_set_physics_substeps.restype = None
            self._lib.vangers_set_physics_substeps.argtypes = [ctypes.c_void_p, ctypes.c_int]

        # Initialize the engine if the function is available
        if hasattr(self._lib, "vangers_engine_init"):
            result = self._lib.vangers_engine_init()
            if result != 1:
                raise RuntimeError(f"Failed to initialize Vangers engine: {result}")

        # Provide a friendly log message
        print("Vangers engine library loaded from:", self.lib_path)

    # -----------------------------------------------------------------------
    # Thin wrapper methods around the raw ctypes library.
    # These provide a stable API callers can use instead of touching
    # self._lib directly and allow graceful handling when some symbols
    # are omitted in minimal builds.
    # -----------------------------------------------------------------------
    def create_instance(self, width: int, height: int):
        if not hasattr(self._lib, "vangers_create_instance"):
            raise RuntimeError("Library missing vangers_create_instance")
        return self._lib.vangers_create_instance(int(width), int(height))

    def destroy_instance(self, instance_ptr):
        if instance_ptr is None:
            return
        if hasattr(self._lib, "vangers_destroy_instance"):
            try:
                self._lib.vangers_destroy_instance(instance_ptr)
            except Exception:
                pass

    def reset_instance(self, instance_ptr) -> int:
        if not hasattr(self._lib, "vangers_reset_instance"):
            return 1
        return int(self._lib.vangers_reset_instance(instance_ptr))

    def step_simulation(self, instance_ptr, num_steps: int) -> int:
        if not hasattr(self._lib, "vangers_step_simulation"):
            return 0
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
        if hasattr(self._lib, "vangers_set_action"):
            try:
                self._lib.vangers_set_action(instance_ptr, int(movement), int(steering), int(fire), int(special1), int(special2))
            except Exception:
                pass

    def get_player_state(self, instance_ptr, state_ptr) -> int:
        if not hasattr(self._lib, "vangers_get_player_state"):
            return 0
        return int(self._lib.vangers_get_player_state(instance_ptr, state_ptr))

    def get_frame_buffer(self, instance_ptr, buffer_ptr, buffer_size: int) -> int:
        if not hasattr(self._lib, "vangers_get_frame_buffer"):
            return 0
        return int(self._lib.vangers_get_frame_buffer(instance_ptr, buffer_ptr, int(buffer_size)))

    def get_events(self, instance_ptr, events_buffer, max_events: int) -> int:
        if not hasattr(self._lib, "vangers_get_events"):
            return 0
        return int(self._lib.vangers_get_events(instance_ptr, events_buffer, int(max_events)))

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
    Single instance of the Vangers game engine.
    Provides deterministic stepping and direct state access.
    """

    def __init__(self, engine_lib: VangersEngineLib, width: int = 640, height: int = 480):
        self.engine_lib = engine_lib
        self.width = width
        self.height = height

        # Create game instance (use wrapper method on engine_lib)
        self.instance_ptr = engine_lib.create_instance(width, height)
        if not self.instance_ptr:
            raise RuntimeError("Failed to create Vangers instance")

        # State buffers
        self.player_state = PlayerState()
        self.frame_buffer = np.zeros((height, width, 3), dtype=np.uint8)
        self.events_buffer = (GameEvent * 32)()  # Max 32 events per step

        # Episode tracking
        self.step_count = 0
        self.total_reward = 0.0
        self.done = False

    def __del__(self):
        if hasattr(self, 'instance_ptr') and self.instance_ptr and self.engine_lib:
            try:
                self.engine_lib.destroy_instance(self.instance_ptr)
            except Exception:
                # Destructor must not raise
                pass

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the game instance and return initial observation and info."""
        result = self.engine_lib.reset_instance(self.instance_ptr)
        if result != 1:
            raise RuntimeError(f"Failed to reset instance: {result}")

        self.step_count = 0
        self.total_reward = 0.0
        self.done = False

        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def step(self, action: np.ndarray, num_substeps: int = 1) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Step the simulation deterministically.

        Args:
            action: Action array [movement, steering, fire, special1, special2]
            num_substeps: Number of physics substeps to execute
        """
        if self.done:
            raise RuntimeError("Episode is done, call reset() first")

        # Ensure action shape/length
        if isinstance(action, (list, tuple)):
            action = np.array(action)
        if action.shape[0] < 5:
            raise ValueError("Action must have length >= 5")

        # Set action in game engine via wrapper
        self.engine_lib.set_action(
            self.instance_ptr,
            int(action[0]), int(action[1]), int(action[2]), int(action[3]), int(action[4])
        )

        # Step simulation deterministically
        total_reward = 0.0
        for _ in range(num_substeps):
            result = self.engine_lib.step_simulation(self.instance_ptr, 1)
            if result != 1:
                # Treat non-success as done (engine-specific semantics)
                self.done = True
                break

            # Update player_state from engine for reward calculation and termination
            self.engine_lib._lib.vangers_get_player_state(self.instance_ptr, ctypes.byref(self.player_state))

            # Process events and calculate reward
            step_reward = self._process_events()
            total_reward += step_reward

        self.step_count += 1
        self.total_reward += total_reward

        # Get new observation
        observation = self._get_observation()

        # Termination/truncation checks
        terminated = self.done or not bool(self.player_state.alive)
        truncated = self.step_count >= 10000  # hard-coded max-episode length

        info = self._get_info()
        info['events'] = self._get_recent_events()

        return observation, total_reward, terminated, truncated, info

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation from game state (screen + compact state vector)."""
        # Update player state (via wrapper)
        try:
            self.engine_lib.get_player_state(self.instance_ptr, ctypes.byref(self.player_state))
        except Exception:
            # If the wrapper/library fails, leave the player_state as-is
            pass

        # Update frame buffer (if supported)
        frame_ptr = self.frame_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        buffer_size = int(self.frame_buffer.size)
        try:
            # Use wrapper which will call into the library if present
            self.engine_lib.get_frame_buffer(self.instance_ptr, frame_ptr, buffer_size)
        except Exception:
            # If frame buffer function is not available or fails, leave zeros
            pass

        # Create compact state vector
        state_vector = np.array([
            # Position normalized by some heuristic scale
            self.player_state.position.x / 1000.0,
            self.player_state.position.y / 1000.0,
            self.player_state.position.z / 1000.0,
            # Velocity normalized
            self.player_state.velocity.x / 100.0,
            self.player_state.velocity.y / 100.0,
            self.player_state.velocity.z / 100.0,
            # Orientation normalized
            self.player_state.angle / (2 * np.pi) if self.player_state.angle is not None else 0.0,
            self.player_state.pitch / (np.pi / 2) if self.player_state.pitch is not None else 0.0,
            self.player_state.roll / (np.pi / 2) if self.player_state.roll is not None else 0.0,
            # Vehicle stats
            self.player_state.armor / 1000.0,
            self.player_state.energy / 1000.0,
            self.player_state.speed / 200.0,
            self.player_state.max_speed / 200.0,
            self.player_state.volume / 100.0,
            self.player_state.max_volume / 100.0,
            self.player_state.money / 10000.0,
            # Status flags
            float(self.player_state.alive),
            float(self.player_state.on_ground),
            float(self.player_state.in_water),
            # Episode progress
            self.step_count / 10000.0,
        ], dtype=np.float32)

        return {
            'screen': self.frame_buffer.copy(),
            'state': state_vector
        }

    def _process_events(self) -> float:
        """Process game events returned by the engine and compute a reward."""
        reward = 0.0

        # Guard for missing functions
        if not hasattr(self.engine_lib._lib, "vangers_get_events"):
            # Provide a small alive bonus and movement bonus
            reward += 0.1
            reward += float(self.player_state.speed) * 0.001
            return reward

        num_events = self.engine_lib.get_events(
            self.instance_ptr, self.events_buffer, len(self.events_buffer)
        )

        for i in range(num_events):
            event = self.events_buffer[i]
            if event.type == 0:  # Collision
                reward -= 5.0
            elif event.type == 1:  # Item collected
                reward += 10.0
            elif event.type == 2:  # Objective completed
                reward += 100.0
                self.done = True
            elif event.type == 3:  # Damage taken
                reward -= float(event.value) * 0.01

        # Clear events if function available
        # Clear events (wrapper handles missing symbol)
        try:
            self.engine_lib.clear_events(self.instance_ptr)
        except Exception:
            pass

        # Base rewards
        reward += 0.1  # alive bonus
        reward += float(self.player_state.speed) * 0.001  # movement bonus

        return reward

    def _get_info(self) -> Dict[str, Any]:
        """Return additional info about the current instance."""
        return {
            'step_count': int(self.step_count),
            'total_reward': float(self.total_reward),
            'player_alive': bool(self.player_state.alive),
            'player_position': (
                float(self.player_state.position.x),
                float(self.player_state.position.y),
                float(self.player_state.position.z)
            ),
            'player_armor': int(self.player_state.armor),
            'player_energy': int(self.player_state.energy),
        }

    def _get_recent_events(self) -> List[Dict[str, Any]]:
        """Return the most recent events from the engine as Python dicts."""
        events: List[Dict[str, Any]] = []

        if not hasattr(self.engine_lib._lib, "vangers_get_events"):
            return events

        num_events = self.engine_lib.get_events(
            self.instance_ptr, self.events_buffer, len(self.events_buffer)
        )

        for i in range(num_events):
            ev = self.events_buffer[i]
            events.append({
                'type': int(ev.type),
                'value': float(ev.value),
                'position': (float(ev.x), float(ev.y)),
                'timestamp': int(ev.timestamp),
            })

        return events


# ---------------------------------------------------------------------------
# Vectorized Gymnasium-compatible environment
# ---------------------------------------------------------------------------

class VangersVectorizedEnv(object):
    """
    Vectorized Vangers environment that runs multiple game instances in parallel.
    Each instance is deterministic and steps only when explicitly called.
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 20}

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
        super().__init__()

        self.num_envs = int(num_envs)
        self.screen_width = int(screen_width)
        self.screen_height = int(screen_height)
        self.frame_skip = int(frame_skip)
        self.max_episode_steps = int(max_episode_steps)
        self.render_mode = render_mode
        self.reward_scale = float(reward_scale)

        # Load engine library
        self.engine_lib = VangersEngineLib(engine_lib_path)

        # Create game instances
        self.instances = [
            VangersInstance(self.engine_lib, screen_width, screen_height)
            for _ in range(self.num_envs)
        ]

        # Define action and observation spaces
        if spaces is not None:
            self.action_space = spaces.MultiDiscrete([3, 3, 2, 2, 2])  # [movement, steering, fire, special1, special2]
            self.observation_space = spaces.Dict({
                'screen': spaces.Box(
                    low=0, high=255,
                    shape=(screen_height, screen_width, 3),
                    dtype=np.uint8
                ),
                'state': spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(20,),  # State vector size
                    dtype=np.float32
                )
            })
        else:
            # Minimal fallback shapes for environments without gymnasium installed
            self.action_space = None
            self.observation_space = None

        # Episode tracking
        self.episode_rewards = np.zeros(self.num_envs, dtype=float)
        self.episode_lengths = np.zeros(self.num_envs, dtype=int)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset all environments and return stacked observations and infos list."""
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

        stacked_obs = self._stack_observations(observations)
        return stacked_obs, infos

    def step(self, actions: Union[np.ndarray, List[np.ndarray]]):
        """Step all environments in parallel with the provided actions.

        Actions may be:
          - a single 1D numpy array (applied to all envs)
          - a 2D numpy array of shape (num_envs, action_dim)
          - a list of action arrays (one per env)
        """
        # Normalize input to a list of length num_envs
        if isinstance(actions, np.ndarray):
            if actions.ndim == 1:
                actions = [actions.copy() for _ in range(self.num_envs)]
            elif actions.ndim == 2:
                actions = [actions[i] for i in range(self.num_envs)]
            else:
                raise ValueError("Unsupported action array shape")
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

            # Update episode trackers
            self.episode_rewards[i] += reward
            self.episode_lengths[i] += 1

            # Auto-reset on termination/truncation (to simplify batch training)
            if terminated or truncated:
                info['episode'] = {
                    'r': float(self.episode_rewards[i]),
                    'l': int(self.episode_lengths[i])
                }
                reset_obs, reset_info = instance.reset()
                observations[i] = reset_obs
                infos[i].update(reset_info)
                self.episode_rewards[i] = 0.0
                self.episode_lengths[i] = 0

        stacked_obs = self._stack_observations(observations)
        return stacked_obs, np.array(rewards, dtype=float), np.array(terminations, dtype=bool), np.array(truncations, dtype=bool), infos

    def _stack_observations(self, observations: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Stack a list of observations into batched observations."""
        screens = np.stack([obs['screen'] for obs in observations])
        states = np.stack([obs['state'] for obs in observations])

        return {
            'screen': screens,
            'state': states
        }

    def render(self, mode: Optional[str] = None):
        """Render the first environment. Supports 'rgb_array' and 'human'."""
        render_mode = mode or self.render_mode

        if render_mode == 'rgb_array':
            return self.instances[0].frame_buffer
        elif render_mode == 'human':
            try:
                import cv2  # type: ignore
                cv2.imshow('Vangers Vectorized Env', self.instances[0].frame_buffer)
                cv2.waitKey(1)
            except Exception:
                print("OpenCV not available for rendering (install opencv-python)")

        return None

    def close(self):
        """Clean up resources for all instances."""
        # Instances are cleaned by their destructors; explicit cleanup can be added here.
        for instance in self.instances:
            try:
                if hasattr(instance, "instance_ptr") and instance.instance_ptr:
                    # best-effort destroy using wrapper
                    self.engine_lib.destroy_instance(instance.instance_ptr)
                    instance.instance_ptr = None
            except Exception:
                pass

        # Optionally call global cleanup for the engine
        try:
            # Global cleanup via wrapper
            self.engine_lib.cleanup()
        except Exception:
            pass

    def get_episode_rewards(self) -> np.ndarray:
        return self.episode_rewards.copy()

    def get_episode_lengths(self) -> np.ndarray:
        return self.episode_lengths.copy()

    def set_time_scale(self, scale: float):
        """Set simulation time scale for all instances (if supported by library)."""
        # Use wrapper to set time scale for each instance; wrapper is no-op if function missing
        for instance in self.instances:
            try:
                self.engine_lib.set_time_scale(instance.instance_ptr, float(scale))
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Convenience single-environment wrapper
# ---------------------------------------------------------------------------

class VangersEnv(gym.Env if gym is not None else object):
    """
    Single Vangers environment that provides a standard Gym interface.
    This is a simple wrapper around VangersInstance for easier use.
    """

    def __init__(self, width: int = 640, height: int = 480, render_mode: Optional[str] = None, engine_lib_path: Optional[str] = None):
        self.width = int(width)
        self.height = int(height)
        self.render_mode = render_mode

        # Initialize engine library
        self.engine_lib = VangersEngineLib(engine_lib_path)

        # Create single instance
        self.instance = VangersInstance(self.engine_lib, width, height)

        # Define action and observation spaces (compatible with Gym)
        if spaces is not None:
            self.action_space = spaces.MultiDiscrete([3, 3, 2, 2, 2])
            self.observation_space = spaces.Dict({
                'screen': spaces.Box(low=0, high=255, shape=(height, width, 3), dtype=np.uint8),
                'state': spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
            })
        else:
            # Fallback if gymnasium is not available
            class SimpleSpace:
                def __init__(self, shape=None, dtype=None, nvec=None):
                    self.shape = shape
                    self.dtype = dtype
                    self.nvec = nvec

                def sample(self):
                    if self.nvec is not None:
                        return [int(np.random.randint(0, n)) for n in self.nvec]
                    return np.random.rand(*self.shape).astype(self.dtype)

            class MultiDiscrete(SimpleSpace):
                def __init__(self, nvec):
                    super().__init__(nvec=nvec)
            self.action_space = MultiDiscrete([3, 3, 2, 2, 2])
            self.observation_space = {
                'screen': SimpleSpace(shape=(height, width, 3), dtype=np.uint8),
                'state': SimpleSpace(shape=(20,), dtype=np.float32)
            }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset the environment and return initial observation and info."""
        if seed is not None:
            np.random.seed(seed)
        observation, info = self.instance.reset()
        return observation, info

    def step(self, action):
        """Take an action and return (observation, reward, terminated, truncated, info)."""
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        observation, reward, terminated, truncated, info = self.instance.step(action)
        return observation, reward, terminated, truncated, info

    def render(self, mode: Optional[str] = None):
        """Render or return the RGB array depending on mode."""
        render_mode = mode or self.render_mode
        if render_mode == "rgb_array":
            observation = self.instance._get_observation()
            return observation.get('screen')
        elif render_mode == "human":
            try:
                import cv2  # type: ignore
                buf = self.instance.frame_buffer
                cv2.imshow("VangersEnv", buf)
                cv2.waitKey(1)
            except Exception:
                # If cv2 not available, just return the array
                return self.instance.frame_buffer
        return None

    def close(self):
        """Close the environment and clean up resources."""
        try:
            if hasattr(self, 'instance'):
                del self.instance
        except Exception:
            pass

        try:
            if hasattr(self, 'engine_lib') and self.engine_lib and hasattr(self.engine_lib._lib, "vangers_engine_cleanup"):
                self.engine_lib._lib.vangers_engine_cleanup()
                del self.engine_lib
        except Exception:
            pass

    def __del__(self):
        self.close()


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_vangers_env(**kwargs) -> VangersVectorizedEnv:
    """Create a single Vangers environment (wrapper around vectorized env with num_envs=1)."""
    kwargs = dict(kwargs)
    kwargs['num_envs'] = 1
    return VangersVectorizedEnv(**kwargs)


# ---------------------------------------------------------------------------
# Simple CLI test when run as a script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing Vangers Vectorized Environment...")

    try:
        env = VangersVectorizedEnv(num_envs=2, screen_width=320, screen_height=240)
        obs, infos = env.reset()
        print(f"Observation shapes: screen={obs['screen'].shape}, state={obs['state'].shape}")
        print(f"Action space: {getattr(env, 'action_space', None)}")

        for step in range(5):
            # Sample actions (fallback to zeros if action_space is not available)
            if hasattr(env, 'action_space') and getattr(env.action_space, 'sample', None):
                actions = [env.action_space.sample() for _ in range(env.num_envs)]
            else:
                actions = [np.zeros(5, dtype=int) for _ in range(env.num_envs)]

            obs, rewards, terms, truncs, infos = env.step(actions)
            print(f"Step {step}: rewards={rewards}, any_done={np.any(terms | truncs)}")

        env.close()
        print("Test completed successfully!")

    except Exception as e:
        print("Test failed:", e)
        print("Make sure Vangers engine library is built and available.")
        print("Build with: cmake -DBUILD_SHARED_ENGINE=ON && make")
