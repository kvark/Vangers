#!/usr/bin/env python3
"""
Common utilities and classes shared across Vangers gym wrapper tests.
Contains C struct definitions, library loading utilities, and helper functions.
"""

import ctypes
import os
import sys
import numpy as np
from pathlib import Path


# C struct definitions (shared across all tests)
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


def find_engine_library():
    """Find the Vangers engine shared library."""
    possible_paths = [
        "../libvangers_engine.so",
        "./libvangers_engine.so",
        "../build/libvangers_engine.so",
        "./build/libvangers_engine.so",
        "../build/gym_wrapper/libvangers_engine.so",
        "./build/gym_wrapper/libvangers_engine.so",
    ]

    for path in possible_paths:
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        "Could not find libvangers_engine.so. "
        "Build with: cmake -DBUILD_SHARED_ENGINE=ON && make"
    )


def load_engine_library(lib_path=None):
    """Load the library and set up function signatures."""
    if lib_path is None:
        lib_path = find_engine_library()

    lib = ctypes.CDLL(lib_path)

    # Engine initialization
    lib.vangers_engine_init.restype = ctypes.c_int
    lib.vangers_engine_init.argtypes = []

    lib.vangers_engine_cleanup.restype = None
    lib.vangers_engine_cleanup.argtypes = []

    lib.vangers_get_version.restype = ctypes.c_char_p
    lib.vangers_get_version.argtypes = []

    lib.vangers_get_num_instances.restype = ctypes.c_int
    lib.vangers_get_num_instances.argtypes = []

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

    lib.vangers_set_time_scale.restype = None
    lib.vangers_set_time_scale.argtypes = [ctypes.c_void_p, ctypes.c_float]

    lib.vangers_pause_simulation.restype = None
    lib.vangers_pause_simulation.argtypes = [ctypes.c_void_p, ctypes.c_bool]

    lib.vangers_set_action.restype = None
    lib.vangers_set_action.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]

    # State query
    lib.vangers_get_player_state.restype = ctypes.c_int
    lib.vangers_get_player_state.argtypes = [ctypes.c_void_p, ctypes.POINTER(PlayerState)]

    lib.vangers_get_frame_buffer.restype = ctypes.c_int
    lib.vangers_get_frame_buffer.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int
    ]

    # Events
    lib.vangers_get_events.restype = ctypes.c_int
    lib.vangers_get_events.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(GameEvent), ctypes.c_int
    ]

    lib.vangers_clear_events.restype = None
    lib.vangers_clear_events.argtypes = [ctypes.c_void_p]

    # Environment configuration
    lib.vangers_set_map.restype = None
    lib.vangers_set_map.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

    lib.vangers_set_render_mode.restype = None
    lib.vangers_set_render_mode.argtypes = [ctypes.c_void_p, ctypes.c_int]

    lib.vangers_set_physics_substeps.restype = None
    lib.vangers_set_physics_substeps.argtypes = [ctypes.c_void_p, ctypes.c_int]

    return lib


class TestInstance:
    """Wrapper for a single test instance with common operations."""
    __test__ = False

    def __init__(self, lib, width=320, height=240):
        self.lib = lib
        self.width = width
        self.height = height
        self.instance_ptr = None
        self.player_state = PlayerState()
        self.frame_buffer = np.zeros((height, width, 3), dtype=np.uint8)
        self.events_buffer = (GameEvent * 32)()

    def create(self):
        """Create the instance."""
        self.instance_ptr = self.lib.vangers_create_instance(self.width, self.height)
        if not self.instance_ptr:
            raise RuntimeError("Failed to create instance")
        return self.instance_ptr

    def reset(self):
        """Reset the instance."""
        if not self.instance_ptr:
            raise RuntimeError("Instance not created")
        result = self.lib.vangers_reset_instance(self.instance_ptr)
        if result != 1:
            raise RuntimeError(f"Failed to reset instance: {result}")
        return result

    def step(self, num_steps=1):
        """Step the simulation."""
        if not self.instance_ptr:
            raise RuntimeError("Instance not created")
        result = self.lib.vangers_step_simulation(self.instance_ptr, num_steps)
        return result

    def set_action(self, movement=0, steering=0, fire=0, special1=0, special2=0):
        """Set action for the instance."""
        if not self.instance_ptr:
            raise RuntimeError("Instance not created")
        self.lib.vangers_set_action(self.instance_ptr, movement, steering, fire, special1, special2)

    def get_player_state(self):
        """Get current player state."""
        if not self.instance_ptr:
            raise RuntimeError("Instance not created")
        result = self.lib.vangers_get_player_state(self.instance_ptr, ctypes.byref(self.player_state))
        return result == 1, self.player_state

    def get_frame_buffer(self):
        """Get current frame buffer."""
        if not self.instance_ptr:
            raise RuntimeError("Instance not created")
        frame_ptr = self.frame_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        result = self.lib.vangers_get_frame_buffer(self.instance_ptr, frame_ptr, self.frame_buffer.size)
        return result == 1, self.frame_buffer

    def get_events(self):
        """Get current events."""
        if not self.instance_ptr:
            raise RuntimeError("Instance not created")
        num_events = self.lib.vangers_get_events(self.instance_ptr, self.events_buffer, 32)
        events = []
        for i in range(num_events):
            events.append({
                'type': self.events_buffer[i].type,
                'value': self.events_buffer[i].value,
                'x': self.events_buffer[i].x,
                'y': self.events_buffer[i].y,
                'timestamp': self.events_buffer[i].timestamp
            })
        return events

    def clear_events(self):
        """Clear events."""
        if not self.instance_ptr:
            raise RuntimeError("Instance not created")
        self.lib.vangers_clear_events(self.instance_ptr)

    def destroy(self):
        """Destroy the instance."""
        if self.instance_ptr:
            self.lib.vangers_destroy_instance(self.instance_ptr)
            self.instance_ptr = None

    def __enter__(self):
        """Context manager entry."""
        self.create()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.destroy()


class TestResults:
    """Helper class for collecting and reporting test results."""
    __test__ = False

    def __init__(self):
        self.tests = []
        self.passed = 0
        self.failed = 0

    def add_test(self, name, passed, message=""):
        """Add a test result."""
        self.tests.append({
            'name': name,
            'passed': passed,
            'message': message
        })
        if passed:
            self.passed += 1
        else:
            self.failed += 1

    def print_summary(self):
        """Print test summary."""
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print(f"{'='*60}")

        for test in self.tests:
            status = "✓ PASS" if test['passed'] else "✗ FAIL"
            message = f" - {test['message']}" if test['message'] else ""
            print(f"{status}: {test['name']}{message}")

        print(f"\nTotal: {len(self.tests)} tests, {self.passed} passed, {self.failed} failed")

        if self.failed == 0:
            print("🎉 ALL TESTS PASSED!")
        else:
            print(f"❌ {self.failed} TEST(S) FAILED")

        return self.failed == 0


def print_player_state(state, title="Player State"):
    """Pretty print player state."""
    print(f"\n{title}:")
    print(f"  Position: ({state.position.x:.1f}, {state.position.y:.1f}, {state.position.z:.1f})")
    print(f"  Velocity: ({state.velocity.x:.1f}, {state.velocity.y:.1f}, {state.velocity.z:.1f})")
    print(f"  Angle: {state.angle:.2f}")
    print(f"  Armor: {state.armor}/{state.max_speed}")
    print(f"  Energy: {state.energy}")
    print(f"  Speed: {state.speed}/{state.max_speed}")
    print(f"  Status: {'Alive' if state.alive else 'Dead'}, {'On Ground' if state.on_ground else 'Airborne'}")


def run_basic_engine_test(lib):
    """Run basic engine functionality test."""
    results = TestResults()

    try:
        # Test version
        version = lib.vangers_get_version().decode('utf-8')
        results.add_test("Get Version", True, f"version={version}")

        # Test instance creation
        with TestInstance(lib, 160, 120) as instance:
            results.add_test("Create Instance", True, f"ptr={hex(instance.instance_ptr)}")

            # Test reset
            reset_result = instance.reset()
            results.add_test("Reset Instance", reset_result == 1)

            # Test player state
            success, state = instance.get_player_state()
            results.add_test("Get Player State", success)

            # Test frame buffer
            success, frame = instance.get_frame_buffer()
            results.add_test("Get Frame Buffer", success, f"size={frame.size}")

            # Test actions and stepping
            instance.set_action(1, 0, 0, 0, 0)  # Move forward
            step_result = instance.step(1)
            results.add_test("Step Simulation", step_result == 1)

            # Test state change after action
            success2, state2 = instance.get_player_state()
            pos_changed = (state2.position.x != state.position.x or
                          state2.position.y != state.position.y)
            results.add_test("State Change After Action", pos_changed,
                           f"pos change: {pos_changed}")

    except Exception as e:
        results.add_test("Basic Engine Test", False, str(e))

    return results


def benchmark_performance(lib, num_instances=1, steps_per_instance=100):
    """Benchmark engine performance."""
    import time

    print(f"\n{'='*60}")
    print(f"PERFORMANCE BENCHMARK")
    print(f"{'='*60}")
    print(f"Testing {num_instances} instance(s), {steps_per_instance} steps each")

    instances = []

    try:
        # Create instances
        start_time = time.time()
        for i in range(num_instances):
            instance = TestInstance(lib, 160, 120)
            instance.create()
            instance.reset()
            instances.append(instance)
        creation_time = time.time() - start_time

        print(f"Instance creation: {creation_time:.3f}s ({creation_time/num_instances:.3f}s per instance)")

        # Run simulation
        start_time = time.time()
        total_steps = 0
        for instance in instances:
            instance.set_action(1, 0, 0, 0, 0)  # Move forward
            for step in range(steps_per_instance):
                instance.step(1)
                total_steps += 1
        simulation_time = time.time() - start_time

        print(f"Simulation: {simulation_time:.3f}s ({total_steps/simulation_time:.1f} steps/sec)")
        print(f"Per instance: {simulation_time/num_instances:.3f}s ({steps_per_instance/simulation_time*num_instances:.1f} steps/sec)")

        # Test state queries
        start_time = time.time()
        for _ in range(100):
            for instance in instances:
                instance.get_player_state()
        query_time = time.time() - start_time
        queries = 100 * num_instances

        print(f"State queries: {query_time:.3f}s ({queries/query_time:.1f} queries/sec)")

    finally:
        # Cleanup
        for instance in instances:
            instance.destroy()

    print(f"✓ Performance benchmark completed")
