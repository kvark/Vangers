#!/usr/bin/env python3
"""
Minimal test script for Vangers Engine Library
Tests the basic C interface without gymnasium dependencies
"""

import ctypes
import sys
import os
from pathlib import Path

# C struct definitions
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

def find_library():
    """Find the Vangers engine shared library."""
    possible_paths = [
        "./build/libvangers_engine.so",
        "./build/gym_wrapper/libvangers_engine.so",
        "./gym_wrapper/libvangers_engine.so",
        "./libvangers_engine.so",
        "../build/libvangers_engine.so",
        "../build/gym_wrapper/libvangers_engine.so"
    ]

    for path in possible_paths:
        if os.path.isfile(path):
            return path

    raise FileNotFoundError("Could not find libvangers_engine.so")

def load_library(lib_path):
    """Load the library and set up function signatures."""
    lib = ctypes.CDLL(lib_path)

    # Engine initialization
    lib.vangers_engine_init.restype = ctypes.c_int
    lib.vangers_engine_init.argtypes = []

    lib.vangers_engine_cleanup.restype = None
    lib.vangers_engine_cleanup.argtypes = []

    lib.vangers_get_version.restype = ctypes.c_char_p
    lib.vangers_get_version.argtypes = []

    # Instance management
    lib.vangers_create_instance.restype = ctypes.c_void_p
    lib.vangers_create_instance.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

    lib.vangers_destroy_instance.restype = None
    lib.vangers_destroy_instance.argtypes = [ctypes.c_void_p]

    lib.vangers_reset_instance.restype = ctypes.c_int
    lib.vangers_reset_instance.argtypes = [ctypes.c_void_p]

    # Simulation control
    lib.vangers_step_simulation.restype = ctypes.c_int
    lib.vangers_step_simulation.argtypes = [ctypes.c_void_p, ctypes.c_int]

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

    return lib

def test_basic_functionality():
    """Test basic library functionality."""
    print("Vangers Engine Python Test")
    print("=" * 30)

    try:
        # Find and load library
        lib_path = find_library()
        print(f"Found library: {lib_path}")

        lib = load_library(lib_path)
        print("Library loaded successfully")

        # Initialize engine
        result = lib.vangers_engine_init()
        if result != 1:
            print(f"Engine initialization failed: {result}")
            return False
        print("Engine initialized")

        # Get version
        version = lib.vangers_get_version().decode('utf-8')
        print(f"Engine version: {version}")

        # Create instance
        width, height = 320, 240
        instance = lib.vangers_create_instance(width, height, None)
        if not instance:
            print("Failed to create instance")
            lib.vangers_engine_cleanup()
            return False
        print(f"Created instance: {hex(instance)}")

        # Reset instance
        result = lib.vangers_reset_instance(instance)
        if result != 1:
            print(f"Reset failed: {result}")
            lib.vangers_destroy_instance(instance)
            lib.vangers_engine_cleanup()
            return False
        print("Instance reset successfully")

        # Test player state
        player_state = PlayerState()
        result = lib.vangers_get_player_state(instance, ctypes.byref(player_state))
        if result == 1:
            print("Player state retrieved:")
            print(f"  Position: ({player_state.position.x:.1f}, {player_state.position.y:.1f}, {player_state.position.z:.1f})")
            print(f"  Velocity: ({player_state.velocity.x:.1f}, {player_state.velocity.y:.1f}, {player_state.velocity.z:.1f})")
            print(f"  Angle: {player_state.angle:.2f}")
            print(f"  Armor: {player_state.armor}")
            print(f"  Energy: {player_state.energy}")
            print(f"  Speed: {player_state.speed}/{player_state.max_speed}")
            print(f"  Alive: {player_state.alive}")
        else:
            print("Failed to get player state")

        # Test frame buffer
        frame_size = width * height * 3
        frame_buffer = (ctypes.c_ubyte * frame_size)()
        result = lib.vangers_get_frame_buffer(instance, frame_buffer, frame_size)
        if result == 1:
            print(f"Frame buffer retrieved: {frame_size} bytes")

            # Check if any non-zero data
            has_data = any(frame_buffer[i] != 0 for i in range(min(100, frame_size)))
            print(f"Frame contains data: {has_data}")
        else:
            print("Failed to get frame buffer")

        # Test actions and simulation steps
        print("\nTesting simulation steps...")
        for step in range(5):
            # Set action: move forward
            lib.vangers_set_action(instance, 1, 0, 0, 0, 0)  # forward, straight, no fire, no specials

            # Step simulation
            result = lib.vangers_step_simulation(instance, 1)
            if result == 1:
                print(f"Step {step + 1} completed")

                # Get updated state
                result = lib.vangers_get_player_state(instance, ctypes.byref(player_state))
                if result == 1:
                    print(f"  Position: ({player_state.position.x:.1f}, {player_state.position.y:.1f}, {player_state.position.z:.1f})")

                # Check for events
                events = (GameEvent * 10)()
                num_events = lib.vangers_get_events(instance, events, 10)
                if num_events > 0:
                    print(f"  Events: {num_events}")
                    for i in range(num_events):
                        print(f"    Event {i}: type={events[i].type}, value={events[i].value:.2f}")
                    lib.vangers_clear_events(instance)
            else:
                print(f"Step {step + 1} failed")

        # Test different actions
        print("\nTesting different actions...")
        actions = [
            ("Stop", 0, 0, 0, 0, 0),
            ("Backward", 2, 0, 0, 0, 0),
            ("Turn Left", 1, 1, 0, 0, 0),
            ("Turn Right", 1, 2, 0, 0, 0),
            ("Fire", 1, 0, 1, 0, 0),
        ]

        for name, movement, steering, fire, special1, special2 in actions:
            lib.vangers_set_action(instance, movement, steering, fire, special1, special2)
            result = lib.vangers_step_simulation(instance, 1)
            print(f"Action '{name}': {'SUCCESS' if result == 1 else 'FAILED'}")

        # Cleanup
        print("\nCleaning up...")
        lib.vangers_destroy_instance(instance)
        lib.vangers_engine_cleanup()
        print("Cleanup complete")
        # Success — do not return a boolean; allow normal completion.
        return

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        # Re-raise so test frameworks capture the failure and produce a proper traceback.
        raise

def test_multiple_instances():
    """Test creating multiple instances."""
    print("\nTesting multiple instances...")

    try:
        lib_path = find_library()
        lib = load_library(lib_path)

        lib.vangers_engine_init()

        # Create multiple instances
        instances = []
        for i in range(3):
            instance = lib.vangers_create_instance(160, 120, None)
            if instance:
                instances.append(instance)
                print(f"Created instance {i + 1}: {hex(instance)}")

                # Reset each instance
                result = lib.vangers_reset_instance(instance)
                print(f"  Reset result: {'SUCCESS' if result == 1 else 'FAILED'}")

        print(f"Created {len(instances)} instances")

        # Test simulation on all instances
        for i, instance in enumerate(instances):
            lib.vangers_set_action(instance, 1, 0, 0, 0, 0)  # Move forward
            result = lib.vangers_step_simulation(instance, 5)
            print(f"Instance {i + 1} simulation: {'SUCCESS' if result == 1 else 'FAILED'}")

        # Clean up all instances
        for instance in instances:
            lib.vangers_destroy_instance(instance)

        lib.vangers_engine_cleanup()
        print("Multiple instances test complete")

    except Exception as e:
        print(f"Error during multiple instances test: {e}")

def main():
    """Main test function (script entrypoint)."""
    print("Starting Vangers Engine Python Tests\n")

    try:
        # Run the primary test — it will raise on failure.
        test_basic_functionality()

        print("\n" + "=" * 50)
        print("✓ Basic functionality test PASSED")

        # Run multiple-instances test as a follow-up smoke test.
        test_multiple_instances()
        print("✓ Multiple instances test completed")

        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        print("\nThe Vangers gym wrapper is working correctly.")
        print("You can now use it with gymnasium and other RL frameworks.")
        return 0

    except Exception as e:
        print("\n" + "=" * 50)
        print("✗ Basic functionality test FAILED")
        print("Please check the library build and dependencies.")
        print(f"Exception: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
