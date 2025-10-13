#!/usr/bin/env python3
"""
Consolidated Vangers Engine Test Suite

This script provides comprehensive testing of the Vangers gymnasium wrapper:
- Basic C library functionality
- Single instance operations
- Multiple instance management
- Gymnasium interface compatibility
- Performance benchmarking

Usage:
    python test_engine.py [options]

Options:
    --basic         Test basic engine functionality (default)
    --multiple      Test multiple instances
    --gymnasium     Test gymnasium interface
    --vectorized    Test vectorized environment
    --performance   Run performance benchmarks
    --all           Run all tests
    --instances N   Number of instances for multi-instance tests (default: 3)
    --steps N       Number of steps per test (default: 50)
    --verbose       Enable verbose output
    --help         Show this help message
"""

import argparse
import sys
import os
import time
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_common import (
    find_engine_library, load_engine_library, TestInstance, TestResults,
    print_player_state, run_basic_engine_test, benchmark_performance,
    Vector3, PlayerState, GameEvent
)


def test_basic_functionality(lib, verbose=False):
    """Test basic engine functionality."""
    print("\n" + "="*60)
    print("BASIC ENGINE FUNCTIONALITY TEST")
    print("="*60)

    results = run_basic_engine_test(lib)

    if verbose:
        # Additional detailed tests
        try:
            with TestInstance(lib, 320, 240) as instance:
                instance.reset()

                if verbose:
                    success, state = instance.get_player_state()
                    if success:
                        print_player_state(state, "Initial State")

                # Test different actions
                actions = [
                    (0, 0, 0, 0, 0, "Stop"),
                    (1, 0, 0, 0, 0, "Forward"),
                    (2, 0, 0, 0, 0, "Backward"),
                    (1, 1, 0, 0, 0, "Forward+Left"),
                    (1, 2, 0, 0, 0, "Forward+Right"),
                    (1, 0, 1, 0, 0, "Forward+Fire"),
                ]

                for movement, steering, fire, sp1, sp2, name in actions:
                    instance.reset()
                    instance.set_action(movement, steering, fire, sp1, sp2)

                    success, state_before = instance.get_player_state()
                    instance.step(5)  # Take 5 steps
                    success, state_after = instance.get_player_state()

                    pos_change = (
                        state_after.position.x - state_before.position.x,
                        state_after.position.y - state_before.position.y
                    )

                    energy_change = state_after.energy - state_before.energy

                    results.add_test(f"Action: {name}", True,
                                   f"pos_Δ=({pos_change[0]:.2f},{pos_change[1]:.2f}), energy_Δ={energy_change}")

                    if verbose:
                        print(f"  {name}: Position change=({pos_change[0]:.2f}, {pos_change[1]:.2f}), Energy change={energy_change}")

        except Exception as e:
            results.add_test("Detailed Action Test", False, str(e))

    return results.print_summary()


def test_multiple_instances(lib, num_instances=3, steps=50, verbose=False):
    """Test multiple instance management."""
    print("\n" + "="*60)
    print(f"MULTIPLE INSTANCES TEST ({num_instances} instances)")
    print("="*60)

    results = TestResults()
    instances = []

    try:
        # Test instance creation
        print(f"Creating {num_instances} instances...")
        for i in range(num_instances):
            instance = TestInstance(lib, 160, 120)
            instance.create()
            instances.append(instance)
            if verbose:
                print(f"  Instance {i+1}: {hex(instance.instance_ptr)}")

        results.add_test("Create Multiple Instances", len(instances) == num_instances,
                        f"{len(instances)}/{num_instances} created")

        # Test parallel reset
        print("Testing parallel reset...")
        for i, instance in enumerate(instances):
            reset_result = instance.reset()
            if reset_result != 1:
                results.add_test(f"Reset Instance {i+1}", False, f"result={reset_result}")
            else:
                results.add_test(f"Reset Instance {i+1}", True)

        # Test state isolation
        print("Testing state isolation...")
        initial_states = []
        for i, instance in enumerate(instances):
            # Set different actions for each instance
            instance.set_action(1, i % 3, 0, 0, 0)  # Different steering for each
            success, state = instance.get_player_state()
            initial_states.append(state)

        # Step each instance different amounts
        for i, instance in enumerate(instances):
            instance.step((i + 1) * 10)  # 10, 20, 30 steps etc.

        # Check that states are different (isolation working)
        final_states = []
        for i, instance in enumerate(instances):
            success, state = instance.get_player_state()
            final_states.append(state)

        # Verify states are different
        positions_different = True
        for i in range(len(final_states)):
            for j in range(i + 1, len(final_states)):
                pos_i = final_states[i].position
                pos_j = final_states[j].position
                if abs(pos_i.x - pos_j.x) < 0.1 and abs(pos_i.y - pos_j.y) < 0.1:
                    positions_different = False
                    break

        results.add_test("State Isolation", positions_different,
                        "Instance states properly isolated")

        if verbose:
            for i, (initial, final) in enumerate(zip(initial_states, final_states)):
                print(f"  Instance {i+1}: "
                      f"({initial.position.x:.1f},{initial.position.y:.1f}) → "
                      f"({final.position.x:.1f},{final.position.y:.1f})")

        # Test concurrent access
        print("Testing concurrent access...")
        def worker_function(instance_id):
            instance = instances[instance_id]
            try:
                for step in range(steps):
                    instance.set_action(1, 0, 0, 0, 0)
                    step_result = instance.step(1)
                    if step_result != 1:
                        return False, f"Step failed at {step}"

                    # Occasionally query state
                    if step % 10 == 0:
                        success, state = instance.get_player_state()
                        if not success:
                            return False, f"State query failed at step {step}"

                return True, "Success"
            except Exception as e:
                return False, str(e)

        # Run concurrent workers
        with ThreadPoolExecutor(max_workers=num_instances) as executor:
            futures = [executor.submit(worker_function, i) for i in range(num_instances)]

            concurrent_results = []
            for i, future in enumerate(as_completed(futures)):
                success, message = future.result()
                concurrent_results.append(success)
                results.add_test(f"Concurrent Worker {i+1}", success, message)

        all_concurrent_passed = all(concurrent_results)
        results.add_test("Concurrent Access", all_concurrent_passed,
                        f"{sum(concurrent_results)}/{len(concurrent_results)} workers succeeded")

    except Exception as e:
        results.add_test("Multiple Instances Test", False, str(e))

    finally:
        # Cleanup
        print("Cleaning up instances...")
        for instance in instances:
            instance.destroy()

    return results.print_summary()


def test_gymnasium_interface(lib, vectorized=False, verbose=False):
    """Test gymnasium interface compatibility."""
    print("\n" + "="*60)
    print(f"GYMNASIUM INTERFACE TEST ({'Vectorized' if vectorized else 'Single'} Environment)")
    print("="*60)

    results = TestResults()

    try:
        if vectorized:
            # Test vectorized environment
            from vangers_env import VangersVectorizedEnv

            env = VangersVectorizedEnv(num_envs=2, width=160, height=120)
            results.add_test("Create Vectorized Environment", True, "2 environments")

            # Test reset
            observations, infos = env.reset()
            results.add_test("Vectorized Reset", len(observations['screen']) == 2,
                           f"Got {len(observations['screen'])} observations")

            if verbose:
                print(f"  Observation shapes: screen={observations['screen'].shape}, state={observations['state'].shape}")

            # Test step
            actions = np.array([[1, 0, 0, 0, 0], [1, 1, 0, 0, 0]])  # Different actions for each env
            observations, rewards, terminated, truncated, infos = env.step(actions)

            results.add_test("Vectorized Step", len(rewards) == 2,
                           f"Got {len(rewards)} rewards: {rewards}")

            # Test multiple steps
            total_rewards = [0.0, 0.0]
            for step in range(20):
                actions = np.random.randint(0, [3, 3, 2, 2, 2], size=(2, 5))
                observations, rewards, terminated, truncated, infos = env.step(actions)
                total_rewards[0] += rewards[0]
                total_rewards[1] += rewards[1]

            results.add_test("Vectorized Multi-Step", True,
                           f"20 steps, rewards: [{total_rewards[0]:.2f}, {total_rewards[1]:.2f}]")

            env.close()

        else:
            # Test single environment
            from vangers_env import VangersEnv

            env = VangersEnv(width=320, height=240)
            results.add_test("Create Single Environment", True)

            # Test reset
            observation, info = env.reset(seed=42)
            results.add_test("Single Reset", 'screen' in observation and 'state' in observation,
                           f"obs keys: {list(observation.keys())}")

            if verbose:
                print(f"  Screen shape: {observation['screen'].shape}")
                print(f"  State shape: {observation['state'].shape}")
                print(f"  Info: {info}")

            # Test action space
            action_space_valid = hasattr(env.action_space, 'sample')
            results.add_test("Action Space", action_space_valid, f"type: {type(env.action_space)}")

            # Test observation space
            obs_space_valid = hasattr(env.observation_space, 'keys')
            results.add_test("Observation Space", obs_space_valid, f"keys: {list(env.observation_space.keys()) if obs_space_valid else 'N/A'}")

            # Test episode
            total_reward = 0
            steps = 0
            for step in range(50):
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1

                if terminated or truncated:
                    if verbose:
                        print(f"  Episode ended at step {steps}")
                    break

            results.add_test("Single Episode", steps > 0,
                           f"{steps} steps, reward: {total_reward:.3f}")

            env.close()

    except ImportError as e:
        results.add_test("Gymnasium Import", False, f"ImportError: {e}")
        print("Note: Install with 'pip install gymnasium' for full gymnasium testing")
    except Exception as e:
        results.add_test("Gymnasium Interface Test", False, str(e))

    return results.print_summary()


def test_performance(lib, num_instances=1, steps_per_instance=1000, verbose=False):
    """Run performance benchmarks."""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK")
    print("="*60)

    benchmark_performance(lib, num_instances, steps_per_instance)

    # Additional performance tests
    if verbose:
        print("\nDetailed Performance Analysis:")

        # Test different instance sizes
        sizes = [(160, 120), (320, 240), (640, 480)]
        for width, height in sizes:
            start_time = time.time()
            with TestInstance(lib, width, height) as instance:
                instance.reset()
                for _ in range(100):
                    instance.step(1)
            duration = time.time() - start_time
            print(f"  {width}x{height}: {duration:.3f}s for 100 steps ({100/duration:.1f} fps)")

    return True


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Vangers Engine Test Suite")
    parser.add_argument('--basic', action='store_true', help='Test basic functionality')
    parser.add_argument('--multiple', action='store_true', help='Test multiple instances')
    parser.add_argument('--gymnasium', action='store_true', help='Test gymnasium interface')
    parser.add_argument('--vectorized', action='store_true', help='Test vectorized environment')
    parser.add_argument('--performance', action='store_true', help='Run performance benchmarks')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--instances', type=int, default=3, help='Number of instances for multi-instance tests')
    parser.add_argument('--steps', type=int, default=50, help='Number of steps per test')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')

    args = parser.parse_args()

    # Default to basic test if no specific tests requested
    if not any([args.basic, args.multiple, args.gymnasium, args.vectorized, args.performance, args.all]):
        args.basic = True

    print("VANGERS ENGINE TEST SUITE")
    print("=" * 80)

    try:
        # Find and load library
        lib_path = find_engine_library()
        print(f"Found library: {lib_path}")

        lib = load_engine_library(lib_path)
        print("Library loaded successfully")

        # Initialize engine
        result = lib.vangers_engine_init()
        if result != 1:
            print(f"❌ Engine initialization failed: {result}")
            return 1

        version = lib.vangers_get_version().decode('utf-8')
        print(f"Engine version: {version}")

        # Track overall results
        all_passed = True

        # Run requested tests
        if args.basic or args.all:
            passed = test_basic_functionality(lib, args.verbose)
            all_passed = all_passed and passed

        if args.multiple or args.all:
            passed = test_multiple_instances(lib, args.instances, args.steps, args.verbose)
            all_passed = all_passed and passed

        if args.gymnasium or args.all:
            passed = test_gymnasium_interface(lib, vectorized=False, verbose=args.verbose)
            all_passed = all_passed and passed

        if args.vectorized or args.all:
            passed = test_gymnasium_interface(lib, vectorized=True, verbose=args.verbose)
            all_passed = all_passed and passed

        if args.performance or args.all:
            passed = test_performance(lib, args.instances, args.steps * 20, args.verbose)
            all_passed = all_passed and passed

        # Cleanup
        lib.vangers_engine_cleanup()

        # Final summary
        print("\n" + "=" * 80)
        if all_passed:
            print("🎉 ALL TESTS PASSED!")
            print("The Vangers gym wrapper is working correctly.")
            return 0
        else:
            print("❌ SOME TESTS FAILED")
            print("Please check the output above for details.")
            return 1

    except FileNotFoundError as e:
        print(f"❌ Library not found: {e}")
        print("Build the library with: cmake -DBUILD_SHARED_ENGINE=ON && make")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
