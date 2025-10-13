#!/usr/bin/env python3
"""
Gymnasium Interface Test for Vangers Engine

Tests the high-level Python gymnasium interface including:
- Single environment (VangersEnv)
- Vectorized environment (VangersVectorizedEnv)
- Action/observation spaces
- Episode management
- Integration with RL frameworks

Usage:
    python test_gymnasium.py [options]

Options:
    --single        Test single environment (default)
    --vectorized    Test vectorized environment
    --episodes N    Number of episodes to run (default: 3)
    --steps N       Max steps per episode (default: 100)
    --envs N        Number of environments for vectorized test (default: 2)
    --verbose       Enable verbose output
    --benchmark     Run performance benchmark
    --help         Show this help message
"""

import argparse
import sys
import os
import time
import numpy as np

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_common import TestResults


def test_single_environment(episodes=3, max_steps=100, verbose=False):
    """Test single VangersEnv environment."""
    print("\n" + "="*60)
    print("SINGLE ENVIRONMENT TEST")
    print("="*60)

    results = TestResults()

    try:
        from vangers_env import VangersEnv

        # Create environment
        env = VangersEnv(width=320, height=240)
        results.add_test("Environment Creation", True)

        # Test spaces
        has_action_space = hasattr(env, 'action_space') and hasattr(env.action_space, 'sample')
        results.add_test("Action Space", has_action_space, f"type: {type(env.action_space)}")

        has_obs_space = hasattr(env, 'observation_space')
        results.add_test("Observation Space", has_obs_space, f"type: {type(env.observation_space)}")

        episode_rewards = []
        episode_lengths = []

        for episode in range(episodes):
            print(f"\nEpisode {episode + 1}:")

            # Reset environment
            observation, info = env.reset(seed=episode * 42)
            results.add_test(f"Episode {episode + 1} Reset",
                           isinstance(observation, dict) and 'screen' in observation,
                           f"obs keys: {list(observation.keys())}")

            if verbose:
                screen_shape = observation['screen'].shape
                state_shape = observation['state'].shape
                print(f"  Initial obs: screen={screen_shape}, state={state_shape}")
                print(f"  Initial info: {info}")

            total_reward = 0
            steps = 0

            # Run episode
            for step in range(max_steps):
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)

                total_reward += reward
                steps += 1

                if verbose and step % 20 == 0:
                    pos = info.get('player_position', (0, 0, 0))
                    print(f"  Step {step}: reward={reward:.3f}, pos=({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")

                if terminated or truncated:
                    break

            episode_rewards.append(total_reward)
            episode_lengths.append(steps)

            print(f"  Completed: {steps} steps, reward={total_reward:.3f}")

        # Test episode statistics
        avg_reward = np.mean(episode_rewards)
        avg_length = np.mean(episode_lengths)

        results.add_test("Episode Completion", len(episode_rewards) == episodes,
                        f"avg_reward={avg_reward:.3f}, avg_length={avg_length:.1f}")

        # Test specific functionality
        env.reset()

        # Test different actions
        test_actions = [
            ([0, 0, 0, 0, 0], "Stop"),
            ([1, 0, 0, 0, 0], "Forward"),
            ([1, 1, 0, 0, 0], "Forward+Left"),
            ([1, 0, 1, 0, 0], "Forward+Fire"),
        ]

        for action, name in test_actions:
            obs, reward, term, trunc, info = env.step(action)
            results.add_test(f"Action: {name}", True, f"reward={reward:.3f}")

        # Test render (if implemented)
        try:
            render_result = env.render()
            results.add_test("Render", True, f"type: {type(render_result)}")
        except Exception as e:
            results.add_test("Render", False, str(e))

        # Cleanup
        env.close()
        results.add_test("Environment Cleanup", True)

    except ImportError as e:
        results.add_test("VangersEnv Import", False, str(e))
    except Exception as e:
        results.add_test("Single Environment Test", False, str(e))

    success = results.print_summary()
    assert success, "One or more single-environment checks failed"


def test_vectorized_environment(num_envs=2, episodes=2, max_steps=100, verbose=False):
    """Test vectorized VangersVectorizedEnv environment."""
    print("\n" + "="*60)
    print(f"VECTORIZED ENVIRONMENT TEST ({num_envs} environments)")
    print("="*60)

    results = TestResults()

    try:
        from vangers_env import VangersVectorizedEnv

        # Create vectorized environment
        env = VangersVectorizedEnv(num_envs=num_envs, screen_width=160, screen_height=120)
        results.add_test("Vectorized Environment Creation", True, f"{num_envs} environments")

        # Test reset
        observations, infos = env.reset()

        screen_shape_correct = observations['screen'].shape[0] == num_envs
        state_shape_correct = observations['state'].shape[0] == num_envs

        results.add_test("Vectorized Reset", screen_shape_correct and state_shape_correct,
                        f"screen: {observations['screen'].shape}, state: {observations['state'].shape}")

        if verbose:
            print(f"  Observations shape: screen={observations['screen'].shape}, state={observations['state'].shape}")
            print(f"  Info count: {len(infos)}")

        # Test step with different actions for each environment
        actions = np.random.randint(0, [3, 3, 2, 2, 2], size=(num_envs, 5))
        observations, rewards, terminated, truncated, infos = env.step(actions)

        rewards_correct = len(rewards) == num_envs
        results.add_test("Vectorized Step", rewards_correct,
                        f"rewards: {rewards}, terminated: {terminated}")

        # Run multiple episodes
        episode_rewards = [[] for _ in range(num_envs)]

        for episode in range(episodes):
            print(f"\nVectorized Episode {episode + 1}:")

            observations, infos = env.reset()
            env_rewards = [0.0] * num_envs
            steps = 0

            for step in range(max_steps):
                # Random actions for each environment
                actions = np.random.randint(0, [3, 3, 2, 2, 2], size=(num_envs, 5))
                observations, rewards, terminated, truncated, infos = env.step(actions)

                for i in range(num_envs):
                    env_rewards[i] += rewards[i]

                steps += 1

                if verbose and step % 25 == 0:
                    print(f"  Step {step}: rewards={rewards}")

                # Check if any environment is done
                if np.any(terminated) or np.any(truncated):
                    if verbose:
                        done_envs = np.where(np.logical_or(terminated, truncated))[0]
                        print(f"  Environments {done_envs} finished at step {steps}")
                    break

            for i in range(num_envs):
                episode_rewards[i].append(env_rewards[i])

            print(f"  Episode rewards: {env_rewards}")

        # Test episode statistics
        for i in range(num_envs):
            avg_reward = np.mean(episode_rewards[i])
            results.add_test(f"Environment {i+1} Episodes", len(episode_rewards[i]) == episodes,
                           f"avg_reward={avg_reward:.3f}")

        # Test parallel performance
        start_time = time.time()
        for _ in range(20):
            actions = np.random.randint(0, [3, 3, 2, 2, 2], size=(num_envs, 5))
            env.step(actions)
        duration = time.time() - start_time

        steps_per_sec = (20 * num_envs) / duration
        results.add_test("Vectorized Performance", steps_per_sec > 100,
                        f"{steps_per_sec:.1f} steps/sec across {num_envs} envs")

        # Cleanup
        env.close()
        results.add_test("Vectorized Environment Cleanup", True)

    except ImportError as e:
        results.add_test("VangersVectorizedEnv Import", False, str(e))
    except Exception as e:
        results.add_test("Vectorized Environment Test", False, str(e))

    success = results.print_summary()
    assert success, "One or more vectorized-environment checks failed"


def benchmark_gymnasium_performance(num_envs=4, steps=1000, verbose=False):
    """Benchmark gymnasium interface performance."""
    print("\n" + "="*60)
    print("GYMNASIUM PERFORMANCE BENCHMARK")
    print("="*60)

    try:
        from vangers_env import VangersEnv, VangersVectorizedEnv

        # Single environment benchmark
        print(f"\nSingle Environment Benchmark ({steps} steps):")
        start_time = time.time()

        env = VangersEnv(width=160, height=120)
        env.reset()

        for step in range(steps):
            action = env.action_space.sample()
            env.step(action)

        duration = time.time() - start_time
        single_fps = steps / duration

        env.close()
        print(f"  Single env: {duration:.3f}s, {single_fps:.1f} steps/sec")

        # Vectorized environment benchmark
        print(f"\nVectorized Environment Benchmark ({num_envs} envs × {steps} steps):")
        start_time = time.time()

        vec_env = VangersVectorizedEnv(num_envs=num_envs, screen_width=160, screen_height=120)
        vec_env.reset()

        for step in range(steps):
            actions = np.random.randint(0, [3, 3, 2, 2, 2], size=(num_envs, 5))
            vec_env.step(actions)

        duration = time.time() - start_time
        vectorized_fps = (steps * num_envs) / duration

        vec_env.close()
        print(f"  Vectorized: {duration:.3f}s, {vectorized_fps:.1f} steps/sec")
        print(f"  Speedup: {vectorized_fps/single_fps:.2f}x")

        # Memory usage test (if psutil available)
        try:
            import psutil
            process = psutil.Process()

            # Create multiple environments
            envs = []
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            for i in range(5):
                env = VangersEnv(width=160, height=120)
                envs.append(env)

            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_per_env = (peak_memory - initial_memory) / 5

            for env in envs:
                env.close()

            print(f"  Memory usage: ~{memory_per_env:.1f} MB per environment")

        except ImportError:
            print("  Memory usage: psutil not available")

        return True

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return False


def test_rl_integration():
    """Test integration with RL frameworks."""
    print("\n" + "="*60)
    print("RL FRAMEWORK INTEGRATION TEST")
    print("="*60)

    results = TestResults()

    # Test with stable-baselines3 (if available)
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
        from vangers_env import VangersEnv

        print("Testing stable-baselines3 integration...")

        env = VangersEnv(width=160, height=120)

        # Check environment compatibility
        check_env(env, warn=True)
        results.add_test("SB3 Environment Check", True)

        # Create and train model (briefly)
        model = PPO("MultiInputPolicy", env, verbose=0)
        model.learn(total_timesteps=100)
        results.add_test("SB3 Model Training", True, "100 timesteps")

        # Test trained model
        obs, info = env.reset()
        for _ in range(5):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()

        results.add_test("SB3 Model Prediction", True)

        env.close()

    except ImportError:
        # Treat missing stable-baselines3 as a skipped check rather than a failure.
        # Record as passed/skipped so the overall RL integration summary does not fail
        # when the optional dependency is not installed in the test environment.
        results.add_test("SB3 Integration (skipped)", True, "stable-baselines3 not available (skipped)")
        print("  stable-baselines3 not installed; skipping SB3 integration tests. Install with: pip install stable-baselines3")

    # Test with gymnasium registration (if possible)
    try:
        import gymnasium as gym
        from vangers_env import VangersEnv

        # Test that environment follows gymnasium API
        env = VangersEnv(width=160, height=120)

        # Check required methods
        required_methods = ['reset', 'step', 'close']
        methods_present = all(hasattr(env, method) for method in required_methods)
        results.add_test("Gymnasium API Compliance", methods_present)

        # Check spaces
        has_action_space = hasattr(env, 'action_space')
        has_obs_space = hasattr(env, 'observation_space')
        results.add_test("Gymnasium Spaces", has_action_space and has_obs_space)

        env.close()

    except ImportError:
        results.add_test("Gymnasium Integration", False, "gymnasium not available")

    success = results.print_summary()
    assert success, "One or more RL-integration checks failed"


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Vangers Gymnasium Interface Test")
    parser.add_argument('--single', action='store_true', help='Test single environment')
    parser.add_argument('--vectorized', action='store_true', help='Test vectorized environment')
    parser.add_argument('--episodes', type=int, default=3, help='Number of episodes to run')
    parser.add_argument('--steps', type=int, default=100, help='Max steps per episode')
    parser.add_argument('--envs', type=int, default=2, help='Number of environments for vectorized test')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    parser.add_argument('--rl-integration', action='store_true', help='Test RL framework integration')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')

    args = parser.parse_args()

    # Default to single environment test if no specific tests requested
    if not any([args.single, args.vectorized, args.benchmark, args.rl_integration, args.all]):
        args.single = True

    print("VANGERS GYMNASIUM INTERFACE TEST SUITE")
    print("=" * 80)

    # Track overall results
    all_passed = True

    try:
        # Run requested tests
        if args.single or args.all:
            passed = test_single_environment(args.episodes, args.steps, args.verbose)
            all_passed = all_passed and passed

        if args.vectorized or args.all:
            passed = test_vectorized_environment(args.envs, args.episodes, args.steps, args.verbose)
            all_passed = all_passed and passed

        if args.benchmark or args.all:
            passed = benchmark_gymnasium_performance(args.envs, args.steps * 10, args.verbose)
            all_passed = all_passed and passed

        if args.rl_integration or args.all:
            passed = test_rl_integration()
            all_passed = all_passed and passed

        # Final summary
        print("\n" + "=" * 80)
        if all_passed:
            print("🎉 ALL GYMNASIUM TESTS PASSED!")
            print("The Vangers gymnasium interface is working correctly.")
            return 0
        else:
            print("❌ SOME GYMNASIUM TESTS FAILED")
            print("Please check the output above for details.")
            return 1

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
