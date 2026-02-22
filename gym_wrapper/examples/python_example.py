#!/usr/bin/env python3

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import sys
import os

DEFAULT_MECHOS_NAME = "OxidizeMonk"

# Ensure repo root is on sys.path so "import gym_wrapper" works when run from source.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(REPO_ROOT)
from gym_wrapper.vangers_env import VangersVectorizedEnv, make_vangers_env


class VectorizedBenchmark:
    """Benchmark suite for vectorized environment performance."""

    def __init__(self, num_envs: int = 4):
        self.num_envs = num_envs
        self.env = None
        self.results = {}

    def setup(self):
        """Setup benchmark environment."""
        print(f"Setting up vectorized environment with {self.num_envs} instances...")
        self.env = VangersVectorizedEnv(
            num_envs=self.num_envs,
            screen_width=320,  # Smaller for better performance
            screen_height=240,
            frame_skip=4,
            render_mode=None,  # Headless for benchmarking
            mechos_name=DEFAULT_MECHOS_NAME,
        )

    def teardown(self):
        """Cleanup benchmark environment."""
        if self.env:
            self.env.close()

    def benchmark_step_performance(self, num_steps: int = 1000) -> Dict[str, float]:
        """Benchmark stepping performance."""
        print(f"Benchmarking {num_steps} steps across {self.num_envs} environments...")

        # Reset environment
        obs, infos = self.env.reset()

        # Warm up
        for _ in range(10):
            actions = [self.env.action_space.sample() for _ in range(self.num_envs)]
            self.env.step(actions)

        # Actual benchmark
        start_time = time.time()
        total_rewards = np.zeros(self.num_envs)
        episode_counts = np.zeros(self.num_envs)

        for step in range(num_steps):
            # Random actions for all environments
            actions = [self.env.action_space.sample() for _ in range(self.num_envs)]

            obs, rewards, terms, truncs, infos = self.env.step(actions)
            total_rewards += rewards

            # Count episode completions
            for i, info in enumerate(infos):
                if 'episode' in info:
                    episode_counts[i] += 1

        end_time = time.time()
        elapsed = end_time - start_time

        results = {
            'total_time': elapsed,
            'steps_per_second': (num_steps * self.num_envs) / elapsed,
            'env_steps_per_second': num_steps / elapsed,
            'avg_episode_reward': np.mean(total_rewards),
            'total_episodes': np.sum(episode_counts),
            'episodes_per_minute': np.sum(episode_counts) / (elapsed / 60.0)
        }

        return results

    def benchmark_memory_usage(self) -> Dict[str, float]:
        """Benchmark memory usage."""
        import psutil

        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Reset and run some steps
        obs, infos = self.env.reset()

        for _ in range(100):
            actions = [self.env.action_space.sample() for _ in range(self.num_envs)]
            obs, rewards, terms, truncs, infos = self.env.step(actions)

        # Get final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB

        return {
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'memory_per_env_mb': (final_memory - initial_memory) / self.num_envs,
            'total_memory_overhead_mb': final_memory - initial_memory
        }

    def run_all_benchmarks(self, num_steps: int = 1000) -> Dict[str, Dict]:
        """Run all benchmarks."""
        results = {}

        # Performance benchmark
        print("\n=== Performance Benchmark ===")
        perf_results = self.benchmark_step_performance(num_steps)
        results['performance'] = perf_results

        print(f"Steps per second: {perf_results['steps_per_second']:.1f}")
        print(f"Environment steps per second: {perf_results['env_steps_per_second']:.1f}")
        print(f"Episodes per minute: {perf_results['episodes_per_minute']:.1f}")

        # Memory benchmark
        print("\n=== Memory Benchmark ===")
        memory_results = self.benchmark_memory_usage()
        results['memory'] = memory_results

        print(f"Memory per environment: {memory_results['memory_per_env_mb']:.1f} MB")
        print(f"Total memory overhead: {memory_results['total_memory_overhead_mb']:.1f} MB")

        return results


class SimpleVectorizedAgent:
    """Simple agent that works with vectorized environments."""

    def __init__(self, action_space, num_envs: int):
        self.action_space = action_space
        self.num_envs = num_envs
        self.step_count = 0

    def act(self, observations: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """Get actions for all environments."""
        self.step_count += 1

        actions = []
        for env_idx in range(self.num_envs):
            # Simple policy: mostly move forward, turn occasionally
            action = np.array([1, 0, 0, 0, 0])  # Forward, straight, no fire, no specials

            # Add some randomness
            if np.random.random() < 0.1:
                action[1] = np.random.choice([1, 2])  # Turn left or right

            if np.random.random() < 0.05:
                action[2] = 1  # Fire occasionally

            actions.append(action)

        return actions


class PPOAgent:
    """Simple PPO agent for demonstration (requires stable-baselines3)."""

    def __init__(self, env):
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.env_util import DummyVecEnv

            self.model = PPO(
                "MultiInputPolicy",
                env,
                verbose=1,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                device='auto'
            )
        except ImportError:
            print("stable-baselines3 not available. Install with: pip install stable-baselines3")
            raise

    def train(self, total_timesteps: int = 100000):
        """Train the PPO agent."""
        print(f"Training PPO agent for {total_timesteps} timesteps...")
        self.model.learn(total_timesteps=total_timesteps)

    def evaluate(self, env, num_episodes: int = 10):
        """Evaluate trained agent."""
        episode_rewards = []

        for episode in range(num_episodes):
            obs, infos = env.reset()
            episode_reward = 0
            done = False

            while not done:
                actions, _ = self.model.predict(obs, deterministic=True)
                obs, rewards, terms, truncs, infos = env.step(actions)
                episode_reward += np.sum(rewards)
                done = np.any(terms | truncs)

            episode_rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'episodes': episode_rewards
        }


def run_basic_example(num_envs: int = 4, num_steps: int = 1000):
    """Run basic vectorized environment example."""
    print(f"\n=== Basic Vectorized Environment Example ===")
    print(f"Environments: {num_envs}, Steps: {num_steps}")

    # Create vectorized environment
    env = VangersVectorizedEnv(
        num_envs=num_envs,
        screen_width=640,
        screen_height=480,
        frame_skip=4
    )

    # Create simple agent
    agent = SimpleVectorizedAgent(env.action_space, num_envs)

    try:
        # Reset environment
        observations, infos = env.reset()
        print(f"Observation shapes: screen={observations['screen'].shape}, state={observations['state'].shape}")

        # Run episodes
        episode_rewards = np.zeros(num_envs)
        episode_lengths = np.zeros(num_envs)
        completed_episodes = 0

        for step in range(num_steps):
            # Get actions from agent
            actions = agent.act(observations)

            # Step environment
            observations, rewards, terminated, truncated, infos = env.step(actions)

            # Track rewards and episodes
            episode_rewards += rewards
            episode_lengths += 1

            # Check for completed episodes
            for i, info in enumerate(infos):
                if 'episode' in info:
                    completed_episodes += 1
                    print(f"Episode completed in env {i}: reward={info['episode']['r']:.2f}, length={info['episode']['l']}")
                    episode_rewards[i] = 0
                    episode_lengths[i] = 0

            # Print progress
            if step % 100 == 0:
                print(f"Step {step}: Mean reward per step = {np.mean(rewards):.3f}")

        print(f"\nCompleted {completed_episodes} episodes")
        print(f"Current episode rewards: {episode_rewards}")

    finally:
        env.close()


def run_benchmark(num_envs_list: List[int] = [1, 2, 4, 8, 16]):
    """Run performance benchmarks across different numbers of environments."""
    print(f"\n=== Performance Benchmark ===")

    results = []

    for num_envs in num_envs_list:
        print(f"\nTesting with {num_envs} environments...")

        benchmark = VectorizedBenchmark(num_envs)
        try:
            benchmark.setup()
            bench_results = benchmark.run_all_benchmarks(num_steps=500)
            bench_results['num_envs'] = num_envs
            results.append(bench_results)
        finally:
            benchmark.teardown()

    # Print summary
    print(f"\n{'Envs':<6} {'Steps/sec':<12} {'Mem/env (MB)':<12} {'Episodes/min':<12}")
    print("-" * 50)

    for result in results:
        perf = result['performance']
        mem = result['memory']
        print(f"{result['num_envs']:<6} "
              f"{perf['steps_per_second']:<12.1f} "
              f"{mem['memory_per_env_mb']:<12.1f} "
              f"{perf['episodes_per_minute']:<12.1f}")

    return results


def run_training_example(algorithm: str = 'ppo', timesteps: int = 50000):
    """Run RL training example."""
    print(f"\n=== RL Training Example ({algorithm.upper()}) ===")

    # Create environment
    env = VangersVectorizedEnv(
        num_envs=4,
        screen_width=320,
        screen_height=240,
        frame_skip=8,  # Faster training
        render_mode=None
    )

    try:
        if algorithm.lower() == 'ppo':
            agent = PPOAgent(env)
            agent.train(total_timesteps=timesteps)

            # Evaluate trained agent
            print("\nEvaluating trained agent...")
            eval_results = agent.evaluate(env, num_episodes=5)
            print(f"Mean evaluation reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
        else:
            print(f"Algorithm {algorithm} not implemented")

    except ImportError as e:
        print(f"Training example requires additional dependencies: {e}")
    finally:
        env.close()


def run_determinism_test(num_envs: int = 2, num_steps: int = 100):
    """Test that environments are deterministic."""
    print(f"\n=== Determinism Test ===")

    # Create two identical environments
    env1 = VangersVectorizedEnv(num_envs=num_envs, screen_width=320, screen_height=240)
    env2 = VangersVectorizedEnv(num_envs=num_envs, screen_width=320, screen_height=240)

    try:
        # Reset both with same seed
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)

        # Check initial states match
        state_diff = np.abs(obs1['state'] - obs2['state']).max()
        print(f"Initial state difference: {state_diff}")

        # Run same actions
        differences = []
        for step in range(num_steps):
            # Use fixed actions for reproducibility
            actions = [np.array([1, 0, 0, 0, 0]) for _ in range(num_envs)]  # All forward

            obs1, rewards1, _, _, _ = env1.step(actions)
            obs2, rewards2, _, _, _ = env2.step(actions)

            state_diff = np.abs(obs1['state'] - obs2['state']).max()
            reward_diff = np.abs(np.array(rewards1) - np.array(rewards2)).max()

            differences.append(state_diff)

            if step % 20 == 0:
                print(f"Step {step}: State diff = {state_diff:.6f}, Reward diff = {reward_diff:.6f}")

        max_diff = max(differences)
        print(f"\nMaximum state difference over {num_steps} steps: {max_diff}")

        if max_diff < 1e-6:
            print("✓ Environments are deterministic")
        else:
            print("✗ Environments show non-deterministic behavior")

    finally:
        env1.close()
        env2.close()


def main():
    parser = argparse.ArgumentParser(description="Vectorized Vangers Environment Examples")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps to run")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    parser.add_argument("--train-agent", action="store_true", help="Train RL agent")
    parser.add_argument("--algorithm", type=str, default="ppo", choices=["ppo"], help="RL algorithm")
    parser.add_argument("--timesteps", type=int, default=50000, help="Training timesteps")
    parser.add_argument("--test-determinism", action="store_true", help="Test deterministic behavior")
    parser.add_argument("--all", action="store_true", help="Run all examples")

    args = parser.parse_args()

    print("Vectorized Vangers Environment Examples")
    print("=" * 50)

    try:
        if args.all:
            # Run all examples
            run_basic_example(args.num_envs, args.steps)
            run_benchmark([1, 2, 4, 8])
            run_determinism_test()
            # Skip training by default in --all mode due to time

        elif args.benchmark:
            # Run benchmarks with different env counts
            env_counts = [1, 2, 4, 8, 16] if args.num_envs == 4 else [args.num_envs]
            run_benchmark(env_counts)

        elif args.train_agent:
            # Run RL training
            run_training_example(args.algorithm, args.timesteps)

        elif args.test_determinism:
            # Test deterministic behavior
            run_determinism_test(args.num_envs, args.steps)

        else:
            # Run basic example
            run_basic_example(args.num_envs, args.steps)

        print("\n" + "=" * 50)
        print("Examples completed successfully!")

    except Exception as e:
        print(f"\nError running examples: {e}")
        print("\nMake sure Vangers engine library is built and available.")
        print("Build with: cmake -DBUILD_SHARED_ENGINE=ON ..")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
