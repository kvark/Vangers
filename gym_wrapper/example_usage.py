#!/usr/bin/env python3
"""
Example usage of the Vangers Gymnasium Environment

This script demonstrates how to use the Vangers gym wrapper for:
1. Basic environment interaction
2. Random agent testing
3. Simple scripted agent
4. Integration with RL frameworks (when available)
"""

import numpy as np
import time
from vangers_env import VangersEnv

def test_basic_usage():
    """Test basic environment functionality."""
    print("=" * 60)
    print("BASIC ENVIRONMENT USAGE")
    print("=" * 60)

    # Create environment
    env = VangersEnv(width=640, height=480, render_mode="rgb_array")

    print(f"Action space: {env.action_space}")
    print(f"Observation space keys: {list(env.observation_space.keys())}")

    # Reset environment
    observation, info = env.reset(seed=42)
    print(f"Initial position: {info.get('player_position', 'N/A')}")
    print(f"Initial armor: {info.get('player_armor', 'N/A')}")

    # Take a few random actions
    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(f"Step {step+1}: action={action}, reward={reward:.3f}, "
              f"position={info.get('player_position', 'N/A')}")

        if terminated or truncated:
            print(f"Episode finished after {step+1} steps")
            break

    print(f"Total reward: {total_reward:.3f}")
    env.close()

def test_random_agent(episodes=3, max_steps=100):
    """Test with a random agent over multiple episodes."""
    print("\n" + "=" * 60)
    print("RANDOM AGENT TESTING")
    print("=" * 60)

    env = VangersEnv(width=320, height=240)

    episode_rewards = []
    episode_lengths = []

    for episode in range(episodes):
        print(f"\nEpisode {episode + 1}:")
        observation, info = env.reset(seed=episode)

        total_reward = 0
        steps = 0

        for step in range(max_steps):
            # Random action
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1

            # Print progress every 20 steps
            if step % 20 == 0:
                print(f"  Step {step}: reward={reward:.3f}, "
                      f"position={info.get('player_position', 'N/A')}")

            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        print(f"  Final reward: {total_reward:.3f}, steps: {steps}")

    print(f"\nRandom Agent Results:")
    print(f"  Average reward: {np.mean(episode_rewards):.3f}")
    print(f"  Average episode length: {np.mean(episode_lengths):.1f}")
    print(f"  Reward std: {np.std(episode_rewards):.3f}")

    env.close()

def test_scripted_agent(episodes=2, max_steps=100):
    """Test with a simple scripted agent."""
    print("\n" + "=" * 60)
    print("SCRIPTED AGENT TESTING")
    print("=" * 60)

    env = VangersEnv(width=320, height=240)

    def scripted_policy(observation, step):
        """Simple scripted policy: move forward most of the time, occasionally turn."""
        if step % 30 < 20:
            # Move forward
            return [1, 0, 0, 0, 0]  # forward, straight, no fire
        elif step % 30 < 25:
            # Turn left
            return [1, 1, 0, 0, 0]  # forward, left
        else:
            # Turn right
            return [1, 2, 0, 0, 0]  # forward, right

    for episode in range(episodes):
        print(f"\nEpisode {episode + 1}:")
        observation, info = env.reset(seed=episode + 100)

        total_reward = 0
        steps = 0

        for step in range(max_steps):
            # Use scripted policy
            action = scripted_policy(observation, step)
            observation, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1

            # Print progress every 25 steps
            if step % 25 == 0:
                pos = info.get('player_position', (0, 0, 0))
                print(f"  Step {step}: reward={reward:.3f}, pos=({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")

            if terminated or truncated:
                break

        print(f"  Final reward: {total_reward:.3f}, steps: {steps}")

    env.close()

def test_observation_details():
    """Examine the observation space in detail."""
    print("\n" + "=" * 60)
    print("OBSERVATION SPACE ANALYSIS")
    print("=" * 60)

    env = VangersEnv(width=320, height=240)
    observation, info = env.reset(seed=123)

    # Analyze screen observation
    screen = observation['screen']
    print(f"Screen observation:")
    print(f"  Shape: {screen.shape}")
    print(f"  Dtype: {screen.dtype}")
    print(f"  Min value: {screen.min()}")
    print(f"  Max value: {screen.max()}")
    print(f"  Mean value: {screen.mean():.2f}")

    # Analyze state vector
    state = observation['state']
    print(f"\nState vector:")
    print(f"  Shape: {state.shape}")
    print(f"  Dtype: {state.dtype}")
    print(f"  Values: {state[:10]}...")  # Show first 10 values

    # State vector components (based on implementation)
    print(f"\nState vector interpretation (first 15 components):")
    print(f"  Position: ({state[0]:.1f}, {state[1]:.1f}, {state[2]:.1f})")
    print(f"  Velocity: ({state[3]:.1f}, {state[4]:.1f}, {state[5]:.1f})")
    print(f"  Angles: ({state[6]:.3f}, {state[7]:.3f}, {state[8]:.3f})")
    print(f"  Armor: {state[9]:.1f}")
    print(f"  Energy: {state[10]:.1f}")
    print(f"  Speed: {state[11]:.1f}")
    print(f"  Max Speed: {state[12]:.1f}")
    print(f"  Alive: {bool(state[13])}")
    print(f"  On Ground: {bool(state[14])}")

    # Info dictionary
    print(f"\nInfo dictionary:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    env.close()

def test_action_effects():
    """Test the effects of different actions."""
    print("\n" + "=" * 60)
    print("ACTION EFFECTS TESTING")
    print("=" * 60)

    env = VangersEnv(width=320, height=240)

    # Test different action types
    test_actions = [
        ([0, 0, 0, 0, 0], "Stop"),
        ([1, 0, 0, 0, 0], "Forward"),
        ([2, 0, 0, 0, 0], "Backward"),
        ([1, 1, 0, 0, 0], "Forward + Left"),
        ([1, 2, 0, 0, 0], "Forward + Right"),
        ([1, 0, 1, 0, 0], "Forward + Fire"),
        ([1, 0, 0, 1, 0], "Forward + Special1"),
    ]

    for action, description in test_actions:
        print(f"\nTesting: {description}")
        observation, info = env.reset(seed=200)

        initial_pos = info.get('player_position', (0, 0, 0))
        initial_energy = info.get('player_energy', 100)

        # Apply action for 5 steps
        total_reward = 0
        for step in range(5):
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        final_pos = info.get('player_position', (0, 0, 0))
        final_energy = info.get('player_energy', 100)

        pos_change = (final_pos[0] - initial_pos[0],
                     final_pos[1] - initial_pos[1],
                     final_pos[2] - initial_pos[2])
        energy_change = final_energy - initial_energy

        print(f"  Position change: ({pos_change[0]:.2f}, {pos_change[1]:.2f}, {pos_change[2]:.2f})")
        print(f"  Energy change: {energy_change}")
        print(f"  Total reward: {total_reward:.3f}")

    env.close()

def demo_rl_integration():
    """Demonstrate integration with RL frameworks (if available)."""
    print("\n" + "=" * 60)
    print("RL FRAMEWORK INTEGRATION DEMO")
    print("=" * 60)

    # Try to use stable-baselines3 (if available)
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env

        print("✓ Stable-Baselines3 found!")

        # Create environment
        env = VangersEnv(width=160, height=120)

        print("Checking environment compatibility...")
        check_env(env, warn=True)
        print("✓ Environment is compatible with stable-baselines3!")

        # Create a simple PPO model
        print("Creating PPO model...")
        model = PPO("MultiInputPolicy", env, verbose=1)

        print("Training for 1000 timesteps (demo)...")
        model.learn(total_timesteps=1000)

        print("Testing trained model...")
        obs, info = env.reset()
        for i in range(10):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {i+1}: reward={reward:.3f}")
            if terminated or truncated:
                obs, info = env.reset()

        env.close()
        print("✓ RL integration test completed!")

    except ImportError:
        print("Stable-Baselines3 not available. Install with:")
        print("  pip install stable-baselines3")
        print("\nYou can still use the environment with other RL frameworks.")
        print("The environment follows standard Gymnasium API.")

def main():
    """Run all examples."""
    print("VANGERS GYMNASIUM ENVIRONMENT EXAMPLES")
    print("=" * 80)

    try:
        # Basic functionality
        test_basic_usage()

        # Random agent
        test_random_agent(episodes=3, max_steps=50)

        # Scripted agent
        test_scripted_agent(episodes=2, max_steps=50)

        # Observation analysis
        test_observation_details()

        # Action effects
        test_action_effects()

        # RL integration demo
        demo_rl_integration()

        print("\n" + "=" * 80)
        print("🎉 ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Install reinforcement learning libraries:")
        print("   pip install stable-baselines3 gymnasium")
        print("2. Create your own agent using the VangersEnv")
        print("3. Train and evaluate RL agents in the Vangers world")
        print("4. Experiment with different reward functions and observations")

    except Exception as e:
        print(f"\n❌ Error during examples: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
