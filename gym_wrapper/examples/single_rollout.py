#!/usr/bin/env python3
"""
Single Rollout Example for Vangers Gym Wrapper

This script performs a single episode rollout using random actions
and renders the gameplay. It demonstrates the raw environment interface
without any neural networks or RL frameworks.
"""

import sys
import os
import time
import argparse

# Ensure we can import the vangers_env module from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from vangers_env import VangersEnv
except ImportError:
    # Fallback if running from a different location or package not installed
    try:
        sys.path.append(os.path.join(os.getcwd(), 'gym_wrapper'))
        from vangers_env import VangersEnv
    except ImportError:
        print("Error: Could not import vangers_env. Make sure you are running this script")
        print("from the examples directory or have installed the package.")
        sys.exit(1)


def run_rollout(width=640, height=480, fps=30):
    print(f"Initializing Vangers Environment ({width}x{height})...")

    # Initialize the environment
    # We use render_mode="human" to attempt to open a window using OpenCV
    try:
        env = VangersEnv(
            width=width,
            height=height,
            render_mode="human"
        )
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease build the Vangers engine shared library first.")
        print("See README.md for build instructions.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError initializing environment: {e}")
        sys.exit(1)

    # Reset environment to start state
    print("Resetting environment...")
    obs, info = env.reset(seed=int(time.time()))

    print("Starting random rollout...")
    print("Controls: Randomly sampled from action space")
    print("Press Ctrl+C to stop early")

    done = False
    truncated = False
    step = 0
    total_reward = 0.0
    start_time = time.time()

    try:
        while not (done or truncated):
            # Render the current frame
            # In 'human' mode, this tries to show a cv2 window
            # If cv2 is not installed, it might just return the frame buffer
            env.render()

            # Sample a random action
            # The action space is MultiDiscrete: [move, turn, fire, special1, special2]
            action = env.action_space.sample()

            # Step the environment
            obs, reward, done, truncated, info = env.step(action)

            total_reward += reward
            step += 1

            # Log progress
            if step % 50 == 0:
                print(f"Step {step}: Reward={reward:.4f} Total={total_reward:.4f} Alive={info.get('player_alive', '?')}")

            # Maintain a reasonable frame rate
            time.sleep(1.0 / fps)

    except KeyboardInterrupt:
        print("\nRollout interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
    finally:
        end_time = time.time()
        duration = end_time - start_time
        print("\n" + "="*40)
        print(f"Episode Finished")
        print(f"Total Steps: {step}")
        print(f"Total Reward: {total_reward:.4f}")
        print(f"Duration: {duration:.2f}s")
        if duration > 0:
            print(f"FPS: {step/duration:.2f}")
        print("="*40)

        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vangers Single Rollout Example")
    parser.add_argument("--width", type=int, default=640, help="Screen width")
    parser.add_argument("--height", type=int, default=480, help="Screen height")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS limit")

    args = parser.parse_args()

    run_rollout(width=args.width, height=args.height, fps=args.fps)
