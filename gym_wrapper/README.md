# Vangers Gym Wrapper

A Python Gym environment wrapper for the Vangers game engine, enabling reinforcement learning and AI training in the Vangers game world.

## Overview

This wrapper provides a standard OpenAI Gym interface to the Vangers C++ game engine, allowing researchers and developers to:
- Train reinforcement learning agents in the Vangers environment
- Create AI controllers for Vangers vehicles
- Conduct experiments with game AI
- Interface with the game programmatically

## Features

- **Standard Gym Interface**: Compatible with OpenAI Gym and modern RL frameworks
- **Multi-modal Observations**: RGB visual observations + structured game state
- **Flexible Action Space**: Control movement, steering, weapons, and special abilities
- **Network Communication**: Python client communicates with C++ game engine via sockets
- **Headless Mode**: Run without graphics for faster training
- **Real-time Control**: Low-latency action execution
- **Event Detection**: Collision detection, item collection, objective completion

## Installation

### Prerequisites

1. **Vangers Game**: You need a compiled Vangers executable with gym interface support
2. **Python Dependencies**:
   ```bash
   pip install gymnasium numpy opencv-python
   ```
3. **C++ Dependencies**:
   - jsoncpp (for JSON communication)
   - pthread (for threading)
   - SDL2 (for game engine)

### Building the Gym Interface

1. **Enable Gym Wrapper in Main CMakeLists.txt**:
   ```cmake
   option(ENABLE_GYM_WRAPPER "Enable Gym wrapper" ON)
   if(ENABLE_GYM_WRAPPER)
       add_subdirectory(gym_wrapper)
   endif()
   ```

2. **Build with Gym Support**:
   ```bash
   mkdir build
   cd build
   cmake -DENABLE_GYM_WRAPPER=ON ..
   make
   ```

3. **Install Python Package**:
   ```bash
   cd gym_wrapper
   pip install -e .
   ```

## Quick Start

### Basic Usage

```python
import gymnasium as gym
from vangers_env import VangersEnv

# Create environment
env = VangersEnv(
    game_path="./vangers",  # Path to Vangers executable
    render_mode="human",    # "human", "rgb_array", or None
    screen_width=800,
    screen_height=600
)

# Reset environment
observation, info = env.reset()

# Run episode
for step in range(1000):
    # Random action for demo
    action = env.action_space.sample()
    
    # Step environment
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Render if needed
    env.render()
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

### Running with Different Agents

```bash
# Random agent
python examples/python_example.py --agent random --episodes 5

# Scripted agent
python examples/python_example.py --agent simple --render human

# Human control (WASD keys)
python examples/python_example.py --agent human --render human
```

## Environment Details

### Action Space

The action space is `MultiDiscrete([3, 3, 2, 2, 2])` representing:

| Index | Action | Values |
|-------|--------|--------|
| 0 | Movement | 0=Stop, 1=Forward, 2=Backward |
| 1 | Steering | 0=Straight, 1=Left, 2=Right |
| 2 | Fire | 0=No Fire, 1=Fire |
| 3 | Special 1 | 0=No Special, 1=Use Special |
| 4 | Special 2 | 0=No Special, 1=Use Special |

### Observation Space

The observation is a dictionary with two components:

1. **Visual Observation** (`screen`): RGB image of shape `(height, width, 3)`
2. **State Vector** (`state`): 32-dimensional vector containing:
   - Position (x, y, z)
   - Velocity (vx, vy, vz)
   - Orientation (angle, pitch, roll)
   - Vehicle stats (armor, energy, speed, etc.)
   - Environment info (terrain, weather, etc.)

### Rewards

Default reward structure:
- **+0.1**: Staying alive per step
- **+speed × 0.01**: Moving forward bonus
- **-damage × 0.001**: Damage penalty  
- **+10.0**: Item collection bonus
- **-5.0**: Collision penalty

Customize rewards by subclassing `VangersEnv` and overriding `_calculate_reward()`.

## Game Integration

### Starting the Game with Gym Interface

The Vangers executable needs to be started with special flags:

```bash
./vangers --gym-interface --screen-width 800 --screen-height 600 --headless
```

Available flags:
- `--gym-interface`: Enable gym communication
- `--screen-width N`: Set screen width
- `--screen-height N`: Set screen height  
- `--headless`: Run without graphics window
- `--fps N`: Set target frame rate

### Network Communication

The gym interface uses two TCP sockets:
- **Control Socket** (port 7777): Receives actions from Python
- **Data Socket** (port 7778): Sends game state and frames to Python

## Advanced Usage

### Custom Environments

```python
class CustomVangersEnv(VangersEnv):
    def _calculate_reward(self, game_state):
        # Custom reward logic
        reward = 0.0
        if 'player' in game_state:
            player = game_state['player']
            # Add your reward calculations
        return reward
    
    def _is_episode_done(self, game_state):
        # Custom termination conditions
        return False  # Your logic here
```

### Integration with RL Libraries

#### Stable-Baselines3

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Create vectorized environment
env = make_vec_env(lambda: VangersEnv(), n_envs=4)

# Train PPO agent
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Save model
model.save("vangers_ppo")
```

#### Ray RLLib

```python
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO

ray.init()

config = {
    "env": VangersEnv,
    "env_config": {
        "render_mode": None,
        "screen_width": 640,
        "screen_height": 480
    },
    "num_workers": 4,
    "framework": "torch"
}

tune.run(PPO, config=config, stop={"training_iteration": 100})
```

## Troubleshooting

### Common Issues

1. **Game Won't Start**:
   - Check that Vangers executable exists and is executable
   - Ensure all game dependencies (SDL2, etc.) are installed
   - Try running game manually first without gym interface

2. **Connection Timeout**:
   - Check that ports 7777 and 7778 are available
   - Ensure firewall isn't blocking connections
   - Try different port numbers in both Python and C++ code

3. **Build Errors**:
   - Install jsoncpp development package: `sudo apt install libjsoncpp-dev`
   - Check that C++17 compiler is available
   - Verify all CMake dependencies are satisfied

4. **Python Import Errors**:
   - Install requirements: `pip install gymnasium numpy opencv-python`
   - Check that gym_wrapper is in Python path
   - Try installing in development mode: `pip install -e .`

### Debug Mode

Enable debug output:

```bash
# C++ debug build
cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_GYM_WRAPPER=ON ..

# Python debug
export VANGERS_GYM_DEBUG=1
python examples/python_example.py --verbose
```

## Architecture

```
┌─────────────────┐    TCP Socket    ┌─────────────────┐
│  Python Client  │◄────────────────►│  Vangers Game   │
│   (Gym Env)     │   Control/Data   │  (C++ Engine)   │
└─────────────────┘                  └─────────────────┘
        │                                      │
        ▼                                      ▼
┌─────────────────┐                  ┌─────────────────┐
│ RL Agent/Human  │                  │ Game Simulation │
│    Control      │                  │  Physics/AI     │
└─────────────────┘                  └─────────────────┘
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and test thoroughly
4. Submit a pull request

### Code Style

- **C++**: Follow existing Vangers codebase style
- **Python**: PEP 8 with 88-character line limit
- **Documentation**: Update README and add docstrings

## License

This gym wrapper follows the same license as the main Vangers project (GPLv3).

## Citation

If you use this gym wrapper in research, please cite:

```bibtex
@software{vangers_gym_wrapper,
  title={Vangers Gym Wrapper: Reinforcement Learning Environment},
  author={Your Name},
  year={2024},
  url={https://github.com/KranX/Vangers}
}
```

## Support

- **Issues**: Report bugs and feature requests on GitHub
- **Documentation**: See examples/ directory for more usage patterns  
- **Community**: Join discussions in GitHub Issues

---

*Happy training! 🚗🤖*