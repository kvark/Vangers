# Vangers Gym Wrapper

A Python-friendly, in-process Gym wrapper for the Vangers C++ game engine.
This package embeds a thin ctypes-based loader that talks to the engine shared library (libvangers_engine.*) and exposes a Gym-compatible API for training and evaluation.

Key goals:
- Run the engine in-process (no TCP sockets) for low-latency, deterministic control.
- Provide multi-modal observations (RGB frame + compact state vector).
- Integrate cleanly with modern RL libraries (stable‑baselines3, RLlib, etc.).
- Offer examples and tooling for training with a live preview.

---

## Quick snapshot

- Package: `gym_wrapper`
- Main environment classes:
  - `VangersEnv` — single-environment Gym wrapper.
  - `VangersVectorizedEnv` — simple vectorized wrapper that runs multiple instances in the same process.
- Example: `examples/train_and_render.py` — trains PPO with a CNN+MLP and shows a live preview of the current policy, running the engine at accelerated time scale (default 20x).

---

## Why in-process (no sockets)

Previous approaches used a TCP socket bridge between Python and the game binary. That adds IPC overhead, synchronization complexity, and packaging friction.

This wrapper runs the engine as a shared library (ctypes.CDLL) inside the Python process. Advantages:
- Minimal latency between policy action and engine response.
- Deterministic execution and easy time-control for reproducible training.
- Simpler packaging: the built `.so` / `.dylib` / `.dll` can be bundled with the Python package or referenced via environment variables.

---

## Installation & build (developer workflow)

Prereqs:
- C++ toolchain (CMake, compiler that supports C++17)
- Python 3.10+ recommended
- Optional (for examples): `stable-baselines3`, `torch`, `opencv-python`, `gymnasium`

1. Build the engine shared library (from repo root):

```bash
mkdir -p build
cd build
cmake -DBUILD_SHARED_ENGINE=ON ..
cmake --build . -j
```

This will produce the engine shared object (e.g. `libvangers_engine.so`) in your build tree. The python wrapper looks for the engine in a few places automatically.

2. Make the library visible to the Python package:

- Preferred: copy or configure your packaging to place the built `libvangers_engine.*` into the `gym_wrapper` package directory (so it becomes part of the wheel).
- Alternative: set an environment variable pointing to the built library:

```bash
export VANGERS_ENGINE_LIB=/path/to/libvangers_engine.so
# or
export VANGERS_ENGINE_PATH=/directory/containing/libvangers_engine.so
```

3. Install the Python package (development mode):

From `Vangers/gym_wrapper`:

```bash
pip install -e .
```

---

## Runtime discovery (how the Python code finds the engine)

The wrapper respects:
- `VANGERS_ENGINE_LIB` — full path to a library file
- `VANGERS_ENGINE_PATH` — a directory to search for `libvangers_engine.*`
- Bundled library inside the installed package (if available)
- `ctypes.util.find_library` fallbacks and several well-known locations near the python module

Call `from gym_wrapper import load_engine` or use `VangersEngineLib()`/`VangersEnv` which will try discovery automatically.

---

## Quick example: train with live preview (recommended)

The repository contains a ready-to-run example which trains a PPO agent and shows a live preview window of the policy in action. The example uses an in-memory inference proxy so the preview does not block training.

Run from the package root:

```bash
# Run training with preview (in-memory renderer; 20x engine speed, preview at 30 FPS)
python3.12 Vangers/gym_wrapper/examples/train_and_render.py \
  --timesteps 50000 \
  --save-interval 5000 \
  --width 160 --height 120 \
  --render-width 320 --render-height 240 \
  --play-speed 20.0 \
  --preview-fps 30
```

If you installed the package (and `gym_wrapper` is importable):

```bash
python3.12 -m gym_wrapper.examples.train_and_render \
  --timesteps 50000 --save-interval 5000
```

Flags of interest:
- `--timesteps` — total training timesteps
- `--save-interval` — how often to save checkpoints (helpful for longer experiments)
- `--play-speed` — engine time scale for preview (20.0 => 20× real-time)
- `--preview-fps` — display FPS for the preview window
- `--no-render` — run training without preview
- `--renderer-separate-process` — legacy option to use disk checkpoints for preview (default behavior uses in-memory snapshotter)
- `--snapshot-interval` — how frequently the snapshotter copies weights to the inference proxy (seconds)

---

## Training notes and architecture

Training flow (in-process):
1. Python loads `libvangers_engine` with ctypes.
2. Create one or more `VangersInstance` objects inside the same process via engine C API wrappers.
3. Interact with each instance by:
   - setting actions (deterministically)
   - stepping the engine
   - querying `player_state`, `frame_buffer`, and events
4. The example uses:
   - `VangersVectorizedEnv` for parallelized training
   - PPO with a `MultiInputPolicy` (screen + state)
   - A combined CNN (for images) + MLP (for state) features extractor

Preview architecture (non-blocking):
- The training model is the source of truth.
- A snapshotter thread periodically clones the training model weights into a separate inference model (lightweight proxy).
- The preview renderer uses that inference model to predict actions and display frames. This avoids contention with training and keeps the UI smooth.
- Debug overlay shows training throughput (timesteps/sec), snapshot age, episode reward, position, armor, energy, and preview FPS.

---

## Action and observation spaces

Action:
- `MultiDiscrete([3, 3, 2, 2, 2])`:
  - Movement: 0=Stop, 1=Forward, 2=Backward
  - Steering: 0=Straight, 1=Left, 2=Right
  - Fire: 0/1
  - Special1: 0/1
  - Special2: 0/1

Observation:
- A Python `dict` containing:
  - `screen`: RGB image, `np.uint8`, shape `(height, width, 3)`
  - `state`: compact `np.float32` vector (position, velocity, orientation, armor, energy, speed, etc.)

The example provides a CNN+MLP extractor that processes both modalities and concatenates features for the policy network.

---

## Example usage inside Python

Minimal usage:

```python
from vangers_env import VangersEnv

env = VangersEnv(width=160, height=120)
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

To train with SB3 (conceptual):

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from vangers_env import VangersEnv

env = DummyVecEnv([lambda: VangersEnv(width=160, height=120)])
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

For the combined CNN+MLP extractor and live preview, use the included example (`examples/train_and_render.py`) which wires everything together.

---

## Troubleshooting

- Library not found:
  - Set `VANGERS_ENGINE_LIB` to the full path of your built `libvangers_engine.*`.
  - Or place the built library inside the `gym_wrapper` package directory prior to packaging/install.

- Preview window not opening:
  - Ensure `opencv-python` is installed.
  - If running headless, use `--no-render`.

- Performance:
  - In-process engine is typically faster than IPC approaches. If training is CPU-bound, tune `num_envs`, frame-skip, and engine physics sub-steps.

- Determinism:
  - The engine wrapper provides deterministic stepping; use `VangersInstance` controls (time scale, substeps) to control reproducibility.

---

## Development & Contribution

- Add tests under `gym_wrapper/tests`.
- Run the test suite from repo root:

```bash
pytest -q
```

- When packaging, ensure the engine shared object is included in `package_data` or installed in a known location, or use the environment variables described above.

---

## Example commands (copy/paste)

Train with preview (recommended):

```bash
python3.12 Vangers/gym_wrapper/examples/train_and_render.py \
  --timesteps 50000 \
  --save-interval 5000 \
  --width 160 --height 120 \
  --render-width 320 --render-height 240 \
  --play-speed 20.0 \
  --preview-fps 30
```

Install & run example after `pip install -e .`:

```bash
python3.12 -m gym_wrapper.examples.train_and_render --timesteps 50000
```

---

## License & Citation

This wrapper follows the Vangers project's license (see top-level repo). If you use it in research, please cite the project repository.

---

If you want, I can:
- Add a small README section that demonstrates how to package the shared library inside the wheel (CMake/packaging snippet).
- Provide a short troubleshooting checklist for common build/linker issues on Linux/macOS/Windows.
