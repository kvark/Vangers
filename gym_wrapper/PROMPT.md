We are writing a gym wrapper for the Vangers game. The wrapper will provide a standardized interface for interacting with the game, allowing for easy integration with reinforcement learning algorithms.
We are not writing shims or mocks, we are actually instantiating the game engine in the gym environment. The ultimate test is "train_and_render.py" script, which should be called with `python3.12`.
Check the code, make it all build, and make train_and_render to work properly.

Need this to build:
```bash
export CLUNK_ROOT=/x/Code/Vangers/external/clunk/install
export LD_LIBRARY_PATH=$CLUNK_ROOT/lib:$LD_LIBRARY_PATH
```
