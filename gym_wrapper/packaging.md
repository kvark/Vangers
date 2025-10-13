# Packaging and publishing the Vangers Gym Wrapper

This document describes the recommended steps to build, package and publish the `vangers-gym` Python package that embeds the native Vangers shared library (the engine built as a shared object / dynamic library). It covers local development steps, a small helper script included in the repository, tips for building wheels, and high-level guidance for CI-based multi-platform packaging.

Goals
- Produce a Python wheel that contains the platform shared library inside the Python package so the library is available after installation.
- Keep the build and packaging process reproducible and automatable (CI-friendly).
- Provide guidance for publishing platform-specific wheels (Linux, macOS, Windows).

Contents
1. Prerequisites
2. Local packaging (quick)
3. Packaging script (`build_package.py`)
4. Building wheels locally (recommended)
5. CI packaging overview (short)
6. Publishing to PyPI (notes)
7. Troubleshooting
8. Best practices and notes for maintainers

1 — Prerequisites
------------------
- Tools needed on the machine where you build the package:
  - CMake (>= 3.1 recommended; use a modern release if possible).
  - A C++ toolchain that supports C++17 (gcc/clang/MSVC).
  - Python 3.7+ (we use Python 3.12 in the examples but the package metadata supports >=3.7).
  - pip, setuptools, wheel and optionally `build` or `cibuildwheel` for wheel building. These are best installed into a virtual environment.

- Recommended local workflow:
  - Create and activate a virtualenv:
    - python3.12 -m venv .venv
    - source .venv/bin/activate   (or on Windows: .venv\Scripts\activate)
  - Upgrade packaging tools:
    - python -m pip install --upgrade pip setuptools wheel build

- Note: The Vangers C++ engine links to some system libraries (SDL2, pthread, jsoncpp). Ensure required system dependencies are installed for your target platform when building.

2 — Local packaging (quick)
----------------------------
This repository includes a helper `build_package.py` (in `Vangers/gym_wrapper/`) that wraps the common steps:

- Configure and build the C++ shared library.
- Copy the resulting shared library into the `gym_wrapper/` package directory (so it is included in the wheel).
- Build a wheel (and optionally an sdist).

Typical quick commands (from repo root):

1. Build & package (preferred):
   - python3.12 gym_wrapper/build_package.py --build-dir gym_wrapper/build --python python3.12 --outdir gym_wrapper/dist

2. If you have already built the C++ library manually and only want to package:
   - python3.12 gym_wrapper/build_package.py --no-build --python python3.12 --outdir gym_wrapper/dist

After a successful run you should find wheel(s) in `Vangers/gym_wrapper/dist/` or the directory passed with `--outdir`.

Important:
- The script tries to use `python -m build`. If that module is missing the script will attempt to `pip install --user build`. In some environments (e.g. restricted CI images) pip may not be available — in that case the helper falls back to `setup.py bdist_wheel`.

3 — Packaging script details
----------------------------
File: `Vangers/gym_wrapper/build_package.py`

Main features:
- Uses CMake to configure/build only the `vangers_engine` target by default (faster).
- Locates built artifacts that look like shared libraries:
  - Linux: `libvangers_engine.so`, `libvangers_engine.so.*`
  - macOS: `libvangers_engine.dylib`
  - Windows: `vangers_engine.dll`
- Copies only library-like artifacts into `Vangers/gym_wrapper/` so they become package data.
- Builds a wheel (and optionally sdist) using `python -m build` (preferred). Falls back to `setup.py`.

Useful flags:
- `--build-dir`: custom CMake build directory (default `build`).
- `--build-type`: `Release` or `Debug`.
- `--no-build`: skip CMake step and use existing build artifacts.
- `--outdir`: output directory for wheel/sdist.
- `--python`: python executable used to build the wheel (useful for building with a specific interpreter).
- `--skip-copy`: do not copy built libraries into the package (useful if you manage copy elsewhere).
- `--keep-existing-lib`: do not overwrite an existing embedded library file.

4 — Building wheels locally (recommended)
-----------------------------------------
For local wheel creation and testing:

1. Create a clean virtualenv and install the tools:
   - python3.12 -m venv .venv
   - source .venv/bin/activate
   - python -m pip install --upgrade pip setuptools wheel build

2. Run the packaging helper:
   - python3.12 gym_wrapper/build_package.py --build-dir gym_wrapper/build --python python3.12 --outdir gym_wrapper/dist

3. Verify the wheel:
   - python -m pip install --force-reinstall gym_wrapper/dist/vangers_gym-*.whl
   - Test that the package loads and the shared library can be found:
     - python -c "import gym_wrapper; print(gym_wrapper.find_engine_library()); lib=gym_wrapper.load_engine(verbose=True); print(lib)"

Notes:
- Building the C++ library produces platform-dependent binaries; a wheel built on Linux will only work on Linux unless you produce manylinux-compliant wheels.
- For reproducible manylinux-compatible Linux wheels, build using manylinux Docker images or use `cibuildwheel` in CI.

5 — CI packaging overview
-------------------------
For publishing platform wheels you should use CI to produce OS-appropriate wheels:

- Linux (manylinux):
  - Use `cibuildwheel` on a Linux runner (or manylinux Docker images).
  - Steps:
    - Build the C++ library for the target platform, copy the shared library into the Python package folder.
    - Run `cibuildwheel` to produce wheels for the configured Python ABIs. `cibuildwheel` will create manylinux-compliant wheels if used in a manylinux container.
    - Upload the resulting wheels to the release or to PyPI.

- macOS:
  - Build on macOS runners and use `cibuildwheel` (or `python -m build`) to create macOS wheels.
  - Ensure the dynamic library naming and code-signing requirements (if any) are handled appropriately for your release workflow.

- Windows:
  - Build the native DLL on Windows or use cross-building strategies.
  - Use `cibuildwheel` on Windows to produce `.whl` files.

High-level CI recipe:
1. Checkout repository.
2. Run CMake build for the vangers shared library.
3. Copy shared library into package dir (`gym_wrapper/`).
4. Run `cibuildwheel` or `python -m build` to create wheels/sdists.
5. Run tests (optional).
6. Upload artifacts to GitHub Releases or PyPI (use `twine`).

If you'd like, I can provide a starter GitHub Actions workflow using `cibuildwheel` to build Linux/macOS/Windows wheels.

6 — Publishing to PyPI
-----------------------
- For each platform you will likely generate different wheels. Publish them to PyPI using `twine` or keep them attached to a GitHub release.
- Typical publish steps:
  1. pip install --upgrade twine
  2. twine upload dist/*

- If you produce manylinux wheels, make sure they are compliant and test installing them in a fresh virtualenv.

7 — Troubleshooting
--------------------
Common problems and quick fixes:

- "Library not found" after installation:
  - Ensure the wheel actually contains the shared library under the installed package path.
  - Verify `gym_wrapper.find_engine_library()` picks up the installed library; check `VANGERS_ENGINE_LIB` env var to override if needed.

- Build fails due to missing system libraries (e.g. SDL2, jsoncpp):
  - Install the appropriate development packages for your OS (apt, brew, pacman, or system packages).
  - On Debian/Ubuntu: apt install libsdl2-dev libjsoncpp-dev build-essential cmake

- `python -m build` fails in CI:
  - Ensure `build` is installed in the build environment or use `pip` to install it before running.
  - On constrained runners without `pip`, fallback to `setup.py` may be necessary.

- Wheel install failing because of missing dependencies at import time:
  - If the shared library has runtime dependencies not bundled in the wheel (e.g., system libs), the dynamic loader may fail. Consider documenting the required system packages or bundling lightweight dependencies.

8 — Best practices and notes for maintainers
--------------------------------------------
- Do not commit large platform binaries into the git history. Prefer running the CMake build and copying into the package directory as part of your release pipeline.
- For reproducible builds across Linux distributions, use `cibuildwheel` or manylinux Docker images.
- Tag releases in the repository and attach built artifacts (wheels) to the release for easy distribution.
- Keep metadata in `pyproject.toml` (PEP 621) as done in this repo to reduce ambiguity between setup.py and PEP517 toolchains.
- Provide a small smoke-test in tests/ that attempts to import the package and calls `gym_wrapper.load_engine()` to validate the embedded library can be loaded.

Appendix — Common commands summary
----------------------------------
- Build & package (local):
  - python3.12 -m venv .venv
  - source .venv/bin/activate
  - python -m pip install --upgrade pip build wheel setuptools
  - python3.12 gym_wrapper/build_package.py --build-dir gym_wrapper/build --python python3.12 --outdir gym_wrapper/dist

- Build only (CMake):
  - mkdir -p gym_wrapper/build
  - cd gym_wrapper/build
  - cmake .. -DBUILD_SHARED_ENGINE=ON -DCMAKE_BUILD_TYPE=Release
  - cmake --build . --target vangers_engine -- -j$(nproc)

- Build CI (conceptual):
  - Checkout
  - Build native library (CMake)
  - Copy artifact into package
  - Run `cibuildwheel` to build wheels for many Pythons and platforms
  - Upload artifacts with `twine` or attach to a release

If you want, I can:
- Add a `packaging-ci.yml` GitHub Actions workflow that demonstrates building the C++ library and running `cibuildwheel` to produce platform wheels.
- Create a smoke-test script that validates the installed wheel loads the embedded shared library and can run a short env test.

Choose one and I’ll prepare it next.