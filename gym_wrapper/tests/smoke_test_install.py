#!/usr/bin/env python3
"""
Smoke-test helper that installs a built wheel into a fresh virtualenv and runs a minimal runtime test.

This script is intended to be run from the repository root and will:

  1. Find a built wheel file (if not given explicitly).
  2. Create an isolated venv.
  3. Install the wheel into the venv (via pip).
  4. Run a minimal test script inside the venv:
       - import `gym_wrapper`
       - attempt to locate and load the embedded native engine
       - create a tiny `VangersVectorizedEnv` (1 env) and step once
  5. Report success/failure and optional logs.

Usage:
  # Automatic wheel discovery (searches common dist locations)
  python3 gym_wrapper/tests/smoke_test_install.py

  # Specify a wheel file explicitly
  python3 gym_wrapper/tests/smoke_test_install.py --wheel /path/to/vangers_gym-0.1.0-py3-none-any.whl

  # Keep the venv around for inspection (--keep-venv)
  python3 gym_wrapper/tests/smoke_test_install.py --keep-venv

Note:
- This is a smoke-test only. It is not a replacement for the project's unit tests or CI.
- The script attempts to be robust in environments where pip is present inside venvs.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
import venv
import os
import glob

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../Vangers
DEFAULT_DIST_DIRS = [
    PROJECT_ROOT / "gym_wrapper" / "dist",
    PROJECT_ROOT / "dist",
    PROJECT_ROOT / "gym_wrapper" / "dist_test",
    PROJECT_ROOT / "gym_wrapper" / "dist_test2",
    PROJECT_ROOT / "gym_wrapper" / "dist_test3",
]


def find_wheel_candidates(search_dirs=None):
    search_dirs = search_dirs or DEFAULT_DIST_DIRS
    candidates = []
    for d in search_dirs:
        try:
            for p in sorted(Path(d).glob("*.whl")):
                candidates.append(p)
        except Exception:
            continue
    # also try top-level wheel patterns
    for p in sorted(PROJECT_ROOT.glob("*.whl")):
        if p not in candidates:
            candidates.append(p)
    return candidates


def choose_wheel(provided: Path | None):
    if provided:
        p = Path(provided)
        if not p.exists():
            raise FileNotFoundError(f"Provided wheel not found: {p}")
        return p.resolve()
    candidates = find_wheel_candidates()
    if not candidates:
        raise FileNotFoundError(
            "No wheel found in expected dist directories. Build one first with the packaging helper "
            "(e.g. `python3 gym_wrapper/build_package.py`).\nSearched dirs: " + ", ".join(str(d) for d in DEFAULT_DIST_DIRS)
        )
    # prefer the most recent by name sort (usually best)
    return candidates[-1].resolve()


def make_venv(venv_dir: Path, python_exe: str | None = None):
    """
    Create a venv at venv_dir. If python_exe is given, we attempt to run that to
    create the venv using the venv module of the current interpreter (best-effort).
    Returns path to the venv Python executable.
    """
    # Use the stdlib venv builder (this uses the current running interpreter's venv module)
    builder = venv.EnvBuilder(with_pip=True)
    builder.create(venv_dir)
    if os.name == "nt":
        venv_python = venv_dir / "Scripts" / "python.exe"
        venv_pip = venv_dir / "Scripts" / "pip.exe"
    else:
        venv_python = venv_dir / "bin" / "python"
        venv_pip = venv_dir / "bin" / "pip"
    if not venv_python.exists():
        raise FileNotFoundError(f"Created venv but python not found at expected location: {venv_python}")
    return venv_dir.resolve(), venv_python.resolve(), venv_pip.resolve()


def run_in_subproc(cmd, cwd=None, env=None, check=True):
    print("+ " + " ".join(map(str, cmd)))
    res = subprocess.run(cmd, cwd=cwd, env=env)
    if check and res.returncode != 0:
        raise subprocess.CalledProcessError(res.returncode, cmd)
    return res.returncode


# ---------------------------------------------------------------------------
# Test script to run inside the venv
# ---------------------------------------------------------------------------

TEST_SCRIPT = textwrap.dedent(
    r"""
    import sys, traceback
    try:
        print("Python:", sys.version)
        # Try to import the package and discover the embedded lib
        import gym_wrapper
        print("gym_wrapper version (if present):", getattr(gym_wrapper, "__version__", "<no-version>"))
        lib_path = None
        try:
            lib_path = gym_wrapper.find_engine_library()
            print("find_engine_library ->", lib_path)
        except Exception as e:
            print("find_engine_library error:", e)
        try:
            lib = gym_wrapper.load_engine(verbose=True)
            print("Loaded engine lib handle:", lib)
        except Exception as e:
            print("load_engine failed:", e)
            raise

        # Minimal integration smoke test using the python env wrapper
        try:
            from gym_wrapper.vangers_env import VangersVectorizedEnv
            print("VangersVectorizedEnv found, creating a small env...")
            env = VangersVectorizedEnv(num_envs=1, screen_width=64, screen_height=48, frame_skip=1)
            obs, infos = env.reset()
            print("reset observation keys:", list(obs.keys()))
            # Sample a sensible action: zeros if action_space absent
            if getattr(env, "action_space", None) is not None and getattr(env.action_space, "sample", None):
                a = [env.action_space.sample()]
            else:
                a = [ [0,0,0,0,0] ]
            obs, rewards, terms, truncs, infos = env.step(a)
            print("step returned rewards:", rewards)
            env.close()
        except Exception as e:
            print("Environment smoke test failed:")
            traceback.print_exc()
            raise

        print("SMOKE TEST: OK")
        sys.exit(0)
    except Exception:
        traceback.print_exc()
        sys.exit(2)
    """
).lstrip()


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Smoke test installing a built wheel into a fresh venv and running a small runtime test.")
    ap.add_argument("--wheel", type=str, help="Path to the .whl file to install. If omitted the script searches dist directories.")
    ap.add_argument("--keep-venv", action="store_true", help="Do not remove the created venv directory (useful for debugging).")
    ap.add_argument("--venv-dir", type=str, help="Create venv at this path instead of a temporary dir.")
    ap.add_argument("--python", type=str, default=sys.executable, help="Python interpreter used to create the venv (default: same Python running this script).")
    ap.add_argument("--iterations", type=int, default=1, help="How many times to run the smoke test inside the created venv (default: 1).")
    args = ap.parse_args(argv)

    try:
        wheel_path = choose_wheel(Path(args.wheel) if args.wheel else None)
        print("Using wheel:", wheel_path)
    except Exception as e:
        print("ERROR locating wheel:", e)
        return 3

    tempdir = None
    venv_dir = Path(args.venv_dir).resolve() if args.venv_dir else None
    try:
        if venv_dir is None:
            tempdir = Path(tempfile.mkdtemp(prefix="vangers-smoke-"))
            venv_dir = tempdir / "venv"
        else:
            venv_dir.mkdir(parents=True, exist_ok=True)

        print("Creating venv at:", venv_dir)
        venv_dir, venv_python, venv_pip = make_venv(venv_dir, python_exe=args.python)
        print("Venv python:", venv_python)
        print("Venv pip:", venv_pip)

        # Upgrade pip in the venv to ensure modern wheel handling
        try:
            run_in_subproc([str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], check=True)
        except subprocess.CalledProcessError:
            print("Warning: failed to upgrade pip/setuptools/wheel inside venv; continuing with available pip")

        # Install the wheel
        print("Installing wheel into venv:", wheel_path)
        run_in_subproc([str(venv_python), "-m", "pip", "install", "--no-cache-dir", str(wheel_path)], check=True)

        # Write test script into temp file
        test_fd = None
        test_path = venv_dir / "smoke_test.py"
        with open(test_path, "w", encoding="utf-8") as f:
            f.write(TEST_SCRIPT)
        print("Written smoke test to:", test_path)

        # Run the test script inside the created venv
        rc = 0
        for i in range(args.iterations):
            print(f"Running smoke test iteration {i+1}/{args.iterations} ...")
            try:
                run_in_subproc([str(venv_python), str(test_path)], check=True)
            except subprocess.CalledProcessError as cpe:
                print("Smoke test failed (returncode):", cpe.returncode)
                rc = cpe.returncode or 1
                break

        if rc == 0:
            print("SMOKE TEST: SUCCESS")
        else:
            print("SMOKE TEST: FAILURE (see logs above)")

        return rc

    finally:
        if tempdir and not args.keep_venv:
            try:
                print("Cleaning up temporary venv dir:", tempdir)
                shutil.rmtree(tempdir)
            except Exception as e:
                print("Warning: failed to remove tempdir:", e)
        elif args.keep_venv:
            print("Kept venv dir for inspection:", venv_dir)


if __name__ == "__main__":
    raise SystemExit(main())
