#!/usr/bin/env python3
"""
Packaging helper for Vangers gym wrapper.

This script automates:
  1) Configuring & building the C++ shared library (vangers_engine) with CMake.
  2) Copying the produced shared library files into the Python package directory
     (so they will be included in the wheel as package_data).
  3) Building a Python wheel (and optionally sdist) for the gym_wrapper package.

Usage examples:
  # Basic build-and-package using current python interpreter:
  python3.12 build_package.py

  # Custom build dir, debug build, and produce wheel only (no sdist):
  python3.12 build_package.py --build-dir out/build --build-type Debug --no-sdist

  # Skip building the C++ shared lib (useful when library already present):
  python3.12 build_package.py --no-build

Notes:
 - This script expects to be placed in the gym_wrapper package directory:
     Vangers/gym_wrapper/build_package.py
 - It will configure & build from that same directory (CMake source dir).
 - The produced shared object(s) (libvangers_engine.* or vangers_engine.*)
   will be copied into the same package directory so they are included in the wheel.
 - The script uses `cmake` and `python -m build` (PEP517). Ensure these tools
   are available on the PATH.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# ---- Configuration defaults -------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent  # Vangers/gym_wrapper
DEFAULT_BUILD_DIR = SCRIPT_DIR / "build"
DEFAULT_BUILD_TYPE = "Release"
LIB_BASENAMES = ("libvangers_engine", "vangers_engine")
LIB_PATTERNS = (
    "libvangers_engine.*",
    "vangers_engine.*",
)

# ---- Utility functions -----------------------------------------------------


def run(cmd: List[str], cwd: Optional[Path] = None, env: Optional[dict] = None) -> None:
    """
    Run a command and stream its output. Raises CalledProcessError on failure.
    """
    logging.info("Running: %s (cwd=%s)", " ".join(cmd), str(cwd) if cwd else None)
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    assert process.stdout is not None
    try:
        for line in process.stdout:
            print(line.rstrip())
        rc = process.wait()
    finally:
        if process.stdout:
            process.stdout.close()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def find_built_libraries(build_dir: Path) -> List[Path]:
    """
    Find matching shared-library files under build_dir. Returns list of absolute paths.

    Only return files that look like shared libraries to avoid copying
    incidental build artifacts such as .pc files or object files into the
    Python package. This function accepts typical shared library name patterns
    including:
      - libvangers_engine.so
      - libvangers_engine.so.1.0.0
      - libvangers_engine.dylib
      - vangers_engine.dll
    """
    matches: List[Path] = []
    if not build_dir.exists():
        return matches

    # Shared library indicative substrings / extensions
    # We'll accept names ending with .so, containing .so., ending with .dylib or .dll
    def looks_like_shared_lib(p: Path) -> bool:
        name = p.name.lower()
        if name.endswith(".dylib") or name.endswith(".dll"):
            return True
        if name.endswith(".so"):
            return True
        if ".so." in name:  # versioned soname like libvangers_engine.so.1.0.0
            return True
        return False

    # Search top-level build dir and immediate subdirectories
    search_dirs = [build_dir, build_dir / "bin", build_dir / "lib", build_dir / "Release", build_dir / "Debug"]

    for d in search_dirs:
        if not d.exists():
            continue
        for pattern in LIB_PATTERNS:
            for p in sorted(d.glob(pattern)):
                if p.is_file() and looks_like_shared_lib(p):
                    matches.append(p.resolve())

    # Also perform a recursive search for anything matching the basename with extension
    for base in LIB_BASENAMES:
        for p in sorted(build_dir.rglob(f"{base}*")):
            if p.is_file() and looks_like_shared_lib(p):
                matches.append(p.resolve())

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for p in matches:
        sp = str(p)
        if sp not in seen:
            seen.add(sp)
            unique.append(p)
    return unique


def copy_libraries_to_package(lib_paths: List[Path], package_dir: Path) -> List[Path]:
    """
    Copy matched libraries into the python package directory.
    Returns list of destination paths.
    """
    dests: List[Path] = []
    package_dir.mkdir(parents=True, exist_ok=True)
    for lib in lib_paths:
        dest = package_dir / lib.name
        logging.info("Copying %s -> %s", lib, dest)
        shutil.copy2(lib, dest)
        # Ensure copied file is writable for packaging tools
        dest.chmod(0o755)
        dests.append(dest.resolve())
    return dests


def build_cmake(source_dir: Path, build_dir: Path, build_type: str, extra_cmake_args: List[str]) -> None:
    """
    Configure and build the vangers_engine target via CMake.
    """
    cmake_cmd = ["cmake", "-S", str(source_dir), "-B", str(build_dir), f"-DCMAKE_BUILD_TYPE={build_type}", "-DBUILD_SHARED_ENGINE=ON"]
    cmake_cmd.extend(extra_cmake_args)
    run(cmake_cmd)

    # Build the specific target (vangers_engine) for faster builds
    build_cmd = ["cmake", "--build", str(build_dir), "--config", build_type, "--target", "vangers_engine", "--", f"-j{os.cpu_count() or 2}"]
    run(build_cmd)


def build_python_package(python_exe: str, package_dir: Path, wheel_outdir: Path, sdist: bool = True) -> None:
    """
    Build wheel (and optional sdist).

    Preferred method is `python -m build`. If the `build` module is not
    available we attempt to install it via pip. If that still fails we
    fall back to the legacy setuptools `setup.py bdist_wheel` path after
    ensuring wheel/setuptools are available.
    """
    # 1) Try using python -m build directly (if available)
    try:
        logging.info("Checking for 'build' module by running: %s -m build --version", python_exe)
        run([python_exe, "-m", "build", "--version"], cwd=package_dir)
        cmd = [python_exe, "-m", "build", "--wheel", "--outdir", str(wheel_outdir)]
        if sdist:
            cmd.append("--sdist")
        logging.info("Building Python package with: %s", " ".join(cmd))
        run(cmd, cwd=package_dir)
        return
    except Exception:
        logging.warning("'build' module not available or failed. Attempting to install it via pip...")

    # 2) Try to install build module and retry
    try:
        run([python_exe, "-m", "pip", "install", "--user", "build"], cwd=package_dir)
        cmd = [python_exe, "-m", "build", "--wheel", "--outdir", str(wheel_outdir)]
        if sdist:
            cmd.append("--sdist")
        logging.info("Retrying: %s", " ".join(cmd))
        run(cmd, cwd=package_dir)
        return
    except Exception:
        logging.warning("Installing 'build' module failed or 'python -m build' still fails. Falling back to setup.py bdist_wheel.")

    # 3) Fallback: ensure wheel/setuptools and create a wheel via setup.py
    try:
        run([python_exe, "-m", "pip", "install", "--user", "wheel", "setuptools"], cwd=package_dir)
    except Exception:
        logging.warning("Failed to ensure wheel/setuptools are installed via pip; attempting setup.py anyway.")

    # Build wheel with setup.py
    logging.info("Building wheel via setup.py bdist_wheel")
    run([python_exe, "setup.py", "bdist_wheel", "-d", str(wheel_outdir)], cwd=package_dir)
    if sdist:
        logging.info("Also creating sdist via setup.py sdist")
        run([python_exe, "setup.py", "sdist", "-d", str(wheel_outdir)], cwd=package_dir)


# ---- Main CLI --------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build C++ engine and package gym_wrapper wheel")
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR, help="CMake build directory (default: %(default)s)")
    parser.add_argument("--build-type", choices=("Release", "Debug"), default=DEFAULT_BUILD_TYPE, help="CMake build type")
    parser.add_argument("--no-build", action="store_true", help="Skip the CMake build step (assume library already built)")
    parser.add_argument("--clean-build", action="store_true", help="Remove build directory before configuring")
    parser.add_argument("--extra-cmake-args", nargs="*", default=[], help="Extra args to pass to cmake configure step")
    parser.add_argument("--python", default=sys.executable, help="Python executable to build the wheel (default: current interpreter)")
    parser.add_argument("--outdir", type=Path, default=SCRIPT_DIR / "dist", help="Output directory for built wheels/sdists")
    parser.add_argument("--no-sdist", action="store_true", help="Do not produce sdist (only wheel)")
    parser.add_argument("--skip-copy", action="store_true", help="Do not copy built libraries into the package directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--keep-existing-lib", action="store_true", help="Do not overwrite an existing embedded library file in package dir")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    source_dir = SCRIPT_DIR  # the CMakeLists.txt for the wrapper lives here
    build_dir = args.build_dir.resolve()
    package_dir = SCRIPT_DIR  # we'll copy into the package directory (the same folder as this script)
    wheel_outdir = args.outdir.resolve()
    python_exe = args.python

    logging.info("source_dir=%s", source_dir)
    logging.info("build_dir=%s", build_dir)
    logging.info("package_dir=%s", package_dir)
    logging.info("wheel_outdir=%s", wheel_outdir)
    logging.info("python=%s", python_exe)

    try:
        # Optionally clean build dir
        if args.clean_build and build_dir.exists():
            logging.info("Removing build directory: %s", build_dir)
            shutil.rmtree(build_dir)

        # Build C++ library
        if not args.no_build:
            logging.info("Configuring and building C++ library with CMake")
            build_cmake(source_dir=source_dir, build_dir=build_dir, build_type=args.build_type, extra_cmake_args=args.extra_cmake_args)
        else:
            logging.info("Skipping C++ build (per --no-build)")

        # Find built libraries
        libs = find_built_libraries(build_dir)
        if not libs:
            logging.warning("No built library artifacts found in build directory: %s", build_dir)
        else:
            logging.info("Found built libraries: %s", ", ".join(str(p) for p in libs))

        # Copy the built libraries into the package directory
        if not args.skip_copy and libs:
            # Optionally avoid overwriting existing embedded files
            libs_to_copy = []
            for lib in libs:
                dest = package_dir / lib.name
                if args.keep_existing_lib and dest.exists():
                    logging.info("Keeping existing embedded library (skip overwrite): %s", dest)
                    continue
                libs_to_copy.append(lib)
            if libs_to_copy:
                copied = copy_libraries_to_package(libs_to_copy, package_dir)
                logging.info("Copied libraries into package: %s", ", ".join(str(p) for p in copied))
            else:
                logging.info("No libraries to copy (either none found or keep_existing_lib prevented overwrite).")
        else:
            if args.skip_copy:
                logging.info("Skipping copy of built libraries into package (per --skip-copy).")
            else:
                logging.info("No libraries found to copy into package.")

        # Build Python wheel (and optional sdist)
        wheel_outdir.mkdir(parents=True, exist_ok=True)
        build_python_package(python_exe=python_exe, package_dir=SCRIPT_DIR, wheel_outdir=wheel_outdir, sdist=not args.no_sdist)

        logging.info("Packaging complete. Artifacts are in: %s", wheel_outdir)
        return 0

    except subprocess.CalledProcessError as cpe:
        logging.error("Command failed: %s", cpe)
        return getattr(cpe, "returncode", 1)
    except Exception as e:
        logging.exception("Packaging failed: %s", e)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
