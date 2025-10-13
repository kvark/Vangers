"""
Gym wrapper package initializer.

This module provides helper utilities for locating and loading the platform
shared library produced by the C++ build (libvangers_engine.so / .dylib / .dll).

Goal:
- When packaging the Python package, the built shared object should be placed
  inside this package (e.g. as `gym_wrapper/libvangers_engine.so`) so it is
  distributed as part of the wheel / sdist. This file provides helper functions
  to:
    * Discover an embedded/bundled library inside the installed package.
    * Respect environment overrides (VANGERS_ENGINE_LIB / VANGERS_ENGINE_PATH).
    * Fall back to system discovery if no bundled copy is available.
    * Load the library via ctypes and expose convenience loader function.

Notes for packaging / build system:
- Ensure that the CMake build copies the produced shared object into this
  package directory (e.g. <repo>/gym_wrapper/libvangers_engine.so) before
  running the Python packaging step. Also add that file to package_data in
  setup.py / pyproject.toml so it is included in the wheel.
- Example CMake step (conceptual):
    configure_file(<build_lib_path> ${CMAKE_CURRENT_SOURCE_DIR}/libvangers_engine.so COPYONLY)
  or install the library into the package dir as part of packaging flow.

This file intentionally does not automatically load the engine on import.
Call `load_engine()` explicitly to control when the .so is loaded.
"""

from __future__ import annotations

import ctypes
import os
import platform
import sys
from pathlib import Path
from typing import Optional

try:
    # Python >=3.9
    from importlib import resources
except Exception:
    # Python 3.8 fallback
    import importlib_resources as resources  # type: ignore

__all__ = [
    "get_default_library_name",
    "find_embedded_library",
    "find_engine_library",
    "load_engine",
    "ENGINE_LIB_PATH",
    "ENGINE",
    "__version__",
]

__version__ = "0.1.0"

# Global holders for loaded library/path
ENGINE = None  # type: Optional[ctypes.CDLL]
ENGINE_LIB_PATH: Optional[str] = None


def get_default_library_name() -> str:
    """Return platform-appropriate library filename basename."""
    plat = sys.platform
    if plat == "darwin":
        return "libvangers_engine.dylib"
    if plat.startswith("win"):
        return "vangers_engine.dll"
    # Default to Linux-style soname
    return "libvangers_engine.so"


def find_embedded_library(package: Optional[str] = None) -> Optional[str]:
    """
    Try to locate a library file that is bundled inside the installed package.

    This will check for the canonical filename in the package directory as well
    as a few common variants (sonames, versioned .so files).

    Returns:
        Absolute filesystem path to the embedded library if found, otherwise None.
    """
    libname = get_default_library_name()
    # Use package name if provided, otherwise current package
    pkg = package or __package__ or "gym_wrapper"

    # Prefer direct file on filesystem (this will be present for non-zipped installs)
    try:
        # resources.files returns Traversable object
        root = resources.files(pkg)
    except Exception:
        # If importlib.resources cannot locate package, fallback to package directory
        try:
            # If the package is a namespace under the repo, attempt to resolve via file system
            pkg_path = Path(__file__).resolve().parent
            candidates = list(pkg_path.glob("libvangers_engine*")) + list(
                pkg_path.glob("vangers_engine*")
            )
            for c in sorted(candidates):
                if c.is_file():
                    return str(c)
            return None
        except Exception:
            return None

    # If resource is a file-system path, we can inspect it
    try:
        pkg_path = root
        # Check exact name first
        candidate = pkg_path.joinpath(libname)
        if candidate.exists():
            with resources.as_file(candidate) as p:
                return str(p)

        # Try a few common patterns (libvangers_engine.so, libvangers_engine.so.1, etc.)
        patterns = ["libvangers_engine.*", "vangers_engine.*"]
        for pat in patterns:
            for match in sorted(pkg_path.glob(pat)):
                # resources.as_file requires a Traversable; joinpath returns Traversable on files()
                try:
                    with resources.as_file(match) as p:
                        if p.exists():
                            return str(p)
                except Exception:
                    # fallback: if match has a filesystem path attribute
                    try:
                        # Some Traversables support .locate() returning pathlib.Path-like object
                        path_candidate = Path(match.__str__())
                        if path_candidate.exists():
                            return str(path_candidate)
                    except Exception:
                        continue
    except Exception:
        # Best-effort: ignore errors and return None
        return None

    return None


def find_engine_library() -> Optional[str]:
    """
    Discover engine shared library using the following order:
    1) Environment override: VANGERS_ENGINE_LIB (file) or VANGERS_ENGINE_PATH (dir)
    2) Embedded library shipped inside the installed package
    3) System-wide discovery using ctypes.util.find_library
    4) Current working directory / common build locations

    Returns:
        Absolute path or soname (suitable for ctypes.CDLL), or None if not found.
    """
    # 1) Environment overrides
    env_file = os.environ.get("VANGERS_ENGINE_LIB")
    env_path = os.environ.get("VANGERS_ENGINE_PATH") or os.environ.get("VANGERS_GYM_LIB")
    if env_file:
        if Path(env_file).is_file():
            return str(Path(env_file).resolve())
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return str(p.resolve())
        if p.is_dir():
            # Search directory for matching files
            for pattern in ("libvangers_engine.*", "vangers_engine.*"):
                for candidate in sorted(p.glob(pattern)):
                    if candidate.is_file():
                        return str(candidate.resolve())

    # 2) Embedded library inside package
    embedded = find_embedded_library()
    if embedded:
        return embedded

    # 3) ctypes.util.find_library
    try:
        from ctypes.util import find_library

        name = find_library("vangers_engine")
        if name:
            return name
    except Exception:
        pass

    # 4) Common local locations (useful during development)
    local_candidates = [
        Path.cwd() / get_default_library_name(),
        Path.cwd() / "gym_wrapper" / get_default_library_name(),
        Path(__file__).resolve().parent / get_default_library_name(),
        Path(__file__).resolve().parent / "build" / get_default_library_name(),
        Path(__file__).resolve().parent / "lib" / get_default_library_name(),
    ]
    for c in local_candidates:
        try:
            if c.exists():
                return str(c.resolve())
        except Exception:
            continue

    return None


def load_engine(lib_path: Optional[str] = None, *, verbose: bool = False) -> ctypes.CDLL:
    """
    Load the engine shared library via ctypes and return the CDLL handle.

    Args:
        lib_path: explicit path or soname to load. If omitted, discovery is attempted.
        verbose: print discovery/loading details if true.

    Returns:
        ctypes.CDLL object

    Raises:
        FileNotFoundError if no library could be discovered or the load fails.
    """
    global ENGINE, ENGINE_LIB_PATH
    if ENGINE is not None:
        return ENGINE

    path_to_try = lib_path or find_engine_library()
    if verbose:
        print(f"[vangers.gym_wrapper] attempting to load engine library: {path_to_try!r}")

    if not path_to_try:
        raise FileNotFoundError(
            "Vangers engine shared library not found. "
            "Build it and place it inside the gym_wrapper package or set VANGERS_ENGINE_LIB."
        )

    try:
        lib = ctypes.CDLL(path_to_try)
    except OSError as e:
        # If path_to_try is a soname (returned by find_library) and OS can't resolve,
        # re-raise with additional guidance.
        raise OSError(
            f"Failed to load engine library at {path_to_try!r}: {e}. "
            "Ensure the file is compatible with your platform and Python interpreter."
        ) from e

    ENGINE = lib
    ENGINE_LIB_PATH = path_to_try
    if verbose:
        print(f"[vangers.gym_wrapper] loaded engine from: {ENGINE_LIB_PATH!r}")

    return lib
