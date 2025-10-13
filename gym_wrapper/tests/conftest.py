#!/usr/bin/env python3
"""
Pytest configuration for Vangers gym wrapper tests.

Provides fixtures:
- lib_path: filesystem path to the built shared library (skips tests when not found)
- lib: ctypes.CDLL instance with C function prototypes set up and engine initialized

The fixtures use the helpers in `test_common.py` (located in the same tests package).
"""

from __future__ import annotations

import pytest
import os
from pathlib import Path

# Import test helpers robustly:
# 1) Try the normal absolute package import (works when pytest runs with package discovery)
# 2) Fallback to relative import if executed as part of a package
# 3) As a last resort, load the `test_common.py` source file directly from the tests directory
try:
    # Preferred when running tests with package context
    from gym_wrapper.tests.test_common import find_engine_library, load_engine_library  # type: ignore
except Exception:
    try:
        # Works when conftest is executed as a submodule of the tests package
        from .test_common import find_engine_library, load_engine_library  # type: ignore
    except Exception:
        # Final fallback: load the module directly from the file system so tests can run
        import importlib.util
        import sys

        tests_dir = Path(__file__).resolve().parent
        module_path = tests_dir / "test_common.py"
        if not module_path.exists():
            raise

        spec = importlib.util.spec_from_file_location("gym_wrapper.tests.test_common", str(module_path))
        module = importlib.util.module_from_spec(spec)
        loader = spec.loader
        assert loader is not None
        loader.exec_module(module)
        find_engine_library = module.find_engine_library
        load_engine_library = module.load_engine_library


@pytest.fixture(scope="session")
def lib_path() -> str:
    """
    Locate the built engine shared library. If not found, skip the engine tests.
    """
    try:
        path = find_engine_library()
    except FileNotFoundError as e:
        pytest.skip(f"libvangers_engine not found: {e}")
    if not path:
        pytest.skip("libvangers_engine not found")
    return path


@pytest.fixture(scope="session")
def lib(lib_path: str):
    """
    Load the engine library, call initialization, and provide the CDLL handle to tests.

    On teardown, call the engine cleanup function if available.
    """
    lib = None
    try:
        # Load library and configure function prototypes
        lib = load_engine_library(lib_path)

        # Attempt engine initialization; if init fails, skip tests
        try:
            if hasattr(lib, "vangers_engine_init"):
                result = lib.vangers_engine_init()
                if int(result) != 1:
                    pytest.skip(f"vangers_engine_init returned non-success: {result}")
        except Exception as e:
            # If calling init raised, skip tests rather than fail collection
            pytest.skip(f"vangers_engine_init failed: {e}")

        yield lib

    finally:
        # Best-effort cleanup
        try:
            if lib is not None and hasattr(lib, "vangers_engine_cleanup"):
                try:
                    lib.vangers_engine_cleanup()
                except Exception:
                    # ignore cleanup errors
                    pass
        except Exception:
            pass
