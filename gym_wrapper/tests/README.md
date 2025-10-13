# Vangers Gymnasium Wrapper - Test Suite

This directory contains the organized test suite for the Vangers gymnasium wrapper. The tests have been consolidated to eliminate redundancy and provide comprehensive coverage of all functionality.

## Test Structure

### Core Test Files

- **`test_common.py`** - Shared utilities, C struct definitions, and helper functions
- **`test_engine.py`** - Consolidated engine-level tests (C library functionality)
- **`test_gymnasium.py`** - High-level gymnasium interface tests
- **`test_minimal.py`** - Basic C library functionality tests (legacy)

### Shared Components (`test_common.py`)

All tests now use shared components to eliminate redundancy:

- **C Structures**: `Vector3`, `PlayerState`, `GameEvent` (defined once, used everywhere)
- **Library Loading**: `find_engine_library()`, `load_engine_library()` 
- **Test Instance Wrapper**: `TestInstance` class for common operations
- **Results Collection**: `TestResults` class for consistent reporting
- **Utilities**: Helper functions for benchmarking and state printing

## Test Categories

### 1. Engine Tests (`test_engine.py`)

Tests the low-level C engine functionality:

```bash
# Basic functionality (default)
python test_engine.py --basic

# Multiple instance management
python test_engine.py --multiple --instances 5

# Performance benchmarking
python test_engine.py --performance --steps 1000

# All engine tests
python test_engine.py --all --verbose
```

**Test Coverage:**
- ✅ Engine initialization and cleanup
- ✅ Instance creation and destruction
- ✅ State queries (player state, frame buffer, events)
- ✅ Action setting and simulation stepping
- ✅ Multiple instance isolation
- ✅ Concurrent access safety
- ✅ Performance benchmarks

### 2. Gymnasium Tests (`test_gymnasium.py`)

Tests the high-level Python gymnasium interface:

```bash
# Single environment (default)
python test_gymnasium.py --single

# Vectorized environment
python test_gymnasium.py --vectorized --envs 4

# Performance benchmark
python test_gymnasium.py --benchmark

# RL framework integration
python test_gymnasium.py --rl-integration

# All gymnasium tests
python test_gymnasium.py --all --verbose
```

**Test Coverage:**
- ✅ VangersEnv single environment
- ✅ VangersVectorizedEnv multi-environment
- ✅ Action/observation spaces
- ✅ Episode management and reset
- ✅ Gymnasium API compliance
- ✅ Performance benchmarks
- ✅ Integration with stable-baselines3 (if available)

## Command Line Options

Both test scripts support consistent command-line options:

### Common Options
- `--verbose, -v` - Enable detailed output
- `--help` - Show help message
- `--all` - Run all available tests

### Engine Test Options
- `--basic` - Basic functionality (default)
- `--multiple` - Multi-instance tests
- `--performance` - Performance benchmarks
- `--instances N` - Number of instances for multi-instance tests (default: 3)
- `--steps N` - Number of steps per test (default: 50)

### Gymnasium Test Options  
- `--single` - Single environment tests (default)
- `--vectorized` - Vectorized environment tests
- `--benchmark` - Performance benchmarks
- `--rl-integration` - Test RL framework integration
- `--episodes N` - Number of episodes (default: 3)
- `--steps N` - Max steps per episode (default: 100)
- `--envs N` - Number of environments for vectorized tests (default: 2)

## Quick Test Commands

### Basic Smoke Test
```bash
# Test everything works
python test_engine.py --basic
python test_gymnasium.py --single --episodes 1 --steps 10
```

### Full Test Suite
```bash
# Comprehensive testing
python test_engine.py --all --verbose
python test_gymnasium.py --all --verbose
```

### Performance Testing
```bash
# Benchmark performance
python test_engine.py --performance --instances 4 --steps 1000
python test_gymnasium.py --benchmark --envs 4
```

### Development Testing
```bash
# Quick validation during development
python test_engine.py --basic --verbose
python test_gymnasium.py --single --episodes 2 --steps 20 --verbose
```

## Test Results Interpretation

### Expected Results

**Engine Tests:**
- All basic functionality tests should **PASS**
- Multi-instance tests should **PASS** (state isolation, concurrent access)
- Performance should achieve >1000 steps/sec on modern hardware

**Gymnasium Tests:**
- Single environment tests should **PASS** 
- Vectorized environment tests should **PASS**
- Action/observation spaces should be valid gymnasium types
- Episodes should complete with reasonable rewards (>0)

### Known Issues

1. **"State Change After Action" may fail** in engine tests - this is expected behavior for the minimal C API implementation
2. **RL framework integration** requires optional dependencies (`pip install stable-baselines3`)
3. **Memory usage** tests require `psutil` package

## Dependencies

### Required
- Python 3.12+
- numpy
- Built `libvangers_engine.so` (copy to `tests/` directory)

### Optional (for full testing)
- gymnasium - for gymnasium API compliance tests
- stable-baselines3 - for RL integration tests  
- psutil - for memory usage analysis

## Troubleshooting

### Library Not Found
```
❌ Library not found: Could not find libvangers_engine.so
```
**Solution:** Copy the built library to the tests directory:
```bash
cp ../build/libvangers_engine.so ./
```

### Engine Initialization Failed
```
❌ Engine initialization failed: 0
```
**Solution:** Check that the library was built correctly and all dependencies are available.

### Import Errors
```
❌ ImportError: cannot import name 'VangersEnv'
```
**Solution:** Ensure you're running tests from the `tests/` directory and the parent `vangers_env.py` is accessible.

## Test Organization Benefits

### ✅ Eliminated Redundancy
- Single definition of C structures (`Vector3`, `PlayerState`, `GameEvent`)
- Shared library loading and initialization code
- Common test utilities and result reporting

### ✅ Consolidated Functionality
- `test_multiple_instances.py` → `test_engine.py --multiple`
- `test_vectorized_env.py` → `test_gymnasium.py --vectorized`
- Multiple test modes in single files via command-line options

### ✅ Improved Maintainability
- Changes to test infrastructure only need to be made in one place
- Consistent test patterns and reporting
- Clear separation of concerns (engine vs. gymnasium level)

### ✅ Better User Experience
- Simple command-line interface
- Comprehensive help messages
- Consistent output formatting
- Quick smoke tests and full test suites

---

*The test suite provides comprehensive validation of the Vangers gymnasium wrapper at both the C engine level and Python gymnasium interface level.*