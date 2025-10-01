# Simulation System for Reinforcement Learning Experiments

This directory contains a comprehensive and professional simulation system for running and analyzing trained reinforcement learning agents. The system has been refactored to eliminate code duplication, improve maintainability, and provide a clean interface for different simulation scenarios.

## Features

- **Modular Architecture**: Clean separation of concerns with specialized components
- **Command-line Interface**: Flexible parameter configuration via command-line arguments
- **Professional Code Structure**: Well-documented, typed, and maintainable code
- **Comprehensive Logging**: Detailed action tracking and state logging
- **Video Recording**: High-quality video output with HUD overlays
- **Robust Error Handling**: Graceful handling of edge cases and failures
- **Resource Management**: Proper cleanup and timeout handling

## Quick Start

### Basic Usage

```bash
# Run a basic simulation
python experimental_simulation_new.py simple_kitchen_circular 2 v3.1 1 competition training_001 50

# With video disabled
python experimental_simulation_new.py simple_kitchen_circular 2 v3.1 1 competition training_001 50 --enable-video false

# Custom duration and settings
python experimental_simulation_new.py simple_kitchen_circular 2 v3.1 1 competition training_001 50 --duration 300 --tick-rate 30
```

### Advanced Usage

```bash
# Full parameter specification
python experimental_simulation_new.py \
    simple_kitchen_circular \     # map name
    2 \                          # number of agents
    v3.1 \                       # intent version
    1 \                          # cooperative (1) or competitive (0)
    competition \                # game version
    training_001 \               # training ID
    50 \                         # checkpoint number
    --enable-video true \        # enable video recording
    --cluster cuenca \           # cluster type
    --duration 180 \             # duration in seconds
    --tick-rate 24 \             # engine tick rate
    --video-fps 24               # video frame rate
```

## Command-line Parameters

### Required Arguments

1. **`map_name`**: Map identifier (e.g., `simple_kitchen_circular`)
2. **`num_agents`**: Number of agents in the simulation (e.g., `2`)
3. **`intent_version`**: Intent version identifier (e.g., `v3.1`)
4. **`cooperative`**: Cooperative mode flag (`1` for cooperative, `0` for competitive)
5. **`game_version`**: Game version identifier (e.g., `competition`)
6. **`training_id`**: Training session identifier (e.g., `training_001`)
7. **`checkpoint_number`**: Checkpoint number to load (e.g., `50`)

### Optional Arguments

- **`--enable-video`**: Enable video recording (`true`/`false`) [default: `true`]
- **`--cluster`**: Cluster type (`brigit`, `cuenca`, `local`) [default: `cuenca`]
- **`--duration`**: Simulation duration in seconds [default: `180`]
- **`--tick-rate`**: Engine tick rate (FPS) [default: `24`]
- **`--video-fps`**: Video recording frame rate [default: `24`]

## Architecture

### Core Components

1. **`utils.py`**: Central utility module containing:
   - `SimulationConfig`: Configuration management
   - `PathManager`: Path and directory management
   - `ControllerManager`: RL controller initialization
   - `RayManager`: Ray cluster management
   - `DataLogger`: State and action logging
   - `ActionTracker`: Detailed action tracking with timing
   - `VideoRecorder`: Video recording with HUD overlays
   - `GameManager`: Game instance creation and setup
   - `SimulationRunner`: Main simulation execution engine

2. **Main Script**:
   - `experimental_simulation_new.py`: Refactored simulation runner

### Directory Structure

```
spoiled_broth/simulations/
├── __init__.py          # Module initialization
├── utils.py             # Core utilities and shared functions
└── README.md            # This documentation

Root directory:
└── experimental_simulation_new.py  # Main simulation script
```

## Key Improvements

### 1. Code Reduction and Organization
- **Before**: Single monolithic file with ~700 lines
- **After**: Modular structure with clear separation of concerns
- **Eliminated**: Code duplication and complex nested logic
- **Added**: Professional class-based architecture

### 2. Enhanced Maintainability
- **Type Hints**: Comprehensive type annotations for better code clarity
- **Documentation**: Detailed docstrings for all components
- **Error Handling**: Robust error handling with informative messages
- **Resource Management**: Proper cleanup and timeout handling

### 3. Improved Flexibility
- **Command-line Interface**: Easy parameter modification without code changes
- **Configurable Paths**: Support for different cluster environments
- **Modular Components**: Easy to extend or modify individual features
- **Professional Logging**: Structured logging with progress reporting

### 4. Better Resource Management
- **Timeout Handling**: Prevents hanging during engine shutdown
- **Memory Management**: Proper resource cleanup and garbage collection
- **Thread Safety**: Safe handling of concurrent operations
- **Graceful Failures**: Robust error recovery and cleanup

## Output Files

The simulation generates a dedicated directory for each run with all associated files:

### Directory Structure
```
simulations/
└── simulation_{YYYY_MM_DD-HH_MM_SS}/
    ├── config.txt                                    # Simulation configuration
    ├── simulation_log_checkpoint_{N}_{timestamp}.csv # State log
    ├── actions_checkpoint_{N}_{timestamp}.csv        # Action log
    └── offline_recording_{timestamp}.mp4             # Video recording (if enabled)
```

### 1. Configuration File
**File**: `config.txt`

**Contains**:
- **Simulation Info**: ID, checkpoint, map, agents, version details
- **Technical Settings**: Cluster, timing, grid size, speeds
- **Video Settings**: Recording options and frame rates
- **Controller Info**: Controller type and LSTM usage
- **Output Files**: List of generated files
- **Paths**: Relevant directories and file locations

**Example content**:
```
# Simulation Configuration File
# Generated on: 2025-10-01 15:30:45

[SIMULATION_INFO]
SIMULATION_ID: 2025_10_01-15_30_45
CHECKPOINT_NUMBER: 50
MAP_NAME: simple_kitchen_circular
NUM_AGENTS: 2
INTENT_VERSION: v3.1
COOPERATIVE: True
GAME_VERSION: competition
TRAINING_ID: training_001

[TECHNICAL_SETTINGS]
CLUSTER: cuenca
DURATION_SECONDS: 180
ENGINE_TICK_RATE: 24
AI_TICK_RATE: 1
AGENT_SPEED_PX_PER_SEC: 32
GRID_SIZE: (8, 8)
TILE_SIZE: 16

[VIDEO_SETTINGS]
ENABLE_VIDEO: True
VIDEO_FPS: 24

[CONTROLLER_INFO]
CONTROLLER_TYPE: standard
USE_LSTM: False

[OUTPUT_FILES]
STATE_CSV: simulation_log_checkpoint_50_2025_10_01-15_30_45.csv
ACTION_CSV: actions_checkpoint_50_2025_10_01-15_30_45.csv
VIDEO_FILE: offline_recording_2025_10_01-15_30_45.mp4

[PATHS]
SIMULATION_DIR: /path/to/simulations/simulation_2025_10_01-15_30_45
CHECKPOINT_DIR: /path/to/checkpoint_50
MAP_FILE: /path/to/maps/simple_kitchen_circular.txt
```

### 2. State Log CSV
**File**: `simulation_log_checkpoint_{checkpoint}_{timestamp}.csv`

**Columns**:
- `agent`: Agent identifier
- `frame`: Frame number
- `second`: Time in seconds
- `x`, `y`: Agent coordinates
- `tile_x`, `tile_y`: Tile coordinates
- `item`: Current item held
- `score`: Current score

### 3. Action Log CSV
**File**: `actions_checkpoint_{checkpoint}_{timestamp}.csv`

**Columns**:
- `agent_id`: Agent identifier
- `decision_frame`, `completion_frame`: Action timing
- `duration_frames`: Action duration
- `action_number`: Sequential action number
- `action_type`: Type of action performed
- `target_tile_type`, `target_tile_x`, `target_tile_y`: Target information
- `decision_x`, `decision_y`: Agent position when action started
- `completion_x`, `completion_y`: Agent position when action completed
- `item_before`, `item_after`: Item state changes

### 4. Video Recording
**File**: `offline_recording_{timestamp}.mp4`

**Features**:
- High-quality video recording
- HUD overlay with real-time scores
- Configurable frame rate
- Multiple codec support

## Technical Details

### Ray Cluster Management
- Automatic initialization and cleanup
- Local mode for development
- Error handling for cluster issues

### Controller Loading
- Automatic detection of LSTM vs standard controllers
- Support for multiple agent types
- Graceful handling of missing checkpoints

### Action Tracking
- Precise timing with minimum duration enforcement
- Detailed state tracking for analysis
- Thread-safe operation with proper locking
- Automatic cleanup of incomplete actions

### Video Recording
- Multiple codec fallback for compatibility
- Automatic frame format conversion
- HUD overlay with score information
- Efficient memory usage

## Migration Guide

### From Old to New Script

**Old Usage:**
```bash
# Required manual editing of script variables
python experimental_simulation.py simple_kitchen_circular 2 v3.1 1 competition training_001 50 true
```

**New Usage:**
```bash
# Flexible command-line interface with clear parameters
python experimental_simulation_new.py simple_kitchen_circular 2 v3.1 1 competition training_001 50 --enable-video true --cluster cuenca
```

### Key Changes

1. **Parameter Handling**: Now uses proper argument parsing instead of sys.argv indexing
2. **Modular Structure**: Code is organized into logical components
3. **Error Handling**: Better error messages and recovery
4. **Resource Management**: Improved cleanup and timeout handling
5. **Logging**: More structured and informative output

## Extension Guide

### Adding New Features

1. **New Logging**: Extend `DataLogger` class with additional CSV columns
2. **New Metrics**: Add methods to track additional simulation metrics
3. **Custom Controllers**: Extend `ControllerManager` for new controller types
4. **Video Features**: Enhance `VideoRecorder` with additional HUD elements

### Customizing Behavior

1. **Timing**: Modify `SimulationConfig` for different timing parameters
2. **Paths**: Customize `PathManager` for different directory structures
3. **Game Setup**: Extend `GameManager` for custom game initialization
4. **Action Tracking**: Enhance `ActionTracker` for specific action types

## Performance Considerations

### Optimization Features
- **Efficient Rendering**: Optimized frame generation and encoding
- **Memory Management**: Proper resource cleanup and garbage collection
- **Threading**: Safe concurrent operations with timeout handling
- **File I/O**: Efficient CSV writing with proper buffering

### Scalability
- **Large Simulations**: Handles long-duration simulations efficiently
- **Multiple Agents**: Scales well with increasing agent count
- **High Frame Rates**: Supports high-frequency data collection
- **Large Maps**: Efficient handling of complex game environments

## Troubleshooting

### Common Issues

1. **Controller Loading Failures**
   - Check checkpoint directory exists
   - Verify controller type configuration
   - Ensure proper file permissions

2. **Video Recording Issues**
   - Check available codecs
   - Verify output directory permissions
   - Monitor disk space availability

3. **Ray Initialization Problems**
   - Check for port conflicts
   - Verify system resources
   - Review Ray logs for details

4. **Path Resolution Errors**
   - Verify cluster configuration
   - Check directory structure
   - Ensure proper file paths

### Performance Issues

1. **Slow Simulation**
   - Reduce tick rate if needed
   - Disable video recording for faster runs
   - Check system resource usage

2. **Memory Usage**
   - Monitor RAM usage during long simulations
   - Adjust buffer sizes if needed
   - Ensure proper cleanup

## Dependencies

- **Ray**: Distributed computing framework
- **OpenCV**: Video recording and image processing
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation (via CSV)
- **Threading**: Concurrent operations
- **Pathlib**: Modern path handling

## Contributing

When extending the simulation system:

1. Follow the established class-based architecture
2. Add comprehensive documentation and type hints
3. Include proper error handling and cleanup
4. Test with different simulation scenarios
5. Update this README for new features

## Support

For issues or questions about the simulation system:

1. Check this README for common solutions
2. Review the code documentation and comments
3. Test with minimal examples to isolate issues
4. Report bugs with detailed reproduction steps