# Simulation Module Refactoring

## Overview

The large `utils.py` file has been refactored into individual modules for better maintainability and organization. Each class now has its own file while maintaining backward compatibility.

## New File Structure

```
spoiled_broth/simulations/
├── __init__.py                 # Package initialization with imports
├── utils.py                    # Original file (kept for backward compatibility)
├── simulation_config.py        # SimulationConfig class
├── path_manager.py            # PathManager class
├── controller_manager.py      # ControllerManager class
├── ray_manager.py             # RayManager class
├── data_logger.py             # DataLogger class
├── action_tracker.py          # ActionTracker class
├── video_recorder.py          # VideoRecorder class
├── game_manager.py            # GameManager class
├── simulation_runner.py       # SimulationRunner class
└── simulation_utils.py        # Utility functions
```

## Classes and Their Files

1. **SimulationConfig** → `simulation_config.py`
   - Configuration class for simulation parameters
   - Handles cluster settings, timing, video settings, grid settings

2. **PathManager** → `path_manager.py`
   - Manages paths and directories for simulation runs
   - Handles path setup and grid size detection from maps

3. **ControllerManager** → `controller_manager.py`
   - Manages RL controller initialization and configuration
   - Determines controller types and initializes agent controllers

4. **RayManager** → `ray_manager.py`
   - Manages Ray cluster initialization and shutdown
   - Simple static methods for Ray operations

5. **DataLogger** → `data_logger.py`
   - Handles logging of simulation state and action data
   - Creates CSV files and configuration files

6. **ActionTracker** → `action_tracker.py`
   - Tracks agent actions with detailed logging and timing
   - Handles action start/end tracking and cleanup

7. **VideoRecorder** → `video_recorder.py`
   - Handles video recording with HUD overlay
   - Manages video encoding and frame processing

8. **GameManager** → `game_manager.py`
   - Manages game instance creation and configuration
   - Handles game state reset and factory creation

9. **SimulationRunner** → `simulation_runner.py`
   - Main class for running complete simulations
   - Orchestrates all other components for full simulation execution

10. **Utility Functions** → `simulation_utils.py`
    - `setup_simulation_argument_parser()`
    - `main_simulation_pipeline()`

## Import Usage

### New Modular Imports (Recommended)
```python
from spoiled_broth.simulations import SimulationConfig, SimulationRunner
from spoiled_broth.simulations import setup_simulation_argument_parser, main_simulation_pipeline

# Or import specific modules
from spoiled_broth.simulations.simulation_config import SimulationConfig
from spoiled_broth.simulations.simulation_runner import SimulationRunner
```

### Legacy Imports (Still Supported)
```python
from spoiled_broth.simulations.utils import (
    SimulationConfig,
    SimulationRunner,
    setup_simulation_argument_parser,
    main_simulation_pipeline
)
```

## Key Dependencies Between Modules

- `simulation_runner.py` imports most other modules as it orchestrates the simulation
- `data_logger.py` imports `action_tracker.py` to avoid circular imports
- `path_manager.py`, `controller_manager.py`, and `game_manager.py` all depend on `simulation_config.py`
- All imports use relative imports (`.module_name`) within the package

## Backward Compatibility

- The original `utils.py` file is preserved
- The `__init__.py` file imports all classes and functions, maintaining the same public API
- Existing code using imports from `spoiled_broth.simulations.utils` will continue to work
- New code can use the more specific module imports for better dependency management

## Benefits of Refactoring

1. **Better Organization**: Each class has its own focused file
2. **Easier Maintenance**: Changes to one class don't require editing a large file
3. **Clearer Dependencies**: Import relationships are more explicit
4. **Improved Testing**: Individual classes can be tested in isolation
5. **Reduced Merge Conflicts**: Multiple developers can work on different classes simultaneously
6. **Better IDE Support**: Faster loading and better autocomplete for smaller files