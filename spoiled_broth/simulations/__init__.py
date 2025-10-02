"""
Simulation utilities package for running reinforcement learning simulations.

This package provides modular components for:
- Configuration management
- Path management
- Controller initialization
- Data logging and tracking
- Video recording
- Game setup and execution
- Ray cluster management
- Complete simulation execution

Author: Samuel Lozano
"""

# Configuration
from .simulation_config import SimulationConfig

# Core managers
from .path_manager import PathManager
from .controller_manager import ControllerManager
from .ray_manager import RayManager
from .game_manager import GameManager

# Data handling
from .data_logger import DataLogger
from .action_tracker import ActionTracker
from .video_recorder import VideoRecorder

# Main simulation runner
from .simulation_runner import SimulationRunner

# Utility functions
from .simulation_utils import (
    setup_simulation_argument_parser,
    main_simulation_pipeline
)

# Meaningful actions analysis
from .meaningful_actions import (
    analyze_meaningful_actions,
    analyze_meaningful_actions_from_files
)

__all__ = [
    # Configuration
    'SimulationConfig',
    
    # Core managers
    'PathManager',
    'ControllerManager', 
    'RayManager',
    'GameManager',
    
    # Data handling
    'DataLogger',
    'ActionTracker',
    'VideoRecorder',
    
    # Main simulation runner
    'SimulationRunner',
    
    # Utility functions
    'setup_simulation_argument_parser',
    'main_simulation_pipeline',
    
    # Meaningful actions analysis
    'analyze_meaningful_actions',
    'analyze_meaningful_actions_from_files'
]