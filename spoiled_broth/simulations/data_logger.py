"""
Data logging utilities for simulation runs.

Author: Samuel Lozano
"""

import os
import csv
import time
from pathlib import Path
from typing import Dict, Any


class DataLogger:
    """Handles logging of simulation state and action data."""
    
    def __init__(self, base_saving_path: Path, checkpoint_number: str, timestamp: str, 
                 simulation_config: Dict[str, Any]):
        self.base_saving_path = base_saving_path
        self.checkpoint_number = checkpoint_number
        self.timestamp = timestamp
        self.simulation_config = simulation_config
        
        # Create simulation-specific directory
        self.simulation_dir = base_saving_path / f"simulation_{timestamp}"
        os.makedirs(self.simulation_dir, exist_ok=True)
        
        self.state_csv_path = self.simulation_dir / f"simulation.csv"
        self.action_csv_path = self.simulation_dir / f"actions.csv"
        self.config_path = self.simulation_dir / "config.txt"
        
        self._initialize_state_logger()
        self._create_config_file()
    
    def _initialize_state_logger(self):
        """Initialize the state logger CSV file."""
        with open(self.state_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["agent_id", "frame", "second", "x", "y", "tile_x", "tile_y", "item", "score"])
    
    def _create_config_file(self):
        """Create a configuration file with all simulation parameters."""
        import time
        import platform
        import sys
        import os
        
        # Calculate derived timing values
        duration = self.simulation_config.get('DURATION', 0)
        init_period = self.simulation_config.get('AGENT_INITIALIZATION_PERIOD', 0.0)
        tick_rate = self.simulation_config.get('TICK_RATE', 24)
        total_time = duration + init_period
        total_frames = int(total_time * tick_rate)
        gameplay_frames = int(duration * tick_rate)
        init_frames = int(init_period * tick_rate)
        
        config_content = [
            "# Simulation Configuration File",
            f"# Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Platform: {platform.system()} {platform.release()}",
            f"# Python: {sys.version.split()[0]}",
            f"# Working Directory: {os.getcwd()}",
            f"# Command Line: {' '.join(sys.argv) if hasattr(sys, 'argv') else 'N/A'}",
            "",
            "[SIMULATION_INFO]",
            f"SIMULATION_ID: {self.timestamp}",
            f"CHECKPOINT_NUMBER: {self.checkpoint_number}",
            f"MAP_NR: {self.simulation_config.get('MAP_NR', 'unknown')}",
            f"NUM_AGENTS: {self.simulation_config.get('NUM_AGENTS', 'unknown')}",
            f"INTENT_VERSION: {self.simulation_config.get('INTENT_VERSION', 'unknown')}",
            f"COOPERATIVE: {self.simulation_config.get('COOPERATIVE', 'unknown')}",
            f"GAME_VERSION: {self.simulation_config.get('GAME_VERSION', 'unknown')}",
            f"TRAINING_ID: {self.simulation_config.get('TRAINING_ID', 'unknown')}",
            "",
            "[TIMING_CONFIGURATION]",
            f"AGENT_INITIALIZATION_PERIOD: {init_period}",
            f"DURATION_SECONDS: {duration}",
            f"TOTAL_SIMULATION_TIME: {total_time}",
            f"ENGINE_TICK_RATE: {tick_rate}",
            f"AI_TICK_RATE: {self.simulation_config.get('AI_TICK_RATE', 1)}",
            f"INITIALIZATION_FRAMES: {init_frames}",
            f"GAMEPLAY_FRAMES: {gameplay_frames}",
            f"TOTAL_FRAMES: {total_frames}",
            "",
            "[TECHNICAL_SETTINGS]",
            f"CLUSTER: {self.simulation_config.get('CLUSTER', 'unknown')}",
            f"AGENT_SPEED_PX_PER_SEC: {self.simulation_config.get('AGENT_SPEED', 32)}",
            f"GRID_SIZE: {self.simulation_config.get('GRID_SIZE', 'unknown')}",
            f"DEFAULT_GRID_SIZE: {self.simulation_config.get('DEFAULT_GRID_SIZE', 'unknown')}",
            f"TILE_SIZE: {self.simulation_config.get('TILE_SIZE', 16)}",
            f"LOCAL_PATH: {self.simulation_config.get('LOCAL_PATH', 'unknown')}",
            "",
            "[VIDEO_SETTINGS]",
            f"ENABLE_VIDEO: {self.simulation_config.get('ENABLE_VIDEO', 'unknown')}",
            f"VIDEO_FPS: {self.simulation_config.get('VIDEO_FPS', 'unknown')}",
            "",
            "[CONTROLLER_INFO]",
            f"CONTROLLER_TYPE: {self.simulation_config.get('CONTROLLER_TYPE', 'unknown')}",
            f"USE_LSTM: {self.simulation_config.get('USE_LSTM', 'unknown')}",
            "",
            "[OUTPUT_FILES]",
            f"STATE_CSV: {self.state_csv_path.name}",
            f"ACTION_CSV: {self.action_csv_path.name}",
            f"VIDEO_FILE: {self.simulation_config.get('video_filename', 'N/A' if not self.simulation_config.get('ENABLE_VIDEO') else 'offline_recording_{}.mp4'.format(self.timestamp))}",
            "",
            "[PATHS]",
            f"SIMULATION_DIR: {self.simulation_dir}",
            f"CHECKPOINT_DIR: {self.simulation_config.get('CHECKPOINT_DIR', 'unknown')}",
            f"MAP_FILE: {self.simulation_config.get('MAP_FILE', 'unknown')}",
            "",
            "[INITIALIZATION_DETAILS]",
            f"# Agent initialization period: {init_period} seconds ({init_frames} frames)",
            f"# During initialization, agents are positioned but perform no actions",
            f"# This prevents teleportation artifacts and ensures system stabilization",
            f"# Actual gameplay begins after initialization period completes",
            "",
            "[TIMING_BREAKDOWN]",
            f"# Total simulation: {total_time} seconds ({total_frames} frames)",
            f"# - Initialization: {init_period} seconds ({init_frames} frames)",
            f"# - Active gameplay: {duration} seconds ({gameplay_frames} frames)",
            f"# Frame timing: {1.0/tick_rate:.4f} seconds per frame",
            ""
        ]
        
        try:
            with open(self.config_path, 'w') as f:
                f.write('\n'.join(config_content))
            print(f"Configuration saved to: {self.config_path}")
        except Exception as e:
            print(f"Warning: Could not create config file: {e}")
    
    def update_config_with_runtime_info(self, game: Any):
        """
        Update the config file with runtime information detected during simulation.
        
        Args:
            game: Game instance with runtime information
        """
        if not hasattr(game, 'agents_start_acting_frame'):
            print("No runtime agent start frame detected - agents may not have started acting yet")
            return
            
        tick_rate = self.simulation_config.get('TICK_RATE', 24)
        agents_start_frame = game.agents_start_acting_frame
        agents_start_time = agents_start_frame / tick_rate
        
        runtime_info = [
            "",
            "[RUNTIME_DETECTION]",
            f"ACTUAL_AGENTS_START_ACTING_FRAME: {agents_start_frame}",
            f"ACTUAL_AGENTS_START_ACTING_TIME: {agents_start_time:.3f}",
            f"# This is the actual frame/time when the first agent performed an action",
            f"# Use this frame as the cutoff for data analysis - ignore data before this frame",
            f"# This frame should be used by positions_extraction and actions_extraction",
        ]
        
        try:
            # Append runtime info to existing config file
            with open(self.config_path, 'a') as f:
                f.write('\n'.join(runtime_info))
            print(f"Runtime info added to config: Agents start acting at frame {agents_start_frame} (time: {agents_start_time:.3f}s)")
        except Exception as e:
            print(f"Warning: Could not update config file with runtime info: {e}")
    
    
    def log_state(self, frame: int, game: Any, tick_rate: int):
        """
        Log current state of all agents.
        
        Args:
            frame: Current frame number
            game: Game instance
            tick_rate: Engine tick rate for time calculation
        """
        with open(self.state_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for agent_id, obj in game.gameObjects.items():
                if agent_id.startswith('ai_rl_'):
                    # Use consistent tile coordinate calculation
                    agent_x = getattr(obj, 'x', 0)
                    agent_y = getattr(obj, 'y', 0)
                    # Use consistent tile calculation: floor division by tile size + 1
                    if agent_x is None or agent_y is None:
                        tile_x, tile_y = 0, 0
                    else:
                        tile_x = int(agent_x // 16 + 1)
                        tile_y = int(agent_y // 16 + 1)
                    
                    writer.writerow([
                        agent_id, 
                        frame, 
                        frame / tick_rate,
                        agent_x, 
                        agent_y,
                        tile_x, 
                        tile_y,
                        getattr(obj, 'item', ''), 
                        getattr(obj, 'score', 0)
                    ])

    def _position_to_tile(x: float, y: float) -> tuple[int, int]:
        """Convert pixel position to tile coordinates consistently."""
        # Use consistent tile calculation: floor division by tile size + 1
        # This ensures all components use the same calculation
        if x is None or y is None:
            return 0, 0
        tile_x = int(x // 16 + 1)
        tile_y = int(y // 16 + 1)
        return tile_x, tile_y