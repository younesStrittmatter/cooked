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
        config_content = [
            "# Simulation Configuration File",
            f"# Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}",
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
            "[TECHNICAL_SETTINGS]",
            f"CLUSTER: {self.simulation_config.get('CLUSTER', 'unknown')}",
            f"DURATION_SECONDS: {self.simulation_config.get('DURATION', 'unknown')}",
            f"ENGINE_TICK_RATE: {self.simulation_config.get('TICK_RATE', 'unknown')}",
            f"AI_TICK_RATE: {self.simulation_config.get('AI_TICK_RATE', 1)}",
            f"AGENT_SPEED_PX_PER_SEC: {self.simulation_config.get('AGENT_SPEED', 32)}",
            f"GRID_SIZE: {self.simulation_config.get('GRID_SIZE', 'unknown')}",
            f"TILE_SIZE: {self.simulation_config.get('TILE_SIZE', 16)}",
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
            ""
        ]
        
        try:
            with open(self.config_path, 'w') as f:
                f.write('\n'.join(config_content))
            print(f"Configuration saved to: {self.config_path}")
        except Exception as e:
            print(f"Warning: Could not create config file: {e}")
    
    def log_state(self, frame: int, game: Any, tick_rate: int):
        """
        Log current state of all agents.
        
        Args:
            frame: Current frame number
            game: Game instance
            tick_rate: Engine tick rate for time calculation
        """
        # Import ActionTracker here to avoid circular imports
        from .action_tracker import ActionTracker
        
        with open(self.state_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for agent_id, obj in game.gameObjects.items():
                if agent_id.startswith('ai_rl_'):
                    # Use consistent tile coordinate calculation with action tracker
                    agent_x = getattr(obj, 'x', 0)
                    agent_y = getattr(obj, 'y', 0)
                    tile_x, tile_y = ActionTracker._position_to_tile(agent_x, agent_y)
                    
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