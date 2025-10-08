"""
Utility functions for running reinforcement learning simulations.

This module provides common functionality for:
- Configuration management
- Controller initialization
- Data logging and tracking
- Video recording
- Game setup and execution
- Ray cluster management

Author: Samuel Lozano
"""

import os
import re
import sys
import time
import csv
import threading
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass

import ray
import numpy as np
import cv2

from .simulation_config import SimulationConfig
from engine.app.ai_only_app import AIOnlySessionApp
from engine.extensions.renderer2d.renderer_2d_offline import Renderer2DOffline
from engine.extensions.renderer2d.renderer_ui import Renderer2DModule



class PathManager:
    """Manages paths and directories for simulation runs."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
    
    def setup_paths(self, map_nr: str, num_agents: int, intent_version: str,
                   cooperative: bool, game_version: str, training_id: str,
                   checkpoint_number: str) -> Dict[str, Path]:
        """
        Set up all necessary paths for a simulation run.
        
        Args:
            map_nr: Name of the map
            num_agents: Number of agents
            intent_version: Intent version identifier
            cooperative: Whether this is a cooperative simulation
            game_version: Game version identifier
            training_id: Training identifier
            checkpoint_number: Checkpoint number to load (integer or "final")
            
        Returns:
            Dictionary containing all relevant paths
        """
        base_path = Path(f"{self.config.local_path}/data/samuel_lozano/cooked/{game_version}/{intent_version}/map_{map_nr}")
        
        training_type = "cooperative" if cooperative else "competitive"
        training_path = base_path / f"{training_type}/Training_{training_id}"
        
        # Resolve checkpoint number (handle "final" case)
        # Validate checkpoint_number
        if checkpoint_number.lower() != "final":
            try:
                int(checkpoint_number)  # Validate it's a number
            except ValueError:
                raise ValueError(f"Invalid checkpoint number: '{checkpoint_number}'. Must be an integer or 'final'")
        
        # Construct paths based on checkpoint type
        if checkpoint_number.lower() == "final":
            checkpoint_dir = training_path / "checkpoint_final"
            saving_path = training_path / "simulations_final"
        else:
            checkpoint_dir = training_path / f"checkpoint_{checkpoint_number}"
            saving_path = training_path / f"simulations_{checkpoint_number}"
        
        # Find project root by looking for spoiled_broth directory
        project_root = Path(__file__).parent.parent.parent  # utils.py is in spoiled_broth/simulations/
        
        paths = {
            'base_path': base_path,
            'training_path': training_path,
            'checkpoint_dir': checkpoint_dir,
            'saving_path': saving_path,
            'path_root': project_root / "spoiled_broth",
            'map_txt_path': project_root / "spoiled_broth" / "maps" / f"{map_nr}.txt",
            'config_path': training_path / "config.txt",
            'checkpoint_number': checkpoint_number  # Keep original checkpoint_number
        }
        
        # Create directories
        os.makedirs(paths['saving_path'], exist_ok=True)
        
        return paths
    
    def get_grid_size_from_map(self, map_txt_path: Path) -> Tuple[int, int]:
        """
        Determine grid size from map file.
        
        Args:
            map_txt_path: Path to map text file
            
        Returns:
            Tuple of (width, height) for the grid
        """
        try:
            with open(map_txt_path) as f:
                lines = [l.rstrip("\n") for l in f.readlines()]
            if lines:
                return (len(lines[0]), len(lines))
        except Exception as e:
            print(f"Warning: Could not read map file {map_txt_path}: {e}")
        
        return self.config.default_grid_size


class ControllerManager:
    """Manages RL controller initialization and configuration."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
    
    def determine_controller_type(self, config_path: Path) -> str:
        """
        Determine which controller type to use based on config file.
        
        Args:
            config_path: Path to training config file
            
        Returns:
            Controller type ('lstm' or 'standard')
        """
        use_lstm = False
        
        if config_path.exists():
            try:
                with open(config_path) as f:
                    for line in f:
                        if line.strip().startswith("USE_LSTM"):
                            use_lstm = line.split(":", 1)[1].strip().lower() == "true"
                            break
            except Exception as e:
                print(f"Warning: Could not read config file {config_path}: {e}")
        
        return 'lstm' if use_lstm else 'standard'
    
    def initialize_controllers(self, num_agents: int, checkpoint_dir: Path,
                             controller_type: str, game_version: str) -> Dict[str, Any]:
        """
        Initialize RL controllers for all agents.
        
        Args:
            num_agents: Number of agents to create controllers for
            checkpoint_dir: Directory containing model checkpoints
            controller_type: Type of controller ('lstm' or 'standard')
            game_version: Game version for competition detection
            
        Returns:
            Dictionary mapping agent IDs to controller instances
        """
        # Import controllers dynamically to avoid import issues
        from spoiled_broth.rl.rllib_controller import RLlibController
        from spoiled_broth.rl.rllib_controller_lstm import RLlibControllerLSTM
        
        ControllerCls = RLlibControllerLSTM if controller_type == 'lstm' else RLlibController
        is_competition = game_version.upper() == "COMPETITION"
        
        controllers = {}
        
        for i in range(1, num_agents + 1):
            agent_id = f"ai_rl_{i}"
            try:
                controllers[agent_id] = ControllerCls(
                    agent_id, 
                    checkpoint_dir, 
                    f"policy_{agent_id}", 
                    competition=is_competition
                )
                print(f"Successfully initialized controller for {agent_id}")
            except Exception as e:
                print(f"Warning: couldn't initialize controller {agent_id}: {e}")
        
        return controllers


class RayManager:
    """Manages Ray cluster initialization and shutdown."""
    
    @staticmethod
    def initialize_ray():
        """Initialize Ray in local mode."""
        try:
            ray.shutdown()
            ray.init(local_mode=True, ignore_reinit_error=True, include_dashboard=False)
            print("Ray initialized successfully")
        except Exception as e:
            print(f"Warning: Ray initialization failed: {e}")
    
    @staticmethod
    def shutdown_ray():
        """Shutdown Ray cluster."""
        try:
            ray.shutdown()
            print("Ray shutdown successfully")
        except Exception as e:
            print(f"Warning: Ray shutdown failed: {e}")


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
        init_period = self.simulation_config.get('AGENT_INITIALIZATION_PERIOD', 15.0)
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


class RawActionLogger:
    """Logs raw integer actions directly from controllers with the same format as ActionTracker."""
    
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.lock = threading.Lock()
        
        # Action ID counters per agent (for action_id - 0-indexed per agent)
        self.action_id_counters = {}
        
        self._initialize_csv()
    
    def _initialize_csv(self):
        """Initialize the raw action logging CSV file."""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'action_id', 'agent_id', 'action_number', 'action_type', 
                'target_tile_type', 'target_tile_x', 'target_tile_y'
            ])
    
    def _get_tile_info(self, game: Any, action: int, clickable_indices: list) -> tuple[str, Optional[int], Optional[int]]:
        """Get tile information for a given action index."""
        tile_type, x_tile, y_tile = '', None, None
        
        try:
            if action == len(clickable_indices):
                # This is a "do_nothing" action
                return '', None, None
            
            if 0 <= action < len(clickable_indices):
                tile_index = clickable_indices[action]
                grid = getattr(game, 'grid', None)
                if grid is not None:
                    x = tile_index % grid.width
                    y = tile_index // grid.width
                    if 0 <= x < grid.width and 0 <= y < grid.height:
                        tile = grid.tiles[x][y]
                        if tile:
                            tile_type = tile.__class__.__name__.lower()
                            x_tile, y_tile = x + 1, y + 1  # Convert to 1-indexed
                            return tile_type, x_tile, y_tile
        except Exception as e:
            print(f"[RAW_ACTION_LOGGER] Error getting tile info: {e}")
        
        return tile_type, x_tile, y_tile
    
    def log_action(self, agent_id: str, raw_action: int, game: Any, clickable_indices: list):
        """Log a raw integer action to CSV."""
        with self.lock:
            try:
                # Get action_id for this agent (0-indexed per agent)
                if agent_id not in self.action_id_counters:
                    self.action_id_counters[agent_id] = 0
                else:
                    self.action_id_counters[agent_id] += 1
                
                action_id = self.action_id_counters[agent_id]
                
                # Determine action type and target info
                if raw_action == len(clickable_indices):
                    # This is a "do_nothing" action
                    action_type = 'do_nothing'
                    tile_type, x_tile, y_tile = '', '', ''
                    action_number = -1
                else:
                    # This is a click action
                    action_type = 'click'
                    tile_type, x_tile, y_tile = self._get_tile_info(game, raw_action, clickable_indices)
                    action_number = raw_action
                
                # Write action immediately to CSV in same format as ActionTracker
                with open(self.csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    row = [
                        action_id,
                        agent_id,
                        action_number,
                        action_type,
                        tile_type or '',
                        x_tile or '',
                        y_tile or ''
                    ]
                    writer.writerow(row)
                    
                print(f"[RAW_ACTION_LOGGER] Logged action {raw_action} for {agent_id} (action_id: {action_id})")
                    
            except Exception as e:
                print(f"[RAW_ACTION_LOGGER] Error logging action for {agent_id}: {e}")


class VideoRecorder:
    """Handles video recording with HUD overlay."""
    
    def __init__(self, output_path: Path, fps: int = 24):
        self.output_path = str(output_path)
        self.fps = fps
        self.writer = None
        self.size = None
        self.hud_height = None  # Will be calculated based on first frame
        
        # Create directory if it doesn't exist
        os.makedirs(output_path.parent, exist_ok=True)
    
    def start(self, frame: np.ndarray):
        """Initialize video writer with first frame."""
        src = self._prepare_frame(frame)
        h, w = src.shape[:2]
        
        # Calculate HUD height (3 lines of text + padding)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 1
        line_height = cv2.getTextSize('Tg', font, scale, thickness)[0][1] + 6
        self.hud_height = max(48, 12 + 3 * line_height)  # 3 lines: Coop, ai_rl_1, ai_rl_2
        
        # Video dimensions include game area + HUD area
        video_width = w
        video_height = h + self.hud_height
        
        # Try different codecs
        fourcc_candidates = ['mp4v', 'avc1', 'H264', 'XVID', 'MJPG']
        
        for code in fourcc_candidates:
            fourcc = cv2.VideoWriter_fourcc(*code)
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (video_width, video_height))
            
            try:
                if self.writer.isOpened():
                    self.size = (video_width, video_height)
                    break
            except Exception:
                pass
            
            if self.writer:
                self.writer.release()
                self.writer = None
        
        if self.writer is None:
            print(f"Warning: could not open VideoWriter for {self.output_path}")
    
    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        """Prepare frame for video encoding."""
        src = frame
        if src.dtype != np.uint8:
            src = (np.clip(src, 0.0, 1.0) * 255).astype(np.uint8)
        if src.ndim == 2:
            src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        return src
    
    def write_frame_with_hud(self, frame: np.ndarray, game: Any):
        """Write frame with HUD overlay showing scores."""
        if self.writer is None:
            self.start(frame)
        
        if self.writer is None:
            return
        
        try:
            src = self._prepare_frame(frame)
            
            # Calculate scores
            coop_score = 0
            scores = {}
            try:
                for agent_id, obj in game.gameObjects.items():
                    if agent_id.startswith('ai_rl_') and obj is not None:
                        score = int(getattr(obj, 'score', 0) or 0)
                        coop_score += score
                        scores[agent_id] = score
            except Exception:
                pass
            
            # Create HUD
            hud_lines = [f"Coop: {coop_score}"]
            for label in ('ai_rl_1', 'ai_rl_2'):
                hud_lines.append(f"{label}: {scores.get(label, 0)}")
            
            # Add HUD to frame (this preserves the original game dimensions)
            frame_with_hud = self._add_hud_to_frame(src, hud_lines)
            
            # Never resize - the frame should already be the correct size
            self.writer.write(frame_with_hud)
            
        except Exception as e:
            print(f"Warning: failed to write video frame: {e}")
    
    def _add_hud_to_frame(self, frame: np.ndarray, hud_lines: List[str]) -> np.ndarray:
        """Add HUD overlay to frame with pure black background."""
        h, w = frame.shape[:2]
        
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 1
        line_height = cv2.getTextSize('Tg', font, scale, thickness)[0][1] + 6
        
        # Use consistent HUD height calculated during initialization
        if self.hud_height is None:
            pad_height = max(48, 12 + len(hud_lines) * line_height)
        else:
            pad_height = self.hud_height
        
        # Create canvas with HUD area - preserve original game frame exactly
        canvas = np.zeros((h + pad_height, w, 3), dtype=np.uint8)
        canvas[0:h, 0:w] = frame
        
        # Draw pure black background for HUD area
        cv2.rectangle(canvas, (0, h), (w, h + pad_height), (0, 0, 0), -1)
        
        # Draw HUD text
        start_y = h + 12 + line_height
        for i, text in enumerate(hud_lines):
            y = start_y + i * line_height
            # Shadow for better readability
            cv2.putText(canvas, text, (9, y+1), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
            # White text
            cv2.putText(canvas, text, (8, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
        return canvas
    
    def stop(self):
        """Stop video recording and release resources."""
        if self.writer:
            self.writer.release()
            self.writer = None


class GameManager:
    """Manages game instance creation and configuration."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
    
    def create_game_factory(self, map_nr: str, grid_size: Tuple[int, int], 
                           intent_version: str) -> Callable:
        """
        Create a game factory function for the simulation.
        
        Args:
            map_nr: Name of the map to load
            grid_size: Grid size tuple (width, height)
            intent_version: Intent version identifier
            
        Returns:
            Game factory function
        """
        def game_factory():
            from spoiled_broth.game import SpoiledBroth as Game
            
            game = Game(
                map_nr=map_nr, 
                grid_size=grid_size, 
                intent_version=intent_version
            )
            
            # Reset game state
            self._reset_game_state(game)
            return game
        
        return game_factory
    
    def _reset_game_state(self, game: Any):
        """Reset game state to clean initial conditions."""
        try:
            grid = getattr(game, 'grid', None)
            if grid is not None:
                from spoiled_broth.world.tiles import Dispenser
                
                # Clear items on non-dispenser tiles
                for x in range(grid.width):
                    for y in range(grid.height):
                        tile = grid.tiles[x][y]
                        if tile is None:
                            continue
                        
                        if hasattr(tile, 'item') and not isinstance(tile, Dispenser):
                            tile.item = None
                        # Reset cutting board state by resetting cut_time_accumulated
                        # (cut_stage is a read-only property calculated from cut_time_accumulated)
                        if hasattr(tile, 'cut_time_accumulated'):
                            tile.cut_time_accumulated = 0
                        if hasattr(tile, 'cut_by'):
                            tile.cut_by = None
                        if hasattr(tile, 'cut_item'):
                            tile.cut_item = None
            
            # Reset agent states
            for agent_id, obj in list(game.gameObjects.items()):
                if agent_id.startswith('ai_rl_'):
                    if hasattr(obj, 'item'):
                        obj.item = None
                    if hasattr(obj, 'action_complete'):
                        obj.action_complete = True
                    if hasattr(obj, 'current_action'):
                        obj.current_action = None
                    if hasattr(obj, 'speed'):
                        obj.speed = self.config.agent_speed_px_per_sec
                        
        except Exception as e:
            print(f"Warning: Error resetting game state: {e}")


class SimulationRunner:
    """Main class for running simulations."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.path_manager = PathManager(config)
        self.controller_manager = ControllerManager(config)
        self.ray_manager = RayManager()
        self.game_manager = GameManager(config)
        
    def run_simulation(self, map_nr: str, num_agents: int, intent_version: str,
                      cooperative: bool, game_version: str, training_id: str,
                      checkpoint_number: str, timestamp: str) -> Dict[str, Path]:
        """
        Run a complete simulation.
        
        Args:
            map_nr: Name of the map
            num_agents: Number of agents
            intent_version: Intent version identifier
            cooperative: Whether this is cooperative
            game_version: Game version identifier
            training_id: Training identifier
            checkpoint_number: Checkpoint number (integer or "final")
            timestamp: Timestamp for file naming
            
        Returns:
            Dictionary containing output file paths
        """
        print(f"Starting simulation: {map_nr}, {num_agents} agents, {intent_version}")
        
        # Initialize Ray
        self.ray_manager.initialize_ray()
        
        try:
            # Setup paths
            paths = self.path_manager.setup_paths(
                map_nr, num_agents, intent_version, cooperative,
                game_version, training_id, checkpoint_number
            )
            
            # Get grid size
            grid_size = self.path_manager.get_grid_size_from_map(paths['map_txt_path'])
            
            # Initialize controllers
            controller_type = self.controller_manager.determine_controller_type(paths['config_path'])
            controllers = self.controller_manager.initialize_controllers(
                num_agents, paths['checkpoint_dir'], controller_type, game_version
            )
            
            if not controllers:
                raise ValueError("No controllers were successfully initialized")
            
            # Setup logging with simulation configuration
            simulation_config = {
                'MAP_NR': map_nr,
                'NUM_AGENTS': num_agents,
                'INTENT_VERSION': intent_version,
                'COOPERATIVE': cooperative,
                'GAME_VERSION': game_version,
                'TRAINING_ID': training_id,
                'CLUSTER': self.config.cluster,
                'DURATION': self.config.duration_seconds,
                'TICK_RATE': self.config.engine_tick_rate,
                'AI_TICK_RATE': self.config.ai_tick_rate,
                'AGENT_SPEED': self.config.agent_speed_px_per_sec,
                'GRID_SIZE': grid_size,
                'TILE_SIZE': self.config.tile_size,
                'ENABLE_VIDEO': self.config.enable_video,
                'VIDEO_FPS': self.config.video_fps,
                'CONTROLLER_TYPE': controller_type,
                'USE_LSTM': controller_type == 'lstm',
                'CHECKPOINT_DIR': str(paths['checkpoint_dir']),
                'MAP_FILE': str(paths['map_txt_path']),
                'AGENT_INITIALIZATION_PERIOD': self.config.agent_initialization_period,
                'TOTAL_SIMULATION_TIME': self.config.total_simulation_time,
                'TOTAL_FRAMES': self.config.total_frames,
                'DEFAULT_GRID_SIZE': self.config.default_grid_size,
                'LOCAL_PATH': self.config.local_path if self.config.local_path else 'N/A'
            }
            
            data_logger = DataLogger(paths['saving_path'], checkpoint_number, timestamp, simulation_config)
            
            # Create raw action logger for capturing integer actions directly from controllers
            raw_action_csv_path = data_logger.simulation_dir / "actions.csv"
            raw_action_logger = RawActionLogger(raw_action_csv_path)
            
            # Setup game
            game_factory = self.game_manager.create_game_factory(
                map_nr, grid_size, intent_version
            )
            
            # Run the simulation
            output_paths = self._execute_simulation(
                game_factory, controllers, paths, data_logger, 
                raw_action_logger, timestamp
            )
            
            return output_paths
            
        finally:
            self.ray_manager.shutdown_ray()
    
    def _execute_simulation(self, game_factory: Callable, controllers: Dict,
                           paths: Dict, data_logger: DataLogger,
                           raw_action_logger: RawActionLogger, 
                           timestamp: str) -> Dict[str, Path]:
        """Execute the main simulation loop."""
        
        # Create engine app
        engine_app = AIOnlySessionApp(
            game_factory=game_factory,
            ui_modules=[Renderer2DModule()],
            agent_map=controllers,
            path_root=paths['path_root'],
            tick_rate=self.config.engine_tick_rate,
            ai_tick_rate=self.config.ai_tick_rate,
            is_max_speed=False,
            agent_initialization_period=self.config.agent_initialization_period
        )
        
        session = engine_app.get_session()
        game = session.engine.game
        
        # Attach raw action logger
        game.raw_action_logger = raw_action_logger
        
        # Attach controllers to agents
        for agent_id, controller in controllers.items():
            agent_obj = game.gameObjects.get(agent_id)
            if agent_obj is not None:
                controller.agent = agent_obj
                print(f"Attached controller to agent {agent_id}")
        
        # Ensure agents have the configured speed (exactly like old code)
        for agent_id, obj in list(game.gameObjects.items()):
            if agent_id.startswith('ai_rl_') and obj is not None:
                try:
                    obj.speed = self.config.agent_speed_px_per_sec
                except Exception:
                    pass
        
        # Setup rendering
        renderer = Renderer2DOffline(
            path_root=paths['path_root'], 
            tile_size=self.config.tile_size
        )
        
        video_recorder = None
        if self.config.enable_video:
            video_path = data_logger.simulation_dir / f"offline_recording_{timestamp}.mp4"
            video_recorder = VideoRecorder(video_path, self.config.video_fps)
        
        try:
            # Start session
            try:
                session.start()
                print("Session started successfully")
            except RuntimeError as e:
                if "threads can only be started once" in str(e):
                    print("Session already started, continuing...")
                else:
                    raise e
            
            time.sleep(0.5)  # Allow startup
            
            # Run simulation loop
            self._run_simulation_loop(
                session, game, renderer, data_logger, 
                video_recorder, timestamp
            )
            
        finally:
            # Cleanup
            self._cleanup_simulation(
                video_recorder, engine_app, 
                game, session.engine, data_logger
            )
        
        return {
            'simulation_dir': data_logger.simulation_dir,
            'config_file': data_logger.config_path,
            'state_csv': data_logger.state_csv_path,
            'action_csv': data_logger.simulation_dir / "actions.csv",
            'video_file': data_logger.simulation_dir / f"offline_recording_{timestamp}.mp4" if self.config.enable_video else None
        }
    
    def _run_simulation_loop(self, session: Any, game: Any, renderer: Any,
                           data_logger: DataLogger, video_recorder: Optional[VideoRecorder],
                           timestamp: str):
        """Run the main simulation loop."""
        
        total_frames = self.config.total_frames
        frame_interval = 1.0 / self.config.engine_tick_rate
        start_time = time.time()
        
        progress_step = max(1, int(total_frames * 0.05))
        next_progress = progress_step
        
        print(f"Starting simulation with {total_frames} frames ({self.config.total_simulation_time} seconds total)")
        print(f"  - Agent initialization: {self.config.agent_initialization_period} seconds")
        print(f"  - Active gameplay: {self.config.duration_seconds} seconds")
        print("Note: Agents will not act during initialization period - this is by design")
        
        prev_state = None
        
        for frame_idx in range(total_frames):
            # Wait for the right time for this frame (exactly like old code)
            target_time = start_time + frame_idx * frame_interval
            current_time = time.time()
            if current_time < target_time:
                time.sleep(target_time - current_time)
            
            # Set frame count on game object for synchronization
            game.frame_count = frame_idx
            
            # Get current state and render FIRST
            curr_state = self._serialize_ui_state(session, game)
            prev_state = prev_state or curr_state
            frame = renderer.render_to_array(prev_state, curr_state, 0.0)
            
            # Log state BEFORE checking action completions to ensure consistent position capture
            data_logger.log_state(frame_idx, game, self.config.engine_tick_rate)
            
            # Check for completed actions after state is logged
            self._check_action_completions(game)
            
            # Record video
            if video_recorder:
                video_recorder.write_frame_with_hud(frame, game)
            
            prev_state = curr_state
            
            # Progress reporting (adjust for initialization period)
            simulation_time = frame_idx / self.config.engine_tick_rate
            if frame_idx + 1 >= next_progress:
                percent = int(100 * (frame_idx + 1) / total_frames)
                if simulation_time < self.config.agent_initialization_period:
                    status = f"(initialization: {simulation_time:.1f}s/{self.config.agent_initialization_period}s)"
                else:
                    active_time = simulation_time - self.config.agent_initialization_period
                    status = f"(active gameplay: {active_time:.1f}s/{self.config.duration_seconds}s)"
                print(f"Simulation progress: {percent}% ({frame_idx + 1}/{total_frames} frames) {status}")
                sys.stdout.flush()
                next_progress += progress_step
    
    def _check_action_completions(self, game: Any):
        """Check if any agents have completed their actions and should end them."""
        # Note: This method is simplified since we no longer use ActionTracker
        # Actions are now logged directly when they occur in the controller
        pass
    
    def _serialize_ui_state(self, session: Any, game: Any) -> Dict:
        """Serialize UI state for rendering."""
        payload = {}
        for module in session.ui_modules:
            payload.update(module.serialize_for_agent(game, session.engine, None))
        payload['tick'] = session.engine.tick_count
        return payload
    
    def _cleanup_simulation(self, video_recorder: Optional[VideoRecorder],
                          engine_app: Any, game: Any, engine: Any,
                          data_logger: DataLogger):
        """Cleanup simulation resources."""
        print("Cleaning up simulation...")
        
        # Stop video recording
        if video_recorder:
            print("Stopping video recorder...")
            video_recorder.stop()
        
        # Stop engine
        print("Stopping engine...")
        self._stop_engine_with_timeout(engine_app)
        
        print("Simulation cleanup completed")
    
    def _stop_engine_with_timeout(self, engine_app: Any, timeout: float = 5.0):
        """Stop engine with timeout to prevent hanging."""
        try:
            # First try direct stop without threading
            engine_app.stop()
            print("Engine stopped successfully")
        except Exception as direct_error:
            print(f"Direct engine stop failed: {direct_error}, trying with timeout thread...")
            
            # Only use threading as fallback
            engine_stop_completed = threading.Event()
            engine_stop_result = {"error": None}
            
            def stop_engine():
                try:
                    engine_app.stop()
                    engine_stop_completed.set()
                except Exception as e:
                    engine_stop_result["error"] = e
                    engine_stop_completed.set()
            
            try:
                # Create and start thread - only when needed
                stop_thread = threading.Thread(target=stop_engine)
                stop_thread.daemon = True
                stop_thread.start()
                
                # Wait with timeout
                if engine_stop_completed.wait(timeout=timeout):
                    if engine_stop_result["error"]:
                        print(f"Error stopping engine: {engine_stop_result['error']}")
                    else:
                        print("Engine stopped successfully via thread")
                else:
                    print(f"Engine stop timed out after {timeout} seconds")
                    
            except RuntimeError as e:
                if "threads can only be started once" in str(e):
                    print("Threading error encountered, engine may already be stopped")
                else:
                    print(f"Threading error during engine stop: {e}")
            except Exception as e:
                print(f"Unexpected error during threaded engine stop: {e}")


def setup_simulation_argument_parser() -> argparse.ArgumentParser:
    """
    Set up command line argument parser for simulations.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description='Run reinforcement learning simulations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'map_nr',
        type=str,
        help='Map name identifier'
    )
    
    parser.add_argument(
        'num_agents',
        type=int,
        help='Number of agents in the simulation'
    )
    
    parser.add_argument(
        'intent_version',
        type=str,
        help='Intent version identifier'
    )
    
    parser.add_argument(
        'cooperative',
        type=int,
        choices=[0, 1],
        help='Whether simulation is cooperative (1) or competitive (0)'
    )
    
    parser.add_argument(
        'game_version',
        type=str,
        help='Game version identifier'
    )
    
    parser.add_argument(
        'training_id',
        type=str,
        help='Training identifier'
    )
    
    parser.add_argument(
        'checkpoint_number',
        type=str,
        help='Checkpoint number to load (integer) or "final" for the latest checkpoint'
    )
    
    parser.add_argument(
        '--enable-video',
        type=str,
        choices=['true', 'false'],
        default='true',
        help='Enable video recording'
    )
    
    parser.add_argument(
        '--cluster',
        type=str,
        choices=['brigit', 'cuenca', 'local'],
        default='cuenca',
        help='Cluster type for path configuration'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=180,
        help='Simulation duration in seconds'
    )
    
    parser.add_argument(
        '--tick-rate',
        type=int,
        default=24,
        help='Engine tick rate (frames per second)'
    )
    
    parser.add_argument(
        '--video-fps',
        type=int,
        default=24,
        help='Video recording frame rate'
    )
    
    return parser


def main_simulation_pipeline(map_nr: str, num_agents: int, intent_version: str,
                           cooperative: bool, game_version: str, training_id: str,
                           checkpoint_number: str, enable_video: bool = True,
                           cluster: str = 'cuenca', duration: int = 180,
                           tick_rate: int = 24, video_fps: int = 24) -> Dict[str, Path]:
    """
    Main simulation pipeline that can be used by different simulation scripts.
    
    Args:
        MAP_NR: Map name identifier
        NUM_AGENTS: Number of agents
        intent_version: Intent version identifier
        cooperative: Whether simulation is cooperative
        game_version: Game version identifier
        training_id: Training identifier
        checkpoint_number: Checkpoint number (integer or "final")
        enable_video: Whether to enable video recording
        cluster: Cluster type
        duration: Simulation duration in seconds
        tick_rate: Engine tick rate
        video_fps: Video frame rate
        
    Returns:
        Dictionary containing output file paths
    """
    # Create configuration
    config = SimulationConfig(
        cluster=cluster,
        engine_tick_rate=tick_rate,
        duration_seconds=duration,
        enable_video=enable_video,
        video_fps=video_fps
    )
    
    # Create timestamp
    timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")
    
    # Create and run simulation
    runner = SimulationRunner(config)
    
    output_paths = runner.run_simulation(
        map_nr=map_nr,
        num_agents=num_agents,
        intent_version=intent_version,
        cooperative=cooperative,
        game_version=game_version,
        training_id=training_id,
        checkpoint_number=checkpoint_number,
        timestamp=timestamp
    )
    
    print(f"Simulation completed successfully!")
    print(f"Simulation directory: {output_paths['simulation_dir']}")
    print(f"Configuration file: {output_paths['config_file']}")
    print(f"State CSV: {output_paths['state_csv']}")
    print(f"Action CSV: {output_paths['action_csv']}")
    if output_paths['video_file']:
        print(f"Video file: {output_paths['video_file']}")
    
    return output_paths