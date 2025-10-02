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

from engine.app.ai_only_app import AIOnlySessionApp
from engine.extensions.renderer2d.renderer_2d_offline import Renderer2DOffline
from engine.extensions.renderer2d.renderer_ui import Renderer2DModule


@dataclass
class SimulationConfig:
    """Configuration class for simulation parameters."""
    
    # Cluster settings
    cluster: str = 'cuenca'
    cluster_paths: Dict[str, str] = None
    
    # Simulation timing
    engine_tick_rate: int = 24
    ai_tick_rate: int = 1
    duration_seconds: int = 180
    agent_speed_px_per_sec: int = 32
    
    # Video settings
    enable_video: bool = True
    video_fps: int = 24
    
    # Grid settings
    default_grid_size: Tuple[int, int] = (8, 8)
    tile_size: int = 16
    
    def __post_init__(self):
        if self.cluster_paths is None:
            self.cluster_paths = {
                'brigit': '/mnt/lustre/home/samuloza',
                'cuenca': '',
                'local': 'D:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado'
            }
    
    def validate_cluster(self):
        """Validate cluster configuration."""
        if self.cluster not in self.cluster_paths:
            raise ValueError(f"Invalid cluster '{self.cluster}'. Choose from {list(self.cluster_paths.keys())}")
    
    @property
    def local_path(self) -> str:
        """Get the local path for the configured cluster."""
        self.validate_cluster()
        return self.cluster_paths[self.cluster]
    
    @property
    def total_frames(self) -> int:
        """Calculate total frames for the simulation."""
        return self.duration_seconds * self.engine_tick_rate


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


class ActionTracker:
    """Tracks agent actions with detailed logging and timing."""
    
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.lock = threading.Lock()
        
        # Track active actions (no longer need completion times for minimum duration)
        self.active_actions = {}
        self.action_completion_times = {}  # Keep for compatibility but not used for duration enforcement
        
        # Action counters per agent (for action_number)
        self.action_counters = {}
        
        # Action ID counters per agent (for action_id - 0-indexed per agent)
        self.action_id_counters = {}
        
        # Store completed actions for writing to CSV
        self.completed_actions = []
        
        self._initialize_csv()
    
    @staticmethod
    def _position_to_tile(x: float, y: float) -> tuple[int, int]:
        """Convert pixel position to tile coordinates consistently."""
        # Use consistent tile calculation: floor division by tile size + 1
        # This ensures all components use the same calculation
        if x is None or y is None:
            return 0, 0
        tile_x = int(x // 16 + 1)
        tile_y = int(y // 16 + 1)
        return tile_x, tile_y
    
    def _initialize_csv(self):
        """Initialize the action tracking CSV file."""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'action_id', 'agent_id', 'action_number', 'action_type', 
                'target_tile_type', 'target_tile_x', 'target_tile_y'
            ])
    
    def _get_tile_info(self, game: Any, target_id: str) -> Tuple[str, Optional[int], Optional[int]]:
        """Get tile information for a given target ID."""
        tile_type, x_tile, y_tile = '', None, None
        try:
            grid = getattr(game, 'grid', None)
            if grid is not None and hasattr(grid, 'tiles'):
                for x in range(grid.width):
                    for y in range(grid.height):
                        t = grid.tiles[x][y]
                        if t and getattr(t, 'id', None) == target_id:
                            tile_type = t.__class__.__name__.lower()
                            x_tile, y_tile = x + 1, y + 1
                            return tile_type, x_tile, y_tile
        except Exception:
            pass
        return tile_type, x_tile, y_tile
    
    def _extract_action_info(self, action: Any, game: Any) -> Tuple[str, str, str, Optional[int], Optional[int]]:
        """Extract action type and target information."""
        action_type = ''
        target_id = ''
        tile_type = ''
        x_tile = None
        y_tile = None
        
        try:
            if action is None:
                return 'do_nothing', '', '', None, None
            
            if isinstance(action, dict):
                action_type = str(action.get('type', action.get('action', 'unknown')))
                target_id = str(action.get('target', ''))
            elif hasattr(action, 'action_type') or hasattr(action, 'type'):
                action_type = str(getattr(action, 'action_type', getattr(action, 'type', 'unknown')))
                target_id = str(getattr(action, 'target', ''))
            else:
                action_type = str(action)
            
            if target_id:
                tile_type, x_tile, y_tile = self._get_tile_info(game, target_id)
                
        except Exception as e:
            action_type = 'unknown'
            print(f"DEBUG: Error extracting action info: {e}")
            
        return action_type, target_id, tile_type, x_tile, y_tile
    
    def start_action(self, agent_id: str, action_type: str, tile_or_target: Any, 
                    timestamp: float, action_number: Optional[int] = None, **kwargs):
        """Start tracking an action for an agent."""
        with self.lock:
            if agent_id in self.active_actions:
                # Log a warning if the agent is already performing an action
                print(f"[ACTION_TRACKER] Agent {agent_id} is already performing action "
                      f"'{self.active_actions[agent_id].get('action_type', 'unknown')}' and cannot start a new one.")
                return

            game = getattr(self, 'game', None)
            engine = getattr(self, 'engine', None)
            
            if game is None:
                print(f"[ACTION_TRACKER] No game reference for {agent_id}")
                return
            
            agent_obj = game.gameObjects.get(agent_id)
            
            if agent_obj is None:
                print(f"[ACTION_TRACKER] No agent object found for {agent_id}")
                return
            
            # Create action dict for processing
            if action_type == "click" and tile_or_target:
                action = {"type": "click", "target": getattr(tile_or_target, 'id', str(tile_or_target))}
            elif action_type == "do_nothing":
                action = None
            else:
                action = {"type": action_type}
            
            # Extract action information
            action_type_str, target_id, tile_type, x_tile, y_tile = self._extract_action_info(action, game)
            
            # Get agent current state at action start time
            agent_x = getattr(agent_obj, 'x', None)
            agent_y = getattr(agent_obj, 'y', None)
            # Capture item state BEFORE the action starts
            item_before = getattr(agent_obj, 'item', None)
            
            # Use provided action number or generate new one
            if action_type == "do_nothing":
                # do_nothing actions should always have action_number = -1
                action_number = -1
            elif action_number is None or action_number == -1:
                # Generate new action number for regular actions
                self.action_counters[agent_id] = self.action_counters.get(agent_id, 0) + 1
                action_number = self.action_counters[agent_id]
            
            # Create action record
            action_record = {
                'action': action,
                'action_type': action_type_str,
                'target_id': target_id,
                'tile_type': tile_type,
                'target_x_tile': x_tile,
                'target_y_tile': y_tile,
                'decision_timestamp': timestamp,
                'decision_x': agent_x,
                'decision_y': agent_y,
                'item_before': item_before,
                'action_number': action_number
            }
            
            # Store active action
            self.active_actions[agent_id] = action_record
            
            # No minimum duration enforcement - actions complete when the game engine says they're complete
            # Remove the action from completion times tracking
            self.action_completion_times.pop(agent_id, None)
                
    # Removed can_complete_action method - actions complete when game engine says they're complete
    
    def end_action(self, agent_id: str, timestamp: Optional[float] = None, current_frame: Optional[int] = None, **kwargs) -> bool:
        """End tracking an action for an agent."""
        with self.lock:
            if agent_id not in self.active_actions:
                return False
            
            action_record = self.active_actions[agent_id]
            game = getattr(self, 'game', None)
            engine = getattr(self, 'engine', None)
            
            if game is None or engine is None:
                self.active_actions.pop(agent_id, None)
                self.action_completion_times.pop(agent_id, None)
                return False
            
            # Use current timestamp if provided, otherwise get current time
            if timestamp is None:
                timestamp = time.time()
            
            # Actions complete when the game engine says they're complete - no duration checks
            
            # Get action_id for this agent (0-indexed per agent)
            if agent_id not in self.action_id_counters:
                self.action_id_counters[agent_id] = 0
            else:
                self.action_id_counters[agent_id] += 1
            
            action_id = self.action_id_counters[agent_id]
            
            # Write action immediately to CSV in simplified format
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                row = [
                    action_id,
                    agent_id,
                    action_record['action_number'],
                    action_record['action_type'],
                    action_record['tile_type'] or '',
                    action_record['target_x_tile'] or '',
                    action_record['target_y_tile'] or ''
                ]
                writer.writerow(row)
                        
            # Remove from active tracking
            self.active_actions.pop(agent_id, None)
            self.action_completion_times.pop(agent_id, None)
            return True
    
    def finalize_actions_with_log_data(self, log_csv_path: Path, tick_rate: int = 24):
        """
        Legacy method kept for compatibility - actions are now written directly during end_action.
        """
        print("[ACTION_TRACKER] Actions are now written directly to CSV - no finalization needed")
        pass
    

    def cleanup(self, game: Any, engine: Any, max_cleanup_time: float = 3.0):
        """Complete any remaining active actions at simulation end."""
        print(f"[ACTION_TRACKER] Starting cleanup for {len(self.active_actions)} active actions")
        
        cleanup_start_time = time.time()
        
        try:
            lock_acquired = self.lock.acquire(timeout=2.0)
            if not lock_acquired:
                print(f"[ACTION_TRACKER] Could not acquire lock for cleanup")
                self.active_actions.clear()
                self.action_completion_times.clear()
                return
            
            try:
                active_agents = list(self.active_actions.keys())
                for agent_id in active_agents:
                    if time.time() - cleanup_start_time > max_cleanup_time:
                        print(f"[ACTION_TRACKER] Cleanup timeout reached")
                        self.active_actions.clear()
                        self.action_completion_times.clear()
                        break
                    
                try:
                    # At simulation end, complete any remaining actions with final timestamp
                    final_timestamp = time.time()
                    print(f"[ACTION_TRACKER] Completing remaining action for {agent_id} at simulation end")
                    self._force_complete_action(agent_id, game, engine, final_timestamp)
                except Exception as action_error:
                    print(f"[ACTION_TRACKER] Error completing action for {agent_id}: {action_error}")
                    # Manually remove from tracking if completion fails
                    self.active_actions.pop(agent_id, None)
                    self.action_completion_times.pop(agent_id, None)
                
            finally:
                self.lock.release()
                
        except Exception as e:
            print(f"[ACTION_TRACKER] Critical error during cleanup: {e}")
            self.active_actions.clear()
            self.action_completion_times.clear()
    
    def _force_complete_action(self, agent_id: str, game: Any, engine: Any, final_timestamp: Optional[float] = None):
        """Force complete an action during cleanup."""
        if agent_id not in self.active_actions:
            return
        
        action_record = self.active_actions[agent_id]
        
        # Use provided final_timestamp or get current time
        if final_timestamp is not None:
            completion_timestamp = final_timestamp
        else:
            completion_timestamp = time.time()
        
        agent_obj = game.gameObjects.get(agent_id) if game else None
        
        # Get completion state
        if agent_obj:
            agent_x = getattr(agent_obj, 'x', action_record['decision_x'] or 0)
            agent_y = getattr(agent_obj, 'y', action_record['decision_y'] or 0)
            item_after = getattr(agent_obj, 'item', None)
        else:
            agent_x = action_record['decision_x'] or 0
            agent_y = action_record['decision_y'] or 0  
            item_after = action_record['item_before']
        
        # Convert positions to tile coordinates
        decision_tile_x, decision_tile_y = self._position_to_tile(
            action_record['decision_x'], action_record['decision_y']
        )
        completion_tile_x, completion_tile_y = self._position_to_tile(agent_x, agent_y)
        
        # Store completed action data WITHOUT frame numbers (to be added later)
        completed_action = {
            'agent_id': agent_id,
            'action_number': action_record['action_number'],
            'action_type': action_record['action_type'],
            'target_tile_type': action_record.get('tile_type') or 'None',
            'target_tile_x': action_record.get('target_x_tile') or 0,
            'target_tile_y': action_record.get('target_y_tile') or 0,
            'decision_x': action_record['decision_x'],
            'decision_y': action_record['decision_y'],
            'decision_tile_x': decision_tile_x,
            'decision_tile_y': decision_tile_y,
            'completion_x': agent_x,
            'completion_y': agent_y,
            'completion_tile_x': completion_tile_x,
            'completion_tile_y': completion_tile_y,
            'item_before': action_record['item_before'],
            'item_after': item_after,
            'decision_timestamp': action_record['decision_timestamp'],
            'completion_timestamp': completion_timestamp
        }
        
        # Add to completed actions list
        self.completed_actions.append(completed_action)
        
        # Remove from tracking
        self.active_actions.pop(agent_id, None)
        self.action_completion_times.pop(agent_id, None)


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
                'MAP_FILE': str(paths['map_txt_path'])
            }
            
            data_logger = DataLogger(paths['saving_path'], checkpoint_number, timestamp, simulation_config)
            action_tracker = ActionTracker(data_logger.action_csv_path)
            
            # Setup game
            game_factory = self.game_manager.create_game_factory(
                map_nr, grid_size, intent_version
            )
            
            # Run the simulation
            output_paths = self._execute_simulation(
                game_factory, controllers, paths, data_logger, 
                action_tracker, timestamp
            )
            
            return output_paths
            
        finally:
            self.ray_manager.shutdown_ray()
    
    def _execute_simulation(self, game_factory: Callable, controllers: Dict,
                           paths: Dict, data_logger: DataLogger,
                           action_tracker: ActionTracker, timestamp: str) -> Dict[str, Path]:
        """Execute the main simulation loop."""
        
        # Create engine app
        engine_app = AIOnlySessionApp(
            game_factory=game_factory,
            ui_modules=[Renderer2DModule()],
            agent_map=controllers,
            path_root=paths['path_root'],
            tick_rate=self.config.engine_tick_rate,
            ai_tick_rate=self.config.ai_tick_rate,
            is_max_speed=False
        )
        
        session = engine_app.get_session()
        game = session.engine.game
        
        # Attach action tracker
        game.action_tracker = action_tracker
        action_tracker.game = game
        action_tracker.engine = session.engine
        
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
                action_tracker, video_recorder, engine_app, 
                game, session.engine, data_logger
            )
        
        return {
            'simulation_dir': data_logger.simulation_dir,
            'config_file': data_logger.config_path,
            'state_csv': data_logger.state_csv_path,
            'action_csv': data_logger.action_csv_path,
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
        
        print(f"Starting simulation with {total_frames} frames ({self.config.duration_seconds} seconds)")
        
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
            
            # Progress reporting
            if frame_idx + 1 >= next_progress:
                percent = int(100 * (frame_idx + 1) / total_frames)
                print(f"Simulation progress: {percent}% ({frame_idx + 1}/{total_frames} frames)")
                sys.stdout.flush()
                next_progress += progress_step
    
    def _check_action_completions(self, game: Any):
        """Check if any agents have completed their actions and should end them."""
        if not hasattr(game, 'action_tracker'):
            return
            
        action_tracker = game.action_tracker
        current_timestamp = time.time()
        
        # Check each agent to see if their action should be completed
        for agent_id, agent_obj in game.gameObjects.items():
            if not agent_id.startswith('ai_rl_'):
                continue
                
            # Check if agent has completed their current action
            if (hasattr(agent_obj, 'action_complete') and 
                getattr(agent_obj, 'action_complete', False) and
                agent_id in action_tracker.active_actions):
                
                # Try to end the action with current timestamp
                action_ended = action_tracker.end_action(agent_id, current_timestamp)
                if action_ended:
                    # Clear the agent's action state
                    agent_obj.current_action = None
                    # Reset action_complete flag
                    agent_obj.action_complete = False
    
    def _serialize_ui_state(self, session: Any, game: Any) -> Dict:
        """Serialize UI state for rendering."""
        payload = {}
        for module in session.ui_modules:
            payload.update(module.serialize_for_agent(game, session.engine, None))
        payload['tick'] = session.engine.tick_count
        return payload
    
    def _cleanup_simulation(self, action_tracker: ActionTracker, 
                          video_recorder: Optional[VideoRecorder],
                          engine_app: Any, game: Any, engine: Any,
                          data_logger: DataLogger):
        """Cleanup simulation resources."""
        print("Cleaning up simulation...")
        
        # Stop video recording
        if video_recorder:
            print("Stopping video recorder...")
            video_recorder.stop()
        
        # Cleanup actions
        print("Cleaning up actions...")
        try:
            action_tracker.cleanup(game, engine)
        except Exception as e:
            print(f"Error during action cleanup: {e}")
        
        # Finalize actions with precise frame timing from logs
        print("Finalizing actions with log data...")
        try:
            action_tracker.finalize_actions_with_log_data(
                data_logger.state_csv_path, 
                tick_rate=self.config.engine_tick_rate
            )
        except Exception as e:
            print(f"Error during action finalization: {e}")
        
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