"""
Main simulation runner class for managing complete simulation execution.

Author: Samuel Lozano
"""

import time
import sys
import threading
from pathlib import Path
from typing import Dict, Any, Callable, Optional

from engine.app.ai_only_app import AIOnlySessionApp
from engine.extensions.renderer2d.renderer_2d_offline import Renderer2DOffline
from engine.extensions.renderer2d.renderer_ui import Renderer2DModule

from .simulation_config import SimulationConfig
from .path_manager import PathManager
from .controller_manager import ControllerManager
from .ray_manager import RayManager
from .game_manager import GameManager
from .data_logger import DataLogger
from .action_tracker import ActionTracker
from .video_recorder import VideoRecorder


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