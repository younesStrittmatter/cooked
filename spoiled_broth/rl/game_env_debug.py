import csv
import os
from pathlib import Path
from datetime import datetime
import numpy as np
from spoiled_broth.rl.game_env import GameEnv
from spoiled_broth.rl.observation_space import game_to_obs_vector
from spoiled_broth.rl.action_space import get_rl_action_space, convert_action_to_tile
from spoiled_broth.world.tiles import Counter


class GameEnvDebug(GameEnv):
    """
    Debug version of GameEnv that captures detailed episode data.
    
    This extends the regular GameEnv to save simulation data for each episode:
    - State data (agent positions, items, scores)
    - Action data (actions taken by each agent)
    - Observation data (observation vectors for each agent)
    - Counter data (items on counters throughout the episode)
    
    All data is saved in episode-specific directories for detailed analysis.
    """
    
    def __init__(self, debug_mode=False, debug_capture_episodes=False, 
                 episodes_dir="", debug_max_episodes_to_capture=100, **kwargs):
        super().__init__(**kwargs)
        
        self.debug_mode = debug_mode
        self.debug_capture_episodes = debug_capture_episodes
        self.episodes_dir = Path(episodes_dir) if episodes_dir else None
        self.debug_max_episodes_to_capture = debug_max_episodes_to_capture
        self.episodes_captured = 0
        
        # Episode-specific data tracking
        self.current_episode_dir = None
        self.episode_states = []
        self.episode_actions = []
        self.episode_observations = []
        self.episode_counters = []
        self.frame_count = 0
        
        # Item state tracking for debugging delivery/salad mismatch
        self.previous_agent_items = {}
        
        # Episode summary data
        self.last_episode_data = {}
        
        print(f"GameEnvDebug initialized:")
        print(f"  Debug mode: {self.debug_mode}")
        print(f"  Capture episodes: {self.debug_capture_episodes}")
        print(f"  Episodes directory: {self.episodes_dir}")
        print(f"  Max episodes to capture: {self.debug_max_episodes_to_capture}")

    def reset(self, seed=None, options=None):
        """Reset environment and start new episode data capture if enabled."""
        # Save previous episode data if we were capturing
        if (self.debug_mode and self.debug_capture_episodes and 
            self.current_episode_dir is not None and 
            (self.episode_states or self.episode_actions)):
            self._save_episode_data()
        
        # Start new episode capture if enabled and within limits
        should_capture = (self.debug_mode and 
                         self.debug_capture_episodes and 
                         self.episodes_captured < self.debug_max_episodes_to_capture and
                         self.episodes_dir is not None)
        
        if should_capture:
            self._start_episode_capture()
        else:
            self.current_episode_dir = None
        
        # Reset frame count
        self.frame_count = 0
        
        # Call parent reset
        observations, infos = super().reset(seed, options)
        
        # Log initial state and observations if capturing
        if self.current_episode_dir is not None:
            self._log_episode_state()
            self._log_episode_observations(observations)
            self._log_episode_counters()
        
        return observations, infos

    def step(self, actions):
        """Step environment and log debug data if enabled."""
        # Store agent items before step for change tracking
        if self.current_episode_dir is not None:
            self._track_agent_items_before_step()
        
        # Store original actions for logging reference
        self._original_actions = actions.copy()
        
        # Call parent step to get the actual validated_actions
        # We need to capture the validated_actions from the parent class
        observations, rewards, terminations, truncations, infos = super().step(actions)
        
        # Log validated actions after step if capturing 
        # Now we can access the validated_actions from the parent step method
        if self.current_episode_dir is not None:
            self._log_validated_actions()
            
        # Increment frame count
        self.frame_count += 1
        
        # Log state after step if capturing
        if self.current_episode_dir is not None:
            self._log_episode_state()
            self._log_episode_observations(observations)
            self._log_episode_counters()
            self._log_item_state_changes()
        
        # If episode is done, prepare summary data
        if any(terminations.values()) or any(truncations.values()):
            self._prepare_episode_summary(rewards, infos)
        
        return observations, rewards, terminations, truncations, infos

    def _start_episode_capture(self):
        """Initialize episode data capture."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        episode_name = f"episode_{self.episode_count}_{timestamp}"
        self.current_episode_dir = self.episodes_dir / episode_name
        os.makedirs(self.current_episode_dir, exist_ok=True)
        
        # Initialize episode data lists
        self.episode_states = []
        self.episode_actions = []
        self.episode_observations = []
        self.episode_counters = []
        
        # Create CSV headers
        self._initialize_episode_csvs()
        
        print(f"Started episode capture {self.episodes_captured + 1}: {episode_name}")

    def _initialize_episode_csvs(self):
        """Initialize CSV files for episode data."""
        # State CSV
        state_csv_path = self.current_episode_dir / "simulation.csv"
        with open(state_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["agent_id", "frame", "second", "x", "y", "tile_x", "tile_y", "item", "score"])
        
        # Action CSV
        action_csv_path = self.current_episode_dir / "actions.csv"
        with open(action_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["agent_id", "frame", "second", "action_index", "action_name", "tile_index", "tile_x", "tile_y"])
        
        # Observation CSV
        obs_csv_path = self.current_episode_dir / "observations.csv"
        with open(obs_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Determine observation size from one agent
            obs_example = self.observe(self.agents[0]) if self.agents else np.array([])
            obs_columns = [f"obs_{i}" for i in range(len(obs_example))]
            writer.writerow(["agent_id", "frame", "second"] + obs_columns)
        
        # Counter CSV - will be initialized when first counter data is logged
        self._counter_csv_initialized = False

    def _log_episode_state(self):
        """Log current state to episode data."""
        if self.current_episode_dir is None:
            return
            
        state_csv_path = self.current_episode_dir / "simulation.csv"
        with open(state_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for agent_id, obj in self.game.gameObjects.items():
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
                        self.frame_count, 
                        self._elapsed_time,  # Use actual episode time
                        agent_x, 
                        agent_y,
                        tile_x, 
                        tile_y,
                        getattr(obj, 'item', ''), 
                        getattr(obj, 'score', 0)
                    ])

    def _log_validated_actions(self):
        """Log only the validated actions (those that were actually processed by the game) to episode data."""
        if self.current_episode_dir is None:
            return
            
        action_csv_path = self.current_episode_dir / "actions.csv"
        with open(action_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Only log actions that were actually validated by the parent GameEnv
            if hasattr(self, '_logging_actions'):
                for agent_id, validated_action in self._logging_actions.items():
                    elapsed_time = validated_action["elapsed_time"]

                    # Extract validated action data
                    action_idx = validated_action['action_idx']
                    action_name = validated_action['action_name'] 
                    tile_index = validated_action['tile_index']
                    
                    # Calculate tile coordinates from validated data
                    tile_x = validated_action.get('x', -1)
                    tile_y = validated_action.get('y', -1)
                    
                    writer.writerow([
                        agent_id, 
                        self.frame_count,
                        elapsed_time,
                        action_idx,
                        action_name,
                        tile_index,
                        tile_x,
                        tile_y
                    ])

    def _log_episode_observations(self, observations):
        """Log observations to episode data."""
        if self.current_episode_dir is None:
            return
            
        obs_csv_path = self.current_episode_dir / "observations.csv"
        with open(obs_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for agent_id, obs in observations.items():
                obs_list = obs.flatten().tolist() if hasattr(obs, 'flatten') else [obs]
                writer.writerow([
                    agent_id,
                    self.frame_count,
                    self._elapsed_time,  # Use actual episode time
                ] + obs_list)

    def _log_episode_counters(self):
        """Log counter states to episode data."""
        if self.current_episode_dir is None:
            return
            
        # Initialize counter CSV on first call
        if not self._counter_csv_initialized:
            self._initialize_counter_csv()
        
        counter_csv_path = self.current_episode_dir / "counters.csv"
        
        # Collect counter items
        counter_data = [self.frame_count, self._elapsed_time]  # Start with frame and time
        
        grid = getattr(self.game, 'grid', None)
        if grid is not None and hasattr(self, '_counter_positions'):
            # Create a mapping of positions to items for quick lookup
            position_to_item = {}
            
            try:
                # Import Counter class to check isinstance                
                for x in range(grid.width):
                    for y in range(grid.height):
                        tile = grid.tiles[x][y]
                        if isinstance(tile, Counter):
                            # Use 1-indexed coordinates for consistency
                            pos = (x + 1, y + 1)
                            item = getattr(tile, 'item', None)
                            position_to_item[pos] = item if item is not None else ''
            except ImportError:
                # Fallback if Counter class is not available
                for x in range(grid.width):
                    for y in range(grid.height):
                        tile = grid.tiles[x][y]
                        if hasattr(tile, '_type') and tile._type == 2:  # Assuming type 2 is counter
                            pos = (x + 1, y + 1)
                            item = getattr(tile, 'item', None)
                            position_to_item[pos] = item if item is not None else ''
            
            # Add items in the same order as headers
            for pos in self._counter_positions:
                counter_data.append(position_to_item.get(pos, ''))
        
        # Write to CSV
        with open(counter_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(counter_data)

    def _initialize_counter_csv(self):
        """Initialize the counter CSV for this episode."""
        # Find all counter positions in the game
        self._counter_positions = []
        grid = getattr(self.game, 'grid', None)
        if grid is not None:
            try:
                # Import Counter class to check isinstance                
                for x in range(grid.width):
                    for y in range(grid.height):
                        tile = grid.tiles[x][y]
                        if isinstance(tile, Counter):
                            # Store counter position (1-indexed for consistency)
                            counter_pos = (x + 1, y + 1)
                            self._counter_positions.append(counter_pos)
            except ImportError:
                # If Counter class is not available, try to identify counter tiles by type
                for x in range(grid.width):
                    for y in range(grid.height):
                        tile = grid.tiles[x][y]
                        if hasattr(tile, '_type') and tile._type == 2:  # Assuming type 2 is counter
                            counter_pos = (x + 1, y + 1)
                            self._counter_positions.append(counter_pos)
        
        # Sort counter positions for consistent ordering
        self._counter_positions.sort()
        
        # Create CSV header
        header = ["frame", "second"]
        for x, y in self._counter_positions:
            header.append(f"counter_{x}_{y}")
        
        # Write header to CSV
        counter_csv_path = self.current_episode_dir / "counters.csv"
        with open(counter_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
        
        self._counter_csv_initialized = True

    def _track_agent_items_before_step(self):
        """Track agent items before step to detect changes."""
        self.previous_agent_items = {}
        for agent_id, obj in self.game.gameObjects.items():
            if agent_id.startswith('ai_rl_'):
                self.previous_agent_items[agent_id] = getattr(obj, 'item', None)

    def _log_item_state_changes(self):
        """Log detailed item state changes to help debug delivery/salad mismatches."""
        if self.current_episode_dir is None:
            return
            
        # Create item changes log file if it doesn't exist
        item_changes_csv_path = self.current_episode_dir / "item_changes.csv"
        file_exists = item_changes_csv_path.exists()
        
        with open(item_changes_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Write header if first time
            if not file_exists:
                writer.writerow([
                    "agent_id", "frame", "second", "action_idx", "action_name", "tile_index", "tile_x", "tile_y",
                    "item_before", "item_after", "item_changed", "change_type",
                    "agent_score_before", "agent_score_after", "score_changed"
                ])
            
            # Check each agent for changes - only log for agents that had validated actions
            for agent_id, obj in self.game.gameObjects.items():
                if (agent_id.startswith('ai_rl_') and 
                    hasattr(self, '_original_actions') and 
                    agent_id in self._original_actions):
                    
                    current_item = getattr(obj, 'item', None)
                    previous_item = self.previous_agent_items.get(agent_id, None)
                    current_score = getattr(obj, 'score', 0)
                    
                    # Determine change type
                    item_changed = previous_item != current_item
                    change_type = "none"
                    if item_changed:
                        if previous_item is None and current_item is not None:
                            change_type = "picked_up"
                        elif previous_item is not None and current_item is None:
                            change_type = "dropped_or_consumed"
                        elif previous_item != current_item:
                            change_type = "item_transformed"
                    
                    # Default action info (for cases where action wasn't validated)
                    action_idx = self._original_actions[agent_id] if hasattr(self, '_original_actions') else -1
                    action_space = get_rl_action_space(self.game_mode)
                    action_name = action_space[action_idx] if 0 <= action_idx < len(action_space) else f"unknown_{action_idx}"
                    
                    # Get tile information - use validated action data if available
                    tile_index = -1
                    tile_x = -1
                    tile_y = -1
                    
                    if hasattr(self, '_logging_actions') and agent_id in self._logging_actions:
                        validated_action = self._logging_actions[agent_id]
                        action_idx = validated_action['action_idx']
                        action_name = validated_action['action_name']
                        tile_index = validated_action['tile_index']
                        tile_x = validated_action.get('x', -1)
                        tile_y = validated_action.get('y', -1)
                    
                    # Log the change (even if no change, for completeness)
                    writer.writerow([
                        agent_id, 
                        self.frame_count, 
                        self._elapsed_time,
                        action_idx,
                        action_name,
                        tile_index,
                        tile_x,
                        tile_y,
                        previous_item or "",
                        current_item or "",
                        item_changed,
                        change_type,
                        self.previous_agent_items.get(f"{agent_id}_score", 0),
                        current_score,
                        current_score != self.previous_agent_items.get(f"{agent_id}_score", 0)
                    ])
                    
                    # Store current score for next iteration
                    self.previous_agent_items[f"{agent_id}_score"] = current_score

    def _prepare_episode_summary(self, rewards, infos):
        """Prepare summary data for this episode."""
        self.last_episode_data = {
            'episode_number': self.episode_count,
            'episode_duration_seconds': self._elapsed_time,  # Use actual episode time
            'total_frames': self.frame_count,
        }
        
        # Add per-agent data
        for agent_id in self.agents:
            agent_obj = self.agent_map.get(agent_id)
            if agent_obj:
                self.last_episode_data[f'final_score_{agent_id}'] = getattr(agent_obj, 'score', 0)
                
            # Add reward information
            if agent_id in rewards:
                self.last_episode_data[f'final_reward_{agent_id}'] = rewards[agent_id]
            
            # Add cumulative rewards
            self.last_episode_data[f'total_rewards_{agent_id}'] = self.cumulated_modified_rewards.get(agent_id, 0)
            
            # Add action statistics
            self.last_episode_data[f'total_actions_{agent_id}'] = self.total_actions_asked.get(agent_id, 0)
            self.last_episode_data[f'actions_not_available_{agent_id}'] = self.total_actions_not_available.get(agent_id, 0)
            self.last_episode_data[f'inaccessible_actions_{agent_id}'] = self.total_actions_inaccessible.get(agent_id, 0)
            
            # Add event statistics
            if agent_id in self.total_agent_events:
                for event_type, count in self.total_agent_events[agent_id].items():
                    self.last_episode_data[f'{event_type}_{agent_id}'] = count

    def _save_episode_data(self):
        """Save episode configuration and summary data."""
        if self.current_episode_dir is None:
            return
        
        # Save episode configuration
        config_path = self.current_episode_dir / "config.txt"
        config_content = [
            f"# Episode Configuration",
            f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "[EPISODE_INFO]",
            f"EPISODE_NUMBER: {self.episode_count}",
            f"MAP_NR: {self.map_nr}",
            f"GAME_MODE: {self.game_mode}",
            f"NUM_AGENTS: {len(self.agents)}",
            f"EPISODE_DURATION_SECONDS: {self._max_seconds_per_episode}",
            f"ACTUAL_DURATION_SECONDS: {self._elapsed_time}",
            f"TOTAL_FRAMES: {self.frame_count}",
            "",
            "[AGENT_CONFIG]",
            f"WALKING_SPEEDS: {self.walking_speeds}",
            f"CUTTING_SPEEDS: {self.cutting_speeds}",
            f"REWARD_WEIGHTS: {self.reward_weights}",
            "",
            "[REWARD_CONFIG]",
            f"PENALTIES_CFG: {self.penalties_cfg}",
            f"REWARDS_CFG: {self.rewards_cfg}",
            "",
            "[FILES]",
            "STATE_CSV: simulation.csv",
            "ACTION_CSV: actions.csv", 
            "OBSERVATION_CSV: observations.csv",
            "COUNTER_CSV: counters.csv",
            "ITEM_CHANGES_CSV: item_changes.csv",
            "",
            "[EPISODE_SUMMARY]",
        ]
        
        # Add episode summary data
        if self.last_episode_data:
            for key, value in self.last_episode_data.items():
                config_content.append(f"{key}: {value}")
        
        with open(config_path, 'w') as f:
            f.write('\n'.join(config_content))
        
        self.episodes_captured += 1
        print(f"Saved episode data to: {self.current_episode_dir}")
        print(f"  Total episodes captured: {self.episodes_captured}")
        
        # Reset episode directory
        self.current_episode_dir = None