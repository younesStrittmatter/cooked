import csv
import os
from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np
from spoiled_broth.config import *
from spoiled_broth.maps.accessibility_maps import get_accessibility_map
import pickle as _pickle
from spoiled_broth.rl.action_space import get_rl_action_space, convert_action_to_tile
from spoiled_broth.rl.observation_space import game_to_obs_vector
from spoiled_broth.rl.classify_action_type import get_action_type_and_agent_events, get_action_type_list, ACTION_TYPE_INACCESSIBLE
from spoiled_broth.rl.reward_analysis import get_rewards
from spoiled_broth.game import SpoiledBroth, random_game_state

BUSY_PENALTY = 0.01  # Penalty for being busy
USELESS_ACTION_PENALTY = 0.2  # Penalty for performing a useless action
DESTRUCTIVE_ACTION_PENALTY = 1.0  # Harsh penalty for performing a destructive action
INACCESSIBLE_TILE_PENALTY = 0.5  # Penalty for trying to access unreachable tiles
WAIT_FOR_ACTION_COMPLETION = True  # Flag to ensure actions complete before next step

REWARDS = {
    "raw_food": 0.2,
    "plate": 0.2,
    "counter": 0.5,
    "cut": 2.0,
    "salad": 5.0,
    "deliver": 10.0,
}

ACTIONS_OBSERVATION_MAPPING_CLASSIC = {
    0: 0, 1: 1, 2: 2, 3: 3, # Two dispensers, cutting board, delivery
    4: 4, 5: 5, # Free counter (closest and midpoint)
    6: 7, 7: 8, # Tomato on counter (closest and midpoint)
    8: 10, 9: 11, # Plate on counter (closest and midpoint)
    10: 13, 11: 14, # Tomato_cut on counter (closest and midpoint)
    12: 16, 13: 17, # Pumpkin on counter (closest and midpoint)
}

ACTIONS_OBSERVATION_MAPPING_COMPETITION = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, # Three dispensers, cutting board, delivery
    5: 5, 6: 6, # Free counter (closest and midpoint)
    7: 8, 8: 9, # Tomato on counter (closest and midpoint)
    9: 11, 10: 12, # Pumpkin on counter (closest and midpoint)
    11: 14, 12: 15, # Plate on counter (closest and midpoint)
    13: 17, 14: 18, # Tomato_cut on counter (closest and midpoint)
    15: 20, 16: 21, # Pumpkin_cut on counter (closest and midpoint)
    17: 23, 18: 24, # Tomato_salad on counter (closest and midpoint)
    19: 26, 20: 27, # Pumpkin_salad on counter (closest and midpoint)
}

class DistanceMatrixWrapper:
    """
    Lightweight wrapper that provides the same .get(from).get(to) lookup
    semantics as the original nested dict, but backed by a numpy matrix
    and index maps for speed.
    """
    def __init__(self, D, pos_from, pos_to):
        # D: numpy array shape (N_from, N_to) with np.nan for missing
        self.D = D
        # pos_from / pos_to are arrays of shape (N,2)
        self.pos_from = [tuple(p) for p in pos_from.tolist()]
        self.pos_to = [tuple(p) for p in pos_to.tolist()]
        self.pos_from_idx = {p: i for i, p in enumerate(self.pos_from)}
        self.pos_to_idx = {p: i for i, p in enumerate(self.pos_to)}

    def get(self, from_xy, default=None):
        i = self.pos_from_idx.get(from_xy)
        if i is None:
            return {}
        # Return a dict-like object for compatibility: mapping to_xy -> distance
        row = self.D[i]
        result = {}
        for j, val in enumerate(row):
            if not np.isnan(val):
                result[self.pos_to[j]] = float(val)
        return result

def init_game(agents, map_nr=1, grid_size=(8, 8), seed=None, game_mode="classic", walking_speeds=None, cutting_speeds=None):
    num_agents = len(agents)
    game = SpoiledBroth(map_nr=map_nr, grid_size=grid_size, num_agents=num_agents, seed=seed, walking_speeds=walking_speeds, cutting_speeds=cutting_speeds)
    clickable_indices = game.clickable_indices
    # New action space: fixed action space for RL agents
    action_spaces = {
        agent: spaces.Discrete(len(get_rl_action_space(game_mode)))
        for agent in agents
    }
    _clickable_mask = np.zeros(game.grid.width * game.grid.height, dtype=np.int8)
    for idx in clickable_indices:
        _clickable_mask[idx] = 1
    for agent_id in agents:
        walk_speed = walking_speeds.get(agent_id, 1) if walking_speeds else 1
        cut_speed = cutting_speeds.get(agent_id, 1) if cutting_speeds else 1
        game.add_agent(agent_id, walk_speed, cut_speed)
        agent = game.gameObjects[agent_id]
        if hasattr(agent, 'game'):
            agent.game.clickable_indices = clickable_indices
    return game, action_spaces, _clickable_mask, clickable_indices

class GameEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "game_v0"}

    def __init__(
        self, 
        reward_weights=None, 
        map_nr=1, 
        game_mode="classic",
        inner_seconds=180,
        path="training_stats.csv",
        grid_size=(8, 8),
        payoff_matrix=[1,1,-2],
        initial_seed=0,
        wait_for_completion=True,  # New parameter to control action completion waiting
        start_epoch=0,
        distance_map=None,
        walking_speeds=None,
        cutting_speeds=None
    ):
        super().__init__()
        self.map_nr = map_nr
        self.episode_count = start_epoch
        self.game_mode = game_mode
        self._max_seconds_per_episode = inner_seconds
        self._elapsed_time = 0.0
        self.render_mode = None
        self.write_header = True
        self.write_csv = False
        self.csv_path = os.path.join(path, "training_stats.csv")
        self.grid_size = grid_size
        self.seed = initial_seed
        self.payoff_matrix = payoff_matrix
        self.walking_speeds = walking_speeds
        self.cutting_speeds = cutting_speeds

        self.clickable_indices = None  # Initialize clickable indices storage
        self.distance_map = None
        if isinstance(distance_map, str):
            # Only support .npz path strings now
            try:
                # direct path
                if distance_map.endswith('.npz') and os.path.exists(distance_map):
                    data = np.load(distance_map)
                    D = data['D']
                    pos_from = data['pos_from']
                    pos_to = data['pos_to']
                    self.distance_map = DistanceMatrixWrapper(D, pos_from, pos_to)
                else:
                    # Try the repo cache folder
                    possible = os.path.join(os.path.dirname(__file__), '..', 'maps', 'distance_cache', distance_map)
                    if possible.endswith('.npz') and os.path.exists(possible):
                        data = np.load(possible)
                        D = data['D']
                        pos_from = data['pos_from']
                        pos_to = data['pos_to']
                        self.distance_map = DistanceMatrixWrapper(D, pos_from, pos_to)
                    else:
                        # if not found or not .npz, set None
                        self.distance_map = None
            except Exception:
                self.distance_map = None
        else:
            # allow direct dict or wrapper being passed (for tests); otherwise None
            self.distance_map = distance_map
                    
        # Load the accessibility map for this map
        self.accessibility_map = get_accessibility_map(map_nr)

        # Determine agent IDs from reward_weights or default to two agents
        if reward_weights is not None:
            self.possible_agents = list(reward_weights.keys())
        else:
            self.possible_agents = ["ai_rl_1", "ai_rl_2"]
        self.agents = self.possible_agents[:]

        default_weights = {agent: (1.0, 0.0) for agent in self.agents}
        self.reward_weights = reward_weights if reward_weights is not None else default_weights
        self.wait_for_completion = wait_for_completion
        self.cumulated_pure_rewards = {agent: 0.0 for agent in self.agents}
        self.cumulated_modified_rewards = {agent: 0.0 for agent in self.agents}

        if self.game_mode == "competition":
            self.total_agent_events = {agent_id: {"deliver_own": 0, "deliver_other": 0, "salad_own": 0, "salad_other": 0, "cut_own": 0, "cut_other": 0, "plate": 0, "raw_food_own": 0, "raw_food_other": 0, "counter": 0} for agent_id in self.agents}
            self.action_obs_mapping = ACTIONS_OBSERVATION_MAPPING_COMPETITION
            self.agent_food_type = {
                "ai_rl_1": "tomato",
                "ai_rl_2": "pumpkin",
                }
        elif self.game_mode == "classic":
            self.total_agent_events = {agent_id: {"deliver": 0, "cut": 0, "salad": 0, "plate": 0, "raw_food": 0, "counter": 0} for agent_id in self.agents}
            self.action_obs_mapping = ACTIONS_OBSERVATION_MAPPING_CLASSIC
        else:
            raise ValueError(f"Unknown game mode: {self.game_mode}")
        
        self.total_action_types = {
            agent_id: {action_type: 0 for action_type in get_action_type_list(self.game_mode)}
            for agent_id in self.agents
        }
        self.total_actions_asked = {agent_id: 0 for agent_id in self.agents}
        self.total_actions_not_performed = {agent_id: 0 for agent_id in self.agents}
        self.total_actions_inaccessible = {agent_id: 0 for agent_id in self.agents}
        self.ACTION_TYPE_INACCESSIBLE = ACTION_TYPE_INACCESSIBLE

        self.game, self.action_spaces, self._clickable_mask, self.clickable_indices = init_game(self.agents, map_nr=self.map_nr, grid_size=self.grid_size, seed=self.seed, game_mode=self.game_mode, walking_speeds=self.walking_speeds, cutting_speeds=self.cutting_speeds)

        self.agent_map = {
            agent_id: self.game.gameObjects[agent_id] for agent_id in self.agents
        }

        # Initialize action completion states for all agents
        for agent_id, agent in self.agent_map.items():
            agent.is_busy = False

        # --- New observation space---
        obs_vector = game_to_obs_vector(self.game, self.agents[0], game_mode=self.game_mode, distance_map=self.distance_map)
        obs_size = obs_vector.size
        self.observation_spaces = {
            agent: spaces.Box(low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32)
            for agent in self.agents
        }

        self.observations = {agent: np.zeros((obs_size,), dtype=np.float32) for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._last_score = 0

        self.obs_actions_mapping = {}

    def reset(self, seed=None, options=None):
        # Initialize or increment reset counter
        if not hasattr(self, '_reset_count'):
            self._reset_count = 0
        self._reset_count += 1

        # Create a unique seed by combining fixed seed and reset counter
        episode_seed = (self.seed + self._reset_count) if self.seed is not None else None

        self.agents = self.possible_agents[:]

        self.game, self.action_spaces, self._clickable_mask, self.clickable_indices = init_game(self.agents, map_nr=self.map_nr, grid_size=self.grid_size, seed=episode_seed, game_mode=self.game_mode, walking_speeds=self.walking_speeds, cutting_speeds=self.cutting_speeds)
        self.game.clickable_indices = self.clickable_indices
        random_game_state(self.game)

        self.agent_map = {agent_id: self.game.gameObjects[agent_id] for agent_id in self.agents}

        self.cumulated_pure_rewards = {agent: 0.0 for agent in self.agents}
        self.cumulated_modified_rewards = {agent: 0.0 for agent in self.agents}

        self._elapsed_time = 0.0

        # Store last (unnormalized) observation for each agent
        self._last_obs_raw = {}
        for agent in self.agents:
            obs_raw = game_to_obs_vector(self.game, agent, game_mode=self.game_mode, distance_map=self.distance_map)
            self._last_obs_raw[agent] = obs_raw

        if self.game_mode == "competition":
            self.total_agent_events = {agent_id: {"deliver_own": 0, "deliver_other": 0, "salad_own": 0, "salad_other": 0, "cut_own": 0, "cut_other": 0} for agent_id in self.agents}
            self.agent_food_type = {
                "ai_rl_1": "tomato",
                "ai_rl_2": "pumpkin",
                }
        elif self.game_mode == "classic":
            self.total_agent_events = {agent_id: {"deliver": 0, "cut": 0, "salad": 0, "plate": 0, "raw_food": 0, "counter": 0} for agent_id in self.agents}
        else:
            raise ValueError(f"Unknown game mode: {self.game_mode}")

        self.total_action_types = {
            agent_id: {action_type: 0 for action_type in get_action_type_list(self.game_mode)}
            for agent_id in self.agents
        }
        self.total_actions_asked = {agent_id: 0 for agent_id in self.agents}
        self.total_actions_not_performed = {agent_id: 0 for agent_id in self.agents}
        self.total_actions_inaccessible = {agent_id: 0 for agent_id in self.agents}

        self.observations = {agent: self.observe(agent) for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._last_score = 0

        return self.observations, self.infos

    def observe(self, agent):
        obs_vector = game_to_obs_vector(self.game, agent, game_mode=self.game_mode, distance_map=self.distance_map)
        obs = obs_vector.flatten().astype(np.float32)
        return obs

    def step(self, actions):
        # --- Time-based step logic ---
        # 1. For each agent, if not busy and action is provided, process action and set busy time
        # 2. Advance simulation by the minimum busy time (or until episode ends)
        # 3. Assign negative reward proportional to busy time at action selection
        # 4. End episode if elapsed time >= max seconds

        validated_actions = {}

        # Tracking for rewards and events
        agent_penalties = {agent_id: 0.0 for agent_id in self.agents}
        if self.game_mode == "competition":
            agent_events = {agent_id: {"deliver_own": 0, "deliver_other": 0, "salad_own": 0, "salad_other": 0, "cut_own": 0, "cut_other": 0, "plate": 0, "raw_food_own": 0, "raw_food_other": 0, "counter": 0} for agent_id in self.agents}
        elif self.game_mode == "classic":
            agent_events = {agent_id: {"deliver": 0, "cut": 0, "salad": 0, "plate": 0, "raw_food": 0, "counter": 0} for agent_id in self.agents}
        else:
            raise ValueError(f"Unknown game mode: {self.game_mode}")


        # Process new actions for available agents
        for agent_id, action_idx in actions.items():
            agent = self.agent_map[agent_id]
            self.total_actions_asked[agent_id] += 1
            print(f'Item on hand for agent {agent_id}: {getattr(agent, "item", None)}')
            if getattr(agent, "busy_until", None) is None or self._elapsed_time >= agent.busy_until:
                # Convert RL action to tile index
                action_name = get_rl_action_space(self.game_mode)[action_idx]
                tile_index = convert_action_to_tile(agent, self.game, action_name, distance_map=self.distance_map)
                if tile_index is None:
                    self.total_actions_not_performed[agent_id] += 1
                    agent_penalties[agent_id] += INACCESSIBLE_TILE_PENALTY
                    continue
                grid_w = self.game.grid.width
                x = tile_index % grid_w
                y = tile_index // grid_w
                tile = self.game.grid.tiles[x][y]
                action_type, agent_events = get_action_type_and_agent_events(self, tile, agent, agent_id, agent_events, x=x, y=y, accessibility_map=self.accessibility_map)
                busy_time = 0.2  # Default minimal busy time for game step

                # Penalty for inaccessible action
                if action_type == self.ACTION_TYPE_INACCESSIBLE:
                    self.total_actions_inaccessible[agent_id] += 1
                    agent_penalties[agent_id] += INACCESSIBLE_TILE_PENALTY
                    agent.busy_until = self._elapsed_time + busy_time
                    continue

                # Penalty for useless actions
                elif action_type.startswith("useless_"):
                    agent_penalties[agent_id] += USELESS_ACTION_PENALTY

                # Penalty for destructive actions
                elif action_type.startswith("destructive_"):
                    agent_penalties[agent_id] += DESTRUCTIVE_ACTION_PENALTY
                
                # Estimate busy time (move + action duration)  
                obs = self.observations.get(agent_id)
                busy_time = obs[self.action_obs_mapping[action_idx]] * getattr(self.game, 'normalization_factor', 1.0)
                agent.busy_until = self._elapsed_time + busy_time

                # Assign negative reward proportional to busy time
                agent_penalties[agent_id] += BUSY_PENALTY * busy_time

                # Store the validated action
                validated_actions[agent_id] = {"type": "click", "target": tile_index}
                self.total_action_types[agent_id][action_type] += 1
            else:
                self.total_actions_not_performed[agent_id] += 1

        # Find minimum time until next agent is available or episode ends
        active_busy_times = [
            agent.busy_until for agent in self.agent_map.values()
            if getattr(agent, "busy_until", None) is not None and agent.busy_until > self._elapsed_time
        ]
        # If no agent is busy, advance by a minimal time step of 0.2 seconds
        if active_busy_times:
            next_time = min(min(active_busy_times), self._max_seconds_per_episode)
        else:
            next_time = min(self._elapsed_time + 0.2, self._max_seconds_per_episode)
        
        # Advance time and game state
        advanced_time = next_time - self._elapsed_time
        self._elapsed_time = next_time
        self.game.step(validated_actions, delta_time=advanced_time)

        # Update busy_times and mark agents as available if their busy_until has passed
        for agent_id, agent in self.agent_map.items():
            if hasattr(agent, 'busy_until') and agent.busy_until is not None and self._elapsed_time >= agent.busy_until:
                agent.busy_until = None

        # Compute rewards
        self.cumulated_pure_rewards, self.cumulated_modified_rewards = get_rewards(self, agent_events, agent_penalties, REWARDS)

        # Check for episode termination
        should_truncate = self._elapsed_time >= self._max_seconds_per_episode
        if should_truncate:
            self.dones = {agent: True for agent in self.agents}
            self._elapsed_time = 0
            self.write_csv = True

        self.infos = {
            agent: {
                "agent_events": agent_events[agent],
                "action_types": self.total_action_types[agent],
                "score": self.agent_map[agent].score
            }
            for agent in self.agents
        }

        # Update last observation for each agent after stepping
        self.observations = {agent: self.observe(agent) for agent in self.agents}
        terminations = self.dones
        truncations = {agent: False for agent in self.agents}

        # If episode is done, aggregate and log
        if self.write_csv:
            print(f"[Episode {self.episode_count}] Logging episode data to csv")
            row = {"epoch": self.episode_count}
            for agent_id in self.agents:
                row[f"pure_reward_{agent_id}"] = float(self.cumulated_pure_rewards[agent_id])
                row[f"modified_reward_{agent_id}"] = float(self.cumulated_modified_rewards[agent_id])

                # Add result events
                for result_event in self.total_agent_events[agent_id]:
                    row[f"{result_event}_{agent_id}"] = self.total_agent_events[agent_id][result_event]
                row[f"actions_asked_{agent_id}"] = self.total_actions_asked[agent_id]
                row[f"actions_not_performed_{agent_id}"] = self.total_actions_not_performed[agent_id]
                row[f"inaccessible_actions_{agent_id}"] = self.total_actions_inaccessible[agent_id]

                # Add action type columns dynamically
                for action_type in get_action_type_list(self.game_mode):
                    row[f"{action_type}_{agent_id}"] = self.total_action_types[agent_id][action_type]

            with open(self.csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if self.write_header:
                    writer.writeheader()
                    self.write_header = False
                writer.writerow(row)

            # Reset log for next episode
            self.episode_infos_log = {agent: [] for agent in self.agents}
            self.episode_count += 1
            self.write_csv = False

        return self.observations, self.rewards, terminations, truncations, self.infos

    def render(self):
        print(f"[Game Render] Agents: {self.agents}")

    def close(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        return self.action_spaces[agent]
