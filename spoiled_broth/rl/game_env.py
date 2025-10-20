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
from spoiled_broth.rl.classify_action_type import get_action_type, get_action_type_list, ACTION_TYPE_INACCESSIBLE
from spoiled_broth.rl.agent_events import get_agent_events
from spoiled_broth.rl.reward_analysis import get_rewards
from spoiled_broth.game import SpoiledBroth, random_game_state

USELESS_ACTION_PENALTY = 0.1 # Penalty for performing a useless action
INACCESSIBLE_TILE_PENALTY = 0.5  # Harsh penalty for trying to access unreachable tiles
WAIT_FOR_ACTION_COMPLETION = True  # Flag to ensure actions complete before next step

REWARDS = {
    "item_cut": 2.0,
    "salad_created": 5.0,
    "delivered": 10.0
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

def init_game(agents, map_nr=1, grid_size=(8, 8), seed=None, game_mode="classic"):
    num_agents = len(agents)
    game = SpoiledBroth(map_nr=map_nr, grid_size=grid_size, num_agents=num_agents, seed=seed)
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
        game.add_agent(agent_id)
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
        step_per_episode=1000,
        path="training_stats.csv",
        grid_size=(8, 8),
        payoff_matrix=[1,1,-2],
        initial_seed=0,
        wait_for_completion=True,  # New parameter to control action completion waiting
        start_epoch=0,
        distance_map=None
    ):
        super().__init__()
        self.map_nr = map_nr
        self._step_counter = 0
        self.episode_count = start_epoch
        self._max_steps_per_episode = step_per_episode
        self.render_mode = None
        self.write_header = True
        self.write_csv = False
        self.csv_path = os.path.join(path, "training_stats.csv")
        self.grid_size = grid_size
        self.seed = initial_seed
        self.clickable_indices = None  # Initialize clickable indices storage
        # distance_map can be either a dict (already loaded) or a filepath to a pickle file.
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
        self.game_mode = game_mode            

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
            self.total_agent_events = {agent_id: {"delivered_own": 0, "delivered_other": 0, "salad_own": 0, "salad_other": 0, "cut_own": 0, "cut_other": 0} for agent_id in self.agents}
            self.agent_food_type = {
                "ai_rl_1": "tomato",
                "ai_rl_2": "pumpkin",
                }
        elif self.game_mode == "classic":
            self.total_agent_events = {agent_id: {"delivered": 0, "salad": 0, "cut": 0} for agent_id in self.agents}
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

        self.game, self.action_spaces, self._clickable_mask, self.clickable_indices = init_game(self.agents, map_nr=self.map_nr, grid_size=self.grid_size, seed=self.seed, game_mode=self.game_mode)

        self.agent_map = {
            agent_id: self.game.gameObjects[agent_id] for agent_id in self.agents
        }

        # Initialize action completion states for all agents
        for agent_id, agent in self.agent_map.items():
            agent.action_complete = True
            agent.current_action = None
            agent.is_busy = False

        # --- New observation space---
        obs_vector = game_to_obs_vector(self.game, self.agents[0], game_mode=self.game_mode, map_nr=self.map_nr, distance_map=self.distance_map)
        obs_size = obs_vector.size
        self.observation_spaces = {
            agent: spaces.Box(low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32)
            for agent in self.agents
        }

        self.rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._last_score = 0

    def reset(self, seed=None, options=None):
        # Initialize or increment reset counter
        if not hasattr(self, '_reset_count'):
            self._reset_count = 0
        self._reset_count += 1
        
        # Create a unique seed by combining fixed seed and reset counter
        episode_seed = (self.seed + self._reset_count) if self.seed is not None else None
        
        self._last_score = 0
        self.agents = self.possible_agents[:]

        self.game, self.action_spaces, self._clickable_mask, self.clickable_indices = init_game(self.agents, map_nr=self.map_nr, grid_size=self.grid_size, seed=episode_seed, game_mode=self.game_mode)
        self.game.clickable_indices = self.clickable_indices
        random_game_state(self.game)

        self.agent_map = {
            agent_id: self.game.gameObjects[agent_id] for agent_id in self.agents
        }

        self.rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._last_score = 0


        self.cumulated_pure_rewards = {agent: 0.0 for agent in self.agents}
        self.cumulated_modified_rewards = {agent: 0.0 for agent in self.agents}
        self.total_agent_events = {agent_id: {"delivered": 0, "cut": 0, "salad": 0} for agent_id in self.agents}

        if self.game_mode == "competition":
            self.total_agent_events = {agent_id: {"delivered_own": 0, "delivered_other": 0, "salad_own": 0, "salad_other": 0, "cut_own": 0, "cut_other": 0} for agent_id in self.agents}
            self.agent_food_type = {
                "ai_rl_1": "tomato",
                "ai_rl_2": "pumpkin",
                }
        elif self.game_mode == "classic":
            self.total_agent_events = {agent_id: {"delivered": 0, "salad": 0, "cut": 0} for agent_id in self.agents}
        else:
            raise ValueError(f"Unknown game mode: {self.game_mode}")

        self.total_action_types = {
            agent_id: {action_type: 0 for action_type in get_action_type_list(self.game_mode)}
            for agent_id in self.agents
        }
        self.total_actions_asked = {agent_id: 0 for agent_id in self.agents}
        self.total_actions_not_performed = {agent_id: 0 for agent_id in self.agents}
        self.total_actions_inaccessible = {agent_id: 0 for agent_id in self.agents}

        assert isinstance(self.infos, dict), f"infos is not a dict: {self.infos}"
        assert all(isinstance(v, dict) for v in self.infos.values()), "infos values must be dicts"

        return {
            agent: self.observe(agent)
            for agent in self.agents
        }, self.infos

    def observe(self, agent):
        obs_vector = game_to_obs_vector(self.game, agent, game_mode=self.game_mode, map_nr=self.map_nr, distance_map=self.distance_map)
        obs = obs_vector.flatten().astype(np.float32)
        return obs

    def step(self, actions):
        self._step_counter += 1
        agent_penalties = {agent_id: 0.0 for agent_id in self.agents}
        validated_actions = {}

        # Convert RL actions to tile clicks
        for agent_id, action_idx in actions.items():
            self.total_actions_asked[agent_id] += 1
            agent_obj = self.game.gameObjects[agent_id]
            action_name = get_rl_action_space(self.game_mode)[action_idx]

            # Check if agent is busy (e.g., cutting)
            if getattr(agent_obj, "is_busy", False):
                self.total_actions_not_performed[agent_id] += 1
                continue

            # Convert RL action to tile index
            tile_index = convert_action_to_tile(agent_obj, self.game, action_name, distance_map=self.distance_map)

            if tile_index is None:
                self.total_actions_not_performed[agent_id] += 1
                continue
            
            else:
                grid_w = self.game.grid.width
                x = tile_index % grid_w
                y = tile_index // grid_w
                tile = self.game.grid.tiles[x][y]
                action_type = get_action_type(tile, agent_obj, x=x, y=y, accessibility_map=self.accessibility_map)
    
                # Penalty for inaccessible action
                if action_type == self.ACTION_TYPE_INACCESSIBLE:
                    self.total_actions_inaccessible[agent_id] += 1
                    agent_penalties[agent_id] += INACCESSIBLE_TILE_PENALTY
                    continue
                
                # Count action type
                self.total_action_types[agent_id][action_type] += 1
    
                # Penalty for useless action
                if action_type.startswith("useless"):
                    agent_penalties[agent_id] += USELESS_ACTION_PENALTY
                agent_obj.action_complete = False
    
                # Store the validated action
                validated_actions[agent_id] = {"type": "click", "target": tile_index}

        # Advance the game state by one step with validated actions
        self.game.step(validated_actions, delta_time=1 / cf_AI_TICK_RATE)

        # Tracking for rewards and events
        agent_events = {agent_id: {"delivered": 0, "cut": 0, "salad": 0} for agent_id in self.agents}

        self.total_agent_events, agent_events, self._last_score = get_agent_events(self, agent_events)
        self.cumulated_pure_rewards, self.cumulated_modified_rewards = get_rewards(self, agent_events, agent_penalties, REWARDS) 

        # Check if episode should end due to step limit
        should_truncate = self._step_counter >= self._max_steps_per_episode
        if should_truncate:
            self.dones = {agent: True for agent in self.agents}
            self._step_counter = 0
            self.write_csv = True

        self.infos = {
            agent: {
                "agent_events": agent_events[agent],  # Add agent events to info
                "action_types": self.total_action_types[agent],  # Add action types to info
                "score": self.agent_map[agent].score
            }
            for agent in self.agents
        }

        observations = {
            agent: self.observe(agent)
            for agent in self.agents
        }

        terminations = self.dones
        truncations = {agent: False for agent in self.agents}  # No truncations, only terminations

        # If episode is done, aggregate and log
        if self.write_csv:
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

        return observations, self.rewards, terminations, truncations, self.infos

    def render(self):
        print(f"[Game Render] Agents: {self.agents}")

    def close(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        return self.action_spaces[agent]
