import csv
import os
from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np
from spoiled_broth.config import *

from spoiled_broth.game import SpoiledBroth, random_game_state, game_to_obs_matrix
from spoiled_broth.world.tiles import Counter

MAX_PLAYERS = 4

COUNTER_PENALTY = 0.1

REWARD_ITEM_CUT = 1.
REWARD_SALAD_CREATED = 2. + REWARD_ITEM_CUT  # (+ REWARD_ITEM_CUT sinc creating a salad "loses a cut item")
REWARD_DELIVERED = 10. + REWARD_SALAD_CREATED  # (+ REWARD_SALAD_CREATED since delivering "loses a salad")

# Action type constants for tracking
ACTION_TYPE_FLOOR = "floor"
ACTION_TYPE_WALL = "wall"
ACTION_TYPE_COUNTER = "counter"
ACTION_TYPE_DISPENSER = "dispenser"
ACTION_TYPE_CUTTING_BOARD = "cutting_board"
ACTION_TYPE_DELIVERY = "delivery"
ACTION_TYPE_NON_CLICKABLE = "non_clickable"

def get_action_type(tile):
    """Determine the type of action based on the tile clicked"""
    if tile is None or not hasattr(tile, '_type'):
        return ACTION_TYPE_FLOOR  # Default to floor for None/invalid tiles
    
    # Map tile types to action types
    type_mapping = {
        0: ACTION_TYPE_FLOOR,
        1: ACTION_TYPE_WALL,
        2: ACTION_TYPE_COUNTER,
        3: ACTION_TYPE_DISPENSER,
        4: ACTION_TYPE_CUTTING_BOARD,
        5: ACTION_TYPE_DELIVERY
    }
    
    return type_mapping.get(tile._type, ACTION_TYPE_FLOOR)  # Default to floor for unknown types

def init_game(agents, map_nr=1, grid_size=(8, 8)):
    game = SpoiledBroth(map_nr=map_nr, grid_size=grid_size)
    for agent_id in agents:
        game.add_agent(agent_id)

    clickable_indices = []
    for x in range(game.grid.width):
        for y in range(game.grid.height):
            tile = game.grid.tiles[x][y]
            if tile and tile.clickable is not None:
                index = y * game.grid.width + x
                clickable_indices.append(index)

    action_spaces = {
        agent: spaces.Discrete(len(clickable_indices))
        for agent in agents
    }
    _clickable_mask = np.zeros(game.grid.width * game.grid.height, dtype=np.int8)
    for idx in clickable_indices:
        _clickable_mask[idx] = 1

    return game, action_spaces, _clickable_mask, clickable_indices


class GameEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "game_v0"}

    def __init__(
            self, 
            reward_weights=None, 
            map_nr=1, 
            cooperative=1, 
            step_per_episode=1000,
            path="training_stats.csv",
            grid_size=(8, 8)
    ):
        super().__init__()
        self.map_nr = map_nr
        self.cooperative = cooperative  # Store cooperative mode as instance variable
        self._step_counter = 0
        self.episode_count = 0
        self._max_steps_per_episode = step_per_episode
        self.render_mode = None
        self.write_header = True
        self.write_csv = False
        self.csv_path = os.path.join(path, "training_stats.csv")
        self.grid_size = grid_size

        self.possible_agents = ["ai_rl_1", "ai_rl_2"]
        self.agents = self.possible_agents[:]

        default_weights = {agent: (1.0, 0.0) for agent in self.agents}
        self.reward_weights = reward_weights if reward_weights is not None else default_weights
        self.cumulated_pure_rewards = {agent: 0.0 for agent in self.agents}
        self.cumulated_modified_rewards = {agent: 0.0 for agent in self.agents}
        self.total_agent_events = {agent_id: {"delivered": 0, "cut": 0, "salad": 0} for agent_id in self.agents}
        self.total_action_types = {
            agent_id: {
                ACTION_TYPE_FLOOR: 0,
                ACTION_TYPE_WALL: 0,
                ACTION_TYPE_COUNTER: 0,
                ACTION_TYPE_DISPENSER: 0,
                ACTION_TYPE_CUTTING_BOARD: 0,
                ACTION_TYPE_DELIVERY: 0
            } for agent_id in self.agents
        }

        self.game, self.action_spaces, self._clickable_mask, self.clickable_indices = init_game(self.agents, map_nr=self.map_nr, grid_size=self.grid_size)

        self.agent_map = {
            agent_id: self.game.gameObjects[agent_id] for agent_id in self.agents
        }

        # --- New observation space: flatten (channels, H, W) + (2, 4) inventory ---
        obs_matrix, agent_inventory = game_to_obs_matrix(self.game, self.agents[0])
        obs_size = obs_matrix.size + agent_inventory.size
        self.observation_spaces = {
            agent: spaces.Box(low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32)
            for agent in self.agents
        }

        self.rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._last_score = 0

    def reset(self, seed=None, options=None):
        self._last_score = 0
        self.agents = self.possible_agents[:]

        self.game, self.action_spaces, self._clickable_mask, self.clickable_indices = init_game(self.agents, map_nr=self.map_nr, grid_size=self.grid_size)

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
        self.total_action_types = {
            agent_id: {
                ACTION_TYPE_FLOOR: 0,
                ACTION_TYPE_WALL: 0,
                ACTION_TYPE_COUNTER: 0,
                ACTION_TYPE_DISPENSER: 0,
                ACTION_TYPE_CUTTING_BOARD: 0,
                ACTION_TYPE_DELIVERY: 0
            } for agent_id in self.agents
        }

        assert isinstance(self.infos, dict), f"infos is not a dict: {self.infos}"
        assert all(isinstance(v, dict) for v in self.infos.values()), "infos values must be dicts"

        return {
            agent: self.observe(agent)
            for agent in self.agents
        }, self.infos

    def observe(self, agent):
        # Use the new spatial observation (channels, H, W) and agent inventory
        obs_matrix, agent_inventory = game_to_obs_matrix(self.game, agent)
        # Option 1: Return as tuple (recommended for custom RL code)
        # return (obs_matrix, agent_inventory)
        # Option 2: Flatten and concatenate for Gym compatibility
        obs = np.concatenate([obs_matrix.flatten(), agent_inventory.flatten()]).astype(np.float32)
        return obs

    def step(self, actions):
        self._step_counter += 1
        # Submit intents from each agent

        agent_penalties = {agent_id: 0.0 for agent_id in self.agents}  # Used for counter penalty
        for agent_id, action in actions.items():
            agent_obj = self.game.gameObjects[agent_id]
            # Only allow a new action if the agent is not moving and has no unfinished intents
            if getattr(agent_obj, "is_moving", False) or getattr(agent_obj, "intents", []):
                continue  # Skip this agent's new action, let it finish its current one

            # Map action to actual clickable tile index
            tile_index = self.clickable_indices[action]
            grid_w = self.game.grid.width
            x = tile_index % grid_w
            y = tile_index // grid_w
            tile = self.game.grid.tiles[x][y]

            # Track action type
            action_type = get_action_type(tile)
            self.total_action_types[agent_id][action_type] += 1

            # Penalty for clicking on counter without holding something
            if isinstance(tile, Counter) and agent_obj.item is None:
                agent_penalties[agent_id] += COUNTER_PENALTY

            tile.click(agent_id)

        def decode_action(action_int):
            return {"type": "click", "target": int(self.clickable_indices[action_int])}

        actions_dict = {agent_id: decode_action(action) for agent_id, action in actions.items()}

        # Advance the game state by one step (simulate one tick)
        self.game.step(actions_dict, delta_time=1 / cf_AI_TICK_RATE)

        # Detect agent that performed the action and reward
        agent_events = {agent_id: {"delivered": 0, "cut": 0, "salad": 0} for agent_id in self.agents}
        
        for thing in self.game.gameObjects.values():
            if hasattr(thing, 'tiles'):
                for row in thing.tiles:
                    for tile in row:
                        if hasattr(tile, "cut_by") and tile.cut_by:
                            if tile.cut_by in agent_events:
                                agent_events[tile.cut_by]["cut"] += 1
                                self.total_agent_events[tile.cut_by]["cut"] += 1
                            tile.cut_by = None

                        if hasattr(tile, "salad_by") and tile.salad_by:
                            if tile.salad_by in agent_events:
                                agent_events[tile.salad_by]["salad"] += 1
                                self.total_agent_events[tile.salad_by]["salad"] += 1
                            tile.salad_by = None

        # Delivery
        new_score = self.game.gameObjects["score"].score
        if (new_score - self._last_score) > 0:
            print(f"Agent delivered item, new score: {new_score}")
            if self.cooperative:
                # Reward all agents for delivery
                for agent_id in agent_events:
                    agent_events[agent_id]["delivered"] += 1
                    self.total_agent_events[agent_id]["delivered"] += 1
            else:
                # Only reward the delivering agent(s)
                for thing in self.game.gameObjects.values():
                    if hasattr(thing, 'tiles'):
                        for row in thing.tiles:
                            for tile in row:
                                if hasattr(tile, "delivered_by") and tile.delivered_by:
                                    if tile.delivered_by in agent_events:
                                        agent_events[tile.delivered_by]["delivered"] += 1
                                        self.total_agent_events[tile.delivered_by]["delivered"] += 1
                                    tile.delivered_by = None
            self._last_score = new_score

        # Get reward from delivering items
        pure_rewards = {agent_id: 0.0 for agent_id in self.agents}
        for agent_id in self.agents:
            # Pure rewards: only positive events, no penalties
            pure_rewards[agent_id] = (
                agent_events[agent_id]["delivered"] * REWARD_DELIVERED +
                agent_events[agent_id]["cut"] * REWARD_ITEM_CUT +
                agent_events[agent_id]["salad"] * REWARD_SALAD_CREATED
            )
            self.cumulated_pure_rewards[agent_id] += pure_rewards[agent_id]

        for agent_id in self.agents:
            alpha, beta = self.reward_weights.get(agent_id, (1.0, 0.0))
            other_agents = [a for a in self.agents if a != agent_id]
            if other_agents:
                avg_other_reward = sum(pure_rewards[a] for a in other_agents) / len(other_agents)
            else:
                avg_other_reward = 0.0  # in case there is only one agent
            # Modified rewards: include penalties
            self.rewards[agent_id] = alpha * (pure_rewards[agent_id] - agent_penalties[agent_id]) + beta * avg_other_reward
            self.cumulated_modified_rewards[agent_id] += self.rewards[agent_id]

        # Check if episode should end due to step limit
        should_truncate = self._step_counter >= self._max_steps_per_episode
        if should_truncate:
            self.dones = {agent: True for agent in self.agents}
            self._step_counter = 0
            self.write_csv = True
        
        self.infos = {
            agent: {
                "pure_reward": pure_rewards[agent],
                "agent_events": agent_events[agent],  # Add agent events to info
                "action_types": self.total_action_types[agent]  # Add action types to info
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
            row = {
                    "epoch": self.episode_count
                }
            
            for agent_id in self.agents:
                row[f"pure_reward_{agent_id}"] = float(self.cumulated_pure_rewards[agent_id])
                row[f"modified_reward_{agent_id}"] = float(self.cumulated_modified_rewards[agent_id])
                row[f"delivered_{agent_id}"] = self.total_agent_events[agent_id]["delivered"]
                row[f"cut_{agent_id}"] = self.total_agent_events[agent_id]["cut"]
                row[f"salad_{agent_id}"] = self.total_agent_events[agent_id]["salad"]

                # Add action type columns
                row[f"floor_actions_{agent_id}"] = self.total_action_types[agent_id][ACTION_TYPE_FLOOR]
                row[f"wall_actions_{agent_id}"] = self.total_action_types[agent_id][ACTION_TYPE_WALL]
                row[f"counter_actions_{agent_id}"] = self.total_action_types[agent_id][ACTION_TYPE_COUNTER]
                row[f"dispenser_actions_{agent_id}"] = self.total_action_types[agent_id][ACTION_TYPE_DISPENSER]
                row[f"cutting_board_actions_{agent_id}"] = self.total_action_types[agent_id][ACTION_TYPE_CUTTING_BOARD]
                row[f"delivery_actions_{agent_id}"] = self.total_action_types[agent_id][ACTION_TYPE_DELIVERY]

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
