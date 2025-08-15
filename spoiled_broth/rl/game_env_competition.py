import csv
import os
from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np
from spoiled_broth.config import *

from spoiled_broth.game import SpoiledBroth, random_game_state, game_to_obs_matrix_competition
#from spoiled_broth.world.tiles import Counter

DO_NOTHING_PENALTY = 0.3  # Penalty for choosing to do nothing
USELESS_ACTION_PENALTY = 0.2 # Penalty for performing a useless action
NO_ACTION_PENALTY = 0.01

REWARDS_BY_VERSION = {
    "v1": {
        "useful_food_dispenser": 0.5,
        "useful_cutting_board": 0.0,
        "useful_plate_dispenser": 0.0,
        "item_cut": 0.0,
        "salad_created": 0.0,
        "delivered": 10.0
    },
    "v2.1": {
        "useful_food_dispenser": 0.5,
        "useful_cutting_board": 1.0,
        "useful_plate_dispenser": 0.0,
        "item_cut": 3.0,
        "salad_created": 0.0,
        "delivered": 10.0
    },
    "v2.2": {
        "useful_food_dispenser": 0.5,
        "useful_cutting_board": 1.0,
        "useful_plate_dispenser": 0.0,
        "item_cut": 3.0,
        "salad_created": 0.0,
        "delivered": 10.0
    },
    "v3.1": {
        "useful_food_dispenser": 0.5,
        "useful_cutting_board": 1.0,
        "useful_plate_dispenser": 2.0,
        "item_cut": 3.0,
        "salad_created": 5.0,
        "delivered": 10.0
    },
    "v3.2": {
        "useful_food_dispenser": 0.5,
        "useful_cutting_board": 1.0,
        "useful_plate_dispenser": 2.0,
        "item_cut": 3.0,
        "salad_created": 5.0,
        "delivered": 10.0
    },
    "default": {
        "useful_dispenser": 0.5,
        "useful_cutting_board": 1.0,
        "useful_plate_dispenser": 1.5,
        "item_cut": 3.0,
        "salad_created": 5.0,
        "delivered": 10.0
    }
}

# Action type constants for tracking
ACTION_TYPE_DO_NOTHING = "do_nothing"
ACTION_TYPE_FLOOR = "floor"
ACTION_TYPE_WALL = "wall"
ACTION_TYPE_USELESS_COUNTER = "useless_counter"
ACTION_TYPE_USEFUL_COUNTER = "useful_counter"
ACTION_TYPE_OWN_USEFUL_FOOD_DISPENSER = "useful_food_dispenser_own"
ACTION_TYPE_OWN_USELESS_FOOD_DISPENSER = "useless_food_dispenser_own"
ACTION_TYPE_OTHER_USEFUL_FOOD_DISPENSER = "useful_food_dispenser_other"
ACTION_TYPE_OTHER_USELESS_FOOD_DISPENSER = "useless_food_dispenser_other"
ACTION_TYPE_USELESS_CUTTING_BOARD = "useless_cutting_board"
ACTION_TYPE_OWN_USEFUL_CUTTING_BOARD = "useful_cutting_board_own"
ACTION_TYPE_OTHER_USEFUL_CUTTING_BOARD = "useful_cutting_board_other"
ACTION_TYPE_USEFUL_PLATE_DISPENSER = "useful_plate_dispenser"
ACTION_TYPE_USELESS_PLATE_DISPENSER = "useless_plate_dispenser"
ACTION_TYPE_USELESS_DELIVERY = "useless_delivery"
ACTION_TYPE_OWN_USEFUL_DELIVERY = "useful_delivery_own"
ACTION_TYPE_OTHER_USEFUL_DELIVERY = "useful_delivery_other"

def get_action_type(tile, agent, agent_id, agent_food_type):
    """
    Determine the type of action based on the tile clicked and agent state.
    Uses agent_food_type to distinguish own food from other food.
    """
    if tile is None or not hasattr(tile, '_type'):
        return ACTION_TYPE_FLOOR  # Default to floor for None/invalid tiles

    default_action = ACTION_TYPE_FLOOR
    holding_item = getattr(agent, "item", None)
    holding_something = holding_item is not None

    own_food = agent_food_type.get(agent_id, None)
    other_food = None

    for aid, food in agent_food_type.items():
        if aid != agent_id:
            other_food = food

    # Tile type 0: floor
    if tile._type == 0:
        return ACTION_TYPE_FLOOR

    # Tile type 1: wall
    if tile._type == 1:
        return ACTION_TYPE_WALL

    # Tile type 2: counter
    if tile._type == 2:
        # Useful if agent is holding something (can drop it), or counter has something and agent can pick it
        item_on_counter = getattr(tile, "item", None)
        if holding_something or item_on_counter is not None:
            return ACTION_TYPE_USEFUL_COUNTER
        else:
            return ACTION_TYPE_USELESS_COUNTER

    # Tile type 3: dispenser
    if tile._type == 3:
        disp_item = getattr(tile, "item", None)
        if disp_item == "plate":
            return ACTION_TYPE_USEFUL_PLATE_DISPENSER if not holding_something else ACTION_TYPE_USELESS_PLATE_DISPENSER
        elif disp_item == own_food:
            return ACTION_TYPE_OWN_USEFUL_FOOD_DISPENSER if not holding_something else ACTION_TYPE_OWN_USELESS_FOOD_DISPENSER
        elif disp_item == other_food:
            return ACTION_TYPE_OTHER_USEFUL_FOOD_DISPENSER if not holding_something else ACTION_TYPE_OTHER_USELESS_FOOD_DISPENSER
            
    # Tile type 4: cutting board
    if tile._type == 4:
        board_item = getattr(tile, "item", None)

        # Detect food type on board
        board_item_type = board_item if isinstance(board_item, str) else None

        # Item in hand can be cut (not already cut/salad)
        item_in_hand = holding_item
        holding_uncut = (
            item_in_hand is not None and
            isinstance(item_in_hand, str) and
            not item_in_hand.endswith("_cut") and
            not item_in_hand.endswith("_salad")
        )

        # Check if it's own or other food
        if holding_uncut:
            if item_in_hand == own_food:
                return ACTION_TYPE_OWN_USEFUL_CUTTING_BOARD
            elif item_in_hand == other_food:
                return ACTION_TYPE_OTHER_USEFUL_CUTTING_BOARD
        elif board_item_type:
            if board_item_type.startswith(own_food):
                return ACTION_TYPE_OWN_USEFUL_CUTTING_BOARD
            elif board_item_type.startswith(other_food):
                return ACTION_TYPE_OTHER_USEFUL_CUTTING_BOARD

        return ACTION_TYPE_USELESS_CUTTING_BOARD

    # Tile type 5: delivery
    if tile._type == 5:
        item = holding_item
        if item is None or not isinstance(item, str):
            return ACTION_TYPE_USELESS_DELIVERY

        # Detect if item is deliverable
        deliverable = False
        if agent.intent_version == "v1":
            deliverable = item in ['tomato_salad', 'pumpkin_salad', 'tomato', 'pumpkin']
        elif agent.intent_version in ["v2.1", "v2.2"]:
            deliverable = item in ['tomato_salad', 'pumpkin_salad', 'tomato_cut', 'pumpkin_cut']
        else:
            deliverable = item in ['tomato_salad', 'pumpkin_salad']

        if deliverable:
            if item.startswith(own_food):
                return ACTION_TYPE_OWN_USEFUL_DELIVERY
            elif item.startswith(other_food):
                return ACTION_TYPE_OTHER_USEFUL_DELIVERY
        else:
            return ACTION_TYPE_USELESS_DELIVERY

    # Default fallback
    return default_action

def init_game(agents, map_nr=1, grid_size=(8, 8), intent_version=None, seed=None):
    num_agents = len(agents)
    game = SpoiledBroth(map_nr=map_nr, grid_size=grid_size, intent_version=intent_version, num_agents=num_agents, seed=seed)
    for agent_id in agents:
        if intent_version is not None:
            game.add_agent(agent_id, intent_version=intent_version)
        else:
            game.add_agent(agent_id)

    clickable_indices = []
    for x in range(game.grid.width):
        for y in range(game.grid.height):
            tile = game.grid.tiles[x][y]
            if tile and tile.clickable is not None:
                index = y * game.grid.width + x
                clickable_indices.append(index)

    action_spaces = {
        agent: spaces.Discrete(len(clickable_indices) + 1)  # last action index = do nothing
        for agent in agents
    }
    _clickable_mask = np.zeros(game.grid.width * game.grid.height, dtype=np.int8)
    for idx in clickable_indices:
        _clickable_mask[idx] = 1

    return game, action_spaces, _clickable_mask, clickable_indices


class GameEnvCompetition(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "game_v0"}

    def __init__(
            self, 
            reward_weights=None, 
            map_nr=1, 
            cooperative=0, 
            step_per_episode=1000,
            path="training_stats.csv",
            grid_size=(8, 8),
            intent_version=None,
            payoff_matrix=[1,1,-2],
            initial_seed=0,
    ):
        super().__init__()
        self.map_nr = map_nr
        self.cooperative = 0  # Always competitive mode
        self._step_counter = 0
        self.episode_count = 0
        self._max_steps_per_episode = step_per_episode
        self.render_mode = None
        self.write_header = True
        self.write_csv = False
        self.csv_path = os.path.join(path, "training_stats.csv")
        self.grid_size = grid_size
        self.intent_version = intent_version
        self.payoff_matrix = payoff_matrix
        self.seed = initial_seed
        self.agent_food_type = {
            "ai_rl_1": "tomato",
            "ai_rl_2": "pumpkin",
        }

        # Determine agent IDs from reward_weights or default to two agents
        if reward_weights is not None:
            self.possible_agents = list(reward_weights.keys())
        else:
            self.possible_agents = ["ai_rl_1", "ai_rl_2"]
        self.agents = self.possible_agents[:]

        default_weights = {agent: (1.0, 0.0) for agent in self.agents}
        self.reward_weights = reward_weights if reward_weights is not None else default_weights
        self.cumulated_pure_rewards = {agent: 0.0 for agent in self.agents}
        self.cumulated_modified_rewards = {agent: 0.0 for agent in self.agents}
        self.total_result_events = {agent_id: {"delivered_own": 0, "delivered_other": 0, "salad_own": 0, "salad_other": 0, "cut_own": 0, "cut_other": 0} for agent_id in self.agents}
        self.total_action_types = {
            agent_id: {
                ACTION_TYPE_DO_NOTHING: 0,
                ACTION_TYPE_FLOOR: 0,
                ACTION_TYPE_WALL: 0,
                ACTION_TYPE_USELESS_COUNTER: 0,
                ACTION_TYPE_USEFUL_COUNTER: 0,
                ACTION_TYPE_OWN_USEFUL_FOOD_DISPENSER: 0,
                ACTION_TYPE_OWN_USELESS_FOOD_DISPENSER: 0,
                ACTION_TYPE_OTHER_USEFUL_FOOD_DISPENSER: 0,
                ACTION_TYPE_OTHER_USELESS_FOOD_DISPENSER: 0,
                ACTION_TYPE_USELESS_CUTTING_BOARD: 0,
                ACTION_TYPE_OWN_USEFUL_CUTTING_BOARD: 0,
                ACTION_TYPE_OTHER_USEFUL_CUTTING_BOARD: 0,
                ACTION_TYPE_USEFUL_PLATE_DISPENSER: 0,
                ACTION_TYPE_USELESS_PLATE_DISPENSER: 0,
                ACTION_TYPE_USELESS_DELIVERY: 0,
                ACTION_TYPE_OWN_USEFUL_DELIVERY: 0,
                ACTION_TYPE_OTHER_USEFUL_DELIVERY: 0
            } for agent_id in self.agents
        }
        self.total_actions_asked = {agent_id: 0 for agent_id in self.agents}
        self.total_actions_not_performed = {agent_id: 0 for agent_id in self.agents}

        self.game, self.action_spaces, self._clickable_mask, self.clickable_indices = init_game(self.agents, map_nr=self.map_nr, grid_size=self.grid_size, intent_version=self.intent_version, seed=self.seed)

        self.agent_map = {
            agent_id: self.game.gameObjects[agent_id] for agent_id in self.agents
        }

        # --- New observation space: flatten (channels, H, W) + inventory ---
        obs_matrix, agent_inventory = game_to_obs_matrix_competition(self.game, self.agents[0], self.agent_food_type)
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
        # Initialize or increment reset counter
        if not hasattr(self, '_reset_count'):
            self._reset_count = 0
        self._reset_count += 1
        
        # Create a unique seed by combining fixed seed and reset counter
        episode_seed = (self.seed + self._reset_count) if self.seed is not None else None
        
        self._last_score = 0
        self.agents = self.possible_agents[:]
        self.agent_food_type = {
            "ai_rl_1": "tomato",
            "ai_rl_2": "pumpkin",
        }        

        self.game, self.action_spaces, self._clickable_mask, self.clickable_indices = init_game(self.agents, map_nr=self.map_nr, grid_size=self.grid_size, intent_version=self.intent_version, seed=episode_seed)

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
        self.total_result_events = {agent_id: {"delivered_own": 0, "delivered_other": 0, "salad_own": 0, "salad_other": 0, "cut_own": 0, "cut_other": 0} for agent_id in self.agents}
        self.total_action_types = {
            agent_id: {
                ACTION_TYPE_DO_NOTHING: 0,
                ACTION_TYPE_FLOOR: 0,
                ACTION_TYPE_WALL: 0,
                ACTION_TYPE_USELESS_COUNTER: 0,
                ACTION_TYPE_USEFUL_COUNTER: 0,
                ACTION_TYPE_OWN_USEFUL_FOOD_DISPENSER: 0,
                ACTION_TYPE_OWN_USELESS_FOOD_DISPENSER: 0,
                ACTION_TYPE_OTHER_USEFUL_FOOD_DISPENSER: 0,
                ACTION_TYPE_OTHER_USELESS_FOOD_DISPENSER: 0,
                ACTION_TYPE_USELESS_CUTTING_BOARD: 0,
                ACTION_TYPE_OWN_USEFUL_CUTTING_BOARD: 0,
                ACTION_TYPE_OTHER_USEFUL_CUTTING_BOARD: 0,
                ACTION_TYPE_USEFUL_PLATE_DISPENSER: 0,
                ACTION_TYPE_USELESS_PLATE_DISPENSER: 0,
                ACTION_TYPE_USELESS_DELIVERY: 0,
                ACTION_TYPE_OWN_USEFUL_DELIVERY: 0,
                ACTION_TYPE_OTHER_USEFUL_DELIVERY: 0
            } for agent_id in self.agents
        }
        self.total_actions_asked = {agent_id: 0 for agent_id in self.agents}
        self.total_actions_not_performed = {agent_id: 0 for agent_id in self.agents}

        assert isinstance(self.infos, dict), f"infos is not a dict: {self.infos}"
        assert all(isinstance(v, dict) for v in self.infos.values()), "infos values must be dicts"

        return {
            agent: self.observe(agent)
            for agent in self.agents
        }, self.infos

    def get_rewards_for_version(self):
        return REWARDS_BY_VERSION.get(self.intent_version, REWARDS_BY_VERSION["default"])


    def observe(self, agent):
        # Use the new spatial observation (channels, H, W) and agent inventory
        obs_matrix, agent_inventory = game_to_obs_matrix_competition(self.game, agent, self.agent_food_type)
        # Option 1: Return as tuple (recommended for custom RL code)
        # return (obs_matrix, agent_inventory)
        # Option 2: Flatten and concatenate for Gym compatibility
        obs = np.concatenate([obs_matrix.flatten(), agent_inventory.flatten()]).astype(np.float32)
        return obs

    def step(self, actions):
        self._step_counter += 1

        agent_penalties = {agent_id: 0.0 for agent_id in self.agents}  # Used for useless and idle penalties
        step_action_types = {agent_id: {ACTION_TYPE_OWN_USEFUL_FOOD_DISPENSER: 0,
                                        ACTION_TYPE_OTHER_USEFUL_FOOD_DISPENSER: 0,
                                        ACTION_TYPE_USEFUL_PLATE_DISPENSER: 0,
                                        ACTION_TYPE_OWN_USEFUL_CUTTING_BOARD: 0,
                                        ACTION_TYPE_OTHER_USEFUL_CUTTING_BOARD: 0
                                        } for agent_id in self.agents}

        for agent_id, action in actions.items():
            self.total_actions_asked[agent_id] += 1
            agent_obj = self.game.gameObjects[agent_id]

             # Check if agent wants to do nothing
            if action == len(self.clickable_indices):
                self.total_action_types[agent_id][ACTION_TYPE_DO_NOTHING] = self.total_action_types[agent_id].get(ACTION_TYPE_DO_NOTHING, 0) + 1
                agent_penalties[agent_id] += NO_ACTION_PENALTY
                continue 

            # Only allow a new action if the agent is not moving and has no unfinished intents
            if getattr(agent_obj, "is_moving", False) or getattr(agent_obj, "intents", []):
                self.total_actions_not_performed[agent_id] += 1
                agent_penalties[agent_id] += NO_ACTION_PENALTY
                continue 

            # Map action to actual clickable tile index
            tile_index = self.clickable_indices[action]
            grid_w = self.game.grid.width
            x = tile_index % grid_w
            y = tile_index // grid_w
            tile = self.game.grid.tiles[x][y]

            # Track action type
            action_type = get_action_type(tile, agent_obj, agent_id, self.agent_food_type)
            if action_type in step_action_types[agent_id]:
                step_action_types[agent_id][action_type] += 1
            self.total_action_types[agent_id][action_type] += 1

            # Penalty for useless action
            if action_type.startswith("useless"):
                agent_penalties[agent_id] += USELESS_ACTION_PENALTY

            tile.click(agent_id)

        def decode_action(action_int):
            if action_int == len(self.clickable_indices):
                # do nothing action
                return None
            else:
                return {"type": "click", "target": int(self.clickable_indices[action_int])}

        actions_dict = {agent_id: decode_action(action) for agent_id, action in actions.items()} # Some entries might now be None (when agent chose to do nothing)
        # Filter out None (do nothing) actions
        filtered_actions_dict = {k: v for k, v in actions_dict.items() if v is not None}

        # Advance the game state by one step (simulate one tick)
        self.game.step(filtered_actions_dict, delta_time=1 / cf_AI_TICK_RATE)

        step_result_events = {agent_id: {"delivered_own": 0, "delivered_other": 0, "salad_own": 0, "salad_other": 0, "cut_own": 0, "cut_other": 0} for agent_id in self.agents}

        # Detect cut and salad events
        for thing in self.game.gameObjects.values():
            if hasattr(thing, 'tiles'):
                for row in thing.tiles:
                    for tile in row:
                        # CUT
                        if hasattr(tile, "cut_by") and tile.cut_by:
                            agent_id = tile.cut_by
                            if agent_id in step_result_events:
                                food = getattr(tile, "cut_item", "")
                                if food.startswith(self.agent_food_type[agent_id]):
                                    step_result_events[agent_id]["cut_own"] += 1
                                    self.total_result_events[agent_id]["cut_own"] += 1
                                else:
                                    step_result_events[agent_id]["cut_other"] += 1
                                    self.total_result_events[agent_id]["cut_other"] += 1
                            tile.cut_by = None
                            tile.cut_item = None

                        # SALAD
                        if hasattr(tile, "salad_by") and tile.salad_by:
                            agent_id = tile.salad_by
                            if agent_id in step_result_events:
                                food = getattr(tile, "salad_item", "")
                                if food.startswith(self.agent_food_type[agent_id]):
                                    step_result_events[agent_id]["salad_own"] += 1
                                    self.total_result_events[agent_id]["salad_own"] += 1
                                else:
                                    step_result_events[agent_id]["salad_other"] += 1
                                    self.total_result_events[agent_id]["salad_other"] += 1
                            tile.salad_by = None
                    tile.salad_item = None

        # Detect DELIVERY
        new_score = self.game.gameObjects["score"].score
        if (new_score - self._last_score) > 0:
            for thing in self.game.gameObjects.values():
                if hasattr(thing, 'tiles'):
                    for row in thing.tiles:
                        for tile in row:
                            if hasattr(tile, "delivered_by") and tile.delivered_by:
                                agent_id = tile.delivered_by
                                if agent_id in step_result_events:
                                    food = getattr(tile, "delivered_item", "")
                                    if food.startswith(self.agent_food_type[agent_id]):
                                        step_result_events[agent_id]["delivered_own"] += 1
                                        self.total_result_events[agent_id]["delivered_own"] += 1
                                    else:
                                        step_result_events[agent_id]["delivered_other"] += 1
                                        self.total_result_events[agent_id]["delivered_other"] += 1
                                tile.delivered_by = None
                                tile.delivered_item = None
        self._last_score = new_score

        # Get reward from delivering items
        rewards_cfg = self.get_rewards_for_version()
        pure_rewards = {agent_id: 0.0 for agent_id in self.agents}
        for agent_id in self.agents:
            # Pure rewards: only positive events, no IDLE or useless penalties
            own_reward = (
                step_result_events[agent_id]["delivered_own"] * rewards_cfg["delivered"]
                + step_result_events[agent_id]["salad_own"] * rewards_cfg["salad_created"]
                + step_result_events[agent_id]["cut_own"] * rewards_cfg["item_cut"]
                + step_action_types[agent_id][ACTION_TYPE_OWN_USEFUL_FOOD_DISPENSER] * rewards_cfg["useful_food_dispenser"]
                + step_action_types[agent_id][ACTION_TYPE_USEFUL_PLATE_DISPENSER] * rewards_cfg["useful_plate_dispenser"]
                + step_action_types[agent_id][ACTION_TYPE_OWN_USEFUL_CUTTING_BOARD] * rewards_cfg["useful_cutting_board"]
            )

            other_reward = (
                step_result_events[agent_id]["delivered_other"] * rewards_cfg["delivered"]
                + step_result_events[agent_id]["salad_other"] * rewards_cfg["salad_created"]
                + step_result_events[agent_id]["cut_other"] * rewards_cfg["item_cut"]
                + step_action_types[agent_id][ACTION_TYPE_OTHER_USEFUL_FOOD_DISPENSER] * rewards_cfg["useful_food_dispenser"]
                + step_action_types[agent_id][ACTION_TYPE_OTHER_USEFUL_CUTTING_BOARD] * rewards_cfg["useful_cutting_board"]
            )

            other_penalty = 0
            for other_agent_id in self.agents:
                if other_agent_id == agent_id:
                    continue
                other_penalty += (
                    step_result_events[other_agent_id]["delivered_other"] * rewards_cfg["delivered"]
                    + step_result_events[other_agent_id]["salad_other"] * rewards_cfg["salad_created"]
                    + step_result_events[other_agent_id]["cut_other"] * rewards_cfg["item_cut"]
                    + step_action_types[other_agent_id][ACTION_TYPE_OTHER_USEFUL_FOOD_DISPENSER] * rewards_cfg["useful_food_dispenser"]
                    + step_action_types[other_agent_id][ACTION_TYPE_OTHER_USEFUL_CUTTING_BOARD] * rewards_cfg["useful_cutting_board"]
                )

            pure_rewards[agent_id] = (self.payoff_matrix[0] * own_reward + self.payoff_matrix[1] * other_reward + self.payoff_matrix[2] * other_penalty)
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
                "result_events": step_result_events[agent],  # Add agent events to info
                "action_types": step_action_types[agent],  # Add action types to info
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
            row = {
                    "epoch": self.episode_count
                }
            
            for agent_id in self.agents:
                row[f"total_deliveries_{agent_id}"] = float(self.total_result_events[agent_id]["delivered_own"]) + float(self.total_result_events[agent_id]["delivered_other"])
                row[f"pure_reward_{agent_id}"] = float(self.cumulated_pure_rewards[agent_id])
                row[f"modified_reward_{agent_id}"] = float(self.cumulated_modified_rewards[agent_id])
                row[f"delivered_own_{agent_id}"] = self.total_result_events[agent_id]["delivered_own"]
                row[f"delivered_other_{agent_id}"] = self.total_result_events[agent_id]["delivered_other"]
                row[f"cut_own_{agent_id}"] = self.total_result_events[agent_id]["cut_own"]
                row[f"cut_other_{agent_id}"] = self.total_result_events[agent_id]["cut_other"]
                row[f"salad_own_{agent_id}"] = self.total_result_events[agent_id]["salad_own"]
                row[f"salad_other_{agent_id}"] = self.total_result_events[agent_id]["salad_other"]

                row[f"actions_asked_{agent_id}"] = self.total_actions_asked[agent_id]
                row[f"actions_not_performed_{agent_id}"] = self.total_actions_not_performed[agent_id]

                # Add action type columns
                row[f"do_nothing_{agent_id}"] = self.total_action_types[agent_id][ACTION_TYPE_DO_NOTHING]
                row[f"floor_actions_{agent_id}"] = self.total_action_types[agent_id][ACTION_TYPE_FLOOR]
                row[f"wall_actions_{agent_id}"] = self.total_action_types[agent_id][ACTION_TYPE_WALL]
                row[f"useless_counter_actions_{agent_id}"] = self.total_action_types[agent_id][ACTION_TYPE_USELESS_COUNTER]
                row[f"useful_counter_actions_{agent_id}"] = self.total_action_types[agent_id][ACTION_TYPE_USEFUL_COUNTER]
                row[f"useless_own_food_dispenser_actions_{agent_id}"] = self.total_action_types[agent_id][ACTION_TYPE_OWN_USELESS_FOOD_DISPENSER]
                row[f"useful_own_food_dispenser_actions_{agent_id}"] = self.total_action_types[agent_id][ACTION_TYPE_OWN_USEFUL_FOOD_DISPENSER]
                row[f"useless_other_food_dispenser_actions_{agent_id}"] = self.total_action_types[agent_id][ACTION_TYPE_OTHER_USELESS_FOOD_DISPENSER]
                row[f"useful_other_food_dispenser_actions_{agent_id}"] = self.total_action_types[agent_id][ACTION_TYPE_OTHER_USEFUL_FOOD_DISPENSER]
                row[f"useless_cutting_board_actions_{agent_id}"] = self.total_action_types[agent_id][ACTION_TYPE_USELESS_CUTTING_BOARD]
                row[f"useful_own_cutting_board_actions_{agent_id}"] = self.total_action_types[agent_id][ACTION_TYPE_OWN_USEFUL_CUTTING_BOARD]
                row[f"useful_other_cutting_board_actions_{agent_id}"] = self.total_action_types[agent_id][ACTION_TYPE_OTHER_USEFUL_CUTTING_BOARD]
                row[f"useless_plate_dispenser_actions_{agent_id}"] = self.total_action_types[agent_id][ACTION_TYPE_USELESS_PLATE_DISPENSER]
                row[f"useful_plate_dispenser_actions_{agent_id}"] = self.total_action_types[agent_id][ACTION_TYPE_USEFUL_PLATE_DISPENSER]
                row[f"useless_delivery_actions_{agent_id}"] = self.total_action_types[agent_id][ACTION_TYPE_USELESS_DELIVERY]
                row[f"useful_own_delivery_actions_{agent_id}"] = self.total_action_types[agent_id][ACTION_TYPE_OWN_USEFUL_DELIVERY]
                row[f"useful_other_delivery_actions_{agent_id}"] = self.total_action_types[agent_id][ACTION_TYPE_OTHER_USEFUL_DELIVERY]

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
