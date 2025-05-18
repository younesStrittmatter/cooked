import random

from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np
from spoiled_broth.config import *
from spoiled_broth.game import SpoiledBroth, game_to_vector, random_game_state

MAX_PLAYERS = 4

NON_CLICKABLE_PENALTY = 0.05
IDLE_PENALTY = 0.01

REWARD_ITEM_CUT = 0.1
REWARD_SALAD_CREATED = 0.2 + REWARD_ITEM_CUT
REWARD_DELIVERED = 5. + REWARD_SALAD_CREATED

STEP_PER_EPISODE = 1000

def init_game(agents, map_nr=1):
    game = SpoiledBroth(map_nr=map_nr)
    for agent_id in agents:
        game.add_agent(agent_id)

    grid_size = game.grid.width * game.grid.height
    action_spaces = {
        agent: spaces.Discrete(grid_size)
        for agent in agents
    }
    _clickable_mask = np.zeros(game.grid.width * game.grid.height, dtype=np.int8)
    for x in range(game.grid.width):
        for y in range(game.grid.height):
            tile = game.grid.tiles[x][y]
            if tile and tile.clickable is not None:
                index = y * game.grid.width + x
                _clickable_mask[index] = 1

    return game, action_spaces, _clickable_mask

class GameEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "game_v0"}

    def __init__(self, reward_weights=None, map_nr=1):
        super().__init__()
        self.map_nr = map_nr
        self._step_counter = 0
        self._max_steps_per_episode = STEP_PER_EPISODE
        self.render_mode = None

        self.possible_agents = ["ai_rl_1", "ai_rl_2"]
        self.agents = self.possible_agents[:]

        default_weights = {agent: (1.0, 0.0) for agent in self.agents}
        self.reward_weights = reward_weights if reward_weights is not None else default_weights
        self.total_agent_events = {agent_id: {"delivered": 0, "cut": 0, "salad": 0} for agent_id in self.agents}

        print('Weighted rewards:\n')
        print(f'Agent 1: {self.reward_weights["ai_rl_1"]}\n')
        print(f'Agent 2: {self.reward_weights["ai_rl_2"]}\n')

        self.game, self.action_spaces, self._clickable_mask = init_game(self.agents, map_nr=self.map_nr)

        self.agent_map = {
            agent_id: self.game.gameObjects[agent_id] for agent_id in self.agents
        }

        example_obs = self.observe(self.agents[0])
        obs_size = example_obs.shape[0]
        self.observation_spaces = {
            agent: spaces.Box(low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32)
            for agent in self.agents
        }

        self.rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {'action_mask': self._clickable_mask} for agent in self.agents}
        self._last_score = 0

    def reset(self, seed=None, options=None):
        self._last_score = 0
        self.agents = self.possible_agents[:]

        self.game, self.action_spaces, self._clickable_mask = init_game(self.agents, map_nr=self.map_nr)
        random_game_state(self.game)

        self.agent_map = {
            agent_id: self.game.gameObjects[agent_id] for agent_id in self.agents
        }

        self.rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {
            agent: {"action_mask": self._clickable_mask.copy()}
            for agent in self.agents
        }
        self._last_score = 0

        return {agent: self.observe(agent) for agent in self.agents}, self.infos

    def observe(self, agent):
        return game_to_vector(self.game, agent)

    def step(self, actions):
        self._step_counter += 1
        agent_penalties = {agent_id: 0.0 for agent_id in self.agents}
        for agent_id, action in actions.items():
            grid_w = self.game.grid.width
            x = action % grid_w
            y = action // grid_w
            tile = self.game.grid.tiles[x][y]

            if tile and tile.clickable is not None:
                tile.click(agent_id)
            else:
                agent_penalties[agent_id] = NON_CLICKABLE_PENALTY

        def decode_action(action_int):
            return {"type": "click", "target": int(action_int)}

        actions_dict = {agent_id: decode_action(action) for agent_id, action in actions.items()}
        self.game.step(actions_dict, delta_time=1 / cf_AI_TICK_RATE)

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

                        if hasattr(tile, "salad_created_by") and tile.salad_created_by:
                            if tile.salad_created_by in agent_events:
                                agent_events[tile.salad_created_by]["salad"] += 1
                                self.total_agent_events[tile.salad_created_by]["salad"] += 1
                            tile.salad_created_by = None

        new_score = self.game.gameObjects["score"].score
        if (new_score - self._last_score) > 0:
            print(f"Agent delivered item, new score: {new_score}")
            for agent_id in agent_events:
                agent_events[agent_id]["delivered"] += 1
                self.total_agent_events[agent_id]["delivered"] += 1
            self._last_score = new_score

        pure_rewards = {}
        for agent_id in self.agents:
            pure_rewards[agent_id] = (
                agent_events[agent_id]["delivered"] * REWARD_DELIVERED +
                agent_events[agent_id]["cut"] * REWARD_ITEM_CUT +
                agent_events[agent_id]["salad"] * REWARD_SALAD_CREATED -
                agent_penalties[agent_id] - IDLE_PENALTY
            )

        for agent_id in self.agents:
            alpha, beta = self.reward_weights.get(agent_id, (1.0, 0.0))
            other_agents = [a for a in self.agents if a != agent_id]
            if other_agents:
                avg_other_reward = sum(pure_rewards[a] for a in other_agents) / len(other_agents)
            else:
                avg_other_reward = 0.0
            self.rewards[agent_id] = alpha * pure_rewards[agent_id] + beta * avg_other_reward

        self.dones = {agent: False for agent in self.agents}
        self.infos = {
            agent: {"action_mask": self._clickable_mask}
            for agent in self.agents
        }

        observations = {
            agent: self.observe(agent)
            for agent in self.agents
        }

        should_truncate = self._step_counter >= self._max_steps_per_episode
        if should_truncate:
            self._step_counter = 0

        terminations = self.dones
        truncations = {agent: should_truncate for agent in self.agents}

        return observations, self.rewards, terminations, truncations, self.infos

    def render(self):
        print(f"[Game Render] Agents: {self.agents}")

    def close(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]