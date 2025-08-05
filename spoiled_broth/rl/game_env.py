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
REWARD_SALAD_CREATED = 0.2 + REWARD_ITEM_CUT  # (+ REWARD_ITEM_CUT sinc creating a salad "loses a cut item")
REWARD_DELIVERED = 5. + REWARD_SALAD_CREATED  # (+ REWARD_SALAD_CREATED since delivering "loses a salad")

STEP_PER_EPISODE = 1000


def init_game(agents):
    map_nr = random.randint(1, 4)
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

    def __init__(self):
        super().__init__()
        self._step_counter = 0
        self._max_steps_per_episode = STEP_PER_EPISODE
        self.render_mode = None

        self.possible_agents = ["ai_rl_1", "ai_rl_2"]
        self.agents = self.possible_agents[:]

        self.game, self.action_spaces, self._clickable_mask = init_game(self.agents)

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
        self.salads_last = 0
        self.cut_last = 0
        self.cut_time_accumulated = 0
        for thing in self.game.gameObjects.values():
            if hasattr(thing, 'tiles'):
                tiles = thing.tiles
                for r in tiles:
                    for t in r:
                        if hasattr(t, "item"):
                            if t.item and t.item.endswith("_salad"):
                                self.salads_last += 1
                            if t.item and t.item.endswith("_cut"):
                                self.cut_last += 1
                        if hasattr(t, "cut_time_accumulated"):
                            self.cut_time_accumulated += t.cut_time_accumulated

            if hasattr(thing, "item"):
                if thing.item and thing.item.endswith("_salad"):
                    self.salads_last += 1
                if thing.item and thing.item.endswith("_cut"):
                    self.cut_last += 1
            if hasattr(thing, "cut_time_accumulated"):
                self.cut_time_accumulated += thing.cut_time_accumulated

    def reset(self, seed=None, options=None):
        self._last_score = 0
        self.agents = self.possible_agents[:]

        self.game, self.action_spaces, self._clickable_mask = init_game(self.agents)

        random_game_state(self.game)

        self.agent_map = {
            agent_id: self.game.gameObjects[agent_id] for agent_id in self.agents
        }

        self.rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {
            agent: {"action_mask": self._clickable_mask.copy()}  # always copy to avoid pointer issues
            for agent in self.agents
        }
        self._last_score = 0

        # since the game is initialzed with random items, we need to set the number of salads and cut items
        self.salads_last = 0
        self.cut_last = 0
        self.cut_time_accumulated = 0
        for thing in self.game.gameObjects.values():
            if hasattr(thing, 'tiles'):
                tiles = thing.tiles
                for r in tiles:
                    for t in r:
                        if hasattr(t, "item"):
                            if t.item and t.item.endswith("_salad"):
                                self.salads_last += 1
                            if t.item and t.item.endswith("_cut"):
                                self.cut_last += 1
                        if hasattr(t, "cut_time_accumulated"):
                            self.cut_time_accumulated += t.cut_time_accumulated

            if hasattr(thing, "item"):
                if thing.item and thing.item.endswith("_salad"):
                    self.salads_last += 1
                if thing.item and thing.item.endswith("_cut"):
                    self.cut_last += 1
            if hasattr(thing, "cut_time_accumulated"):
                self.cut_time_accumulated += thing.cut_time_accumulated

        assert isinstance(self.infos, dict), f"infos is not a dict: {self.infos}"
        assert all(isinstance(v, dict) for v in self.infos.values()), "infos values must be dicts"

        return {
            agent: self.observe(agent)
            for agent in self.agents
        }, self.infos

    def observe(self, agent):
        return game_to_vector(self.game, agent)

    def step(self, actions):
        self._step_counter += 1
        # Submit intents from each agent

        # If the agent clicks on a "non-clickable" tile, apply a penalty
        agent_penalties = {agent_id: 0.0 for agent_id in self.agents}
        for agent_id, action in actions.items():
            # Convert flat index to tile
            grid_w = self.game.grid.width
            x = action % grid_w
            y = action // grid_w
            tile = self.game.grid.tiles[x][y]

            if tile and tile.clickable is not None:
                tile.click(agent_id)
            else:
                agent_penalties[agent_id] = NON_CLICKABLE_PENALTY

        # Advance the game state by one step (simulate one tick)
        self.game.step({}, delta_time=1 / cf_AI_TICK_RATE)

        # Calculate intermediate rewards based on the number of salads and cut items (this is combined for all agents)
        salads_total = 0
        cut_items = 0
        cut_time_accumulated = 0
        for thing in self.game.gameObjects.values():
            if hasattr(thing, 'tiles'):
                tiles = thing.tiles
                for r in tiles:
                    for t in r:
                        if hasattr(t, "item"):
                            if t.item and t.item.endswith("_salad"):
                                salads_total += 1
                            if t.item and t.item.endswith("_cut"):
                                cut_items += 1
                        if hasattr(t, "cut_time_accumulated"):
                            cut_time_accumulated += t.cut_time_accumulated

            if hasattr(thing, "item"):
                if thing.item and thing.item.endswith("_salad"):
                    salads_total += 1
                if thing.item and thing.item.endswith("_cut"):
                    cut_items += 1
            if hasattr(thing, "cut_time_accumulated"):
                cut_time_accumulated += thing.cut_time_accumulated

        intermediate_rewards = ((salads_total - self.salads_last) * REWARD_SALAD_CREATED
                                + (cut_items - self.cut_last) * REWARD_ITEM_CUT +
                                (cut_time_accumulated - self.cut_time_accumulated) * REWARD_ITEM_CUT / 8)

        # Only give the reward once
        self.salads_last = salads_total
        self.cut_last = cut_items
        self.cut_time_accumulated = cut_time_accumulated

        # Get reward from delivering items

        new_score = self.game.gameObjects["score"].score
        reward = ((new_score - self._last_score) * REWARD_DELIVERED
                  + intermediate_rewards
                  - IDLE_PENALTY)

        if (new_score - self._last_score) > 0:
            print(f"[GameEnv] Agent delivered item, new score: {new_score}")
        self._last_score = new_score
        for agent_id in self.agents:
            self.rewards[agent_id] = reward - agent_penalties[agent_id]

        # You can customize these based on game logic later
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
        truncations = {agent: should_truncate for agent in self.agents}  # or your own logic

        return observations, self.rewards, terminations, truncations, self.infos

    def render(self):
        print(f"[Game Render] Agents: {self.agents}")

    def close(self):
        pass
