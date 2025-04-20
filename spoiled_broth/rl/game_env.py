import random

from pettingzoo import ParallelEnv
from pettingzoo.utils import agent_selector
from gymnasium import spaces
import numpy as np

from spoiled_broth.game import SpoiledBroth, game_to_vector, random_game_state

MAX_PLAYERS = 4


class GameEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "game_v0"}

    def __init__(self):
        super().__init__()
        self._step_counter = 0
        self._max_steps_per_episode = 10000
        self.render_mode = None
        self.game = SpoiledBroth()
        self.possible_agents = ["ai_rl_1", "ai_rl_2"]
        self.agents = self.possible_agents[:]
        self.agent_selector = agent_selector(self.agents)
        self._agent_selector = self.agent_selector

        for agent_id in self.agents:
            self.game.add_agent(agent_id)

        self.agent_map = {
            agent_id: self.game.gameObjects[agent_id] for agent_id in self.agents
        }

        example_obs = self.observe(self.agents[0])
        obs_size = example_obs.shape[0]
        self.observation_spaces = {
            agent: spaces.Box(low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32)
            for agent in self.agents
        }

        grid_size = self.game.grid.width * self.game.grid.height
        self.action_spaces = {
            agent: spaces.Discrete(grid_size)
            for agent in self.agents
        }
        self._clickable_mask = np.zeros(self.game.grid.width * self.game.grid.height, dtype=np.int8)
        for x in range(self.game.grid.width):
            for y in range(self.game.grid.height):
                tile = self.game.grid.tiles[x][y]
                if tile and tile.clickable is not None:
                    index = y * self.game.grid.width + x
                    self._clickable_mask[index] = 1

        self.rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {'action_mask': self._clickable_mask} for agent in self.agents}
        self._last_score = 0

    def reset(self, seed=None, options=None):
        self._last_score = 0
        self.agents = self.possible_agents[:]

        self.game = SpoiledBroth()
        for agent_id in self.agents:
            self.game.add_agent(agent_id)

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
        for agent_id, action in actions.items():
            # Convert flat index to tile
            grid_w = self.game.grid.width
            x = action % grid_w
            y = action // grid_w
            tile = self.game.grid.tiles[x][y]

            if tile and tile.clickable is not None:
                tile.click(agent_id)

        # Advance the game state by one step (simulate one tick)
        self.game.step({}, delta_time=1.)  # You can adjust delta_time for training speed

        # Update rewards (here: use the agents' internal score values directly)
        new_score = self.game.gameObjects["score"].score
        reward = new_score - self._last_score -.01
        if reward > 0:
            print(f"Agent {agent_id} received reward: {reward}")
        self._last_score = new_score
        for agent_id in self.agents:
            self.rewards[agent_id] = reward

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
            self._step_counter = 0  # 👈 Reset step count
            print("[Env] 🔁 Truncating episode and triggering reset...")

        terminations = self.dones
        truncations = {agent: should_truncate for agent in self.agents}  # or your own logic

        return observations, self.rewards, terminations, truncations, self.infos

    def render(self):
        print(f"[Game Render] Agents: {self.agents}")

    def close(self):
        pass
