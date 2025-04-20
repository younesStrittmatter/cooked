from engine.base_game import BaseGame

from engine.extensions.topDownGridWorld.grid import Grid
from spoiled_broth.agent.base import Agent

from spoiled_broth.world.tiles import COLOR_MAP
from spoiled_broth.ui.score import Score


class SpoiledBroth(BaseGame):
    def __init__(self):
        super().__init__()
        self.grid = None
        self.grid = Grid("grid", 8, 8, 16)
        self.grid.init_from_img("spoiled_broth/maps/1.png", COLOR_MAP, self)
        self.score = Score()

        self.gameObjects['grid'] = self.grid
        self.gameObjects['score'] = self.score

    def add_agent(self, agent_id):
        print(f'Adding agent {agent_id}')

        agent = Agent(agent_id, self.grid)

        self.gameObjects[agent_id] = agent

    def step(self, actions: dict, delta_time: float):
        super().step(actions, delta_time)
