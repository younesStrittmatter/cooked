from engine.base_game import BaseGame
from template_game.agent.base import Agent


class Game(BaseGame):
    def __init__(self):
        super().__init__()

    def add_agent(self, agent_id):
        agent = Agent(agent_id)
        self.gameObjects[agent_id] = agent

    def step(self, actions: dict, delta_time: float):
        super().step(actions, delta_time)
