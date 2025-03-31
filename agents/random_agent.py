import random
from agents.base_agent import BaseAgent

class RandomAgent(BaseAgent):
    def choose_action(self, observation):
        return random.choice(["left", "right"])
