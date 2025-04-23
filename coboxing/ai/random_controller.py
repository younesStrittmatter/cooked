from engine.ai.controller import Controller
import random


class RandomWalk(Controller):
    def __init__(self, agent_id):
        super().__init__(agent_id)
        self.x = 0.5
        self.y = 0.5

    def choose_action(self, observation: dict):
        self.x += (random.random() - .5) * .1
        self.y += (random.random() - .5) * .1
        if self.x > 1:
            self.x = 1
        if self.x < 0:
            self.x = 0
        if self.y > 1:
            self.y = 1
        if self.y < 0:
            self.y = 0

        return {'type': 'mouse',
                'x': self.x,
                'y': self.y,
                'leftDown': True,
                'rightDown': False}
