import random

from werkzeug.routing import Map

from engine.base_game import BaseGame
from coboxing.agent.base import Agent
from coboxing.game_objects import Box, Target
from engine.extensions.renderer2d.io import Mouse


class CoBoxing(BaseGame):
    def __init__(self, map_nr=None):
        super().__init__()
        self.box = Box(x=.4, y=.4)
        self.target = Target(x=.75, y=.75)

        self.gameObjects[self.box.id] = self.box
        self.gameObjects[self.target.id] = self.target
        self.order = random.choice([0, 1])

        self.running = True

    @property
    def nr_agents(self):
        return len([o for _, o in self.gameObjects.items() if isinstance(o, Agent)])

    @property
    def agents(self):
        return [o for _, o in self.gameObjects.items() if isinstance(o, Agent)]

    def add_agent(self, agent_id):
        if self.nr_agents % 2 == self.order:
            agent = Agent(agent_id, 'purple')
        else:
            agent = Agent(agent_id, 'green')
        agent_mouse = Mouse(agent_id=agent_id)
        agent.add_mouse(agent_mouse)
        self.gameObjects[agent.id] = agent

    def step(self, actions: dict, delta_time: float):
        if not self.running:
            return
        agents_pos_before = {}

        for agent in self.agents:
            agents_pos_before[agent.id] = {'x': agent.x, 'y': agent.y}

        super().step(actions, delta_time)
        self.collision(agents_pos_before)

        if self.check_win():
            self.target.drawable.left = random.random()
            self.target.drawable.top = random.random()

    def check_win(self):
        dx = (self.box.x - self.target.x) ** 2
        dy = (self.box.y - self.target.y) ** 2
        distance = (dx + dy) ** 0.5
        return distance < self.box.width / 4

    def collision(self, agents_pos_before):

        box_dx = 0
        box_dy = 0
        for agent in self.agents:
            # agent push box
            overlap_x, overlap_y = get_overlap(agent, self.box)
            added_width = (agent.width + self.box.width) / 2
            added_height = (agent.height + self.box.height) / 2

            _dx = agent.x - agents_pos_before[agent.id]['x']
            _dy = agent.y - agents_pos_before[agent.id]['y']

            if agent.kind == 'purple' and overlap_x > 0 and overlap_y > .5 * added_height:
                box_dx += _dx

            if agent.kind == 'green' and overlap_y > 0 and overlap_x > .5 * added_width:
                box_dy += _dy
        self.box.x += box_dx
        self.box.y += box_dy

        # box push agent
        for agent in self.agents:
            overlap_x, overlap_y = get_overlap(self.box, agent)
            if overlap_x > 0 and overlap_y > 0:
                added_width = (agent.width + self.box.width) / 2
                added_height = (agent.height + self.box.height) / 2

                if overlap_x < overlap_y:
                    if agent.x < self.box.x:
                        agent.x = self.box.x - added_width
                    else:
                        agent.x = self.box.x + added_width
                else:
                    if agent.y < self.box.y:
                        agent.y = self.box.y - added_height
                    else:
                        agent.y = self.box.y + added_height

            agent.sync()


def get_overlap(a, b):
    dx = (a.width + b.width) / 2 - abs(a.x - b.x)
    dy = (a.height + b.height) / 2 - abs(a.y - b.y)
    return dx, dy
