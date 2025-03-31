from engine.base_game import BaseGame
import random

from engine.extensions.gridworld import grid
from engine.extensions.renderer2d.basic_2d import Basic2D
from engine.game_object import GameObject
from engine.extensions.gridworld.grid import Grid, Tile


class SpoiledBroth(BaseGame):
    def __init__(self):
        super().__init__()
        self.grid = None
        self.grid = Grid("grid", 8, 8, 20)
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                tile = FloorTile(id=None, game=self)
                self.grid.add_tile(tile, x, y)

        self.gameObjects['grid'] = self.grid

    def add_agent(self, agent_id):
        pic = Basic2D(f'{agent_id}_image',
                      0,
                      0,
                      20,
                      20,
                      'agent1.jpeg')

        agent = PlayerObject(agent_id, pic)
        self.gameObjects[agent_id] = agent


class FloorTile(Tile):
    def __init__(self, id, game=None):
        def on_click(agent_id):
            if agent_id in game.gameObjects:
                game.gameObjects[agent_id].set_move_target(self.x, self.y)

        super().__init__(id, 'floor/basic-floor.png', on_click=on_click)


class PlayerObject(GameObject):
    def __init__(self, id, image):
        super().__init__(id=id)
        self.x = random.randint(50, 100)
        self.y = random.randint(50, 100)
        self.image = image
        self.target = None
        self.move_target_x = None
        self.move_target_y = None
        self.speed = 10

    def set_move_target(self, x, y):
        self.move_target_x = x
        self.move_target_y = y

    def move(self, delta_time):

        if self.move_target_x is not None:
            if not close(self.x, self.move_target_x):
                if self.x < self.move_target_x:
                    self.x += self.speed * delta_time
                elif self.x > self.move_target_x:
                    self.x -= self.speed * delta_time
        if self.move_target_y is not None:
            if not close(self.y, self.move_target_y):
                if self.y < self.move_target_y:
                    self.y += self.speed * delta_time
                elif self.y > self.move_target_y:
                    self.y -= self.speed * delta_time

        if close(self.x, self.move_target_x) and close(self.y, self.move_target_y):
            self.move_target_x = None
            self.move_target_y = None

    def sync(self):
        self.image.x = self.x
        self.image.y = self.y

    def update(self, actions: dict, delta_time: float):
        self.move(delta_time)
        self.sync()
        pass

    def serialize(self) -> dict:
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
        }


def close(a, b, tolerance=2):
    if a is None or b is None:
        return True
    return abs(a - b) < tolerance
