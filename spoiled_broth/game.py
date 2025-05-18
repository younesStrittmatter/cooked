from engine.base_game import BaseGame

from engine.extensions.topDownGridWorld.grid import Grid
from spoiled_broth.world.tiles import Counter, CuttingBoard
from spoiled_broth.agent.base import Agent

from spoiled_broth.world.tiles import COLOR_MAP
from spoiled_broth.ui.score import Score

from pathlib import Path

import numpy as np
import random

class SpoiledBroth(BaseGame):
    def __init__(self, map_nr=None):
        super().__init__()
        if map_nr is None:
            map_nr = random.randint(1, 4)
        self.grid = None
        img_path = Path(__file__).parent / "maps" / f"{map_nr}.png"
        self.grid = Grid("grid", 8, 8, 16)
        self.grid.init_from_img(img_path, COLOR_MAP, self)
        self.score = Score()

        self.gameObjects['grid'] = self.grid
        self.gameObjects['score'] = self.score

    def add_agent(self, agent_id):
        agent = Agent(agent_id, self.grid, self)
        # set agent's initial position to walkable tile
        choices = []
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                tile = self.grid.tiles[x][y]
                if tile and tile.is_walkable:
                    choices.append(tile)
        start_tile = random.choice(choices)
        agent.x = start_tile.slot_x * self.grid.tile_size + self.grid.tile_size // 2
        agent.y = start_tile.slot_y * self.grid.tile_size + self.grid.tile_size // 2
        self.gameObjects[agent_id] = agent

    def step(self, actions: dict, delta_time: float):
        super().step(actions, delta_time)

MAX_PLAYERS = 4  # or import if needed

def game_to_vector(game, agent_id):
    grid = game.grid
    agent = game.gameObjects[agent_id]
    tile_obs = []
    for y in range(grid.height):
        for x in range(grid.width):
            tile = grid.tiles[x][y]
            if tile:
                tile_obs.extend(tile.to_vector())

    self_obs = agent.to_vector()
    len_agent_vector = len(self_obs)

    other_obs = []
    for other_id in game.gameObjects:
        if other_id != agent_id and hasattr(game.gameObjects[other_id], "to_vector"):
            other_obs.extend(game.gameObjects[other_id].to_vector())

    n_other = len([aid for aid in game.gameObjects if aid != agent_id and hasattr(game.gameObjects[aid], "to_vector")])
    if n_other < MAX_PLAYERS - 1:
        other_obs.extend([0.0] * (len_agent_vector * (MAX_PLAYERS - 1 - n_other)))

    return np.concatenate([tile_obs, self_obs, other_obs]).astype(np.float32)


def random_game_state(game, item_list=['tomato', 'plate', 'tomato_cut', 'tomato_salad']):
    chance = random.random()
    if chance < 0.4: # easy
        _item_list = item_list + ['tomato_salad']
    elif chance < 0.6: # mid
        _item_list = [i for i in item_list if not i.endswith('salad')]
    elif chance < 0.8: # hard
        _item_list = [i for i in item_list if not (i.endswith('cut') or i.endswith('salad'))] + [None]
    else:
        _item_list = [None]

    for x in range(game.grid.width):
        for y in range(game.grid.height):
            tile = game.grid.tiles[x][y]
            if isinstance(tile, Counter):
                if random.random() < .6:
                    tile.item = random.choice(_item_list)
            if isinstance(tile, CuttingBoard):
                if random.random() < .6 and chance < 0.6:
                    tile.item = random.choice(['tomato', None])

    # Optionally randomize agent inventory
    for agent in game.gameObjects.values():
        if hasattr(agent, "item"):
            if random.random() < .3:
                agent.item = random.choice(_item_list)


