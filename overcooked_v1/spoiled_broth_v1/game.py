from engine.base_game import BaseGame

from engine.extensions.topDownGridWorld.grid import Grid
from spoiled_broth.world.tiles import Counter, CuttingBoard
from spoiled_broth.agent.base import Agent

from spoiled_broth.world.tiles import COLOR_MAP, CHAR_MAP
from spoiled_broth.ui.score import Score

from pathlib import Path

import numpy as np
import random

class SpoiledBroth(BaseGame):
    def __init__(self, map_nr=None, grid_size=(8, 8)):
        super().__init__()
        if map_nr is None:
            map_nr = str(random.randint(1, 4))  # default maps
        width, height = grid_size
        self.grid = Grid("grid", width, height, 16)
        map_path_img = Path(__file__).parent / "maps" / f"{map_nr}.png"
        map_path_txt = Path(__file__).parent / "maps" / f"{map_nr}.txt"

        if map_path_img.exists():
            # Load map through image
            self.grid.init_from_img(map_path_img, COLOR_MAP, self)
        elif map_path_txt.exists():
            # Load map through text
            self.grid.init_from_text(map_path_txt, CHAR_MAP, self)
        else:
            raise FileNotFoundError(f"Map'{map_nr}' not found, neither as image nor as text.")
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

MAX_PLAYERS = 4 # or import if needed

### LEGACY
#def game_to_vector(game, agent_id):
#    # Get both agents' info
#    agent_ids = [aid for aid in game.gameObjects if aid.startswith('ai_rl_')]
#    agent_vecs = []
#    tile_vecs = []
#    for aid in agent_ids:
#        agent = game.gameObjects[aid]
#        agent_vecs.append(agent.to_vector())
#        # Find the tile the agent is currently targeting (if any intent exists)
#        if hasattr(agent, 'intents') and agent.intents:
#            # Assume first intent is MoveToIntent or similar, get its target
#            intent = agent.intents[0]
#            target_tile = getattr(intent, 'target', None)
#            if target_tile is not None:
#                tile_vecs.append(target_tile.to_vector())
#            else:
#                tile_vecs.append(np.zeros(1 + 4 + 1, dtype=np.float32))  # type + 4 items + progress
#        else:
#            tile_vecs.append(np.zeros(1 + 4 + 1, dtype=np.float32))
#    # Flatten all vectors
#    obs = np.concatenate(agent_vecs + tile_vecs).astype(np.float32)
#    return obs


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

def game_to_obs_matrix(game, agent_id):
    """
    Returns a spatial observation for the current game state from the perspective of agent_id.
    - obs_matrix: shape (channels, H, W)
    - agent_inventory: shape (2, 4) (one-hot for [self, other] agent's held item)
    Channels:
      0: Self agent position
      1: Other agent position
      2: Counter tiles
      3: Dispenser tiles
      4: Cutting board tiles
      5: Delivery tiles
      6: Tomato on tile
      7: Plate on tile
      8: Tomato_cut on tile
      9: Tomato_salad on tile
      10: Progress (normalized, 0 for non-cutting board)
    """
    grid = game.grid
    H, W = grid.height, grid.width
    channels = 11
    obs = np.zeros((channels, H, W), dtype=np.float32)
    # Get agent ids and order as [self, other]
    all_agent_ids = [aid for aid in game.gameObjects if aid.startswith('ai_rl_')]
    if agent_id not in all_agent_ids:
        raise ValueError(f"agent_id {agent_id} not found in gameObjects")
    other_agent_id = [aid for aid in all_agent_ids if aid != agent_id]
    agent_order = [agent_id] + other_agent_id
    item_names = ["tomato", "plate", "tomato_cut", "tomato_salad"]
    # --- Agent positions ---
    for idx, aid in enumerate(agent_order):
        agent = game.gameObjects[aid]
        x, y = agent.slot_x, agent.slot_y
        if 0 <= x < W and 0 <= y < H:
            obs[idx, y, x] = 1.0  # Channel 0: self, Channel 1: other
    # --- Tile types and items ---
    for y in range(H):
        for x in range(W):
            tile = grid.tiles[x][y]
            if tile is None:
                continue
            t = getattr(tile, '_type', None)
            if t == 2:
                obs[2, y, x] = 1.0
            elif t == 3:
                obs[3, y, x] = 1.0
            elif t == 4:
                obs[4, y, x] = 1.0
            elif t == 5:
                obs[5, y, x] = 1.0
            item = getattr(tile, 'item', None)
            if item in item_names:
                obs[6 + item_names.index(item), y, x] = 1.0
            if t == 4:
                cut_stage = getattr(tile, 'cut_stage', 0)
                norm_prog = min(3, max(0, cut_stage)) / 3.0
                obs[10, y, x] = norm_prog
    # --- Agent inventory ---
    agent_inventory = np.zeros((2, 4), dtype=np.float32)
    for idx, aid in enumerate(agent_order):
        agent = game.gameObjects[aid]
        item = getattr(agent, 'item', None)
        if item in item_names:
            agent_inventory[idx, item_names.index(item)] = 1.0
    return obs, agent_inventory


