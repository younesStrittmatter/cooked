import re
from typing import Callable

from engine.base_game import BaseGame

from engine.extensions.topDownGridWorld.grid import Grid

from spoiled_broth.world.tiles import Counter, CuttingBoard, Wall
from spoiled_broth.agent.base import Agent

from spoiled_broth.world.tiles import COLOR_MAP
from spoiled_broth.ui.score import Score
from spoiled_broth.ui.message import Message

from pathlib import Path

import numpy as np
import random

MAP_ID = 'encouraged_division_of_labor_large'

TICK_LOGGING = True



HAS_TRAIN_PHASE = True
TICK_PER_SECOND = 24
TRAIN_TICKS = 2 * 60 * TICK_PER_SECOND
TEST_TICKS = 3 * 60 * TICK_PER_SECOND
SET_TIMER = 5

HAS_COLLISION = True

SUM_TICKS = TEST_TICKS
if HAS_TRAIN_PHASE:
    SUM_TICKS = TRAIN_TICKS + TEST_TICKS + SET_TIMER * TICK_PER_SECOND

MAP_WIDTH = 10
MAP_HEIGHT = 10

map_dims = {
    'baseline_division_of_labor_v2': [8, 8],
}
map_has_training = {
    'baseline_division_of_labor_v2': False,
}
map_sum_tick = {
    'baseline_division_of_labor_v2': TEST_TICKS,
}

def reset_collision_grid(grid):
    for row in grid.tiles:
        for tile in row:
            tile.is_occupied = False

class SpoiledBroth(BaseGame):
    def __init__(self, map_nr=MAP_ID, url_params=None, tick_log_path=None):
        super().__init__()
        if map_nr is None:
            map_nr = random.randint(1, 4)
        if url_params and 'map' in url_params:
            map_nr = url_params['map'][0]
        self.map_nr = map_nr
        # self.grid = None
        img_path = Path(__file__).parent / "maps" / f"{map_nr}.png"
        width, height = MAP_WIDTH, MAP_HEIGHT
        if map_nr in map_dims:
            width, height = map_dims[map_nr]

        self.gameObjects['grid'] = Grid("grid", width, height, 16)


        self.gameObjects['grid'].init_from_img(img_path, COLOR_MAP, self)
        if HAS_COLLISION:
            reset_collision_grid(self.gameObjects['grid'])

        self.gameObjects['score'] = Score()

        self.tick_logging = False
        self.tick_count = 0
        # LOGGING
        if tick_log_path is not None:
            print(tick_log_path)
            self.tick_logging = True
            self._tick_rows = []
            self._tick_log_path = tick_log_path
        has_training = HAS_TRAIN_PHASE
        if map_nr in map_has_training:
            has_training = map_has_training[map_nr]
        if has_training:
            self.phase = "train"
            self.set_timer = SET_TIMER

    def redirect_link(self):
        return "https://spoiledbrothwaitingroom-7135b.web.app/endPage?score=" + str(self.gameObjects['score'].score)

    def add_agent(self, agent_id, url_params=None, **kwargs):

        grid = self.gameObjects['grid']
        if not 'player_nr' in kwargs:
            player_nr = len(self.get_agents()) + 1
        else:
            player_nr = kwargs['player_nr']
        agent = Agent(agent_id, grid, self, additional_info=url_params, player_nr=player_nr, is_collision=HAS_COLLISION)

        # set agent's initial position to walkable tile
        choices = []
        for x in range(grid.width):
            for y in range(grid.height):
                tile = grid.tiles[x][y]
                if tile and tile.is_walkable:
                    choices.append(tile)
        start_tile = random.choice(choices)

        if not 'x' in kwargs:
            agent.x = start_tile.slot_x * grid.tile_size + grid.tile_size // 2
        else:
            agent.x = kwargs['x']
        if url_params and f'slot_x_p{player_nr}' in url_params:
            agent.x = (int(url_params[f'slot_x_p{player_nr}'][0]) * grid.tile_size + grid.tile_size // 2)
        agent.start_x = agent.x
        if not 'y' in kwargs:
            agent.y = start_tile.slot_y * grid.tile_size + grid.tile_size // 2
        else:
            agent.y = kwargs['y']
        if url_params and f'slot_y_p{player_nr}' in url_params:
            agent.y = (int(url_params[f'slot_y_p{player_nr}'][0]) * grid.tile_size + grid.tile_size // 2)
        agent.start_y = agent.y
        self.gameObjects[agent_id] = agent

    def remove_agent(self, agent_id):
        if agent_id in self.gameObjects:
            del self.gameObjects[agent_id]

    def get_agents(self):
        return [obj for obj in self.gameObjects.values() if hasattr(obj, '_is_agent') and obj._is_agent]

    def reset(self, all=False):

        # self.grid = None

        agents = self.get_agents()
        for agent in agents:
            agent.x = agent.start_x
            agent.y = agent.start_y
            agent.move_target = None
            agent.item = None
            agent.action = None
            agent.score = 0
            agent.reset()

        if not all:
            return
        # 1) Remove all non-agent, non-message objects
        keys_to_delete = [
            key for key, obj in self.gameObjects.items()
            if key != "message" and not (hasattr(obj, "_is_agent") and obj._is_agent)
        ]
        for key in keys_to_delete:
            del self.gameObjects[key]

        img_path = Path(__file__).parent / "maps" / f"{self.map_nr}.png"
        width, height = MAP_WIDTH, MAP_HEIGHT
        if self.map_nr in map_dims:
            width, height = map_dims[self.map_nr]
        grid = Grid("grid", width, height, 16)
        grid.init_from_img(img_path, COLOR_MAP, self)


        self.gameObjects["grid"] = grid
        if HAS_COLLISION:
            reset_collision_grid(self.gameObjects['grid'])
        self.gameObjects["score"] = Score()

        # 3) Keep a direct grid attribute if other code uses game.grid
        self.grid = grid

        # 4) Re-wire agents to the new grid (if they keep a reference)
        for agent in agents:
            if hasattr(agent, "grid"):
                agent.grid = grid

    def step(self, actions: dict, delta_time: float):
        if HAS_COLLISION:
            reset_collision_grid(self.gameObjects['grid'])
            agents = self.get_agents()
            for agent in agents:
                if hasattr(agent, "set_collision"):
                    agent.set_collision()
        super().step(actions, delta_time)
        self.tick_count += 1
        has_training = HAS_TRAIN_PHASE

        if self.map_nr in map_has_training:
            has_training = map_has_training[self.map_nr]
        if has_training and self.tick_count >= TRAIN_TICKS and self.phase == 'train':
            self.phase = 'ready'
            self.reset(all=True)
            self.gameObjects['message'] = Message(message=
                "This was the practice round.\n"
                "The real game begins now.\n"
                "From now on, your score will be recorded.\n"
                f"Starting in {self.set_timer} seconds...\n"
            )
        if has_training and self.phase == 'ready':
            self.reset(all=False)
            if self.tick_count % 24 == 0:
                self.set_timer -= 1
                self.gameObjects['message'].set_message(
                    "This was the practice round.\n"
                    "The real game begins now.\n"
                    "From now on, your score will be recorded.\n"
                    f"Starting in {self.set_timer} seconds...\n"
                )

        if has_training and self.set_timer < 0 and self.phase == "ready":
            self.reset(all=True )
            self.phase = "test"
            self.gameObjects['message'].set_message(message="")

        if self.tick_logging:
            self._store_tick_data()

    def _flatten_serialized(self, obj: dict, prefix: str, key_filter: Callable[[str], bool] = lambda k: True) -> dict:
        flat = {}
        for key, value in obj.items():
            if key == "children":
                continue  # Skip children here; they'll be visited separately

            full_key = f"{prefix}_{key}"
            if not key_filter(full_key):
                continue

            if isinstance(value, dict):
                nested = self._flatten_serialized(value, prefix=full_key, key_filter=key_filter)
                flat.update(nested)
            else:
                flat[full_key] = value

        return flat
        # Store tick data for debugging or analysis

    def _store_tick_data(self):
        if not self.tick_logging:
            return
        try:
            import pandas as pd
        except ImportError:
            print("Pandas is required for tick logging. Please install pandas.")
            # Build snapshot for this tick
        snapshot = {"tick": self.tick_count}
        has_training = HAS_TRAIN_PHASE
        if self.map_nr in map_has_training:
            has_training = map_has_training[self.map_nr]
        if has_training:
            snapshot['phase'] = self.phase
        snapshot['collision'] = HAS_COLLISION

        def process(obj):
            serialized = obj.full_serialize()
            flat = self._flatten_serialized(
                serialized["data"],
                prefix=obj.id,
                key_filter=is_not_excludes
            )
            snapshot.update(flat)
            for child in getattr(obj, "children", []):
                process(child)

        for obj in self.gameObjects.values():
            process(obj)

        # Buffer this tick
        self._tick_rows.append(snapshot)

        # If the session just ended, write the whole thing once
        sum_ticks = SUM_TICKS
        if self.map_nr in map_sum_tick:
            sum_ticks = map_sum_tick[self.map_nr]
        if self.tick_count == sum_ticks:
            print("Writing")

            if not self._tick_rows:
                return  # nothing to write
            print('Data found. Writing...')

            df = pd.DataFrame(self._tick_rows)

            # keep 'tick' as the first column
            cols = ["tick"] + [c for c in df.columns if c != "tick"]
            df = df[cols]

            # Write exactly once
            out_path = getattr(self, "_tick_log_path", "ticks.csv")

            df.to_csv(out_path, index=False)
            print("wrote to {}".format(out_path))

            # Optional: keep in memory for immediate analysis
            self._tick_df = df

    def initial_args(self) -> dict:
        """
        Returns the initial state of the game.
        This can be used to reset the game or for training purposes.
        """
        return {
            "map_nr": self.map_nr,
        }

    def agent_initial_state(self, agent_id: str) -> dict:
        """
        Returns the initial state of a specific agent.
        This can be used to reset the agent's state or for training purposes.
        """
        return {
            "agent_id": agent_id,
            "x": self.gameObjects[agent_id].x,
            'y': self.gameObjects[agent_id].y,
            "item": self.gameObjects[agent_id].item,
            "url_params": self.gameObjects[agent_id].additional_info,
            "cut_speed": getattr(self.gameObjects[agent_id], "cut_speed", 1.0),
            "walk_speed": getattr(self.gameObjects[agent_id], "speed", 1.0),
            "player_nr": self.gameObjects[agent_id].player_nr,
        }


def is_not_excludes(key: str) -> bool:
    if re.search(r'_drawable_', key):
        return False
    if re.search(r'_text_', key):
        return False
    if re.search('_additional_info', key):
        if not re.search(r'_PROLIFIC_PID', key):
            return False
    if key.endswith('id'):
        return False
    if key == 'grid_tiles':
        return False
    if key.startswith('Floor_'):
        return False
    if key.startswith('Wall_'):
        return False
    if key.endswith('left'):
        return False
    if key.endswith('top'):
        return False
    if key.endswith('width'):
        return False
    if key.endswith('height'):
        return False
    if key.endswith('isClickable'):
        return False
    if key == 'grid_tile_size':
        return False
    if key.endswith('clickable'):
        return False
    if key.endswith('class'):
        return False
    return True


MAX_PLAYERS = 4  # or import if needed


def game_to_vector(game, agent_id):
    """
    For RL Agents, transform the game state into a vector representation.
    """
    grid = game.gameObjects['grid']
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


def game_to_prompt(game, agent_id):
    """
    For LLM Agents, transform the game state into a prompt representation.
    """
    grid = game.gameObjects['grid']
    agent = game.gameObjects[agent_id]
    _p = f"""
You are playing a cooperative grid-based game with other agents as a team.
The shared goal is to deliver as many *tomato_salad* as possible.

You interact with the game by choosing a tile.
Choosing a tile makes you walk there and interact (e.g., pick up, place, cut, deliver).

You must think carefully: choose the tile that let's your team deliver a salad quickest.

Game Basics:

- To deliver a *tomato_salad*, you must hold it and place it on the delivery tile.
- To make a *tomato_salad*, you must hold a *tomato_cut* and place it on a *counter* that already contains a *plate*.
- To make a *tomato_cut*, you must choose a *cutting_board* that contains a *tomato*.
- To place a *tomato* on the cutting board, you must hold it and choose the cutting board.
- To place any item on a counter, you must hold it and place it on a counter.
- To pick up an item, you must choose the tile that contains it (dispenser, counter, cutting board).

Choosing a tile means walking to it (if reachable) and interacting with it. You cannot walk through walls or counters. Distance matters — far-away tiles take longer to reach. Try to think about the layout of the game since you might have to walk around counters or walls.
Here is the current state of the game, tile by tile including the action taken when chosing the tile. The floor tiles are omitted. The full grid is 8x8. The tile coordinates are (x, y) where x is the column and y is the row. The top left corner is (0, 0) and the bottom right corner is (7, 7):
"""

    for y in range(grid.height):
        for x in range(grid.width):
            tile = grid.tiles[x][y]
            if tile and not tile.is_walkable and not (
                    isinstance(tile, Wall) and (x == 0 or x == 7 or y == 0 or y == 7)):
                _p += f"\n{tile.to_prompt(agent)}"

    options = []
    for y in range(grid.height):
        for x in range(grid.width):
            tile = grid.tiles[x][y]
            if tile and not tile.is_walkable and not isinstance(tile, Wall):
                if hasattr(tile, "to_prompt") and '[None]' not in tile.to_prompt(agent):
                    options.append(f"({x}, {y})")

    _p += """
Some tiles show on_choose: [None].  These are not useful right now, but they help you understand the layout or other agents."""

    _p += """
Descriptions of the other agents:
"""

    for other_id in game.gameObjects:
        object = game.gameObjects[other_id]
        if other_id != agent_id and isinstance(object, Agent) and hasattr(object, "to_prompt"):
            _p += f"{object.to_prompt(False)}"

    _p += """
your own state state:
"""

    _p += f"{agent.to_prompt(True)}"

    _p += """
Before choosing, ask yourself:

- What ist the next step for your team?"""
    if agent.item:
        _p += f"""
- You are currently holding a {agent.item}."""
    if agent.item in ['tomato', 'pumpkin', 'cabbage']:
        _p += f"""
- You can cut the {agent.item} on a cutting board or on a counter. Is there an empty cutting board?
- Is it better to cut it yourself or to place it on a counter up for someone else to pick it up and cut it? (Consider walking distance for you and other agents.)
- If there is no empty cutting board, you should place your item on a counter or get a different item from a dispenser."""
    if agent.item in ['plate']:
        _p += f"""
- You can put the {agent.item} on a counter.
- Where should you place it? Or is it unnecessary and you need to pick up a tomato instead?"""
    if agent.item in ['tomato_cut', 'pumpkin_cut', 'cabbage_cut']:
        _p += f"""
- Can you place this item on a counter with a plate on it to assemble a salad? Is there a counter with a plate?
- Should you place it on an empty counter instead? For example, to get a plate first or for another agent to pick it up."""
    if agent.item in ['tomato_salad', 'pumpkin_salad', 'cabbage_salad']:
        _p += f"""
- You can deliver the {agent.item} or put it a counter. Only put it on a counter if you are sure that you or someone else will pick it up soon.
"""

    if agent.item is None:
        _p += f"""
- What item must you hold next to enable the next step toward delivering a tomato_salad? (Think about the full sequence: cut → assemble → deliver.) You need all the items so if there is a counter with a plate, you need a tomato_cut but if there is a tomato_cut, you need a plate.
- What items are currently available on tiles (counters, dispensers, cutting boards)?
- Is it possible to immediately cut, assemble, or deliver based on the available items? (If yes, act; if not, set up for the next step.)
- Which available tiles allow you to make progress with the fewest moves? (Consider walking distance and interaction steps.)
"""
    _p += f"""
Work with your team. You don’t need to do everything yourself. Others can also pick up, cut, assemble, or deliver. Sometimes it’s best to set up for someone else — especially if it saves walking time.
Now choose the tile that enables the next useful step for your team.

Format your answer as (x, y) — or '' if you don’t want to choose anything right now."""

    _p += f"""
The allowed tiles are: {options}.
You must choose only from the given list of allowed tiles.
"""
    _p += f"""Only give the tile, and nothing else."""
    return _p


def random_game_state(game, item_list=['tomato', 'plate', 'tomato_cut', 'tomato_salad']):
    chance = random.random()
    if chance < 0.4:  # easy
        _item_list = item_list + ['tomato_salad']
    elif chance < 0.6:  # mid
        _item_list = [i for i in item_list if not i.endswith('salad')]
    elif chance < 0.8:  # hard
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
