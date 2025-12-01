from typing import List
from engine.game_object import GameObject
import math
from engine.extensions.topDownGridWorld.a_star import (
    grid_to_world, find_path, find_path_ghost, Node, path_length
)
from engine.extensions.topDownGridWorld.intent._base_intent import Intent

PHYSICS_STEPS = 100
TILE_SNAP_THRESHOLD = .2


class Agent(GameObject):
    _is_agent = True

    def __init__(self,
                 id,
                 grid,
                 x=None,
                 y=None,

                 ):
        super().__init__(id=id)
        self.x = 8 if x is None else x
        self.y = 8 if y is None else y
        self.grid = grid
        self.intents = []

        # Initialize pathfinding variables

        self.speed = 50
        self.path = []
        self.ghost_path = []
        self.ghost_path_index = 0
        self.is_partial = False
        self.path = []
        self.path_index = 0
        self.drawables = []
        self.move_target_x = None
        self.move_target_y = None

    def reset(self):
        self.path = []
        self.path_index = 0
        self.move_target_x = None
        self.move_target_y = None
        self.intents = []

    @property
    def slot_x(self):
        return int(self.x // self.grid.tile_size)

    @property
    def slot_y(self):
        return int(self.y // self.grid.tile_size)

    @property
    def node(self):
        return Node(self.slot_x, self.slot_y)

    def add_drawable(self, drawable):
        self.drawables.append(drawable)

    def set_move_target(self, tile):
        start = self.node
        goal = Node(tile.slot_x, tile.slot_y)
        self.path = []
        path = find_path(self.grid, start, goal)
        if path:
            self.path = path[1:]  # Skip current tile
            self.is_partial = False
            self.path_index = 0
            return
        ghost_path = find_path_ghost(self.grid, start, goal)
        if ghost_path:
            # Convert ghost path to partial real path
            partial = []
            for node in ghost_path[1:]:
                tile_obj = self.grid.tiles[node.x][node.y]
                if hasattr(tile_obj, "is_occupied") and tile_obj.is_occupied:
                    break  # stop at first real blocker
                partial.append(node)

            self.path = partial
            self.is_partial = True
            self.path_index = 0
            return

        # If even ghost A* fails -> unreachable (rare)
        self.path = []

    def get_path(self, tile):

        start = self.node
        if not isinstance(tile, Node):
            goal = Node(tile.slot_x, tile.slot_y)
        else:
            goal = tile
        path = find_path(self.grid, start, goal)
        if path:
            return path
        return None

    def get_path_ghost(self, tile):
        start = self.node
        if not isinstance(tile, Node):
            goal = Node(tile.slot_x, tile.slot_y)
        else:
            goal = tile
        path = find_path_ghost(self.grid, start, goal)
        if path:
            return path
        return None

    def get_distance(self, tile):
        path = self.get_path(tile)
        if path:
            return path_length(path)
        return None

    def get_distance_ghost(self, tile):
        path = self.get_path_ghost(tile)
        if path:
            return path_length(path)
        return None

    def move(self, delta_time):

        # if self.path_index >= len(self.path):
        #     return  # No path or finished
        #
        # next_tile = self.path[self.path_index]
        # target_pos = grid_to_world(next_tile, self.grid)

        d_t = delta_time / PHYSICS_STEPS

        for i in range(PHYSICS_STEPS):
            if self.path_index >= len(self.path):
                return  # No path or finished

            # on_ghost_path = False
            # if self.ghost_path_index < len(self.ghost_path):
            #     next_tile = self.ghost_path[self.ghost_path_index]
            #     on_ghost_path = True
            #else:
            next_tile = self.path[self.path_index]
            target_pos = grid_to_world(next_tile, self.grid)

            next_tile_on_grid = self.grid.tiles[next_tile.x][next_tile.y]
            if hasattr(next_tile_on_grid, 'is_occupied'):
                if next_tile_on_grid.is_occupied and next_tile_on_grid.is_occupied != self.id:
                    return

            dx = target_pos['x'] - self.x
            dy = target_pos['y'] - self.y

            magnitude = math.sqrt(dx ** 2 + dy ** 2)
            if magnitude > 0:
                self.x += dx / magnitude * self.speed * d_t
                self.y += dy / magnitude * self.speed * d_t

            # Snap to tile and advance index
            if close(self.x, target_pos['x']) and close(self.y, target_pos['y']):
                # if self.path_index == len(self.path) - 1:  # Check if it's the last tile
                #     self.x = target_pos['x']
                #     self.y = target_pos['y']
                #if not on_ghost_path:
                self.path_index += 1
                # else:
                #     self.ghost_path_index += 1
                # break

    def sync(self):
        for drawable in self.drawables:
            drawable.left = self.x - int(self.grid.tile_size // 2)
            drawable.top = self.y - int(self.grid.tile_size // 2)

    def set_intents(self, intents: List[Intent]):
        self.intents = intents

    def update(self, actions: dict, delta_time: float):

        if len(self.intents):
            current = self.intents[0]
            current.update(self, delta_time)
            if current.finished(self):
                self.intents.pop(0)
        self.sync()

    def serialize(self) -> dict:
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
        }


def close(a, b, tolerance=TILE_SNAP_THRESHOLD):
    if a is None or b is None:
        return True
    return abs(a - b) < tolerance

# time_total
# speed = (px/s)

# t/100 * self.speed =

# 1s = 1000 / 100 * 30  10 *30 = 30px
