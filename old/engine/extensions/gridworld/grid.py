# File: extensions/gridworld/grid.py
from engine.game_object import GameObject
from engine.extensions.renderer2d.basic_2d import Basic2D
from engine.extensions.renderer2d.clickable_2d import Clickable2D


class Grid(GameObject):
    def __init__(self,
                 id,
                 width,
                 height,
                 tile_size):
        super().__init__(id)
        self.width = width
        self.height = height
        self.tile_size = tile_size

        self.tiles = [
            [None for _ in range(height)]
            for _ in range(width)
        ]

    def add_tile(self, tile):
        if tile.slot_x < self.width and tile.slot_y < self.height:
            self.tiles[tile.slot_x][tile.slot_y] = tile
            return True
        print('Invalid slot coordinates, tile not added')
        return False

    @property
    def children(self):
        return [tile for row in self.tiles for tile in row]

    @property
    def tiles_flattened(self):
        return [tile for row in self.tiles for tile in row if tile]

    def update(self, actions, delta_time):
        for child in self.children:
            child.update(actions, delta_time)

    def serialize(self):
        return {'id': self.id}  # Grid itself is abstract


class Tile(GameObject):
    """
    A tile is a clickable object that can be placed on the grid.
    """
    slot_x = 0
    slot_y = 0

    left = 0
    top = 0
    tile_size = 16

    clickable = None

    def __init__(self, id=None,
                 left=0,
                 top=0,
                 tiles_size=16,
                 on_click=None):
        super().__init__(id)
        self.drawables = []

        self._tiles_size = tiles_size

        if on_click:
            self.clickable = Clickable2D(None, left, top, tiles_size, tiles_size, on_click)

    def update(self, actions, delta_time):
        # Process click actions that target this tile.
        # Controllers submit intents as a dict: agent_id -> action.
        # If an action is a click and its target matches this tile's id,
        # invoke the tile click handler so the agent receives the intents
        # (e.g., MoveToIntent + PickUpIntent).
        if actions:
            for agent_id, action in list(actions.items()):
                try:
                    if isinstance(action, dict) and action.get('type') == 'click' and action.get('target') == self.id:
                        # call the tile click which invokes the on_click handler
                        try:
                            self.click(agent_id)
                        except Exception:
                            # non-fatal: continue processing other actions
                            pass
                except Exception:
                    # ignore malformed actions
                    pass

    def add_drawable(self, drawable: Basic2D):
        self.drawables.append(drawable)
        drawable.left = self.left
        drawable.top = self.top

    def add_to_grid(self,
                    grid,
                    slot_x,
                    slot_y,
                    tile_size=16,
                    on_click=None):

        self.slot_x = slot_x
        self.slot_y = slot_y

        _tile_size = tile_size or self.tile_size

        self.left = slot_x * _tile_size
        self.top = slot_y * _tile_size

        if on_click:
            if not self.clickable:
                self.clickable = Clickable2D(self.id,
                                             slot_x * _tile_size,
                                             slot_y * _tile_size,
                                             _tile_size,
                                             _tile_size, on_click)
        self.sync()
        grid.add_tile(self)

    def sync(self):
        if self.clickable:
            self.clickable.left = self.left
            self.clickable.top = self.top
            self.clickable.width = self.tile_size
            self.clickable.height = self.tile_size
        for drawable in self.drawables:
            drawable.left = self.left
            drawable.top = self.top
            drawable.width = self.tile_size
            drawable.height = self.tile_size
            drawable.src_w = self.tile_size
            drawable.src_h = self.tile_size

    def serialize(self):
        return {"id": self.id}
