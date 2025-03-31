# File: extensions/gridworld/grid.py
from engine.game_object import GameObject
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

    def add_tile(self, tile, slot_x, slot_y):
        if slot_x < self.width and slot_y < self.height:
            tile.x = slot_x * self.tile_size
            tile.y = slot_y * self.tile_size
            tile.w = tile.h = self.tile_size
            self.tiles[slot_x][slot_y] = tile
            return True
        print('Invalid slot coordinates, tile not added')
        return False

    @property
    def children(self):
        return [tile for row in self.tiles for tile in row]

    def update(self, actions, delta_time):
        for child in self.children:
            child.update(actions, delta_time)

    def serialize(self):
        return {'id': self.id} # Grid itself is abstract


class Tile(Clickable2D):
    def __init__(self, id, src, on_click=None):
        super().__init__(id, x=0, y=0, w=0, h=0, src=src, on_click=on_click)

    def serialize(self):
        data = super().serialize()
        return data
