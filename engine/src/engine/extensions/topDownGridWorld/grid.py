from engine.extensions.gridworld import grid
from PIL import Image


class Grid(grid.Grid):
    def init_from_img(self, path, color_map, game):
        # load the image and check the size
        img = Image.open(path)
        img = img.convert("RGB")
        img = img.resize((self.width, self.height))
        if img.size != (self.width, self.height):
            raise ValueError("Image size does not match grid size")
        for x in range(self.width):
            for y in range(self.height):
                color = img.getpixel((x, y))
                tile = color_map[str(color)]['class'](game=game, slot_id=f'{x}{y}', **color_map[str(color)]['kwargs'])
                tile.add_to_grid(self, x, y, self.tile_size)

    def serialize(self) -> dict:
        return {
            'id': self.id,
            'width': self.width,
            'height': self.height,
            'tile_size': self.tile_size,
            'tiles': [[tile.full_serialize() if tile else None for tile in row] for row in self.tiles]
        }

    @classmethod
    def deserialize(cls, data: dict, game=None):
        grid_instance = cls(data['id'], data['width'], data['height'], data['tile_size'])
        for x in range(data['width']):
            for y in range(data['height']):
                tile_data = data['tiles'][x][y]
                if tile_data:
                    tile = grid_instance.tiles[x][y] = grid.Tile.full_deserialize(tile_data, game=game)
                    tile.slot_x = x
                    tile.slot_y = y
        return grid_instance



class Tile(grid.Tile):
    def __init__(self, id=None,
                 is_walkable=False,
                 game=None,
                 tile_size=16):
        if game is None:
            raise ValueError("Game cannot be None")
        super().__init__(id, tiles_size=tile_size)
        self.game = game
        self.is_walkable = is_walkable

    def click(self, agent_id: str):
        if self.clickable and self.clickable.on_click:
            self.clickable.on_click(agent_id)

    def add_to_grid(self, grid=None, slot_x=0, slot_y=0, tile_size=16):
        # only define on_click if get_intent is not the default
        _on_click = None

        # Replace Tile with the actual base class name where get_intent is defined
        if self.__class__.get_intent is not Tile.get_intent:
            def _on_click(agent_id):
                agent = self.game.gameObjects.get(agent_id)
                if agent:
                    intent = self.get_intent(agent)
                    if intent:
                        agent.set_intents(intent)

        super().add_to_grid(grid, slot_x, slot_y, tile_size, _on_click)

    def get_intent(self, agent):
        return None

    def serialize(self) -> dict:
        return {
            "id": self.id,
            "class": self.__class__.__name__,
            "slot_x": self.slot_x,
            "slot_y": self.slot_y,
            "tile_size": self.tile_size,
            "is_walkable": self.is_walkable,
        }

    @classmethod
    def deserialize(cls, data: dict, game=None):
        tile = cls(
            id=data.get("id"),
            is_walkable=data.get("is_walkable", False),
            game=game,
            tile_size=data.get("tile_size", 16)
        )
        tile.slot_x = data.get("slot_x", 0)
        tile.slot_y = data.get("slot_y", 0)
        return tile

