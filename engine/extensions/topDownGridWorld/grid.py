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
                tile = color_map[str(color)]['class'](game=game, **color_map[str(color)]['kwargs'])
                tile.add_to_grid(self, x, y, self.tile_size)


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
