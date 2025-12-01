from engine.extensions.topDownGridWorld.ai_controller._base_controller import Controller
from engine.extensions.topDownGridWorld.grid import Grid
import random


class RandomTileClicker(Controller):
    def choose_action(self, observation: dict):

        if observation is None or observation is {}:
            return None

        options = []

        for key, value in observation.items():
            if isinstance(value, Grid):
                for tile in value.tiles_flattened:
                    if hasattr(tile, 'clickable') and tile.clickable is not None:
                        options.append(tile)

        if options:
            tile = random.choice(options)

            return {
                "type": "click",
                "target": tile.id
            }
        return None
