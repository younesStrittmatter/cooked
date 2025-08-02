from abc import abstractmethod

from engine.extensions.topDownGridWorld.grid import Tile
from engine.extensions.renderer2d.basic_2d import Basic2D
from engine.extensions.topDownGridWorld.intent.intents import MoveToIntent
from spoiled_broth.agent.intents import PickUpIntent, ItemExchangeIntent, CuttingBoardIntent, DeliveryIntent
import random
import numpy as np

TYPE_TILES = 6

ITEM_LIST = [
    "tomato", "plate", "tomato_cut", "tomato_salad"
]

class SoiledBrothTile(Tile):
    _type = 0

    def _position_vec(self):
        return [self.slot_x / self.game.grid.width, self.slot_y / self.game.grid.height]

    def _clickable_vec(self):
        return [0] if self.clickable is None else [1]

    def _type_vec(self):
        vec = [0] * TYPE_TILES
        if hasattr(self, '_type'):
            vec[self._type] = 1
        return vec

    def _walkable_vec(self):
        return [1] if self.is_walkable else [0]

    def _item_vec(self):
        vec = [0] * len(ITEM_LIST)
        if hasattr(self, 'item') and self.item in ITEM_LIST:
            vec[ITEM_LIST.index(self.item)] = 1
        return vec

    def _progress_vec(self):
        return [0] * 3
    
    #def _get_normalized_progress(self):
    #    """Improved progress vector with proper normalization"""
    #    if hasattr(self, 'progress'):
    #        if hasattr(self, 'max_progress'):
    #            # Normalize current progress
    #            normalized = min(1.0, self.progress / self.max_progress)
    #            return [normalized]
    #        elif hasattr(self, 'progress_steps'):
    #            # One-hot for discrete progress steps
    #            vec = [0] * len(self.progress_steps)
    #            current_step = min(len(self.progress_steps)-1, 
    #                              bisect.bisect_left(self.progress_steps, self.progress))
    #            vec[current_step] = 1
    #            return vec
    #    # Default no progress
    #    return [0] * 3  # Maintains same length as original

    ## LEGACY FROM game_to_vector in game.py
    #def to_vector(self):
    #    # Only encode type (as int), item (one-hot for 4), and progress (if relevant)
    #    # All tiles are clickable, so skip clickable_vec
    #    # Omit walkable_vec and position_vec for simplicity
    #    type_val = getattr(self, '_type', 0)
    #    item_vec = self._item_vec()
    #    # For cutting board, include progress; else, 0
    #    progress = self._progress_vec() if hasattr(self, '_progress_vec') else [0]
    #    return np.array([type_val] + item_vec + progress, dtype=np.float32)

    def to_language(self, agent):
        raise NotImplementedError()


class Floor(SoiledBrothTile):
    _type = 0

    def __init__(self, game):
        super().__init__(game=game)
        self.is_walkable = True
        self.clickable = None
        src_y = random.randint(1, 28) * 16
        self.add_drawable(Basic2D(src='world/basic-floor.png', z_index=0, src_y=src_y, normalize=False))


class Wall(SoiledBrothTile):
    _type = 1

    def __init__(self, game):
        super().__init__(game=game)
        self.is_walkable = False
        self.clickable = None
        self.add_drawable(Basic2D(src='world/basic-wall.png', z_index=0, normalize=False))


class Counter(SoiledBrothTile):
    _type = 2

    def __init__(self, game):
        super().__init__(game=game)
        self.is_walkable = False
        self.item = None
        self.add_drawable(Basic2D(src='world/basic-counter.png', z_index=0, normalize=False))
        self.add_drawable(Basic2D(src='world/item-on-counter.png', z_index=1, normalize=False))
        self.drawables[1].width = 0
        self.drawables[1].height = 0
        self.salad_by = None  # Register the agent
        self.intent_version = None

    def update(self, agent, delta_time):
        super().update(agent, delta_time)
        if self.item is not None:
            self.drawables[1].width = 16
            self.drawables[1].height = 16
            if self.item == 'tomato':
                self.drawables[1].src_y = 0
            if self.item == 'pumpkin':
                self.drawables[1].src_y = 16
            if self.item == 'cabbage':
                self.drawables[1].src_y = 32
            if self.item == 'plate':
                self.drawables[1].src_y = 48
            if self.item == 'tomato_cut':
                self.drawables[1].src_y = 9 * 16
            if self.item == 'pumpkin_cut':
                self.drawables[1].src_y = 10 * 16
            if self.item == 'cabbage_cut':
                self.drawables[1].src_y = 11 * 16
            if self.item == 'tomato_salad':
                self.drawables[1].src_y = 15 * 16
            if self.item == 'pumpkin_salad':
                self.drawables[1].src_y = 16 * 16
            if self.item == 'cabbage_salad':
                self.drawables[1].src_y = 17 * 16
        else:
            self.drawables[1].width = 0
            self.drawables[1].height = 0
        
        # The salad_by field is now set in ItemExchangeIntent when salad is created

    def get_intent(self, agent):
        self.intent_version = agent.intent_version
        return [MoveToIntent(self), ItemExchangeIntent(self, version=agent.intent_version)]


class Dispenser(SoiledBrothTile):
    _type = 3

    def __init__(self, game, item):
        super().__init__(game=game)
        self.is_walkable = False
        self.add_drawable(Basic2D(src='world/basic-counter.png', z_index=0, normalize=False))
        self.item = item
        src_y = 0
        if item == 'tomato':
            src_y = 0
        if item == 'pumpkin':
            src_y = 16
        if item == 'cabbage':
            src_y = 32
        if item == 'plate':
            src_y = 48
        self.add_drawable(Basic2D(src='world/item-dispenser.png', z_index=0, src_y=src_y, normalize=False))
        self.intent_version = None

    def get_intent(self, agent):
        self.intent_version = agent.intent_version
        return [MoveToIntent(self), PickUpIntent(self, version=self.intent_version)]


class CuttingBoard(SoiledBrothTile):
    _type = 4

    def __init__(self, game):
        super().__init__(game=game)
        self.is_walkable = False
        self.add_drawable(Basic2D(src='world/basic-counter.png', z_index=0, normalize=False))
        self.add_drawable(Basic2D(src='world/cutting-board.png', z_index=1, normalize=False))
        self.add_drawable(Basic2D(src='world/item-on-board.png', z_index=2, normalize=False))
        self.drawables[2].width = 0
        self.drawables[2].height = 0
        self.item = None
        self.cut_time_accumulated = 0
        self.cut_by = None  # New field to register the agent
        self.intent_version = None

    @property
    def cut_stage(self):
        if self.intent_version == "v2":
        # v2: cutting is instant, so always consider it done
            return 3
        elif self.intent_version == "v3":
            # In v3: cutting one time is enough
            if self.cut_time_accumulated >= 1:
                return 3 
            else:
                return 0
        else:
            if 0 <= self.cut_time_accumulated < 1:
                return 0
            if 1 <= self.cut_time_accumulated < 2:
                return 1
            elif 2 <= self.cut_time_accumulated < 3:
                return 2
            else:
                return 3

    def update(self, agent, delta_time):
        super().update(agent, delta_time)
        self.cut_time_accumulated = min(4, self.cut_time_accumulated)

        if self.item is not None:
            self.drawables[2].width = 16
            self.drawables[2].height = 16
            if self.item == 'tomato':
                self.drawables[2].src_y = 0
            if self.item == 'pumpkin':
                self.drawables[2].src_y = 16
            if self.item == 'cabbage':
                self.drawables[2].src_y = 32
            self.drawables[2].src_x = self.cut_stage * 16
        else:
            self.drawables[2].width = 0
            self.drawables[2].height = 0

        # The cut_by field is now set in CuttingBoardIntent when item is cut

    def get_intent(self, agent):
        self.intent_version = agent.intent_version
        return [MoveToIntent(self), CuttingBoardIntent(self, version=self.intent_version)]

    def _progress_vec(self):
        if self.item is None:
            return [0, 0, 0]
        stage = self.cut_stage  # 0–3
        vec = [0, 0, 0]
        if 1 <= stage < 2:
            vec[0] = 1  # early chop
        elif 2 <= stage < 3:
            vec[1] = 1  # mid chop
        elif stage >= 3:
            vec[2] = 1  # done
        return vec


class Delivery(SoiledBrothTile):
    _type = 5

    def __init__(self, game):
        super().__init__(game=game)
        self.is_walkable = False
        self.add_drawable(Basic2D(src='world/delivery.png', z_index=0, normalize=False))
        self.delivered_by = None  # Register which agent delivered
        self.intent_version = None

    def update(self, agent, delta_time):
        super().update(agent, delta_time)
        # The delivered_by field is now set in DeliveryIntent when delivery occurs

    def get_intent(self, agent):
        self.intent_version = agent.intent_version
        return [MoveToIntent(self), DeliveryIntent(self, version=self.intent_version)]


COLOR_MAP = {
    '(0, 0, 0)': {'class': Floor, 'kwargs': {}},
    '(50, 50, 50)': {'class': Wall, 'kwargs': {}},
    '(255, 0, 0)': {'class': Dispenser, 'kwargs': {'item': 'tomato'}},
    '(255, 255, 0)': {'class': Dispenser, 'kwargs': {'item': 'pumpkin'}},
    '(0, 255, 0)': {'class': Dispenser, 'kwargs': {'item': 'cabbage'}},
    '(100, 100, 100)': {'class': Dispenser, 'kwargs': {'item': 'plate'}},
    '(0, 255, 255)': {'class': CuttingBoard, 'kwargs': {}},
    '(255, 255, 255)': {'class': Counter, 'kwargs': {}},
    '(255, 0, 255)': {'class': Delivery, 'kwargs': {}},
}

CHAR_MAP = {
    'W': {'class': Wall, 'kwargs': {}},
    ' ': {'class': Floor, 'kwargs': {}},
    'T': {'class': Dispenser, 'kwargs': {'item': 'tomato'}},
    'P': {'class': Dispenser, 'kwargs': {'item': 'pumpkin'}},
    'C': {'class': Dispenser, 'kwargs': {'item': 'cabbage'}},
    'X': {'class': Dispenser, 'kwargs': {'item': 'plate'}},
    'B': {'class': CuttingBoard, 'kwargs': {}},
    'M': {'class': Counter, 'kwargs': {}},
    'D': {'class': Delivery, 'kwargs': {}},
}
