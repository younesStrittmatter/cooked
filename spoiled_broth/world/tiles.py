from abc import abstractmethod
import math

from engine.extensions.topDownGridWorld.grid import Tile
from engine.extensions.renderer2d.basic_2d import Basic2D
from engine.extensions.topDownGridWorld.intent.intents import MoveToIntent
from spoiled_broth.agent.intents import PickUpIntent, ItemExchangeIntent, CuttingBoardIntent, DeliveryIntent
import random
import numpy as np

from engine.extensions.topDownGridWorld.a_star import Node, get_neighbors
from template_game.game import Game

TYPE_TILES = 6

ITEM_LIST = [
    "tomato", "pumpkin", "cabbage", "plate",
    "tomato_cut", "pumpkin_cut", "cabbage_cut",
    "tomato_salad", "pumpkin_salad", "cabbage_salad",
    "unknown", None  # fallback types
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

    def to_vector(self):
        return np.array(
            self._position_vec() +
            self._clickable_vec() +
            self._type_vec() +
            self._walkable_vec() +
            self._item_vec() +
            self._progress_vec(),
            dtype=np.float32
        )

    @abstractmethod
    def _type_name(self):
        ...

    @abstractmethod
    def _on_click_description(self, agent=None):
        ...

    @property
    def _item(self):
        if hasattr(self, 'item'):
            return self.item
        return None

    def get_distance(self, agent):
        target_node = Node(self.slot_x, self.slot_y)
        neighbors = get_neighbors(agent.grid, target_node, include_diagonal=False)
        if len(neighbors) == 0:
            return 99
        else:
            d = 9999
            target = None
            for neighbor in neighbors:
                _d = agent.get_distance(neighbor)
                if _d is not None and _d < d:
                    d = _d
                    target = neighbor

            if target is None:
                return 99
            return d

    def to_prompt(self, agent=None):
        _p = f"""\
Tile ({self._type_name()}):
    coordinates: ({self.slot_x}, {self.slot_y})
    type: {self._type_name()}
    distance: {round(self.get_distance(agent), 1)}"""

        if hasattr(self, 'item'):
            _p += f"\n    contains: {self._item}"

        if self.clickable is not None and self._on_click_description:
            _p += f"\n    on_choose: {self._on_click_description(agent=agent)}\n"

        return _p

    def serialize(self):
        return {
            'class': self.__class__.__name__,
            'id': self.id,
            'slot_x': self.slot_x,
            'slot_y': self.slot_y,
            'item': self._item,
            'clickable': self.clickable is not None
        }

    @classmethod
    def deserialize(cls, data, game=None):
        obj = cls(game=game)
        obj.id = data['id']
        obj.slot_x = data['slot_x']
        obj.slot_y = data['slot_y']
        obj.item = data.get('item', None)
        return obj


class Floor(SoiledBrothTile):
    _type = 0

    def __init__(self, game, slot_id=None):
        if slot_id is None:
            slot_id = random.randint(0, TYPE_TILES)
        _id = f'Floor_{slot_id}'
        super().__init__(game=game, id=_id)
        self.is_walkable = True
        self.clickable = None
        src_y = random.randint(1, 28) * 16
        self.add_drawable(
            Basic2D(id=f'{_id}_drawable', src='world/basic-floor.png', z_index=0, src_y=src_y, normalize=False))

    def _type_name(self):
        return "floor"

    def _on_click_description(self, agent=None):
        return "[None]"


class Wall(SoiledBrothTile):
    _type = 1

    def __init__(self, game, slot_id=None):
        if slot_id is None:
            slot_id = random.randint(0, TYPE_TILES)
        _id = f'Wall_{slot_id}'
        super().__init__(game=game, id=_id)
        self.is_walkable = False
        self.clickable = None
        self.add_drawable(Basic2D(id=f'{_id}_drawable', src='world/basic-wall.png', z_index=0, normalize=False))

    def _type_name(self):
        return "wall"

    def _on_click_description(self, agent=None):
        return "[None]"


class Counter(SoiledBrothTile):
    _type = 2

    def __init__(self, game, slot_id=None):
        if slot_id is None:
            slot_id = random.randint(0, TYPE_TILES)
        _id = f'Counter_{slot_id}'
        super().__init__(game=game, id=_id)
        self.is_walkable = False
        self.item = None
        self.add_drawable(Basic2D(id=f'{_id}_drawable_base', src='world/basic-counter.png', z_index=0, normalize=False))
        self.add_drawable(
            Basic2D(id=f'{_id}_drawable_item', src='world/item-on-counter.png', z_index=1, normalize=False))
        self.drawables[1].width = 0
        self.drawables[1].height = 0

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

    def get_intent(self, agent):
        return [MoveToIntent(self), ItemExchangeIntent(self)]

    def _type_name(self):
        return "counter"

    def _on_click_description(self, agent=None):
        if agent is None:
            return "If you are holding an item you will place it here. If the counter has an item you will pick it up. In case you hold a cut tomato and the counter has a plate you will place the tomato on the plate and assemble a salad."
        if agent is not None:
            item_held = agent.item
            if item_held is None:
                if self.item is None:
                    return "[None]"
                if self.item is not None:
                    return f"Pick up a {self.item}"
            else:
                if self.item is None:
                    return f"Place a {item_held}"
                if self.item is not None:
                    if self.item == 'plate' and item_held in ['tomato_cut', 'pumpkin_cut', 'cabbage_cut']:
                        return f"Assemble a {item_held}_salad"
                    elif self.item == item_held:
                        return "[None]"
                    else:
                        return f"Place a {item_held} and pick up a {self.item}"


class Dispenser(SoiledBrothTile):
    _type = 3

    def __init__(self, game, item, slot_id=None):
        if slot_id is None:
            slot_id = random.randint(0, TYPE_TILES)
        _id = f'Dispenser_{slot_id}'
        super().__init__(game=game, id=_id)
        self.is_walkable = False
        self.add_drawable(Basic2D(id=f'{_id}_drawable_base', src='world/basic-counter.png', z_index=0, normalize=False))
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
        self.add_drawable(
            Basic2D(id=f'{_id}_drawable_item', src='world/item-dispenser.png', z_index=0, src_y=src_y, normalize=False))

    def get_intent(self, agent):
        return [MoveToIntent(self), PickUpIntent(self)]

    def _type_name(self):
        return "dispenser"

    def _on_click_description(self, agent):
        if agent is not None and agent.item is not None:
            if self.item.endswith('salad'):
                return f"Do not choose this tile, you will destroy the {agent.item}"
            if self.item.endswith('cut'):
                return f"Do not choose this tile, you will destroy the {agent.item}"
            return f"Pick a {self.item}"
        return f"Pick up a {self.item}"

    @classmethod
    def deserialize(cls, data, game=None):
        obj = cls(game=game, item=data.get('item'))
        obj.id = data['id']
        obj.slot_x = data['slot_x']
        obj.slot_y = data['slot_y']
        obj.item = data.get('item', None)
        return obj


class CuttingBoard(SoiledBrothTile):
    _type = 4

    def __init__(self, game, slot_id=None):
        if slot_id is None:
            slot_id = random.randint(0, TYPE_TILES)
        _id = f'CuttingBoard_{slot_id}'
        super().__init__(game=game, id=_id)
        self.is_walkable = False
        self.add_drawable(Basic2D(id=f'{_id}_drawable_base', src='world/basic-counter.png', z_index=0, normalize=False))
        self.add_drawable(
            Basic2D(id=f'{_id}_drawable_board', src='world/cutting-board.png', z_index=1, normalize=False))
        self.add_drawable(Basic2D(id=f'{_id}_drawable_item', src='world/item-on-board.png', z_index=2, normalize=False))
        self.add_drawable(
            Basic2D(id=f'{_id}_drawable_knife', src='world/knife.png', z_index=3, src_x=0, src_w=16, src_h=16,
                    normalize=False))
        self.drawables[2].width = 0
        self.drawables[2].height = 0
        self.item = None
        self.cut_time_accumulated = 0

    @property
    def cut_stage(self):
        if 0 <= self.cut_time_accumulated < 1:
            return 0
        if 1 <= self.cut_time_accumulated < 2:
            return 1
        elif 2 <= self.cut_time_accumulated < 3:
            return 2
        else:
            return 3

    @property
    def _item(self):
        if self.cut_stage >= 3:
            return f'{self.item}_cut'
        return self.item

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
        self.drawables[3].src_x = 0
        # Configuration
        frame_duration = 0.07
        frames = [0, 16, 32]
        total_duration = len(frames) * frame_duration

        # Calculate current frame index
        time_in_cycle = self.cut_time_accumulated % total_duration
        frame_index = int(time_in_cycle // frame_duration)
        self.drawables[3].src_x = frames[frame_index]
        if self.cut_stage >= 3 or self.cut_time_accumulated <= frame_duration:
            self.drawables[3].src_x = 0

    def serialize(self):
        return {
            'class': self.__class__.__name__,
            'id': self.id,
            'slot_x': self.slot_x,
            'slot_y': self.slot_y,
            'item': self._item,
            'clickable': self.clickable is not None,
            'cut_time_accumulated': self.cut_time_accumulated,
            'cut_stage': self.cut_stage
        }



    def get_intent(self, agent):
        return [MoveToIntent(self), CuttingBoardIntent(self)]

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

    def _type_name(self):
        return "cutting board"

    def _on_click_description(self, agent):
        if agent is None or agent.item is None:
            if self.item is None:
                return '[None]'
            else:
                if self.cut_time_accumulated < 3:
                    return f"Cut the {self.item}."
                else:
                    return f"Pick up a {self.item}_cut."
        elif agent.item in ['tomato', 'pumpkin', 'cabbage']:
            if self.item is None:
                return f"Cut the {agent.item}."
            elif self.cut_stage == 3:
                return f"Place a {agent.item} and pick up a {self._item}_cut."
        return "[None]"


class Delivery(SoiledBrothTile):
    _type = 5

    def __init__(self, game, slot_id=None):
        if slot_id is None:
            slot_id = random.randint(0, TYPE_TILES)
        _id = f'Delivery_{slot_id}'
        super().__init__(game=game, id=_id)
        self.is_walkable = False
        self.add_drawable(Basic2D(id=f'{_id}_drawable', src='world/delivery.png', z_index=0, normalize=False))

    def update(self, agent, delta_time):
        super().update(agent, delta_time)

    def get_intent(self, agent):
        return [MoveToIntent(self), DeliveryIntent(self)]

    def _type_name(self):
        return "delivery"

    def _on_click_description(self, agent):
        if agent is not None and agent.item in ['tomato_salad', 'pumpkin_salad', 'cabbage_salad']:
            return f"Deliver the {agent.item}"
        return f"[None]"


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
