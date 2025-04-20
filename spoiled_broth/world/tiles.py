from engine.extensions.topDownGridWorld.grid import Tile
from engine.extensions.renderer2d.basic_2d import Basic2D
from engine.extensions.topDownGridWorld.intent.intents import MoveToIntent
from spoiled_broth.agent.intents import PickUpIntent, ItemExchangeIntent, CuttingBoardIntent, DeliveryIntent
import random


class Floor(Tile):
    def __init__(self, game):
        super().__init__(game=game)
        self.is_walkable = True
        self.clickable = None
        src_y = random.randint(1, 28) * 16
        self.add_drawable(Basic2D(src='world/basic-floor.png', z_index=0, src_y=src_y))


class Wall(Tile):
    def __init__(self, game):
        super().__init__(game=game)
        self.is_walkable = False
        self.clickable = None
        self.add_drawable(Basic2D(src='world/basic-wall.png', z_index=0))


class Counter(Tile):
    def __init__(self, game):
        super().__init__(game=game)
        self.is_walkable = False
        self.item = None
        self.add_drawable(Basic2D(src='world/basic-counter.png', z_index=0))
        self.add_drawable(Basic2D(src='world/item-on-counter.png', z_index=1))
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


class Dispenser(Tile):
    def __init__(self, game, item):
        super().__init__(game=game)
        self.is_walkable = False
        self.add_drawable(Basic2D(src='world/basic-counter.png', z_index=0))
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
        self.add_drawable(Basic2D(src='world/item-dispenser.png', z_index=0, src_y=src_y))

    def get_intent(self, agent):
        return [MoveToIntent(self), PickUpIntent(self)]


class CuttingBoard(Tile):
    def __init__(self, game):
        super().__init__(game=game)
        self.is_walkable = False
        self.add_drawable(Basic2D(src='world/basic-counter.png', z_index=0))
        self.add_drawable(Basic2D(src='world/cutting-board.png', z_index=1))
        self.add_drawable(Basic2D(src='world/item-on-board.png', z_index=2))
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

    def update(self, agent, delta_time):
        super().update(agent, delta_time)

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

    def get_intent(self, agent):
        return [MoveToIntent(self), CuttingBoardIntent(self)]


class Delivery(Tile):
    def __init__(self, game):
        super().__init__(game=game)
        self.is_walkable = False
        self.add_drawable(Basic2D(src='world/delivery.png', z_index=0))



    def update(self, agent, delta_time):
        super().update(agent, delta_time)


    def get_intent(self, agent):
        return [MoveToIntent(self), DeliveryIntent(self)]


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
