import numpy as np

from engine.extensions.renderer2d.basic_2d import Basic2D
from engine.extensions.topDownGridWorld import agent
import random
from spoiled_broth.world.tiles import ITEM_LIST


class Agent(agent.Agent):
    def __init__(self, agent_id, grid, game):
        super().__init__(agent_id, grid)
        self.game = game
        self.path = []
        self.path_index = 0
        self.move_target = None

        self.item = None
        self.cut_speed = 1
        self.action = None
        self.score = 0
        hair_n = random.randint(0, 8)
        mustache_n = random.randint(0, 8)
        skin_n = random.randint(0, 8)

        husk_2d = Basic2D(src='agent/cook-husk.png', z_index=1, src_y=0, src_w=16, src_h=16, width=16, height=16,
                          normalize=False)
        hair_2d = Basic2D(src=f'agent/hair/{hair_n}.png', z_index=2, src_w=16, src_h=16, width=16, height=16,
                          normalize=False)
        mustache_2d = Basic2D(src=f'agent/mustache/{mustache_n}.png', z_index=2, src_w=16, src_h=16, width=16,
                              height=16, normalize=False)
        head_2d = Basic2D(src=f'agent/skin/{skin_n}.png', z_index=2, src_y=0, src_w=16, src_h=16, width=16, height=16,
                          normalize=False)
        hands_no_item_2d = Basic2D(src=f'agent/skin/{skin_n}.png', z_index=2, src_y=16, src_w=16, src_h=16, width=16,
                                   height=16, normalize=False)
        hands_with_item_2d = Basic2D(src=f'agent/skin/{skin_n}.png', z_index=2, src_y=32, src_w=16, src_h=16, width=0,
                                     height=0, normalize=False)
        item_held_2d = Basic2D(src='agent/items-held.png', z_index=3, src_y=0, src_w=16, src_h=16, width=0, height=0,
                               normalize=False)

        self.src_x = 0
        self.src_x_n = 0
        self.animation_time = 0

        self.add_drawable(husk_2d)
        self.add_drawable(hair_2d)
        self.add_drawable(mustache_2d)
        self.add_drawable(head_2d)
        self.add_drawable(hands_no_item_2d)
        self.add_drawable(hands_with_item_2d)
        self.add_drawable(item_held_2d)

    @property
    def is_moving(self):
        return self.path_index < len(self.path)

    def update(self, actions: dict, delta_time: float):
        super().update(actions, delta_time)
        if not self.is_moving:
            self.animation_time = 0
            self.x = self.slot_x * self.grid.tile_size + self.grid.tile_size // 2
            self.y = self.slot_y * self.grid.tile_size + self.grid.tile_size // 2
            self.src_x_n = 0
        else:
            self.animation_time += delta_time
            if self.animation_time > 0.1:
                self.src_x_n += 1
                self.src_x_n %= 5
                self.animation_time = 0
        self.src_x = self.src_x_n * 16

        for drawable in self.drawables:
            drawable.src_x = self.src_x

        if self.item is not None:
            self.drawables[4].width = 0
            self.drawables[4].height = 0
            self.drawables[5].width = 16
            self.drawables[5].height = 16
            self.drawables[6].width = 16
            self.drawables[6].height = 16
            if self.item == 'tomato':
                self.drawables[6].src_y = 0
            #if self.item == 'pumpkin':
            #    self.drawables[6].src_y = 16
            #if self.item == 'cabbage':
            #    self.drawables[6].src_y = 32
            if self.item == 'plate':
                self.drawables[6].src_y = 48
            if self.item == 'tomato_cut':
                self.drawables[6].src_y = 9 * 16
            #if self.item == 'pumpkin_cut':
            #    self.drawables[6].src_y = 10 * 16
            #if self.item == 'cabbage_cut':
            #    self.drawables[6].src_y = 11 * 16
            if self.item == 'tomato_salad':
                self.drawables[6].src_y = 15 * 16
            #if self.item == 'pumpkin_salad':
            #    self.drawables[6].src_y = 16 * 16
            #if self.item == 'cabbage_salad':
            #    self.drawables[6].src_y = 17 * 16
        else:
            self.drawables[4].width = 16
            self.drawables[4].height = 16
            self.drawables[5].width = 0
            self.drawables[5].height = 0
            self.drawables[6].width = 0
            self.drawables[6].height = 0

    #def to_vector(self):
    #    _pos = [self.slot_x / self.grid.width, self.slot_y / self.grid.height]
    #    _item = [0] * len(ITEM_LIST)
    #    if self.item in ITEM_LIST:
    #        _item[ITEM_LIST.index(self.item)] = 1
    #    return np.array(_pos + _item, dtype=np.float32)
