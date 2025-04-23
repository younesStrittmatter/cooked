from engine.game_object import GameObject
from engine.extensions.renderer2d.basic_2d import Basic2D


class Box(GameObject):

    def __init__(self, x=.4, y=.4, width=.1, height=.1):
        super().__init__('box')
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self.drawable = Basic2D(src=f'box.png', width=self.width, height=self.height)

    @property
    def children(self):
        return [self.drawable]

    def update(self, actions: dict, delta_time: float):
        self.sync()

    def sync(self):
        self.drawable.left = self.x - self.width / 2
        self.drawable.top = self.y - self.height / 2

    def serialize(self) -> dict:
        return super().serialize()


class Target(GameObject):
    def __init__(self, x=.5, y=.5, width=.1, height=.1):
        super().__init__('target')
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self.drawable = Basic2D(src=f'target.png', width=self.width, height=self.height)

    @property
    def children(self):
        return [self.drawable]

    def sync(self):
        self.drawable.left = self.x - self.width / 2
        self.drawable.top = self.y - self.height / 2

    def update(self, actions: dict, delta_time: float):
        self.sync()

    def serialize(self) -> dict:
        return super().serialize()
