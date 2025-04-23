from engine.extensions.renderer2d.io import Mouse
from engine.extensions.renderer2d.basic_2d import Basic2D
from engine.game_object import GameObject


class Agent(GameObject):
    def __init__(self, id, kind='', x=.5, y=.5, width=.1, height=.1):
        super().__init__(id=id)
        self.x = x
        self.y = y
        self.target_x = 0
        self.target_y = 0
        self.mouse = None
        self.kind = kind
        self.width = width
        self.height = height

        self.drawable = Basic2D(src=f'{kind}.png', width=self.width, height=self.height)

    @property
    def children(self):
        return [self.drawable]

    def sync(self):
        self.drawable.left = self.x - self.width / 2
        self.drawable.top = self.y - self.height / 2

    def add_mouse(self, agent_mouse: Mouse):
        self.mouse = agent_mouse

    def move(self, delta_time):
        d_x = self.target_x - self.x
        d_y = self.target_y - self.y
        distance = (d_x ** 2 + d_y ** 2) ** 0.5
        if distance > 0.01:
            d_x /= distance
            d_y /= distance
            self.x += d_x * .3 * delta_time
            self.y += d_y * .3 * delta_time

    def update(self, actions: dict, delta_time: float):

        self.mouse.update(actions, delta_time)

        if self.mouse.leftDown:
            self.target_x = self.mouse.x
            self.target_y = self.mouse.y
        else:
            self.target_x = self.x
            self.target_y = self.y
        self.move(delta_time)
        self.sync()

    def serialize(self) -> dict:
        return {
            "id": self.id,
        }
