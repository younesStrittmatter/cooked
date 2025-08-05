from engine.game_object import GameObject


class Text(GameObject):
    def update(self, actions: dict, delta_time: float):
        pass

    def __init__(self, id=None,
                 left=None,
                 right=None,
                 top=None,
                 bottom=None,
                 content='',
                 color='#000000', font_size=16,

                 z_index=0):
        super().__init__(id=id)
        if left is None and right is None:
            raise ValueError("Either left or right must be specified")
        if left is not None and right is not None:
            raise ValueError("Only one of left or right can be specified")
        if top is None and bottom is None:
            raise ValueError("Either top or bottom must be specified")
        if top is not None and bottom is not None:
            raise ValueError("Only one of top or bottom can be specified")
        if bottom is None:
            self.baseline = 'top'
        if top is None:
            self.baseline = 'bottom'
        if left is None:
            self.align = 'right'
        if right is None:
            self.align = 'left'
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.content = content
        self.color = color
        self.font_size = font_size
        self.z_index = z_index

    def serialize(self) -> dict:
        return {
            "id": self.id,
            "class": self.__class__.__name__,
            "left": self.left,
            "right": self.right,
            "align": self.align,
            "top": self.top,
            "bottom": self.bottom,
            "baseline": self.baseline,
            "content": self.content,
            "color": self.color,
            "fontSize": self.font_size,
            "zIndex": self.z_index
        }

    @classmethod
    def deserialize(cls, data: dict, game=None):
        obj = cls(
            id=data.get("id"),
            left=data.get("left", 0),
            right=data.get("right"),
            top=data.get("top", 0),
            bottom=data.get("bottom"),
            content=data.get("content", ''),
            color=data.get("color", '#000000'),
            font_size=data.get("fontSize", 16),
            z_index=data.get("zIndex", 0)
        )
        return obj
