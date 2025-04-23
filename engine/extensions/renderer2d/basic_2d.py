from engine.game_object import GameObject


class Basic2D(GameObject):
    def __init__(self, id=None,
                 left=0, top=0, width=0, height=0, src=None,
                 src_x=0, src_y=0, src_w=None, src_h=None,
                 normalize=True,
                 z_index=0):
        super().__init__(id=id)
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.src = src
        self.src_x = src_x
        self.src_y = src_y
        self.src_w = src_w
        self.src_h = src_h
        self.normalize = normalize
        self.z_index = z_index

    def update(self, actions: dict, delta_time: float):
        pass

    def serialize(self) -> dict:
        return {
            "id": self.id,
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
            "src": self.src,
            "srcX": self.src_x,
            "srcY": self.src_y,
            "srcW": self.src_w,
            "srcH": self.src_h,
            "normalize": self.normalize,
            "zIndex": self.z_index
        }
