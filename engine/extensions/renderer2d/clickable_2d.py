from collections.abc import Callable
import secrets

from jinja2.utils import consume

from engine.extensions.renderer2d.basic_2d import Basic2D


class Clickable2D(Basic2D):
    is_clickable = True

    def __init__(self, id, x, y, w, h, src,
                 src_x=0, src_y=0, src_w=None, src_h=None,
                 on_click=Callable):
        super().__init__(id=id, x=x, y=y, w=w, h=h, src=src,
                         src_x=src_x, src_y=src_y, src_w=src_w,
                         src_h=src_h)
        if id is None:
            self.id = secrets.token_hex(8)
        self.on_click = on_click

    def update(self, actions: dict, delta_time: float):
        for agent_id, action in actions.items():
            if action.get("type") == "click" and action.get("target") == self.id:
                if self.on_click:
                    self.on_click(agent_id)

    def serialize(self) -> dict:
        data = super().serialize()
        data["isClickable"] = self.is_clickable
        data['id'] = self.id
        return data
