from collections.abc import Callable
from engine.game_object import GameObject

class Clickable2D(GameObject):
    is_clickable = True

    def __init__(self, id=None, left=0, top=0, width=1, height=1, on_click: Callable=None) -> None:
        super().__init__(id=id)
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.on_click = on_click

    def update(self, actions: dict, delta_time: float):
        for agent_id, action in actions.items():
            if action.get("type") == "click" and action.get("target") == self.id:
                if self.on_click:
                    self.on_click(agent_id)

    def serialize(self) -> dict:
        data = super().serialize()
        data["left"] = self.left
        data["top"] = self.top
        data["width"] = self.width
        data["height"] = self.height
        data["isClickable"] = self.is_clickable
        data['id'] = self.id
        return data
