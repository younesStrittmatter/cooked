from collections.abc import Callable
from engine.game_object import GameObject


class Mouse(GameObject):

    def __init__(self, id=None, agent_id=None) -> None:
        super().__init__(id=id)
        self.x = 0
        self.y = 0
        self.leftDown = False
        self.rightDown = False
        self.agent_id = agent_id

    def update(self, actions: dict, delta_time: float):
        if self.agent_id in actions:
            action = actions[self.agent_id]
            if action.get("type") == "mouse":
                self.x = action.get("x")
                self.y = action.get("y")
                self.leftDown = action.get("leftDown")
                self.rightDown = action.get("rightDown")

    def serialize(self) -> dict:
        data = super().serialize()
        data["x"] = self.x
        data["y"] = self.y
        data["leftDown"] = self.leftDown
        data["rightDown"] = self.rightDown
        data['id'] = self.id
        return data
