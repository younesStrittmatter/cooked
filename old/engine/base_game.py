from abc import ABC, abstractmethod


class BaseGame(ABC):
    def __init__(self):
        self.done = False
        self.gameObjects = {}

    def step(self, actions: dict, delta_time: float):
        for obj in list(self.gameObjects.values()):
            self._update_recursive(obj, actions, delta_time)

    def _update_recursive(self, obj, actions: dict, delta_time: float):
        if hasattr(obj, 'update'):
            obj.update(actions, delta_time)
        for child in getattr(obj, 'children', []):
            if child is not None:
                self._update_recursive(child, actions, delta_time)

    def get_observations(self) -> dict:
        return self.gameObjects
        # return self.gameObjects
