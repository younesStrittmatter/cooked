from abc import ABC, abstractmethod


class BaseGame(ABC):
    def __init__(self):
        self.done = False
        self.gameObjects = {}

    def step(self, actions: dict, delta_time: float):
        for obj in self.gameObjects.values():
            obj.update(actions, delta_time)

    def get_observations(self) -> dict:
        return self.gameObjects
