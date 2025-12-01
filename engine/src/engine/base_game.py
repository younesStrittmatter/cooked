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

    @abstractmethod
    def initial_args(self) -> dict:
        """
        Returns the initial state of the game.
        This can be used to reset the game or for training purposes.
        """
        return {}

    @abstractmethod
    def agent_initial_state(self, agent_id: str) -> dict:
        """
        Returns the initial state of a specific agent.
        This can be used to reset the agent's state or for training purposes.
        """
        return {}


    def serialize_initial_state(self) -> dict:
        return {
            "class": self.__class__.__name__,
            "gameObjects": {k: v.full_serialize() for k, v in self.gameObjects.items()},
            "init_args": self.initial_args()
        }


    @classmethod
    def from_state(cls, state_dict: dict):
        init_args = state_dict.get("init_args", {})
        init_args['tick_log_path'] = state_dict.get("tick_log_path")
        game = cls(**init_args)
        return game
