from abc import abstractmethod
from typing import Optional


class Controller:
    is_controller = True
    def __init__(self,
                 agent_id: str):
        self.agent_id = agent_id
        self.agent = None

    @abstractmethod
    def choose_action(self,
                      observation: dict,
                      tick_count: Optional[int] = None) -> dict:
        raise NotImplementedError("Override this in subclasses.")
