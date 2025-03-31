from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def choose_action(self, observation) -> str:
        pass