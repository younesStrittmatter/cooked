from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.extensions.topDownGridWorld.agent import Agent


class Intent(ABC):
    @abstractmethod
    def update(self, agent: Agent, delta_time: float) -> None:
        pass

    @abstractmethod
    def finished(self, agent: Agent) -> bool:
        pass
