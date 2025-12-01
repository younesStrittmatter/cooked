from abc import ABC, abstractmethod
from pathlib import Path
from flask import Flask


class UIModule(ABC):
    @abstractmethod
    def serialize_for_agent(self, game, engine, agent_id: str) -> dict:
        pass

    def get_static_path(self) -> Path | None:
        return None

    def register_routes(self, app: Flask):
        pass
