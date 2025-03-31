from engine.interface.ui_module import UIModule


class Game2DUI(UIModule):
    def __init__(self, game, engine):
        super().__init__(game, engine)

    def serialize_for_agent(self, game, engine, agent_id: str) -> dict:
        return super().serialize_for_agent(game, engine, agent_id)
