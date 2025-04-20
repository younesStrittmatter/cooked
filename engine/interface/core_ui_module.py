from engine.interface.ui_module import UIModule
from pathlib import Path


class CoreUIModule(UIModule):
    """
    Core UI module that serves as a base for all UI modules.
    It provides the basic structure and functionality for rendering game objects.
    """

    def serialize_for_agent(self, game, engine, agent_id: str) -> dict:
        """
        Serialize game state for a specific agent.
        This method should be overridden by subclasses to provide specific serialization logic.
        """
        objects = []

        def collect_serialized(obj):
            objects.append(obj.serialize())
            if hasattr(obj, 'children'):
                for child in obj.children:
                    collect_serialized(child)

        for _, obj in game.gameObjects.items():
            collect_serialized(obj)


        return {'gameObjects': objects}



    def get_static_path(self):
        return Path(__file__).parent / "static"
