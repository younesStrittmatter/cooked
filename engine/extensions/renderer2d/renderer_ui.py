from engine.interface.ui_module import UIModule
from flask import send_from_directory
from pathlib import Path

class Renderer2DModule(UIModule):
    def __init__(self, sprite_folder_name="sprites"):
        self.sprite_folder_name = sprite_folder_name
        self._path_root = None  # This will be injected by the engine

    def serialize_for_agent(self, game, engine, agent_id: str) -> dict:
        return {
            "objects": getattr(game, "objects", [])
        }

    def get_static_path(self):
        return Path(__file__).parent / "static"

    def register_routes(self, app):
        @app.route("/sprites/<path:filename>")
        def serve_sprite(filename):
            if not self._path_root:
                raise RuntimeError("Renderer2DModule requires `path_root` to be set.")
            sprite_path = self._path_root / "static" / self.sprite_folder_name
            return send_from_directory(sprite_path, filename)

    # 🆕 Inject path_root from the engine during setup
    def set_path_root(self, path_root):
        self._path_root = Path(path_root)
