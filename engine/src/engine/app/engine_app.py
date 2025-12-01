from flask import Flask
from pathlib import Path
from engine.core import Engine
from engine.interface.engine_routes import register_engine_routes
from engine.interface.static_routes import register_static_routes
from engine.engine_runner import EngineRunner
from engine.session.single_session_manager import SingleSessionManager
from engine.interface.core_ui_module import CoreUIModule

class EngineApp:
    def __init__(self,
                 game_factory,
                 ui_modules=None,
                 agent_map=None,
                 tick_rate=5,
                 path_root=None):

        self.app = Flask(__name__)
        self.path_root = Path(path_root) if path_root else Path(__file__).resolve().parents[2] / "game"
        self.ui_modules = ui_modules or []
        self.ui_modules += [CoreUIModule()]
        for module in self.ui_modules:
            if hasattr(module, "set_path_root"):
                module.set_path_root(self.path_root)

        self.game = game_factory()
        self.engine = Engine(self.game, tick_rate=tick_rate)

        self.runner = EngineRunner(
            game=self.game,
            engine=self.engine,
            agent_map=agent_map or {},
            tick_rate=tick_rate
        )
        self.runner.start()

        # Wrap engine and modules into session-like interface
        session_manager = SingleSessionManager(self.engine, self.ui_modules)

        # Register API routes and static routes
        register_engine_routes(self.app, session_manager)
        register_static_routes(self.app, self.path_root, ui_modules=self.ui_modules)

    def run(self, **kwargs):
        self.app.run(**kwargs)
