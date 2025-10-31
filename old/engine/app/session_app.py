from flask import Flask
from pathlib import Path
from engine.session.session_manager import SessionManager
from engine.interface.static_routes import register_static_routes
from engine.interface.engine_routes import register_engine_routes
from engine.interface.core_ui_module import CoreUIModule


class SessionApp:
    def __init__(self,
                 game_factory,
                 ui_modules=None,
                 agent_map=None,
                 path_root=None,
                 tick_rate=24,
                 ai_tick_rate=5,
                 n_players=2,
                 is_max_speed=False):
        """
        Initialize the Session App (Flask app)

        :param game_factory: The game factory to use
        :param ui_modules: Additional UI modules to load
        :param agent_map: A dictionary mapping agent IDs to agent instances
        :param path_root: The root to the game
        :param tick_rate: How often the game updates per second
        :param ai_tick_rate: How often the agents make decisions per seconds
        :param n_players: How many players have to join the game to start it
        :param is_max_speed: Whether the game should run at max speed (useful for training agents)
        """

        assert not is_max_speed or n_players == 0, "Training mode is not"

        self.app = Flask(__name__)
        self.path_root = Path(path_root) if path_root else Path(__file__).resolve().parents[2] / "game"
        self.ui_modules = ui_modules or []
        self.ui_modules += [CoreUIModule()]
        for module in self.ui_modules:
            if hasattr(module, "set_path_root"):
                module.set_path_root(self.path_root)

        self.session_manager = SessionManager(
            game_factory=game_factory,
            ui_modules=self.ui_modules,
            agent_map=agent_map or {},
            tick_rate=tick_rate,
            ai_tick_rate=ai_tick_rate,
            n_players=n_players,
            is_max_speed=is_max_speed
        )

        # Unified route and asset registration
        register_engine_routes(self.app, self.session_manager)
        register_static_routes(self.app, self.path_root, ui_modules=self.ui_modules)
