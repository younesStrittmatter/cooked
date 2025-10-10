from flask import Flask
from pathlib import Path
from engine.session.single_session_manager import SingleSessionManager
from engine.interface.static_routes import register_static_routes
from engine.interface.ai_only_engine_routes import register_ai_only_engine_routes
from engine.interface.core_ui_module import CoreUIModule


class AIOnlySessionApp:
    """
    AI-only version of SessionApp that doesn't require human players.
    Designed specifically for video recording and AI-only gameplay.
    """
    
    def __init__(self,
                 game_factory,
                 ui_modules=None,
                 agent_map=None,
                 path_root=None,
                 tick_rate=24,
                 ai_tick_rate=5,
                 is_max_speed=False,
                 agent_initialization_period=0.0):
        """
        Initialize the AI-Only Session App (Flask app)

        :param game_factory: The game factory to use
        :param ui_modules: Additional UI modules to load
        :param agent_map: A dictionary mapping agent IDs to agent instances
        :param path_root: The root to the game
        :param tick_rate: How often the game updates per second
        :param ai_tick_rate: How often the agents make decisions per seconds
        :param is_max_speed: Whether the game should run at max speed (useful for training agents)
        :param agent_initialization_period: The agent initialization period in seconds
        """

        self.app = Flask(__name__)
        self.path_root = Path(path_root) if path_root else Path(__file__).resolve().parents[2] / "game"
        self.ui_modules = ui_modules or []
        self.ui_modules += [CoreUIModule()]
        for module in self.ui_modules:
            if hasattr(module, "set_path_root"):
                module.set_path_root(self.path_root)

        # Use SingleSessionManager for AI-only operation
        self.session_manager = SingleSessionManager(
            game_factory=game_factory,
            ui_modules=self.ui_modules,
            agent_map=agent_map or {},
            tick_rate=tick_rate,
            ai_tick_rate=ai_tick_rate,
            is_max_speed=is_max_speed,
            agent_initialization_period=agent_initialization_period
        )

        # Use AI-only routes instead of regular engine routes
        register_ai_only_engine_routes(self.app, self.session_manager)
        register_static_routes(self.app, self.path_root, ui_modules=self.ui_modules)
        
        # Start the session immediately since no human players are needed
        self.session_manager.start_session()
    
    def get_session(self):
        """Get the current session"""
        return self.session_manager.get_session()
    
    def stop(self):
        """Stop the session"""
        if hasattr(self.session_manager, 'stop_session'):
            self.session_manager.stop_session() 