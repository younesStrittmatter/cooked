from flask import Flask, request
from flask_socketio import SocketIO, emit, disconnect
from pathlib import Path
from engine.session.session_manager import SessionManager
from engine.interface.static_routes import register_static_routes
from engine.interface.engine_routes import register_engine_routes
from engine.interface.core_ui_module import CoreUIModule
import json
from urllib.parse import parse_qs


class SessionApp:
    def __init__(self,
                 game_factory,
                 ui_modules=None,
                 agent_map=None,
                 path_root=None,
                 tick_rate=24,
                 ai_tick_rate=5,
                 n_players=2,
                 is_max_speed=False,
                 max_game_time=None,
                 redirect_link=None):

        self.app = Flask(__name__)
        self.app.config["SECRET_KEY"] = "replace-this-if-needed"

        self.socketio = SocketIO(self.app,
                                 cors_allowed_origins="*",
                                 async_mode="eventlet",
                                 ping_interval=10,
                                 ping_timeout=10)

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
            is_max_speed=is_max_speed,
            socketio=self.socketio,
            max_game_time=max_game_time,
            redirect_link=redirect_link
        )

        register_engine_routes(self.app, self.session_manager)
        register_static_routes(self.app, self.path_root, ui_modules=self.ui_modules)

        self._register_websocket_routes()

    def _register_websocket_routes(self):
        @self.socketio.on("connect")
        def on_connect():
            sid = request.sid
            query_string = request.query_string.decode()
            params = parse_qs(query_string)
            agent_id = self.session_manager.generate_agent_id()
            session = self.session_manager.find_or_create_session(params)
            session.add_agent(agent_id, url_params=params)
            session.register_websocket(agent_id, sid)

            if session.is_full(self.session_manager.n_players):
                print(f"[SessionApp] Session {session.id} is full, starting game")
                session.start()

            emit("joined", {
                "agent_id": agent_id,
                "session_id": session.id
            })

        @self.socketio.on("intent")
        def handle_intent(data):
            session = self.session_manager.get_session(data["session_id"])
            if session:
                session.engine.submit_intent(data["agent_id"], data["action"])

        @self.socketio.on("disconnect")
        def on_disconnect():
            sid = request.sid
            print(f"[WS] disconnect handler triggered for sid={sid}")
            for session in self.session_manager.sessions.values():
                agent_id = session.get_agent_id_by_sid(sid)
                if agent_id:
                    if not session.is_full(self.session_manager.n_players):
                        print(f"[WS] {agent_id} disconnected before game start, removing agent")
                        if hasattr(session.engine.game, "remove_agent"):
                            session.engine.game.remove_agent(agent_id)
                        session._agents.remove(agent_id)
                        session.recorder.unregister_agent(agent_id)
                    print(f"[WS] {agent_id} disconnected")
                    session.websockets.pop(agent_id, None)
                    session.agent_to_sid.pop(agent_id, None)
                    session.sid_to_agent.pop(sid, None)
                    break
