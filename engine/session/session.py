import secrets
from engine.core import Engine
from engine.engine_runner import EngineRunner
from engine.logging.replay_recorder import ReplayRecorder
import time


class Session:
    def __init__(self,
                 game_factory,
                 ui_modules,
                 agent_map,
                 tick_rate,
                 ai_tick_rate,
                 is_max_speed,
                 socketio,
                 max_game_time=None,
                 redirect_link=None,
                 url_params=None):
        self.id = secrets.token_hex(8)
        self.recorder = ReplayRecorder(self.id)

        self.socketio = socketio
        self.sid_to_agent = {}
        self.agent_to_sid = {}
        self.engine = Engine(game_factory(url_params=url_params),
                             tick_rate=tick_rate,
                             is_max_speed=is_max_speed,
                             replay_recorder=self.recorder)
        self.engine.add_on_tick_callback(self.broadcast_state)
        self.engine.add_on_game_end_callback(self.handle_game_end)
        self.recorder.set_config(self.engine.game.serialize_initial_state())
        self.ui_modules = ui_modules
        self._agents = []
        self.websockets = {}
        self.max_game_time = max_game_time
        self.redirect_link = redirect_link
        self.game_start_time = None

        self._runner = EngineRunner(
            game=self.engine.game,
            engine=self.engine,
            agent_map=agent_map,
            tick_rate=tick_rate,
            ai_tick_rate=ai_tick_rate,
            is_max_speed=is_max_speed,
            session=self
        )

    def register_websocket(self, agent_id, sid):
        self.agent_to_sid[agent_id] = sid
        self.sid_to_agent[sid] = agent_id

    def get_agent_id_by_sid(self, sid):
        return self.sid_to_agent.get(sid)

    def has_active_websockets(self):
        return any(sid in self.sid_to_agent for sid in self.sid_to_agent)

    def add_agent(self, agent_id, url_params=None):
        if hasattr(self.engine.game, "add_agent"):
            self.engine.game.add_agent(agent_id, url_params=url_params)
        self._agents.append(agent_id)
        self.recorder.register_agent(agent_id, self.engine.game.agent_initial_state(agent_id))

    def register_agent(self, agent_id, initial_state):
        self.recorder.register_agent(agent_id, initial_state)

    def is_full(self, max_agents):
        return len(self._agents) >= max_agents

    def start(self):
        self.game_start_time = time.time()
        self._runner.start()

    def broadcast_state(self):
        game = self.engine.game
        engine = self.engine

        for agent_id, sid in self.agent_to_sid.items():
            payload = {
                "type": "state",
                "you": agent_id,
                "tick": engine.tick_count
            }
            for module in self.ui_modules:
                payload.update(module.serialize_for_agent(game, engine, agent_id))
            try:
                self.socketio.emit("state", payload, to=sid)
            except Exception as e:
                print(f"[WS] Error sending to {agent_id}: {e}")

    def handle_game_end(self):
        print(f"[Session {self.id}] Game ended. Disconnecting clients and saving replay.")
        self.recorder.save()
        for sid in list(self.sid_to_agent.keys()):
            try:
                if self.redirect_link is not None:
                    if hasattr(self.engine.game, "redirect_link"):
                        redirect_link = self.engine.game.redirect_link()
                    else:
                        redirect_link = self.redirect_link
                    self.socketio.emit("redirect", {"url": redirect_link}, to=sid)
                    self.socketio.sleep(0.5)
                self.socketio.server.disconnect(sid)
            except Exception as e:
                print(f"[WS] Error disconnecting {sid}: {e}")
