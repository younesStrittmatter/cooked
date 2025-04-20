import secrets
from engine.core import Engine
from engine.engine_runner import EngineRunner


class Session:
    def __init__(self,
                 game_factory,
                 ui_modules,
                 agent_map,
                 tick_rate,
                 ai_tick_rate,
                 is_max_speed):
        self.id = secrets.token_hex(8)
        self.engine = Engine(game_factory(),
                             tick_rate=tick_rate,
                             is_max_speed=is_max_speed)
        self.ui_modules = ui_modules
        self._agents = []
        self._runner = EngineRunner(
            game=self.engine.game,
            engine=self.engine,
            agent_map=agent_map,
            tick_rate=tick_rate,
            ai_tick_rate=ai_tick_rate,
            is_max_speed=is_max_speed
        )

    def add_agent(self, agent_id):
        if hasattr(self.engine.game, "add_agent"):
            self.engine.game.add_agent(agent_id)
        self._agents.append(agent_id)

    def is_full(self, max_agents):
    
        return len(self._agents) >= max_agents

    def start(self):
        self._runner.start()
