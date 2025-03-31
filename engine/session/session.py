import secrets
from engine.core import Engine
from engine.engine_runner import EngineRunner


class Session:
    def __init__(self,
                 game_factory,
                 ui_modules,
                 agent_map,
                 tick_rate):
        self.id = secrets.token_hex(4)
        self.engine = Engine(game_factory(), tick_rate=tick_rate)
        self.ui_modules = ui_modules
        self._agents = []
        self._runner = EngineRunner(
            game=self.engine.game,
            engine=self.engine,
            agent_map=agent_map,
            tick_rate=tick_rate,
        )

    def add_agent(self, agent_id):
        if hasattr(self.engine.game, "add_agent"):
            self.engine.game.add_agent(agent_id)
        self._agents.append(agent_id)

    def is_full(self, max_agents):
        return len(self._agents) >= max_agents

    def start(self):
        self._runner.start()
