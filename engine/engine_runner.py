from threading import Thread
import time

class EngineRunner:
    def __init__(self, game, engine, agent_map=None, tick_rate=5):
        self.game = game
        self.engine = engine
        self.agent_map = agent_map or {}
        self.tick_rate = tick_rate
        self.engine_thread = Thread(target=self.engine.start_loop, daemon=True)
        self.agent_thread = None

    def start(self):
        self.engine_thread.start()

        if self.agent_map:
            for agent_id in self.agent_map:
                if hasattr(self.game, "add_agent"):
                    self.game.add_agent(agent_id)

            self.agent_thread = Thread(target=self._run_agents, daemon=True)
            self.agent_thread.start()

    def _run_agents(self):
        while not self.game.done:
            observations = self.game.get_observations()
            for agent_id, agent in self.agent_map.items():
                obs = observations.get(agent_id)
                if obs is not None:
                    action = agent.choose_action(obs)
                    self.engine.submit_intent(agent_id, action)
            time.sleep(1.0 / self.tick_rate)
