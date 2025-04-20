from threading import Thread
import time


class EngineRunner:
    def __init__(self,
                 game,
                 engine,
                 agent_map=None,
                 tick_rate=5,
                 ai_tick_rate=5,
                 is_max_speed=False):
        self.game = game
        self.engine = engine
        self.agent_map = agent_map or {}
        self.tick_rate = tick_rate
        self.ai_tick_rate = ai_tick_rate
        self.is_max_speed = is_max_speed
        self.engine_thread = Thread(target=self.engine.start_loop, daemon=True)
        self.agent_thread = None

    def start(self):
        self.engine_thread.start()
        if self.agent_map:
            for agent_id, controller in self.agent_map.items():
                if hasattr(self.game, "add_agent"):
                    self.game.add_agent(agent_id)
                controller.agent = self.game.gameObjects[agent_id]
            self.agent_thread = Thread(target=self._run_agents, daemon=True)
            self.agent_thread.start()

    def _run_agents(self):
        last_action_time = 0.0
        decision_interval = 1.0 / self.ai_tick_rate


        while not self.game.done:
            sim_time = getattr(self.engine, "sim_time", self.engine.tick_count * self.engine.tick_interval)

            if sim_time - last_action_time >= decision_interval:
                last_action_time = sim_time

                observations = self.game.get_observations()
                for agent_id, agent in self.agent_map.items():
                    action = agent.choose_action(observations)
                    self.engine.submit_intent(agent_id, action)
            else:
                if not self.is_max_speed:
                    time.sleep(0.001)
