from threading import Thread
import time


class EngineRunner:
    def __init__(self,
                 game,
                 engine,
                 agent_map=None,
                 tick_rate=5,
                 ai_tick_rate=5,
                 is_max_speed=False,
                 agent_initialization_period=0.0):
        self.game = game
        self.engine = engine
        self.agent_map = agent_map or {}
        self.tick_rate = tick_rate
        self.ai_tick_rate = ai_tick_rate
        self.is_max_speed = is_max_speed
        self.agent_initialization_period = agent_initialization_period
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
        agents_started_acting = False
        
        while not self.game.done:
            sim_time = getattr(self.engine, "sim_time", self.engine.tick_count * self.engine.tick_interval)

            # Check if initialization period has passed
            if sim_time < self.agent_initialization_period:
                # During initialization period, agents should not act
                if not self.is_max_speed:
                    time.sleep(0.001)
                continue
            
            # Log when agents first start acting (only once)
            if not agents_started_acting:
                frame_count = getattr(self.engine, 'tick_count', 0)
                print(f"Agents starting to act at frame {frame_count} (time: {sim_time:.3f}s)")
                agents_started_acting = True

            if sim_time - last_action_time >= decision_interval:
                last_action_time = sim_time

                observations = self.game.get_observations()
                for agent_id, agent in self.agent_map.items():
                    action = agent.choose_action(observations)
                    self.engine.submit_intent(agent_id, action)
            
            if not self.is_max_speed:
                time.sleep(0.001)