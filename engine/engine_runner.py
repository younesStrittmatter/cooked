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
                 session=None):
        self.game = game
        self.engine = engine
        self.agent_map = agent_map or {}
        self.tick_rate = tick_rate
        self.ai_tick_rate = ai_tick_rate
        self.is_max_speed = is_max_speed
        self.engine_thread = Thread(target=self.engine.start_loop, daemon=True)
        self.agent_thread = None
        self.last_ping_time = time.time()
        self.timeout_seconds = 180
        self.watchdog_thread = None
        self.on_tick_callbacks = []
        self.session = session

    def add_on_tick_callback(self, fn):
        self.on_tick_callbacks.append(fn)

    def ping(self):
        self.last_ping_time = time.time()

    def start(self):
        self.engine_thread.start()
        if self.agent_map:
            for agent_id, controller in self.agent_map.items():
                print('[EngineRunner] Starting agent {} with controller {}'.format(agent_id,
                                                                                   controller.__class__.__name__))
                if hasattr(self.game, "add_agent"):
                    init_config = {}
                    if hasattr(controller, "agent_init_config"):
                        print(f"[EngineRunner] Initializing agent {agent_id} to game")
                        init_config = controller.agent_init_config()
                    self.game.add_agent(agent_id, **init_config)
                    self.session.register_agent(
                        agent_id,
                        self.game.agent_initial_state(agent_id))
                controller.agent = self.game.gameObjects[agent_id]

            self.agent_thread = Thread(target=self._run_agents, daemon=True)
            self.agent_thread.start()
        self.watchdog_thread = Thread(target=self._watchdog_loop, daemon=True)
        self.watchdog_thread.start()

    def _run_agents(self):
        last_action_time = 0.0
        decision_interval = 1.0 / self.ai_tick_rate

        while not self.game.done:
            if time.time() - self.last_ping_time > self.timeout_seconds:
                print("Session timed out. Stopping agent thread.")
                break
            sim_time = getattr(self.engine, "sim_time", self.engine.tick_count * self.engine.tick_interval)

            if sim_time - last_action_time >= decision_interval:
                last_action_time = sim_time
                observations = self.game.get_observations()
                for agent_id, agent in self.agent_map.items():
                    action = agent.choose_action(observations, self.engine.tick_count)
                    if action:
                        self.engine.submit_intent(agent_id, action)
            else:
                if not self.is_max_speed:
                    time.sleep(0.001)

    def _watchdog_loop(self):
        print('[EngineRunner] Watchdog started')
        while not self.game.done:
            if self.session.max_game_time and self.session.game_start_time:
                if time.time() - self.session.game_start_time > self.session.max_game_time:
                    print("[Watchdog] Max game time reached. Ending game.")
                    self.session.engine.game.done = True
                    break
            if not self.session.has_active_websockets():
                print('[EngineRunner] No active websockets')
                print("No active clients. Shutting down.")
                self.game.done = True
                break
            time.sleep(1)
