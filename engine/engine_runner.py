# EngineRunner (Eventlet/SocketIO friendly)
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

        self.startup_grace = 2.0

        # NOTE: no OS Thread objects here; we'll start background tasks in start()
        self.engine_task = None
        self.agent_task = None
        self.watchdog_task = None

        self.last_ping_time = time.time()
        self.timeout_seconds = 180
        self.on_tick_callbacks = []
        self.session = session  # must expose .socketio

    def add_on_tick_callback(self, fn):
        self.on_tick_callbacks.append(fn)

    def ping(self):
        self.last_ping_time = time.time()

    def start(self):
        print('[EngineRunner] Starting engine')

        # Run engine loop as a cooperative background task.
        # If Engine.start_loop uses time.sleep, eventlet monkey_patch already makes it cooperative.
        self.engine_task = self.session.socketio.start_background_task(self.engine.start_loop)

        # Start AI controllers (cooperative)
        if self.agent_map:
            for agent_id, controller in self.agent_map.items():
                print('[EngineRunner] Starting agent {} with controller {}'.format(
                    agent_id, controller.__class__.__name__))
                if hasattr(self.game, "add_agent"):
                    init_config = {}
                    if hasattr(controller, "agent_init_config"):
                        print(f"[EngineRunner] Initializing agent {agent_id} to game")
                        init_config = controller.agent_init_config()
                    self.game.add_agent(agent_id, **init_config)
                    self.session.register_agent(
                        agent_id,
                        self.game.agent_initial_state(agent_id)
                    )
                controller.agent = self.game.gameObjects[agent_id]

            self.agent_task = self.session.socketio.start_background_task(self._run_agents)

        # Watchdog (cooperative)
        self.watchdog_task = self.session.socketio.start_background_task(self._watchdog_loop)

    def _run_agents(self):
        print('[Engine Runner] Agent loop started')
        decision_dt = 1.0 / max(self.ai_tick_rate, 1e-6)
        last_action_sim_t = -1e9  # ensure first decision runs immediately

        while not self.game.done:
            # Timeout check
            if time.time() - self.last_ping_time > self.timeout_seconds:
                print("Session timed out. Stopping agent loop.")
                break

            # Use sim time (your current behavior)
            sim_time = getattr(self.engine, "sim_time",
                               self.engine.tick_count * self.engine.tick_interval)

            if (sim_time - last_action_sim_t) + 1e-12 >= decision_dt:
                last_action_sim_t = sim_time
                observations = self.game.get_observations()
                for agent_id, controller in self.agent_map.items():
                    action = controller.choose_action(observations, self.engine.tick_count)
                    if action:
                        self.engine.submit_intent(agent_id, action)
                # yield a tiny bit so other greenlets can run
                self.session.socketio.sleep(0)
            else:
                # Sleep until the next decision boundary (cooperative, no spin)
                remaining = decision_dt - (sim_time - last_action_sim_t)
                self.session.socketio.sleep(max(0.0, remaining))

            if self.is_max_speed:
                # still yield cooperatively rather than busy-spinning
                self.session.socketio.sleep(0)

    def _watchdog_loop(self):
        print('[EngineRunner] Watchdog started')
        while not self.game.done:
            if self.session.max_game_time and self.session.game_start_time:
                if self.engine.tick_count >= self.session.max_game_time * self.tick_rate:
                    print(f"[Watchdog] Max ticks reached at {self.engine.tick_count}")
                    self.session.engine.game.done = True
                    break

            # NEW: grace period
            if self.session.game_start_time and (time.time() - self.session.game_start_time) < self.startup_grace:
                if not self.is_max_speed:
                    time.sleep(0.2)
                continue

            if not self.session.has_active_websockets():
                print(f"[Watchdog] No active websockets at tick={self.engine.tick_count}; ending")
                self.game.done = True
                break

            time.sleep(0 if self.is_max_speed else 1)
