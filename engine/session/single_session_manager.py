from engine.session.session import Session


class SingleSessionManager:
    """
    Single Session Manager for AI-only operation.
    Manages a single session that starts immediately without waiting for human players.
    """
    
    def __init__(self, game_factory, ui_modules, agent_map, tick_rate, ai_tick_rate, is_max_speed, agent_initialization_period=15.0):
        """
        Initialize the Single Session Manager for AI-only operation.
        
        :param game_factory: The game factory to use
        :param ui_modules: The UI modules to use
        :param agent_map: A dictionary mapping agent IDs to agent instances
        :param tick_rate: The tick rate of the game
        :param ai_tick_rate: The decision rate of the agents
        :param is_max_speed: Whether the game should run at max speed (useful for training agents)
        :param agent_initialization_period: The agent initialization period in seconds
        """
        self.game_factory = game_factory
        self.ui_modules = ui_modules
        self.agent_map = agent_map
        self.tick_rate = tick_rate
        self.ai_tick_rate = ai_tick_rate
        self.is_max_speed = is_max_speed
        self.agent_initialization_period = agent_initialization_period
        self.session = None
        self.session_id = None
        
        # Add compatibility attributes for engine routes
        self.n_players = 0  # No human players required
        self.sessions = {}  # For compatibility with SessionManager interface

    def start_session(self):
        """Start a new session immediately"""
        if self.session is None:
            self.session = Session(
                self.game_factory,
                self.ui_modules,
                self.agent_map,
                self.tick_rate,
                self.ai_tick_rate,
                self.is_max_speed,
                self.agent_initialization_period
            )
            self.session_id = self.session.id
            
            # Store session in sessions dict for compatibility
            self.sessions[self.session_id] = self.session
            
            # Add all AI agents to the session
            for agent_id in self.agent_map.keys():
                self.session.add_agent(agent_id)
            
            # Ensure agents are properly positioned before starting the engine
            import time
            time.sleep(0.1)  # Small delay to ensure all agents are properly initialized
            
            # Verify agent positions are set
            game = self.session.engine.game
            for agent_id in self.agent_map.keys():
                agent = game.gameObjects.get(agent_id)
                if agent and hasattr(agent, 'x') and hasattr(agent, 'y'):
                    print(f"Agent {agent_id} initialized at position ({agent.x}, {agent.y})")
            
            # Start the session
            self.session.start()
            print(f"AI-only session started with ID: {self.session_id}")
            print(f"Added {len(self.agent_map)} AI agents to the session")

    def get_session(self, session_id=None):
        """Get the current session (ignores session_id for single session)"""
        return self.session

    def stop_session(self):
        """Stop the current session"""
        if self.session:
            self.session.engine.stop()
            self.session = None
            self.session_id = None
            self.sessions.clear()
            print("AI-only session stopped")

    def generate_agent_id(self):
        """Generate an agent ID (not used in AI-only mode)"""
        return None

    def find_or_create_session(self):
        """Find or create a session (always returns the single session)"""
        if self.session is None:
            self.start_session()
        return self.session


class DummySession:
    def __init__(self, engine, ui_modules):
        self.id = "single"
        self.engine = engine
        self.ui_modules = ui_modules
        self._started = False

    def add_agent(self, agent_id):
        if hasattr(self.engine.game, "add_agent"):
            self.engine.game.add_agent(agent_id)

    def is_full(self, _):
        return True

    def start(self):
        if not self._started:
            self._started = True
