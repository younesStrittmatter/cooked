import secrets
from engine.session.session import Session


class SessionManager:
    def __init__(self, game_factory, ui_modules, agent_map, tick_rate, ai_tick_rate, n_players, is_max_speed):
        """
        Initialize the Session Manager. This is used to manage the sessions of the game. For example, to
        make sure that the game is not started until all players have joined.

        It also manages that a new session is created if the current one is full.
        :param game_factory: The game factory to use
        :param ui_modules: The UI modules to use
        :param agent_map: A dictionary mapping agent IDs to agent instances
        :param tick_rate: The tick rate of the game
        :param ai_tick_rate: The decision rate of the agents
        :param n_players: How many players have to join the game to start it
        :param is_max_speed: Weather the game should run at max speed (useful for training agents)
        """
        self.game_factory = game_factory
        self.ui_modules = ui_modules
        self.agent_map = agent_map
        self.tick_rate = tick_rate
        self.ai_tick_rate = ai_tick_rate
        self.n_players = n_players
        self.is_max_speed = is_max_speed
        self.sessions = {}

    def generate_agent_id(self):
        return f"player_{secrets.token_hex(4)}"

    def find_or_create_session(self):
        for session in self.sessions.values():
            if not session.is_full(self.n_players):
                return session
        new_session = Session(self.game_factory,
                              self.ui_modules,
                              self.agent_map,
                              self.tick_rate,
                              self.ai_tick_rate,
                              self.is_max_speed
                              )
        self.sessions[new_session.id] = new_session
        return new_session

    def get_session(self, session_id):
        return self.sessions.get(session_id)
