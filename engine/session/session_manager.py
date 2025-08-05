import secrets
from engine.session.session import Session


class SessionManager:
    def __init__(self,
                 game_factory, ui_modules, agent_map, tick_rate, ai_tick_rate, n_players, is_max_speed, socketio,
                 max_game_time=None, redirect_link=None):
        """
        Initialize the Session Manager. This is used to manage the sessions of the game. For example, to
        make sure that the game is not started until all players have joined.

        It also manages that a new session is created if the current one is full.
        """
        self.game_factory = game_factory
        self.ui_modules = ui_modules
        self.agent_map = agent_map
        self.tick_rate = tick_rate
        self.ai_tick_rate = ai_tick_rate
        self.n_players = n_players
        self.is_max_speed = is_max_speed
        self.socketio = socketio
        self.sessions = {}
        self.max_game_time = max_game_time
        self.redirect_link = redirect_link

    def generate_agent_id(self):
        return f"player_{secrets.token_hex(4)}"

    def find_or_create_session(self, params=None):
        for session in self.sessions.values():
            if not session.is_full(self.n_players):
                return session
        new_session = Session(
            game_factory=self.game_factory,
            ui_modules=self.ui_modules,
            agent_map=self.agent_map,
            tick_rate=self.tick_rate,
            ai_tick_rate=self.ai_tick_rate,
            is_max_speed=self.is_max_speed,
            socketio=self.socketio,
            max_game_time=self.max_game_time,
            redirect_link=self.redirect_link,
            url_params=params
        )
        self.sessions[new_session.id] = new_session
        return new_session

    def get_session(self, session_id):
        return self.sessions.get(session_id)
