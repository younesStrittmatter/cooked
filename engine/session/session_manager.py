import secrets
from engine.session.session import Session


class SessionManager:
    def __init__(self, game_factory, ui_modules, agent_map, tick_rate, max_agents):
        self.game_factory = game_factory
        self.ui_modules = ui_modules
        self.agent_map = agent_map
        self.tick_rate = tick_rate
        self.max_agents = max_agents
        self.sessions = {}

    def generate_agent_id(self):
        return f"player_{secrets.token_hex(4)}"

    def find_or_create_session(self):
        for session in self.sessions.values():
            if not session.is_full(self.max_agents):
                return session
        new_session = Session(self.game_factory, self.ui_modules, self.agent_map, self.tick_rate)
        self.sessions[new_session.id] = new_session
        return new_session

    def get_session(self, session_id):
        return self.sessions.get(session_id)
