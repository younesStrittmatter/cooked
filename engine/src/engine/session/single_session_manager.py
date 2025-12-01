class SingleSessionManager:
    def __init__(self, engine, ui_modules):
        self._session = DummySession(engine, ui_modules)

    @property
    def max_agents(self):
        return 1

    def generate_agent_id(self):
        return "player_1"

    def find_or_create_session(self):
        return self._session

    def get_session(self, session_id):
        return self._session


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
