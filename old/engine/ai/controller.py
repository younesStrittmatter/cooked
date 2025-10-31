class Controller:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.agent = None

    def choose_action(self, observation: dict):
        raise NotImplementedError("Override this in subclasses.")
