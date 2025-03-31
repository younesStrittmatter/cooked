from engine.base_game import BaseGame


class Game2d(BaseGame):
    def __init__(self):
        super().__init__()
        self.objects = [
            {"x": 2, "y": 3, "src": "agent1.jpeg"}
        ]
        self.id = None

    def add_agent(self, agent_id):
        self.id = agent_id

    def step(self, actions: dict):
        for aid, act in actions.items():
            if act == "left":
                self.objects[0]['x'] -= 1
            elif act == "right":
                self.objects[0]['x'] += 1

    def get_observations(self):
        return {}
