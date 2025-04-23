from engine.game_object import GameObject


class Agent(GameObject):
    def __init__(self, id):
        super().__init__(id=id)

    def move(self, delta_time):
        pass

    def update(self, actions: dict, delta_time: float):
        self.move(delta_time)

    def serialize(self) -> dict:
        return {
            "id": self.id,
        }
