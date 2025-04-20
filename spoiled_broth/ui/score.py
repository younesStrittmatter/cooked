from engine.game_object import GameObject


class Score(GameObject):
    def __init__(self):
        super().__init__()
        self.score = 0

    def serialize(self) -> dict:
        return {
            'score': self.score
        }

    def update(self, actions: dict, delta_time: float):
        pass
