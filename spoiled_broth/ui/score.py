from mpmath.libmp import normalize

from engine.game_object import GameObject
from engine.extensions.renderer2d.text import Text


class Score(GameObject):
    def __init__(self, id=None):
        super().__init__(id=id)
        self.score = 0
        self.text_obj = Text(
            id=f"{self.id}_text",
            right=0.1,
            bottom=0.1,
            content=f"{self.score}",
            color="#FFFFFF",
            font_size=28,
            z_index=0,
        )

    def serialize(self) -> dict:
        return {
            'id': self.id,
            'score': self.score
        }

    def update(self, actions: dict, delta_time: float):
        self.text_obj.content = f"{getRank(self.score)}\n{self.score}"
        self.text_obj.color = getScoreColor(self.score)

    @classmethod
    def deserialize(cls, data: dict, game=None):
        obj = cls(id=data['id'])
        obj.score = data['score']
        return obj


def getRank(score):
    if score <= 1:
        return "🔪 Tomato Trainee"
    if score <= 3:
        return "🍅 Cherry Chopper"
    if score <= 5:
        return "🥗 Caprese Crafter"
    return "👑 Heirloom Hero"


def getScoreColor(score):
    if score <= 1: return "#CCCCCC"
    if score <= 3: return "#BBBB88"
    if score <= 5: return "#FFC107"
    return "#FFF380"
