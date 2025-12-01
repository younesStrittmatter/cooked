from engine.game_object import GameObject
from engine.extensions.renderer2d.text import Text


class Message(GameObject):
    def __init__(self, id=None, message=None):
        super().__init__(id=id)
        self.message = message
        self.text_obj = Text(
            id=f"{self.id}_text",
            left=200,
            top=200,
            content=f"{self.message}",
            color="#FFFFFF",
            font_size=28,
            z_index=-3,
        )
        self.is_show = True

    def serialize(self) -> dict:
        return {
            'id': self.id,
            'message': self.message,
        }

    def set_message(self, message):
        self.message = message

    def update(self, actions: dict, delta_time: float):
        if self.is_show:
            self.text_obj.content = f"{self.message}"
        else:
            self.text_obj.content = ''


    @classmethod
    def deserialize(cls, data: dict, game=None):
        obj = cls(id=data['id'])
        obj.message = data['message']
        return obj


