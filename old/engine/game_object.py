from abc import abstractmethod
import secrets


class GameObject:
    def __init__(self, id: str = None):
        if id is None:
            id = secrets.token_hex(8)
        self.id = id

    @abstractmethod
    def update(self, actions: dict, delta_time: float):
        pass

    @property
    def children(self):
        children = []
        for attr in vars(self).values():
            if isinstance(attr, GameObject):
                children.append(attr)
            elif isinstance(attr, list):
                children += [item for item in attr if isinstance(item, GameObject)]
        return children

    @abstractmethod
    def serialize(self) -> dict:
        return {'id': self.id}
