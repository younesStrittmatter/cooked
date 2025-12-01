from abc import abstractmethod
import secrets

CLASS_REGISTRY = {}

class GameObject:
    def __init__(self, id: str = None):
        self.id = id or secrets.token_hex(8)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        CLASS_REGISTRY[cls.__name__] = cls

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
        # Should only return subclass-specific fields
        return {}

    def full_serialize(self):
        children_data = []
        for attr_name, attr_value in vars(self).items():
            if isinstance(attr_value, GameObject):
                children_data.append({
                    "attr": attr_name,
                    "data": attr_value.full_serialize()
                })
            elif isinstance(attr_value, list):
                for i, item in enumerate(attr_value):
                    if isinstance(item, GameObject):
                        children_data.append({
                            "attr": f"{attr_name}[{i}]",
                            "data": item.full_serialize()
                        })
        return {
            "id": self.id,
            "class": self.__class__.__name__,
            "data": self.serialize(),
            "children": children_data
        }

    @classmethod
    def full_deserialize(cls, data: dict, game=None):
        class_name = data["class"]
        obj_cls = CLASS_REGISTRY[class_name]
        obj = obj_cls.deserialize(data["data"], game=game)
        obj.id = data["id"]

        for child_info in data.get("children", []):
            attr = child_info["attr"]
            child = GameObject.full_deserialize(child_info["data"], game=game)

            if "[" in attr:  # Handle list elements
                attr_base, index = attr[:-1].split("[")
                index = int(index)
                if not hasattr(obj, attr_base):
                    setattr(obj, attr_base, [])
                lst = getattr(obj, attr_base)
                while len(lst) <= index:
                    lst.append(None)
                lst[index] = child
            else:
                setattr(obj, attr, child)

        return obj

    @classmethod
    def deserialize(cls, data: dict, game=None):
        raise NotImplementedError(f"{data} has no method `deserialize()`."
                                  " Each subclass must implement `deserialize()`")
