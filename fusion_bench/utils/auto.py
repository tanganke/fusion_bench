from omegaconf import DictConfig

from fusion_bench.utils import import_object


class BaseFactoryClass:
    _registry = {}

    @classmethod
    def from_config(cls, config: DictConfig):
        name = config.name
        if name not in cls._registry:
            raise ValueError(
                f"Unknown name: {name}, available names: {cls._registry.keys()}. "
                f"You can register a new item using `{cls.__name__}.register()` method."
            )

        item_cls = cls._registry[name]
        if isinstance(item_cls, str):
            if item_cls.startswith("."):
                item_cls = f"{cls.__module__}.{item_cls[1:]}"
            item_cls = import_object(item_cls)
        return item_cls(config)

    @classmethod
    def register(cls, name: str, item_cls):
        cls._registry[name] = item_cls

    @classmethod
    def available_items(cls):
        return list(cls._registry.keys())
