from typing import Any, ClassVar
from pydantic import BaseModel, ConfigDict

class ComponentSettings(BaseModel):

    model_config = ConfigDict(extra="forbid", frozen=True)
    _CONFIG_PATH: ClassVar[str] = ""

    @classmethod
    def _slice(cls, config: dict[str, Any]) -> dict[str, Any]:
        node: Any = config or {}
        if not cls._CONFIG_PATH:
            return node if isinstance(node, dict) else {}
        for key in cls._CONFIG_PATH.split("."):
            if not isinstance(node, dict):
                return {}
            node = node.get(key, {})
        return node if isinstance(node, dict) else {}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ComponentSettings":
        sliced = cls._slice(config)
        known = {k: v for k, v in sliced.items() if k in cls.model_fields}
        return cls(**known)

    def with_overrides(self, overrides: dict[str, Any]) -> "ComponentSettings":
        if not overrides:
            return self
        valid = {k: v for k, v in overrides.items() if k in type(self).model_fields}
        return self.model_copy(update=valid) if valid else self
