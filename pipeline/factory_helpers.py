from typing import Any, Callable

from components._base import ComponentSettings

def build_component(
    component_cls: type,
    settings_cls: type[ComponentSettings],
    config: dict[str, Any],
    deps_builder: Callable[[ComponentSettings, dict[str, Any]], dict[str, Any]] | None = None,
) -> Any:
    settings = settings_cls.from_config(config)
    deps = deps_builder(settings, config) if deps_builder else {}
    return component_cls(settings=settings, **deps)
