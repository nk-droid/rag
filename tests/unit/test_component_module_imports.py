import importlib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPONENTS_ROOT = PROJECT_ROOT / "components"

def _component_module_names() -> list[str]:
    modules: list[str] = []
    for path in sorted(COMPONENTS_ROOT.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        relative = path.relative_to(COMPONENTS_ROOT).with_suffix("")
        modules.append("components." + ".".join(relative.parts))
    return modules

MODULE_NAMES = _component_module_names()

def test_component_module_inventory_is_non_empty() -> None:
    assert MODULE_NAMES, "No component modules found to test."

@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_component_modules_import(module_name: str) -> None:
    imported = importlib.import_module(module_name)
    assert imported is not None
