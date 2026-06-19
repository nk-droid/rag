from typing import Iterable

from components._base import ComponentSettings
from components.shared_types import MemoryRecord

class MemoryFilterSettings(ComponentSettings):
    _CONFIG_PATH = "memory.filter"

class MemoryFilter:
    def __init__(self, settings: MemoryFilterSettings) -> None:
        self.settings = settings

    def filter(self, memories: Iterable[MemoryRecord]) -> list[MemoryRecord]:
        return list(memories)
