from typing import Any

from components._base import ComponentSettings
from components.memory.memory_store import MemoryStore
from components.shared_types import MemoryRecord

class MemoryWriterSettings(ComponentSettings):
    _CONFIG_PATH = "memory.writer"

class MemoryWriter:
    def __init__(self, settings: MemoryWriterSettings, store: MemoryStore) -> None:
        self.settings = settings
        self.store = store

    def write(self, interaction: dict[str, Any]) -> MemoryRecord:
        record = MemoryRecord(
            id=str(interaction.get("id", "memory-0")),
            content=str(interaction.get("content", "")),
            metadata={},
        )
        return self.store.add(record)
