from components._base import ComponentSettings
from components.shared_types import MemoryRecord

class MemoryStoreSettings(ComponentSettings):
    _CONFIG_PATH = "memory.store"

    max_entries: int = 10_000

class MemoryStore:
    def __init__(self, settings: MemoryStoreSettings) -> None:
        self.settings = settings
        self._items: list[MemoryRecord] = []

    def add(self, record: MemoryRecord) -> MemoryRecord:
        self._items.append(record)
        if len(self._items) > self.settings.max_entries:
            self._items = self._items[-self.settings.max_entries:]
        return record

    def all(self) -> list[MemoryRecord]:
        return list(self._items)

    def search(self, query: str, top_k: int = 5) -> list[MemoryRecord]:
        text = str(query or "").strip().lower()
        if not text:
            return self._items[: max(0, int(top_k))]

        hits = [
            item
            for item in self._items
            if text in item.content.lower()
        ]
        return hits[: max(0, int(top_k))]
