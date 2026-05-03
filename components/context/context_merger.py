from components._base import ComponentSettings
from components.shared_types import RetrievedChunk

class ContextMergerSettings(ComponentSettings):
    _CONFIG_PATH = "context.merger"

class ContextMerger:
    def __init__(self, settings: ContextMergerSettings) -> None:
        self.settings = settings

    def merge(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        merged: list[RetrievedChunk] = []
        seen_texts: set[str] = set()
        for chunk in chunks:
            if chunk.text in seen_texts:
                continue
            seen_texts.add(chunk.text)
            merged.append(chunk)
        return merged
