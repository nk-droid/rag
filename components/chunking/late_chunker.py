from components._base import ComponentSettings
from components.shared_types import Chunk

class LateChunkerSettings(ComponentSettings):
    _CONFIG_PATH = "chunking.late"

class LateChunker:
    def __init__(self, settings: LateChunkerSettings) -> None:
        self.settings = settings

    def chunk(self, text: str) -> list[Chunk]:
        raise NotImplementedError
