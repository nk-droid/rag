from langchain_text_splitters import RecursiveCharacterTextSplitter

from components._base import ComponentSettings
from components.shared_types import Chunk

class RecursiveChunkerSettings(ComponentSettings):
    _CONFIG_PATH = "chunking.recursive"

    chunk_size: int = 512
    chunk_overlap: int = 50

class RecursiveChunker:
    def __init__(self, settings: RecursiveChunkerSettings) -> None:
        self.settings = settings
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def chunk(self, text: str) -> list[Chunk]:
        chunks = self._splitter.split_text(text)
        return [Chunk(text=chunk, index=idx) for idx, chunk in enumerate(chunks)]
