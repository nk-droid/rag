from pathlib import Path

from components._base import ComponentSettings
from components.ingestion.base_loader import BaseLoader
from components.ingestion.ingestion_schema import SourceDocument

class TextLoaderSettings(ComponentSettings):
    _CONFIG_PATH = "ingestion.text"

class TextLoader(BaseLoader):
    def __init__(self, settings: TextLoaderSettings) -> None:
        self.settings = settings

    def load(self, source: str) -> list[SourceDocument]:
        path = Path(source)
        text = path.read_text(encoding="utf-8")
        return [
            SourceDocument(
                text=text,
                source=str(path),
                metadata={"loader": self.__class__.__name__, "suffix": path.suffix},
            )
        ]
