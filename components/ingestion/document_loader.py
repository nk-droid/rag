from pathlib import Path

from components._base import ComponentSettings
from components.ingestion.ingestion_schema import SourceDocument
from components.ingestion.markdown_loader import MarkdownLoader
from components.ingestion.text_loader import TextLoader

class DocumentLoaderSettings(ComponentSettings):
    _CONFIG_PATH = "ingestion.document"

class DocumentLoader:
    def __init__(
        self,
        settings: DocumentLoaderSettings,
        markdown_loader: MarkdownLoader,
        text_loader: TextLoader,
    ) -> None:
        self.settings = settings
        self.markdown_loader = markdown_loader
        self.text_loader = text_loader

    def load(self, source: str) -> list[SourceDocument]:
        path = Path(source)

        if path.suffix.lower() in {".md", ".markdown"}:
            return self.markdown_loader.load(source)

        if path.suffix.lower() in {".txt", ".log"}:
            return self.text_loader.load(source)

        return []
