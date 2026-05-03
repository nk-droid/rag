from pathlib import Path

from components._base import ComponentSettings
from components.ingestion.document_loader import DocumentLoader
from components.ingestion.ingestion_schema import SourceDocument

class DirectoryLoaderSettings(ComponentSettings):
    _CONFIG_PATH = "ingestion.directory"

    recursive: bool = True

class DirectoryLoader:
    def __init__(self, settings: DirectoryLoaderSettings, loader: DocumentLoader) -> None:
        self.settings = settings
        self.loader = loader

    def load(self, source: str) -> list[SourceDocument]:
        root = Path(source)
        pattern = "**/*" if self.settings.recursive else "*"
        documents: list[SourceDocument] = []

        for path in sorted(root.glob(pattern)):
            if not path.is_file():
                continue
            documents.extend(self.loader.load(str(path)))

        return documents
