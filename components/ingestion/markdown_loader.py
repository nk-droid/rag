from pathlib import Path

from components.ingestion.base_loader import BaseLoader
from components.ingestion.ingestion_schema import SourceDocument

class MarkdownLoader(BaseLoader):
    """Load Markdown files into source documents."""

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
