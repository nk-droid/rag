from pathlib import Path

from components.ingestion.ingestion_schema import SourceDocument
from components.ingestion.markdown_loader import MarkdownLoader
from components.ingestion.text_loader import TextLoader

class DocumentLoader:
    """Route supported sources to the appropriate file loader."""

    def __init__(
        self,
        markdown_loader: MarkdownLoader | None = None,
        text_loader: TextLoader | None = None,
    ) -> None:
        self.markdown_loader = markdown_loader or MarkdownLoader()
        self.text_loader = text_loader or TextLoader()

    def load(self, source: str) -> list[SourceDocument]:
        path = Path(source)

        if path.suffix.lower() in {".md", ".markdown"}:
            return self.markdown_loader.load(source)

        if path.suffix.lower() in {".txt", ".log"}:
            return self.text_loader.load(source)

        return []
