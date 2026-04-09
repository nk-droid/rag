from pathlib import Path

from components.ingestion.document_loader import DocumentLoader
from components.ingestion.ingestion_schema import SourceDocument

class DirectoryLoader:
    """Walk a directory and load supported files."""

    def __init__(self, loader: DocumentLoader | None = None, recursive: bool = True) -> None:
        self.loader = loader or DocumentLoader()
        self.recursive = recursive

    def load(self, source: str) -> list[SourceDocument]:
        root = Path(source)
        pattern = "**/*" if self.recursive else "*"
        documents: list[SourceDocument] = []

        for path in sorted(root.glob(pattern)):
            if not path.is_file():
                continue
            documents.extend(self.loader.load(str(path)))

        return documents
    
if __name__ == "__main__":
    loader = DirectoryLoader()
    data = loader.load("/Users/nidhishkumar/Personal/rag/data/raw/docs")
    print(data)
    print(len(data))
