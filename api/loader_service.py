from pathlib import Path

from components.ingestion.directory_loader import DirectoryLoader, DirectoryLoaderSettings
from components.ingestion.document_loader import DocumentLoader, DocumentLoaderSettings
from components.ingestion.ingestion_schema import SourceDocument
from components.ingestion.markdown_loader import MarkdownLoader, MarkdownLoaderSettings
from components.ingestion.repo_loader import RepoLoader, RepoLoaderSettings
from components.ingestion.text_loader import TextLoader, TextLoaderSettings

from api.schemas import SourceRecord

class LoaderService:
    def __init__(self) -> None:
        self._markdown_loader = MarkdownLoader(MarkdownLoaderSettings())
        self._text_loader = TextLoader(TextLoaderSettings())
        self._document_loader = DocumentLoader(
            DocumentLoaderSettings(),
            markdown_loader=self._markdown_loader,
            text_loader=self._text_loader,
        )
        self._directory_loader = DirectoryLoader(
            DirectoryLoaderSettings(),
            loader=self._document_loader,
        )
        self._repo_loader = RepoLoader(RepoLoaderSettings())

    def load_sources(self, sources: list[SourceRecord]) -> list[SourceDocument]:
        documents: list[SourceDocument] = []

        for source in sources:
            path = Path(source.path)
            if not path.exists():
                continue

            loaded = self._load_single(source)
            for item in loaded:
                item.metadata.setdefault("source_id", source.id)
                item.metadata.setdefault("source_name", source.name)
                item.metadata.setdefault("source_type", source.source_type)
            documents.extend(loaded)

        return documents

    def _load_single(self, source: SourceRecord) -> list[SourceDocument]:
        if source.loader == "markdown_loader":
            return self._markdown_loader.load(source.path)

        if source.loader == "text_loader":
            return self._text_loader.load(source.path)

        if source.loader == "directory_loader":
            return self._directory_loader.load(source.path)

        if source.loader == "document_loader":
            return self._document_loader.load(source.path)

        if source.loader == "repo_loader":
            return self._repo_loader.load(
                source.path,
                metadata={
                    "source_id": source.id,
                    "source_name": source.name,
                    "repo_url": source.repo_url,
                    "branch": source.branch,
                    "commit_sha": source.commit_sha,
                },
            )

        return []
