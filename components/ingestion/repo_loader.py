from typing import Any
from pathlib import Path

from components._base import ComponentSettings
from components.ingestion.code_loader import CodeLoader, CodeLoaderSettings
from components.ingestion.ingestion_schema import SourceDocument
from components.ingestion.repo_file_filter import RepoFileFilter, RepoFileFilterSettings

class RepoLoaderSettings(ComponentSettings):
    _CONFIG_PATH = "ingestion.repo"

class RepoLoader:
    def __init__(
        self, 
        settings: RepoLoaderSettings,
        file_filter: RepoFileFilter | None = None,
        code_loader: CodeLoader | None = None
    ) -> None:
        self.settings = settings
        self.file_filter = file_filter or RepoFileFilter(RepoFileFilterSettings())
        self.code_loader = code_loader or CodeLoader(CodeLoaderSettings())
    
    def load(self, source: str, metadata: dict[str, Any] | None = None) -> list[SourceDocument]:
        repo_root = Path(source).resolve()
        source_metadata =  metadata or {}
        documents: list[SourceDocument] = []

        for path in self.file_filter.iter_files(repo_root):
            relative_path = path.relative_to(repo_root).as_posix()
            for document in self.code_loader.load(str(path)):
                document.source = relative_path
                document.metadata.update(source_metadata)
                document.metadata.update({
                    "loader": self.__class__.__name__,
                    "source_type": "repo_code",
                    "repo_root": str(repo_root),
                    "path": relative_path,
                    "relative_path": relative_path,
                })
                documents.append(document)

        return documents
