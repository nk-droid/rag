from abc import ABC, abstractmethod

from components.ingestion.ingestion_schema import SourceDocument

class BaseLoader(ABC):
    """Common interface for ingestion loaders."""

    @abstractmethod
    def load(self, source: str) -> list[SourceDocument]:
        raise NotImplementedError
