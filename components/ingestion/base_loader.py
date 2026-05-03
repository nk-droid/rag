from abc import ABC, abstractmethod

from components.ingestion.ingestion_schema import SourceDocument

class BaseLoader(ABC):
    @abstractmethod
    def load(self, source: str) -> list[SourceDocument]:
        raise NotImplementedError
