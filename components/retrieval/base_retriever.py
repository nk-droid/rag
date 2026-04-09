from abc import ABC, abstractmethod
from components.shared_types import RetrievedChunk

class BaseRetriever(ABC):
    """Common interface for retrievers."""

    def __init__(self, store: object | None = None) -> None:
        self.store = store

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        raise NotImplementedError
