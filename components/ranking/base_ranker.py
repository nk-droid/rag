from abc import ABC, abstractmethod
from components.shared_types import RetrievedChunk

class BaseRanker(ABC):
    """Common interface for rerankers."""

    def __init__(self, model: object | None = None, top_n: int = None) -> None:
        self.model = model
        self.top_n = top_n

    @abstractmethod
    def rank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        raise NotImplementedError
