from abc import ABC, abstractmethod

from components._base import ComponentSettings
from components.shared_types import RetrievedChunk

class BaseRetrieverSettings(ComponentSettings):
    pass

class BaseRetriever(ABC):
    def __init__(self, settings: BaseRetrieverSettings, store: object | None = None) -> None:
        self.settings = settings
        self.store = store

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        raise NotImplementedError
