from abc import ABC, abstractmethod

from components._base import ComponentSettings
from components.shared_types import RetrievedChunk

class BaseRankerSettings(ComponentSettings):
    top_n: int = 5

class BaseRanker(ABC):
    def __init__(self, settings: BaseRankerSettings, model: object | None = None) -> None:
        self.settings = settings
        self.model = model

    @property
    def top_n(self) -> int:
        return self.settings.top_n

    @abstractmethod
    def rank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        raise NotImplementedError
