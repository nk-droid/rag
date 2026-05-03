from components.retrieval.base_retriever import BaseRetriever, BaseRetrieverSettings
from components.shared_types import RetrievedChunk

class MemoryRetrieverSettings(BaseRetrieverSettings):
    _CONFIG_PATH = "retrieval.memory"

class MemoryRetriever(BaseRetriever):
    def __init__(self, settings: MemoryRetrieverSettings) -> None:
        super().__init__(settings=settings, store=None)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        raise NotImplementedError
