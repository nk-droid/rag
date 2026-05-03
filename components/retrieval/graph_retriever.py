from components.retrieval.base_retriever import BaseRetriever, BaseRetrieverSettings
from components.shared_types import RetrievedChunk

class GraphRetrieverSettings(BaseRetrieverSettings):
    _CONFIG_PATH = "retrieval.graph"

class GraphRetriever(BaseRetriever):
    def __init__(self, settings: GraphRetrieverSettings) -> None:
        super().__init__(settings=settings, store=None)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        raise NotImplementedError
