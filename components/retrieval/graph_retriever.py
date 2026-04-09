from components.shared_types import RetrievedChunk

from components.retrieval.base_retriever import BaseRetriever

class GraphRetriever(BaseRetriever):
    """Retrieve context by traversing graph relationships."""

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        raise NotImplementedError
