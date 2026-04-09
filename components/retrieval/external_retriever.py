from components.shared_types import RetrievedChunk

from components.retrieval.base_retriever import BaseRetriever

class ExternalRetriever(BaseRetriever):
    """Retrieve context from external systems or APIs."""

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        raise NotImplementedError
