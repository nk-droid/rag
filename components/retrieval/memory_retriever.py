from components.shared_types import RetrievedChunk

from components.retrieval.base_retriever import BaseRetriever

class MemoryRetriever(BaseRetriever):
    """Retrieve context from conversation or long-term memory."""

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        raise NotImplementedError
