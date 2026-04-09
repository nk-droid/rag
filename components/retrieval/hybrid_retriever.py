from components.shared_types import RetrievedChunk
from components.retrieval.base_retriever import BaseRetriever

class HybridRetriever(BaseRetriever):
    """Combine multiple retrieval strategies."""
    def __init__(self, dense_retriever=None, sparse_retriever=None, candidate_multiplier: int = 4):
        super().__init__(store=None)
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.candidate_multiplier = max(1, int(candidate_multiplier))

    def retrieve_candidates(self, query: str, top_k: int = 5) -> dict[str, list[RetrievedChunk]]:
        candidate_k = max(top_k, top_k * self.candidate_multiplier)
        sparse = self.sparse_retriever.retrieve(query, top_k=candidate_k) if self.sparse_retriever else []
        dense = self.dense_retriever.retrieve(query, top_k=candidate_k) if self.dense_retriever else []
        return {"sparse": sparse, "dense": dense}

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        c = self.retrieve_candidates(query, top_k=top_k)
        return c["sparse"] + c["dense"]
