from components.retrieval.base_retriever import BaseRetriever, BaseRetrieverSettings
from components.retrieval.coarse_retriever import CoarseRetriever
from components.retrieval.fine_retriever import FineRetriever
from components.shared_types import RetrievedChunk

class HybridRetrieverSettings(BaseRetrieverSettings):
    _CONFIG_PATH = "retrieval.hybrid"

    candidate_multiplier: int = 4
    rrf_k: int = 60
    sparse_weight: float = 0.45
    dense_weight: float = 0.55
    fuse: bool = False

class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        settings: HybridRetrieverSettings,
        dense_retriever: FineRetriever | None = None,
        sparse_retriever: CoarseRetriever | None = None,
    ) -> None:
        super().__init__(settings=settings, store=None)
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever

    @property
    def candidate_multiplier(self) -> int:
        return max(1, int(self.settings.candidate_multiplier))

    def retrieve_candidates(self, query: str, top_k: int = 5) -> dict[str, list[RetrievedChunk]]:
        candidate_k = max(top_k, top_k * self.candidate_multiplier)
        sparse = self.sparse_retriever.retrieve(query, top_k=candidate_k) if self.sparse_retriever else []
        dense = self.dense_retriever.retrieve(query, top_k=candidate_k) if self.dense_retriever else []
        return {"sparse": sparse, "dense": dense}

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        c = self.retrieve_candidates(query, top_k=top_k)
        return c["sparse"] + c["dense"]
