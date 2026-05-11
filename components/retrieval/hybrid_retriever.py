from components.retrieval.base_retriever import BaseRetriever, BaseRetrieverSettings
from components.retrieval.coarse_retriever import CoarseRetriever
from components.retrieval.fine_retriever import FineRetriever
from components.shared_types import RetrievedChunk

class HybridRetrieverSettings(BaseRetrieverSettings):
    _CONFIG_PATH = "retrieval.hybrid"

    candidate_multiplier: int = 4
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
        candidates = self.retrieve_candidates(query, top_k=top_k)
        sparse = self._normalize(candidates["sparse"])
        dense = self._normalize(candidates["dense"])

        merged: dict[str, RetrievedChunk] = {}
        for chunk, weight in self._weighted_stream(sparse, dense):
            key = chunk.id or f"text::{hash(chunk.text)}"
            weighted_score = chunk.score * weight
            incumbent = merged.get(key)
            if incumbent is None:
                merged[key] = RetrievedChunk(
                    id=chunk.id, text=chunk.text, score=weighted_score, metadata=dict(chunk.metadata)
                )
            elif weighted_score > incumbent.score:
                incumbent.score = weighted_score
                incumbent.metadata.update(chunk.metadata)

        return sorted(merged.values(), key=lambda c: c.score, reverse=True)[:top_k]

    def _weighted_stream(self, sparse: list[RetrievedChunk], dense: list[RetrievedChunk]):
        for chunk in sparse:
            yield chunk, self.settings.sparse_weight
        for chunk in dense:
            yield chunk, self.settings.dense_weight

    @staticmethod
    def _normalize(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []
        scores = [c.score for c in chunks]
        lo, hi = min(scores), max(scores)
        span = hi - lo
        if span == 0:
            return [RetrievedChunk(id=c.id, text=c.text, score=1.0, metadata=dict(c.metadata)) for c in chunks]
        return [
            RetrievedChunk(id=c.id, text=c.text, score=(c.score - lo) / span, metadata=dict(c.metadata))
            for c in chunks
        ]