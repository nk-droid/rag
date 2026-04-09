from components.shared_types import RetrievedChunk
from components.ranking.base_ranker import BaseRanker
from sentence_transformers import CrossEncoder

class CrossEncoderRanker(BaseRanker):
    """Score candidates with a cross-encoder model."""

    def __init__(self, model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2", top_n = 3):
        super().__init__(
            model=CrossEncoder(model_name),
            top_n=top_n
        )

    def rank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        pairs = [(query, candidate.text) for candidate in candidates]
        scores = self.model.predict(pairs)

        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:self.top_n]]
