from components.shared_types import RetrievedChunk
from components.ranking.base_ranker import BaseRanker

class ColBERTRanker(BaseRanker):
    """Score candidates with a ColBERT-style late interaction model."""

    def rank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        raise NotImplementedError
