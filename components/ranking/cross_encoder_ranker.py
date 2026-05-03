from sentence_transformers import CrossEncoder

from components.ranking.base_ranker import BaseRanker, BaseRankerSettings
from components.shared_types import RetrievedChunk

class CrossEncoderRankerSettings(BaseRankerSettings):
    _CONFIG_PATH = "ranking.cross_encoder"

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 3

class CrossEncoderRanker(BaseRanker):
    def __init__(self, settings: CrossEncoderRankerSettings) -> None:
        super().__init__(
            settings=settings,
            model=CrossEncoder(settings.model_name),
        )

    def rank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        pairs = [(query, candidate.text) for candidate in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[: self.top_n]]
