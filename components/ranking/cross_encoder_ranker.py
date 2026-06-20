from components.ranking.base_ranker import BaseRanker, BaseRankerSettings
from components.shared_types import RetrievedChunk

class CrossEncoderRankerSettings(BaseRankerSettings):
    _CONFIG_PATH = "ranking.cross_encoder"

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 3

class CrossEncoderRanker(BaseRanker):
    def __init__(self, settings: CrossEncoderRankerSettings) -> None:
        super().__init__(settings=settings, model=None)

    def _load_model(self):
        if self.model is None:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.settings.model_name)
        return self.model

    def rank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not candidates:
            return []
        model = self._load_model()
        pairs = [(query, candidate.text) for candidate in candidates]
        scores = model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[: self.top_n]]
