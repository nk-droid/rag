from components.ranking.base_ranker import BaseRanker, BaseRankerSettings
from components.shared_types import RetrievedChunk

class ColBERTRankerSettings(BaseRankerSettings):
    _CONFIG_PATH = "ranking.colbert"

class ColBERTRanker(BaseRanker):
    def __init__(self, settings: ColBERTRankerSettings) -> None:
        super().__init__(settings=settings, model=None)

    def rank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        raise NotImplementedError
