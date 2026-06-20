from typing import Literal

import numpy as np
from langchain_ollama import OllamaEmbeddings

from components.ranking.base_ranker import BaseRanker, BaseRankerSettings
from components.ranking.scoring_utils import CosineScoring, MMRScoring, ScoringStrategy
from components.shared_types import RetrievedChunk
from infra.cache.cache_keys import text_hash

class EmbeddingRankerSettings(BaseRankerSettings):
    _CONFIG_PATH = "ranking.embedding"

    model_name: str = "qwen3-embedding:0.6b"
    top_n: int = 5
    use_cache: bool = True
    strategy: Literal["mmr", "cosine"] = "mmr"
    lambda_param: float = 0.7

def _build_strategy(settings: EmbeddingRankerSettings) -> ScoringStrategy:
    if settings.strategy == "mmr":
        return MMRScoring(lambda_param=settings.lambda_param)
    if settings.strategy == "cosine":
        return CosineScoring()
    return MMRScoring(lambda_param=settings.lambda_param)

class EmbeddingRanker(BaseRanker):
    def __init__(self, settings: EmbeddingRankerSettings) -> None:
        super().__init__(
            settings=settings,
            model=OllamaEmbeddings(model=settings.model_name),
        )
        self.strategy = _build_strategy(settings)
        self._query_embedding_cache: dict[str, list[float]] = {}
        self._document_embedding_cache: dict[str, list[float]] = {}

    def _query_cache_key(self, query: str) -> str:
        return f"{self.settings.model_name}:query:{query.strip()}"

    def _doc_cache_key(self, candidate: RetrievedChunk) -> str:
        if candidate.id:
            return f"{self.settings.model_name}:doc:{candidate.id}"
        return f"{self.settings.model_name}:doc_hash:{text_hash(candidate.text)}"

    def rank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not candidates:
            return []

        if not self.settings.use_cache:
            query_vec = np.array(self.model.embed_query(query)).reshape(1, -1)
            doc_vecs = np.array(self.model.embed_documents([c.text for c in candidates]))
            return self.strategy.select(query_vec, doc_vecs, candidates, self.top_n)

        query_key = self._query_cache_key(query)
        if query_key not in self._query_embedding_cache:
            self._query_embedding_cache[query_key] = self.model.embed_query(query)
        query_vec = np.array(self._query_embedding_cache[query_key]).reshape(1, -1)

        missing_keys: list[str] = []
        missing_texts: list[str] = []
        ordered_keys: list[str] = []
        for candidate in candidates:
            key = self._doc_cache_key(candidate)
            ordered_keys.append(key)
            if key not in self._document_embedding_cache:
                missing_keys.append(key)
                missing_texts.append(candidate.text)

        if missing_texts:
            for key, vec in zip(missing_keys, self.model.embed_documents(missing_texts)):
                self._document_embedding_cache[key] = vec

        doc_vecs = np.array([self._document_embedding_cache[k] for k in ordered_keys])
        return self.strategy.select(query_vec, doc_vecs, candidates, self.top_n)
