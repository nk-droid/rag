import numpy as np
from components.shared_types import RetrievedChunk
from components.ranking.base_ranker import BaseRanker
from components.ranking.scoring_utils import ScoringStrategy, MMRScoring
from langchain_ollama import OllamaEmbeddings
from infra.cache.cache_keys import text_hash

class EmbeddingRanker(BaseRanker):
    def __init__(
        self,
        model_name,
        strategy: ScoringStrategy = MMRScoring(),
        top_n=5,
        use_cache: bool = True,
    ): # TODO: Pass strategy from config
        resolved_model_name = model_name or "qwen3-embedding:4b"
        super().__init__(
            model = OllamaEmbeddings(model=resolved_model_name), # FIXME: Make this configurable
            top_n = top_n
        )
        self.strategy = strategy
        self.model_name = resolved_model_name
        self.use_cache = use_cache
        self._query_embedding_cache: dict[str, list[float]] = {}
        self._document_embedding_cache: dict[str, list[float]] = {}

    def _query_cache_key(self, query: str) -> str:
        return f"{self.model_name}:query:{query.strip()}"

    def _doc_cache_key(self, candidate: RetrievedChunk) -> str:
        if candidate.id:
            return f"{self.model_name}:doc:{candidate.id}"
        return f"{self.model_name}:doc_hash:{text_hash(candidate.text)}"

    def rank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not candidates:
            return []

        if not self.use_cache:
            query_vec = np.array(self.model.embed_query(query)).reshape(1, -1)
            doc_vecs = np.array(self.model.embed_documents([candidate.text for candidate in candidates]))
            return self.strategy.select(query_vec, doc_vecs, candidates, self.top_n)

        query_key = self._query_cache_key(query)
        if query_key not in self._query_embedding_cache:
            self._query_embedding_cache[query_key] = self.model.embed_query(query)
        query_vec = np.array(self._query_embedding_cache[query_key]).reshape(1, -1)

        missing_doc_keys: list[str] = []
        missing_doc_texts: list[str] = []
        ordered_doc_keys: list[str] = []
        for candidate in candidates:
            doc_key = self._doc_cache_key(candidate)
            ordered_doc_keys.append(doc_key)
            if doc_key not in self._document_embedding_cache:
                missing_doc_keys.append(doc_key)
                missing_doc_texts.append(candidate.text)

        if missing_doc_texts:
            embedded_docs = self.model.embed_documents(missing_doc_texts)
            for key, vector in zip(missing_doc_keys, embedded_docs):
                self._document_embedding_cache[key] = vector

        doc_vecs = np.array(
            [self._document_embedding_cache[key] for key in ordered_doc_keys]
        )

        return self.strategy.select(query_vec, doc_vecs, candidates, self.top_n)
