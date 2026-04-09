import numpy as np
from components.shared_types import RetrievedChunk
from components.ranking.base_ranker import BaseRanker
from components.ranking.scoring_utils import ScoringStrategy, MMRScoring
from langchain_ollama import OllamaEmbeddings

class EmbeddingRanker(BaseRanker):
    def __init__(self, model_name, strategy: ScoringStrategy = MMRScoring(), top_n=5): # TODO: Pass strategy from config
        super().__init__(
            model = OllamaEmbeddings(model=model_name), # FIXME: Make this configurable
            top_n = top_n
        )
        self.strategy = strategy

    def rank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        query_vec = np.array(
            self.model.embed_query(query)
        ).reshape(1, -1)

        doc_vecs = np.array(
            self.model.embed_documents(
                [candidate.text for candidate in candidates]
            )
        )

        return self.strategy.select(query_vec, doc_vecs, candidates, self.top_n)
