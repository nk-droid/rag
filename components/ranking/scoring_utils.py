from components.shared_types import RetrievedChunk

def normalize_scores(candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
    if not candidates:
        return []
    max_score = max(candidate.score for candidate in candidates) or 1.0
    return [
        RetrievedChunk(
            id=candidate.id,
            text=candidate.text,
            score=candidate.score / max_score,
            metadata=dict(candidate.metadata),
        )
        for candidate in candidates
    ]

def sort_by_score(candidates: list[RetrievedChunk], descending: bool = True) -> list[RetrievedChunk]:
    return sorted(candidates, key=lambda candidate: candidate.score, reverse=descending)

# TODO: Move this
from abc import ABC, abstractmethod
import numpy as np

class ScoringStrategy(ABC):
    @abstractmethod
    def score(self, query_vec, doc_vecs):
        raise NotImplementedError

    @abstractmethod
    def select(self, query_vec, doc_vecs, docs, top_n):
        raise NotImplementedError

from sklearn.metrics.pairwise import cosine_similarity

class CosineScoring(ScoringStrategy):
    def score(self, query_vec, doc_vecs):
        return cosine_similarity(query_vec, doc_vecs)[0]

    def select(self, query_vec, doc_vecs, docs, top_n):
        scores = self.score(query_vec, doc_vecs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_n]]
    
class MMRScoring(ScoringStrategy):
    def __init__(self, lambda_param=0.7):
        self.lambda_param = lambda_param

    def score(self, query_vec, doc_vecs):
        # not directly used for ranking
        return cosine_similarity(query_vec, doc_vecs)[0]

    def select(self, query_vec, doc_vecs, docs, top_n):
        query_sim = cosine_similarity(query_vec, doc_vecs)[0]

        selected_indices = []
        first_idx = int(np.argmax(query_sim))
        selected_indices.append(first_idx)

        while len(selected_indices) < min(top_n, len(docs)):
            mmr_scores = []

            for i in range(len(docs)):
                if i in selected_indices:
                    continue

                relevance = query_sim[i]

                diversity = max(
                    cosine_similarity(
                        doc_vecs[i].reshape(1, -1),
                        doc_vecs[selected_indices]
                    )[0]
                )

                score = self.lambda_param * relevance - (1 - self.lambda_param) * diversity

                mmr_scores.append((i, score))

            next_idx = max(mmr_scores, key=lambda x: x[1])[0]
            selected_indices.append(next_idx)

        return [docs[i] for i in selected_indices]
