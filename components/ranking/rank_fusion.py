from typing import Optional

from components._base import ComponentSettings
from components.shared_types import RetrievedChunk

class RankFusionSettings(ComponentSettings):
    _CONFIG_PATH = "ranking.fusion"

    method: str = "rrf"
    rrf_k: int = 60
    weights: Optional[list[float]] = None
    normalize_output: bool = True

class RankFusion:
    def __init__(self, settings: RankFusionSettings) -> None:
        self.settings = settings

    def fuse(self, result_sets: list[list[RetrievedChunk]]) -> list[RetrievedChunk]:
        if not result_sets:
            return []

        weights = self._resolve_weights(len(result_sets))
        k = max(self.settings.rrf_k, 1)

        scores: dict[str, float] = {}
        best_chunk: dict[str, RetrievedChunk] = {}
        sources: dict[str, list[int]] = {}

        for set_index, result_set in enumerate(result_sets):
            weight = weights[set_index]
            if weight <= 0:
                continue
            for rank, chunk in enumerate(result_set):
                if chunk is None:
                    continue
                key = self._dedup_key(chunk)
                scores[key] = scores.get(key, 0.0) + weight / (k + rank + 1)
                sources.setdefault(key, []).append(set_index)
                if key not in best_chunk:
                    best_chunk[key] = chunk

        return self._materialize(scores, best_chunk, sources)

    def _resolve_weights(self, n: int) -> list[float]:
        configured = self.settings.weights
        if not configured:
            return [1.0] * n
        if len(configured) != n:
            raise ValueError(
                f"RankFusion.weights has {len(configured)} entries but received {n} result sets."
            )
        return [float(w) for w in configured]

    @staticmethod
    def _dedup_key(chunk: RetrievedChunk) -> str:
        return f"id::{chunk.id}" if chunk.id else f"text::{hash(chunk.text)}"

    def _materialize(
        self,
        scores: dict[str, float],
        best_chunk: dict[str, RetrievedChunk],
        sources: dict[str, list[int]],
    ) -> list[RetrievedChunk]:
        ordered_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
        fused: list[RetrievedChunk] = []
        for key in ordered_keys:
            origin = best_chunk[key]
            metadata = dict(origin.metadata)
            metadata["fused_from"] = sorted(set(sources[key]))
            metadata["fusion_score"] = scores[key]
            fused.append(RetrievedChunk(id=origin.id, text=origin.text, score=scores[key], metadata=metadata))

        if self.settings.normalize_output and fused:
            lo = min(c.score for c in fused)
            hi = max(c.score for c in fused)
            span = hi - lo
            if span == 0:
                for c in fused:
                    c.score = 1.0
            else:
                for c in fused:
                    c.score = (c.score - lo) / span
        return fused
