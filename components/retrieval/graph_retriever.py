import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from components.retrieval.base_retriever import BaseRetriever, BaseRetrieverSettings
from components.shared_types import RetrievedChunk

class GraphRetrieverSettings(BaseRetrieverSettings):
    _CONFIG_PATH = "retrieval.graph"

    path: str = "data/indices/repo_graph.json"
    max_depth: int = 2
    max_neighbors: int = 20
    score_decay: float = 0.85
    min_score: float = 0.05

class GraphRetriever(BaseRetriever):
    def __init__(self, settings: GraphRetrieverSettings) -> None:
        super().__init__(settings=settings, store=None)
        self.graph_path = Path(settings.path)
        self._graph: dict[str, Any] | None = None
        self._graph_signature: tuple[int, int] | None = None

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        query = query.strip()
        if not query or top_k <= 0:
            return []

        graph = self._load_graph()
        if not graph:
            return []

        chunks_by_id = {
            str(chunk.get("id")): chunk
            for chunk in graph.get("chunks", [])
            if isinstance(chunk, dict) and str(chunk.get("id") or "")
        }
        if not chunks_by_id:
            return []

        tokens = self._tokens(query)
        phrase = query.lower()
        scored_chunks = self._score_chunks(phrase, tokens, chunks_by_id)
        seed_nodes = self._seed_nodes(phrase, tokens, graph)

        if seed_nodes:
            adjacency = self._adjacency(graph)
            traversed_scores = self._traverse(seed_nodes, adjacency)
            for chunk_id, score in traversed_scores.items():
                current = scored_chunks.get(chunk_id, 0.0)
                scored_chunks[chunk_id] = max(current, score)

        ranked = sorted(
            scored_chunks.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        results: list[RetrievedChunk] = []
        for chunk_id, score in ranked:
            if score < float(self.settings.min_score):
                continue

            raw = chunks_by_id.get(chunk_id)
            if not raw:
                continue

            metadata = dict(raw.get("metadata") or {})
            metadata["retrieval_source"] = "graph_retriever"
            metadata["graph_score"] = float(score)
            results.append(
                RetrievedChunk(
                    id=chunk_id,
                    text=str(raw.get("text") or ""),
                    score=float(score),
                    metadata=metadata,
                )
            )
            if len(results) >= top_k:
                break

        return results

    def _load_graph(self) -> dict[str, Any]:
        signature = self._index_signature()
        if self._graph_signature == signature and self._graph is not None:
            return self._graph

        if signature is None:
            self._graph = {}
            self._graph_signature = None
            return self._graph

        try:
            payload = json.loads(self.graph_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            payload = {}

        self._graph = payload if isinstance(payload, dict) else {}
        self._graph_signature = signature
        return self._graph

    def _index_signature(self) -> tuple[int, int] | None:
        if not self.graph_path.exists():
            return None
        
        stat = self.graph_path.stat()
        return int(stat.st_size), int(stat.st_mtime_ns)

    def _seed_nodes(
        self,
        phrase: str,
        tokens: set[str],
        graph: dict[str, Any],
    ) -> dict[str, float]:
        seeds: dict[str, float] = {}
        for node in graph.get("nodes", []):
            if not isinstance(node, dict):
                continue

            node_id = str(node.get("id") or "")
            if not node_id:
                continue

            metadata = dict(node.get("metadata") or {})
            haystack = " ".join(
                [
                    node_id,
                    str(node.get("label") or ""),
                    str(node.get("type") or ""),
                    str(metadata.get("path") or ""),
                    str(metadata.get("module") or ""),
                ]
            ).lower()
            score = self._lexical_score(phrase, tokens, haystack)
            if score > 0:
                seeds[node_id] = score

        return seeds

    def _score_chunks(
        self,
        phrase: str,
        tokens: set[str],
        chunks_by_id: dict[str, dict[str, Any]],
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        for chunk_id, chunk in chunks_by_id.items():
            metadata = dict(chunk.get("metadata") or {})
            haystack = " ".join(
                [
                    str(chunk.get("text") or ""),
                    str(metadata.get("path") or ""),
                    str(metadata.get("relative_path") or ""),
                    str(metadata.get("symbol") or ""),
                    str(metadata.get("title") or ""),
                    str(metadata.get("chunk_type") or ""),
                ]
            ).lower()
            score = self._lexical_score(phrase, tokens, haystack)
            if score > 0:
                scores[chunk_id] = score

        return scores

    def _adjacency(self, graph: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        adjacency: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for edge in graph.get("edges", []):
            if not isinstance(edge, dict):
                continue
            
            source = str(edge.get("source") or "")
            target = str(edge.get("target") or "")
            if not source or not target:
                continue

            adjacency[source].append(edge)

            reverse = dict(edge)
            reverse["source"], reverse["target"] = target, source
            reverse["relation"] = f"REVERSE_{edge.get('relation', '')}"
            adjacency[target].append(reverse)

        return adjacency

    def _traverse(
        self,
        seeds: dict[str, float],
        adjacency: dict[str, list[dict[str, Any]]],
    ) -> dict[str, float]:
        evidence_scores: dict[str, float] = {}
        best_node_scores: dict[str, float] = dict(seeds)
        queue: deque[tuple[str, int, float]] = deque(
            (node_id, 0, score) for node_id, score in seeds.items()
        )

        while queue:
            node_id, depth, node_score = queue.popleft()
            if depth > int(self.settings.max_depth):
                continue

            neighbors = adjacency.get(node_id, [])[: int(self.settings.max_neighbors)]
            for edge in neighbors:
                relation_weight = self._relation_weight(str(edge.get("relation") or ""))
                edge_score = node_score * relation_weight * float(self.settings.score_decay)
                evidence_chunk_id = str(edge.get("evidence_chunk_id") or "")
                if evidence_chunk_id:
                    evidence_scores[evidence_chunk_id] = max(
                        evidence_scores.get(evidence_chunk_id, 0.0),
                        edge_score,
                    )

                target = str(edge.get("target") or "")
                if not target:
                    continue

                previous = best_node_scores.get(target, 0.0)
                if edge_score <= previous:
                    continue

                best_node_scores[target] = edge_score
                queue.append((target, depth + 1, edge_score))

        return evidence_scores

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[A-Za-z0-9_./-]+", text.lower())
            if len(token) > 1
        }

    @staticmethod
    def _lexical_score(phrase: str, tokens: set[str], haystack: str) -> float:
        if not haystack:
            return 0.0

        score = 0.0
        if phrase and phrase in haystack:
            score += 1.0

        if not tokens:
            return score

        matched = sum(1 for token in tokens if token in haystack)
        if matched:
            score += matched / len(tokens)

        return min(score, 2.0)

    @staticmethod
    def _relation_weight(relation: str) -> float:
        weights = {
            "DEFINES": 1.0,
            "TESTS": 0.95,
            "DEFINES_CONFIG": 0.9,
            "CONTAINS": 0.8,
            "IMPORTS": 0.7,
        }

        is_reverse = relation.startswith("REVERSE_")
        normalized = relation.removeprefix("REVERSE_")
        weight = weights.get(normalized, 0.6)
        return weight * 0.8 if is_reverse else weight
