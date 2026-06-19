import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from components._base import ComponentSettings
from components.shared_types import RetrievedChunk

class GraphExpanderSettings(ComponentSettings):
    _CONFIG_PATH = "retrieval.graph"

    path: str = "data/indices/repo_graph.json"
    max_depth: int = 2
    max_neighbors: int = 20
    max_expanded_chunks: int = 20
    score_decay: float = 0.85

class GraphExpander:
    def __init__(self, settings: GraphExpanderSettings) -> None:
        self.settings = settings
        self.graph_path = Path(settings.path)
        self._graph: dict[str, Any] | None = None

    def expand(self, chunks: list[RetrievedChunk], top_k: int | None = None) -> list[RetrievedChunk]:
        graph = self._load_graph()
        if not graph or not chunks:
            return []

        nodes_by_id = {str(node.get("id")): node for node in graph.get("nodes", []) if isinstance(node, dict)}
        chunks_by_id = {str(chunk.get("id")): chunk for chunk in graph.get("chunks", []) if isinstance(chunk, dict)}

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

        seeds = self._seed_nodes(chunks, nodes_by_id, graph)
        if not seeds:
            return []

        evidence_ids = self._traverse(seeds, adjacency)
        original_ids = {self._chunk_id(chunk) for chunk in chunks}
        max_chunks = int(top_k or self.settings.max_expanded_chunks)

        expanded: list[RetrievedChunk] = []
        for rank, chunk_id in enumerate(evidence_ids):
            if chunk_id in original_ids:
                continue

            raw = chunks_by_id.get(chunk_id)
            if not raw:
                continue

            metadata = dict(raw.get("metadata") or {})
            metadata["retrieval_source"] = "graph_expander"
            metadata["graph_expanded"] = True
            expanded.append(
                RetrievedChunk(
                    id=chunk_id,
                    text=str(raw.get("text") or ""),
                    score=max(0.0, 1.0 - (rank * 0.01)) * float(self.settings.score_decay),
                    metadata=metadata,
                )
            )
            if len(expanded) >= max_chunks:
                break

        return expanded

    def _load_graph(self) -> dict[str, Any]:
        if self._graph is not None:
            return self._graph
        
        if not self.graph_path.exists():
            self._graph = {}
            return self._graph
        
        try:
            self._graph = json.loads(self.graph_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._graph = {}

        return self._graph

    def _seed_nodes(
        self,
        chunks: list[RetrievedChunk],
        nodes_by_id: dict[str, dict[str, Any]],
        graph: dict[str, Any],
    ) -> list[str]:
        seeds: list[str] = []
        chunk_to_nodes: dict[str, set[str]] = defaultdict(set)
        path_to_nodes: dict[str, set[str]] = defaultdict(set)
        symbol_to_nodes: dict[str, set[str]] = defaultdict(set)

        for node_id, node in nodes_by_id.items():
            metadata = dict(node.get("metadata") or {})
            path = str(metadata.get("path") or "")
            label = str(node.get("label") or "")
            if path:
                path_to_nodes[path].add(node_id)

            if label:
                symbol_to_nodes[label].add(node_id)

        for edge in graph.get("edges", []):
            if not isinstance(edge, dict):
                continue

            evidence = str(edge.get("evidence_chunk_id") or "")
            if evidence:
                chunk_to_nodes[evidence].add(str(edge.get("source")))
                chunk_to_nodes[evidence].add(str(edge.get("target")))

        for chunk in chunks:
            metadata = dict(chunk.metadata or {})
            chunk_id = self._chunk_id(chunk)
            path = str(metadata.get("relative_path") or metadata.get("path") or "")
            symbol = str(metadata.get("symbol") or "")

            seeds.extend(sorted(chunk_to_nodes.get(chunk_id, set())))
            seeds.extend(sorted(path_to_nodes.get(path, set())))
            if symbol:
                seeds.extend(sorted(symbol_to_nodes.get(symbol, set())))

        return list(dict.fromkeys(item for item in seeds if item))

    def _traverse(self, seeds: list[str], adjacency: dict[str, list[dict[str, Any]]]) -> list[str]:
        visited: set[str] = set()
        evidence: list[str] = []
        queue: deque[tuple[str, int]] = deque((seed, 0) for seed in seeds)

        while queue:
            node_id, depth = queue.popleft()
            if node_id in visited or depth > int(self.settings.max_depth):
                continue

            visited.add(node_id)

            neighbors = adjacency.get(node_id, [])[: int(self.settings.max_neighbors)]
            for edge in neighbors:
                evidence_chunk_id = str(edge.get("evidence_chunk_id") or "")
                if evidence_chunk_id and evidence_chunk_id not in evidence:
                    evidence.append(evidence_chunk_id)
                    
                target = str(edge.get("target") or "")
                if target and target not in visited:
                    queue.append((target, depth + 1))

        return evidence

    @staticmethod
    def _chunk_id(chunk: RetrievedChunk) -> str:
        metadata = dict(chunk.metadata or {})
        return str(metadata.get("chunk_id") or chunk.id or "")