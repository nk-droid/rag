import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from components._base import ComponentSettings
from components.indexer.indexer_schema import IndexRecord

class RepoGraphIndexerSettings(ComponentSettings):
    _CONFIG_PATH = "indexers.graph"

    path: str = "data/indices/repo_graph.json"

class RepoGraphIndexer:
    def __init__(self, settings: RepoGraphIndexerSettings) -> None:
        self.settings = settings
        self.index_path = Path(settings.path)

    def index(self, chunks: list[Any]) -> list[IndexRecord]:
        nodes: dict[str, dict[str, Any]] = {}
        edges: dict[tuple[str, str, str], dict[str, Any]] = {}
        chunk_store: dict[str, dict[str, Any]] = {}

        def add_node(node_id: str, node_type: str, label: str, metadata: dict[str, Any] | None = None) -> None:
            if node_id not in nodes:
                nodes[node_id] = {"id": node_id, "type": node_type, "label": label, "metadata": metadata or {}}
            
            elif metadata:
                nodes[node_id]["metadata"].update(metadata)

        def add_edge(
            source: str,
            relation: str,
            target: str,
            evidence_chunk_id: str | None,
            metadata: dict[str, Any] | None = None,
        ) -> None:
            key = (source, relation, target)
            if key not in edges:
                edges[key] = {
                    "source": source,
                    "relation": relation,
                    "target": target,
                    "evidence_chunk_id": evidence_chunk_id,
                    "metadata": metadata or {},
                }

        records: list[IndexRecord] = []
        by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for index, chunk in enumerate(chunks):
            text = self._chunk_text(chunk).strip()
            if not text:
                continue

            metadata = self._chunk_metadata(chunk)
            path = str(metadata.get("relative_path") or metadata.get("path") or metadata.get("source") or "unknown")
            source_id = str(metadata.get("source_id") or metadata.get("repo_name") or "repo")
            chunk_id = str(metadata.get("chunk_id") or self._chunk_id(path, text, index))
            metadata["chunk_id"] = chunk_id
            metadata.setdefault("path", path)

            chunk_store[chunk_id] = {"id": chunk_id, "text": text, "metadata": metadata}
            by_file[path].append({"text": text, "metadata": metadata, "chunk_id": chunk_id})

            repo_node = f"repo:{source_id}"
            file_node = f"file:{path}"
            add_node(repo_node, "Repository", source_id, {"source_id": source_id})
            add_node(file_node, "File", path, {"path": path, "source_id": source_id})
            add_edge(repo_node, "CONTAINS", file_node, chunk_id)

            symbol = metadata.get("symbol")
            if symbol:
                symbol_text = str(symbol)
                symbol_node = f"symbol:{path}:{symbol_text}"
                node_type = self._symbol_node_type(str(metadata.get("chunk_type") or "symbol"))
                add_node(
                    symbol_node,
                    node_type,
                    symbol_text,
                    {
                        "path": path,
                        "source_id": source_id,
                        "start_line": metadata.get("start_line"),
                        "end_line": metadata.get("end_line"),
                    },
                )

                add_edge(file_node, "DEFINES", symbol_node, chunk_id)

                if "." in symbol_text:
                    parent = symbol_text.split(".", 1)[0]
                    parent_node = f"symbol:{path}:{parent}"
                    add_node(parent_node, "Class", parent, {"path": path, "source_id": source_id})
                    add_edge(parent_node, "DEFINES", symbol_node, chunk_id)

            for module in self._extract_imports(text):
                module_node = f"module:{module}"
                add_node(module_node, "Module", module, {"module": module})
                add_edge(file_node, "IMPORTS", module_node, chunk_id)

            for config_key in self._extract_config_keys(text, path):
                key_node = f"config:{path}:{config_key}"
                add_node(key_node, "ConfigKey", config_key, {"path": path, "source_id": source_id})
                add_edge(file_node, "DEFINES_CONFIG", key_node, chunk_id)

            records.append(IndexRecord(id=chunk_id, text=text, embedding=[], metadata=metadata))

        self._add_test_edges(nodes, edges, by_file)

        payload = {
            "version": 1,
            "nodes": list(nodes.values()),
            "edges": list(edges.values()),
            "chunks": list(chunk_store.values()),
        }
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return records

    @staticmethod
    def _chunk_text(chunk: Any) -> str:
        if hasattr(chunk, "text"):
            return str(chunk.text)
        
        if isinstance(chunk, dict):
            return str(chunk.get("text", ""))
        
        if isinstance(chunk, str):
            return chunk
        
        return ""

    @staticmethod
    def _chunk_metadata(chunk: Any) -> dict[str, Any]:
        if hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
            return dict(chunk.metadata)
        
        if isinstance(chunk, dict) and isinstance(chunk.get("metadata"), dict):
            return dict(chunk["metadata"])
        
        return {}

    @staticmethod
    def _chunk_id(path: str, text: str, index: int) -> str:
        digest = hashlib.sha1(f"{path}:{index}:{text}".encode("utf-8")).hexdigest()[:16]
        return f"chunk:{digest}"

    @staticmethod
    def _symbol_node_type(chunk_type: str) -> str:
        mapping = {"class": "Class", "method": "Method", "function": "Function"}
        return mapping.get(chunk_type, "Symbol")

    @staticmethod
    def _extract_imports(text: str) -> list[str]:
        modules: set[str] = set()
        for line in text.splitlines():
            stripped = line.strip()
            match = re.match(r"^(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", stripped)
            if match:
                module = match.group(1) or match.group(2)
                if module:
                    modules.add(module.split(".")[0] if not module.startswith(".") else module)
        
        return sorted(modules)

    @staticmethod
    def _extract_config_keys(text: str, path: str) -> list[str]:
        suffix = Path(path).suffix.lower()
        if suffix not in {".yaml", ".yml", ".json", ".toml"} and Path(path).name not in {"Dockerfile", "Makefile"}:
            return []

        keys: set[str] = set()
        for line in text.splitlines():
            stripped = line.strip().strip('"')
            if not stripped or stripped.startswith(("#", "//")):
                continue

            yaml_match = re.match(r"^([A-Za-z0-9_.-]+)\s*:", stripped)
            toml_match = re.match(r"^([A-Za-z0-9_.-]+)\s*=", stripped)
            section_match = re.match(r"^\[+([A-Za-z0-9_.-]+)\]+", stripped)
            value = None
            if yaml_match:
                value = yaml_match.group(1)

            elif toml_match:
                value = toml_match.group(1)

            elif section_match:
                value = section_match.group(1)

            if value:
                keys.add(value)

        return sorted(keys)[:100]

    @staticmethod
    def _add_test_edges(
        nodes: dict[str, dict[str, Any]],
        edges: dict[tuple[str, str, str], dict[str, Any]],
        by_file: dict[str, list[dict[str, Any]]],
    ) -> None:
        symbol_nodes_by_name: dict[str, list[str]] = defaultdict(list)
        for node_id, node in nodes.items():
            if node.get("type") in {"Class", "Method", "Function", "Symbol"}:
                label = node.get("label", "")
                symbol_nodes_by_name[label.lower()].append(node_id)
                symbol_nodes_by_name[label.split(".")[-1].lower()].append(node_id)

        for path, chunks in by_file.items():
            name = Path(path).name.lower()
            if not (name.startswith("test_") or name.endswith("_test.py") or "/tests/" in f"/{path}"):
                continue

            file_node = f"file:{path}"
            combined_text = "\n".join(str(item["text"]) for item in chunks)
            evidence_chunk_id = str(chunks[0].get("chunk_id")) if chunks else None
            for candidate in re.findall(r"test_([A-Za-z0-9_]+)", combined_text):
                cleaned = candidate.lower()
                for symbol_node in symbol_nodes_by_name.get(cleaned, [])[:5]:
                    key = (file_node, "TESTS", symbol_node)
                    edges.setdefault(
                        key,
                        {
                            "source": file_node,
                            "relation": "TESTS",
                            "target": symbol_node,
                            "evidence_chunk_id": evidence_chunk_id,
                            "metadata": {"heuristic": "test_name_match"},
                        },
                    )