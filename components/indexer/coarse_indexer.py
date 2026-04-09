import hashlib
import json
from pathlib import Path
from typing import Any

from components.indexer.indexer_schema import IndexRecord

class CoarseIndexer:
    def __init__(self, index_path: str | Path = "data/indexes/coarse_index.json") -> None:
        self.index_path = Path(index_path)

    def index(
        self,
        chunks: list[Any],
        config: dict[str, Any],
        vector_db_path: str | Path | None = None,
    ) -> list[IndexRecord]:
        if not chunks:
            return []

        path = Path(vector_db_path) if vector_db_path else Path(
            config.get("coarse_index", {}).get("path", str(self.index_path))
        )

        docs: list[dict[str, Any]] = []
        records: list[IndexRecord] = []

        for i, chunk in enumerate(chunks):
            text = self._chunk_text(chunk).strip()
            if not text:
                continue

            metadata = self._chunk_metadata(chunk)
            doc_id = self._record_id(text, metadata, i)

            docs.append({"id": doc_id, "text": text, "metadata": metadata})
            records.append(IndexRecord(id=doc_id, text=text, embedding=[], metadata=metadata))

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"version": 1, "documents": docs}, ensure_ascii=False, indent=4), encoding="utf-8")
        return records

    def _chunk_text(self, chunk: Any) -> str:
        if hasattr(chunk, "text"):
            return str(chunk.text)
        if isinstance(chunk, dict):
            return str(chunk.get("text", ""))
        if isinstance(chunk, str):
            return chunk
        return ""

    def _chunk_metadata(self, chunk: Any) -> dict[str, Any]:
        if hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
            return dict(chunk.metadata)
        if isinstance(chunk, dict) and isinstance(chunk.get("metadata"), dict):
            return dict(chunk["metadata"])
        return {}

    def _record_id(self, text: str, metadata: dict[str, Any], index: int) -> str:
        source = metadata.get("source", "unknown")
        source_index = metadata.get("source_index", 0)
        chunk_index = metadata.get("chunk_index", index)
        fingerprint = hashlib.sha1(f"{source}:{text}".encode("utf-8")).hexdigest()[:12]
        return f"{source_index}:{chunk_index}:{source}:{fingerprint}"
