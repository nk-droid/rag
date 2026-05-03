import hashlib
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from components._base import ComponentSettings
from components.indexer.indexer_schema import IndexRecord

class EmbeddingIndexerSettings(ComponentSettings):
    _CONFIG_PATH = "indexers.embedding"

    path: str = "data/indices/faiss_index"
    vector_store: dict[str, Any] = {"provider": "faiss"}

class EmbeddingIndexer:
    def __init__(self, settings: EmbeddingIndexerSettings, vector_store: object) -> None:
        self.settings = settings
        self.vector_db_path = Path(settings.path)
        self._vector_store = vector_store

    def index(self, chunks: list[Any]) -> list[IndexRecord]:
        if not chunks:
            return []

        documents: list[Document] = []
        ids: list[str] = []
        records: list[IndexRecord] = []

        for index, chunk in enumerate(chunks):
            text = self._chunk_text(chunk)
            if not text:
                continue

            metadata = self._chunk_metadata(chunk)
            record_id = self._record_id(text, metadata, index)
            ids.append(record_id)
            documents.append(Document(page_content=text, metadata=metadata))
            records.append(IndexRecord(id=record_id, text=text, embedding=[], metadata=metadata))

        if not documents:
            return []

        self._vector_store.add_documents(documents=documents, ids=ids)
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
