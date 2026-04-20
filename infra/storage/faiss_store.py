from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

class LangChainFAISSStore:
    """Small wrapper around LangChain FAISS persistence."""

    def __init__(self, embeddings, persist_path: str | Path = "data/embeddings/faiss_index") -> None:
        self.embeddings = embeddings
        self.persist_path = Path(persist_path)
        self._cached_store = None
        self._cached_signature: tuple[tuple[str, bool, int | None, int | None], ...] | None = None

    def _index_signature(self) -> tuple[tuple[str, bool, int | None, int | None], ...] | None:
        if not self.persist_path.exists():
            return None

        artifact_paths = (
            self.persist_path / "index.faiss",
            self.persist_path / "index.pkl",
        )
        signature: list[tuple[str, bool, int | None, int | None]] = []
        for artifact in artifact_paths:
            if artifact.exists():
                stat = artifact.stat()
                signature.append((str(artifact), True, int(stat.st_size), int(stat.st_mtime_ns)))
            else:
                signature.append((str(artifact), False, None, None))
        return tuple(signature)

    def _load_or_create(self):
        current_signature = self._index_signature()
        if (
            self._cached_store is not None
            and self._cached_signature is not None
            and self._cached_signature == current_signature
        ):
            return self._cached_store

        if current_signature is None:
            self._cached_store = None
            self._cached_signature = None
            return None

        self._cached_store = FAISS.load_local(
            str(self.persist_path),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        self._cached_signature = current_signature
        return self._cached_store

    def add_documents(self, documents: list[Document], ids: list[str] | None = None):
        if not documents:
            return None

        vector_store = self._load_or_create()
        if vector_store is None:
            vector_store = FAISS.from_documents(documents, self.embeddings, ids=ids)
        else:
            vector_store.add_documents(documents, ids=ids)

        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(self.persist_path))
        self._cached_store = vector_store
        self._cached_signature = self._index_signature()
        return vector_store

    def similarity_search_with_score(self, query: str, k: int = 3):
        vector_store = self._load_or_create()
        if vector_store is None:
            return []
        return vector_store.similarity_search_with_score(query=query, k=k)

    def similarity_search(self, query: str, k: int = 3):
        vector_store = self._load_or_create()
        if vector_store is None:
            return []
        return vector_store.similarity_search(query=query, k=k)
