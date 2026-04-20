import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from components.shared_types import RetrievedChunk
from components.retrieval.base_retriever import BaseRetriever

class CoarseRetriever(BaseRetriever):
    def __init__(self, index_path: str | Path = "data/indexes/coarse_index.json") -> None:
        super().__init__(store=None)
        self.index_path = Path(index_path)
        self._retriever: BM25Retriever | None = None
        self._retriever_signature: tuple[int, int] | None = None

    def _index_signature(self) -> tuple[int, int] | None:
        if not self.index_path.exists():
            return None
        stat = self.index_path.stat()
        return int(stat.st_size), int(stat.st_mtime_ns)

    def _load(self) -> BM25Retriever | None:
        signature = self._index_signature()
        if self._retriever_signature == signature:
            return self._retriever

        if signature is None:
            self._retriever = None
            self._retriever_signature = None
            return None

        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        docs_payload = payload.get("documents", [])

        documents = [
            Document(
                page_content=str(item.get("text", "")),
                metadata={"id": item.get("id"), **(item.get("metadata") or {})},
            )
            for item in docs_payload
            if str(item.get("text", "")).strip()
        ]

        if not documents:
            self._retriever = None
            self._retriever_signature = signature
            return None

        self._retriever = BM25Retriever.from_documents(documents)
        self._retriever_signature = signature
        return self._retriever

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        retriever = self._load()
        if retriever is None:
            return []

        retriever.k = top_k
        processed_query = retriever.preprocess_func(query)
        scores = retriever.vectorizer.get_scores(processed_query)

        if len(scores) == 0:
            return []

        k = min(int(top_k), len(scores))
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: float(scores[i]),
            reverse=True,
        )[:k]

        results: list[RetrievedChunk] = []
        for rank, idx in enumerate(ranked_indices, start=1):
            doc = retriever.docs[idx]
            score = float(scores[idx])

            metadata = dict(doc.metadata or {})
            metadata["source"] = "bm25"
            metadata["rank_sparse"] = rank
            metadata["bm25_score"] = score

            doc_id = str(metadata.get("id", f"coarse-{idx}"))
            results.append(
                RetrievedChunk(
                    id=doc_id,
                    text=doc.page_content,
                    score=score,
                    metadata=metadata,
                )
            )
            
        return results
