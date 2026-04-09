import hashlib

from components.shared_types import RetrievedChunk
from components.retrieval.base_retriever import BaseRetriever

class FineRetriever(BaseRetriever):
    """Fetch a narrow, precision-oriented set of candidates."""

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if self.store is None or not query.strip() or top_k <= 0:
            return []

        docs_with_score = self.store.similarity_search_with_score(
            query=query,
            k=top_k
        )

        results: list[RetrievedChunk] = []
        for doc, raw_score in docs_with_score:
            text = str(getattr(doc, "page_content", "") or "")
            if not text.strip():
                continue

            metadata = dict(getattr(doc, "metadata", {}) or {})
            chunk_id = self._resolve_chunk_id(text, metadata, fallback_index=len(results))
            score = float(raw_score)

            metadata.setdefault("source", "dense")
            metadata["rank_dense"] = len(results) + 1
            metadata["dense_score"] = score

            results.append(
                RetrievedChunk(
                    id=chunk_id,
                    text=text,
                    score=score,
                    metadata=metadata,
                )
            )

        return results

    def _resolve_chunk_id(self, text: str, metadata: dict, fallback_index: int) -> str:
        if metadata.get("id"):
            return str(metadata["id"])

        source = metadata.get("source", "unknown")
        source_index = metadata.get("source_index", 0)
        chunk_index = metadata.get("chunk_index", fallback_index)
        fingerprint = hashlib.sha1(f"{source}:{text}".encode("utf-8")).hexdigest()[:12]
        return f"{source_index}:{chunk_index}:{source}:{fingerprint}"
