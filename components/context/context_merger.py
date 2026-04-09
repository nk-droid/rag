from components.shared_types import RetrievedChunk

class ContextMerger:
    """Merge overlapping or duplicated chunks."""

    def merge(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        merged: list[RetrievedChunk] = []
        seen_texts: set[str] = set()
        for chunk in chunks:
            if chunk.text in seen_texts:
                continue
            seen_texts.add(chunk.text)
            merged.append(chunk)
        return merged
