from components.shared_types import RetrievedChunk

class ContextBuilder:
    """Assemble retrieved chunks into a single context block."""

    def build(self, chunks: list[RetrievedChunk]) -> str:
        return "\n\n".join(chunk.text for chunk in chunks)
