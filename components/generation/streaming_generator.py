from components.shared_types import StreamingText

class StreamingGenerator:
    """Yield a response incrementally."""

    def __init__(self, llm: object | None = None) -> None:
        self.llm = llm

    def stream(self, query: str, context: str) -> StreamingText:
        raise NotImplementedError
