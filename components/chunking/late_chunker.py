from components.shared_types import Chunk

class LateChunker:
    def chunk(self, text: str) -> list[Chunk]:
        raise NotImplementedError
