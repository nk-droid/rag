from components.shared_types import Chunk
from langchain_text_splitters import RecursiveCharacterTextSplitter

class RecursiveChunker(RecursiveCharacterTextSplitter):
    def __init__(self, config) -> None:
        chunking_configs = config.get("chunking", {})
        chunk_size = chunking_configs.get("chunk_size", 512)
        chunk_overlap = chunking_configs.get("chunk_overlap", 50)
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split_text(self, text):
        chunks = super().split_text(text)
        return [Chunk(text=chunk, index=idx) for idx, chunk in enumerate(chunks)]

    def chunk(self, text: str) -> list:
        chunks = self.split_text(text)
        return chunks