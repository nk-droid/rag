from components._base import ComponentSettings
from components.shared_types import RetrievedChunk

class ContextBuilderSettings(ComponentSettings):
    _CONFIG_PATH = "context.builder"

class ContextBuilder:
    def __init__(self, settings: ContextBuilderSettings) -> None:
        self.settings = settings

    def build(self, chunks: list[RetrievedChunk]) -> str:
        blocks = []
        for chunk in chunks:
            metadata = chunk.metadata or {}
            path = metadata.get("relative_path") or metadata.get("path") or metadata.get("source") or "unknown"
            symbol = metadata.get("symbol") or metadata.get("title") or ""
            start = metadata.get("start_line")
            end = metadata.get("end_line")

            header = f"[source: {path}"
            if symbol:
                header += f" | symbol: {symbol}"
            if start and end:
                header += f" | lines: {start}-{end}"
            header += "]"

            blocks.append(f"{header}\n{chunk.text}")

        return "\n\n".join(blocks)
