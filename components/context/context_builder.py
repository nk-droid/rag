from components._base import ComponentSettings
from components.shared_types import RetrievedChunk

class ContextBuilderSettings(ComponentSettings):
    _CONFIG_PATH = "context.builder"

class ContextBuilder:
    def __init__(self, settings: ContextBuilderSettings) -> None:
        self.settings = settings

    def build(self, chunks: list[RetrievedChunk]) -> str:
        return "\n\n".join(chunk.text for chunk in chunks)
