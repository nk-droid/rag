from components._base import ComponentSettings
from components.shared_types import StreamingText

class StreamingGeneratorSettings(ComponentSettings):
    _CONFIG_PATH = "generation.streaming"

class StreamingGenerator:
    def __init__(self, settings: StreamingGeneratorSettings, llm: object | None = None) -> None:
        self.settings = settings
        self.llm = llm

    def stream(self, query: str, context: str) -> StreamingText:
        raise NotImplementedError
