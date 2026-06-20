from langchain_core.prompts import PromptTemplate

from components._base import ComponentSettings
from components.shared_types import StreamingText
from infra.llm.llm_wrapper import BaseLLM

class StreamingGeneratorSettings(ComponentSettings):
    _CONFIG_PATH = "generation.streaming"

class StreamingGenerator:
    def __init__(self, settings: StreamingGeneratorSettings, llm: BaseLLM) -> None:
        self.settings = settings
        self.llm = llm

    def stream(self, prompt: PromptTemplate, inputs: dict) -> StreamingText:
        rendered = prompt.format(**inputs)
        for piece in self.llm.stream(rendered):
            if piece is None:
                continue
            yield str(piece)
