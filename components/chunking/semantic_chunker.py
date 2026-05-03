from typing import Any

from components._base import ComponentSettings
from components.generation.generator import Generator
from components.generation.output_parser import OutputParser
from components.generation.prompt_builder import PromptBuilder
from components.shared_types import Chunk

class SemanticChunkerSettings(ComponentSettings):
    _CONFIG_PATH = "chunking.semantic"

    template_name: str = "chunk.yaml"
    parser_model: str = "SemanticChunks"
    llm: dict[str, Any] = {"provider": "ollama", "model_name": "llama3.2:latest"}

class SemanticChunker:
    def __init__(
        self,
        settings: SemanticChunkerSettings,
        prompt_builder: PromptBuilder,
        generator: Generator,
        parser: OutputParser,
    ) -> None:
        self.settings = settings
        self.prompt_builder = prompt_builder
        self.generator = generator
        self.parser = parser

    def chunk(self, text: str) -> list[Chunk]:
        try:
            prompt = self.prompt_builder.build(
                self.settings.template_name, self.settings.parser_model
            )
            output = self.generator.generate(prompt, {"text": text}).content
            chunks = self.parser.parse(output, self.settings.parser_model).chunks
            return [Chunk(text=t, index=i) for i, t in enumerate(chunks)]
        except Exception:
            return []
