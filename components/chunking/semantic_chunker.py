from pathlib import Path
from components.shared_types import Chunk
from infra.llm.llm_factory import get_llm
from components.generation.prompt_builder import PromptBuilder
from components.generation.generator import Generator
from components.generation.output_parser import OutputParser
from infra.logging.logger import get_logger

logger = get_logger(__name__, "INFO")

class SemanticChunker:
    def __init__(self, config):
        semantic_chunking_configs = config.get("chunking", {}).get("semantic", {})
        self.template_name = semantic_chunking_configs.get("template_name")
        self.parser_model = semantic_chunking_configs.get("parser")
        self.prompt_builder = PromptBuilder(Path(__file__).parent / "templates")
        self.generator = Generator(get_llm(semantic_chunking_configs))
        self.parser = OutputParser()

    def split_text(self, text):
        prompt = self.prompt_builder.build(self.template_name, self.parser_model)
        output = self.generator.generate(prompt, {"text": text}).content
        return self.parser.parse(output, self.parser_model)

    def chunk(self, text: str) -> list[Chunk]:
        try:
            chunks = self.split_text(text).chunks
            result = [Chunk(text=text, index=index) for index, text in enumerate(chunks)]
            return result
        except:
            return []
