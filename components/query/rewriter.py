from components._base import ComponentSettings
from components.generation.generator import Generator
from components.generation.output_parser import OutputParser
from components.generation.prompt_builder import PromptBuilder

class QueryRewriterSettings(ComponentSettings):
    _CONFIG_PATH = "retrieval.query_rewrite"

    template_name: str = "rewrite_query.yaml"
    parser_model: str = "RewrittenQuery"

class QueryRewriter:
    def __init__(
        self,
        settings: QueryRewriterSettings,
        generator: Generator,
        prompt_builder: PromptBuilder,
        parser: OutputParser,
    ) -> None:
        self.settings = settings
        self.generator = generator
        self.prompt_builder = prompt_builder
        self.parser = parser

    @staticmethod
    def _to_text(output) -> str:
        if isinstance(output, str):
            return output
        content = getattr(output, "content", None)
        if isinstance(content, str):
            return content
        if content is not None:
            return str(content)
        return str(output)

    def rewrite(self, query: str) -> str:
        cleaned = query.strip()
        if not cleaned:
            return ""

        try:
            prompt = self.prompt_builder.build(
                self.settings.template_name, self.settings.parser_model
            )
            raw = self.generator.generate(prompt, {"query": cleaned})
            parsed = self.parser.parse(self._to_text(raw), self.settings.parser_model)
            rewritten = parsed.query.strip()
            return rewritten or cleaned
        except Exception:
            return cleaned
