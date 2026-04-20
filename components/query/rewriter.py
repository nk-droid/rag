from pathlib import Path

from components.generation.generator import Generator
from components.generation.output_parser import OutputParser
from components.generation.prompt_builder import PromptBuilder

class QueryRewriter:
    """Rewrite a user query into a more effective retrieval query."""

    def __init__(
        self,
        generator: Generator | None = None,
        prompt_builder: PromptBuilder | None = None,
        parser: OutputParser | None = None,
        template_name: str = "rewrite_query.yaml",
        parser_model: str = "RewrittenQuery",
    ) -> None:
        self.generator = generator
        self.prompt_builder = prompt_builder or PromptBuilder(
            template_dir=Path(__file__).parent / "templates"
        )
        self.parser = parser or OutputParser()
        self.template_name = template_name
        self.parser_model = parser_model

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

        if self.generator is None:
            return cleaned

        try:
            prompt = self.prompt_builder.build(self.template_name, self.parser_model)
            raw = self.generator.generate(prompt, {"query": cleaned})
            parsed = self.parser.parse(self._to_text(raw), self.parser_model)
            rewritten = parsed.query.strip()
            return rewritten or cleaned
        except Exception:
            return cleaned