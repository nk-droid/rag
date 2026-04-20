from pathlib import Path

from components.generation.generator import Generator
from components.generation.output_parser import OutputParser
from components.generation.prompt_builder import PromptBuilder

class MultiQueryGenerator:
    """Expand a single query into multiple search variations."""

    def __init__(
        self,
        generator: Generator | None = None,
        prompt_builder: PromptBuilder | None = None,
        parser: OutputParser | None = None,
        max_queries: int = 3,
        template_name: str = "multi_query.yaml",
        parser_model: str = "QueryVariants",
    ) -> None:
        self.generator = generator
        self.prompt_builder = prompt_builder or PromptBuilder(
            template_dir=Path(__file__).parent / "templates"
        )
        self.parser = parser or OutputParser()
        self.max_queries = max(1, int(max_queries))
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

    def generate(self, query: str) -> list[str]:
        cleaned = query.strip()
        if not cleaned:
            return []

        if self.generator is None:
            return [cleaned]

        try:
            prompt = self.prompt_builder.build(self.template_name, self.parser_model)
            raw = self.generator.generate(
                prompt,
                {"query": cleaned, "max_queries": self.max_queries},
            )
            parsed = self.parser.parse(self._to_text(raw), self.parser_model)
            variants = [q.strip() for q in parsed.queries if isinstance(q, str) and q.strip()]

            deduped: list[str] = []
            seen: set[str] = set()
            for q in variants:
                key = q.lower()
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(q)

            if cleaned.lower() not in seen:
                deduped.insert(0, cleaned)

            return deduped[:self.max_queries]
        except Exception:
            return [cleaned]
