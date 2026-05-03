from components._base import ComponentSettings
from components.generation.generator import Generator
from components.generation.output_parser import OutputParser
from components.generation.prompt_builder import PromptBuilder

class MultiQueryGeneratorSettings(ComponentSettings):
    _CONFIG_PATH = "retrieval.query_expansion"

    max_queries: int = 3
    template_name: str = "multi_query.yaml"
    parser_model: str = "QueryVariants"

class MultiQueryGenerator:
    def __init__(
        self,
        settings: MultiQueryGeneratorSettings,
        generator: Generator,
        prompt_builder: PromptBuilder,
        parser: OutputParser,
    ) -> None:
        self.settings = settings
        self.generator = generator
        self.prompt_builder = prompt_builder
        self.parser = parser

    @property
    def max_queries(self) -> int:
        return max(1, int(self.settings.max_queries))

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

        try:
            prompt = self.prompt_builder.build(
                self.settings.template_name, self.settings.parser_model
            )
            raw = self.generator.generate(
                prompt,
                {"query": cleaned, "max_queries": self.max_queries},
            )
            parsed = self.parser.parse(self._to_text(raw), self.settings.parser_model)
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

            return deduped[: self.max_queries]
        except Exception:
            return [cleaned]
