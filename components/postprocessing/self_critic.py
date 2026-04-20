import json
from pathlib import Path
from typing import Any

from components.generation.generator import Generator
from components.generation.output_parser import OutputParser, SelfCritique
from components.generation.prompt_builder import PromptBuilder

class SelfCritic:
    """LLM-based critic that evaluates answer quality against context."""

    def __init__(
        self,
        generator: Generator | None = None,
        prompt_builder: PromptBuilder | None = None,
        parser: OutputParser | None = None,
        template_name: str = "self_critic.yaml",
        parser_model: str = "SelfCritique",
    ) -> None:
        self.generator = generator
        self.prompt_builder = prompt_builder or PromptBuilder(
            template_dir=Path(__file__).parent / "templates"
        )
        self.parser = parser or OutputParser()
        self.template_name = template_name
        self.parser_model = parser_model

    @staticmethod
    def _to_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        content = getattr(value, "content", None)
        if isinstance(content, str):
            return content
        return str(value)

    @staticmethod
    def _extract_answer_text(answer_text: str) -> str:
        stripped = answer_text.strip()
        if not stripped:
            return ""

        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return stripped

        if isinstance(payload, dict) and "answer" in payload:
            return str(payload.get("answer", "")).strip()
        return stripped

    def critique(self, answer: Any, context: str) -> dict[str, Any]:
        answer_text = self._extract_answer_text(self._to_text(answer))
        context_text = (context or "").strip()

        if not answer_text:
            return SelfCritique(
                needs_refine=True,
                grounded=False,
                issues=["empty_answer"],
                suggestions=["Provide a direct answer grounded in context."],
            ).model_dump()

        if self.generator is None:
            return SelfCritique(
                needs_refine=False,
                grounded=True,
                issues=[],
                suggestions=[],
            ).model_dump()

        try:
            prompt = self.prompt_builder.build(self.template_name, self.parser_model)
            raw = self.generator.generate(
                prompt,
                {"answer": answer_text, "context": context_text},
            )
            parsed = self.parser.parse(self._to_text(raw), self.parser_model)
            return parsed.model_dump()

        except Exception:
            return SelfCritique(
                needs_refine=False,
                grounded=True,
                issues=[],
                suggestions=[],
            ).model_dump()
