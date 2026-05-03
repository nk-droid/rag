import json
from typing import Any

from components._base import ComponentSettings
from components.generation.generator import Generator
from components.generation.output_parser import OutputParser
from components.generation.prompt_builder import PromptBuilder

class RefinerSettings(ComponentSettings):
    _CONFIG_PATH = "postprocessing.refiner"

    template_name: str = "refine_answer.yaml"
    parser_model: str = "RefinedAnswer"

class Refiner:
    def __init__(
        self,
        settings: RefinerSettings,
        generator: Generator,
        prompt_builder: PromptBuilder,
        parser: OutputParser,
    ) -> None:
        self.settings = settings
        self.generator = generator
        self.prompt_builder = prompt_builder
        self.parser = parser

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

    def refine(self, answer: Any, critique: dict[str, Any]) -> str:
        raw_answer_text = self._to_text(answer).strip()
        answer_text = self._extract_answer_text(raw_answer_text)

        if not critique or not critique.get("needs_refine", False):
            return raw_answer_text

        try:
            prompt = self.prompt_builder.build(
                self.settings.template_name, self.settings.parser_model
            )
            raw = self.generator.generate(
                prompt,
                {
                    "answer": answer_text,
                    "critique": json.dumps(critique, ensure_ascii=False),
                },
            )
            parsed = self.parser.parse(self._to_text(raw), self.settings.parser_model)
            refined = getattr(parsed, "answer", "").strip()
            if not refined:
                return raw_answer_text
            return json.dumps({"answer": refined}, ensure_ascii=False)

        except Exception:
            return raw_answer_text
