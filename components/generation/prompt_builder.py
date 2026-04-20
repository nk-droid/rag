import yaml
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from components.generation.output_parser import pydantic_models #FIXME

formatting = """
Return the result in the following format:
{format_instructions}

Don't return anything else
"""

class PromptBuilder:
    """Build prompts from a named template and runtime inputs."""

    def __init__(self, template_dir: str | Path | None = None, use_cache: bool = True) -> None:
        self.template_dir = Path(template_dir) if template_dir else Path(__file__).parent / "templates"
        self.use_cache = use_cache
        self._compiled_prompts: dict[tuple[str, int, str | None], PromptTemplate] = {}

    def _template_signature(self, template_path: Path) -> tuple[str, int]:
        stat = template_path.stat()
        return str(template_path.resolve()), int(stat.st_mtime_ns)

    def build(self, template_name: str, parser_model: str | None = None) -> PromptTemplate:
        template_path = self.template_dir / template_name
        template_key = (*self._template_signature(template_path), parser_model)
        if self.use_cache:
            cached = self._compiled_prompts.get(template_key)
            if cached is not None:
                return cached

        with open(template_path, "r", encoding="utf-8") as prompt_file:
            prompt_config = yaml.safe_load(prompt_file)

        variables = prompt_config.get("variables", {})
        if not isinstance(variables, dict):
            variables = {}

        partial_variables = {}
        if parser_model:
            model_cls = pydantic_models.get(parser_model)
            if model_cls is None:
                raise ValueError(f"Unsupported parser model: {parser_model}")
            partial_variables = {
                "format_instructions": PydanticOutputParser(
                    pydantic_object=model_cls
                ).get_format_instructions()
            }

        prompt = PromptTemplate(
            template=(
                prompt_config.get("template")
                if not parser_model
                else f"{prompt_config.get('template')}\n{formatting}"
            ),
            input_variables=list(variables.keys()),
            partial_variables=partial_variables,
        )

        if self.use_cache:
            self._compiled_prompts[template_key] = prompt
        return prompt
