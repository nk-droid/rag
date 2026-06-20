import re
import yaml
from pathlib import Path

from api.schemas import PromptTemplateItem

PROMPT_TEMPLATE_DIR = Path("components/generation/templates")
_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")
_FILENAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

def _display_name(stem: str) -> str:
    words = stem.replace("_", " ").replace("-", " ").split()
    return " ".join(word.capitalize() for word in words) or stem

def _parse_content(content: str) -> tuple[str, list[str]]:
    template = ""
    variables: list[str] = []
    if yaml is not None:
        try:
            data = yaml.safe_load(content) or {}
        except yaml.YAMLError:
            data = {}
        if isinstance(data, dict):
            template = str(data.get("template") or "")
            raw_variables = data.get("variables")
            if isinstance(raw_variables, dict):
                variables = [str(key) for key in raw_variables.keys()]
    if not template:
        template = content
    return template.strip("\n"), variables

def _item_from_path(path: Path) -> PromptTemplateItem:
    content = path.read_text(encoding="utf-8")
    template, variables = _parse_content(content)
    return PromptTemplateItem(
        name=path.name,
        label=_display_name(path.stem),
        template=template,
        variables=variables,
        content=content,
    )

def list_prompt_templates() -> list[PromptTemplateItem]:
    if not PROMPT_TEMPLATE_DIR.exists():
        return []
    return [_item_from_path(path) for path in sorted(PROMPT_TEMPLATE_DIR.glob("*.yaml"))]

def _normalize_filename(name: str) -> str:
    candidate = name.strip()
    if not candidate:
        raise ValueError("Prompt name is required.")
    if not candidate.endswith((".yaml", ".yml")):
        candidate = f"{candidate}.yaml"
    if not _FILENAME_RE.match(candidate):
        raise ValueError("Prompt name may only contain letters, numbers, '.', '_' and '-'.")
    return candidate

def _render_yaml(template: str, variables: list[str]) -> str:
    indented = "\n".join(f"  {line}" if line.strip() else "" for line in template.splitlines())
    lines = ["template: |", indented, "", "variables:"]
    if variables:
        lines.extend(f"  {variable}: null" for variable in variables)
    else:
        lines.append("  {}")
    return "\n".join(lines).rstrip() + "\n"

def create_prompt_template(name: str, template: str, overwrite: bool = False) -> PromptTemplateItem:
    template_body = template.strip("\n")
    if not template_body.strip():
        raise ValueError("Prompt template body is required.")

    filename = _normalize_filename(name)
    path = PROMPT_TEMPLATE_DIR / filename
    if path.exists() and not overwrite:
        raise FileExistsError(f"Prompt '{filename}' already exists.")

    variables = list(dict.fromkeys(_PLACEHOLDER_RE.findall(template_body)))
    PROMPT_TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_yaml(template_body, variables), encoding="utf-8")
    return _item_from_path(path)
