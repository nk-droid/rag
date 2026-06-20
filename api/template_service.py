import re
import yaml
from typing import Any
from pathlib import Path

from api.schemas import PipelineTemplate

PIPELINE_CONFIG_DIR = Path("configs/pipeline")
STEP_RE = re.compile(r"-\s*\{(?P<body>.+)\}")

def _display_name(template_id: str) -> str:
    words = template_id.replace("_", " ").replace("-", " ").split()
    special = {"bm25": "BM25", "rag": "RAG", "api": "API"}
    return " ".join(special.get(word.lower(), word.capitalize()) for word in words)

def _description(path: Path) -> str | None:
    lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            if lines:
                break
            continue
        if not stripped.startswith("#"):
            break
        lines.append(stripped.lstrip("#").strip())
    return " ".join(lines) if lines else None

def _steps(payload: dict[str, Any], section: str) -> list[dict[str, object]]:
    raw_steps = payload.get(section, {}).get("steps", [])
    if not isinstance(raw_steps, list):
        return []
    return [step for step in raw_steps if isinstance(step, dict)]

def _parse_component(value: str) -> str | list[str]:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        return [
            item.strip()
            for item in value[1:-1].split(",")
            if item.strip()
        ]
    return value

def _parse_step_line(line: str) -> dict[str, object] | None:
    match = STEP_RE.search(line.strip())
    if not match:
        return None
    body = match.group("body")
    name_match = re.search(r"(?:^|,\s*)name:\s*([^,}]+)", body)
    component_match = re.search(r"(?:^|,\s*)component:\s*(\[[^\]]+\]|[^,}]+)", body)
    if not name_match or not component_match:
        return None
    step: dict[str, object] = {
        "name": name_match.group(1).strip(),
        "component": _parse_component(component_match.group(1)),
    }
    for key in ("template_name", "parser"):
        value_match = re.search(rf"(?:^|,\s*){key}:\s*([^,}}]+)", body)
        if value_match:
            step[key] = value_match.group(1).strip()
    return step

def _fallback_payload(path: Path) -> dict[str, Any]:
    section: str | None = None
    payload: dict[str, Any] = {
        "init_pipeline": {"steps": []},
        "pipeline": {"steps": []},
    }
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "init_pipeline:":
            section = "init_pipeline"
            continue
        if stripped == "pipeline:":
            section = "pipeline"
            continue
        if section is None:
            continue
        step = _parse_step_line(stripped)
        if step:
            payload[section]["steps"].append(step)
    return payload

def _flatten_components(steps: list[dict[str, object]]) -> list[str]:
    components: list[str] = []
    for step in steps:
        component = step.get("component")
        if isinstance(component, str):
            components.append(component)
        elif isinstance(component, list):
            components.extend(str(item) for item in component if item)
    seen: set[str] = set()
    ordered: list[str] = []
    for component in components:
        if component in seen:
            continue
        seen.add(component)
        ordered.append(component)
    return ordered

def _tags(components: list[str]) -> list[str]:
    tags: list[str] = []
    if "hybrid_retriever" in components:
        tags.append("Hybrid")
    if any("ranker" in component for component in components):
        tags.append("Reranker")
    if "external_retriever" in components:
        tags.append("External fallback")
    if "repo_graph_indexer" in components or "graph_expander" in components:
        tags.append("Graph")
    if "self_critic" in components or "refiner" in components:
        tags.append("Refinement")
    if not tags:
        tags.append("Baseline")
    return tags

def list_pipeline_templates() -> list[PipelineTemplate]:
    templates: list[PipelineTemplate] = []
    for path in sorted(PIPELINE_CONFIG_DIR.glob("*.yaml")):
        if yaml is None:
            payload = _fallback_payload(path)
        else:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            continue
        init_steps = _steps(payload, "init_pipeline")
        run_steps = _steps(payload, "pipeline")
        components = _flatten_components(init_steps + run_steps)
        template_id = path.stem
        templates.append(
            PipelineTemplate(
                id=template_id,
                name=_display_name(template_id),
                file=str(path),
                description=_description(path),
                tags=_tags(components),
                components=components,
                init_steps=init_steps,
                run_steps=run_steps,
            )
        )
    return templates