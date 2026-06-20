from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pipeline.config import load_yaml

_DEFAULT_METRICS = [
    "recall_at_k",
    "context_precision_at_k",
    "faithfulness_lexical",
    "answer_relevancy_lexical",
    "latency_ms",
]

@dataclass(slots=True)
class Variant:
    name: str
    pipeline: str
    config: dict[str, Any] = field(default_factory=dict)  # optional config overrides

@dataclass(slots=True)
class Experiment:
    name: str
    dataset: str
    sources: str
    variants: list[Variant]
    metrics: list[str] = field(default_factory=lambda: list(_DEFAULT_METRICS))
    runtime: str = "eval"
    env: str = "dev"
    parallelism: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dataset": self.dataset,
            "sources": self.sources,
            "metrics": list(self.metrics),
            "runtime": self.runtime,
            "env": self.env,
            "parallelism": self.parallelism,
            "variants": [
                {"name": v.name, "pipeline": v.pipeline, "config": v.config}
                for v in self.variants
            ],
        }

def experiment_from_mapping(raw: dict[str, Any], label: str = "<memory>") -> Experiment:
    block = raw.get("experiment", raw) if isinstance(raw, dict) else {}

    required = ("name", "dataset", "sources", "variants")
    missing = [key for key in required if not block.get(key)]
    if missing:
        raise ValueError(f"Experiment '{label}' missing required keys: {missing}")

    variants: list[Variant] = []
    seen: set[str] = set()
    for entry in block["variants"]:
        if not isinstance(entry, dict) or not entry.get("name") or not entry.get("pipeline"):
            raise ValueError(f"Each variant needs 'name' and 'pipeline': {entry!r}")
        name = str(entry["name"])
        if name in seen:
            raise ValueError(f"Duplicate variant name: {name}")
        seen.add(name)
        variants.append(
            Variant(
                name=name,
                pipeline=str(entry["pipeline"]),
                config=entry.get("config", {}) or {},
            )
        )

    return Experiment(
        name=str(block["name"]),
        dataset=str(block["dataset"]),
        sources=str(block["sources"]),
        variants=variants,
        metrics=list(block.get("metrics", _DEFAULT_METRICS)),
        runtime=str(block.get("runtime", "eval")),
        env=str(block.get("env", "dev")),
        parallelism=int(block.get("parallelism", 1)),
    )

def load_experiment(path: str | Path) -> Experiment:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Experiment config not found: {path}")

    return experiment_from_mapping(load_yaml(path), label=str(path))