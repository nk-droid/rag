import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from components.evaluation.metrics import METRIC_REGISTRY

_KNOWN_FIELDS = ("question", "ground_truth", "answer", "reference_contexts")

@dataclass(slots=True)
class EvalSample:
    question: str
    ground_truth: str | None = None
    reference_contexts: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "ground_truth": self.ground_truth,
            "reference_contexts": self.reference_contexts,
        }

def _records_from_dict(raw: dict[str, Any]) -> list[dict[str, Any]]:
    questions = raw.get("question") or raw.get("questions") or []
    if not isinstance(questions, list):
        raise ValueError("dataset 'question' must be a list.")
    records: list[dict[str, Any]] = []
    for index, question in enumerate(questions):
        record: dict[str, Any] = {"question": question}
        for key in ("ground_truth", "reference_contexts"):
            values = raw.get(key)
            if isinstance(values, list) and index < len(values):
                record[key] = values[index]
        records.append(record)
    return records

def _load_raw(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    if suffix == ".json":
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return _records_from_dict(data)
        if isinstance(data, list):
            return data
        raise ValueError("JSON dataset must be a dict-of-lists or list-of-records.")

    raise ValueError(f"Unsupported dataset format: {path.suffix} (use .json or .jsonl)")

def load_dataset(path: str | Path) -> list[EvalSample]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    samples: list[EvalSample] = []
    for record in _load_raw(path):
        if not isinstance(record, dict):
            continue
        question = str(record.get("question", "")).strip()
        if not question:
            continue
        refs = record.get("reference_contexts")
        samples.append(
            EvalSample(
                question=question,
                ground_truth=record.get("ground_truth"),
                reference_contexts=list(refs) if isinstance(refs, list) else None,
                metadata={k: v for k, v in record.items() if k not in _KNOWN_FIELDS},
            )
        )

    if not samples:
        raise ValueError(f"No usable questions found in {path}.")
    return samples

def check_metric_requirements(samples: list[EvalSample], metric_names: list[str]) -> list[str]:
    has_ground_truth = any(s.ground_truth for s in samples)
    has_references = any(s.reference_contexts for s in samples)
    available = {
        "ground_truth": has_ground_truth,
        "reference_contexts": has_references,
    }

    from components.evaluation.ragas_metrics import RAGAS_METRIC_NAMES

    ragas_needs_reference = {"context_precision", "context_recall"}

    warnings: list[str] = []
    for name in metric_names:
        if name in RAGAS_METRIC_NAMES:
            if name in ragas_needs_reference and not has_ground_truth:
                warnings.append(
                    f"metric '{name}' needs ground_truth in the dataset; it will report null."
                )
            continue
        spec = METRIC_REGISTRY.get(name)
        if spec is None:
            warnings.append(f"unknown metric '{name}' — it will be skipped.")
            continue
        if spec.requires and not any(available.get(field, False) for field in spec.requires):
            need = " or ".join(spec.requires)
            warnings.append(
                f"metric '{name}' needs {need} in the dataset; it will report null."
            )
    return warnings