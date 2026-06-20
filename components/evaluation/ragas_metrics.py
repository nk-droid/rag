import math
from typing import Any

RAGAS_METRIC_NAMES = {
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
}

def _usable(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in records if r.get("answer") and not r.get("error")]

def _sample_row(record: dict[str, Any]) -> tuple[str, str, list[str], str]:
    return (
        record.get("question", ""),
        record.get("answer", ""),
        list(record.get("contexts") or []),
        record.get("ground_truth") or "",
    )

def _cell(value: float | None, count: int, error: str | None = None) -> dict[str, Any]:
    cell = {"value": value, "count": count, "higher_is_better": True}
    if error:
        cell["error"] = error[:200]
    return cell

def _mean(values: list[Any]) -> float | None:
    clean = [
        float(v)
        for v in values
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    ]
    return sum(clean) / len(clean) if clean else None

def _build_evaluator(config: dict[str, Any] | None):
    from components.evaluation.ragas_eval import RagasEvaluator, RagasEvaluatorSettings

    settings = (
        RagasEvaluatorSettings.from_config(config) if config else RagasEvaluatorSettings()
    )
    return RagasEvaluator(settings)

def ragas_aggregate(
    records: list[dict[str, Any]],
    metric_names: list[str],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return ragas_aggregate_batch({"_": records}, metric_names, config).get("_", {})

def ragas_aggregate_batch(
    records_by_variant: dict[str, list[dict[str, Any]]],
    metric_names: list[str],
    config: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    order = list(records_by_variant)
    selected = [name for name in metric_names if name in RAGAS_METRIC_NAMES]
    if not selected:
        return {variant: {} for variant in order}

    # Build the combined sample set and remember each variant's row slice.
    combined = {"question": [], "answer": [], "retrieved_contexts": [], "ground_truth": []}
    slices: dict[str, tuple[int, int]] = {}
    cursor = 0
    for variant in order:
        usable = _usable(records_by_variant[variant])
        slices[variant] = (cursor, len(usable))
        cursor += len(usable)
        for record in usable:
            question, answer, contexts, reference = _sample_row(record)
            combined["question"].append(question)
            combined["answer"].append(answer)
            combined["retrieved_contexts"].append(contexts)
            combined["ground_truth"].append(reference)

    result = {
        variant: {name: _cell(None, slices[variant][1]) for name in selected}
        for variant in order
    }
    if cursor == 0:
        return result

    try:
        frame, name_map = _build_evaluator(config).score_frame(combined, metrics=selected)
    except Exception as error:
        for variant in order:
            count = slices[variant][1]
            result[variant] = {name: _cell(None, count, str(error)) for name in selected}
        return result

    if frame is None:
        return result

    for variant in order:
        start, count = slices[variant]
        if count == 0:
            continue
        sub = frame.iloc[start : start + count]
        for name in selected:
            column = name_map.get(name, name)
            column = column if column in sub else (name if name in sub else None)
            if column is None:
                continue
            values = sub[column].tolist()
            result[variant][name] = _cell(_mean(values), len([v for v in values if v is not None]))
    return result