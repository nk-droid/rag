import re
import string
from dataclasses import dataclass
from typing import Any, Callable, Optional

_PUNCT = str.maketrans("", "", string.punctuation)

def _normalize(text: str) -> str:
    text = (text or "").lower().translate(_PUNCT)
    return re.sub(r"\s+", " ", text).strip()

def _tokens(text: str) -> list[str]:
    return _normalize(text).split()

def _coverage(needle: list[str], haystack_tokens: set[str]) -> float:
    if not needle:
        return 0.0
    found = sum(1 for token in needle if token in haystack_tokens)
    return found / len(needle)

def _f1(pred: str, gold: str) -> float:
    pred_tokens = _tokens(pred)
    gold_tokens = _tokens(gold)
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common: dict[str, int] = {}
    gold_counts: dict[str, int] = {}
    for token in gold_tokens:
        gold_counts[token] = gold_counts.get(token, 0) + 1
    overlap = 0
    seen: dict[str, int] = {}
    for token in pred_tokens:
        seen[token] = seen.get(token, 0) + 1
        if seen[token] <= gold_counts.get(token, 0):
            overlap += 1

    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def _matches(reference: str, context: str, threshold: float = 0.6) -> bool:
    ref_tokens = _tokens(reference)
    if not ref_tokens:
        return False
    norm_ref = _normalize(reference)
    norm_ctx = _normalize(context)
    if norm_ref and norm_ref in norm_ctx:
        return True
    return _coverage(ref_tokens, set(_tokens(context))) >= threshold

def _recall_at_k(record: dict[str, Any]) -> Optional[float]:
    contexts = record.get("contexts") or []
    references = record.get("reference_contexts")
    if references:
        if not contexts:
            return 0.0
        matched = sum(
            1 for ref in references if any(_matches(ref, ctx) for ctx in contexts)
        )
        return matched / len(references)

    ground_truth = record.get("ground_truth")
    if ground_truth:
        union = set(_tokens(" ".join(contexts)))
        return _coverage(_tokens(ground_truth), union)

    return None

def _context_precision_at_k(record: dict[str, Any]) -> Optional[float]:
    contexts = record.get("contexts") or []
    if not contexts:
        return None
    references = record.get("reference_contexts")
    if references:
        relevant = sum(
            1 for ctx in contexts if any(_matches(ref, ctx) for ref in references)
        )
        return relevant / len(contexts)

    ground_truth = record.get("ground_truth")
    if ground_truth:
        gt_tokens = _tokens(ground_truth)
        relevant = sum(
            1 for ctx in contexts if _coverage(gt_tokens, set(_tokens(ctx))) >= 0.3
        )
        return relevant / len(contexts)

    return None

def _answer_f1(record: dict[str, Any]) -> Optional[float]:
    ground_truth = record.get("ground_truth")
    if not ground_truth:
        return None
    return _f1(record.get("answer", ""), ground_truth)

def _answer_em(record: dict[str, Any]) -> Optional[float]:
    ground_truth = record.get("ground_truth")
    if not ground_truth:
        return None
    return float(_normalize(record.get("answer", "")) == _normalize(ground_truth))

def _faithfulness_lexical(record: dict[str, Any]) -> Optional[float]:
    answer = record.get("answer", "")
    contexts = record.get("contexts") or []
    answer_tokens = _tokens(answer)
    if not answer_tokens or not contexts:
        return None
    return _coverage(answer_tokens, set(_tokens(" ".join(contexts))))

def _answer_relevancy_lexical(record: dict[str, Any]) -> Optional[float]:
    answer_tokens = set(_tokens(record.get("answer", "")))
    question_tokens = set(_tokens(record.get("question", "")))
    if not answer_tokens or not question_tokens:
        return None
    intersection = answer_tokens & question_tokens
    union = answer_tokens | question_tokens
    return len(intersection) / len(union) if union else None

def _latency_ms(record: dict[str, Any]) -> Optional[float]:
    value = record.get("latency_ms")
    return float(value) if value is not None else None

@dataclass(frozen=True)
class MetricSpec:
    name: str
    fn: Callable[[dict[str, Any]], Optional[float]]
    higher_is_better: bool
    requires: tuple[str, ...]

METRIC_REGISTRY: dict[str, MetricSpec] = {
    "recall_at_k": MetricSpec(
        "recall_at_k", _recall_at_k, True, ("reference_contexts", "ground_truth")
    ),
    "context_precision_at_k": MetricSpec(
        "context_precision_at_k",
        _context_precision_at_k,
        True,
        ("reference_contexts", "ground_truth"),
    ),
    "answer_f1": MetricSpec("answer_f1", _answer_f1, True, ("ground_truth",)),
    "answer_em": MetricSpec("answer_em", _answer_em, True, ("ground_truth",)),
    "faithfulness_lexical": MetricSpec(
        "faithfulness_lexical", _faithfulness_lexical, True, ()
    ),
    "answer_relevancy_lexical": MetricSpec(
        "answer_relevancy_lexical", _answer_relevancy_lexical, True, ()
    ),
    "latency_ms": MetricSpec("latency_ms", _latency_ms, False, ()),
}

def aggregate(records: list[dict[str, Any]], metric_names: list[str]) -> dict[str, Any]:
    """Average each metric over the records, skipping `None` scores."""
    summary: dict[str, Any] = {}
    for name in metric_names:
        spec = METRIC_REGISTRY.get(name)
        if spec is None:
            summary[name] = {"value": None, "count": 0, "error": "unknown metric"}
            continue
        scores = [spec.fn(record) for record in records]
        valid = [s for s in scores if s is not None]
        summary[name] = {
            "value": (sum(valid) / len(valid)) if valid else None,
            "count": len(valid),
            "higher_is_better": spec.higher_is_better,
        }
    return summary