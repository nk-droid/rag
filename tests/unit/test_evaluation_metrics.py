"""Unit tests for lexical metrics, dataset loading, and ragas aggregation glue."""
import json

import pandas as pd
import pytest

from components.evaluation import metrics as m
from components.evaluation import ragas_metrics as rm
from components.evaluation.dataset import (
    EvalSample,
    check_metric_requirements,
    load_dataset,
)


# --------------------------------------------------------------------------- #
# lexical metrics
# --------------------------------------------------------------------------- #
def test_normalize_and_tokens():
    assert m._normalize("  Hello, WORLD!! ") == "hello world"
    assert m._tokens("a a b") == ["a", "a", "b"]


def test_f1_edge_cases():
    assert m._f1("", "") == 1.0
    assert m._f1("a", "") == 0.0
    assert m._f1("cat dog", "dog cat") == 1.0
    assert m._f1("totally different", "nothing shared") == 0.0
    assert 0 < m._f1("the cat sat", "the cat") <= 1.0


def test_matches_substring_and_coverage():
    assert m._matches("hello world", "say hello world now") is True
    assert m._matches("", "x") is False
    assert m._matches("alpha beta gamma", "alpha beta gamma delta", threshold=0.6) is True
    assert m._matches("alpha beta gamma", "alpha only", threshold=0.6) is False


def test_recall_at_k_paths():
    assert m._recall_at_k({"reference_contexts": ["a b"], "contexts": []}) == 0.0
    assert m._recall_at_k({"reference_contexts": ["hello"], "contexts": ["hello there"]}) == 1.0
    assert m._recall_at_k({"ground_truth": "cat", "contexts": ["a cat sat"]}) == 1.0
    assert m._recall_at_k({"answer": "x"}) is None


def test_context_precision_paths():
    assert m._context_precision_at_k({"contexts": []}) is None
    val = m._context_precision_at_k({"reference_contexts": ["cat"], "contexts": ["cat", "dog"]})
    assert val == 0.5
    val = m._context_precision_at_k({"ground_truth": "cat sat", "contexts": ["cat sat here", "nope"]})
    assert 0 <= val <= 1
    assert m._context_precision_at_k({"contexts": ["x"]}) is None


def test_answer_f1_em_faithfulness_relevancy_latency():
    assert m._answer_f1({"answer": "cat", "ground_truth": "cat"}) == 1.0
    assert m._answer_f1({"answer": "cat"}) is None
    assert m._answer_em({"answer": "Cat!", "ground_truth": "cat"}) == 1.0
    assert m._answer_em({"answer": "x"}) is None
    assert m._faithfulness_lexical({"answer": "cat", "contexts": ["a cat"]}) == 1.0
    assert m._faithfulness_lexical({"answer": "", "contexts": []}) is None
    assert m._answer_relevancy_lexical({"answer": "cat dog", "question": "cat"}) == 0.5
    assert m._answer_relevancy_lexical({"answer": "", "question": "q"}) is None
    assert m._latency_ms({"latency_ms": "12.5"}) == 12.5
    assert m._latency_ms({}) is None


def test_aggregate_averages_and_unknown():
    records = [
        {"answer": "cat", "ground_truth": "cat"},
        {"answer": "dog", "ground_truth": "cat"},
    ]
    out = m.aggregate(records, ["answer_em", "nope_metric"])
    assert out["answer_em"]["value"] == 0.5 and out["answer_em"]["count"] == 2
    assert out["nope_metric"]["error"] == "unknown metric"


def test_aggregate_all_none_value():
    out = m.aggregate([{"answer": "x"}], ["answer_f1"])
    assert out["answer_f1"]["value"] is None and out["answer_f1"]["count"] == 0


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
def test_load_dataset_jsonl(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"question": "q1", "ground_truth": "a"}\n\n{"question": "q2"}\n')
    samples = load_dataset(p)
    assert [s.question for s in samples] == ["q1", "q2"]
    assert samples[0].ground_truth == "a"


def test_load_dataset_json_dict_of_lists(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps({"question": ["q1", "q2"], "ground_truth": ["g1"], "reference_contexts": [["c1"], ["c2"]]}))
    samples = load_dataset(p)
    assert samples[0].ground_truth == "g1"
    assert samples[1].reference_contexts == ["c2"]


def test_load_dataset_json_list_of_records_and_metadata(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps([{"question": "q", "extra": 1}, {"no_question": True}, "skip"]))
    samples = load_dataset(p)
    assert len(samples) == 1 and samples[0].metadata == {"extra": 1}


def test_load_dataset_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "missing.json")
    bad = tmp_path / "d.txt"
    bad.write_text("x")
    with pytest.raises(ValueError):
        load_dataset(bad)
    empty = tmp_path / "e.json"
    empty.write_text(json.dumps({"question": []}))
    with pytest.raises(ValueError):
        load_dataset(empty)
    notlist = tmp_path / "n.json"
    notlist.write_text(json.dumps({"question": "notalist"}))
    with pytest.raises(ValueError):
        load_dataset(notlist)
    scalar = tmp_path / "s.json"
    scalar.write_text(json.dumps(42))
    with pytest.raises(ValueError):
        load_dataset(scalar)


def test_eval_sample_to_record():
    s = EvalSample(question="q", ground_truth="g", reference_contexts=["c"])
    assert s.to_record() == {"question": "q", "ground_truth": "g", "reference_contexts": ["c"]}


def test_check_metric_requirements_warns():
    samples = [EvalSample(question="q")]  # no ground_truth / references
    warnings = check_metric_requirements(
        samples, ["recall_at_k", "context_recall", "faithfulness_lexical", "nope"]
    )
    assert any("recall_at_k" in w for w in warnings)
    assert any("context_recall" in w for w in warnings)
    assert any("unknown metric 'nope'" in w for w in warnings)
    # faithfulness_lexical has no requirements -> no warning
    assert not any("faithfulness_lexical" in w for w in warnings)


def test_check_metric_requirements_satisfied():
    samples = [EvalSample(question="q", ground_truth="g", reference_contexts=["c"])]
    assert check_metric_requirements(samples, ["recall_at_k", "context_recall"]) == []


# --------------------------------------------------------------------------- #
# ragas aggregation glue (without real ragas)
# --------------------------------------------------------------------------- #
def test_ragas_helpers():
    recs = [{"answer": "a"}, {"answer": "", "error": None}, {"answer": "b", "error": "boom"}]
    assert rm._usable(recs) == [{"answer": "a"}]
    assert rm._sample_row({"question": "q", "answer": "a", "contexts": ["c"], "ground_truth": "g"}) == (
        "q", "a", ["c"], "g",
    )
    assert rm._cell(0.5, 3)["value"] == 0.5
    assert rm._cell(None, 1, "x" * 300)["error"] == "x" * 200
    assert rm._mean([1.0, None, float("nan"), 3.0]) == 2.0
    assert rm._mean([None]) is None


def test_ragas_aggregate_batch_no_ragas_metrics():
    out = rm.ragas_aggregate_batch({"v1": [{"answer": "a"}]}, ["answer_f1"])
    assert out == {"v1": {}}


def test_ragas_aggregate_batch_empty_records():
    out = rm.ragas_aggregate_batch({"v1": []}, ["faithfulness"])
    assert out["v1"]["faithfulness"]["value"] is None


def test_ragas_aggregate_batch_evaluator_error(monkeypatch):
    def _boom(config):
        raise RuntimeError("no model")

    monkeypatch.setattr(rm, "_build_evaluator", _boom)
    out = rm.ragas_aggregate_batch({"v1": [{"answer": "a", "question": "q"}]}, ["faithfulness"])
    assert "no model" in out["v1"]["faithfulness"]["error"]


def test_ragas_aggregate_batch_success(monkeypatch):
    class _FakeEval:
        def score_frame(self, combined, metrics):
            frame = pd.DataFrame({"faithfulness": [1.0, 0.0]})
            return frame, {"faithfulness": "faithfulness"}

    monkeypatch.setattr(rm, "_build_evaluator", lambda config: _FakeEval())
    records = {"v1": [{"answer": "a", "question": "q"}, {"answer": "b", "question": "q2"}]}
    out = rm.ragas_aggregate_batch(records, ["faithfulness"])
    assert out["v1"]["faithfulness"]["value"] == 0.5


def test_ragas_aggregate_single_variant_wrapper(monkeypatch):
    monkeypatch.setattr(rm, "ragas_aggregate_batch", lambda r, n, c=None: {"_": {"faithfulness": {"value": 1.0}}})
    assert rm.ragas_aggregate([], ["faithfulness"])["faithfulness"]["value"] == 1.0
