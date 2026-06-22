"""Unit tests for the experiment framework: config parsing, comparison report,
and the variant runner (orchestrator + process pool mocked out)."""
import pytest

from pipeline.experiment import config as exp_config
from pipeline.experiment import report as exp_report
from pipeline.experiment import runner as exp_runner
from pipeline.experiment.config import Experiment, Variant, experiment_from_mapping, load_experiment


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def test_experiment_from_mapping_valid():
    raw = {
        "experiment": {
            "name": "exp",
            "dataset": "data.json",
            "sources": "src",
            "variants": [{"name": "v1", "pipeline": "simple"}],
            "metrics": ["recall_at_k"],
            "parallelism": 2,
        }
    }
    exp = experiment_from_mapping(raw)
    assert exp.name == "exp" and exp.parallelism == 2
    assert exp.variants[0].pipeline == "simple"
    assert exp.to_dict()["variants"][0]["name"] == "v1"


def test_experiment_from_mapping_defaults_metrics():
    exp = experiment_from_mapping(
        {"name": "e", "dataset": "d", "sources": "s", "variants": [{"name": "v", "pipeline": "p"}]}
    )
    assert exp.metrics == exp_config._DEFAULT_METRICS
    assert exp.runtime == "eval" and exp.env == "dev"


def test_experiment_from_mapping_errors():
    with pytest.raises(ValueError):
        experiment_from_mapping({"name": "e"})  # missing keys
    with pytest.raises(ValueError):
        experiment_from_mapping(
            {"name": "e", "dataset": "d", "sources": "s", "variants": [{"name": "v"}]}
        )  # variant missing pipeline
    with pytest.raises(ValueError):
        experiment_from_mapping(
            {
                "name": "e", "dataset": "d", "sources": "s",
                "variants": [{"name": "v", "pipeline": "p"}, {"name": "v", "pipeline": "q"}],
            }
        )  # duplicate variant


def test_load_experiment(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_experiment(tmp_path / "missing.yaml")
    p = tmp_path / "exp.yaml"
    p.write_text(
        "experiment:\n  name: e\n  dataset: d\n  sources: s\n  variants:\n    - {name: v, pipeline: simple}\n"
    )
    exp = load_experiment(p)
    assert exp.name == "e"


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
class _FakeStore:
    def __init__(self, metrics_by_variant):
        self._metrics = metrics_by_variant

    def list_variants(self, run_dir):
        return list(self._metrics)

    def load_metrics(self, run_dir, variant):
        return self._metrics[variant]


def test_build_comparison_picks_best():
    store = _FakeStore(
        {
            "a": {"recall_at_k": {"value": 0.5, "higher_is_better": True}, "latency_ms": {"value": 5.0, "higher_is_better": False}},
            "b": {"recall_at_k": {"value": 0.8, "higher_is_better": True}, "latency_ms": {"value": 2.0, "higher_is_better": False}},
        }
    )
    comparison = exp_report.build_comparison("rd", ["recall_at_k", "latency_ms", "missing"], store=store)
    assert comparison["best"]["recall_at_k"] == "b"  # higher better
    assert comparison["best"]["latency_ms"] == "b"  # lower better
    assert comparison["best"]["missing"] is None


def test_format_and_render():
    assert exp_report._format(None) == "—"
    assert exp_report._format(0.12345) == "0.1235"
    assert exp_report._format(5000.0) == "5000.0"
    assert exp_report._format("x") == "x"
    comparison = {
        "metrics": ["recall_at_k"],
        "variants": {"a": {"recall_at_k": {"value": 0.5}}, "b": {"recall_at_k": {"value": None}}},
        "best": {"recall_at_k": "a"},
    }
    md = exp_report.render_markdown(comparison)
    assert "**0.5000**" in md and "—" in md
    exp_report.render_console(comparison)  # smoke (no raise)


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
def test_build_variant_config_merges_and_applies_workspace():
    config = exp_runner.build_variant_config(
        {"pipeline": "simple", "config": {"retrieval": {"top_k": 9}}},
        {"runtime": "eval", "env": "dev"},
    )
    assert config["retrieval"]["top_k"] == 9
    assert "workspace" in config and config["workspace"]["id"]


def test_run_variant_invalid_config(monkeypatch):
    monkeypatch.setattr(exp_runner, "build_variant_config", lambda v, e: {"workspace": {"id": "w"}})
    monkeypatch.setattr(exp_runner, "validate_config", lambda c: ["bad key flow"])
    result = exp_runner.run_variant({"name": "v", "pipeline": "p"}, {"sources": "s"}, [{"question": "q"}])
    assert "invalid config" in result["error"]


def test_run_variant_init_failure(monkeypatch):
    def _boom(v, e):
        raise RuntimeError("nope")

    monkeypatch.setattr(exp_runner, "build_variant_config", _boom)
    result = exp_runner.run_variant({"name": "v", "pipeline": "p"}, {"sources": "s"}, [{"question": "q"}])
    assert "init failed" in result["error"]


def test_run_variant_happy_path(monkeypatch):
    import pipeline.orchestrator as orch_mod

    monkeypatch.setattr(exp_runner, "build_variant_config", lambda v, e: {"workspace": {"id": "w"}})
    monkeypatch.setattr(exp_runner, "validate_config", lambda c: [])

    class _FakeOrch:
        def __init__(self, config):
            pass

        def initialize(self, state):
            return state

        def run(self, state):
            return {"answer": f"A:{state['query']}", "retrieved": []}

    monkeypatch.setattr(orch_mod, "RAGOrchestrator", _FakeOrch)
    monkeypatch.setattr(exp_runner, "extract_answer", lambda s: s["answer"])
    monkeypatch.setattr(exp_runner, "extract_contexts", lambda s: [])

    result = exp_runner.run_variant(
        {"name": "v", "pipeline": "p"}, {"sources": "s"}, [{"question": "q1"}, {"question": "q2"}]
    )
    assert result["error"] is None
    assert [r["answer"] for r in result["records"]] == ["A:q1", "A:q2"]


def test_run_variant_records_per_sample_error(monkeypatch):
    import pipeline.orchestrator as orch_mod

    monkeypatch.setattr(exp_runner, "build_variant_config", lambda v, e: {"workspace": {"id": "w"}})
    monkeypatch.setattr(exp_runner, "validate_config", lambda c: [])

    class _FakeOrch:
        def __init__(self, config):
            pass

        def initialize(self, state):
            return state

        def run(self, state):
            raise RuntimeError("run boom")

    monkeypatch.setattr(orch_mod, "RAGOrchestrator", _FakeOrch)
    result = exp_runner.run_variant({"name": "v", "pipeline": "p"}, {"sources": "s"}, [{"question": "q"}])
    assert result["records"][0]["error"] == "run boom"


def test_run_experiment_sequential(monkeypatch):
    monkeypatch.setattr(exp_runner, "run_variant", lambda v, e, s: {"variant": v["name"]})
    experiment = {"variants": [{"name": "a"}, {"name": "b"}], "parallelism": 1}
    results = exp_runner.run_experiment(experiment, [{"question": "q"}])
    assert [r["variant"] for r in results] == ["a", "b"]


def test_run_experiment_parallel(monkeypatch):
    class _FakeFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class _FakeExecutor:
        def __init__(self, max_workers=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def submit(self, fn, *args):
            return _FakeFuture(fn(*args))

    monkeypatch.setattr(exp_runner, "ProcessPoolExecutor", _FakeExecutor)
    monkeypatch.setattr(exp_runner, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(exp_runner, "run_variant", lambda v, e, s: {"variant": v["name"]})

    experiment = {"variants": [{"name": "a"}, {"name": "b"}], "parallelism": 2}
    results = exp_runner.run_experiment(experiment, [{"question": "q"}])
    assert sorted(r["variant"] for r in results) == ["a", "b"]
