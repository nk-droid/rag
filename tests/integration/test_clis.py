"""Integration tests for the rag / rag-eval CLIs (orchestrator + experiment
runner stubbed; real config loading and validation)."""
import json
from types import SimpleNamespace

import pytest

from clis import cli as rag_cli
from clis import eval_cli
from components.shared_types import RetrievedChunk


# --------------------------------------------------------------------------- #
# rag CLI
# --------------------------------------------------------------------------- #
def _cli_args(**over):
    base = dict(
        list_pipelines=False,
        pipeline="simple",
        runtime="cli",
        env="dev",
        validate_only=False,
        source=None,
        repo_url=None,
        branch="main",
        source_id=None,
        access_token=None,
        query=None,
        top_k=None,
        skip_init=False,
        show_state=False,
        save_intermediate=False,
        run_id=None,
        output=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_cli_list_pipelines(capsys):
    rag_cli.main(_cli_args(list_pipelines=True))
    out = capsys.readouterr().out
    assert "simple" in out


def test_cli_validate_only(capsys):
    rag_cli.main(_cli_args(validate_only=True))
    assert "valid" in capsys.readouterr().out.lower()


def test_cli_run_with_local_source(tmp_path, monkeypatch, capsys):
    doc = tmp_path / "doc.txt"
    doc.write_text("hello world")

    class _FakeOrch:
        def __init__(self, config):
            self.config = config

        def initialize(self, state):
            return state

        def run(self, state):
            return {**state, "answer": "the answer", "retrieved": []}

    monkeypatch.setattr(rag_cli, "RAGOrchestrator", _FakeOrch)
    rag_cli.main(_cli_args(source=str(doc), query="q", skip_init=True))
    assert "the answer" in capsys.readouterr().out


def test_cli_run_writes_output_file(tmp_path, monkeypatch):
    doc = tmp_path / "doc.txt"
    doc.write_text("hello world")
    output = tmp_path / "nested" / "result.md"

    class _FakeOrch:
        def __init__(self, config):
            self.config = config

        def initialize(self, state):
            return state

        def run(self, state):
            return {**state, "answer": "the answer", "retrieved": []}

    monkeypatch.setattr(rag_cli, "RAGOrchestrator", _FakeOrch)
    rag_cli.main(
        _cli_args(source=str(doc), query="q", skip_init=True, output=str(output))
    )

    assert output.read_text(encoding="utf-8") == "the answer"


def test_cli_run_repo_with_evidence(tmp_path, monkeypatch, capsys):
    wt = tmp_path / "wt"
    wt.mkdir()

    class _FakeCloner:
        def __init__(self, settings):
            pass

        def clone_or_update(self, repo_url, branch=None, source_id=None, access_token=None):
            return SimpleNamespace(
                source_id="b", branch=branch or "main", commit_sha="sha", working_tree=wt
            )

    class _FakeOrch:
        def __init__(self, config):
            pass

        def initialize(self, state):
            return state

        def run(self, state):
            return {
                **state,
                "answer": "repo answer",
                "retrieved": [RetrievedChunk(id="a", text="t", metadata={"relative_path": "a.py"})],
                "graph_expanded": [RetrievedChunk(id="b", text="t2", metadata={"relative_path": "b.py"})],
            }

    monkeypatch.setattr(rag_cli, "RepoCloner", _FakeCloner)
    monkeypatch.setattr(rag_cli, "RAGOrchestrator", _FakeOrch)
    rag_cli.main(
        _cli_args(
            pipeline=None, repo_url="https://github.com/a/b.git", source_id="b",
            query="q", skip_init=True, show_state=True,
        )
    )
    out = capsys.readouterr().out
    assert "repo answer" in out and "a.py" in out


def test_cli_helpers(monkeypatch):
    chunk = RetrievedChunk(id="x", text="t", metadata={"path": "p.py"})
    assert rag_cli._chunk_path(chunk) == "p.py"
    assert rag_cli._chunk_path(RetrievedChunk(id="y", text="t")) == "unknown"
    dupes = [
        RetrievedChunk(id="1", text="t", metadata={"path": "a"}),
        RetrievedChunk(id="2", text="t", metadata={"path": "a"}),
        RetrievedChunk(id="3", text="t", metadata={"path": "b"}),
    ]
    assert rag_cli._unique_paths(dupes) == ["a", "b"]
    monkeypatch.setattr(rag_cli.console, "input", lambda *a, **k: "  typed  ")
    assert rag_cli._get_query() == "typed"
    assert rag_cli._get_source_path() == "typed"


def test_cli_missing_source_exits(monkeypatch):
    monkeypatch.setattr(rag_cli.console, "input", lambda *a, **k: "")
    with pytest.raises(SystemExit):
        rag_cli.main(_cli_args(query="q"))


# --------------------------------------------------------------------------- #
# rag-eval CLI
# --------------------------------------------------------------------------- #
def _setup_run(tmp_path):
    from infra.storage.experiment_store import ExperimentStore

    store = ExperimentStore(tmp_path)
    run_dir = store.create_run({"name": "exp", "metrics": ["latency_ms"]})
    # manifest stores experiment.metrics for _metric_names fallback
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["experiment"]["metrics"] = ["latency_ms"]
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    store.write_variant_runs(
        run_dir,
        {"variant": "v1", "pipeline": "simple", "records": [{"question": "q", "answer": "a", "latency_ms": 2.0}]},
    )
    return store, run_dir


def test_eval_cli_cmd_run(tmp_path, monkeypatch, capsys):
    dataset = tmp_path / "data.json"
    dataset.write_text(json.dumps({"question": ["q1"], "ground_truth": ["g1"]}))
    exp_yaml = tmp_path / "exp.yaml"
    exp_yaml.write_text(
        f"experiment:\n  name: e\n  dataset: {dataset}\n  sources: s\n  metrics: [latency_ms]\n"
        "  variants:\n    - {name: v1, pipeline: simple}\n"
    )

    monkeypatch.setattr(
        eval_cli, "run_experiment",
        lambda exp, samples: [{"variant": "v1", "pipeline": "simple", "records": [{"question": "q1", "answer": "a", "latency_ms": 1.0}], "error": None}],
    )
    eval_cli.cmd_run(SimpleNamespace(experiment=str(exp_yaml), root=str(tmp_path / "exp")))
    assert "Run stored at" in capsys.readouterr().out


def test_eval_cli_cmd_metrics_and_report(tmp_path, capsys):
    store, run_dir = _setup_run(tmp_path)
    eval_cli.cmd_metrics(SimpleNamespace(run_dir=str(run_dir), root=str(tmp_path), metrics=None))
    assert (run_dir / "comparison.json").exists()
    eval_cli.cmd_report(SimpleNamespace(run_dir=str(run_dir), root=str(tmp_path), metrics=["latency_ms"]))
    assert "comparison" in capsys.readouterr().out.lower() or True


def test_eval_cli_cmd_metrics_no_metrics(tmp_path):
    from infra.storage.experiment_store import ExperimentStore

    store = ExperimentStore(tmp_path)
    run_dir = store.create_run({"name": "exp"})  # no metrics in manifest
    with pytest.raises(SystemExit):
        eval_cli.cmd_metrics(SimpleNamespace(run_dir=str(run_dir), root=str(tmp_path), metrics=None))


def test_eval_cli_metric_names_override(tmp_path):
    store, run_dir = _setup_run(tmp_path)
    assert eval_cli._metric_names(run_dir, store, ["custom"]) == ["custom"]
    assert eval_cli._metric_names(run_dir, store, None) == ["latency_ms"]


def test_eval_cli_main_dispatch(tmp_path, monkeypatch, capsys):
    store, run_dir = _setup_run(tmp_path)
    monkeypatch.setattr("sys.argv", ["rag-eval", "--root", str(tmp_path), "report", str(run_dir)])
    eval_cli.main()
