"""Integration tests for the FastAPI app via TestClient. Heavy service calls
(orchestrator runs, cloning, streaming) are stubbed; routing/validation is real."""
import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.routers import pipelines as pipelines_router
from api.routers import sources as sources_router
from api.source_store import SourceStore


@pytest.fixture()
def client():
    return TestClient(app)


# --------------------------------------------------------------------------- #
# basic
# --------------------------------------------------------------------------- #
def test_health_and_catalog(client):
    assert client.get("/health").json() == {"status": "ok"}
    catalog = client.get("/api/components/catalog").json()
    assert catalog["groups"] and "defaults" in catalog


# --------------------------------------------------------------------------- #
# prompts
# --------------------------------------------------------------------------- #
def test_prompts_list_and_create(client, tmp_path, monkeypatch):
    monkeypatch.setattr("api.prompt_service.PROMPT_TEMPLATE_DIR", tmp_path)
    assert client.get("/api/prompts").json()["prompts"] == []
    created = client.post("/api/prompts", json={"name": "greet", "template": "Hi {query}"})
    assert created.status_code == 200 and created.json()["prompt"]["name"] == "greet.yaml"
    dup = client.post("/api/prompts", json={"name": "greet", "template": "x"})
    assert dup.status_code == 409
    bad = client.post("/api/prompts", json={"name": "bad name!", "template": "x"})
    assert bad.status_code == 400


# --------------------------------------------------------------------------- #
# sources
# --------------------------------------------------------------------------- #
@pytest.fixture()
def store_override(tmp_path):
    store = SourceStore(manifest_path=tmp_path / "manifest.json")
    app.dependency_overrides[sources_router.get_store] = lambda: store
    app.dependency_overrides[pipelines_router.get_store] = lambda: store
    yield store
    app.dependency_overrides.pop(sources_router.get_store, None)
    app.dependency_overrides.pop(pipelines_router.get_store, None)


def test_sources_list_and_register_path(client, store_override, tmp_path, monkeypatch):
    monkeypatch.setattr("api.source_store.UPLOAD_DIR", tmp_path / "uploads")
    assert client.get("/api/sources").json()["sources"] == []

    f = tmp_path / "note.md"
    f.write_text("# hi")
    ok = client.post("/api/sources/register-path", json={"path": str(f)})
    assert ok.status_code == 200 and ok.json()["source"]["loader"] == "markdown_loader"

    missing = client.post("/api/sources/register-path", json={"path": str(tmp_path / "nope.md")})
    assert missing.status_code == 404

    bad = tmp_path / "x.pdf"
    bad.write_text("data")
    unsupported = client.post("/api/sources/register-path", json={"path": str(bad)})
    assert unsupported.status_code == 400


def test_sources_register_repo(client, store_override, tmp_path):
    from components.ingestion.repo_cloner import RepoCheckout

    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / "a.py").write_text("x")

    class _FakeCloner:
        def clone_or_update(self, repo_url, branch=None):
            return RepoCheckout(
                source_id="b", repo_url=repo_url, branch=branch or "main",
                working_tree=wt, commit_sha="sha", manifest_path=tmp_path / "m.json",
            )

    app.dependency_overrides[sources_router.get_repo_cloner] = lambda: _FakeCloner()
    try:
        ok = client.post("/api/sources/register-repo", json={"repo_url": "https://github.com/a/b.git"})
        assert ok.status_code == 200 and ok.json()["source"]["source_type"] == "repository"
        bad = client.post("/api/sources/register-repo", json={"repo_url": "not-a-url"})
        assert bad.status_code == 400
    finally:
        app.dependency_overrides.pop(sources_router.get_repo_cloner, None)


def test_sources_upload(client, store_override, tmp_path, monkeypatch):
    monkeypatch.setattr("api.source_store.UPLOAD_DIR", tmp_path / "uploads")
    ok = client.post("/api/sources/upload", files={"files": ("a.txt", b"hello", "text/plain")})
    assert ok.status_code == 200 and ok.json()["sources"][0]["loader"] == "text_loader"
    bad = client.post("/api/sources/upload", files={"files": ("a.exe", b"x", "application/octet-stream")})
    assert bad.status_code == 400


# --------------------------------------------------------------------------- #
# pipelines
# --------------------------------------------------------------------------- #
def _selection(**over):
    base = {
        "chunking": ["recursive_chunker"],
        "retrieval": ["coarse_retriever"],
        "generation": ["llm_generator"],
    }
    base.update(over)
    return base


def test_pipeline_templates_and_preview(client):
    templates = client.get("/api/pipelines/templates").json()["templates"]
    assert any(t["id"] == "simple" for t in templates)

    ok = client.post("/api/pipelines/preview", json={"query": "q", "selection": _selection()})
    assert ok.status_code == 200 and ok.json()["plan"]["pipeline"]

    invalid = client.post(
        "/api/pipelines/preview",
        json={"query": "q", "selection": _selection(retrieval=["coarse_retriever", "hybrid_retriever"])},
    )
    assert invalid.status_code == 400


def _registered_source(store, tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("hello world")
    return store.add_source(
        name="doc.txt", source_type="file", loader="text_loader", path=str(f), size_bytes=11
    )


def test_pipeline_run_and_initialize(client, store_override, tmp_path, monkeypatch):
    source = _registered_source(store_override, tmp_path)

    def _fake_run(request, sources, loader_service):
        from api.pipeline_service import build_pipeline_plan

        plan = build_pipeline_plan(request).plan
        return plan, ["w"], {"run_id": "r1", "pipeline_id": "p", "pipeline_name": "N", "query": request.query, "steps": []}

    monkeypatch.setattr(pipelines_router, "run_pipeline", _fake_run)
    body = {"query": "q", "source_ids": [source.id], "selection": _selection()}
    ok = client.post("/api/pipelines/run", json=body)
    assert ok.status_code == 200 and ok.json()["run_id"] == "r1"

    # invalid source id
    bad = client.post("/api/pipelines/run", json={"query": "q", "source_ids": ["nope"], "selection": _selection()})
    assert bad.status_code == 400

    def _fake_init(request, sources, loader_service):
        from api.pipeline_service import build_pipeline_plan
        from api.schemas import PipelineInitializeResponse

        return PipelineInitializeResponse(
            initialization_id="init_x", plan=build_pipeline_plan(request).plan, source_count=1, document_count=1
        )

    monkeypatch.setattr(pipelines_router, "initialize_pipeline", _fake_init)
    init = client.post("/api/pipelines/initialize", json=body)
    assert init.status_code == 200 and init.json()["initialization_id"] == "init_x"


def test_pipeline_stream(client, store_override, tmp_path, monkeypatch):
    source = _registered_source(store_override, tmp_path)

    def _fake_stream(request, sources, loader_service):
        yield ("plan", {"plan": {}, "warnings": []})
        yield ("token", {"piece": "hi"})
        yield ("done", {"result": {}, "steps": []})

    monkeypatch.setattr(pipelines_router, "stream_pipeline_run", _fake_stream)
    body = {"query": "q", "source_ids": [source.id], "selection": _selection(generation=["streaming_generator"])}
    resp = client.post("/api/pipelines/stream", json=body)
    assert resp.status_code == 200
    assert "event: token" in resp.text and "event: done" in resp.text

    # non-streaming selection rejected
    bad = client.post("/api/pipelines/stream", json={"query": "q", "source_ids": [source.id], "selection": _selection()})
    assert bad.status_code == 400


# --------------------------------------------------------------------------- #
# experiments (list / detail / queries)
# --------------------------------------------------------------------------- #
@pytest.fixture()
def experiment_root(tmp_path, monkeypatch):
    from api.routers import experiments as exp_router
    from infra.storage.experiment_store import ExperimentStore

    root = tmp_path / "experiments"
    monkeypatch.setattr(exp_router, "EXPERIMENT_ROOT", root)
    store = ExperimentStore(root)
    run_dir = store.create_run({"name": "exp"})
    store.write_variant_runs(
        run_dir,
        {
            "variant": "v1",
            "pipeline": "simple",
            "records": [{"question": "q1", "answer": "a1", "contexts": ["c"], "latency_ms": 1.0}],
        },
    )
    store.write_comparison(run_dir, {"metrics": ["latency_ms"], "best": {"latency_ms": "v1"}})
    return root, run_dir


def test_experiments_run_via_yaml(client, tmp_path, monkeypatch):
    from api.routers import experiments as exp_router

    monkeypatch.setattr(exp_router, "EXPERIMENT_ROOT", tmp_path / "experiments")
    dataset = tmp_path / "data.json"
    dataset.write_text(json.dumps({"question": ["q1"], "ground_truth": ["g1"]}))

    monkeypatch.setattr(
        exp_router, "run_experiment",
        lambda exp, samples: [
            {"variant": "v1", "pipeline": "simple",
             "records": [{"question": "q1", "answer": "a", "contexts": [], "latency_ms": 1.0}], "error": None}
        ],
    )
    yaml_text = (
        f"experiment:\n  name: e\n  dataset: {dataset}\n  sources: s\n  metrics: [latency_ms]\n"
        "  variants:\n    - {name: v1, pipeline: simple}\n"
    )
    resp = client.post("/api/experiments/run", json={"yaml_text": yaml_text})
    assert resp.status_code == 200
    assert resp.json()["name"] == "e" and resp.json()["comparison"]["metrics"] == ["latency_ms"]


def test_experiments_list_detail_queries(client, experiment_root):
    root, run_dir = experiment_root
    listing = client.get("/api/experiments").json()["experiments"]
    assert listing and listing[0]["name"] == "exp"

    detail = client.get(f"/api/experiments/exp/{run_dir.name}").json()
    assert detail["run_id"] == run_dir.name and detail["comparison"]["metrics"] == ["latency_ms"]

    queries = client.get(f"/api/experiments/exp/{run_dir.name}/queries").json()
    assert queries["queries"][0]["question"] == "q1"

    missing = client.get("/api/experiments/exp/nonexistent-run")
    assert missing.status_code == 404
