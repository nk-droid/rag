"""Unit tests for API service-layer logic: catalog, prompt/template services,
source store, loader service, and pipeline planning/serialization helpers."""
import types
from pathlib import Path

import pytest

from api import catalog
from api import pipeline_service as ps
from api import prompt_service
from api import template_service
from api.schemas import PipelineRequest, PipelineSelection
from api.source_store import (
    SourceStore,
    _repo_name_from_url,
    validate_public_repo_url,
)


# --------------------------------------------------------------------------- #
# catalog
# --------------------------------------------------------------------------- #
def test_catalog_payload_and_helpers():
    payload = catalog.as_json_payload()
    assert payload["groups"] and "defaults" in payload
    assert catalog.get_group("chunking").id == "chunking"
    assert catalog.get_group("nope") is None
    assert catalog.status_for("late_chunker") == "not_implemented"
    assert catalog.status_for("unknown") == "experimental"
    assert catalog.is_implemented("recursive_chunker") is True
    assert catalog.is_implemented("late_chunker") is False
    assert set(catalog.default_selection()) == set(catalog.GROUP_ORDER)


# --------------------------------------------------------------------------- #
# prompt_service
# --------------------------------------------------------------------------- #
def test_prompt_service_list_and_create(tmp_path, monkeypatch):
    monkeypatch.setattr(prompt_service, "PROMPT_TEMPLATE_DIR", tmp_path)
    assert prompt_service.list_prompt_templates() == []
    item = prompt_service.create_prompt_template("my_prompt", "Answer {query} using {context}")
    assert item.name == "my_prompt.yaml"
    assert set(item.variables) == {"query", "context"}
    listed = prompt_service.list_prompt_templates()
    assert any(p.name == "my_prompt.yaml" for p in listed)


def test_prompt_service_create_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(prompt_service, "PROMPT_TEMPLATE_DIR", tmp_path)
    with pytest.raises(ValueError):
        prompt_service.create_prompt_template("x", "   ")
    with pytest.raises(ValueError):
        prompt_service.create_prompt_template("bad name!", "body {q}")
    prompt_service.create_prompt_template("dup", "body")
    with pytest.raises(FileExistsError):
        prompt_service.create_prompt_template("dup", "body")
    # overwrite allowed
    prompt_service.create_prompt_template("dup", "newbody", overwrite=True)


def test_prompt_service_missing_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(prompt_service, "PROMPT_TEMPLATE_DIR", tmp_path / "missing")
    assert prompt_service.list_prompt_templates() == []


def test_prompt_service_parse_content_yaml(tmp_path, monkeypatch):
    monkeypatch.setattr(prompt_service, "PROMPT_TEMPLATE_DIR", tmp_path)
    (tmp_path / "p.yaml").write_text("template: |\n  Hi {name}\nvariables:\n  name: null\n")
    items = prompt_service.list_prompt_templates()
    assert items[0].variables == ["name"]


# --------------------------------------------------------------------------- #
# template_service
# --------------------------------------------------------------------------- #
def test_template_service_lists_pipeline_templates():
    templates = template_service.list_pipeline_templates()
    ids = {t.id for t in templates}
    assert "simple" in ids and "repo_hybrid_graph" in ids
    graph = next(t for t in templates if t.id == "repo_hybrid_graph")
    assert "Graph" in graph.tags
    assert graph.components


def test_template_service_helpers():
    assert template_service._display_name("repo_hybrid_graph") == "Repo Hybrid Graph"
    assert template_service._tags(["hybrid_retriever", "cross_encoder_ranker"]) == ["Hybrid", "Reranker"]
    assert template_service._tags([]) == ["Baseline"]
    assert template_service._parse_component("[a, b]") == ["a", "b"]
    assert template_service._parse_component("solo") == "solo"
    step = template_service._parse_step_line("- {name: x, component: coarse_retriever}")
    assert step == {"name": "x", "component": "coarse_retriever"}
    assert template_service._parse_step_line("not a step") is None


def test_template_service_flatten_components():
    steps = [{"component": "a"}, {"component": ["b", "a", ""]}]
    assert template_service._flatten_components(steps) == ["a", "b"]


# --------------------------------------------------------------------------- #
# source_store
# --------------------------------------------------------------------------- #
def _store(tmp_path):
    return SourceStore(manifest_path=tmp_path / "manifest.json")


def test_source_store_add_list_and_lookup(tmp_path):
    store = _store(tmp_path)
    assert store.list_sources() == []
    s = store.add_source(name="a.txt", source_type="file", loader="text_loader", path="/a.txt", size_bytes=10)
    listed = store.list_sources()
    assert len(listed) == 1 and listed[0].id == s.id
    assert store.get_sources_by_ids([s.id, "missing"]) == [listed[0]]


def test_source_store_resolve_loader(tmp_path):
    store = _store(tmp_path)
    d = tmp_path / "dir"
    d.mkdir()
    assert store.resolve_loader_for_path(d) == ("directory", "directory_loader")
    assert store.resolve_loader_for_path(Path("x.md")) == ("file", "markdown_loader")
    assert store.resolve_loader_for_path(Path("x.txt")) == ("file", "text_loader")
    assert store.resolve_loader_for_path(Path("x.pdf")) == ("file", "document_loader")


def test_source_store_persist_uploaded_file(tmp_path, monkeypatch):
    import api.source_store as ss

    monkeypatch.setattr(ss, "UPLOAD_DIR", tmp_path / "uploads")
    store = _store(tmp_path)
    rec = store.persist_uploaded_file(filename="note.md", contents=b"# hi")
    assert rec.loader == "markdown_loader" and rec.size_bytes == 4
    with pytest.raises(ValueError):
        store.persist_uploaded_file(filename="bad.exe", contents=b"x")


def test_source_store_add_repository_source(tmp_path):
    from components.ingestion.repo_cloner import RepoCheckout

    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / "f.py").write_text("x")
    store = _store(tmp_path)
    checkout = RepoCheckout(
        source_id="repo",
        repo_url="https://github.com/a/b.git",
        branch="main",
        working_tree=wt,
        commit_sha="deadbeef",
        manifest_path=tmp_path / "m.json",
    )
    rec = store.add_repository_source(checkout)
    assert rec.source_type == "repository" and rec.commit_sha == "deadbeef"
    assert rec.name == "b@main"


def test_source_store_corrupt_manifest(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text("{ not json")
    store = SourceStore(manifest_path=p)
    assert store.list_sources() == []


def test_validate_public_repo_url():
    assert validate_public_repo_url("https://github.com/a/b.git").startswith("https://")
    assert validate_public_repo_url("git@github.com:a/b.git").startswith("git@")
    with pytest.raises(ValueError):
        validate_public_repo_url("  ")
    with pytest.raises(ValueError):
        validate_public_repo_url("not-a-url")


def test_repo_name_from_url():
    assert _repo_name_from_url("https://github.com/a/MyRepo.git") == "MyRepo"
    assert _repo_name_from_url("git@github.com:a/b") == "b"


# --------------------------------------------------------------------------- #
# pipeline_service — serialization helpers
# --------------------------------------------------------------------------- #
def test_prompt_text_and_short_text():
    assert ps._prompt_text(None) is None
    assert ps._prompt_text("p") == "p"
    assert ps._prompt_text(types.SimpleNamespace(template="t")) == "t"
    assert ps._prompt_text(types.SimpleNamespace(x=1))  # falls to str()
    assert ps._short_text("ab", limit=5) == "ab"
    long = ps._short_text("x" * 20, limit=5)
    assert "truncated" in long


def test_serialize_payload_variants(tmp_path):
    assert ps._serialize_payload(None) is None
    assert ps._serialize_payload(Path("/a")) == "/a"
    assert ps._serialize_payload({"a": 1}) == {"a": 1}
    assert ps._serialize_payload(list(range(60))) == list(range(50))  # capped
    assert ps._serialize_payload(types.SimpleNamespace(a=1, _hidden=2)) == {"a": 1}

    class _M:
        def model_dump(self):
            return {"m": 1}

    assert ps._serialize_payload(_M()) == {"m": 1}


def test_serialize_chunks_and_answer():
    chunks = [
        types.SimpleNamespace(id="a", text="t", score=0.5, metadata={"k": 1}),
        {"id": "b", "content": "c", "score": 0.1},
        12345,
    ]
    out = ps._serialize_chunks(chunks)
    assert out[0]["id"] == "a" and out[1]["text"] == "c"
    assert ps._serialize_chunks("notalist") == []
    assert ps._answer_from_state({"parsed_output": types.SimpleNamespace(answer="A")}) == "A"
    assert ps._answer_from_state({"answer": "raw"}) == "raw"
    assert ps._answer_from_state({"answer": types.SimpleNamespace(content="C")}) == "C"
    assert ps._answer_from_state({}) == ""


def test_step_output_and_summary():
    state = {
        "query": "q",
        "retrieved": [{"id": "x", "text": "t"}],
        "context": "ctx",
        "answer": "two words",
        "evaluation": {"m": 1},
    }
    assert ps._step_output("query", state)["query"] == "q"
    assert ps._step_output("retrieved_documents", state)["documents"]
    assert ps._step_output("context", state)["context"] == "ctx"
    assert "answer" in ps._step_output("generated_answer", state)
    assert ps._step_output("metrics", state)["metrics"] == {"m": 1}
    assert "state_keys" in ps._step_output("raw_json", state)
    docs_out = {"documents": [1, 2]}
    assert ps._step_summary("retrieved_documents", docs_out) == "2 chunks"
    assert ps._step_summary("generated_answer", {"answer": "two words"}) == "2 answer words"
    assert ps._step_summary("unknown", {}) is None


def test_build_run_steps_filters_phases():
    state = {
        "step_timings": [
            {"phase": "init", "step_name": "chunk", "component": "recursive_chunker", "latency_ms": 1.0},
            {"phase": "run", "step_name": "retrieve", "component": "coarse_retriever", "latency_ms": 2.0},
        ],
        "retrieved": [{"id": "x", "text": "t"}],
    }
    run_steps = ps._build_run_steps(state)  # excludes init
    assert len(run_steps) == 1 and run_steps[0]["component"] == "coarse_retriever"
    init_steps = ps._build_run_steps(state, phases={"init"})
    assert len(init_steps) == 1 and init_steps[0]["component"] == "recursive_chunker"


# --------------------------------------------------------------------------- #
# pipeline_service — planning
# --------------------------------------------------------------------------- #
def _req(retrieval=("coarse_retriever",), ranking=(), generation=("llm_generator",), postprocessing=(), **over):
    sel = PipelineSelection(
        chunking=["recursive_chunker"],
        retrieval=list(retrieval),
        ranking=list(ranking),
        generation=list(generation),
        postprocessing=list(postprocessing),
        **{k: v for k, v in over.items() if k in PipelineSelection.model_fields},
    )
    return PipelineRequest(query="hello", selection=sel, source_ids=["s1"])


def test_build_plan_basic_autoinserts_generator_and_indexer():
    planned = ps.build_pipeline_plan(_req())
    components = [s.component for s in planned.plan.pipeline]
    assert "prompt_builder" in components and "llm_generator" in components
    assert planned.plan.indexers == ["coarse_indexer"]


def test_build_plan_hybrid_with_reranker_identity():
    planned = ps.build_pipeline_plan(_req(retrieval=["hybrid_retriever"], ranking=["embedding_ranker"]))
    _, name = ps._pipeline_identity(planned.plan)
    assert name == "Hybrid RAG + Reranker"
    assert set(["embedding_indexer", "coarse_indexer"]) <= set(planned.plan.indexers)


def test_build_plan_external_fallback_and_only():
    planned = ps.build_pipeline_plan(_req(retrieval=["coarse_retriever", "external_retriever"]))
    names = [s.name for s in planned.plan.pipeline]
    assert "retrieve_external" in names and any("external_retriever" in w for w in planned.warnings)
    only = ps.build_pipeline_plan(_req(retrieval=["external_retriever"]))
    _, name = ps._pipeline_identity(only.plan)
    assert name == "External Search RAG"


def test_build_plan_streaming():
    planned = ps.build_pipeline_plan(_req(generation=["streaming_generator"]))
    assert ps.plan_has_streaming_generator(planned.plan) is True
    _, name = ps._pipeline_identity(planned.plan)
    assert "Streaming" in name


def test_build_plan_validation_errors():
    with pytest.raises(ps.PipelineValidationError):
        ps.build_pipeline_plan(_req(retrieval=["coarse_retriever", "hybrid_retriever"]))  # 2 primaries
    bad = PipelineRequest(query="q", selection=PipelineSelection(chunking=["recursive_chunker"], retrieval=[]))
    with pytest.raises(ps.PipelineValidationError):
        ps.build_pipeline_plan(bad)
    notimpl = PipelineRequest(
        query="q", selection=PipelineSelection(chunking=["late_chunker"], retrieval=["coarse_retriever"])
    )
    with pytest.raises(ps.PipelineValidationError):
        ps.build_pipeline_plan(notimpl)


def test_required_indexers_and_dedupe():
    assert ps._required_indexers_for_retriever("hybrid_retriever") == ["embedding_indexer", "coarse_indexer"]
    assert ps._required_indexers_for_retriever("fine_retriever") == ["embedding_indexer"]
    assert ps._required_indexers_for_retriever("graph_retriever") == ["repo_graph_indexer"]
    assert ps._required_indexers_for_retriever("unknown") == []
    assert ps._dedupe([" a ", "a", "b", ""]) == ["a", "b"]


# --------------------------------------------------------------------------- #
# pipeline_service — initialization tracking
# --------------------------------------------------------------------------- #
def test_initialization_record_and_require_flow():
    ps._INITIALIZED_PIPELINES.clear()
    from api.schemas import SourceRecord
    from datetime import datetime, timezone

    sources = [
        SourceRecord(
            id="s1", name="n", source_type="file", loader="text_loader", path="/p",
            created_at=datetime.now(timezone.utc),
        )
    ]
    req = _req()
    planned = ps.build_pipeline_plan(req)
    init_id = ps._record_initialization(req, sources, planned.plan, document_count=3)
    assert init_id in ps._INITIALIZED_PIPELINES

    # require with matching id passes
    req_with_id = _req()
    req_with_id.initialization_id = init_id
    req_with_id.skip_initialization = True
    ps._require_initialized_pipeline(req_with_id, sources, planned.plan)

    # missing id
    with pytest.raises(ps.PipelineValidationError):
        ps._require_initialized_pipeline(_req(), sources, planned.plan)

    # unknown id
    bad = _req()
    bad.initialization_id = "init_nope"
    with pytest.raises(ps.PipelineValidationError):
        ps._require_initialized_pipeline(bad, sources, planned.plan)


def test_purge_expired_initializations():
    ps._INITIALIZED_PIPELINES.clear()
    ps._INITIALIZED_PIPELINES["old"] = ps.InitializedPipelineRecord(
        initialization_id="old", fingerprint="f", source_ids=(), created_at=0.0,
        document_count=0, source_count=0,
    )
    ps._purge_expired_initializations(now=ps.INITIALIZATION_TTL_SEC + 10)
    assert "old" not in ps._INITIALIZED_PIPELINES


def test_source_paths_exist(tmp_path):
    from api.schemas import SourceRecord
    from datetime import datetime, timezone

    real = tmp_path / "f.txt"
    real.write_text("x")
    sources = [
        SourceRecord(id="ok", name="n", source_type="file", loader="text_loader", path=str(real), created_at=datetime.now(timezone.utc)),
        SourceRecord(id="missing", name="n", source_type="file", loader="text_loader", path="/no/such", created_at=datetime.now(timezone.utc)),
    ]
    assert ps.source_paths_exist(sources) == ["missing"]
