"""Integration coverage for RAGOrchestrator: step expansion, source collection,
init-skip manifest logic, and the public intermediate helpers."""
import json
from pathlib import Path

import pytest

import pipeline.orchestrator as orchestrator_module
from pipeline.orchestrator import RAGOrchestrator


def _noop(state, config):
    return {**state, "noop_ran": True}


def _base_config(**over):
    cfg = {
        "runtime": {"mode": "api"},
        "intermediate": {"enabled": False},
        "init_pipeline": {"steps": []},
        "pipeline": {"steps": [{"name": "noop", "component": "noop_component"}]},
    }
    cfg.update(over)
    return cfg


@pytest.fixture(autouse=True)
def _register_noop(monkeypatch):
    monkeypatch.setitem(orchestrator_module.REGISTRY, "noop_component", _noop)


# --------------------------------------------------------------------------- #
# _expand_step
# --------------------------------------------------------------------------- #
def test_expand_step_single_and_list_with_options():
    orch = RAGOrchestrator(_base_config())
    assert orch._expand_step({"name": "s", "component": "x", "top_k": 9}) == [("s", "x", {"top_k": 9})]
    expanded = orch._expand_step(
        {"name": "s", "component": ["a", "b"], "shared": 1, "options": {"a": {"extra": 2}}}
    )
    assert expanded[0] == ("s:a", "a", {"shared": 1, "extra": 2})
    assert expanded[1] == ("s:b", "b", {"shared": 1})


def test_expand_step_invalid_types_raise():
    orch = RAGOrchestrator(_base_config())
    with pytest.raises(TypeError):
        orch._expand_step({"name": "s", "component": 123})
    with pytest.raises(TypeError):
        orch._expand_step({"name": "s", "component": [123]})


# --------------------------------------------------------------------------- #
# _collect_source_files
# --------------------------------------------------------------------------- #
def test_collect_source_files_dir_file_dict_and_dedup(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.txt").write_text("b")
    orch = RAGOrchestrator(_base_config())
    files = orch._collect_source_files({"sources": str(tmp_path)})
    names = {p.name for p in files}
    assert names == {"a.txt", "b.txt"}
    # single file + dict source + missing path
    one = orch._collect_source_files(
        {"sources": [str(tmp_path / "a.txt"), {"source": str(tmp_path / "a.txt")}, "/no/such/file"]}
    )
    assert any(p.name == "a.txt" for p in one)


# --------------------------------------------------------------------------- #
# manifest / artifact helpers
# --------------------------------------------------------------------------- #
def test_init_manifest_enabled_and_path():
    assert RAGOrchestrator(_base_config())._init_manifest_enabled() is False
    cfg = _base_config(cache={"enabled": True, "manifest_path": "/tmp/m.json"})
    orch = RAGOrchestrator(cfg)
    assert orch._init_manifest_enabled() is True
    assert orch._manifest_path() == Path("/tmp/m.json")
    cfg2 = _base_config(cache={"enabled": True, "features": {"init_manifest": False}})
    assert RAGOrchestrator(cfg2)._init_manifest_enabled() is False


def test_load_manifest_missing_corrupt_valid(tmp_path):
    cfg = _base_config(cache={"enabled": True, "manifest_path": str(tmp_path / "m.json")})
    orch = RAGOrchestrator(cfg)
    assert orch._load_manifest() is None
    (tmp_path / "m.json").write_text("{ not json")
    assert orch._load_manifest() is None
    (tmp_path / "m.json").write_text(json.dumps({"fingerprint": "abc"}))
    assert orch._load_manifest() == {"fingerprint": "abc"}


def test_artifact_ready_dir_and_file(tmp_path):
    f = tmp_path / "f.json"
    f.write_text("x")
    assert RAGOrchestrator._artifact_ready(f) is True
    d = tmp_path / "emb"
    d.mkdir()
    assert RAGOrchestrator._artifact_ready(d) is False
    (d / "index.faiss").write_text("a")
    (d / "index.pkl").write_text("b")
    assert RAGOrchestrator._artifact_ready(d) is True


def test_index_artifacts_exist_no_indexers_is_false():
    assert RAGOrchestrator(_base_config())._index_artifacts_exist() is False


# --------------------------------------------------------------------------- #
# initialize + skip
# --------------------------------------------------------------------------- #
def test_initialize_writes_manifest_then_skips_on_second_run(tmp_path, monkeypatch):
    emb = tmp_path / "emb"

    def _fake_indexer(state, config):
        path = Path(config["indexers"]["embedding"]["path"])
        path.mkdir(parents=True, exist_ok=True)
        (path / "index.faiss").write_text("x")
        (path / "index.pkl").write_text("y")
        return {**state, "indexed": True}

    monkeypatch.setitem(orchestrator_module.REGISTRY, "embedding_indexer", _fake_indexer)

    cfg = _base_config(
        cache={"enabled": True, "manifest_path": str(tmp_path / "manifest.json")},
        indexers={"embedding": {"path": str(emb)}},
        init_pipeline={"steps": [{"name": "index", "component": "embedding_indexer"}]},
    )

    first = RAGOrchestrator(cfg).initialize({"sources": []})
    assert first["init_skipped"] is False
    assert (tmp_path / "manifest.json").exists()

    second_orch = RAGOrchestrator(cfg)
    assert second_orch.can_skip_initialize({"sources": []}) is True
    second = second_orch.initialize({"sources": []})
    assert second["init_skipped"] is True


def test_should_skip_initialize_false_without_manifest_feature():
    orch = RAGOrchestrator(_base_config(init_pipeline={"steps": [{"name": "i", "component": "embedding_indexer"}]}))
    assert orch._should_skip_initialize({"sources": []}) is False


# --------------------------------------------------------------------------- #
# run + partial + intermediate public API
# --------------------------------------------------------------------------- #
def test_run_executes_steps_and_records_timings():
    orch = RAGOrchestrator(_base_config())
    out = orch.run({"query": "hi"})
    assert out["noop_ran"] is True
    assert out["step_timings"][0]["component"] == "noop_component"


def test_execute_steps_partial_phase():
    orch = RAGOrchestrator(_base_config())
    out = orch.execute_steps([{"name": "noop", "component": "noop_component"}], {"query": "q"})
    assert out["noop_ran"] is True


def test_intermediate_public_helpers(tmp_path):
    cfg = _base_config(intermediate={"enabled": True, "path": str(tmp_path)})
    orch = RAGOrchestrator(cfg)
    state = orch.snapshot_intermediate_state({"intermediate_run_id": "r1", "query": "q"})
    assert isinstance(state, dict)
    state = orch.record_intermediate_step(
        {"intermediate_run_id": "r1", "query": "q"},
        phase="run",
        step_name="manual",
        component_name="noop_component",
    )
    orch.finalize_intermediate(state)
    assert (tmp_path / "r1" / "final.json").exists()
