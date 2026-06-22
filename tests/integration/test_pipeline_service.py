"""Integration tests for pipeline_service run/initialize/stream flows with the
orchestrator, runtime config, and component factories stubbed."""
import types
from datetime import datetime, timezone

import pytest

from api import pipeline_service as ps
from api.schemas import PipelineRequest, PipelineSelection, SourceRecord


def _sources():
    return [
        SourceRecord(
            id="s1", name="n", source_type="file", loader="text_loader", path="/p",
            created_at=datetime.now(timezone.utc),
        )
    ]


def _request(generation=("llm_generator",)):
    return PipelineRequest(
        query="q",
        source_ids=["s1"],
        selection=PipelineSelection(
            chunking=["recursive_chunker"], retrieval=["coarse_retriever"], generation=list(generation)
        ),
    )


class _Loader:
    def load_sources(self, sources):
        return [types.SimpleNamespace(text="doc", metadata={})]


class _FakeOrch:
    def __init__(self, config):
        pass

    def initialize(self, state):
        return {**state, "init_skipped": False, "step_timings": []}

    def run(self, state):
        return {**state, "answer": "the answer", "retrieved": [], "step_timings": []}


@pytest.fixture(autouse=True)
def _stub_runtime(monkeypatch):
    monkeypatch.setattr(
        ps, "_runtime_config_for_plan",
        lambda plan: {"runtime": {"mode": "api"}, "intermediate": {"enabled": False},
                      "init_pipeline": {"steps": []}, "pipeline": {"steps": []}},
    )
    monkeypatch.setattr(ps, "RAGOrchestrator", _FakeOrch)


def test_run_pipeline_happy():
    plan, warnings, result = ps.run_pipeline(_request(), _sources(), _Loader())
    assert result["answer"] == "the answer"
    assert result["run_id"]


def test_run_pipeline_no_sources():
    with pytest.raises(ps.PipelineValidationError):
        ps.run_pipeline(_request(), [], _Loader())


def test_run_pipeline_no_documents():
    class _Empty:
        def load_sources(self, sources):
            return []

    with pytest.raises(ps.PipelineValidationError):
        ps.run_pipeline(_request(), _sources(), _Empty())


def test_initialize_pipeline_happy():
    response = ps.initialize_pipeline(_request(), _sources(), _Loader())
    assert response.initialization_id.startswith("init_")
    assert response.document_count == 1


def test_run_pipeline_skip_initialization_requires_record():
    req = _request()
    req.skip_initialization = True
    with pytest.raises(ps.PipelineValidationError):
        ps.run_pipeline(req, _sources(), _Loader())


# --------------------------------------------------------------------------- #
# streaming
# --------------------------------------------------------------------------- #
class _StreamOrch:
    def __init__(self, config):
        pass

    def initialize(self, state):
        return state

    def execute_steps(self, steps, state):
        # pre-steps must leave a prompt + context in state
        return {**state, "prompt": types.SimpleNamespace(template="t"), "context": "ctx", "step_timings": []}

    def snapshot_intermediate_state(self, state):
        return {}

    def record_intermediate_step(self, state, **kwargs):
        state.setdefault("step_timings", []).append({"phase": "stream", "step_name": "streaming_generator", "component": "streaming_generator", "latency_ms": 1.0})
        return state

    def finalize_intermediate(self, state):
        pass


def test_stream_pipeline_run(monkeypatch):
    monkeypatch.setattr(ps, "RAGOrchestrator", _StreamOrch)
    monkeypatch.setitem(
        ps.COMPONENT_FACTORIES,
        ps.STREAMING_GENERATOR_KEY,
        lambda config: types.SimpleNamespace(stream=lambda prompt, inputs: iter(["he", "llo"])),
    )
    events = list(ps.stream_pipeline_run(_request(generation=["streaming_generator"]), _sources(), _Loader()))
    types_seen = [e[0] for e in events]
    assert "plan" in types_seen and "token" in types_seen and "done" in types_seen
    tokens = "".join(e[1]["piece"] for e in events if e[0] == "token")
    assert tokens == "hello"


def test_stream_requires_streaming_generator():
    with pytest.raises(ps.PipelineValidationError):
        list(ps.stream_pipeline_run(_request(), _sources(), _Loader()))
