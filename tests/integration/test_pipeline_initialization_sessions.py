from datetime import datetime, timezone

import pytest

import api.pipeline_service as pipeline_service
from api.pipeline_service import PipelineValidationError, initialize_pipeline, run_pipeline
from api.schemas import PipelineRequest, PipelineSelection, SourceRecord
from components.ingestion.ingestion_schema import SourceDocument


class _Loader:
    def load_sources(self, sources):
        return [
            SourceDocument(
                text="hello world",
                source=sources[0].path,
                metadata={"source_id": sources[0].id},
            )
        ]


class _Orchestrator:
    def __init__(self, config):
        self.config = config

    def initialize(self, state):
        return {
            **state,
            "step_timings": [
                {
                    "phase": "init",
                    "step_name": "chunk",
                    "component": "recursive_chunker",
                    "latency_ms": 1.0,
                }
            ],
        }

    def run(self, state):
        return {
            **state,
            "answer": "done",
            "retrieved": [],
            "ranked": [],
            "step_timings": [
                *state.get("step_timings", []),
                {
                    "phase": "run",
                    "step_name": "retrieve_local",
                    "component": "coarse_retriever",
                    "latency_ms": 1.0,
                },
            ],
        }


def _source() -> SourceRecord:
    return SourceRecord(
        id="src_1",
        name="Example",
        source_type="file",
        loader="document_loader",
        path="/tmp/example.md",
        created_at=datetime.now(timezone.utc),
    )


def _request(*, top_k: int = 5, initialization_id: str | None = None) -> PipelineRequest:
    return PipelineRequest(
        query="What is this?",
        source_ids=["src_1"],
        selection=PipelineSelection(
            chunking=["recursive_chunker"],
            retrieval=["coarse_retriever"],
            generation=["prompt_builder", "llm_generator"],
        ),
        top_k=top_k,
        initialization_id=initialization_id,
        skip_initialization=initialization_id is not None,
    )


@pytest.fixture(autouse=True)
def _fake_orchestrator(monkeypatch):
    pipeline_service._INITIALIZED_PIPELINES.clear()
    monkeypatch.setattr(pipeline_service, "RAGOrchestrator", _Orchestrator)
    yield
    pipeline_service._INITIALIZED_PIPELINES.clear()


def test_skip_initialization_requires_initialization_id() -> None:
    request = _request()
    request.skip_initialization = True

    with pytest.raises(PipelineValidationError, match="has not been initialized"):
        run_pipeline(request, [_source()], _Loader())


def test_initialized_pipeline_id_allows_query_run() -> None:
    initialized = initialize_pipeline(_request(), [_source()], _Loader())

    request = _request(initialization_id=initialized.initialization_id)
    _, _, result = run_pipeline(request, [_source()], _Loader())

    assert result["answer"] == "done"


def test_changed_configuration_rejects_stale_initialization_id() -> None:
    initialized = initialize_pipeline(_request(top_k=5), [_source()], _Loader())

    with pytest.raises(PipelineValidationError, match="configuration changed"):
        run_pipeline(
            _request(top_k=10, initialization_id=initialized.initialization_id),
            [_source()],
            _Loader(),
        )
