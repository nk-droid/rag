import json

from components.shared_types import RetrievedChunk
from infra.storage.intermediate_store import IntermediateStore


def test_intermediate_store_writes_manifest_step_and_final(tmp_path) -> None:
    store = IntermediateStore(
        {
            "intermediate": {
                "enabled": True,
                "path": str(tmp_path),
                "max_text_chars": 8,
            }
        }
    )
    config = {
        "init_pipeline": {"steps": []},
        "pipeline": {"steps": [{"name": "retrieve", "component": "graph_retriever"}]},
    }
    state = {
        "intermediate_run_id": "debug run",
        "query": "Where is the service?",
        "documents": ["inherited document"],
        "retrieved": [
            RetrievedChunk(
                id="chunk-1",
                text="abcdefghijklmnopqrstuvwxyz",
                score=0.9,
                metadata={"path": "service.py"},
            )
        ],
        "_internal": "should not persist",
    }

    state = store.start_run(state, config)
    before_snapshot = store.snapshot_state(
        {
            "intermediate_run_id": "debug run",
            "query": "Where is the service?",
            "documents": ["inherited document"],
            "_internal": "should not persist",
        }
    )
    store.write_step(
        phase="run",
        step_index=1,
        step_name="retrieve",
        component_name="graph_retriever",
        state=state,
        before_snapshot=before_snapshot,
    )
    store.write_final(state)

    run_path = tmp_path / "debug-run"
    assert (run_path / "manifest.json").exists()
    assert (run_path / "001-run-retrieve-graph_retriever.json").exists()
    assert (run_path / "final.json").exists()

    step_payload = json.loads((run_path / "001-run-retrieve-graph_retriever.json").read_text())
    assert step_payload["added_keys"] == ["retrieved"]
    assert step_payload["changed_keys"] == []
    assert step_payload["removed_keys"] == []
    assert "query" not in step_payload["outputs"]
    assert "documents" not in step_payload["outputs"]
    assert "_internal" not in step_payload["outputs"]
    assert step_payload["outputs"]["retrieved"][0]["id"] == "chunk-1"
    assert step_payload["outputs"]["retrieved"][0]["text"]["text"] == "abcdefgh"

    final_payload = json.loads((run_path / "final.json").read_text())
    assert "documents" not in final_payload["outputs"]
    assert "config" not in final_payload["outputs"]
    assert final_payload["outputs"]["retrieved"][0]["id"] == "chunk-1"


def test_intermediate_store_disabled_writes_nothing(tmp_path) -> None:
    store = IntermediateStore({"intermediate": {"enabled": False, "path": str(tmp_path)}})
    state = store.start_run({"intermediate_run_id": "noop"}, {})

    store.write_final(state)

    assert not any(tmp_path.iterdir())
