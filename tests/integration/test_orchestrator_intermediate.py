import json

import pipeline.orchestrator as orchestrator_module
from pipeline.orchestrator import RAGOrchestrator


def test_orchestrator_records_intermediate_steps(tmp_path, monkeypatch) -> None:
    def _noop_step(state, config):
        payload = dict(state)
        payload["noop_ran"] = True
        return payload

    monkeypatch.setitem(orchestrator_module.REGISTRY, "noop_component", _noop_step)

    config = {
        "runtime": {"mode": "api"},
        "intermediate": {
            "enabled": True,
            "path": str(tmp_path),
        },
        "init_pipeline": {"steps": []},
        "pipeline": {
            "steps": [
                {"name": "noop", "component": "noop_component"},
            ]
        },
    }
    orchestrator = RAGOrchestrator(config)
    state = {
        "intermediate_run_id": "orchestrator-test",
        "query": "hello",
    }

    state = orchestrator.run(state)

    run_path = tmp_path / "orchestrator-test"
    assert state["intermediate_path"] == str(run_path)
    assert (run_path / "manifest.json").exists()
    assert (run_path / "001-run-noop-noop_component.json").exists()
    assert (run_path / "final.json").exists()

    step_payload = json.loads((run_path / "001-run-noop-noop_component.json").read_text())
    assert step_payload["added_keys"] == ["noop_ran"]
    assert step_payload["outputs"] == {"noop_ran": True}
    assert "query" not in step_payload["outputs"]

    final_payload = json.loads((run_path / "final.json").read_text())
    assert final_payload["outputs"]["noop_ran"] is True
