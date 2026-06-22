import pytest

from api.routers import experiments as experiment_router
from api.schemas import (
    ExperimentConfigSaveRequest,
    ExperimentConfigValidateRequest,
    ExperimentRunRequest,
    ExperimentRunResponse,
)


EXAMPLE_YAML = """
experiment:
  name: tiny_eval
  dataset: data/raw/eval_set.json
  sources: data/raw/docs-short
  runtime: eval
  env: dev
  parallelism: 1
  metrics:
    - latency_ms
  variants:
    - { name: simple_bm25, pipeline: simple }
"""


@pytest.fixture()
def experiment_config_root(tmp_path, monkeypatch):
    root = tmp_path / "configs" / "experiments"
    root.mkdir(parents=True)
    monkeypatch.setattr(experiment_router, "CONFIG_ROOT", root)
    return root


def test_validate_experiment_config_accepts_valid_yaml() -> None:
    response = experiment_router.validate_experiment_config(
        ExperimentConfigValidateRequest(yaml_text=EXAMPLE_YAML)
    )

    assert response.valid is True
    assert response.config is not None
    assert response.config.name == "tiny_eval"
    assert response.config.variants == ["simple_bm25"]


def test_validate_experiment_config_reports_invalid_yaml() -> None:
    response = experiment_router.validate_experiment_config(
        ExperimentConfigValidateRequest(yaml_text="experiment:\n  name: missing_bits\n")
    )

    assert response.valid is False
    assert response.errors


def test_save_and_list_experiment_config(experiment_config_root) -> None:
    experiment_router.save_experiment_config(
        ExperimentConfigSaveRequest(
            file_name="tiny.yaml",
            yaml_text=EXAMPLE_YAML,
            overwrite=False,
        )
    )

    response = experiment_router.list_experiment_configs()

    assert [config.file for config in response.configs] == ["tiny.yaml"]
    assert response.configs[0].name == "tiny_eval"
    assert (experiment_config_root / "tiny.yaml").exists()


def test_run_existing_experiment_config_loads_selected_yaml(experiment_config_root, monkeypatch) -> None:
    (experiment_config_root / "tiny.yaml").write_text(EXAMPLE_YAML, encoding="utf-8")
    seen = {}

    def fake_run(experiment):
        seen["name"] = experiment.name
        return ExperimentRunResponse(
            name=experiment.name,
            run_id="20260616-000000",
            path="data/experiments/tiny_eval/20260616-000000",
            comparison={},
            warnings=[],
        )

    monkeypatch.setattr(experiment_router, "_run_loaded_experiment", fake_run)

    response = experiment_router.run_experiment_endpoint(
        ExperimentRunRequest(config_file="tiny.yaml")
    )

    assert response.name == "tiny_eval"
    assert seen["name"] == "tiny_eval"
