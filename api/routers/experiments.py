import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
import yaml

from api.schemas import (
    ExperimentConfigItem,
    ExperimentConfigSaveRequest,
    ExperimentConfigSaveResponse,
    ExperimentConfigsResponse,
    ExperimentConfigValidateRequest,
    ExperimentConfigValidateResponse,
    ExperimentDetailResponse,
    ExperimentListItem,
    ExperimentQueriesResponse,
    ExperimentRunRequest,
    ExperimentRunResponse,
    ExperimentsResponse,
)
from components.evaluation.dataset import check_metric_requirements, load_dataset
from components.evaluation.metrics import aggregate
from components.evaluation.ragas_metrics import RAGAS_METRIC_NAMES, ragas_aggregate_batch
from infra.storage.experiment_store import ExperimentStore
from pipeline.config import load_config
from pipeline.experiment.config import Experiment, experiment_from_mapping, load_experiment
from pipeline.experiment.report import build_comparison, render_markdown
from pipeline.experiment.runner import run_experiment

router = APIRouter(prefix="/api/experiments", tags=["experiments"])

EXPERIMENT_ROOT = Path("data/experiments")
CONFIG_ROOT = Path("configs/experiments")

def _store() -> ExperimentStore:
    return ExperimentStore(EXPERIMENT_ROOT)

def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}

def _run_dir(name: str, run_id: str) -> Path:
    root = EXPERIMENT_ROOT.resolve()
    candidate = (EXPERIMENT_ROOT / name / run_id).resolve()
    if root not in candidate.parents:
        raise HTTPException(status_code=400, detail="Invalid experiment path.")
    if not (candidate / "manifest.json").exists():
        raise HTTPException(status_code=404, detail="Experiment run not found.")
    return candidate

def _query_count(store: ExperimentStore, run_dir: Path, variants: list[str]) -> int:
    for variant in variants:
        runs = store.load_runs(run_dir, variant)
        if runs:
            return len(runs)
    return 0

def _safe_config_file(file_name: str) -> Path:
    name = Path(file_name).name
    if not name.endswith((".yaml", ".yml")):
        raise HTTPException(status_code=400, detail="Experiment config must be a .yaml or .yml file.")
    root = CONFIG_ROOT.resolve()
    candidate = (CONFIG_ROOT / name).resolve()
    if candidate.parent != root:
        raise HTTPException(status_code=400, detail="Invalid experiment config path.")
    return candidate

def _experiment_from_yaml_text(yaml_text: str, label: str = "<draft>") -> Experiment:
    try:
        raw = yaml.safe_load(yaml_text) or {}
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError("Experiment YAML must be a mapping.")
    return experiment_from_mapping(raw, label=label)

def _config_item(file_name: str, experiment: Experiment | None = None, error: str | None = None) -> ExperimentConfigItem:
    if experiment is None:
        return ExperimentConfigItem(file=file_name, valid=False, error=error)
    return ExperimentConfigItem(
        file=file_name,
        name=experiment.name,
        dataset=experiment.dataset,
        sources=experiment.sources,
        runtime=experiment.runtime,
        env=experiment.env,
        parallelism=experiment.parallelism,
        metrics=list(experiment.metrics),
        variants=[variant.name for variant in experiment.variants],
        valid=True,
    )

def _load_config_item(path: Path) -> ExperimentConfigItem:
    try:
        return _config_item(path.name, load_experiment(path))
    except Exception as error:
        return _config_item(path.name, None, str(error))

def _compute_and_store(
    store: ExperimentStore,
    run_dir: Path,
    metric_names: list[str],
) -> dict[str, Any]:
    lexical_names = [name for name in metric_names if name not in RAGAS_METRIC_NAMES]
    ragas_names = [name for name in metric_names if name in RAGAS_METRIC_NAMES]

    variants = store.list_variants(run_dir)
    records_by_variant = {variant: store.load_runs(run_dir, variant) for variant in variants}
    metrics_by_variant = {
        variant: aggregate(records_by_variant[variant], lexical_names)
        for variant in variants
    }

    if ragas_names:
        ragas_config = load_config([])
        ragas_by_variant = ragas_aggregate_batch(
            records_by_variant,
            ragas_names,
            config=ragas_config,
        )
        for variant in variants:
            metrics_by_variant[variant].update(ragas_by_variant.get(variant, {}))

    for variant in variants:
        store.write_variant_metrics(run_dir, variant, metrics_by_variant[variant])

    comparison = build_comparison(run_dir, metric_names, store=store)
    store.write_comparison(run_dir, comparison)
    (run_dir / "comparison.md").write_text(render_markdown(comparison), encoding="utf-8")
    return comparison

def _run_loaded_experiment(experiment: Experiment) -> ExperimentRunResponse:
    try:
        samples = load_dataset(experiment.dataset)
        warnings = check_metric_requirements(samples, experiment.metrics)
        results = run_experiment(
            experiment.to_dict(),
            [sample.to_record() for sample in samples],
        )

        store = _store()
        run_dir = store.create_run(experiment.to_dict())
        for result in results:
            store.write_variant_runs(run_dir, result)
            if result.get("error"):
                warnings.append(f"{result['variant']}: {result['error']}")

        comparison = _compute_and_store(store, run_dir, experiment.metrics)
        return ExperimentRunResponse(
            name=experiment.name,
            run_id=run_dir.name,
            path=str(run_dir),
            status="completed",
            comparison=comparison,
            warnings=warnings,
        )
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Experiment run failed: {error}") from error

@router.get("/configs", response_model=ExperimentConfigsResponse)
def list_experiment_configs() -> ExperimentConfigsResponse:
    configs = [
        _load_config_item(path)
        for path in sorted(CONFIG_ROOT.glob("*.y*ml"))
        if path.is_file()
    ]
    return ExperimentConfigsResponse(configs=configs)

@router.post("/configs/validate", response_model=ExperimentConfigValidateResponse)
def validate_experiment_config(payload: ExperimentConfigValidateRequest) -> ExperimentConfigValidateResponse:
    try:
        experiment = _experiment_from_yaml_text(payload.yaml_text)
    except Exception as error:
        return ExperimentConfigValidateResponse(valid=False, errors=[str(error)])
    return ExperimentConfigValidateResponse(
        valid=True,
        config=_config_item("<draft>", experiment),
    )

@router.post("/configs", response_model=ExperimentConfigSaveResponse)
def save_experiment_config(payload: ExperimentConfigSaveRequest) -> ExperimentConfigSaveResponse:
    path = _safe_config_file(payload.file_name)
    if path.exists() and not payload.overwrite:
        raise HTTPException(status_code=409, detail="Experiment config already exists.")

    try:
        experiment = _experiment_from_yaml_text(payload.yaml_text, label=path.name)
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload.yaml_text, encoding="utf-8")
    return ExperimentConfigSaveResponse(config=_config_item(path.name, experiment))

@router.post("/run", response_model=ExperimentRunResponse)
def run_experiment_endpoint(payload: ExperimentRunRequest) -> ExperimentRunResponse:
    if payload.yaml_text:
        if payload.save_as:
            save_experiment_config(
                ExperimentConfigSaveRequest(
                    file_name=payload.save_as,
                    yaml_text=payload.yaml_text,
                    overwrite=payload.overwrite,
                )
            )
        try:
            experiment = _experiment_from_yaml_text(payload.yaml_text, label=payload.save_as or "<draft>")
        except Exception as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return _run_loaded_experiment(experiment)

    if not payload.config_file:
        raise HTTPException(status_code=400, detail="Select an experiment config or provide YAML text.")

    path = _safe_config_file(payload.config_file)
    try:
        experiment = load_experiment(path)
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return _run_loaded_experiment(experiment)

@router.get("", response_model=ExperimentsResponse)
def list_experiments() -> ExperimentsResponse:
    store = _store()
    items: list[ExperimentListItem] = []
    manifests = sorted(
        EXPERIMENT_ROOT.glob("*/*/manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    for manifest_path in manifests:
        run_dir = manifest_path.parent
        name = run_dir.parent.name
        run_id = run_dir.name
        manifest = _read_json(manifest_path)
        comparison = _read_json(run_dir / "comparison.json")
        variants = store.list_variants(run_dir)
        status = "completed" if comparison else "running"
        items.append(
            ExperimentListItem(
                name=name,
                run_id=run_id,
                path=str(run_dir),
                created_at_utc=str(manifest.get("created_at_utc") or ""),
                status=status,
                variants=len(variants),
                queries=_query_count(store, run_dir, variants),
                metrics=list(comparison.get("metrics", [])),
                best=dict(comparison.get("best", {})),
            )
        )

    return ExperimentsResponse(experiments=items)

@router.get("/{name}/{run_id}", response_model=ExperimentDetailResponse)
def get_experiment(name: str, run_id: str) -> ExperimentDetailResponse:
    run_dir = _run_dir(name, run_id)
    store = _store()
    summaries = [
        _read_json(run_dir / "variants" / variant / "summary.json")
        for variant in store.list_variants(run_dir)
    ]
    return ExperimentDetailResponse(
        name=name,
        run_id=run_id,
        path=str(run_dir),
        manifest=_read_json(run_dir / "manifest.json"),
        comparison=_read_json(run_dir / "comparison.json"),
        summaries=summaries,
    )

@router.get("/{name}/{run_id}/queries", response_model=ExperimentQueriesResponse)
def get_experiment_queries(
    name: str,
    run_id: str,
    limit: int = 25,
) -> ExperimentQueriesResponse:
    run_dir = _run_dir(name, run_id)
    store = _store()
    variants = store.list_variants(run_dir)
    runs_by_variant = {variant: store.load_runs(run_dir, variant) for variant in variants}
    max_len = max((len(records) for records in runs_by_variant.values()), default=0)
    count = max(0, min(limit, max_len))

    queries: list[dict[str, object]] = []
    for index in range(count):
        first_record = next(
            (
                records[index]
                for records in runs_by_variant.values()
                if index < len(records)
            ),
            {},
        )
        variants_payload: dict[str, object] = {}
        for variant, records in runs_by_variant.items():
            if index >= len(records):
                continue
            record = records[index]
            variants_payload[variant] = {
                "answer": record.get("answer", ""),
                "contexts": record.get("contexts", []),
                "latency_ms": record.get("latency_ms"),
                "error": record.get("error"),
            }
        queries.append(
            {
                "index": index,
                "question": first_record.get("question", ""),
                "ground_truth": first_record.get("ground_truth"),
                "reference_contexts": first_record.get("reference_contexts"),
                "variants": variants_payload,
            }
        )

    return ExperimentQueriesResponse(name=name, run_id=run_id, queries=queries)