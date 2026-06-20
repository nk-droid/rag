import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from pipeline.config import deep_merge, load_config
from pipeline.results import extract_answer, extract_contexts
from pipeline.validator import validate_config
from pipeline.workspace import apply_workspace

REPO_ROOT = Path(__file__).resolve().parents[2]

def _config_path(*parts: str) -> str:
    return str(REPO_ROOT.joinpath(*parts))

def build_variant_config(variant: dict[str, Any], experiment: dict[str, Any]) -> dict[str, Any]:
    config = load_config(
        [
            _config_path("configs", "pipeline", f"{variant['pipeline']}.yaml"),
            _config_path("configs", "runtime", f"{experiment['runtime']}.yaml"),
            _config_path("configs", "env", f"{experiment['env']}.yaml"),
        ]
    )
    overrides = variant.get("config") or {}
    if overrides:
        config = deep_merge(config, overrides)
    return apply_workspace(config)

def run_variant(
    variant: dict[str, Any],
    experiment: dict[str, Any],
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "variant": variant["name"],
        "pipeline": variant.get("pipeline"),
        "workspace_id": None,
        "config_snapshot": None,
        "records": [],
        "error": None,
    }

    try:
        config = build_variant_config(variant, experiment)
        errors = validate_config(config)
        if errors:
            result["error"] = "invalid config: " + "; ".join(errors)
            return result

        result["workspace_id"] = config.get("workspace", {}).get("id")
        result["config_snapshot"] = config

        from pipeline.orchestrator import RAGOrchestrator

        orchestrator = RAGOrchestrator(config)
        sources = experiment["sources"]
        orchestrator.initialize({"query": samples[0]["question"], "sources": sources})
    
    except Exception as error:
        result["error"] = f"init failed: {error}"
        return result

    for sample in samples:
        record: dict[str, Any] = {
            "question": sample["question"],
            "ground_truth": sample.get("ground_truth"),
            "reference_contexts": sample.get("reference_contexts"),
            "answer": "",
            "contexts": [],
            "latency_ms": None,
            "error": None,
        }
        
        try:
            started = time.perf_counter()
            state = orchestrator.run({"query": sample["question"], "sources": sources})
            record["latency_ms"] = (time.perf_counter() - started) * 1000.0
            record["answer"] = extract_answer(state)
            record["contexts"] = extract_contexts(state)
        
        except Exception as error:
            record["error"] = str(error)
        
        result["records"].append(record)

    return result

def run_experiment(
    experiment: dict[str, Any],
    samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    variants = experiment["variants"]
    parallelism = max(1, int(experiment.get("parallelism", 1)))

    if parallelism == 1 or len(variants) == 1:
        return [run_variant(variant, experiment, samples) for variant in variants]

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=parallelism) as executor:
        futures = {
            executor.submit(run_variant, variant, experiment, samples): variant["name"]
            for variant in variants
        }
        for future in as_completed(futures):
            results.append(future.result())

    order = {variant["name"]: index for index, variant in enumerate(variants)}
    results.sort(key=lambda r: order.get(r["variant"], len(order)))
    return results