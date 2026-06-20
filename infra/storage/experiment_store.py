import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None

def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

class ExperimentStore:
    def __init__(self, root: str | Path = "data/experiments") -> None:
        self.root = Path(root)

    def create_run(self, experiment: dict[str, Any]) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_dir = self.root / experiment["name"] / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            run_dir / "manifest.json",
            {
                "schema_version": 1,
                "name": experiment["name"],
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": _git_commit(),
                "experiment": experiment,
            },
        )
        return run_dir

    def write_variant_runs(self, run_dir: Path, result: dict[str, Any]) -> None:
        variant_dir = run_dir / "variants" / result["variant"]
        variant_dir.mkdir(parents=True, exist_ok=True)

        with (variant_dir / "runs.jsonl").open("w", encoding="utf-8") as handle:
            for record in result.get("records", []):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        if result.get("config_snapshot") is not None:
            _write_json(variant_dir / "config.snapshot.json", result["config_snapshot"])

        _write_json(
            variant_dir / "summary.json",
            {
                "variant": result["variant"],
                "pipeline": result.get("pipeline"),
                "workspace_id": result.get("workspace_id"),
                "error": result.get("error"),
                "n_records": len(result.get("records", [])),
                "n_errors": sum(
                    1 for r in result.get("records", []) if r.get("error")
                ),
            },
        )

    def write_variant_metrics(
        self, run_dir: Path, variant: str, metrics: dict[str, Any]
    ) -> None:
        _write_json(run_dir / "variants" / variant / "metrics.json", metrics)

    def write_comparison(self, run_dir: Path, comparison: dict[str, Any]) -> None:
        _write_json(run_dir / "comparison.json", comparison)

    def list_variants(self, run_dir: Path) -> list[str]:
        variants_dir = Path(run_dir) / "variants"
        if not variants_dir.is_dir():
            return []
        return sorted(p.name for p in variants_dir.iterdir() if p.is_dir())

    def load_runs(self, run_dir: Path, variant: str) -> list[dict[str, Any]]:
        path = Path(run_dir) / "variants" / variant / "runs.jsonl"
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def load_metrics(self, run_dir: Path, variant: str) -> dict[str, Any]:
        path = Path(run_dir) / "variants" / variant / "metrics.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def load_manifest(self, run_dir: Path) -> dict[str, Any]:
        path = Path(run_dir) / "manifest.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))