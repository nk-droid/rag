import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from infra.cache.cache_keys import file_signature, stable_hash
from infra.logging.runtime.factory import get_runtime
from infra.storage.intermediate_store import IntermediateStore
from pipeline.registry import REGISTRY

# Components whose init artifacts gate the init-skip / manifest logic.
_INDEXER_COMPONENTS = {
    "embedding_indexer",
    "coarse_indexer",
    "repo_graph_indexer",
    "graph_indexer",
}

class RAGOrchestrator:
    def __init__(self, config):
        self.config = config
        self.runtime = get_runtime(config)
        self.setup_steps = config.get("init_pipeline", {}).get("steps", [])
        self.steps = config["pipeline"]["steps"]
        self.intermediate_store = IntermediateStore(config)
        self._started = False

    def _cache_config(self) -> dict[str, Any]:
        cache_cfg = self.config.get("cache", {})
        return cache_cfg if isinstance(cache_cfg, dict) else {}

    def _init_manifest_enabled(self) -> bool:
        cache_cfg = self._cache_config()
        if not bool(cache_cfg.get("enabled", False)):
            return False

        features = cache_cfg.get("features", {})
        if isinstance(features, dict) and "init_manifest" in features:
            return bool(features.get("init_manifest"))
        return True

    def _manifest_path(self) -> Path:
        cache_cfg = self._cache_config()
        configured = cache_cfg.get("manifest_path", "data/indices/init_manifest.json")
        return Path(str(configured))

    def _init_indexer_components(self) -> list[str]:
        names: list[str] = []
        for step in self.setup_steps:
            component = step.get("component") if isinstance(step, dict) else None
            components = component if isinstance(component, list) else [component]
            for name in components:
                if name in _INDEXER_COMPONENTS:
                    names.append(str(name))
        return names

    def _index_paths(self) -> dict[str, Path]:
        from pipeline.registry_utils import _get_index_path

        return {
            name: Path(_get_index_path(self.config, name))
            for name in self._init_indexer_components()
        }

    @staticmethod
    def _artifact_ready(path: Path) -> bool:
        if path.is_dir():
            return (path / "index.faiss").exists() and (path / "index.pkl").exists()
        return path.exists()

    def _index_artifacts_exist(self) -> bool:
        paths = self._index_paths()
        if not paths:
            return False
        return all(self._artifact_ready(path) for path in paths.values())

    def _collect_source_files(self, state: dict[str, Any]) -> list[Path]:
        raw_sources = state.get("sources") or state.get("data_sources") or []
        if isinstance(raw_sources, (str, Path)):
            raw_sources = [raw_sources]

        resolved: list[Path] = []
        for source in raw_sources:
            source_path: Path | None = None
            if isinstance(source, Path):
                source_path = source
            elif isinstance(source, str):
                source_path = Path(source)
            elif isinstance(source, dict):
                source_value = source.get("source")
                if isinstance(source_value, str):
                    source_path = Path(source_value)

            if source_path is None:
                continue

            if source_path.is_dir():
                for path in sorted(source_path.rglob("*")):
                    if path.is_file():
                        resolved.append(path.resolve())
            elif source_path.exists() and source_path.is_file():
                resolved.append(source_path.resolve())
            else:
                resolved.append(source_path.resolve())

        unique: dict[str, Path] = {}
        for path in resolved:
            unique[str(path)] = path
        return [unique[key] for key in sorted(unique.keys())]

    def _init_payload(self, state: dict[str, Any]) -> dict[str, Any]:
        source_files = self._collect_source_files(state)
        source_signatures = [file_signature(path) for path in source_files]
        index_paths = {name: str(path) for name, path in self._index_paths().items()}
        return {
            "sources": source_signatures,
            "chunking": self.config.get("chunking", {}),
            "embedding_model": self.config.get("models", {}).get("embedding", {}),
            "init_components": sorted(index_paths.keys()),
            "index_paths": index_paths,
        }

    def _init_fingerprint(self, state: dict[str, Any]) -> str:
        return stable_hash(self._init_payload(state))

    def _load_manifest(self) -> dict[str, Any] | None:
        path = self._manifest_path()
        if not path.exists():
            return None

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

        return payload if isinstance(payload, dict) else None

    def _write_manifest(self, payload: dict[str, Any]) -> None:
        path = self._manifest_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _should_skip_initialize(self, state: dict[str, Any]) -> bool:
        if not self._init_manifest_enabled() or not self.setup_steps:
            return False

        manifest = self._load_manifest()
        if not manifest:
            return False

        expected = manifest.get("fingerprint")
        actual = self._init_fingerprint(state)
        if expected != actual:
            return False

        return self._index_artifacts_exist()

    def _ensure_started(self, msg: str):
        if not self._started:
            self.runtime.start(msg)
            self._started = True

    def _expand_step(self, step: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
        name = step["name"]
        raw_component = step["component"]
        step_options = {k: v for k, v in step.items() if k not in {"name", "component", "options"}}
        per_component_options = step.get("options", {})

        if isinstance(raw_component, str):
            return [(name, raw_component, step_options)]

        if not isinstance(raw_component, list):
            raise TypeError(
                f"Step '{name}' has invalid component type {type(raw_component)!r}. "
                "Expected str or list[str]."
            )

        expanded: list[tuple[str, str, dict[str, Any]]] = []
        for component_name in raw_component:
            if not isinstance(component_name, str):
                raise TypeError(
                    f"Step '{name}' has invalid list component type {type(component_name)!r}. "
                    "Expected list[str]."
                )

            component_step_options = dict(step_options)
            if isinstance(per_component_options, dict):
                scoped = per_component_options.get(component_name, {})
                if isinstance(scoped, dict):
                    component_step_options.update(scoped)

            expanded.append((f"{name}:{component_name}", component_name, component_step_options))

        return expanded

    def _ensure_intermediate_run(self, state: dict[str, Any]) -> dict[str, Any]:
        return self.intermediate_store.start_run(state, self.config)

    def _record_intermediate_step(
        self,
        state: dict[str, Any],
        *,
        phase: str,
        step_name: str,
        component_name: str,
        before_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = self._ensure_intermediate_run(state)
        step_index = int(state.get("_intermediate_step_index", 0)) + 1
        state["_intermediate_step_index"] = step_index
        self.intermediate_store.write_step(
            phase=phase,
            step_index=step_index,
            step_name=step_name,
            component_name=component_name,
            state=state,
            before_snapshot=before_snapshot,
        )
        return state

    def _execute_steps(self, steps, state, phase: str = "run"):
        state = self._ensure_intermediate_run(state)
        for step in steps:
            expanded_steps = self._expand_step(step)
            for run_name, component_name, step_options in expanded_steps:
                component = REGISTRY[component_name]
                before_snapshot = self.intermediate_store.snapshot_state(state)

                step_state = dict(state)
                step_state["_step"] = {
                    "name": run_name,
                    "component": component_name,
                    **step_options,
                }

                step_started = time.perf_counter()
                state = self.runtime.run_step(
                    run_name,
                    component,
                    step_state,
                    self.config
                )
                step_latency_ms = (time.perf_counter() - step_started) * 1000.0

                state.pop("_step", None)
                state.setdefault("step_timings", []).append(
                    {
                        "phase": phase,
                        "step_name": run_name,
                        "component": component_name,
                        "latency_ms": step_latency_ms,
                    }
                )
                state = self._record_intermediate_step(
                    state,
                    phase=phase,
                    step_name=run_name,
                    component_name=component_name,
                    before_snapshot=before_snapshot,
                )

        return state

    def initialize(self, state):
        self._ensure_started("Initializing RAG System")
        state = self._ensure_intermediate_run(state)

        if self._should_skip_initialize(state):
            self.runtime.log("Skipping init pipeline (cache manifest hit).")
            state["init_skipped"] = True
            state["config"] = self.config

            if self._started:
                self.runtime.stop("System Ready")
                self._started = False
            return state

        for step in self.setup_steps:
            for run_name, _, _ in self._expand_step(step):
                self.runtime.add_step(run_name)

        if self.setup_steps:
            state = self._execute_steps(self.setup_steps, state, phase="init")
            if self._init_manifest_enabled():
                init_payload = self._init_payload(state)
                self._write_manifest(
                    {
                        "schema_version": 1,
                        "fingerprint": stable_hash(init_payload),
                        **init_payload,
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                )
            state["init_skipped"] = False

        if self._started:
            self.runtime.stop("System Ready")
            self._started = False

        return state

    def can_skip_initialize(self, state: dict[str, Any]) -> bool:
        return self._should_skip_initialize(state)

    def run(self, state):
        self._ensure_started("Inferencing the System")
        state = self._ensure_intermediate_run(state)
        for step in self.steps:
            for run_name, _, _ in self._expand_step(step):
                self.runtime.add_step(run_name)
        state = self._execute_steps(self.steps, state, phase="run")
        self.intermediate_store.write_final(state)

        if self._started:
            self.runtime.stop("")
            self._started = False

        return state

    def execute_steps(self, steps, state):
        return self._execute_steps(steps, state, phase="partial")

    def record_intermediate_step(
        self,
        state: dict[str, Any],
        *,
        phase: str,
        step_name: str,
        component_name: str,
        before_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._record_intermediate_step(
            state,
            phase=phase,
            step_name=step_name,
            component_name=component_name,
            before_snapshot=before_snapshot,
        )

    def finalize_intermediate(self, state: dict[str, Any]) -> None:
        self.intermediate_store.write_final(state)

    def snapshot_intermediate_state(self, state: dict[str, Any]) -> dict[str, Any]:
        state = self._ensure_intermediate_run(state)
        return self.intermediate_store.snapshot_state(state)
