import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from infra.cache.cache_keys import file_signature, stable_hash
from pipeline.registry import REGISTRY
from infra.logging.runtime.factory import get_runtime

class RAGOrchestrator:
    def __init__(self, config):
        self.config = config
        self.runtime = get_runtime(config)
        self.setup_steps = config.get("init_pipeline", {}).get("steps", [])
        self.steps = config["pipeline"]["steps"]
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

    def _embedding_index_path(self) -> Path:
        vector_store = self.config.get("vector_store", {})
        embedding_cfg = vector_store.get("embedding_indexer", {})
        if isinstance(embedding_cfg, dict) and embedding_cfg.get("path"):
            return Path(str(embedding_cfg["path"]))

        legacy = vector_store.get("path")
        if legacy:
            return Path(str(legacy))
        return Path("data/indices/faiss_index")

    def _coarse_index_path(self) -> Path:
        vector_store = self.config.get("vector_store", {})
        coarse_cfg = vector_store.get("coarse_indexer", {})
        if isinstance(coarse_cfg, dict) and coarse_cfg.get("path"):
            return Path(str(coarse_cfg["path"]))

        legacy = self.config.get("coarse_index", {}).get("path")
        if legacy:
            return Path(str(legacy))
        return Path("data/indices/coarse_index.json")

    def _index_artifacts_exist(self) -> bool:
        embedding_path = self._embedding_index_path()
        coarse_path = self._coarse_index_path()

        if embedding_path.is_dir():
            embedding_ready = (
                (embedding_path / "index.faiss").exists()
                and (embedding_path / "index.pkl").exists()
            )
        else:
            embedding_ready = embedding_path.exists()

        return embedding_ready and coarse_path.exists()

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
        return {
            "sources": source_signatures,
            "chunking": self.config.get("chunking", {}),
            "embedding_model": self.config.get("models", {}).get("embedding", {}),
            "embedding_index_path": str(self._embedding_index_path()),
            "coarse_index_path": str(self._coarse_index_path()),
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

    def _execute_steps(self, steps, state):
        for step in steps:
            expanded_steps = self._expand_step(step)
            for run_name, component_name, step_options in expanded_steps:
                component = REGISTRY[component_name]

                step_state = dict(state)
                step_state["_step"] = {
                    "name": run_name,
                    "component": component_name,
                    **step_options,
                }

                state = self.runtime.run_step(
                    run_name,
                    component,
                    step_state,
                    self.config
                )

                state.pop("_step", None)

        return state

    def initialize(self, state):
        self._ensure_started("Initializing RAG System")

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
            state = self._execute_steps(self.setup_steps, state)
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

    def run(self, state):
        self._ensure_started("Inferencing the System")
        for step in self.steps:
            for run_name, _, _ in self._expand_step(step):
                self.runtime.add_step(run_name)
        state = self._execute_steps(self.steps, state)

        if self._started:
            self.runtime.stop("")
            self._started = False

        return state
