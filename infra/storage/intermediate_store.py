import json
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class IntermediateStore:
    def __init__(self, config: dict[str, Any]) -> None:
        settings = config.get("intermediate", {})
        self.settings = settings if isinstance(settings, dict) else {}
        self.enabled = bool(self.settings.get("enabled", False))
        self.base_path = Path(str(self.settings.get("path", "data/intermediate")))
        self.include_text = bool(self.settings.get("include_text", True))
        self.include_prompt = bool(self.settings.get("include_prompt", True))
        self.include_config = bool(self.settings.get("include_config", False))
        self.include_embeddings = bool(self.settings.get("include_embeddings", False))
        self.max_text_chars = int(self.settings.get("max_text_chars", 4000))
        self.max_list_items = int(self.settings.get("max_list_items", 200))

    def start_run(self, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            return state

        run_id = self._run_id(state)
        run_path = self.base_path / run_id
        run_path.mkdir(parents=True, exist_ok=True)

        state["intermediate_run_id"] = run_id
        state["intermediate_path"] = str(run_path)

        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "path": str(run_path),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "query": self._serialize(state.get("query")),
            "sources": self._serialize(state.get("sources")),
            "init_pipeline": self._serialize(config.get("init_pipeline", {})),
            "pipeline": self._serialize(config.get("pipeline", {})),
        }
        if self.include_config:
            manifest["config"] = self._serialize(config)

        self._write_json(run_path / "manifest.json", manifest)
        return state

    def write_step(
        self,
        *,
        phase: str,
        step_index: int,
        step_name: str,
        component_name: str,
        state: dict[str, Any],
        before_snapshot: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return

        run_path = self._run_path(state)
        if run_path is None:
            return

        baseline = before_snapshot
        if baseline is None:
            baseline = self._serialize_step_state(state)
        step_outputs = self._step_outputs(baseline, state)
        filename = (
            f"{step_index:03d}-"
            f"{self._slug(phase)}-"
            f"{self._slug(step_name)}-"
            f"{self._slug(component_name)}.json"
        )
        self._write_json(
            run_path / filename,
            {
                "phase": phase,
                "step_index": step_index,
                "step_name": step_name,
                "component": component_name,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "added_keys": step_outputs["added_keys"],
                "changed_keys": step_outputs["changed_keys"],
                "removed_keys": step_outputs["removed_keys"],
                "outputs": step_outputs["outputs"],
            },
        )

    def write_final(self, state: dict[str, Any]) -> None:
        if not self.enabled:
            return

        run_path = self._run_path(state)
        if run_path is None:
            return

        self._write_json(
            run_path / "final.json",
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "outputs": self._serialize_final_outputs(state),
            },
        )

    def _serialize_state(self, state: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in state.items():
            if str(key).startswith("_"):
                continue
            if key == "config" and not self.include_config:
                payload[key] = self._config_summary(value)
                continue
            if key == "prompt" and not self.include_prompt:
                payload[key] = self._summary(value)
                continue
            payload[key] = self._serialize(value)
        return payload

    def snapshot_state(self, state: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            return {}
        return self._serialize_step_state(state)

    def _serialize_step_state(self, state: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in state.items():
            key_str = str(key)
            if not self._include_step_key(key_str):
                continue
            payload[key_str] = self._serialize(value)
        return payload

    def _step_outputs(
        self,
        before_snapshot: dict[str, Any],
        after_state: dict[str, Any],
    ) -> dict[str, Any]:
        after_snapshot = self._serialize_step_state(after_state)

        added_keys = sorted(key for key in after_snapshot if key not in before_snapshot)
        changed_keys = sorted(
            key
            for key in after_snapshot
            if key in before_snapshot and after_snapshot[key] != before_snapshot[key]
        )
        removed_keys = sorted(key for key in before_snapshot if key not in after_snapshot)
        output_keys = added_keys + changed_keys

        return {
            "added_keys": added_keys,
            "changed_keys": changed_keys,
            "removed_keys": removed_keys,
            "outputs": {key: after_snapshot[key] for key in output_keys},
        }

    def _serialize_final_outputs(self, state: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in state.items():
            key_str = str(key)
            if not self._include_final_key(key_str):
                continue
            payload[key_str] = self._serialize(value)
        return payload

    def _serialize(self, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float)):
            return value

        if isinstance(value, str):
            return self._serialize_text(value)

        if isinstance(value, Path):
            return str(value)

        if is_dataclass(value) and not isinstance(value, type):
            return self._serialize(asdict(value))

        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            return self._serialize(model_dump())

        dict_method = getattr(value, "dict", None)
        if callable(dict_method):
            return self._serialize(dict_method())

        if isinstance(value, dict):
            return {
                str(key): self._serialize(item)
                for key, item in value.items()
                if self._include_key(str(key))
            }

        if isinstance(value, (list, tuple, set)):
            items = list(value)
            serialized = [
                self._serialize(item)
                for item in items[: self.max_list_items]
            ]
            if len(items) > self.max_list_items:
                serialized.append(
                    {
                        "__truncated_items__": len(items) - self.max_list_items,
                    }
                )
            return serialized

        if hasattr(value, "template"):
            return {
                "__type__": type(value).__name__,
                "template": self._serialize_text(str(getattr(value, "template", ""))),
                "input_variables": self._serialize(getattr(value, "input_variables", [])),
                "partial_variables": self._serialize(getattr(value, "partial_variables", {})),
            }

        return self._summary(value)

    def _serialize_text(self, text: str) -> str | dict[str, Any]:
        if not self.include_text:
            return {
                "__omitted__": "text",
                "chars": len(text),
            }

        if len(text) <= self.max_text_chars:
            return text

        return {
            "text": text[: self.max_text_chars],
            "__truncated_chars__": len(text) - self.max_text_chars,
        }

    def _include_key(self, key: str) -> bool:
        lowered = key.lower()
        if not self.include_embeddings and ("embedding" in lowered or lowered == "vector"):
            return False
        return lowered not in {"llm", "client", "store", "_vector_store"}

    def _include_step_key(self, key: str) -> bool:
        if key.startswith("_"):
            return False
        if key in {"config", "intermediate_run_id", "intermediate_path", "step_timings"}:
            return False
        return self._include_key(key)

    def _include_final_key(self, key: str) -> bool:
        if not self._include_step_key(key):
            return False
        return key not in {
            "chunks",
            "context",
            "data_sources",
            "dense_retrieved",
            "documents",
            "graph_expanded",
            "index_records",
            "prompt",
            "sources",
            "sparse_retrieved",
        }

    def _config_summary(self, value: Any) -> Any:
        if not isinstance(value, dict):
            return self._summary(value)
        return {
            "app": self._serialize(value.get("app", {})),
            "models": self._serialize(value.get("models", {})),
            "init_pipeline": self._serialize(value.get("init_pipeline", {})),
            "pipeline": self._serialize(value.get("pipeline", {})),
        }

    @staticmethod
    def _summary(value: Any) -> dict[str, str]:
        return {
            "__type__": type(value).__name__,
            "__repr__": repr(value),
        }

    def _run_id(self, state: dict[str, Any]) -> str:
        configured = str(state.get("intermediate_run_id") or "").strip()
        if configured:
            return self._slug(configured)
        return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")

    def _run_path(self, state: dict[str, Any]) -> Path | None:
        raw_path = state.get("intermediate_path")
        if not raw_path:
            return None
        return Path(str(raw_path))

    @staticmethod
    def _slug(value: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
        slug = re.sub(r"-+", "-", slug).strip("-._")
        return slug or "run"

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(f"{path.suffix}.tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(path)
