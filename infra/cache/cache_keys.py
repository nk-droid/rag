import dataclasses
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

def _normalize(value: Any) -> Any:
    """Convert values into deterministic JSON-serializable structures."""
    if dataclasses.is_dataclass(value):
        return _normalize(dataclasses.asdict(value))

    if isinstance(value, Path):
        return str(value.resolve())

    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()

    if isinstance(value, Mapping):
        normalized_items = [(str(key), _normalize(val)) for key, val in value.items()]
        normalized_items.sort(key=lambda item: item[0])
        return {key: val for key, val in normalized_items}

    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]

    if isinstance(value, (set, frozenset)):
        normalized_items = [_normalize(item) for item in value]
        return sorted(normalized_items, key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False))

    if isinstance(value, bytes):
        return value.hex()

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _normalize(model_dump())

    dict_method = getattr(value, "dict", None)
    if callable(dict_method):
        return _normalize(dict_method())

    if hasattr(value, "__dict__"):
        return _normalize(vars(value))

    return value


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Serialize payload deterministically for stable hashing."""
    normalized = _normalize(payload)
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def stable_hash(payload: Mapping[str, Any], algorithm: str = "sha256") -> str:
    """Return a deterministic hash for the given payload."""
    encoded = canonical_json(payload).encode("utf-8")
    return hashlib.new(algorithm, encoded).hexdigest()


def _sanitize_key_part(part: str) -> str:
    return (
        part.strip()
        .replace(" ", "_")
        .replace(":", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def make_cache_key(
    namespace: str,
    version: str,
    env: str,
    feature: str,
    payload: Mapping[str, Any],
) -> str:
    """Build a namespaced cache key from metadata + payload hash."""
    parts = [
        _sanitize_key_part(namespace),
        _sanitize_key_part(version),
        _sanitize_key_part(env),
        _sanitize_key_part(feature),
        stable_hash(payload),
    ]
    return ":".join(parts)


def text_hash(text: str, algorithm: str = "sha256") -> str:
    return hashlib.new(algorithm, text.encode("utf-8")).hexdigest()


def file_signature(path: str | Path) -> dict[str, Any]:
    """Return stable metadata for a file path."""
    file_path = Path(path)
    resolved = file_path.resolve()

    if not resolved.exists():
        return {
            "path": str(resolved),
            "exists": False,
            "size": None,
            "mtime_ns": None,
        }

    stat = resolved.stat()
    return {
        "path": str(resolved),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def fingerprint_files(paths: Sequence[str | Path], algorithm: str = "sha256") -> str:
    signatures = [file_signature(path) for path in paths]
    signatures.sort(key=lambda item: item["path"])
    encoded = canonical_json({"files": signatures}).encode("utf-8")
    return hashlib.new(algorithm, encoded).hexdigest()
