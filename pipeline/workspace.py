from copy import deepcopy
from pathlib import Path
from typing import Any

from infra.cache.cache_keys import stable_hash

WORKSPACE_ROOT = "data/workspaces"

MANAGED_PATHS: list[tuple[str, str]] = [
    ("indexers.embedding.path", "faiss_index"),
    ("indexers.coarse.path", "coarse_index.json"),
    ("indexers.graph.path", "repo_graph.json"),
    ("retrieval.graph.path", "repo_graph.json"),
    ("cache.manifest_path", "init_manifest.json"),
    ("intermediate.path", "intermediate"),
]

def workspace_id(config: dict[str, Any]) -> str:
    models = config.get("models", {})
    embedding = models.get("embedding", {}) if isinstance(models, dict) else {}
    return stable_hash(
        {
            "init_pipeline": config.get("init_pipeline", {}),
            "pipeline": config.get("pipeline", {}),
            "chunking": config.get("chunking", {}),
            "embedding": embedding,
        }
    )[:16]

def _get_dotted(config: dict[str, Any], dotted: str) -> Any:
    node: Any = config
    for key in dotted.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(key)
        if node is None:
            return None
    return node

def _set_dotted(config: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = config
    for key in keys[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[keys[-1]] = value

def apply_workspace(config: dict[str, Any]) -> dict[str, Any]:
    ws_cfg = config.get("workspace")
    ws_cfg = ws_cfg if isinstance(ws_cfg, dict) else {}
    if ws_cfg.get("enabled") is False:
        return config

    wid = workspace_id(config)
    root = str(ws_cfg.get("root", WORKSPACE_ROOT))
    base = Path(root) / wid

    config = deepcopy(config)
    for dotted, default_name in MANAGED_PATHS:
        current = _get_dotted(config, dotted)
        name = Path(str(current)).name if current else default_name
        _set_dotted(config, dotted, str(base / name))

    config["workspace"] = {**ws_cfg, "id": wid, "root": root, "path": str(base)}
    return config
