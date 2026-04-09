from pathlib import Path
from typing import List

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}

def deep_merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and k in a:
            a[k] = deep_merge(a[k], v)
        else:
            a[k] = v
    return a

def load_config(paths: List[str]):
    config = {}
    resolved_paths = []

    if BASE_CONFIG_PATH.exists():
        resolved_paths.append(str(BASE_CONFIG_PATH))

    resolved_paths.extend(paths)

    for path in resolved_paths:
        config = deep_merge(config, load_yaml(path))

    return config
