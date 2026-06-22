# Contributing

Contributions are welcome - bug fixes, new components, config improvements, and documentation. This guide explains how the codebase is structured so you can contribute without breaking things.

---

## Table of Contents

- [Project Layout](#project-layout)
- [Setup](#setup)
- [How Components Work](#how-components-work)
- [Adding a New Component](#adding-a-new-component)
- [Good First Contributions](#good-first-contributions)
- [Code Style](#code-style)
- [Submitting a PR](#submitting-a-pr)

---

## Project Layout

The parts of the codebase you'll most likely touch:

```
components/         Domain logic - implement your component here
pipeline/
  component_factories.py   Wire your component to a factory function
  registry.py              Bind the factory to a YAML step name
  registry_handlers.py     Define how your component is called in pipeline state
configs/pipeline/   Add your component to a pipeline step sequence
```

---

## Setup

```bash
git clone https://github.com/nk-droid/rag
cd rag

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For local inference (default config):
```bash
ollama pull llama3.2:latest
ollama pull qwen3-embedding:4b
```

Run the pipeline to confirm your environment is working:
```bash
rag --pipeline custom --runtime cli --env dev
```

(`pip install -e` registers the `rag` and `rag-eval` console scripts. Without an
install you can run the equivalents with `python -m clis.cli` and `python -m clis.eval_cli`.)

---

## How Components Work

Every component follows the same four-step registration pattern:

```
1. Implement   →   components/<domain>/your_component.py
2. Factory     →   pipeline/component_factories.py
3. Registry    →   pipeline/registry.py  +  registry_handlers.py
4. Config      →   configs/pipeline/*.yaml
```

When you run `rag --pipeline custom`, the orchestrator reads the YAML, looks up each step name in `REGISTRY`, calls the bound factory to build the component, and passes it to the handler which reads/writes the shared pipeline `state` dict. Your component never touches config loading, caching, or orchestration - it just does its job.

The registry also caches built components by config hash, so factories are called at most once per config:

```python
# pipeline/registry.py
_COMPONENT_CACHE: dict[tuple[str, str], Any] = {}

def _build_component(name: str, config: dict) -> Any:
    cache_key = (name, _config_cache_key(config))
    if cache_key not in _COMPONENT_CACHE:
        _COMPONENT_CACHE[cache_key] = COMPONENT_FACTORIES[name](config)
    return _COMPONENT_CACHE[cache_key]

def bind(component_name, handler):
    return lambda state, config: handler(_build_component(component_name, config), state, config)
```

---

## Adding a New Component

This walkthrough uses a new retriever as the example. The same four steps apply to any domain.

### Step 1 - Implement the component

Every retriever implements `BaseRetriever`:

```python
# components/retrieval/base_retriever.py
class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        ...
```

Create your file:

```python
# components/retrieval/your_retriever.py
from components.shared_types import RetrievedChunk
from components.retrieval.base_retriever import BaseRetriever

class YourRetriever(BaseRetriever):
    def __init__(self, your_param: str) -> None:
        super().__init__(store=None)
        self.your_param = your_param

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        # your implementation
        return []
```

Base classes by domain:

| Domain | Base class | Required method |
|--------|-----------|-----------------|
| Retrieval | `BaseRetriever` | `retrieve(query, top_k) -> list[RetrievedChunk]` |
| Ranking | `BaseRanker` | `rank(query, candidates) -> list[RetrievedChunk]` |
| Ingestion | `BaseLoader` | `load(source) -> list[SourceDocument]` |

Key shared types from `components/shared_types.py`:

```python
@dataclass(slots=True)
class RetrievedChunk:
    id: str
    text: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class Chunk:
    text: str
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class EvaluationResult:
    metric: str
    score: float
    details: dict[str, Any] = field(default_factory=dict)
```

### Step 2 - Export from the package

```python
# components/retrieval/__init__.py
from components.retrieval.your_retriever import YourRetriever
```

### Step 3 - Register a factory

```python
# pipeline/component_factories.py
from components.retrieval.your_retriever import YourRetriever

COMPONENT_FACTORIES: dict[str, ComponentFactory] = {
    # ... existing entries ...
    "your_retriever": lambda config: YourRetriever(
        your_param=config.get("retrieval", {}).get("your_param", "default")
    ),
}
```

The factory receives the full merged config dict. Pull only the values your component needs. Config reads belong here - not inside the component itself.

### Step 4 - Bind a handler and register

If your component is called the same way as an existing one, reuse the handler:

```python
# pipeline/registry.py
"your_retriever": bind("your_retriever", _retrieve_with),
```

`_retrieve_with` calls `component.retrieve(query, top_k)` and writes results to `state["retrieved"]`. If your component needs different pipeline state behaviour, add a handler to `registry_handlers.py`:

```python
# pipeline/registry_handlers.py
def _your_retrieve_with(component: YourRetriever, state: dict, config: dict) -> dict:
    query = state["query"]
    top_k = config.get("retrieval", {}).get("top_k", 5)
    results = component.retrieve(query, top_k=top_k)
    return {**state, "retrieved": results}
```

Then bind it:

```python
# pipeline/registry.py
"your_retriever": bind("your_retriever", _your_retrieve_with),
```

### Step 5 - Use it in a pipeline config

```yaml
# configs/pipeline/custom.yaml
pipeline:
  steps:
    - name: retriever
      component: your_retriever
```

Run with:
```bash
rag --pipeline custom --runtime cli --env dev
```

---

## Good First Contributions

These are unimplemented components with a clear contract and well-defined scope.

---

### 1. `MemoryRetriever`

**File:** `components/retrieval/memory_retriever.py`

The registry key exists but the class is a stub. A reasonable starting scope:

- Read from the conversational `MemoryStore` (`components/memory/`)
- At query time, retrieve the most relevant prior turns/notes for the current query
- Return them as `RetrievedChunk`s so they can be merged into context

---

### 2. `LateChunker`

**File:** `components/chunking/late_chunker.py`

Chunking that operates on full-document token embeddings before splitting, preserving semantic context across chunk boundaries. A working approach:

- Embed the full document token-by-token
- Find semantic boundary positions by detecting embedding similarity drops
- Split at those positions

---

## Code Style

- **Python 3.12+.** Use `list[X]`, `dict[str, Any]`, `X | None` - not `List`, `Dict`, `Optional`.
- **Type hints on all public methods.**
- **No print statements.** Use `self.runtime.log(...)` or the structured logger in `infra/logging/`.
- **Config reads in the factory, not the component.** Components receive typed values, not raw config dicts.
- **No hardcoded paths.** Resolve paths relative to config values or the repo root - never absolute local paths.

---

## Submitting a PR

1. Fork the repo and create a branch: `git checkout -b feat/graph-retriever`
2. Make your changes
3. Run the pipeline end-to-end to confirm nothing is broken: `rag --pipeline custom --runtime cli --env dev`
4. Open a PR with:
   - What the change does and why
   - Any config keys it reads and their defaults
   - For new components: example YAML showing how to use it in a pipeline

If you're implementing one of the planned components from the Roadmap in README, mention it so it can be tracked.