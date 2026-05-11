# Modular RAG

A config-driven retrieval-augmented generation framework. Define a pipeline as a YAML step sequence - the orchestrator resolves, builds, and wires every component at runtime. Swap models, retrievers, or postprocessors by editing config.

```bash
python cli.py --pipeline custom --runtime cli --env dev
```

---

## Why This Exists

Most RAG implementations are tightly coupled - swapping the retriever means editing generation code, changing the LLM means touching the pipeline. This system treats every component (chunker, retriever, ranker, generator, critic) as a registered, independently configurable unit. The orchestrator wires them together at runtime based purely on YAML.

---

## Architecture

```
CLI  /  FastAPI  /  Notebook
        │
        ▼
RAGOrchestrator          ← merges base + env + pipeline + runtime YAML
 │
 ├── initialize()        ← load → chunk → index  (skipped on fingerprint cache hit)
 │
 └── run()               ← clean → rewrite → expand → retrieve → rank → generate → critique → refine → parse
         │
         ▼
     REGISTRY            ← 39 component keys, each bound to a factory + handler
         │
         ▼
     Components          ← stateless, independently configurable units
         │
         ▼
     Infra               ← LLM, embeddings, cache, vector store
```

**Config resolution:** `configs/base.yaml` → `configs/env/{dev|prod}.yaml` → `configs/pipeline/{name}.yaml` → `configs/runtime/{cli|api|notebook}.yaml`. Each layer overrides the previous. Components receive already-merged values, never raw config dicts.

---

## Active Pipeline: `custom`

The `custom` pipeline is the primary maintained path. It runs end-to-end locally with Ollama.

**Init phase** (skipped on cache hit):
```
directory_loader → recursive_chunker → coarse_indexer
```

**Query phase:**
```
query_cleaner
    → query_rewriter          ← LLM rewrites query for better retrieval
    → multi_query_generator   ← LLM expands into multiple query variants
    → coarse_retriever        ← BM25 sparse retrieval
    → rank_fusion             ← combines result sets
    → embedding_ranker        ← reranks by embedding similarity
    → prompt_builder          ← YAML template + Pydantic schema injection
    → llm_generator
    → self_critic             ← LLM checks answer grounding against context
    → refiner                 ← LLM rewrites answer if critic flags issues
    → output_parser
```

> `hybrid_retriever` (dense + sparse) is wired and available but not active in the current `custom.yaml`. Enabling it also requires `embedding_indexer` in the init phase.

### Smart re-indexing

Before each init, the orchestrator fingerprints the configuration:

```python
fingerprint = stable_hash({
    "sources":             [file_signature(path) for path in source_files],
    "chunking":            config["chunking"],
    "embedding_model":     config["models"]["embedding"],
    "embedding_index_path": str(embedding_index_path),
    "coarse_index_path":    str(coarse_index_path),
})
```

If the fingerprint matches the saved manifest and index artifacts exist, the entire load → chunk → index phase is skipped. Changing a source file, the chunking config, or the embedding model triggers a full re-index automatically.

---

## Components

### ✅ Implemented

| Domain | Components |
|--------|-----------|
| **Ingestion** | `DirectoryLoader`, `DocumentLoader`, `MarkdownLoader`, `TextLoader`, `SourceNormalizer` |
| **Chunking** | `RecursiveChunker`, `SemanticChunker` (LLM-guided boundary detection) |
| **Indexing** | `EmbeddingIndexer` (FAISS), `CoarseIndexer` (BM25, JSON-persisted) |
| **Query** | `QueryCleaner`, `QueryRewriter` (LLM), `MultiQueryGenerator` (LLM) |
| **Retrieval** | `CoarseRetriever` (BM25), `FineRetriever` (FAISS dense), `HybridRetriever` |
| **Ranking** | `RankFusion`, `EmbeddingRanker`, `CrossEncoderRanker` (`ms-marco-MiniLM-L-6-v2`) |
| **Context** | `ContextBuilder`, `ContextMerger`, `ContextTruncator` |
| **Generation** | `PromptBuilder`, `Generator` / `LLMGenerator`, `OutputParser` |
| **Postprocessing** | `SelfCritic` (LLM grounding check), `Refiner` (LLM rewrite on critic fail) |
| **Memory** | `MemoryStore`, `MemoryWriter`, `MemoryFilter` |
| **External** | `ExternalRetriever` (Tavily web search) |
| **Evaluation** | `Evaluator` (base), `RagasEvaluator` (faithfulness, answer relevancy, context precision/recall) |

### 🔲 Planned (registry keys exist; implementations are stubs)

- `LateChunker`
- `GraphRetriever`
- `MemoryRetriever`
- `ColBERTRanker`
- `StreamingGenerator`

---

## Infrastructure

### LLM Providers

| Provider | Notes |
|----------|-------|
| Ollama | Default - `llama3.2:latest` |
| OpenAI | Requires `OPENAI_API_KEY` |
| Anthropic | Requires `ANTHROPIC_API_KEY` |
| HuggingFace | Local inference |

Switch via `configs/base.yaml` - no other changes needed:

```yaml
models:
  llm:
    provider: anthropic
    model_name: claude-sonnet-4-20250514
```

### Embedding Backends

| Provider | Notes |
|----------|-------|
| Ollama | Default - `qwen3-embedding:4b` |
| OpenAI | `text-embedding-*` models |
| HuggingFace | Local sentence-transformers |

### Cache

Two backends sharing a `BaseCache` interface:

**In-memory (default):** TTL + LRU eviction, bounded by `max_entries`. Configurable per pipeline stage.

**Redis:** JSON-serialised values with namespaced key tracking and scoped `clear()`.

```yaml
# configs/base.yaml
cache:
  type: in_memory      # in_memory | redis
  default_ttl_sec: 900
  max_entries: 2000
  features:
    retrieval: true
    ranking_embeddings: true
    prompt: true
    generation: false
```

---

## Setup

**Requirements:** Python 3.11+ · Ollama (for default config)

```bash
git clone https://github.com/nk-droid/rag
cd rag

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pull default models:
```bash
ollama pull llama3.2:latest
ollama pull qwen3-embedding:4b
```

For OpenAI, Anthropic, or Tavily web retrieval, create a `.env`:
```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...              # only if using ExternalRetriever
REDIS_URL=redis://localhost:6379     # only if using Redis cache
```

---

## Running

### CLI

```bash
python cli.py --pipeline custom --runtime cli --env dev
```

The CLI prompts interactively for the source directory and query.

| Flag | Options | Effect |
|------|---------|--------|
| `--pipeline` | `custom`, `simple`, `advanced`, `debug` | Which pipeline YAML to load |
| `--runtime` | `cli`, `api`, `notebook` | Selects runtime adapter (Rich progress vs. silent vs. notebook-friendly) |
| `--env` | `dev`, `prod` | Sets log level, cache TTL, temperature |
| `-e`, `--eval` | flag | Reserved for post-run evaluation (currently a no-op; wiring is commented in `cli.py`) |

### API + Frontend

```bash
python scripts/run_api.py            # FastAPI on :8000
cd frontend && npm install && npm run dev   # Vite dev server on :5173
```

The API exposes `/api/sources`, `/api/components/catalog`, and `/api/pipelines/{preview,run}`. The React + MUI frontend lets you register source paths or upload files, compose a pipeline visually, and run it against the orchestrator.

### Evaluation

```bash
python -c "from scripts.evaluate import run_evaluation, load_samples; ..."
```

`scripts/evaluate.py` loads `data/raw/eval_set.json`, runs each question through a silent-runtime orchestrator, and feeds the answers + retrieved contexts to `RagasEvaluator`.

### Tests

```bash
pytest
```

Component factory, behaviour, and module-import smoke tests live under `tests/`.

---

## Configuration

### Enabling hybrid retrieval

To switch from BM25-only to hybrid (dense + sparse) in `custom.yaml`:

```yaml
# 1. Uncomment embedding_indexer in init_pipeline
- name: index
  component:
    - embedding_indexer
    - coarse_indexer

# 2. Switch retriever in pipeline
- name: retriever
  component: hybrid_retriever
```

### Switching ranker

```yaml
- name: rerank
  component: cross_encoder_ranker   # or: embedding_ranker
```

### Running multiple components on one step

```yaml
- name: index
  component:
    - embedding_indexer
    - coarse_indexer
```

The orchestrator runs both and merges results into the shared state.

---

## Project Structure

```
rag/
├── components/
│   ├── chunking/          RecursiveChunker, SemanticChunker, LateChunker (stub)
│   ├── context/           ContextBuilder, ContextMerger, ContextTruncator
│   ├── evaluation/        Evaluator (base), RagasEvaluator
│   ├── generation/        PromptBuilder, Generator, OutputParser, StreamingGenerator (stub)
│   ├── indexer/           EmbeddingIndexer (FAISS), CoarseIndexer (BM25)
│   ├── ingestion/         DirectoryLoader, DocumentLoader, MarkdownLoader, TextLoader, SourceNormalizer
│   ├── memory/            MemoryStore, MemoryWriter, MemoryFilter
│   ├── postprocessing/    SelfCritic, Refiner
│   ├── query/             QueryCleaner, QueryRewriter, MultiQueryGenerator
│   ├── ranking/           EmbeddingRanker, CrossEncoderRanker, RankFusion, ColBERTRanker (stub)
│   └── retrieval/         CoarseRetriever, FineRetriever, HybridRetriever, ExternalRetriever,
│                          GraphRetriever (stub), MemoryRetriever (stub)
├── configs/
│   ├── base.yaml
│   ├── env/               dev.yaml, prod.yaml
│   ├── pipeline/          custom (maintained), simple, advanced, debug
│   └── runtime/           cli, api, notebook
├── infra/
│   ├── cache/             InMemoryCache (TTL+LRU), RedisCache
│   ├── embeddings/        Provider factory (OpenAI, HuggingFace, Ollama)
│   ├── llm/               Provider factory (OpenAI, Anthropic, HuggingFace, Ollama)
│   ├── logging/           Rich / silent / simple runtime adapters, structured formatters
│   ├── observability/     Reserved for tracing
│   └── storage/           FAISS store, vector store factory
├── pipeline/
│   ├── orchestrator.py    Init + run orchestration, fingerprint-based cache
│   ├── registry.py        39 component bindings
│   ├── registry_handlers.py
│   ├── registry_utils.py
│   ├── component_factories.py
│   └── config.py          Layered YAML resolution
├── api/                   FastAPI app (sources, components catalog, pipeline preview/run)
│   ├── main.py
│   ├── routers/           components.py, pipelines.py, sources.py
│   ├── catalog.py         Component metadata for the frontend
│   ├── loader_service.py
│   ├── pipeline_service.py
│   ├── source_store.py
│   └── schemas.py
├── frontend/              React + Vite + MUI client
│   └── src/               App.tsx, api.ts, theme.ts, types.ts
├── scripts/               run_api.py (uvicorn launcher), evaluate.py
├── tests/                 pytest: factories, behaviors, module-import smoke
├── data/
│   ├── raw/               Source documents + eval_set.json
│   ├── uploads/           Uploaded sources from the API
│   ├── processed/
│   └── indices/           FAISS + BM25 artifacts + init_manifest.json
├── cli.py
├── CONTRIBUTING.md
└── requirements.txt
```

---

## Known Limitations

| Issue | Detail |
|-------|--------|
| Several registry keys map to stub components | `late_chunker`, `graph_retriever`, `memory_retriever`, `colbert_ranker`, `streaming_generator` resolve, but the underlying classes only define `NotImplementedError`-shaped stubs |
| `infra/observability/` | Reserved directory, no implementation yet |

---

## Roadmap

- [ ] `LateChunker` - embedding-aware chunking over full-document representations
- [ ] `GraphRetriever` - graph traversal over document relationship graph
- [ ] `StreamingGenerator` - token streaming end-to-end, including through the API
- [ ] Tracing in `infra/observability/`
- [ ] `docker-compose.yml` - zero-dependency local setup

---

## Stack

**Backend**
- Python 3.11+
- LangChain (`langchain`, `langchain-{ollama,openai,anthropic,community,pinecone,tavily,text-splitters}`)
- FAISS (`faiss-cpu`), `rank_bm25`, `sentence-transformers`
- Pydantic, PyYAML, python-dotenv
- Ragas (evaluation), `datasets`
- Redis (optional cache backend)
- Rich (terminal runtime), pytest
- FastAPI + Uvicorn (API), `python-multipart` (uploads)

**Frontend**
- React 18 + Vite 6 + TypeScript
- MUI 5 (`@mui/material`, `@mui/icons-material`, `@emotion/{react,styled}`)

**External services**
- Ollama (default LLM + embeddings)
- Tavily (web retrieval via `ExternalRetriever`)
- Pinecone (vector store option, wired but not active in `custom`)