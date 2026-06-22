# Modular RAG

A configuration-driven Retrieval-Augmented Generation framework where a "RAG type"
is a YAML file, not a code branch. The same orchestrator runs everything from a
one-line BM25 pipeline to a repo-aware Hybrid GraphRAG, and a separate evaluation
CLI runs multiple pipelines in parallel and compares them on metrics.

```bash
rag --pipeline custom --source data/raw/docs --query "How does indexing work?"
```

---

## 1. What this project is

This is a **composable RAG engine**. Every stage — ingestion, chunking, indexing,
query transformation, retrieval, ranking, context building, generation, and
post-processing — is a swappable component. A pipeline is declared as an ordered
list of steps in a config file; the orchestrator wires components together through
a shared `state` dictionary.

Two goals drive the design:

1. **Run any RAG type from config.** If the components exist, you can arrange them
   into a pipeline and run it. A static contract validator checks the arrangement
   *before* execution, so an invalid pipeline fails with a clear message instead of
   a mid-run `KeyError`.
2. **Compare RAG variants on metrics.** A dedicated experiment runner executes
   multiple pipelines over one dataset, isolates each variant's index/cache,
   stores results, and reports metrics like recall, precision, and faithfulness
   side by side.

Configs are layered and deep-merged: `base.yaml` (defaults for every component) →
`pipeline/<name>.yaml` (the steps) → `runtime/<name>.yaml` (output/determinism) →
`env/<name>.yaml` (environment). Anything in a later layer overrides earlier ones.
Components receive already-merged values, never raw config dicts.

---

## 2. Supported RAG modes

Each mode is a pipeline config in [`configs/pipeline/`](configs/pipeline/). Modes
are not hard-coded categories — they are illustrative compositions you can copy and
edit.

| Mode | Config | Retrieval strategy | Notable stages |
|------|--------|--------------------|----------------|
| Simple document RAG | `simple.yaml` | BM25 (sparse) | clean → retrieve → generate |
| Custom | `custom.yaml` | coarse + rank fusion | query rewrite, multi-query, embedding rerank, self-critic, refine |
| Advanced | `advanced.yaml` | hybrid (BM25 + dense) | cross-encoder rerank, critic + refiner |
| Debug | `debug.yaml` | sparse, verbose | per-step tracing for development |
| Web-fallback RAG | `external_fallback.yaml` | local BM25, Tavily web fallback | tops up results only when local is thin |
| Repo Hybrid GraphRAG | `repo_hybrid_graph.yaml` | hybrid + graph expansion | repo loader, code-aware chunking, code-graph index, graph expander |

**Composable capabilities** available to any mode: query cleaning, query rewriting,
multi-query expansion, dense/sparse/graph retrieval, RRF rank fusion, embedding /
cross-encoder reranking, context building + truncation, self-critique +
refinement, conversational memory, and streaming generation.

**Pluggable backends:**
- **LLM providers:** Ollama (default), OpenAI, Anthropic, HuggingFace
- **Embedding providers:** Ollama (default), OpenAI, HuggingFace
- **Vector store:** FAISS
- **Cache:** in-memory or Redis
- **Web search:** Tavily

---

## 3. Interface support matrix

| Capability | CLI (`rag`) | Eval CLI (`rag-eval`) | HTTP API | Web UI (`frontend/`) |
|------------|:---:|:---:|:---:|:---:|
| Run a single pipeline | ✅ | — | ✅ `POST /api/pipelines/run` | ✅ |
| Streaming generation | — | — | ✅ `POST /api/pipelines/stream` (SSE) | ✅ |
| List / preview pipelines | ✅ `--list-pipelines` | — | ✅ `POST /api/pipelines/preview` | ✅ |
| Validate config (pre-flight) | ✅ `--validate-only` | implicit per variant | ✅ (preview) | ✅ |
| Clone & index a Git repo | ✅ `--repo-url` | via cloned path in `sources` | — | — |
| Register / upload sources | manual path | — | ✅ `/api/sources/*` | ✅ |
| Component catalog | — | — | ✅ `GET /api/components/catalog` | ✅ |
| List / create prompts | — | — | ✅ `/api/prompts` | ✅ |
| Run experiments (multi-variant) | — | ✅ `run` | — | — |
| Recompute metrics / report | — | ✅ `metrics`, `report` | — | — |
| Per-step intermediate snapshots | ✅ `--save-intermediate` | — | ✅ (returned) | ✅ |
| Health check | — | — | ✅ `GET /health` | — |

The CLI is the primary surface for running and validating pipelines; the Eval CLI
owns comparison; the API + React/MUI frontend offer interactive composition.

---

## 4. Architecture

```
┌──────────── interfaces ─────────────────────────────────────────┐
│  clis/ (rag · rag-eval)   api/ (FastAPI)   frontend/ (React+MUI)  │
└───────────────┬─────────────────────────────────────────────────┘
                │
        configs/  (base → pipeline → runtime → env, deep-merged)
                │
                ▼
┌──────────────── pipeline/ ──────────────────────────────────────┐
│  orchestrator      – runs init_pipeline then pipeline steps      │
│  registry          – component name → handler binding (44 keys)  │
│  registry_handlers – per-component state read/write logic        │
│  contracts         – declared requires/produces per component    │
│  validator         – static pre-flight check of a composed config│
│  workspace         – per-config index/artifact isolation         │
│  results           – normalized answer/context extraction        │
│  experiment/       – config, parallel runner, comparison report  │
└───────────────┬─────────────────────────────────────────────────┘
                │  builds via pipeline/component_factories.py
                ▼
┌──────────────── components/ ─────────────────────────────────────┐
│  ingestion · chunking · indexer · retrieval · ranking ·          │
│  context · generation · postprocessing · memory · evaluation     │
└───────────────┬──────────────────────────────────────────────────┘
                │
                ▼
┌──────────────── infra/ ──────────────────────────────────────────┐
│  llm (providers) · embeddings (providers) · storage (faiss,      │
│  intermediate, experiment) · cache (in-memory/redis) · logging   │
└──────────────────────────────────────────────────────────────────┘
```

**Execution model.** The orchestrator runs two phases. `init_pipeline` ingests,
chunks, and indexes sources (cached and skipped on re-runs via a fingerprint
manifest). `pipeline` runs the query-time steps. Components communicate only
through a `state` dict; each component's read/write keys are declared in
`pipeline/contracts.py` and enforced statically by `pipeline/validator.py`.

**Workspace isolation.** `pipeline/workspace.py` rewrites every index/manifest path
to `data/workspaces/<id>/`, where `<id>` is a hash of the pipeline shape + chunking
+ embedding model. Different configs never clobber each other's indexes, and
identical configs reuse them.

**Smart re-indexing.** Before each init, the orchestrator fingerprints the source
files, chunking config, embedding model, and index paths. If the fingerprint
matches the saved manifest and index artifacts exist, the entire load → chunk →
index phase is skipped (`init_skipped: true`). Changing a source file, the chunking
config, or the embedding model triggers a full re-index automatically.

---

## 5. Setup

**Requirements:** Python 3.12+ · Ollama (for the default config)

```bash
git clone https://github.com/nk-droid/rag
cd rag

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The editable install registers two console scripts — `rag` (run/validate pipelines)
and `rag-eval` (experiments). Without installing, run them as `python -m clis.cli`
and `python -m clis.eval_cli`.

Pull the local default models (from `configs/base.yaml`):

```bash
ollama pull llama3.2:latest          # secondary LLM (query rewrite / critique / refine)
ollama pull qwen3-embedding:0.6b     # document embeddings
ollama pull qwen3-embedding:4b       # rerank embeddings
```

The primary generation LLM defaults to `gpt-oss:120b-cloud`, an Ollama **cloud**
model — run `ollama signin` to use it (no pull needed). For a fully local setup,
point `models.llm.model_name` in `configs/base.yaml` at `llama3.2:latest` instead.

For OpenAI, Anthropic, or Tavily web retrieval, create a `.env`:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...               # only if using ExternalRetriever / external_fallback
REDIS_URL=redis://localhost:6379      # only if using the Redis cache backend
```

Switch providers in `configs/base.yaml` — no code changes needed:

```yaml
models:
  llm:
    provider: anthropic               # ollama | openai | anthropic | huggingface
    model_name: claude-sonnet-4-20250514
```

### Verify your setup

After a clean clone, these five commands exercise the whole stack end to end —
dependency install, static validation, the test suite, a document-RAG run, and a
repo GraphRAG run:

```bash
make setup        # install Python dependencies
make validate     # statically validate the bundled pipelines (no LLM)
make test         # run the pytest suite
make demo-doc     # document RAG over data/raw/docs-short
make demo-repo    # clone + index a Git repo, then answer a change-impact question
```

`setup`, `validate`, and `test` need no model backend. `demo-doc` and `demo-repo`
require a running Ollama (or a configured provider) and network access for the
repo clone.

---

## 6. Quickstart: document RAG

Run a pipeline against a folder of documents:

```bash
# See available RAG types
rag --list-pipelines

# Validate a config without running it (instant, no LLM)
rag --pipeline advanced --validate-only

# Run end to end
rag \
  --pipeline simple \
  --source data/raw/docs \
  --query "What is the role of indexing in RAG before answer generation?" \
  --show-state
```

The first run indexes into `data/workspaces/<id>/`; re-running the same pipeline
reuses that index (`init_skipped: true`). The CLI prompts interactively for the
source and query when they are omitted.

### CLI flags

| Flag | Options | Effect |
|------|---------|--------|
| `--pipeline` | `simple`, `custom`, `advanced`, `debug`, `external_fallback`, `repo_hybrid_graph` | Which pipeline YAML to load (defaults to `repo_hybrid_graph` with `--repo-url`, else `custom`) |
| `--runtime` | `cli`, `api`, `notebook`, `eval` | Runtime adapter (Rich progress vs. silent vs. notebook-friendly) |
| `--env` | `dev`, `prod` | Log level, cache TTL, temperature |
| `--list-pipelines` | flag | List available pipeline configs and exit |
| `--validate-only` | flag | Statically validate the composed pipeline (component contracts) and exit |
| `--query` | string | Question to ask (skips the interactive prompt) |
| `--source` | path | Local file or directory source |
| `--repo-url` / `--branch` / `--source-id` | URL / ref / id | Clone and index a Git repo |
| `--access-token` | string | GitHub token for private HTTPS repos (do not commit or log) |
| `--top-k` | int | Override retrieval `top_k` |
| `--skip-init` | flag | Reuse existing indexes instead of re-indexing |
| `--show-state` | flag | Print final-state metadata (workspace id, components, evidence) |
| `--save-intermediate` / `--run-id` | flag / string | Write per-step snapshots under `data/intermediate/` |

---

## 7. Quickstart: repo Hybrid GraphRAG

Index a Git repository and ask questions about its code. The CLI clones the repo
(filtering out binaries, secrets, and vendored dirs), chunks it with the
code-aware chunker, and builds a code graph plus hybrid indexes.

```bash
rag \
  --repo-url https://github.com/nk-droid/AutoPR \
  --branch main \
  --source-id autopr \
  --pipeline repo_hybrid_graph \
  --query "Which file implements the Redis-backed webhook queue and what keys does it use?" \
  --show-state
```

`--show-state` prints an evidence summary (top retrieved files and graph-expanded
files) so you can see *why* the answer was produced. The clone lands under
`data/repos/<source-id>/working-tree`; to re-run against an existing clone, point
`--source` at that path and add `--skip-init`.

---

## 8. Running experiments

An experiment compares pipeline variants over one dataset. Define it in
[`configs/experiments/`](configs/experiments/):

```yaml
experiment:
  name: autopr
  dataset: data/raw/autopr_eval.json     # question + (optional) ground_truth / reference_contexts
  sources: data/repos/autopr/working-tree
  runtime: eval                          # silent + temperature 0 for fair comparison
  env: dev
  parallelism: 1                         # variants run in separate processes
  metrics: [context_recall, context_precision, faithfulness, answer_relevancy, latency_ms]
  variants:
    - { name: simple_bm25,       pipeline: simple }
    - { name: advanced,          pipeline: advanced }
    - { name: custom,            pipeline: custom }
    - { name: hybrid_graph,      pipeline: repo_hybrid_graph }
```

```bash
# Run all variants, store results, compute metrics, print the comparison table
rag-eval run --experiment configs/experiments/autopr.yaml

# Recompute / add metrics from stored answers — no pipeline re-run
rag-eval metrics data/experiments/autopr/<timestamp> \
  --metrics recall_at_k faithfulness_lexical

# Reprint the comparison from stored metrics
rag-eval report data/experiments/autopr/<timestamp>
```

Each run writes a timestamped directory under `data/experiments/<name>/<ts>/`:
`manifest.json` (config + git commit), per-variant `runs.jsonl`,
`config.snapshot.json`, `metrics.json`, and `comparison.{json,md}`. Because runs
are stored separately from metrics, you can tweak metrics without re-running the
LLM. Add a `config:` block to a variant to override single components (e.g. swap
the reranker) while keeping the same pipeline.

Example comparison output (illustrative):

| variant | context_recall | context_precision | faithfulness | latency |
|---------|---------------:|------------------:|-------------:|--------:|
| simple_bm25  | 0.61 | **0.25** | 0.72 | **2.1s** |
| hybrid_graph | **0.67** | 0.10 | **0.75** | 4.8s |

Graph expansion roughly doubles the context set, trading precision for marginal
recall — the kind of signal the harness is built to surface.

---

## 9. Evaluation metrics

Two metric families are supported and can be mixed in one experiment:

**Deterministic lexical metrics** (offline, dependency-free, reproducible) in
[`components/evaluation/metrics.py`](components/evaluation/metrics.py):

| Metric | Measures | Needs | Direction |
|--------|----------|-------|-----------|
| `recall_at_k` | coverage of reference content by retrieved context | `reference_contexts` or `ground_truth` | higher |
| `context_precision_at_k` | fraction of retrieved chunks that are relevant | `reference_contexts` or `ground_truth` | higher |
| `answer_f1` / `answer_em` | token F1 / normalized exact match vs ground truth | `ground_truth` | higher |
| `faithfulness_lexical` | fraction of answer tokens grounded in retrieved context | contexts (always) | higher |
| `answer_relevancy_lexical` | token overlap between answer and question | answer + question | higher |
| `latency_ms` | per-question wall-clock run time | — | lower |

**Ragas LLM-judge metrics** (semantic, batched) in
[`components/evaluation/ragas_metrics.py`](components/evaluation/ragas_metrics.py):
`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`. These
use the configured LLM/embeddings (`evaluation.ragas` in `base.yaml`).

The eval CLI routes each requested metric to the right family automatically and
merges the results into a single per-variant report. Each metric returns `null`
when a record lacks the field it needs; aggregation averages the non-null scores,
and the loader warns when a requested metric lacks its required dataset field
rather than failing.

A dataset is `question` (required) plus optional `ground_truth` (str) and
`reference_contexts` (list).

---

## 10. API + Frontend

```bash
python scripts/run_api.py                     # FastAPI on :8000
cd frontend && npm install && npm run dev     # Vite dev server on :5173
```

The FastAPI app exposes:

- `/api/sources/*` — register paths / upload files / register public repos
- `/api/components/catalog` — component metadata for visual composition
- `/api/pipelines/{preview,initialize,run,stream,templates}` — plan, run, and SSE-stream pipelines
- `/api/prompts` — list available generation prompts and create new ones
- `/api/experiments/*` — list runs, configs, and per-query comparisons

The React + MUI client lets you register/upload sources, compose a pipeline
visually from the component catalog, pick or add a generation **prompt** (with a
raw-prompt preview), run it against the orchestrator, and inspect per-step output
including streamed answers.

---

## 11. Components

| Domain | Implemented |
|--------|-------------|
| **Ingestion** | `DirectoryLoader`, `DocumentLoader`, `MarkdownLoader`, `TextLoader`, `CodeLoader`, `RepoLoader`, `RepoCloner`, `SourceNormalizer` |
| **Chunking** | `RecursiveChunker`, `SemanticChunker` (LLM-guided), `CodeAwareChunker` |
| **Indexing** | `EmbeddingIndexer` (FAISS), `CoarseIndexer` (BM25, JSON-persisted), `RepoGraphIndexer` (code graph) |
| **Query** | `QueryCleaner`, `QueryRewriter` (LLM), `MultiQueryGenerator` (LLM) |
| **Retrieval** | `CoarseRetriever` (BM25), `FineRetriever` (FAISS dense), `HybridRetriever`, `GraphRetriever`, `GraphExpander`, `ExternalRetriever` (Tavily) |
| **Ranking** | `RankFusion` (RRF), `EmbeddingRanker`, `CrossEncoderRanker` (`ms-marco-MiniLM-L-6-v2`) |
| **Context** | `ContextBuilder`, `ContextMerger`, `ContextTruncator` |
| **Generation** | `PromptBuilder`, `Generator` / `LLMGenerator`, `StreamingGenerator`, `OutputParser` |
| **Postprocessing** | `SelfCritic` (grounding check), `Refiner` (rewrite on critic fail) |
| **Memory** | `MemoryStore`, `MemoryWriter`, `MemoryFilter` |
| **Evaluation** | lexical metrics, `RagasEvaluator` (faithfulness, relevancy, precision/recall) |

**Planned** (registry keys exist; classes are stubs): `LateChunker`,
`ColBERTRanker`, `MemoryRetriever`.

---

## 12. Cache

Two backends share a `BaseCache` interface and are configurable per pipeline stage:

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

- **In-memory (default):** TTL + LRU eviction, bounded by `max_entries`.
- **Redis:** JSON-serialised values with namespaced key tracking and scoped `clear()`.

---

## 13. Project structure

```
rag/
├── components/
│   ├── chunking/          RecursiveChunker, SemanticChunker, CodeAwareChunker, LateChunker (stub)
│   ├── context/           ContextBuilder, ContextMerger, ContextTruncator
│   ├── evaluation/        lexical metrics, RagasEvaluator, dataset loader
│   ├── generation/        PromptBuilder, Generator, StreamingGenerator, OutputParser, templates/
│   ├── indexer/           EmbeddingIndexer (FAISS), CoarseIndexer (BM25), RepoGraphIndexer
│   ├── ingestion/         Directory/Document/Markdown/Text/Code/Repo loaders, RepoCloner, file filter
│   ├── memory/            MemoryStore, MemoryWriter, MemoryFilter
│   ├── postprocessing/    SelfCritic, Refiner
│   ├── query/             QueryCleaner, QueryRewriter, MultiQueryGenerator
│   ├── ranking/           RankFusion, EmbeddingRanker, CrossEncoderRanker, ColBERTRanker (stub)
│   └── retrieval/         Coarse/Fine/Hybrid/Graph/External retrievers, GraphExpander, MemoryRetriever (stub)
├── configs/
│   ├── base.yaml
│   ├── env/               dev.yaml, prod.yaml
│   ├── pipeline/          simple, custom, advanced, debug, external_fallback, repo_hybrid_graph
│   ├── runtime/           cli, api, notebook, eval
│   └── experiments/       autopr.yaml, example.yaml
├── infra/
│   ├── cache/             InMemoryCache (TTL+LRU), RedisCache
│   ├── embeddings/        Provider factory (Ollama, OpenAI, HuggingFace)
│   ├── llm/               Provider factory (Ollama, OpenAI, Anthropic, HuggingFace)
│   ├── logging/           Rich / silent / simple runtime adapters
│   └── storage/           FAISS store, intermediate store, experiment store, vector store factory
├── pipeline/
│   ├── orchestrator.py    Init + run orchestration, fingerprint-based cache
│   ├── registry.py        44 component bindings
│   ├── registry_handlers.py / registry_utils.py / component_factories.py
│   ├── contracts.py       Declared requires/produces per component
│   ├── validator.py       Static pre-flight config check
│   ├── workspace.py       Per-config index/artifact isolation
│   ├── results.py         Normalized answer/context extraction
│   └── experiment/        config, parallel runner, comparison report
├── api/                   FastAPI app
│   ├── main.py
│   ├── routers/           components, pipelines, sources, prompts, experiments
│   ├── catalog.py / loader_service.py / pipeline_service.py / template_service.py
│   ├── prompt_service.py / source_store.py / schemas.py
├── frontend/              React + Vite + MUI client (src/: App.tsx, api.ts, theme.ts, types.ts)
├── scripts/               run_api.py (uvicorn launcher)
├── tests/                 pytest: factories, behaviors, module-import smoke
├── data/
│   ├── raw/               Source documents + eval datasets
│   ├── repos/             Cloned Git repositories
│   ├── uploads/           Uploaded sources from the API
│   ├── workspaces/        Per-config FAISS + BM25 + graph artifacts
│   └── experiments/       Stored experiment runs, metrics, comparisons
├── clis/                 CLI entry points: cli.py (`rag`), eval_cli.py (`rag-eval`)
├── CONTRIBUTING.md
└── pyproject.toml         Project metadata, pinned deps, console scripts, pytest config
```

---

## 14. Design trade-offs

- **Config composition over a plugin DSL.** A RAG type is just YAML. Flexibility is
  high, but introducing a *new component type* still requires Python (factory +
  handler + contract). Config rearranges existing components; it doesn't define new
  ones.
- **Static contracts vs. dynamic flexibility.** `contracts.py` + `validator.py`
  catch missing producers before execution. The cost is that every component must
  declare its state keys; the benefit is that arbitrary valid arrangements run, and
  invalid ones fail fast with a readable error.
- **Workspace isolation by config hash.** Guarantees clean, uncontaminated
  comparisons and correct cache reuse, at the cost of disk (each distinct config
  re-indexes into its own directory).
- **Two metric families.** Deterministic lexical metrics are reproducible and
  offline; ragas LLM-judge metrics are semantic but require a model and add latency.
- **Process-based variant parallelism.** Isolates the module-level component cache,
  FAISS, and LLM clients per variant — at higher memory use than threads.
- **Tolerant config slicing.** `from_config` filters a config slice to a model's
  declared fields, so sibling components can share a config path (e.g. graph
  retriever and graph expander both read `retrieval.graph`). Direct construction
  still forbids unknown keys to catch typos.

---

## 15. Limitations

- **Experiment results persist only after all variants finish.** A slow variant
  delays the whole run; incremental per-variant persistence is a planned
  improvement.
- **Lexical metrics are proxies.** `answer_relevancy_lexical` in particular is weak
  for code Q&A (answers use identifiers absent from the question). Use the ragas
  metrics for semantic scoring.
- **Latency is confounded under `parallelism > 1`.** Variants contend for the same
  embedding server / LLM. Use `parallelism: 1` for trustworthy latency numbers.
- **Hybrid retrieval needs a dense index.** A pipeline using `hybrid_retriever` must
  include `embedding_indexer` in `init_pipeline`, or the dense side has nothing to
  read. The validator checks state-key flow, not on-disk index prerequisites.
- **Graph construction is repo/code-oriented.** The graph indexer extracts
  code-shaped entities (files, symbols, imports, config keys, tests) via heuristics;
  a fully generic, ingestor-agnostic graph extractor is not yet implemented.
- **Local-model / API dependence.** Defaults assume Ollama; cloud models add latency
  and require credentials. Web fallback needs `TAVILY_API_KEY`.
- **Repo ingestion caps.** The file filter enforces limits (default 5000 files,
  512 KB/file) and skips binaries, lockfiles, and secret-like paths.
- **Single-machine, no API auth.** The HTTP API has no authentication and the system
  is not designed for multi-node distribution.

---

## 16. Stack

**Backend** — Python 3.12+, LangChain (`langchain-{ollama,openai,anthropic,community,
tavily,text-splitters}`), FAISS (`faiss-cpu`), `rank_bm25`, `sentence-transformers`,
Pydantic, PyYAML, python-dotenv, Ragas + `datasets` (evaluation), Redis (optional
cache), Rich (terminal runtime), pytest, FastAPI + Uvicorn, `python-multipart`.

**Frontend** — React 18 + Vite 6 + TypeScript, MUI 5 (`@mui/material`,
`@mui/icons-material`, `@emotion/{react,styled}`).

**External services** — Ollama (default LLM + embeddings), Tavily (web retrieval).
