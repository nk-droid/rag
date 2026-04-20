from pathlib import Path
from typing import Any, Callable

from components.chunking import LateChunker, RecursiveChunker, SemanticChunker
from components.context import ContextBuilder, ContextMerger, ContextTruncator
from components.evaluation import Evaluator, RagasEvaluator, TruLensEvaluator
from components.generation import Generator, OutputParser, PromptBuilder, StreamingGenerator
from components.ingestion import DirectoryLoader, DocumentLoader, MarkdownLoader, SourceNormalizer, TextLoader
from components.indexer import EmbeddingIndexer, CoarseIndexer
from components.memory import MemoryFilter, MemoryStore, MemoryWriter
from components.postprocessing import Refiner, SelfCritic
from components.query import MultiQueryGenerator, QueryCleaner, QueryRewriter
from components.ranking import (
    BaseRanker,
    ColBERTRanker,
    CrossEncoderRanker,
    EmbeddingRanker,
    RankFusion,
)
from components.retrieval import (
    BaseRetriever,
    CoarseRetriever,
    ExternalRetriever,
    FineRetriever,
    GraphRetriever,
    HybridRetriever,
    MemoryRetriever,
)
from components.shared_types import MemoryRecord, RetrievedChunk
from infra.cache.base_cache import BaseCache
from infra.cache.cache_keys import file_signature, make_cache_key, stable_hash, text_hash
from infra.cache.in_memory_cache import InMemoryCache
from infra.cache.redis_cache import RedisCache
from infra.storage.vector_store_factory import get_vector_store
from infra.llm.llm_factory import get_llm

ComponentCallable = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
ComponentFactory = Callable[[dict[str, Any]], Any]

def _ensure_state(state: dict[str, Any] | None) -> dict[str, Any]:
    return state if isinstance(state, dict) else {}

_COMPONENT_CACHE: dict[tuple[str, str], Any] = {}
_CACHE_CLIENTS: dict[str, BaseCache] = {}
_REPO_ROOT = Path(__file__).resolve().parent.parent
_QUERY_TEMPLATE_DIR = _REPO_ROOT / "components" / "query" / "templates"
_POSTPROCESS_TEMPLATE_DIR = _REPO_ROOT / "components" / "postprocessing" / "templates"

def _config_cache_key(config: dict[str, Any]) -> str:
    vector_store = config.get("vector_store", {})
    models = config.get("models", {})
    retrieval = config.get("retrieval", {})
    ranking = config.get("ranking", {})
    chunking = config.get("chunking", {})
    return repr(
        {
            "vector_store": vector_store,
            "models": models,
            "retrieval": retrieval,
            "ranking": ranking,
            "chunking": chunking,
        }
    )

def _cache_config(config: dict[str, Any]) -> dict[str, Any]:
    cache_cfg = config.get("cache", {})
    return cache_cfg if isinstance(cache_cfg, dict) else {}

def _cache_enabled(config: dict[str, Any], feature: str) -> bool:
    cache_cfg = _cache_config(config)
    if not bool(cache_cfg.get("enabled", False)):
        return False

    features = cache_cfg.get("features", {})
    if isinstance(features, dict) and feature in features:
        return bool(features.get(feature))
    return True

def _cache_ttl(
    config: dict[str, Any],
    feature: str,
    fallback: int | None = None,
) -> int | None:
    cache_cfg = _cache_config(config)
    per_feature = cache_cfg.get("ttl_sec", {})
    if isinstance(per_feature, dict) and feature in per_feature:
        value = per_feature.get(feature)
        if value is None:
            return None
        return int(value)

    default_ttl = cache_cfg.get("default_ttl_sec", fallback)
    if default_ttl is None:
        return None
    return int(default_ttl)

def _cache_env(config: dict[str, Any]) -> str:
    app_cfg = config.get("app", {})
    if isinstance(app_cfg, dict):
        return str(app_cfg.get("env", "default"))
    return "default"

def _cache_key(config: dict[str, Any], feature: str, payload: dict[str, Any]) -> str:
    cache_cfg = _cache_config(config)
    return make_cache_key(
        namespace=str(cache_cfg.get("namespace", "rag")),
        version=str(cache_cfg.get("version", "v1")),
        env=_cache_env(config),
        feature=feature,
        payload=payload,
    )

def _get_cache(config: dict[str, Any]) -> BaseCache | None:
    if not _cache_enabled(config, feature="base"):
        return None

    cache_cfg = _cache_config(config)
    cache_type = str(cache_cfg.get("type", "in_memory")).strip().lower()
    cache_signature = stable_hash(
        {
            "type": cache_type,
            "config": cache_cfg,
            "env": _cache_env(config),
        }
    )

    if cache_signature in _CACHE_CLIENTS:
        return _CACHE_CLIENTS[cache_signature]

    default_ttl = _cache_ttl(config, feature="default", fallback=900)
    if cache_type == "in_memory":
        cache_client: BaseCache = InMemoryCache(
            max_entries=int(cache_cfg.get("max_entries", 2000)),
            default_ttl_sec=default_ttl,
        )
    elif cache_type == "redis":
        namespace = str(cache_cfg.get("namespace", "rag"))
        version = str(cache_cfg.get("version", "v1"))
        env = _cache_env(config)
        key_prefix = str(cache_cfg.get("key_prefix", f"{namespace}:{version}:{env}:"))
        index_key = str(cache_cfg.get("index_key", "__cache_index__"))
        redis_url = str(cache_cfg.get("redis_url", "redis://localhost:6379/0"))
        cache_client = RedisCache(
            redis_url=redis_url,
            default_ttl_sec=default_ttl,
            key_prefix=key_prefix,
            index_key=index_key,
        )
    else:
        raise ValueError(f"Unsupported cache type: {cache_type}")

    _CACHE_CLIENTS[cache_signature] = cache_client
    return cache_client

def _mark_cache_hit(payload: dict[str, Any], feature: str, hit: bool) -> None:
    cache_hit = payload.get("cache_hit")
    if not isinstance(cache_hit, dict):
        cache_hit = {}
    cache_hit[feature] = hit
    payload["cache_hit"] = cache_hit

def _serialize_chunks(chunks: list[RetrievedChunk] | list[Any]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        if isinstance(chunk, RetrievedChunk):
            chunk_id = chunk.id
            text = chunk.text
            score = chunk.score
            metadata = chunk.metadata
        elif isinstance(chunk, dict):
            chunk_id = chunk.get("id", f"chunk-{idx}")
            text = chunk.get("text") or chunk.get("content") or ""
            score = chunk.get("score", 0.0)
            metadata = chunk.get("metadata", {})
        else:
            chunk_id = getattr(chunk, "id", f"chunk-{idx}")
            text = getattr(chunk, "text", "")
            score = getattr(chunk, "score", 0.0)
            metadata = getattr(chunk, "metadata", {})

        serialized.append(
            {
                "id": str(chunk_id),
                "text": str(text),
                "score": float(score) if score is not None else 0.0,
                "metadata": dict(metadata) if isinstance(metadata, dict) else {},
            }
        )
    return serialized

def _deserialize_chunks(payload: list[dict[str, Any]] | Any) -> list[RetrievedChunk]:
    if not isinstance(payload, list):
        return []

    chunks: list[RetrievedChunk] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            continue

        score = item.get("score", 0.0)
        try:
            normalized_score = float(score)
        except (TypeError, ValueError):
            normalized_score = 0.0

        chunks.append(
            RetrievedChunk(
                id=str(item.get("id", f"chunk-{idx}")),
                text=str(item.get("text", "")),
                score=normalized_score,
                metadata=dict(item.get("metadata", {}))
                if isinstance(item.get("metadata"), dict)
                else {},
            )
        )

    return chunks

def _answer_text(answer: Any) -> str:
    if answer is None:
        return ""

    if isinstance(answer, str):
        return answer

    content = getattr(answer, "content", None)
    if content is None:
        return str(answer)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        pieces: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if text is not None:
                    pieces.append(str(text))
            else:
                pieces.append(str(part))
        return "\n".join(piece for piece in pieces if piece)

    return str(content)

def _retrieval_queries(payload: dict[str, Any]) -> list[str]:
    primary = str(payload.get("query", "")).strip()
    raw_queries = payload.get("queries", [])

    candidates: list[str] = []
    if isinstance(raw_queries, list):
        for item in raw_queries:
            text = str(item).strip()
            if text:
                candidates.append(text)

    ordered: list[str] = []
    seen: set[str] = set()

    if primary:
        ordered.append(primary)
        seen.add(primary.lower())

    for item in candidates:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(item)

    return ordered

def _as_retrieved_chunk(item: Any, fallback_index: int) -> RetrievedChunk | None:
    if isinstance(item, RetrievedChunk):
        return item

    if isinstance(item, dict):
        text = str(item.get("text") or item.get("content") or "").strip()
        if not text:
            return None

        score = item.get("score", 0.0)
        try:
            normalized_score = float(score)
        except (TypeError, ValueError):
            normalized_score = 0.0

        return RetrievedChunk(
            id=str(item.get("id", f"chunk-{fallback_index}")),
            text=text,
            score=normalized_score,
            metadata=dict(item.get("metadata", {}))
            if isinstance(item.get("metadata"), dict)
            else {},
        )

    text = str(getattr(item, "text", "")).strip()
    if not text:
        return None

    raw_score = getattr(item, "score", 0.0)
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        score = 0.0

    metadata = getattr(item, "metadata", {})
    return RetrievedChunk(
        id=str(getattr(item, "id", f"chunk-{fallback_index}")),
        text=text,
        score=score,
        metadata=dict(metadata) if isinstance(metadata, dict) else {},
    )

def _merge_retrieval_chunks(chunks: list[Any], top_k: int) -> list[RetrievedChunk]:
    merged: dict[str, RetrievedChunk] = {}
    insertion_order: list[str] = []

    for idx, item in enumerate(chunks):
        chunk = _as_retrieved_chunk(item, fallback_index=idx)
        if chunk is None:
            continue

        key = chunk.id or text_hash(chunk.text)
        if key not in merged:
            merged[key] = chunk
            insertion_order.append(key)
            continue

        current = merged[key]
        if chunk.score > current.score:
            combined_metadata = dict(current.metadata)
            combined_metadata.update(chunk.metadata)
            merged[key] = RetrievedChunk(
                id=chunk.id,
                text=chunk.text,
                score=chunk.score,
                metadata=combined_metadata,
            )

    ordered = [merged[key] for key in insertion_order]
    ordered.sort(key=lambda item: item.score, reverse=True)
    if top_k <= 0:
        return []
    return ordered[:top_k]

def _index_fingerprint(config: dict[str, Any]) -> str:
    embedding_index_path = Path(_get_index_path(config, "embedding_indexer"))
    coarse_index_path = Path(_get_index_path(config, "coarse_indexer"))

    if embedding_index_path.is_dir():
        embedding_artifacts = [
            file_signature(embedding_index_path / "index.faiss"),
            file_signature(embedding_index_path / "index.pkl"),
        ]
    else:
        embedding_artifacts = [file_signature(embedding_index_path)]

    payload = {
        "embedding_artifacts": embedding_artifacts,
        "coarse_artifact": file_signature(coarse_index_path),
        "embedding_model": config.get("models", {}).get("embedding", {}),
        "vector_store": config.get("vector_store", {}),
    }
    return stable_hash(payload)

def _generation_cacheable(config: dict[str, Any]) -> bool:
    llm_cfg = config.get("models", {}).get("llm", {})
    raw_temperature = llm_cfg.get("temperature", 0)
    try:
        temperature = float(raw_temperature)
    except (TypeError, ValueError):
        temperature = 0.0

    if temperature == 0.0:
        return True
    return bool(_cache_config(config).get("allow_nondeterministic_generation", False))

def _get_index_path(config: dict[str, Any], indexer_key: str) -> str:
    vector_store = config.get("vector_store", {})
    indexer_cfg = vector_store.get(indexer_key, {})
    if isinstance(indexer_cfg, dict):
        path = indexer_cfg.get("path")
        if path:
            return str(path)

    # Backward-compatible fallback for embedding index path.
    if indexer_key == "embedding_indexer":
        legacy_path = vector_store.get("path")
        if legacy_path:
            return str(legacy_path)
        return "data/embeddings/faiss_index"

    # Backward-compatible fallback for coarse index path.
    if indexer_key == "coarse_indexer":
        legacy_path = config.get("coarse_index", {}).get("path")
        if legacy_path:
            return str(legacy_path)
        return "data/indexes/coarse_index.json"

    raise ValueError(f"Unsupported indexer key: {indexer_key}")

def _resolve_index_path(indexer: Any, state: dict[str, Any], config: dict[str, Any]) -> str:
    payload = _ensure_state(state)
    step = payload.get("_step", {})
    if isinstance(step, dict):
        step_path = step.get("index_path") or step.get("path")
        if step_path:
            return str(step_path)

    if isinstance(indexer, CoarseIndexer):
        return _get_index_path(config, "coarse_indexer")

    return _get_index_path(config, "embedding_indexer")

def _with_vector_store_path(config: dict[str, Any], path: str) -> dict[str, Any]:
    resolved = dict(config)
    vector_store = dict(resolved.get("vector_store", {}))
    vector_store["path"] = path
    resolved["vector_store"] = vector_store
    return resolved

def _build_component(name: str, config: dict[str, Any]) -> Any:
    cache_key = (name, _config_cache_key(config))
    if cache_key not in _COMPONENT_CACHE:
        _COMPONENT_CACHE[cache_key] = COMPONENT_FACTORIES[name](config)
    return _COMPONENT_CACHE[cache_key]

def _document_to_payload(document: Any) -> dict[str, Any]:
    if hasattr(document, "text") and hasattr(document, "source") and hasattr(document, "metadata"):
        return {
            "text": document.text,
            "metadata": {
                "source": document.source,
                **(document.metadata if isinstance(document.metadata, dict) else {}),
            },
        }

    if isinstance(document, dict):
        text = document.get("text") or document.get("content") or document.get("body") or ""
        metadata = document.get("metadata", {})
        source = document.get("source")
        merged_metadata = dict(metadata) if isinstance(metadata, dict) else {}
        if source:
            merged_metadata.setdefault("source", source)
        return {"text": text, "metadata": merged_metadata}

    if isinstance(document, str):
        return {"text": document, "metadata": {}}

    return {"text": "", "metadata": {}}

def _extract_chunk_inputs(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    source_collections = (
        payload.get("data_sources"),
        payload.get("sources"),
        payload.get("documents"),
    )

    extracted: list[tuple[str, dict[str, Any]]] = []
    for collection in source_collections:
        if not collection:
            continue

        if isinstance(collection, str):
            extracted.append((collection, {}))
            continue

        if not isinstance(collection, list):
            continue

        for item in collection:
            payload_item = _document_to_payload(item)
            text = payload_item.get("text", "")
            metadata = payload_item.get("metadata", {})
            if text:
                extracted.append((text, metadata if isinstance(metadata, dict) else {}))

    if extracted:
        return extracted

    fallback_text = payload.get("text")
    if fallback_text:
        return [(fallback_text, {})]

    fallback_query = payload.get("query")
    if fallback_query:
        return [(fallback_query, {"source": "query"})]

    return []

def _ingest_with(loader: Any, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    payload = _ensure_state(state)
    sources = payload.get("sources") or payload.get("data_sources") or []
    if isinstance(sources, str):
        sources = [sources]

    documents = []
    for source in sources:
        documents.extend(loader.load(source))

    payload["documents"] = documents
    payload["ingestion_loader"] = loader.__class__.__name__
    payload["config"] = config
    return payload

def _normalize_sources_with(
    normalizer: SourceNormalizer,
    state: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    payload = _ensure_state(state)
    documents = payload.get("documents", [])
    payload["data_sources"] = normalizer.normalize(documents)
    payload["config"] = config
    return payload

def _chunk_with(chunker: Any, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    payload = _ensure_state(state)
    chunk_inputs = _extract_chunk_inputs(payload)
    chunks = []

    for source_index, (text, metadata) in enumerate(chunk_inputs[1:]): #FIXME: need to skip first element for semantic chunker. reason not known...
        source_chunks = chunker.chunk(text)
        for chunk_index, chunk in enumerate(source_chunks):
            if hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
                chunk.metadata.update(metadata)
                chunk.metadata.setdefault("source_index", source_index)
                chunk.metadata.setdefault("chunk_index", chunk_index)
        chunks.extend(source_chunks)

    payload["chunks"] = chunks
    payload["chunker"] = chunker.__class__.__name__
    payload["config"] = config
    return payload

def _index_with(indexer: Any, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    payload = _ensure_state(state)
    chunks = payload.get("chunks", [])
    vector_db_path = _resolve_index_path(indexer, payload, config)
    records = indexer.index(chunks, config=config, vector_db_path=vector_db_path)
    payload["index_records"] = records
    payload["indexed_count"] = len(records)
    payload["vector_store_path"] = str(vector_db_path)
    payload["indexer"] = indexer.__class__.__name__
    payload["config"] = config
    return payload

def _retrieve_with(retriever: Any, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    payload = _ensure_state(state)
    queries = _retrieval_queries(payload)
    query = queries[0] if queries else str(payload.get("query", "")).strip()
    top_k = int(payload.get("top_k", config.get("retrieval", {}).get("top_k", 5)))
    step_cfg = payload.get("_step", {}) if isinstance(payload.get("_step"), dict) else {}

    cache = _get_cache(config)
    cache_key = None
    if cache is not None and _cache_enabled(config, "retrieval"):
        key_payload = {
            "retriever": retriever.__class__.__name__,
            "query": str(query).strip(),
            "queries": queries,
            "top_k": top_k,
            "step": step_cfg,
            "index_fingerprint": _index_fingerprint(config),
        }
        cache_key = _cache_key(config, "retrieval", key_payload)
        cached = cache.get(cache_key)
        if isinstance(cached, dict):
            payload["retrieved"] = _deserialize_chunks(cached.get("retrieved", []))
            cached_queries = cached.get("retrieval_queries", queries)
            payload["retrieval_queries"] = (
                list(cached_queries) if isinstance(cached_queries, list) else queries
            )
            payload["retriever"] = retriever.__class__.__name__
            payload["config"] = config
            _mark_cache_hit(payload, "retrieval", True)
            return payload

    queries_to_run = queries or ([query] if query else [])
    if len(queries_to_run) <= 1:
        results = retriever.retrieve(query, top_k=top_k) if query else []
    else:
        gathered: list[Any] = []
        for query_variant in queries_to_run:
            gathered.extend(retriever.retrieve(query_variant, top_k=top_k))
        results = _merge_retrieval_chunks(gathered, top_k=top_k)

    payload["retrieved"] = results
    payload["retrieval_queries"] = queries_to_run
    payload["retriever"] = retriever.__class__.__name__
    payload["config"] = config
    _mark_cache_hit(payload, "retrieval", False)

    if cache is not None and cache_key is not None:
        cache.set(
            cache_key,
            {
                "retrieved": _serialize_chunks(results),
                "retrieval_queries": queries_to_run,
            },
            ttl_sec=_cache_ttl(config, "retrieval", fallback=900),
        )

    return payload

def _hybrid_retrieve_with(
    retriever: HybridRetriever,
    state: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    payload = _ensure_state(state)
    step_cfg = payload.get("_step", {}) if isinstance(payload.get("_step"), dict) else {}
    retrieval_cfg = config.get("retrieval", {})

    queries = _retrieval_queries(payload)
    query = queries[0] if queries else str(payload.get("query", "")).strip()
    top_k = int(step_cfg.get("top_k", payload.get("top_k", retrieval_cfg.get("top_k", 5))))

    cache = _get_cache(config)
    cache_key = None
    if cache is not None and _cache_enabled(config, "retrieval"):
        key_payload = {
            "retriever": retriever.__class__.__name__,
            "query": str(query).strip(),
            "queries": queries,
            "top_k": top_k,
            "candidate_multiplier": int(getattr(retriever, "candidate_multiplier", 1)),
            "fuse": bool(step_cfg.get("fuse", False)),
            "step": step_cfg,
            "index_fingerprint": _index_fingerprint(config),
            "retrieval_cfg": retrieval_cfg,
        }
        cache_key = _cache_key(config, "retrieval", key_payload)
        cached = cache.get(cache_key)
        if isinstance(cached, dict):
            sparse = _deserialize_chunks(cached.get("sparse", []))
            dense = _deserialize_chunks(cached.get("dense", []))
            combined = _deserialize_chunks(cached.get("retrieved", []))

            payload["sparse_retrieved"] = sparse
            payload["dense_retrieved"] = dense
            payload["retrieved"] = combined
            cached_queries = cached.get("retrieval_queries", queries)
            payload["retrieval_queries"] = (
                list(cached_queries) if isinstance(cached_queries, list) else queries
            )
            payload["retriever"] = retriever.__class__.__name__
            payload["config"] = config
            _mark_cache_hit(payload, "retrieval", True)
            return payload

    sparse_candidates: list[Any] = []
    dense_candidates: list[Any] = []
    queries_to_run = queries or ([query] if query else [])

    for query_variant in queries_to_run:
        candidate_sets = retriever.retrieve_candidates(query_variant, top_k=top_k)
        sparse_candidates.extend(list(candidate_sets.get("sparse", [])))
        dense_candidates.extend(list(candidate_sets.get("dense", [])))

    candidate_multiplier = int(getattr(retriever, "candidate_multiplier", 1))
    candidate_cap = max(top_k, top_k * candidate_multiplier)
    sparse = _merge_retrieval_chunks(sparse_candidates, top_k=candidate_cap)
    dense = _merge_retrieval_chunks(dense_candidates, top_k=candidate_cap)

    combined = sparse + dense
    if bool(step_cfg.get("fuse", False)):
        fusion = _build_component("rank_fusion", config)
        combined = fusion.fuse([sparse, dense])
    combined = _merge_retrieval_chunks(combined, top_k=top_k)

    payload["sparse_retrieved"] = sparse
    payload["dense_retrieved"] = dense
    payload["retrieved"] = combined
    payload["retrieval_queries"] = queries_to_run
    payload["retriever"] = retriever.__class__.__name__
    payload["config"] = config
    _mark_cache_hit(payload, "retrieval", False)

    if cache is not None and cache_key is not None:
        cache.set(
            cache_key,
            {
                "sparse": _serialize_chunks(sparse),
                "dense": _serialize_chunks(dense),
                "retrieved": _serialize_chunks(combined),
                "retrieval_queries": queries_to_run,
            },
            ttl_sec=_cache_ttl(config, "retrieval", fallback=900),
        )

    return payload

def _rank_fusion_with(fusion: RankFusion, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    payload = _ensure_state(state)
    step_cfg = payload.get("_step", {}) if isinstance(payload.get("_step"), dict) else {}
    retrieval_cfg = config.get("retrieval", {})
    top_k = int(step_cfg.get("top_k", payload.get("top_k", retrieval_cfg.get("top_k", 5))))

    sparse = payload.get("sparse_retrieved", [])
    dense = payload.get("dense_retrieved", [])

    if sparse or dense:
        fused = fusion.fuse([sparse, dense])
    else:
        # Backward-compatible fallback when only `retrieved` is present.
        fused = payload.get("retrieved", [])

    payload["retrieved"] = fused[:top_k] if top_k > 0 else fused
    payload["fusion"] = fusion.__class__.__name__
    payload["config"] = config
    return payload

def _rank_with(ranker: BaseRanker, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    payload = _ensure_state(state)
    query = payload.get("query", "")
    candidates = payload.get("retrieved", [])
    payload["ranked"] = ranker.rank(query, candidates)
    payload["ranker"] = ranker.__class__.__name__
    payload["config"] = config
    return payload

def _generate_with(generator: Generator, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    payload = _ensure_state(state)
    context = payload.get("context", "")
    if not context and payload.get("ranked"):
        context = "\n\n".join(
            chunk.text if isinstance(chunk, RetrievedChunk) else chunk["content"]
            for chunk in payload["ranked"]
        )
    inputs = {
        "query": payload.get("query", ""),
        "context": context
    }
    prompt = payload["prompt"]

    cache = _get_cache(config)
    cache_key = None
    generation_cache_enabled = (
        cache is not None
        and _cache_enabled(config, "generation")
        and _generation_cacheable(config)
    )
    if generation_cache_enabled:
        partial_vars = getattr(prompt, "partial_variables", {})
        if not isinstance(partial_vars, dict):
            partial_vars = {"value": str(partial_vars)}

        key_payload = {
            "query_hash": text_hash(str(inputs["query"])),
            "context_hash": text_hash(str(inputs["context"])),
            "prompt_template_hash": text_hash(str(getattr(prompt, "template", ""))),
            "prompt_partial_vars_hash": stable_hash({"partial_variables": partial_vars}),
            "llm": config.get("models", {}).get("llm", {}),
            "parser_model": payload.get("_step", {}).get("parser", None),
        }
        cache_key = _cache_key(config, "generation", key_payload)
        cached = cache.get(cache_key)
        if isinstance(cached, dict) and "answer_text" in cached:
            payload["answer"] = str(cached.get("answer_text", ""))
            payload["generator"] = generator.__class__.__name__
            payload["config"] = config
            _mark_cache_hit(payload, "generation", True)
            return payload

    payload["answer"] = generator.generate(prompt, inputs)
    payload["generator"] = generator.__class__.__name__
    payload["config"] = config
    _mark_cache_hit(payload, "generation", False)

    if generation_cache_enabled and cache_key is not None:
        cache.set(
            cache_key,
            {"answer_text": _answer_text(payload["answer"])},
            ttl_sec=_cache_ttl(config, "generation", fallback=3600),
        )

    return payload

def _context_build_with(builder: ContextBuilder, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    payload = _ensure_state(state)
    chunks = payload.get("ranked") or payload.get("retrieved", [])
    payload["context"] = builder.build(chunks)
    payload["config"] = config
    return payload

def _context_merge_with(merger: ContextMerger, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    payload = _ensure_state(state)
    chunks = payload.get("retrieved", [])
    payload["retrieved"] = merger.merge(chunks)
    payload["config"] = config
    return payload

def _context_truncate_with(
    truncator: ContextTruncator,
    state: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    payload = _ensure_state(state)
    max_tokens = int(payload.get("max_tokens", 256))
    payload["context"] = truncator.truncate(payload.get("context", ""), max_tokens)
    payload["config"] = config
    return payload

def _clean_query_with(cleaner: QueryCleaner, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    payload = _ensure_state(state)
    payload["query"] = cleaner.clean(payload.get("query", ""))
    payload["config"] = config
    return payload

def _rewrite_query_with(rewriter: QueryRewriter, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    payload = _ensure_state(state)
    payload["query"] = rewriter.rewrite(payload.get("query", ""))
    payload["config"] = config
    return payload

def _multi_query_with(
    generator: MultiQueryGenerator,
    state: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    payload = _ensure_state(state)
    payload["queries"] = generator.generate(payload.get("query", ""))
    payload["config"] = config
    return payload

def _stream_generate_with(
    generator: StreamingGenerator,
    state: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    payload = _ensure_state(state)
    payload["stream"] = list(generator.stream(payload.get("query", ""), payload.get("context", "")))
    payload["config"] = config
    return payload

def _build_prompt_with(builder: PromptBuilder, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    payload = _ensure_state(state)
    template_name = payload.get("_step", {}).get("template_name")
    payload["prompt"] = builder.build(
        template_name=template_name,
        parser_model=payload.get("_step", {}).get("parser", None)
    )
    payload["config"] = config
    return payload

def _parse_output_with(parser: OutputParser, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    payload = _ensure_state(state)
    answer_text = _answer_text(payload.get("answer"))
    payload["parsed_output"] = parser.parse(
        answer_text,
        parser_model=payload.get("_step", {}).get("parser", None)
    )
    payload["config"] = config
    return payload

def _memory_write_with(writer: MemoryWriter, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    payload = _ensure_state(state)
    interaction = {
        "id": payload.get("memory_id", "memory-0"),
        "content": _answer_text(payload.get("answer", "")),
    }
    payload["memory_record"] = writer.write(interaction)
    payload["config"] = config
    return payload

def _memory_retrieve_with(store: MemoryStore, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    payload = _ensure_state(state)
    top_k = int(payload.get("top_k", 5))
    payload["memories"] = store.search(payload.get("query", ""), top_k=top_k)
    payload["config"] = config
    return payload

def _memory_filter_with(
    memory_filter: MemoryFilter,
    state: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    payload = _ensure_state(state)
    memories = payload.get("memories", [])
    payload["memories"] = memory_filter.filter(memories)
    payload["config"] = config
    return payload

def _evaluate_with(evaluator: Any, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    payload = _ensure_state(state)
    payload["evaluation"] = evaluator.evaluate(payload)
    payload["config"] = config
    return payload

def _critique_with(critic: SelfCritic, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    payload = _ensure_state(state)
    payload["critique"] = critic.critique(payload.get("answer", ""), payload.get("context", ""))
    payload["config"] = config
    return payload

def _refine_with(refiner: Refiner, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    payload = _ensure_state(state)
    payload["answer"] = refiner.refine(payload.get("answer", ""), payload.get("critique", {}))
    payload["config"] = config
    return payload

_MEMORY_STORE = MemoryStore()
COMPONENT_FACTORIES: dict[str, ComponentFactory] = {
    "text_loader": lambda config: TextLoader(),
    "markdown_loader": lambda config: MarkdownLoader(),
    "document_loader": lambda config: DocumentLoader(),
    "directory_loader": lambda config: DirectoryLoader(),
    "source_normalizer": lambda config: SourceNormalizer(),
    "embedding_indexer": lambda config: EmbeddingIndexer(
        vector_db_path=_get_index_path(config, "embedding_indexer")
    ),
    "coarse_indexer": lambda config: CoarseIndexer(
        index_path=_get_index_path(config, "coarse_indexer")
    ),
    "query_cleaner": lambda config: QueryCleaner(),
    "query_rewriter": lambda config: QueryRewriter(
        generator=Generator(get_llm(config)),
        prompt_builder=PromptBuilder(
            template_dir=_QUERY_TEMPLATE_DIR,
            use_cache=_cache_enabled(config, "prompt"),
        ),
        parser=OutputParser(),
    ),
    "multi_query_generator": lambda config: MultiQueryGenerator(
        generator=Generator(get_llm(config)),
        prompt_builder=PromptBuilder(
            template_dir=_QUERY_TEMPLATE_DIR,
            use_cache=_cache_enabled(config, "prompt"),
        ),
        parser=OutputParser(),
        max_queries=int(
            config.get("retrieval", {})
            .get("query_expansion", {})
            .get("max_queries", 3)
        ),
    ),
    "coarse_retriever": lambda config: CoarseRetriever(
        index_path=_get_index_path(config, "coarse_indexer")
    ),
    "fine_retriever": lambda config: FineRetriever(
        store=get_vector_store(
            _with_vector_store_path(config, _get_index_path(config, "embedding_indexer"))
        )
    ),
    "hybrid_retriever": lambda config: HybridRetriever(
        dense_retriever=FineRetriever(
            store=get_vector_store(
                _with_vector_store_path(config, _get_index_path(config, "embedding_indexer"))
            )
        ),
        sparse_retriever=CoarseRetriever(
            index_path=_get_index_path(config, "coarse_indexer")
        ),
        candidate_multiplier=int(
            config.get("retrieval", {})
            .get("hybrid", {})
            .get("candidate_multiplier", 4)
        ),
    ),
    "memory_retriever": lambda config: MemoryRetriever(),
    "graph_retriever": lambda config: GraphRetriever(),
    "external_retriever": lambda config: ExternalRetriever(),
    "late_chunker": lambda config: LateChunker(),
    "semantic_chunker": lambda config: SemanticChunker(config),
    "recursive_chunker": lambda config: RecursiveChunker(config),
    "embedding_ranker": lambda config: EmbeddingRanker(
        model_name=config.get("ranking", {}).get(
            "embedding_model_name"
        ),
        top_n=int(config.get("ranking", {}).get("top_k", 3)),
        use_cache=_cache_enabled(config, "ranking_embeddings"),
    ),
    "colbert_ranker": lambda config: ColBERTRanker(),
    "cross_encoder_ranker": lambda config: CrossEncoderRanker(
        model_name=config.get("ranking", {}).get(
            "cross_encoder_model_name"
        ),
        top_n=int(config.get("ranking", {}).get("top_k", 3))
    ),
    "rank_fusion": lambda config: RankFusion(),
    "generator": lambda config: Generator(get_llm(config)),
    "streaming_generator": lambda config: StreamingGenerator(),
    "prompt_builder": lambda config: PromptBuilder(
        use_cache=_cache_enabled(config, "prompt")
    ),
    "output_parser": lambda config: OutputParser(),
    "memory_store": lambda config: _MEMORY_STORE,
    "memory_writer": lambda config: MemoryWriter(_MEMORY_STORE),
    "memory_filter": lambda config: MemoryFilter(),
    "evaluator": lambda config: Evaluator(),
    "ragas_evaluator": lambda config: RagasEvaluator(),
    "trulens_evaluator": lambda config: TruLensEvaluator(),
    "self_critic": lambda config: SelfCritic(
        generator=Generator(get_llm(config)),
        prompt_builder=PromptBuilder(
            template_dir=_POSTPROCESS_TEMPLATE_DIR,
            use_cache=_cache_enabled(config, "prompt"),
        ),
        parser=OutputParser(),
    ),
    "refiner": lambda config: Refiner(
        generator=Generator(get_llm(config)),
        prompt_builder=PromptBuilder(
            template_dir=_POSTPROCESS_TEMPLATE_DIR,
            use_cache=_cache_enabled(config, "prompt"),
        ),
        parser=OutputParser(),
    ),
    "context_builder": lambda config: ContextBuilder(),
    "context_merger": lambda config: ContextMerger(),
    "context_truncator": lambda config: ContextTruncator(),
}

REGISTRY: dict[str, ComponentCallable] = {
    "text_loader": lambda state, config: _ingest_with(_build_component("text_loader", config), state, config),
    "markdown_loader": lambda state, config: _ingest_with(_build_component("markdown_loader", config), state, config),
    "document_loader": lambda state, config: _ingest_with(_build_component("document_loader", config), state, config),
    "directory_loader": lambda state, config: _ingest_with(_build_component("directory_loader", config), state, config),
    "source_normalizer": lambda state, config: _normalize_sources_with(
        _build_component("source_normalizer", config), state, config
    ),
    "query_cleaner": lambda state, config: _clean_query_with(_build_component("query_cleaner", config), state, config),
    "query_rewriter": lambda state, config: _rewrite_query_with(_build_component("query_rewriter", config), state, config),
    "multi_query_generator": lambda state, config: _multi_query_with(
        _build_component("multi_query_generator", config), state, config
    ),
    "coarse_retriever": lambda state, config: _retrieve_with(_build_component("coarse_retriever", config), state, config),
    "fine_retriever": lambda state, config: _retrieve_with(_build_component("fine_retriever", config), state, config),
    "hybrid_retriever": lambda state, config: _hybrid_retrieve_with(
        _build_component("hybrid_retriever", config), state, config
    ),
    "memory_retriever": lambda state, config: _retrieve_with(_build_component("memory_retriever", config), state, config),
    "graph_retriever": lambda state, config: _retrieve_with(_build_component("graph_retriever", config), state, config),
    "external_retriever": lambda state, config: _retrieve_with(_build_component("external_retriever", config), state, config),
    "late_chunker": lambda state, config: _chunk_with(_build_component("late_chunker", config), state, config),
    "semantic_chunker": lambda state, config: _chunk_with(_build_component("semantic_chunker", config), state, config),
    "recursive_chunker": lambda state, config: _chunk_with(_build_component("recursive_chunker", config), state, config),
    "embedding_ranker": lambda state, config: _rank_with(_build_component("embedding_ranker", config), state, config),
    "colbert_ranker": lambda state, config: _rank_with(_build_component("colbert_ranker", config), state, config),
    "cross_encoder_ranker": lambda state, config: _rank_with(_build_component("cross_encoder_ranker", config), state, config),
    "rank_fusion": lambda state, config: _rank_fusion_with(
        _build_component("rank_fusion", config), state, config
    ),
    "generator": lambda state, config: _generate_with(_build_component("generator", config), state, config),
    "streaming_generator": lambda state, config: _stream_generate_with(
        _build_component("streaming_generator", config), state, config
    ),
    "prompt_builder": lambda state, config: _build_prompt_with(_build_component("prompt_builder", config), state, config),
    "output_parser": lambda state, config: _parse_output_with(_build_component("output_parser", config), state, config),
    "memory_store": lambda state, config: _memory_retrieve_with(_build_component("memory_store", config), state, config),
    "memory_writer": lambda state, config: _memory_write_with(_build_component("memory_writer", config), state, config),
    "memory_filter": lambda state, config: _memory_filter_with(_build_component("memory_filter", config), state, config),
    "evaluator": lambda state, config: _evaluate_with(_build_component("evaluator", config), state, config),
    "ragas_evaluator": lambda state, config: _evaluate_with(_build_component("ragas_evaluator", config), state, config),
    "trulens_evaluator": lambda state, config: _evaluate_with(_build_component("trulens_evaluator", config), state, config),
    "self_critic": lambda state, config: _critique_with(_build_component("self_critic", config), state, config),
    "refiner": lambda state, config: _refine_with(_build_component("refiner", config), state, config),
    "context_builder": lambda state, config: _context_build_with(_build_component("context_builder", config), state, config),
    "context_merger": lambda state, config: _context_merge_with(_build_component("context_merger", config), state, config),
    "context_truncator": lambda state, config: _context_truncate_with(
        _build_component("context_truncator", config), state, config
    ),
    "llm_generator": lambda state, config: _generate_with(_build_component("generator", config), state, config),
    "hybrid_merger": lambda state, config: _context_merge_with(_build_component("context_merger", config), state, config),
    "embedding_indexer": lambda state, config: _index_with(_build_component("embedding_indexer", config), state, config),
    "coarse_indexer": lambda state, config: _index_with(_build_component("coarse_indexer", config), state, config)
}

COMPONENT_CLASSES = {
    "BaseRetriever": BaseRetriever,
    "BaseRanker": BaseRanker,
    "MemoryRecord": MemoryRecord,
    "RetrievedChunk": RetrievedChunk,
}
