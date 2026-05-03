from pathlib import Path
from typing import Any

from components.shared_types import RetrievedChunk
from infra.cache.base_cache import BaseCache
from infra.cache.cache_keys import file_signature, make_cache_key, stable_hash, text_hash
from infra.cache.in_memory_cache import InMemoryCache
from infra.cache.redis_cache import RedisCache

_CACHE_CLIENTS: dict[str, BaseCache] = {}

def _ensure_state(state: dict[str, Any] | None) -> dict[str, Any]:
    return state if isinstance(state, dict) else {}

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

_INDEXER_ALIASES = {
    "embedding_indexer": "embedding",
    "coarse_indexer": "coarse",
}

def _get_index_path(config: dict[str, Any], indexer_key: str) -> str:
    """Resolve the on-disk path for an indexer by config slice.

    Priority:
      1. New schema: `indexers.{embedding,coarse}.path`
      2. Legacy schema: `vector_store.{embedding,coarse}_indexer.path`
      3. Older legacy: `vector_store.path` (embedding only) /
         `coarse_index.path` (coarse only)
      4. Built-in default
    """
    canonical = _INDEXER_ALIASES.get(indexer_key, indexer_key.replace("_indexer", ""))

    indexers = config.get("indexers", {})
    if isinstance(indexers, dict):
        slice_cfg = indexers.get(canonical, {})
        if isinstance(slice_cfg, dict):
            path = slice_cfg.get("path")
            if path:
                return str(path)

    vector_store = config.get("vector_store", {})
    if isinstance(vector_store, dict):
        indexer_cfg = vector_store.get(indexer_key, {})
        if isinstance(indexer_cfg, dict):
            path = indexer_cfg.get("path")
            if path:
                return str(path)

    if canonical == "embedding":
        legacy_path = vector_store.get("path") if isinstance(vector_store, dict) else None
        if legacy_path:
            return str(legacy_path)
        return "data/indices/faiss_index"

    if canonical == "coarse":
        legacy_path = config.get("coarse_index", {}).get("path")
        if legacy_path:
            return str(legacy_path)
        return "data/indices/coarse_index.json"

    raise ValueError(f"Unsupported indexer key: {indexer_key}")

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
    extracted: list[tuple[str, dict[str, Any]]] = []

    # Prefer normalized/document payloads and avoid raw source path strings.
    for key in ("data_sources", "documents"):
        collection = payload.get(key)
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

    raw_sources = payload.get("sources")
    if isinstance(raw_sources, str):
        source_text = raw_sources.strip()
        if source_text and not Path(source_text).exists():
            extracted.append((source_text, {}))
    elif isinstance(raw_sources, list):
        for item in raw_sources:
            payload_item = _document_to_payload(item)
            text = str(payload_item.get("text", "")).strip()
            metadata = payload_item.get("metadata", {})
            if not text:
                continue
            if Path(text).exists():
                continue
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
