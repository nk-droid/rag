from typing import Any

from components.shared_types import RetrievedChunk
from pipeline.component_factories import COMPONENT_FACTORIES
from pipeline.registry_utils import (
    _answer_text,
    _cache_enabled,
    _cache_key,
    _cache_ttl,
    _config_cache_key,
    _deserialize_chunks,
    _ensure_state,
    _extract_chunk_inputs,
    _generation_cacheable,
    _get_cache,
    _index_fingerprint,
    _mark_cache_hit,
    _merge_retrieval_chunks,
    _retrieval_queries,
    _serialize_chunks,
    stable_hash,
    text_hash,
)

_AUX_COMPONENT_CACHE: dict[tuple[str, str], Any] = {}
def _build_aux_component(name: str, config: dict[str, Any]) -> Any:
    cache_key = (name, _config_cache_key(config))
    if cache_key not in _AUX_COMPONENT_CACHE:
        _AUX_COMPONENT_CACHE[cache_key] = COMPONENT_FACTORIES[name](config)
    return _AUX_COMPONENT_CACHE[cache_key]

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
    normalizer: Any,
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

    for source_index, (text, metadata) in enumerate(chunk_inputs):
        source_chunks = chunker.chunk(text)

        for chunk_index, chunk in enumerate(source_chunks):
            if not hasattr(chunk, "metadata") or not isinstance(chunk.metadata, dict):
                continue

            chunk.metadata.update(metadata)
            chunk.metadata.setdefault("source_index", source_index)
            chunk.metadata.setdefault("chunk_index", chunk_index)

            source_key = str(
                chunk.metadata.get("relative_path")
                or chunk.metadata.get("path")
                or chunk.metadata.get("source")
                or chunk.metadata.get("doc_id")
                or chunk.metadata.get("title")
                or source_index
            )

            chunk.metadata["chunk_id"] = (
                f"chunk:{text_hash(f'{source_key}:{chunk_index}:{chunk.text}')[:16]}"
            )

        chunks.extend(source_chunks)

    payload["chunks"] = chunks
    payload["chunker"] = chunker.__class__.__name__
    payload["config"] = config
    return payload

def _index_with(indexer: Any, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    payload = _ensure_state(state)
    chunks = payload.get("chunks", [])
    records = indexer.index(chunks)
    payload["index_records"] = records
    payload["indexed_count"] = len(records)
    payload["vector_store_path"] = str(getattr(indexer, "index_path", "") or getattr(indexer, "vector_db_path", ""))
    payload["indexer"] = indexer.__class__.__name__
    payload["config"] = config
    return payload

def _retrieve_with(retriever: Any, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    payload = _ensure_state(state)
    queries = _retrieval_queries(payload)
    query = queries[0] if queries else str(payload.get("query", "")).strip()
    top_k = int(payload.get("top_k", config.get("retrieval", {}).get("top_k", 5)))
    step_cfg = payload.get("_step", {}) if isinstance(payload.get("_step"), dict) else {}

    if_under = step_cfg.get("if_under")
    if if_under is not None and len(payload.get("retrieved", [])) >= int(if_under):
        payload["retrieval_skipped"] = retriever.__class__.__name__
        payload["config"] = config
        return payload

    merge_with_existing = bool(step_cfg.get("merge_with_existing", False))
    existing = list(payload.get("retrieved", [])) if merge_with_existing else []

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
            fresh = _deserialize_chunks(cached.get("retrieved", []))
            payload["retrieved"] = (
                _merge_retrieval_chunks(existing + fresh, top_k=top_k) if merge_with_existing else fresh
            )
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

    fresh_results = results
    if merge_with_existing:
        results = _merge_retrieval_chunks(existing + results, top_k=top_k)

    payload["retrieved"] = results
    payload["retrieval_queries"] = queries_to_run
    payload["retriever"] = retriever.__class__.__name__
    payload["config"] = config
    _mark_cache_hit(payload, "retrieval", False)

    if cache is not None and cache_key is not None:
        cache.set(
            cache_key,
            {
                "retrieved": _serialize_chunks(fresh_results),
                "retrieval_queries": queries_to_run,
            },
            ttl_sec=_cache_ttl(config, "retrieval", fallback=900),
        )

    return payload

def _hybrid_retrieve_with(
    retriever: Any,
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

    if bool(step_cfg.get("fuse", False)):
        fusion = _build_aux_component("rank_fusion", config)
        combined = fusion.fuse([sparse, dense])
        combined = _merge_retrieval_chunks(combined, top_k=top_k)
    else:
        # Delegate to HybridRetriever.retrieve() so per-source normalisation + weighting
        # happen before any cross-source comparison.
        combined = retriever.retrieve(query, top_k=top_k)

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

def _rank_fusion_with(fusion: Any, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
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

def _rank_with(ranker: Any, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    payload = _ensure_state(state)
    query = payload.get("query", "")
    candidates = payload.get("retrieved", [])
    payload["ranked"] = ranker.rank(query, candidates)
    payload["ranker"] = ranker.__class__.__name__
    payload["config"] = config
    return payload

def _generate_with(generator: Any, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    payload = _ensure_state(state)
    context = payload.get("context", "")
    if not context and payload.get("ranked"):
        context = "\n\n".join(
            chunk.text if isinstance(chunk, RetrievedChunk) else chunk["content"]
            for chunk in payload["ranked"]
        )
        
    payload["context"] = context
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

def _graph_expand_with(expander: Any, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    payload = _ensure_state(state)
    step_cfg = payload.get("_step", {}) if isinstance(payload.get("_step"), dict) else {}
    retrieval_cfg = config.get("retrieval", {})

    retrieved = list(payload.get("retrieved", []) or [])
    max_expanded = int(
        step_cfg.get(
            "max_expanded_chunks",
            getattr(expander.settings, "max_expanded_chunks", 20),
        )
    )

    payload["retrieved_before_graph_expand"] = retrieved
    expanded = expander.expand(retrieved, top_k=max_expanded)

    top_k = int(step_cfg.get("top_k", payload.get("top_k", retrieval_cfg.get("top_k", 5))))
    merged_cap = max(top_k, len(retrieved) + len(expanded))

    payload["graph_expanded"] = expanded
    payload["retrieved"] = _merge_retrieval_chunks(retrieved + expanded, top_k=merged_cap)
    payload["graph_expander"] = expander.__class__.__name__
    payload["config"] = config
    return payload

def _context_build_with(builder: Any, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    payload = _ensure_state(state)
    chunks = payload.get("ranked") or payload.get("retrieved", [])
    payload["context"] = builder.build(chunks)
    payload["config"] = config
    return payload

def _context_merge_with(merger: Any, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    payload = _ensure_state(state)
    chunks = payload.get("retrieved", [])
    payload["retrieved"] = merger.merge(chunks)
    payload["config"] = config
    return payload

def _context_truncate_with(
    truncator: Any,
    state: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    payload = _ensure_state(state)
    raw_max = payload.get("max_tokens")
    max_tokens = int(raw_max) if raw_max is not None else None
    payload["context"] = truncator.truncate(payload.get("context", ""), max_tokens)
    payload["config"] = config
    return payload

def _clean_query_with(cleaner: Any, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    payload = _ensure_state(state)
    payload["query"] = cleaner.clean(payload.get("query", ""))
    payload["config"] = config
    return payload

def _rewrite_query_with(rewriter: Any, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    payload = _ensure_state(state)
    payload["query"] = rewriter.rewrite(payload.get("query", ""))
    payload["config"] = config
    return payload

def _multi_query_with(
    generator: Any,
    state: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    payload = _ensure_state(state)
    payload["queries"] = generator.generate(payload.get("query", ""))
    payload["config"] = config
    return payload

def _stream_generate_with(
    generator: Any,
    state: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    payload = _ensure_state(state)
    prompt = payload.get("prompt")
    if prompt is None:
        raise ValueError(
            "streaming_generator requires a prompt — add prompt_builder upstream."
        )

    context = payload.get("context", "")
    if not context and payload.get("ranked"):
        context = "\n\n".join(
            chunk.text if isinstance(chunk, RetrievedChunk) else chunk["content"]
            for chunk in payload["ranked"]
        )
    inputs = {"query": payload.get("query", ""), "context": context}

    pieces: list[str] = []
    for piece in generator.stream(prompt, inputs):
        pieces.append(piece)

    payload["stream"] = pieces
    payload["answer"] = "".join(pieces)
    payload["generator"] = generator.__class__.__name__
    payload["config"] = config
    return payload

def _build_prompt_with(builder: Any, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    payload = _ensure_state(state)
    template_name = payload.get("_step", {}).get("template_name")
    payload["prompt"] = builder.build(
        template_name=template_name,
        parser_model=payload.get("_step", {}).get("parser", None)
    )
    payload["config"] = config
    return payload

def _parse_output_with(parser: Any, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    payload = _ensure_state(state)
    answer_text = _answer_text(payload.get("answer"))
    payload["parsed_output"] = parser.parse(
        answer_text,
        parser_model=payload.get("_step", {}).get("parser", None)
    )
    payload["config"] = config
    return payload

def _memory_write_with(writer: Any, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    payload = _ensure_state(state)
    interaction = {
        "id": payload.get("memory_id", "memory-0"),
        "content": _answer_text(payload.get("answer", "")),
    }
    payload["memory_record"] = writer.write(interaction)
    payload["config"] = config
    return payload

def _memory_retrieve_with(store: Any, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    payload = _ensure_state(state)
    top_k = int(payload.get("top_k", 5))
    payload["memories"] = store.search(payload.get("query", ""), top_k=top_k)
    payload["config"] = config
    return payload

def _memory_filter_with(
    memory_filter: Any,
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

def _critique_with(critic: Any, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    payload = _ensure_state(state)
    payload["critique"] = critic.critique(payload.get("answer", ""), payload.get("context", ""))
    payload["config"] = config
    return payload

def _refine_with(refiner: Any, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    payload = _ensure_state(state)
    payload["answer"] = refiner.refine(payload.get("answer", ""), payload.get("critique", {}))
    payload["config"] = config
    return payload
