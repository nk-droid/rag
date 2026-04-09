from typing import Any, Callable

from components.chunking import LateChunker, RecursiveChunker, SemanticChunker
from components.context import ContextBuilder, ContextMerger, ContextTruncator
from components.evaluation import Evaluator, RagasEvaluator, TruLensEvaluator
from components.generation import Generator, OutputParser, PromptBuilder, StreamingGenerator
from components.ingestion import DirectoryLoader, DocumentLoader, MarkdownLoader, SourceNormalizer, TextLoader
from components.indexer import EmbeddingIndexer, CoarseIndexer
from components.memory import MemoryFilter, MemoryStore, MemoryWriter
from components.postprocessing import AnswerCleaner, Refiner, SelfCritic
from components.query import MultiQueryGenerator, QueryCleaner, QueryRewriter, RetrievalDecider
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
from infra.storage.vector_store_factory import get_vector_store
from infra.llm.llm_factory import get_llm

ComponentCallable = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
ComponentFactory = Callable[[dict[str, Any]], Any]

def _ensure_state(state: dict[str, Any] | None) -> dict[str, Any]:
    return state if isinstance(state, dict) else {}

_COMPONENT_CACHE: dict[tuple[str, str], Any] = {}

def _config_cache_key(config: dict[str, Any]) -> str:
    vector_store = config.get("vector_store", {})
    models = config.get("models", {})
    embedding = models.get("embedding", {})
    return repr(
        {
            "vector_store": vector_store,
            "embedding": embedding,
        }
    )

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
    query = payload.get("query", "")
    top_k = int(payload.get("top_k", 5))
    results = retriever.retrieve(query, top_k=top_k)
    payload["retrieved"] = results
    payload["retriever"] = retriever.__class__.__name__
    payload["config"] = config
    return payload

def _hybrid_retrieve_with(
    retriever: HybridRetriever,
    state: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    payload = _ensure_state(state)
    step_cfg = payload.get("_step", {}) if isinstance(payload.get("_step"), dict) else {}
    retrieval_cfg = config.get("retrieval", {})

    query = payload.get("query", "")
    top_k = int(step_cfg.get("top_k", payload.get("top_k", retrieval_cfg.get("top_k", 5))))

    candidate_sets = retriever.retrieve_candidates(query, top_k=top_k)
    sparse = list(candidate_sets.get("sparse", []))
    dense = list(candidate_sets.get("dense", []))

    # Option A: simple combined list
    combined = sparse + dense

    # Option B: fuse immediately if step sets `fuse: true`
    if bool(step_cfg.get("fuse", False)):
        fusion = _build_component("rank_fusion", config)
        combined = fusion.fuse([sparse, dense])

    payload["sparse_retrieved"] = sparse
    payload["dense_retrieved"] = dense
    payload["retrieved"] = combined
    payload["retriever"] = retriever.__class__.__name__
    payload["config"] = config
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
        context = "\n\n".join(chunk.text if isinstance(chunk, RetrievedChunk) else chunk["content"] for chunk in payload["ranked"])
    inputs = {
        "query": payload.get("query", ""),
        "context": context
    }
    prompt = payload["prompt"]
    payload["answer"] = generator.generate(prompt, inputs)
    payload["generator"] = generator.__class__.__name__
    payload["config"] = config
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

def _decide_retrieval_with(
    decider: RetrievalDecider,
    state: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    payload = _ensure_state(state)
    payload["retrieval_plan"] = decider.decide(payload.get("query", ""))
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
    payload["parsed_output"] = parser.parse(
        payload.get("answer").content,
        parser_model=payload.get("_step", {}).get("parser", None)
    )
    payload["config"] = config
    return payload

def _memory_write_with(writer: MemoryWriter, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    payload = _ensure_state(state)
    interaction = {
        "id": payload.get("memory_id", "memory-0"),
        "content": payload.get("answer", ""),
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

def _clean_answer_with(cleaner: AnswerCleaner, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    payload = _ensure_state(state)
    payload["answer"] = cleaner.clean(payload.get("answer", ""))
    payload["config"] = config
    return payload

def _identity(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    payload = _ensure_state(state)
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
    "query_rewriter": lambda config: QueryRewriter(),
    "multi_query_generator": lambda config: MultiQueryGenerator(),
    "retrieval_decider": lambda config: RetrievalDecider(),
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
        top_n=int(config.get("ranking", {}).get("top_k", 3))
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
    "prompt_builder": lambda config: PromptBuilder(),
    "output_parser": lambda config: OutputParser(),
    "memory_store": lambda config: _MEMORY_STORE,
    "memory_writer": lambda config: MemoryWriter(_MEMORY_STORE),
    "memory_filter": lambda config: MemoryFilter(),
    "evaluator": lambda config: Evaluator(),
    "ragas_evaluator": lambda config: RagasEvaluator(),
    "trulens_evaluator": lambda config: TruLensEvaluator(),
    "self_critic": lambda config: SelfCritic(),
    "refiner": lambda config: Refiner(),
    "answer_cleaner": lambda config: AnswerCleaner(),
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
    "retrieval_decider": lambda state, config: _decide_retrieval_with(
        _build_component("retrieval_decider", config), state, config
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
    "answer_cleaner": lambda state, config: _clean_answer_with(_build_component("answer_cleaner", config), state, config),
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
