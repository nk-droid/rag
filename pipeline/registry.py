from typing import Any, Callable

from components._base import ComponentSettings
from components.ranking import BaseRanker
from components.retrieval import BaseRetriever
from components.shared_types import MemoryRecord, RetrievedChunk
from pipeline.component_factories import COMPONENT_FACTORIES
from pipeline.registry_handlers import (
    _build_prompt_with,
    _chunk_with,
    _clean_query_with,
    _context_build_with,
    _context_merge_with,
    _context_truncate_with,
    _critique_with,
    _evaluate_with,
    _generate_with,
    _hybrid_retrieve_with,
    _index_with,
    _ingest_with,
    _memory_filter_with,
    _memory_retrieve_with,
    _memory_write_with,
    _multi_query_with,
    _normalize_sources_with,
    _parse_output_with,
    _rank_fusion_with,
    _rank_with,
    _refine_with,
    _retrieve_with,
    _rewrite_query_with,
    _stream_generate_with,
)
from pipeline.registry_utils import _config_cache_key

ComponentCallable = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]

_COMPONENT_CACHE: dict[tuple[str, str], Any] = {}

def _build_component(name: str, config: dict[str, Any]) -> Any:
    cache_key = (name, _config_cache_key(config))
    if cache_key not in _COMPONENT_CACHE:
        _COMPONENT_CACHE[cache_key] = COMPONENT_FACTORIES[name](config)
    return _COMPONENT_CACHE[cache_key]

def _apply_step_overrides(component: Any, state: dict[str, Any]) -> Any:
    """Return a per-call clone of `component` with step overrides merged into settings.

    Reads `state["_step"]` (set by the orchestrator) and filters out reserved
    keys (`name`, `component`). The remaining keys are passed to
    `settings.with_overrides`, which only applies fields the Settings model
    declares — unknown step keys (e.g. `template_name` for the prompt builder)
    are left for handlers to read directly from `_step`.
    """
    step_meta = state.get("_step") or {}
    overrides = {k: v for k, v in step_meta.items() if k not in {"name", "component"}}
    if not overrides:
        return component
    settings = getattr(component, "settings", None)
    if not isinstance(settings, ComponentSettings):
        return component
    new_settings = settings.with_overrides(overrides)
    if new_settings is settings:
        return component
    clone = type(component).__new__(type(component))
    clone.__dict__.update(component.__dict__)
    clone.settings = new_settings
    return clone

def bind(component_name, handler):
    def run(state, config):
        component = _build_component(component_name, config)
        component = _apply_step_overrides(component, state)
        return handler(component, state, config)

    return run

REGISTRY: dict[str, ComponentCallable] = {
    "text_loader": bind("text_loader", _ingest_with),
    "markdown_loader": bind("markdown_loader", _ingest_with),
    "document_loader": bind("document_loader", _ingest_with),
    "directory_loader": bind("directory_loader", _ingest_with),
    "source_normalizer": bind("source_normalizer", _normalize_sources_with),
    "query_cleaner": bind("query_cleaner", _clean_query_with),
    "query_rewriter": bind("query_rewriter", _rewrite_query_with),
    "multi_query_generator": bind("multi_query_generator", _multi_query_with),
    "coarse_retriever": bind("coarse_retriever", _retrieve_with),
    "fine_retriever": bind("fine_retriever", _retrieve_with),
    "hybrid_retriever": bind("hybrid_retriever", _hybrid_retrieve_with),
    "memory_retriever": bind("memory_retriever", _retrieve_with),
    "graph_retriever": bind("graph_retriever", _retrieve_with),
    "external_retriever": bind("external_retriever", _retrieve_with),
    "late_chunker": bind("late_chunker", _chunk_with),
    "semantic_chunker": bind("semantic_chunker", _chunk_with),
    "recursive_chunker": bind("recursive_chunker", _chunk_with),
    "embedding_ranker": bind("embedding_ranker", _rank_with),
    "colbert_ranker": bind("colbert_ranker", _rank_with),
    "cross_encoder_ranker": bind("cross_encoder_ranker", _rank_with),
    "rank_fusion": bind("rank_fusion", _rank_fusion_with),
    "generator": bind("generator", _generate_with),
    "streaming_generator": bind("streaming_generator", _stream_generate_with),
    "prompt_builder": bind("prompt_builder", _build_prompt_with),
    "output_parser": bind("output_parser", _parse_output_with),
    "memory_store": bind("memory_store", _memory_retrieve_with),
    "memory_writer": bind("memory_writer", _memory_write_with),
    "memory_filter": bind("memory_filter", _memory_filter_with),
    "evaluator": bind("evaluator", _evaluate_with),
    "ragas_evaluator": bind("ragas_evaluator", _evaluate_with),
    "trulens_evaluator": bind("trulens_evaluator", _evaluate_with),
    "self_critic": bind("self_critic", _critique_with),
    "refiner": bind("refiner", _refine_with),
    "context_builder": bind("context_builder", _context_build_with),
    "context_merger": bind("context_merger", _context_merge_with),
    "context_truncator": bind("context_truncator", _context_truncate_with),
    "llm_generator": bind("generator", _generate_with),
    "hybrid_merger": bind("context_merger", _context_merge_with),
    "embedding_indexer": bind("embedding_indexer", _index_with),
    "coarse_indexer": bind("coarse_indexer", _index_with),
}

COMPONENT_CLASSES = {
    "BaseRetriever": BaseRetriever,
    "BaseRanker": BaseRanker,
    "MemoryRecord": MemoryRecord,
    "RetrievedChunk": RetrievedChunk,
}
