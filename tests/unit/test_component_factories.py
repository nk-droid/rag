import pytest

from api.catalog import CATALOG
from pipeline.component_factories import COMPONENT_FACTORIES
from pipeline.registry import REGISTRY

EXPECTED_PRIMARY_METHOD = {
    "text_loader": "load",
    "code_loader": "load",
    "markdown_loader": "load",
    "document_loader": "load",
    "directory_loader": "load",
    "repo_loader": "load",
    "source_normalizer": "normalize",
    "embedding_indexer": "index",
    "coarse_indexer": "index",
    "repo_graph_indexer": "index",
    "query_cleaner": "clean",
    "query_rewriter": "rewrite",
    "multi_query_generator": "generate",
    "coarse_retriever": "retrieve",
    "fine_retriever": "retrieve",
    "hybrid_retriever": "retrieve",
    "memory_retriever": "retrieve",
    "graph_retriever": "retrieve",
    "graph_expander": "expand",
    "external_retriever": "retrieve",
    "late_chunker": "chunk",
    "semantic_chunker": "chunk",
    "recursive_chunker": "chunk",
    "code_aware_chunker": "chunk",
    "embedding_ranker": "rank",
    "colbert_ranker": "rank",
    "cross_encoder_ranker": "rank",
    "rank_fusion": "fuse",
    "generator": "generate",
    "streaming_generator": "stream",
    "prompt_builder": "build",
    "output_parser": "parse",
    "memory_store": "search",
    "memory_writer": "write",
    "memory_filter": "filter",
    "evaluator": "evaluate",
    "ragas_evaluator": "evaluate",
    "self_critic": "critique",
    "refiner": "refine",
    "context_builder": "build",
    "context_merger": "merge",
    "context_truncator": "truncate",
}

@pytest.mark.parametrize("component_name", sorted(COMPONENT_FACTORIES.keys()))
def test_component_factories_build(component_name, minimal_config, patched_factory_dependencies):
    component = COMPONENT_FACTORIES[component_name](minimal_config)
    assert component is not None

    expected_method = EXPECTED_PRIMARY_METHOD.get(component_name)
    assert expected_method is not None, f"Missing expected method map for {component_name}"
    assert callable(getattr(component, expected_method, None))

def test_registry_keys_have_factory_or_alias() -> None:
    aliases = {
        "llm_generator": "generator",
        "hybrid_merger": "context_merger",
    }

    missing = [
        key
        for key in REGISTRY
        if key not in COMPONENT_FACTORIES and key not in aliases
    ]
    assert missing == []

def test_factory_keys_are_reachable_from_registry_or_registry_aliases() -> None:
    registry_keys = set(REGISTRY.keys())
    registry_alias_targets = {
        "generator",
        "context_merger",
    }

    unreachable = [
        key
        for key in COMPONENT_FACTORIES
        if key not in registry_keys and key not in registry_alias_targets
    ]
    assert unreachable == []

def test_component_catalog_matches_selectable_backend_steps() -> None:
    expected_picker_components = {
        "recursive_chunker",
        "semantic_chunker",
        "code_aware_chunker",
        "late_chunker",
        "coarse_indexer",
        "embedding_indexer",
        "repo_graph_indexer",
        "query_cleaner",
        "query_rewriter",
        "multi_query_generator",
        "coarse_retriever",
        "fine_retriever",
        "hybrid_retriever",
        "external_retriever",
        "memory_retriever",
        "graph_retriever",
        "rank_fusion",
        "graph_expander",
        "embedding_ranker",
        "cross_encoder_ranker",
        "colbert_ranker",
        "context_merger",
        "context_builder",
        "context_truncator",
        "prompt_builder",
        "llm_generator",
        "output_parser",
        "streaming_generator",
        "self_critic",
        "refiner",
        "ragas_evaluator",
        "evaluator",
    }
    catalog_components = {
        subcomponent.id
        for group in CATALOG
        for subcomponent in group.subcomponents
    }

    assert catalog_components == expected_picker_components
    assert catalog_components <= set(REGISTRY.keys())
