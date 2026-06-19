from dataclasses import dataclass

@dataclass(frozen=True)
class StepContract:
    requires: tuple[frozenset[str], ...]
    produces: frozenset[str]
    phase: str = "run"

def _c(requires: list[set[str]], produces: set[str], phase: str = "run") -> StepContract:
    return StepContract(
        requires=tuple(frozenset(group) for group in requires),
        produces=frozenset(produces),
        phase=phase,
    )

# Components that read from a source and emit `documents` (handler: _ingest_with).
_LOADERS = [
    "text_loader",
    "markdown_loader",
    "document_loader",
    "directory_loader",
    "code_loader",
    "repo_loader",
]

# Components that split text into `chunks` (handler: _chunk_with).
_CHUNKERS = [
    "late_chunker",
    "semantic_chunker",
    "recursive_chunker",
    "code_aware_chunker",
]

# Components that persist an index from `chunks` (handler: _index_with).
_INDEXERS = [
    "embedding_indexer",
    "coarse_indexer",
    "repo_graph_indexer",
    "graph_indexer",
]

# Components that pull candidates by `query` into `retrieved` (handler: _retrieve_with).
_SIMPLE_RETRIEVERS = [
    "coarse_retriever",
    "fine_retriever",
    "memory_retriever",
    "graph_retriever",
    "external_retriever",
]

# Components that rerank `retrieved` into `ranked` (handler: _rank_with).
_RANKERS = [
    "embedding_ranker",
    "colbert_ranker",
    "cross_encoder_ranker",
]

CONTRACTS: dict[str, StepContract] = {}

for _name in _LOADERS:
    CONTRACTS[_name] = _c([{"sources", "data_sources"}], {"documents"}, phase="init")

for _name in _CHUNKERS:
    CONTRACTS[_name] = _c([{"documents", "data_sources"}], {"chunks"}, phase="init")

for _name in _INDEXERS:
    CONTRACTS[_name] = _c([{"chunks"}], {"index_records"}, phase="init")

for _name in _SIMPLE_RETRIEVERS:
    CONTRACTS[_name] = _c([{"query"}], {"retrieved", "retrieval_queries"})

for _name in _RANKERS:
    CONTRACTS[_name] = _c([{"retrieved"}], {"ranked"})

CONTRACTS.update(
    {
        "source_normalizer": _c([{"documents"}], {"data_sources"}, phase="init"),
        "query_cleaner": _c([{"query"}], {"query"}),
        "query_rewriter": _c([{"query"}], {"query"}),
        "multi_query_generator": _c([{"query"}], {"queries"}),
        "hybrid_retriever": _c(
            [{"query"}],
            {"retrieved", "sparse_retrieved", "dense_retrieved", "retrieval_queries"},
        ),
        "graph_expander": _c([{"retrieved"}], {"retrieved", "graph_expanded"}),
        "rank_fusion": _c(
            [{"sparse_retrieved", "dense_retrieved", "retrieved"}], {"retrieved"}
        ),
        "context_builder": _c([{"ranked", "retrieved"}], {"context"}),
        "context_merger": _c([{"retrieved"}], {"retrieved"}),
        "hybrid_merger": _c([{"retrieved"}], {"retrieved"}),
        "context_truncator": _c([{"context"}], {"context"}),
        "prompt_builder": _c([], {"prompt"}),
        "generator": _c([{"prompt"}], {"answer"}),
        "llm_generator": _c([{"prompt"}], {"answer"}),
        "streaming_generator": _c([{"prompt"}], {"answer", "stream"}),
        "output_parser": _c([{"answer"}], {"parsed_output"}),
        "memory_store": _c([{"query"}], {"memories"}),
        "memory_writer": _c([{"answer"}], {"memory_record"}),
        "memory_filter": _c([{"memories"}], {"memories"}),
        "self_critic": _c([{"answer"}], {"critique"}),
        "refiner": _c([{"answer", "critique"}], {"answer"}),
        "evaluator": _c([], {"evaluation"}),
        "ragas_evaluator": _c([], {"evaluation"}),
    }
)
