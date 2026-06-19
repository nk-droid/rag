from pathlib import Path
from typing import Any, Callable

from components.chunking import (
    LateChunker,
    LateChunkerSettings,
    RecursiveChunker,
    RecursiveChunkerSettings,
    SemanticChunker,
    SemanticChunkerSettings,
)
from components.chunking.code_aware_chunker import CodeAwareChunker, CodeAwareChunkerSettings
from components.context import (
    ContextBuilder,
    ContextBuilderSettings,
    ContextMerger,
    ContextMergerSettings,
    ContextTruncator,
    ContextTruncatorSettings,
)
from components.evaluation import (
    Evaluator,
    EvaluatorSettings,
    RagasEvaluator,
    RagasEvaluatorSettings,
)
from components.generation import (
    Generator,
    GeneratorSettings,
    OutputParser,
    OutputParserSettings,
    PromptBuilder,
    PromptBuilderSettings,
    StreamingGenerator,
    StreamingGeneratorSettings,
)
from components.indexer import (
    CoarseIndexer,
    CoarseIndexerSettings,
    EmbeddingIndexer,
    EmbeddingIndexerSettings,
)
from components.indexer.repo_graph_indexer import RepoGraphIndexer, RepoGraphIndexerSettings
from components.ingestion import (
    DirectoryLoader,
    DirectoryLoaderSettings,
    DocumentLoader,
    DocumentLoaderSettings,
    MarkdownLoader,
    MarkdownLoaderSettings,
    SourceNormalizer,
    SourceNormalizerSettings,
    TextLoader,
    TextLoaderSettings,
)
from components.ingestion.code_loader import CodeLoader, CodeLoaderSettings
from components.ingestion.repo_loader import RepoLoader, RepoLoaderSettings
from components.memory import (
    MemoryFilter,
    MemoryFilterSettings,
    MemoryStore,
    MemoryStoreSettings,
    MemoryWriter,
    MemoryWriterSettings,
)
from components.postprocessing import (
    Refiner,
    RefinerSettings,
    SelfCritic,
    SelfCriticSettings,
)
from components.query import (
    MultiQueryGenerator,
    MultiQueryGeneratorSettings,
    QueryCleaner,
    QueryCleanerSettings,
    QueryRewriter,
    QueryRewriterSettings,
)
from components.ranking import (
    ColBERTRanker,
    ColBERTRankerSettings,
    CrossEncoderRanker,
    CrossEncoderRankerSettings,
    EmbeddingRanker,
    EmbeddingRankerSettings,
    RankFusion,
    RankFusionSettings,
)
from components.retrieval import (
    CoarseRetriever,
    CoarseRetrieverSettings,
    ExternalRetriever,
    ExternalRetrieverSettings,
    FineRetriever,
    FineRetrieverSettings,
    GraphRetriever,
    GraphRetrieverSettings,
    HybridRetriever,
    HybridRetrieverSettings,
    MemoryRetriever,
    MemoryRetrieverSettings,
)
from components.retrieval.graph_expander import GraphExpander, GraphExpanderSettings
from infra.llm.llm_factory import get_llm
from infra.storage.vector_store_factory import get_vector_store
from pipeline.factory_helpers import build_component

ComponentFactory = Callable[[dict[str, Any]], Any]

_REPO_ROOT = Path(__file__).resolve().parent.parent
_QUERY_TEMPLATES = _REPO_ROOT / "components" / "query" / "templates"
_POSTPROCESS_TEMPLATES = _REPO_ROOT / "components" / "postprocessing" / "templates"
_CHUNKING_TEMPLATES = _REPO_ROOT / "components" / "chunking" / "templates"
_GENERATION_TEMPLATES = _REPO_ROOT / "components" / "generation" / "templates"

_MEMORY_STORE_SINGLETON: MemoryStore | None = None
def _memory_store_singleton(config: dict[str, Any]) -> MemoryStore:
    global _MEMORY_STORE_SINGLETON
    if _MEMORY_STORE_SINGLETON is None:
        _MEMORY_STORE_SINGLETON = build_component(MemoryStore, MemoryStoreSettings, config)
    return _MEMORY_STORE_SINGLETON

def _output_parser(config: dict[str, Any]) -> OutputParser:
    return build_component(OutputParser, OutputParserSettings, config)

def _prompt_builder(config: dict[str, Any], template_dir: Path) -> PromptBuilder:
    return build_component(
        PromptBuilder,
        PromptBuilderSettings,
        config,
        deps_builder=lambda settings, cfg: {"template_dir": template_dir},
    )

def _generator(config: dict[str, Any]) -> Generator:
    return build_component(
        Generator,
        GeneratorSettings,
        config,
        deps_builder=lambda settings, cfg: {"llm": get_llm(cfg)},
    )

def _streaming_generator(config: dict[str, Any]) -> StreamingGenerator:
    return build_component(
        StreamingGenerator,
        StreamingGeneratorSettings,
        config,
        deps_builder=lambda settings, cfg: {"llm": get_llm(cfg)},
    )

def _semantic_chunker_generator(settings: SemanticChunkerSettings) -> Generator:
    llm_cfg = {"models": {"llm": dict(settings.llm)}}
    return Generator(settings=GeneratorSettings(), llm=get_llm(llm_cfg))

def _document_loader(config: dict[str, Any]) -> DocumentLoader:
    return build_component(
        DocumentLoader,
        DocumentLoaderSettings,
        config,
        deps_builder=lambda settings, cfg: {
            "markdown_loader": build_component(MarkdownLoader, MarkdownLoaderSettings, cfg),
            "text_loader": build_component(TextLoader, TextLoaderSettings, cfg),
        },
    )

def _build_fine_retriever(config: dict[str, Any]) -> FineRetriever:
    return build_component(
        FineRetriever,
        FineRetrieverSettings,
        config,
        deps_builder=lambda settings, cfg: {
            "store": get_vector_store({
                **cfg,
                "vector_store": {
                    **cfg.get("vector_store", {}),
                    "path": settings.path,
                },
            }),
        },
    )

COMPONENT_FACTORIES: dict[str, ComponentFactory] = {
    # ingestion
    "text_loader": lambda c: build_component(TextLoader, TextLoaderSettings, c),
    "markdown_loader": lambda c: build_component(MarkdownLoader, MarkdownLoaderSettings, c),
    "document_loader": _document_loader,
    "directory_loader": lambda c: build_component(
        DirectoryLoader,
        DirectoryLoaderSettings,
        c,
        deps_builder=lambda s, cfg: {"loader": _document_loader(cfg)},
    ),
    "code_loader": lambda c: build_component(CodeLoader, CodeLoaderSettings, c),
    "repo_loader": lambda c: build_component(RepoLoader, RepoLoaderSettings, c),
    "source_normalizer": lambda c: build_component(SourceNormalizer, SourceNormalizerSettings, c),

    # chunking
    "recursive_chunker": lambda c: build_component(RecursiveChunker, RecursiveChunkerSettings, c),
    "late_chunker": lambda c: build_component(LateChunker, LateChunkerSettings, c),
    "semantic_chunker": lambda c: build_component(
        SemanticChunker,
        SemanticChunkerSettings,
        c,
        deps_builder=lambda s, cfg: {
            "prompt_builder": _prompt_builder(cfg, _CHUNKING_TEMPLATES),
            "generator": _semantic_chunker_generator(s),
            "parser": _output_parser(cfg),
        },
    ),
    "code_aware_chunker": lambda c: build_component(CodeAwareChunker, CodeAwareChunkerSettings, c),

    # indexers
    "coarse_indexer": lambda c: build_component(CoarseIndexer, CoarseIndexerSettings, c),
    "embedding_indexer": lambda c: build_component(
        EmbeddingIndexer,
        EmbeddingIndexerSettings,
        c,
        deps_builder=lambda s, cfg: {
            "vector_store": get_vector_store({
                **cfg,
                "vector_store": {**dict(s.vector_store), "path": s.path},
            }),
        },
    ),
    "repo_graph_indexer": lambda c: build_component(RepoGraphIndexer, RepoGraphIndexerSettings, c),

    # query
    "query_cleaner": lambda c: build_component(QueryCleaner, QueryCleanerSettings, c),
    "query_rewriter": lambda c: build_component(
        QueryRewriter,
        QueryRewriterSettings,
        c,
        deps_builder=lambda s, cfg: {
            "generator": _generator(cfg),
            "prompt_builder": _prompt_builder(cfg, _QUERY_TEMPLATES),
            "parser": _output_parser(cfg),
        },
    ),
    "multi_query_generator": lambda c: build_component(
        MultiQueryGenerator,
        MultiQueryGeneratorSettings,
        c,
        deps_builder=lambda s, cfg: {
            "generator": _generator(cfg),
            "prompt_builder": _prompt_builder(cfg, _QUERY_TEMPLATES),
            "parser": _output_parser(cfg),
        },
    ),

    # retrieval
    "coarse_retriever": lambda c: build_component(CoarseRetriever, CoarseRetrieverSettings, c),
    "fine_retriever": _build_fine_retriever,
    "hybrid_retriever": lambda c: build_component(
        HybridRetriever,
        HybridRetrieverSettings,
        c,
        deps_builder=lambda s, cfg: {
            "dense_retriever": _build_fine_retriever(cfg),
            "sparse_retriever": build_component(CoarseRetriever, CoarseRetrieverSettings, cfg),
        },
    ),
    "memory_retriever": lambda c: build_component(MemoryRetriever, MemoryRetrieverSettings, c),
    "graph_retriever": lambda c: build_component(GraphRetriever, GraphRetrieverSettings, c),
    "graph_expander": lambda c: build_component(GraphExpander, GraphExpanderSettings, c),
    "external_retriever": lambda c: build_component(ExternalRetriever, ExternalRetrieverSettings, c),

    # ranking
    "embedding_ranker": lambda c: build_component(EmbeddingRanker, EmbeddingRankerSettings, c),
    "cross_encoder_ranker": lambda c: build_component(CrossEncoderRanker, CrossEncoderRankerSettings, c),
    "colbert_ranker": lambda c: build_component(ColBERTRanker, ColBERTRankerSettings, c),
    "rank_fusion": lambda c: build_component(RankFusion, RankFusionSettings, c),

    # generation
    "generator": _generator,
    "streaming_generator": _streaming_generator,
    "prompt_builder": lambda c: _prompt_builder(c, _GENERATION_TEMPLATES),
    "output_parser": _output_parser,

    # postprocessing
    "self_critic": lambda c: build_component(
        SelfCritic,
        SelfCriticSettings,
        c,
        deps_builder=lambda s, cfg: {
            "generator": _generator(cfg),
            "prompt_builder": _prompt_builder(cfg, _POSTPROCESS_TEMPLATES),
            "parser": _output_parser(cfg),
        },
    ),
    "refiner": lambda c: build_component(
        Refiner,
        RefinerSettings,
        c,
        deps_builder=lambda s, cfg: {
            "generator": _generator(cfg),
            "prompt_builder": _prompt_builder(cfg, _POSTPROCESS_TEMPLATES),
            "parser": _output_parser(cfg),
        },
    ),

    # context
    "context_builder": lambda c: build_component(ContextBuilder, ContextBuilderSettings, c),
    "context_merger": lambda c: build_component(ContextMerger, ContextMergerSettings, c),
    "context_truncator": lambda c: build_component(ContextTruncator, ContextTruncatorSettings, c),

    # memory
    "memory_store": _memory_store_singleton,
    "memory_writer": lambda c: build_component(
        MemoryWriter,
        MemoryWriterSettings,
        c,
        deps_builder=lambda s, cfg: {"store": _memory_store_singleton(cfg)},
    ),
    "memory_filter": lambda c: build_component(MemoryFilter, MemoryFilterSettings, c),

    # evaluation
    "evaluator": lambda c: build_component(Evaluator, EvaluatorSettings, c),
    "ragas_evaluator": lambda c: build_component(RagasEvaluator, RagasEvaluatorSettings, c),
}
