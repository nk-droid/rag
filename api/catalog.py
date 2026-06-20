from dataclasses import dataclass
from typing import Literal

ComponentStatus = Literal["ready", "not_implemented", "experimental"]

@dataclass(frozen=True, slots=True)
class SubcomponentDefinition:
    id: str
    label: str
    description: str
    status: ComponentStatus = "ready"

@dataclass(frozen=True, slots=True)
class ComponentGroupDefinition:
    id: str
    label: str
    description: str
    multi_select: bool
    required: bool
    default: list[str]
    subcomponents: tuple[SubcomponentDefinition, ...]

CATALOG: tuple[ComponentGroupDefinition, ...] = (
    ComponentGroupDefinition(
        id="chunking",
        label="Chunking",
        description="Split source documents into retrievable chunks.",
        multi_select=False,
        required=True,
        default=["recursive_chunker"],
        subcomponents=(
            SubcomponentDefinition(
                id="recursive_chunker",
                label="Recursive Chunker",
                description="Default text splitter using recursive boundaries.",
                status="ready",
            ),
            SubcomponentDefinition(
                id="semantic_chunker",
                label="Semantic Chunker",
                description="LLM-aware chunking strategy using semantic boundaries.",
                status="ready",
            ),
            SubcomponentDefinition(
                id="code_aware_chunker",
                label="Code-aware Chunker",
                description="Split repository files using code structure, symbols, and line metadata.",
                status="ready",
            ),
            SubcomponentDefinition(
                id="late_chunker",
                label="Late Chunker",
                description="Late interaction chunking strategy.",
                status="not_implemented",
            ),
        ),
    ),
    ComponentGroupDefinition(
        id="indexing",
        label="Indexing",
        description="Build one or more indices used by retrievers.",
        multi_select=True,
        required=False,
        default=["coarse_indexer"],
        subcomponents=(
            SubcomponentDefinition(
                id="coarse_indexer",
                label="Coarse Indexer",
                description="Build sparse lexical index for coarse retrieval.",
                status="ready",
            ),
            SubcomponentDefinition(
                id="embedding_indexer",
                label="Embedding Indexer",
                description="Build dense vector index for fine retrieval.",
                status="ready",
            ),
            SubcomponentDefinition(
                id="repo_graph_indexer",
                label="Repository Graph Indexer",
                description="Build a code graph index for graph retrieval and graph expansion.",
                status="ready",
            ),
        ),
    ),
    ComponentGroupDefinition(
        id="query",
        label="Query",
        description="Optional query preprocessing and expansion steps.",
        multi_select=True,
        required=False,
        default=["query_cleaner", "query_rewriter", "multi_query_generator"],
        subcomponents=(
            SubcomponentDefinition(
                id="query_cleaner",
                label="Query Cleaner",
                description="Normalize and clean user query text.",
                status="ready",
            ),
            SubcomponentDefinition(
                id="query_rewriter",
                label="Query Rewriter",
                description="Rewrite query for stronger retrieval recall.",
                status="ready",
            ),
            SubcomponentDefinition(
                id="multi_query_generator",
                label="Multi Query Generator",
                description="Generate query variants for query expansion.",
                status="ready",
            ),
        ),
    ),
    ComponentGroupDefinition(
        id="retrieval",
        label="Retriever",
        description="Pick one primary retriever (coarse / fine / hybrid). Optionally also pick external_retriever to fall back to web search when the primary returns too few chunks.",
        multi_select=True,
        required=True,
        default=["coarse_retriever"],
        subcomponents=(
            SubcomponentDefinition(
                id="coarse_retriever",
                label="Coarse Retriever",
                description="Sparse lexical retrieval over coarse index.",
                status="ready",
            ),
            SubcomponentDefinition(
                id="fine_retriever",
                label="Fine Retriever",
                description="Dense embedding retrieval over vector index.",
                status="ready",
            ),
            SubcomponentDefinition(
                id="hybrid_retriever",
                label="Hybrid Retriever",
                description="Dense + sparse retrieval with candidate merge.",
                status="ready",
            ),
            SubcomponentDefinition(
                id="external_retriever",
                label="External Retriever",
                description="Retrieve from external search APIs.",
                status="experimental",
            ),
            SubcomponentDefinition(
                id="memory_retriever",
                label="Memory Retriever",
                description="Retrieve from conversation or long-term memory.",
                status="not_implemented",
            ),
            SubcomponentDefinition(
                id="graph_retriever",
                label="Graph Retriever",
                description="Retrieve by traversing graph relationships.",
                status="ready",
            ),
        ),
    ),
    ComponentGroupDefinition(
        id="ranking",
        label="Ranking",
        description="Optional reranking, fusion, and retrieval-result expansion.",
        multi_select=True,
        required=False,
        default=["rank_fusion", "embedding_ranker"],
        subcomponents=(
            SubcomponentDefinition(
                id="rank_fusion",
                label="Rank Fusion",
                description="Fuse dense and sparse result lists.",
                status="ready",
            ),
            SubcomponentDefinition(
                id="graph_expander",
                label="Graph Expander",
                description="Expand retrieved chunks through related code graph evidence.",
                status="ready",
            ),
            SubcomponentDefinition(
                id="embedding_ranker",
                label="Embedding Ranker",
                description="Rerank candidates using embedding similarity.",
                status="ready",
            ),
            SubcomponentDefinition(
                id="cross_encoder_ranker",
                label="Cross Encoder Ranker",
                description="Rerank with a cross-encoder model.",
                status="ready",
            ),
            SubcomponentDefinition(
                id="colbert_ranker",
                label="ColBERT Ranker",
                description="Late interaction ColBERT-style ranker.",
                status="not_implemented",
            ),
        ),
    ),
    ComponentGroupDefinition(
        id="context",
        label="Context",
        description="Build and shape the context passed into generation.",
        multi_select=True,
        required=False,
        default=["context_builder"],
        subcomponents=(
            SubcomponentDefinition(
                id="context_merger",
                label="Context Merger",
                description="Merge duplicate or overlapping retrieved chunks before context building.",
                status="ready",
            ),
            SubcomponentDefinition(
                id="context_builder",
                label="Context Builder",
                description="Convert retrieved or ranked chunks into prompt context with source metadata.",
                status="ready",
            ),
            SubcomponentDefinition(
                id="context_truncator",
                label="Context Truncator",
                description="Trim generated context to the configured token budget.",
                status="ready",
            ),
        ),
    ),
    ComponentGroupDefinition(
        id="generation",
        label="Generation",
        description="Prompting, model generation, and output parsing.",
        multi_select=True,
        required=True,
        default=["prompt_builder", "llm_generator", "output_parser"],
        subcomponents=(
            SubcomponentDefinition(
                id="prompt_builder",
                label="Prompt Builder",
                description="Prepare the prompt template and parser binding.",
                status="ready",
            ),
            SubcomponentDefinition(
                id="llm_generator",
                label="LLM Generator",
                description="Generate answer from prompt and context.",
                status="ready",
            ),
            SubcomponentDefinition(
                id="output_parser",
                label="Output Parser",
                description="Parse response into typed structured output.",
                status="ready",
            ),
            SubcomponentDefinition(
                id="streaming_generator",
                label="Streaming Generator",
                description="Token streaming generation. Pick instead of llm_generator; downstream consumers see the joined answer.",
                status="ready",
            ),
        ),
    ),
    ComponentGroupDefinition(
        id="postprocessing",
        label="Postprocessing",
        description="Optional critique and refinement after generation.",
        multi_select=True,
        required=False,
        default=["self_critic", "refiner"],
        subcomponents=(
            SubcomponentDefinition(
                id="self_critic",
                label="Self Critic",
                description="Generate critique from answer + context.",
                status="ready",
            ),
            SubcomponentDefinition(
                id="refiner",
                label="Refiner",
                description="Refine answer based on critique.",
                status="ready",
            ),
        ),
    ),
    ComponentGroupDefinition(
        id="evaluation",
        label="Evaluation",
        description="Optional scoring steps after the answer is produced.",
        multi_select=True,
        required=False,
        default=[],
        subcomponents=(
            SubcomponentDefinition(
                id="ragas_evaluator",
                label="RAGAS Evaluator",
                description="Run optional LLM-judged RAGAS metrics when evaluation data is available.",
                status="experimental",
            ),
            SubcomponentDefinition(
                id="evaluator",
                label="Base Evaluator",
                description="Abstract evaluator hook for custom metric implementations.",
                status="not_implemented",
            ),
        ),
    ),
)

GROUP_ORDER = [group.id for group in CATALOG]

def get_group(group_id: str) -> ComponentGroupDefinition | None:
    for group in CATALOG:
        if group.id == group_id:
            return group
    return None

def status_for(component_id: str) -> ComponentStatus:
    for group in CATALOG:
        for subcomponent in group.subcomponents:
            if subcomponent.id == component_id:
                return subcomponent.status
    return "experimental"

def is_implemented(component_id: str) -> bool:
    return status_for(component_id) != "not_implemented"

def default_selection() -> dict[str, list[str]]:
    return {group.id: list(group.default) for group in CATALOG}

def as_json_payload() -> dict[str, object]:
    groups = []
    for group in CATALOG:
        groups.append(
            {
                "id": group.id,
                "label": group.label,
                "description": group.description,
                "multi_select": group.multi_select,
                "required": group.required,
                "default": list(group.default),
                "subcomponents": [
                    {
                        "id": subcomponent.id,
                        "label": subcomponent.label,
                        "description": subcomponent.description,
                        "status": subcomponent.status,
                    }
                    for subcomponent in group.subcomponents
                ],
            }
        )

    return {
        "groups": groups,
        "defaults": default_selection(),
    }
