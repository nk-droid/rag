from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterator
from uuid import uuid4

from api.catalog import CATALOG, GROUP_ORDER, is_implemented
from api.loader_service import LoaderService
from api.schemas import (
    PipelineInitializeResponse,
    PipelinePlan,
    PipelineRequest,
    PipelineSelection,
    PipelineStep,
    SourceRecord,
)
from pipeline.component_factories import COMPONENT_FACTORIES
from pipeline.config import load_config
from pipeline.orchestrator import RAGOrchestrator

class PipelineValidationError(ValueError):
    pass

STREAMING_GENERATOR_KEY = "streaming_generator"
EXTERNAL_RETRIEVER_KEY = "external_retriever"
EXTERNAL_FALLBACK_THRESHOLD = 3
INITIALIZATION_TTL_SEC = 60 * 60 * 6

@dataclass(slots=True)
class InitializedPipelineRecord:
    initialization_id: str
    fingerprint: str
    source_ids: tuple[str, ...]
    created_at: float
    document_count: int
    source_count: int

_INITIALIZED_PIPELINES: dict[str, InitializedPipelineRecord] = {}

OUTPUT_TYPE_BY_COMPONENT: dict[str, str] = {
    "text_loader": "documents",
    "markdown_loader": "documents",
    "document_loader": "documents",
    "directory_loader": "documents",
    "code_loader": "documents",
    "repo_loader": "documents",
    "source_normalizer": "documents",
    "late_chunker": "chunks",
    "semantic_chunker": "chunks",
    "recursive_chunker": "chunks",
    "code_aware_chunker": "chunks",
    "embedding_indexer": "index",
    "coarse_indexer": "index",
    "repo_graph_indexer": "index",
    "graph_indexer": "index",
    "query_cleaner": "query",
    "query_rewriter": "query",
    "multi_query_generator": "query",
    "coarse_retriever": "retrieved_documents",
    "fine_retriever": "retrieved_documents",
    "hybrid_retriever": "retrieved_documents",
    "memory_retriever": "retrieved_documents",
    "graph_retriever": "retrieved_documents",
    "external_retriever": "retrieved_documents",
    "graph_expander": "retrieved_documents",
    "rank_fusion": "retrieved_documents",
    "embedding_ranker": "reranked_documents",
    "colbert_ranker": "reranked_documents",
    "cross_encoder_ranker": "reranked_documents",
    "context_builder": "context",
    "context_merger": "context",
    "context_truncator": "context",
    "prompt_builder": "prompt",
    "generator": "generated_answer",
    "llm_generator": "generated_answer",
    "streaming_generator": "generated_answer",
    "refiner": "generated_answer",
    "self_critic": "critique",
    "output_parser": "parsed_output",
    "evaluator": "metrics",
    "ragas_evaluator": "metrics",
}

def _prompt_text(prompt: Any) -> str | None:
    if prompt is None:
        return None
    if isinstance(prompt, str):
        return prompt
    for attr in ("template", "text"):
        value = getattr(prompt, attr, None)
        if isinstance(value, str):
            return value
    return str(prompt)

def _short_text(value: Any, limit: int = 6000) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n[truncated {len(text) - limit} chars]"

def _serialize_payload(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _serialize_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize_payload(item) for item in list(value)[:50]]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _serialize_payload(model_dump())
    dict_method = getattr(value, "dict", None)
    if callable(dict_method):
        return _serialize_payload(dict_method())
    if hasattr(value, "__dict__"):
        return _serialize_payload(
            {
                key: item
                for key, item in vars(value).items()
                if not str(key).startswith("_")
            }
        )
    return _short_text(value)

def _pipeline_identity(plan: PipelinePlan) -> tuple[str, str]:
    selected = plan.selected_components
    retrieval = selected.get("retrieval", [])
    ranking = selected.get("ranking", [])
    generation = selected.get("generation", [])
    postprocessing = selected.get("postprocessing", [])

    if "hybrid_retriever" in retrieval and any("ranker" in item for item in ranking):
        name = "Hybrid RAG + Reranker"
    elif "hybrid_retriever" in retrieval:
        name = "Hybrid RAG"
    elif "fine_retriever" in retrieval:
        name = "Vector RAG"
    elif "external_retriever" in retrieval and len(retrieval) == 1:
        name = "External Search RAG"
    elif "coarse_retriever" in retrieval:
        name = "Baseline BM25 RAG"
    else:
        name = "Custom RAG"

    if STREAMING_GENERATOR_KEY in generation:
        name = f"{name} Streaming"
    if postprocessing:
        name = f"{name} + Refinement"

    pipeline_id = "-".join(
        item
        for group in ("retrieval", "ranking", "generation", "postprocessing")
        for item in selected.get(group, [])
    ) or "custom-rag"
    return pipeline_id.replace("_", "-"), name

def _step_output(output_type: str, state: dict[str, Any]) -> dict[str, object]:
    if output_type == "query":
        return {
            "query": str(state.get("query", "")),
            "retrieval_queries": _serialize_payload(state.get("retrieval_queries", [])),
        }

    if output_type == "retrieved_documents":
        return {
            "documents": _serialize_chunks(state.get("retrieved", [])),
            "retrieval_queries": _serialize_payload(state.get("retrieval_queries", [])),
        }

    if output_type == "reranked_documents":
        return {
            "documents": _serialize_chunks(state.get("ranked", []) or state.get("retrieved", [])),
        }

    if output_type == "context":
        return {
            "context": _short_text(state.get("context", "")),
        }

    if output_type == "prompt":
        return {
            "prompt": _short_text(_prompt_text(state.get("prompt"))),
        }

    if output_type == "generated_answer":
        return {
            "answer": _answer_from_state(state),
            "generator": _serialize_payload(state.get("generator")),
            "stream": _serialize_payload(state.get("stream", [])),
        }

    if output_type == "critique":
        return {
            "critique": _short_text(
                state.get("critique")
                or state.get("self_critique")
                or state.get("feedback")
                or ""
            ),
        }

    if output_type == "parsed_output":
        return {
            "parsed_output": _serialize_payload(state.get("parsed_output")),
        }

    if output_type == "metrics":
        return {
            "metrics": _serialize_payload(
                state.get("evaluation")
                or state.get("metrics")
                or state.get("scores")
                or {}
            ),
        }

    if output_type == "chunks":
        return {
            "chunks": _serialize_payload(state.get("chunks", [])),
        }

    if output_type == "documents":
        documents = state.get("documents", [])
        return {
            "count": len(documents) if isinstance(documents, list) else None,
        }

    return {
        "state_keys": sorted(
            key for key in state.keys() if isinstance(key, str) and not key.startswith("_")
        )
    }

def _step_summary(output_type: str, output: dict[str, object]) -> str | None:
    if output_type in {"retrieved_documents", "reranked_documents"}:
        documents = output.get("documents")
        if isinstance(documents, list):
            return f"{len(documents)} chunks"
    if output_type == "prompt":
        prompt = output.get("prompt")
        if isinstance(prompt, str):
            return f"{len(prompt.split())} prompt words"
    if output_type == "generated_answer":
        answer = output.get("answer")
        if isinstance(answer, str):
            return f"{len(answer.split())} answer words"
    if output_type == "metrics":
        metrics = output.get("metrics")
        if isinstance(metrics, dict):
            return f"{len(metrics)} metrics"
    return None

def _build_run_steps(
    state: dict[str, Any],
    *,
    phases: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Shape per-step timings into the UI's backend-driven step contract."""
    timings = [
        timing
        for timing in state.get("step_timings", [])
        if isinstance(timing, dict)
        and (timing.get("phase") in phases if phases is not None else timing.get("phase") != "init")
    ]
    steps: list[dict[str, Any]] = []
    for index, timing in enumerate(timings):
        component = str(timing.get("component") or "")
        output_type = OUTPUT_TYPE_BY_COMPONENT.get(component, "raw_json")
        output = _step_output(output_type, state)
        steps.append(
            {
                "step_id": f"{index}:{timing.get('step_name')}",
                "step_name": str(timing.get("step_name") or component),
                "component": component,
                "status": "completed",
                "latency_ms": round(float(timing.get("latency_ms") or 0.0), 1),
                "output_type": output_type,
                "output": output,
                "summary": _step_summary(output_type, output),
            }
        )
    return steps

def _build_run_result(
    request: PipelineRequest,
    plan: PipelinePlan,
    state: dict[str, Any],
) -> dict[str, object]:
    pipeline_id, pipeline_name = _pipeline_identity(plan)
    steps = _build_run_steps(state)
    answer = _answer_from_state(state)
    retrieved = _serialize_chunks(state.get("retrieved", []))
    ranked = _serialize_chunks(state.get("ranked", []))
    run_id = str(state.get("intermediate_run_id") or f"run_{uuid4().hex[:12]}")
    prompt = _prompt_text(state.get("prompt"))
    evaluation = _serialize_payload(
        state.get("evaluation")
        or state.get("metrics")
        or state.get("scores")
        or {}
    )

    return {
        "run_id": run_id,
        "pipeline_id": pipeline_id,
        "pipeline_name": pipeline_name,
        "status": "completed",
        "query": request.query,
        "answer": answer,
        "retrieved": retrieved,
        "ranked": ranked,
        "retrieval_queries": _serialize_payload(state.get("retrieval_queries", [])),
        "prompt": _short_text(prompt),
        "evaluation": evaluation,
        "steps": steps,
        "trace": [
            {
                "step_id": step["step_id"],
                "step_name": step["step_name"],
                "component": step["component"],
                "latency_ms": step.get("latency_ms"),
                "status": step.get("status"),
                "output_type": step.get("output_type"),
            }
            for step in steps
        ],
        "init_skipped": bool(state.get("init_skipped", False)),
        "intermediate_run_id": state.get("intermediate_run_id"),
        "intermediate_path": state.get("intermediate_path"),
    }

def plan_has_streaming_generator(plan: PipelinePlan) -> bool:
    for step in plan.pipeline:
        component = step.component
        if isinstance(component, str) and component == STREAMING_GENERATOR_KEY:
            return True
        if isinstance(component, list) and STREAMING_GENERATOR_KEY in component:
            return True
    return False

@dataclass(slots=True)
class PlannedPipeline:
    plan: PipelinePlan
    warnings: list[str]

def _initialization_fingerprint(
    request: PipelineRequest,
    sources: list[SourceRecord],
    plan: PipelinePlan,
) -> str:
    payload = {
        "source_ids": [source.id for source in sources],
        "sources": [
            {
                "id": source.id,
                "path": source.path,
                "loader": source.loader,
                "source_type": source.source_type,
                "repo_url": source.repo_url,
                "branch": source.branch,
                "commit_sha": source.commit_sha,
            }
            for source in sources
        ],
        "selection": plan.selected_components,
        "init_pipeline": [_dump_model(step) for step in plan.init_pipeline],
        "pipeline": [_dump_model(step) for step in plan.pipeline],
        "top_k": request.top_k,
        "template_name": request.template_name,
        "parser_name": request.parser_name,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def _purge_expired_initializations(now: float | None = None) -> None:
    current = time.time() if now is None else now
    expired = [
        initialization_id
        for initialization_id, record in _INITIALIZED_PIPELINES.items()
        if current - record.created_at > INITIALIZATION_TTL_SEC
    ]
    for initialization_id in expired:
        _INITIALIZED_PIPELINES.pop(initialization_id, None)

def _record_initialization(
    request: PipelineRequest,
    sources: list[SourceRecord],
    plan: PipelinePlan,
    *,
    document_count: int,
) -> str:
    _purge_expired_initializations()
    initialization_id = f"init_{uuid4().hex[:16]}"
    record = InitializedPipelineRecord(
        initialization_id=initialization_id,
        fingerprint=_initialization_fingerprint(request, sources, plan),
        source_ids=tuple(source.id for source in sources),
        created_at=time.time(),
        document_count=document_count,
        source_count=len(sources),
    )
    _INITIALIZED_PIPELINES[initialization_id] = record
    return initialization_id

def _require_initialized_pipeline(
    request: PipelineRequest,
    sources: list[SourceRecord],
    plan: PipelinePlan,
) -> None:
    _purge_expired_initializations()
    if not request.initialization_id:
        raise PipelineValidationError(
            "Pipeline has not been initialized. Initialize the pipeline before running a query."
        )

    record = _INITIALIZED_PIPELINES.get(request.initialization_id)
    if record is None:
        raise PipelineValidationError(
            "Pipeline initialization has expired or was not found. Re-initialize before running a query."
        )

    expected = _initialization_fingerprint(request, sources, plan)
    if record.fingerprint != expected:
        raise PipelineValidationError(
            "Pipeline configuration changed after initialization. Re-initialize before running a query."
        )

def _selection_dict(selection: PipelineSelection) -> dict[str, list[str]]:
    return {
        "chunking": list(selection.chunking),
        "indexing": list(selection.indexing),
        "query": list(selection.query),
        "retrieval": list(selection.retrieval),
        "ranking": list(selection.ranking),
        "context": list(selection.context),
        "generation": list(selection.generation),
        "postprocessing": list(selection.postprocessing),
        "evaluation": list(selection.evaluation),
    }

def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        key = value.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered

def _ensure_group_rules(selection: dict[str, list[str]]) -> None:
    for group in CATALOG:
        chosen = _dedupe(selection.get(group.id, []))
        valid_ids = {item.id for item in group.subcomponents}

        invalid = [item for item in chosen if item not in valid_ids]
        if invalid:
            raise PipelineValidationError(
                f"Invalid subcomponent(s) for '{group.id}': {', '.join(invalid)}"
            )

        if group.required and not chosen:
            raise PipelineValidationError(f"Group '{group.id}' requires at least one subcomponent.")

        if not group.multi_select and len(chosen) > 1:
            raise PipelineValidationError(
                f"Group '{group.id}' supports exactly one subcomponent. Received: {chosen}"
            )

        for component_id in chosen:
            if not is_implemented(component_id):
                raise PipelineValidationError(
                    f"Subcomponent '{component_id}' is currently not implemented."
                )

def _required_indexers_for_retriever(retriever: str) -> list[str]:
    if retriever == "hybrid_retriever":
        return ["embedding_indexer", "coarse_indexer"]

    if retriever == "fine_retriever":
        return ["embedding_indexer"]

    if retriever == "coarse_retriever":
        return ["coarse_indexer"]

    if retriever == "graph_retriever":
        return ["repo_graph_indexer"]

    return []

def _make_step(name: str, component: str, **kwargs: Any) -> PipelineStep:
    data: dict[str, Any] = {"name": name, "component": component}
    data.update(kwargs)
    validator = getattr(PipelineStep, "model_validate", None)
    if callable(validator):
        return validator(data)
    parser = getattr(PipelineStep, "parse_obj")
    return parser(data)

def build_pipeline_plan(request: PipelineRequest) -> PlannedPipeline:
    selection = _selection_dict(request.selection)
    selection = {key: _dedupe(value) for key, value in selection.items()}
    _ensure_group_rules(selection)

    warnings: list[str] = []

    query_steps = [_make_step(name=item, component=item) for item in selection.get("query", [])]

    retrieval_components = selection.get("retrieval", [])
    external_selected = EXTERNAL_RETRIEVER_KEY in retrieval_components
    primary_candidates = [r for r in retrieval_components if r != EXTERNAL_RETRIEVER_KEY]

    if len(primary_candidates) > 1:
        raise PipelineValidationError(
            "At most one primary retriever can be selected. "
            f"external_retriever can be added as a fallback alongside it. Received: {primary_candidates}"
        )
    if not primary_candidates and not external_selected:
        raise PipelineValidationError("At least one retriever must be selected.")

    primary_retriever = primary_candidates[0] if primary_candidates else None
    retriever = primary_retriever or EXTERNAL_RETRIEVER_KEY

    retrieval_options: dict[str, Any] = {}
    if request.top_k is not None:
        retrieval_options["top_k"] = int(request.top_k)

    pipeline_steps: list[PipelineStep] = []
    pipeline_steps.extend(query_steps)

    if primary_retriever is not None:
        pipeline_steps.append(
            _make_step(name="retrieve_local", component=primary_retriever, **retrieval_options)
        )
        if external_selected:
            pipeline_steps.append(
                _make_step(
                    name="retrieve_external",
                    component=EXTERNAL_RETRIEVER_KEY,
                    if_under=EXTERNAL_FALLBACK_THRESHOLD,
                    merge_with_existing=True,
                )
            )
            warnings.append(
                f"external_retriever runs only when {primary_retriever} returns fewer than "
                f"{EXTERNAL_FALLBACK_THRESHOLD} chunks; merged + deduped against local results."
            )
    else:
        pipeline_steps.append(
            _make_step(name="retrieve_external", component=EXTERNAL_RETRIEVER_KEY, **retrieval_options)
        )

    ranking_steps = selection.get("ranking", [])
    for ranking_component in ranking_steps:
        if ranking_component == "rank_fusion" and retriever != "hybrid_retriever":
            warnings.append(
                "rank_fusion is most effective with hybrid_retriever; keeping it as selected."
            )
        pipeline_steps.append(_make_step(name=ranking_component, component=ranking_component))

    for context_component in selection.get("context", []):
        pipeline_steps.append(_make_step(name=context_component, component=context_component))

    generation_steps = selection.get("generation", [])
    GENERATOR_COMPONENTS = {"llm_generator", "streaming_generator"}
    has_generator = any(step in GENERATOR_COMPONENTS for step in generation_steps)

    if has_generator and "prompt_builder" not in generation_steps:
        generation_steps = ["prompt_builder", *generation_steps]
        warnings.append("Auto-inserted prompt_builder because the chosen generator requires it.")

    if "prompt_builder" not in generation_steps:
        generation_steps.insert(0, "prompt_builder")
        warnings.append("Auto-inserted prompt_builder to keep generation pipeline valid.")

    if not has_generator:
        generation_steps.append("llm_generator")
        warnings.append("Auto-inserted llm_generator because generation output is required.")

    for generation_component in generation_steps:
        if generation_component == "prompt_builder":
            pipeline_steps.append(
                _make_step(
                    name="prompt_builder",
                    component="prompt_builder",
                    template_name=request.template_name or "summarize.yaml",
                    parser=request.parser_name or "Answer",
                )
            )
            continue

        if generation_component == "output_parser":
            pipeline_steps.append(
                _make_step(
                    name="output_parser",
                    component="output_parser",
                    parser=request.parser_name or "Answer",
                )
            )
            continue

        pipeline_steps.append(
            _make_step(name=generation_component, component=generation_component)
        )

    for postprocessing_component in selection.get("postprocessing", []):
        pipeline_steps.append(
            _make_step(name=postprocessing_component, component=postprocessing_component)
        )

    for evaluation_component in selection.get("evaluation", []):
        pipeline_steps.append(
            _make_step(name=evaluation_component, component=evaluation_component)
        )

    chunker = selection.get("chunking", [])[0]
    retriever_required_indexers = _required_indexers_for_retriever(retriever)
    selected_indexers = list(selection.get("indexing", []))

    if not selected_indexers and retriever_required_indexers:
        selected_indexers = list(retriever_required_indexers)
        warnings.append(
            "Auto-selected indexers based on retriever compatibility."
        )

    missing_indexers = [
        indexer for indexer in retriever_required_indexers
        if indexer not in selected_indexers
    ]
    if missing_indexers:
        selected_indexers.extend(missing_indexers)
        warnings.append(
            "Added missing indexers required by the selected retriever: "
            + ", ".join(missing_indexers)
        )

    indexers = selected_indexers

    init_steps: list[PipelineStep] = [_make_step(name="chunk", component=chunker)]
    if indexers:
        if len(indexers) == 1:
            init_steps.append(_make_step(name="index", component=indexers[0]))
        else:
            init_steps.append(_make_step(name="index", component=indexers))

    selected_components = {key: list(selection.get(key, [])) for key in GROUP_ORDER}

    plan = PipelinePlan(
        init_pipeline=init_steps,
        pipeline=pipeline_steps,
        indexers=indexers,
        selected_components=selected_components,
    )
    return PlannedPipeline(plan=plan, warnings=warnings)

def _runtime_config_for_plan(plan: PipelinePlan) -> dict[str, Any]:
    config = load_config([
        "configs/runtime/api.yaml",
        "configs/env/dev.yaml",
    ])
    resolved = deepcopy(config)

    resolved["init_pipeline"] = {
        "steps": [
            _dump_model(step)
            for step in plan.init_pipeline
        ]
    }
    resolved["pipeline"] = {
        "steps": [
            _dump_model(step)
            for step in plan.pipeline
        ]
    }
    return resolved

def _serialize_chunks(chunks: Any) -> list[dict[str, Any]]:
    if not isinstance(chunks, list):
        return []

    serialized: list[dict[str, Any]] = []
    for item in chunks:
        if hasattr(item, "id") and hasattr(item, "text"):
            serialized.append(
                {
                    "id": str(getattr(item, "id", "")),
                    "text": str(getattr(item, "text", "")),
                    "score": float(getattr(item, "score", 0.0) or 0.0),
                    "metadata": dict(getattr(item, "metadata", {}) or {}),
                }
            )
            continue

        if isinstance(item, dict):
            serialized.append(
                {
                    "id": str(item.get("id", "")),
                    "text": str(item.get("text") or item.get("content") or ""),
                    "score": float(item.get("score", 0.0) or 0.0),
                    "metadata": dict(item.get("metadata", {}) or {}),
                }
            )

    return serialized

def _answer_from_state(state: dict[str, Any]) -> str:
    parsed = state.get("parsed_output")
    if parsed is not None:
        answer = getattr(parsed, "answer", None)
        if answer is not None:
            return str(answer)

    raw = state.get("answer")
    if raw is None:
        return ""

    if isinstance(raw, str):
        return raw

    content = getattr(raw, "content", None)
    if content is not None:
        return str(content)

    return str(raw)

def run_pipeline(request: PipelineRequest, sources: list[SourceRecord], loader_service: LoaderService) -> tuple[PipelinePlan, list[str], dict[str, object]]:
    if not sources:
        raise PipelineValidationError("At least one source must be selected before running the pipeline.")

    planned = build_pipeline_plan(request)
    config = _runtime_config_for_plan(planned.plan)
    if request.save_intermediate is not None:
        config.setdefault("intermediate", {})["enabled"] = bool(request.save_intermediate)

    documents = loader_service.load_sources(sources)
    if not documents:
        raise PipelineValidationError(
            "No documents were loaded from selected sources. Check source type and file content."
        )

    state: dict[str, Any] = {
        "query": request.query,
        "documents": documents,
        "sources": [source.path for source in sources],
    }
    if request.top_k is not None:
        state["top_k"] = int(request.top_k)
    if request.intermediate_run_id:
        state["intermediate_run_id"] = request.intermediate_run_id

    orchestrator = RAGOrchestrator(config)
    if not request.skip_initialization:
        state = orchestrator.initialize(state)
    else:
        _require_initialized_pipeline(request, sources, planned.plan)
    state = orchestrator.run(state)

    result = _build_run_result(request, planned.plan, state)

    return planned.plan, planned.warnings, result

def initialize_pipeline(
    request: PipelineRequest,
    sources: list[SourceRecord],
    loader_service: LoaderService,
) -> PipelineInitializeResponse:
    if not sources:
        raise PipelineValidationError("At least one source must be selected before initializing the pipeline.")

    planned = build_pipeline_plan(request)
    config = _runtime_config_for_plan(planned.plan)
    documents = loader_service.load_sources(sources)
    if not documents:
        raise PipelineValidationError(
            "No documents were loaded from selected sources. Check source type and file content."
        )

    state: dict[str, Any] = {
        "query": request.query,
        "documents": documents,
        "sources": [source.path for source in sources],
    }
    if request.top_k is not None:
        state["top_k"] = int(request.top_k)

    orchestrator = RAGOrchestrator(config)
    state = orchestrator.initialize(state)
    init_steps = _build_run_steps(state, phases={"init"})
    initialization_id = _record_initialization(
        request,
        sources,
        planned.plan,
        document_count=len(documents),
    )

    return PipelineInitializeResponse(
        initialization_id=initialization_id,
        plan=planned.plan,
        warnings=planned.warnings,
        status="completed",
        init_skipped=bool(state.get("init_skipped", False)),
        source_count=len(sources),
        document_count=len(documents),
        steps=init_steps,
    )

def stream_pipeline_run(
    request: PipelineRequest,
    sources: list[SourceRecord],
    loader_service: LoaderService,
) -> Iterator[tuple[str, dict[str, Any]]]:
    if not sources:
        raise PipelineValidationError("At least one source must be selected before running the pipeline.")

    planned = build_pipeline_plan(request)
    if not plan_has_streaming_generator(planned.plan):
        raise PipelineValidationError(
            "Streaming endpoint requires streaming_generator in the generation selection."
        )

    raw_steps = [_dump_model(step) for step in planned.plan.pipeline]
    stream_idx = next(
        (
            idx
            for idx, step in enumerate(raw_steps)
            if step.get("component") == STREAMING_GENERATOR_KEY
        ),
        None,
    )
    if stream_idx is None:
        raise PipelineValidationError(
            "Streaming endpoint cannot split a step that bundles streaming_generator with other components."
        )

    config = _runtime_config_for_plan(planned.plan)
    if request.save_intermediate is not None:
        config.setdefault("intermediate", {})["enabled"] = bool(request.save_intermediate)
    documents = loader_service.load_sources(sources)
    if not documents:
        raise PipelineValidationError(
            "No documents were loaded from selected sources. Check source type and file content."
        )

    state: dict[str, Any] = {
        "query": request.query,
        "documents": documents,
        "sources": [source.path for source in sources],
    }
    if request.top_k is not None:
        state["top_k"] = int(request.top_k)
    if request.intermediate_run_id:
        state["intermediate_run_id"] = request.intermediate_run_id

    orchestrator = RAGOrchestrator(config)
    if not request.skip_initialization:
        state = orchestrator.initialize(state)
    else:
        _require_initialized_pipeline(request, sources, planned.plan)

    pre_steps = raw_steps[:stream_idx]
    post_steps = raw_steps[stream_idx + 1:]

    yield (
        "plan",
        {
            "plan": _dump_model(planned.plan),
            "warnings": planned.warnings,
        },
    )

    state = orchestrator.execute_steps(pre_steps, state)
    for step in _build_run_steps(state):
        yield ("step", {"step": step})

    prompt = state.get("prompt")
    if prompt is None:
        raise PipelineValidationError(
            "Pipeline missing prompt_builder before streaming_generator."
        )

    context = state.get("context", "")
    if not context and state.get("ranked"):
        context = "\n\n".join(
            chunk.text if hasattr(chunk, "text") else str(chunk.get("content", ""))
            for chunk in state["ranked"]
        )
    inputs = {"query": state.get("query", ""), "context": context}

    generator = COMPONENT_FACTORIES[STREAMING_GENERATOR_KEY](config)
    before_stream_snapshot = orchestrator.snapshot_intermediate_state(state)

    pieces: list[str] = []
    stream_started = time.perf_counter()
    for piece in generator.stream(prompt, inputs):
        text = str(piece)
        pieces.append(text)
        yield ("token", {"piece": text})
    stream_latency_ms = (time.perf_counter() - stream_started) * 1000.0

    state["answer"] = "".join(pieces)
    state["stream"] = pieces
    state["generator"] = generator.__class__.__name__
    state.setdefault("step_timings", []).append(
        {
            "phase": "stream",
            "step_name": "streaming_generator",
            "component": STREAMING_GENERATOR_KEY,
            "latency_ms": stream_latency_ms,
        }
    )
    state = orchestrator.record_intermediate_step(
        state,
        phase="stream",
        step_name="streaming_generator",
        component_name=STREAMING_GENERATOR_KEY,
        before_snapshot=before_stream_snapshot,
    )
    yield ("step", {"step": _build_run_steps(state)[-1]})

    state = orchestrator.execute_steps(post_steps, state)
    orchestrator.finalize_intermediate(state)

    result = _build_run_result(request, planned.plan, state)
    yield ("done", {"result": result, "steps": result["steps"]})

def source_paths_exist(sources: list[SourceRecord]) -> list[str]:
    missing: list[str] = []
    for source in sources:
        if not Path(source.path).exists():
            missing.append(source.id)
    return missing

def _dump_model(model: Any) -> dict[str, Any]:
    dumper = getattr(model, "model_dump", None)
    if callable(dumper):
        return dumper(exclude_none=True)
    serializer = getattr(model, "dict")
    return serializer(exclude_none=True)
