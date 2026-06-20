import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from api.loader_service import LoaderService
from api.pipeline_service import (
    PipelineValidationError,
    build_pipeline_plan,
    initialize_pipeline,
    plan_has_streaming_generator,
    run_pipeline,
    source_paths_exist,
    stream_pipeline_run,
)
from api.schemas import (
    PipelineInitializeResponse,
    PipelinePreviewResponse,
    PipelineRequest,
    PipelineRunResponse,
    PipelineTemplatesResponse,
)
from api.source_store import SourceStore
from api.template_service import list_pipeline_templates

router = APIRouter(prefix="/api/pipelines", tags=["pipelines"])

def get_store() -> SourceStore:
    return SourceStore()

def get_loader_service() -> LoaderService:
    return LoaderService()

@router.get("/templates", response_model=PipelineTemplatesResponse)
def get_pipeline_templates() -> PipelineTemplatesResponse:
    return PipelineTemplatesResponse(templates=list_pipeline_templates())

@router.post("/preview", response_model=PipelinePreviewResponse)
def preview_pipeline(payload: PipelineRequest) -> PipelinePreviewResponse:
    try:
        planned = build_pipeline_plan(payload)
    except PipelineValidationError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return PipelinePreviewResponse(plan=planned.plan, warnings=planned.warnings)

@router.post("/initialize", response_model=PipelineInitializeResponse)
def initialize_pipeline_endpoint(
    payload: PipelineRequest,
    store: SourceStore = Depends(get_store),
    loader_service: LoaderService = Depends(get_loader_service),
) -> PipelineInitializeResponse:
    source_ids = [item for item in dict.fromkeys(payload.source_ids) if item]

    sources = store.get_sources_by_ids(source_ids)
    if len(sources) != len(source_ids):
        raise HTTPException(status_code=400, detail="One or more source_ids are invalid.")

    missing = source_paths_exist(sources)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Source path no longer exists for source_ids: {', '.join(missing)}",
        )

    try:
        return initialize_pipeline(payload, sources, loader_service)
    except PipelineValidationError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Pipeline initialization failed: {error}") from error

@router.post("/run", response_model=PipelineRunResponse)
def execute_pipeline(
    payload: PipelineRequest,
    store: SourceStore = Depends(get_store),
    loader_service: LoaderService = Depends(get_loader_service),
) -> PipelineRunResponse:
    source_ids = [item for item in dict.fromkeys(payload.source_ids) if item]

    sources = store.get_sources_by_ids(source_ids)
    if len(sources) != len(source_ids):
        raise HTTPException(status_code=400, detail="One or more source_ids are invalid.")

    missing = source_paths_exist(sources)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Source path no longer exists for source_ids: {', '.join(missing)}",
        )

    try:
        plan, warnings, result = run_pipeline(payload, sources, loader_service)
    except PipelineValidationError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Pipeline run failed: {error}") from error

    return PipelineRunResponse(
        plan=plan,
        result=result,
        warnings=warnings,
        run_id=str(result.get("run_id") or ""),
        pipeline_id=str(result.get("pipeline_id") or ""),
        pipeline_name=str(result.get("pipeline_name") or ""),
        status="completed",
        query=str(result.get("query") or payload.query),
        steps=result.get("steps", []),
    )

def _sse_event(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, default=str)}\n\n"

@router.post("/stream")
def execute_pipeline_stream(
    payload: PipelineRequest,
    store: SourceStore = Depends(get_store),
    loader_service: LoaderService = Depends(get_loader_service),
):
    source_ids = [item for item in dict.fromkeys(payload.source_ids) if item]

    sources = store.get_sources_by_ids(source_ids)
    if len(sources) != len(source_ids):
        raise HTTPException(status_code=400, detail="One or more source_ids are invalid.")

    missing = source_paths_exist(sources)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Source path no longer exists for source_ids: {', '.join(missing)}",
        )

    try:
        planned = build_pipeline_plan(payload)
    except PipelineValidationError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    if not plan_has_streaming_generator(planned.plan):
        raise HTTPException(
            status_code=400,
            detail="Streaming endpoint requires streaming_generator in the generation selection.",
        )

    def event_stream():
        try:
            for event_type, data in stream_pipeline_run(payload, sources, loader_service):
                yield _sse_event(event_type, data)
        except PipelineValidationError as error:
            yield _sse_event("error", {"detail": str(error), "code": "validation_error"})
        except Exception as error:
            yield _sse_event("error", {"detail": str(error), "code": "runtime_error"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
