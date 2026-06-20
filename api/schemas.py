from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

SourceType = Literal["file", "directory", "repository"]
StepStatus = Literal["pending", "running", "completed", "skipped", "error"]

class SourceRecord(BaseModel):
    id: str
    name: str
    source_type: SourceType
    loader: str
    path: str
    size_bytes: int | None = None
    repo_url: str | None = None
    branch: str | None = None
    commit_sha: str | None = None
    created_at: datetime

class RegisterPathRequest(BaseModel):
    path: str = Field(..., min_length=1)

class RegisterPathResponse(BaseModel):
    source: SourceRecord

class RegisterRepoRequest(BaseModel):
    repo_url: str = Field(..., min_length=1)
    branch: str | None = None

class RegisterRepoResponse(BaseModel):
    source: SourceRecord

class SourcesResponse(BaseModel):
    sources: list[SourceRecord]

class UploadSourcesResponse(BaseModel):
    sources: list[SourceRecord]

class PipelineSelection(BaseModel):
    chunking: list[str] = Field(default_factory=list)
    indexing: list[str] = Field(default_factory=list)
    query: list[str] = Field(default_factory=list)
    retrieval: list[str] = Field(default_factory=list)
    ranking: list[str] = Field(default_factory=list)
    context: list[str] = Field(default_factory=list)
    generation: list[str] = Field(default_factory=list)
    postprocessing: list[str] = Field(default_factory=list)
    evaluation: list[str] = Field(default_factory=list)

class PipelineRequest(BaseModel):
    query: str = Field(..., min_length=1)
    source_ids: list[str] = Field(default_factory=list)
    selection: PipelineSelection
    top_k: int | None = Field(default=None, ge=1, le=100)
    template_name: str | None = Field(default="summarize.yaml")
    parser_name: str | None = Field(default="Answer")
    save_intermediate: bool | None = None
    intermediate_run_id: str | None = None
    initialization_id: str | None = None
    skip_initialization: bool = False

class PipelineStep(BaseModel):
    name: str
    component: str | list[str]
    options: dict[str, object] | None = None
    parser: str | None = None
    template_name: str | None = None
    top_k: int | None = None
    fuse: bool | None = None
    if_under: int | None = None
    merge_with_existing: bool | None = None

class PipelinePlan(BaseModel):
    init_pipeline: list[PipelineStep]
    pipeline: list[PipelineStep]
    indexers: list[str]
    selected_components: dict[str, list[str]]

class PipelinePreviewResponse(BaseModel):
    plan: PipelinePlan
    warnings: list[str] = Field(default_factory=list)

class PipelineRunStep(BaseModel):
    step_id: str
    step_name: str
    component: str
    status: StepStatus = "completed"
    latency_ms: float | None = None
    output_type: str = "raw_json"
    output: dict[str, object] = Field(default_factory=dict)
    summary: str | None = None
    error: str | None = None

class PipelineRunResponse(BaseModel):
    plan: PipelinePlan
    result: dict[str, object]
    warnings: list[str] = Field(default_factory=list)
    run_id: str | None = None
    pipeline_id: str | None = None
    pipeline_name: str | None = None
    status: StepStatus = "completed"
    query: str | None = None
    steps: list[PipelineRunStep] = Field(default_factory=list)

class PipelineInitializeResponse(BaseModel):
    initialization_id: str
    plan: PipelinePlan
    warnings: list[str] = Field(default_factory=list)
    status: StepStatus = "completed"
    init_skipped: bool = False
    source_count: int = 0
    document_count: int = 0
    steps: list[PipelineRunStep] = Field(default_factory=list)

class PipelineTemplate(BaseModel):
    id: str
    name: str
    file: str
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    components: list[str] = Field(default_factory=list)
    init_steps: list[dict[str, object]] = Field(default_factory=list)
    run_steps: list[dict[str, object]] = Field(default_factory=list)

class PipelineTemplatesResponse(BaseModel):
    templates: list[PipelineTemplate]

class PromptTemplateItem(BaseModel):
    name: str
    label: str
    template: str
    variables: list[str] = Field(default_factory=list)
    content: str

class PromptTemplatesResponse(BaseModel):
    prompts: list[PromptTemplateItem]

class PromptCreateRequest(BaseModel):
    name: str = Field(..., min_length=1)
    template: str = Field(..., min_length=1)
    overwrite: bool = False

class PromptCreateResponse(BaseModel):
    prompt: PromptTemplateItem

class ExperimentListItem(BaseModel):
    name: str
    run_id: str
    path: str
    created_at_utc: str | None = None
    status: str = "completed"
    variants: int = 0
    queries: int = 0
    metrics: list[str] = Field(default_factory=list)
    best: dict[str, str | None] = Field(default_factory=dict)

class ExperimentsResponse(BaseModel):
    experiments: list[ExperimentListItem]

class ExperimentDetailResponse(BaseModel):
    name: str
    run_id: str
    path: str
    manifest: dict[str, object] = Field(default_factory=dict)
    comparison: dict[str, object] = Field(default_factory=dict)
    summaries: list[dict[str, object]] = Field(default_factory=list)

class ExperimentQueriesResponse(BaseModel):
    name: str
    run_id: str
    queries: list[dict[str, object]] = Field(default_factory=list)

class ExperimentConfigItem(BaseModel):
    file: str
    name: str | None = None
    dataset: str | None = None
    sources: str | None = None
    runtime: str | None = None
    env: str | None = None
    parallelism: int | None = None
    metrics: list[str] = Field(default_factory=list)
    variants: list[str] = Field(default_factory=list)
    valid: bool = True
    error: str | None = None

class ExperimentConfigsResponse(BaseModel):
    configs: list[ExperimentConfigItem]

class ExperimentConfigValidateRequest(BaseModel):
    yaml_text: str = Field(..., min_length=1)

class ExperimentConfigValidateResponse(BaseModel):
    valid: bool
    config: ExperimentConfigItem | None = None
    errors: list[str] = Field(default_factory=list)

class ExperimentConfigSaveRequest(BaseModel):
    file_name: str = Field(..., min_length=1)
    yaml_text: str = Field(..., min_length=1)
    overwrite: bool = False

class ExperimentConfigSaveResponse(BaseModel):
    config: ExperimentConfigItem

class ExperimentRunRequest(BaseModel):
    config_file: str | None = None
    yaml_text: str | None = None
    save_as: str | None = None
    overwrite: bool = False

class ExperimentRunResponse(BaseModel):
    name: str
    run_id: str
    path: str
    status: str = "completed"
    comparison: dict[str, object] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
