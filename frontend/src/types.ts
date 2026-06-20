export type ComponentStatus = 'ready' | 'not_implemented' | 'experimental';

export interface Subcomponent {
  id: string;
  label: string;
  description: string;
  status: ComponentStatus;
}

export interface ComponentGroup {
  id: string;
  label: string;
  description: string;
  multi_select: boolean;
  required: boolean;
  default: string[];
  subcomponents: Subcomponent[];
}

export interface ComponentCatalogResponse {
  groups: ComponentGroup[];
  defaults: Record<string, string[]>;
}

export interface SourceRecord {
  id: string;
  name: string;
  source_type: 'file' | 'directory' | 'repository';
  loader: string;
  path: string;
  size_bytes?: number | null;
  repo_url?: string | null;
  branch?: string | null;
  commit_sha?: string | null;
  created_at: string;
}

export interface SourcesResponse {
  sources: SourceRecord[];
}

export interface PipelineSelection {
  chunking: string[];
  indexing: string[];
  query: string[];
  retrieval: string[];
  ranking: string[];
  context: string[];
  generation: string[];
  postprocessing: string[];
  evaluation: string[];
}

export interface PipelineRequest {
  query: string;
  source_ids: string[];
  selection: PipelineSelection;
  top_k?: number;
  template_name?: string;
  parser_name?: string;
  save_intermediate?: boolean;
  intermediate_run_id?: string;
  initialization_id?: string | null;
  skip_initialization?: boolean;
}

export interface PipelineStep {
  name: string;
  component: string | string[];
  options?: Record<string, unknown> | null;
  parser?: string | null;
  template_name?: string | null;
  top_k?: number | null;
  fuse?: boolean | null;
  if_under?: number | null;
  merge_with_existing?: boolean | null;
}

export interface PipelinePlan {
  init_pipeline: PipelineStep[];
  pipeline: PipelineStep[];
  indexers: string[];
  selected_components: Record<string, string[]>;
}

export interface PipelinePreviewResponse {
  plan: PipelinePlan;
  warnings: string[];
}

export interface ChunkResult {
  id: string;
  text: string;
  score: number;
  metadata: Record<string, unknown>;
}

export type StepStatus = 'pending' | 'running' | 'completed' | 'skipped' | 'error';

export interface PipelineRunStep {
  step_id: string;
  step_name: string;
  component: string;
  status: StepStatus;
  latency_ms?: number | null;
  output_type: string;
  output: Record<string, unknown>;
  summary?: string | null;
  error?: string | null;
}

export interface PipelineRunResult {
  run_id?: string | null;
  pipeline_id?: string | null;
  pipeline_name?: string | null;
  status?: StepStatus;
  query?: string;
  answer: string;
  retrieved: ChunkResult[];
  ranked: ChunkResult[];
  retrieval_queries: string[];
  prompt?: string | null;
  evaluation?: Record<string, unknown>;
  steps?: PipelineRunStep[];
  trace?: Record<string, unknown>[];
  init_skipped: boolean;
  intermediate_run_id?: string | null;
  intermediate_path?: string | null;
}

export interface PipelineRunResponse {
  plan: PipelinePlan;
  warnings: string[];
  result: PipelineRunResult;
  run_id?: string | null;
  pipeline_id?: string | null;
  pipeline_name?: string | null;
  status?: StepStatus;
  query?: string | null;
  steps?: PipelineRunStep[];
}

export interface PipelineInitializeResponse {
  initialization_id: string;
  plan: PipelinePlan;
  warnings: string[];
  status: StepStatus;
  init_skipped: boolean;
  source_count: number;
  document_count: number;
  steps: PipelineRunStep[];
}

export interface PipelineTemplate {
  id: string;
  name: string;
  file: string;
  description?: string | null;
  tags: string[];
  components: string[];
  init_steps: Record<string, unknown>[];
  run_steps: Record<string, unknown>[];
}

export interface PipelineTemplatesResponse {
  templates: PipelineTemplate[];
}

export interface PromptTemplateItem {
  name: string;
  label: string;
  template: string;
  variables: string[];
  content: string;
}

export interface PromptTemplatesResponse {
  prompts: PromptTemplateItem[];
}

export interface PromptCreateResponse {
  prompt: PromptTemplateItem;
}

export interface ExperimentListItem {
  name: string;
  run_id: string;
  path: string;
  created_at_utc?: string | null;
  status: string;
  variants: number;
  queries: number;
  metrics: string[];
  best: Record<string, string | null>;
}

export interface ExperimentsResponse {
  experiments: ExperimentListItem[];
}

export interface ExperimentDetailResponse {
  name: string;
  run_id: string;
  path: string;
  manifest: Record<string, unknown>;
  comparison: {
    metrics?: string[];
    variants?: Record<string, Record<string, { value?: number | null; count?: number; higher_is_better?: boolean }>>;
    best?: Record<string, string | null>;
  };
  summaries: Record<string, unknown>[];
}

export interface ExperimentQueryVariant {
  answer?: string;
  contexts?: string[];
  latency_ms?: number | null;
  error?: string | null;
}

export interface ExperimentQueryItem {
  index: number;
  question: string;
  ground_truth?: string | null;
  reference_contexts?: string[] | null;
  variants: Record<string, ExperimentQueryVariant>;
}

export interface ExperimentQueriesResponse {
  name: string;
  run_id: string;
  queries: ExperimentQueryItem[];
}

export interface ExperimentConfigItem {
  file: string;
  name?: string | null;
  dataset?: string | null;
  sources?: string | null;
  runtime?: string | null;
  env?: string | null;
  parallelism?: number | null;
  metrics: string[];
  variants: string[];
  valid: boolean;
  error?: string | null;
}

export interface ExperimentConfigsResponse {
  configs: ExperimentConfigItem[];
}

export interface ExperimentConfigValidateResponse {
  valid: boolean;
  config?: ExperimentConfigItem | null;
  errors: string[];
}

export interface ExperimentConfigSaveResponse {
  config: ExperimentConfigItem;
}

export interface ExperimentRunRequest {
  config_file?: string | null;
  yaml_text?: string | null;
  save_as?: string | null;
  overwrite?: boolean;
}

export interface ExperimentRunResponse {
  name: string;
  run_id: string;
  path: string;
  status: string;
  comparison: ExperimentDetailResponse['comparison'];
  warnings: string[];
}
