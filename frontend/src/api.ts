import type {
  ComponentCatalogResponse,
  ExperimentConfigSaveResponse,
  ExperimentConfigsResponse,
  ExperimentConfigValidateResponse,
  ExperimentDetailResponse,
  ExperimentQueriesResponse,
  ExperimentRunRequest,
  ExperimentRunResponse,
  ExperimentsResponse,
  PipelineInitializeResponse,
  PipelinePreviewResponse,
  PipelineRequest,
  PipelineRunResponse,
  PipelineTemplatesResponse,
  PromptCreateResponse,
  PromptTemplatesResponse,
  SourcesResponse,
} from './types';

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000';

async function handleResponse<T>(response: Response): Promise<T> {
  if (response.ok) {
    return (await response.json()) as T;
  }

  let detail = `Request failed with status ${response.status}`;
  try {
    const payload = (await response.json()) as { detail?: string };
    if (payload.detail) {
      detail = payload.detail;
    }
  } catch {
    // Ignore JSON parse failures and use default detail.
  }

  throw new Error(detail);
}

export async function getCatalog(): Promise<ComponentCatalogResponse> {
  const response = await fetch(`${API_BASE}/api/components/catalog`);
  return handleResponse<ComponentCatalogResponse>(response);
}

export async function getSources(): Promise<SourcesResponse> {
  const response = await fetch(`${API_BASE}/api/sources`);
  return handleResponse<SourcesResponse>(response);
}

export async function uploadSources(files: File[]): Promise<SourcesResponse> {
  const form = new FormData();
  files.forEach((file) => {
    form.append('files', file);
  });

  const response = await fetch(`${API_BASE}/api/sources/upload`, {
    method: 'POST',
    body: form,
  });

  return handleResponse<SourcesResponse>(response);
}

export async function registerSourcePath(path: string): Promise<SourcesResponse> {
  const response = await fetch(`${API_BASE}/api/sources/register-path`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ path }),
  });

  await handleResponse<{ source: unknown }>(response);
  return getSources();
}

export async function registerPublicRepo(repoUrl: string, branch?: string): Promise<SourcesResponse> {
  const response = await fetch(`${API_BASE}/api/sources/register-repo`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      repo_url: repoUrl,
      branch: branch?.trim() || null,
    }),
  });

  await handleResponse<{ source: unknown }>(response);
  return getSources();
}

export async function previewPipeline(payload: PipelineRequest): Promise<PipelinePreviewResponse> {
  const response = await fetch(`${API_BASE}/api/pipelines/preview`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  return handleResponse<PipelinePreviewResponse>(response);
}

export async function initializePipeline(payload: PipelineRequest): Promise<PipelineInitializeResponse> {
  const response = await fetch(`${API_BASE}/api/pipelines/initialize`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  return handleResponse<PipelineInitializeResponse>(response);
}

export async function runPipeline(payload: PipelineRequest): Promise<PipelineRunResponse> {
  const response = await fetch(`${API_BASE}/api/pipelines/run`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  return handleResponse<PipelineRunResponse>(response);
}

export async function getPipelineTemplates(): Promise<PipelineTemplatesResponse> {
  const response = await fetch(`${API_BASE}/api/pipelines/templates`);
  return handleResponse<PipelineTemplatesResponse>(response);
}

export async function getPrompts(): Promise<PromptTemplatesResponse> {
  const response = await fetch(`${API_BASE}/api/prompts`);
  return handleResponse<PromptTemplatesResponse>(response);
}

export async function createPrompt(
  name: string,
  template: string,
  overwrite = false,
): Promise<PromptCreateResponse> {
  const response = await fetch(`${API_BASE}/api/prompts`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ name, template, overwrite }),
  });
  return handleResponse<PromptCreateResponse>(response);
}

export async function getExperiments(): Promise<ExperimentsResponse> {
  const response = await fetch(`${API_BASE}/api/experiments`);
  return handleResponse<ExperimentsResponse>(response);
}

export async function getExperimentDetail(name: string, runId: string): Promise<ExperimentDetailResponse> {
  const response = await fetch(`${API_BASE}/api/experiments/${encodeURIComponent(name)}/${encodeURIComponent(runId)}`);
  return handleResponse<ExperimentDetailResponse>(response);
}

export async function getExperimentQueries(
  name: string,
  runId: string,
  limit = 25,
): Promise<ExperimentQueriesResponse> {
  const response = await fetch(
    `${API_BASE}/api/experiments/${encodeURIComponent(name)}/${encodeURIComponent(runId)}/queries?limit=${limit}`,
  );
  return handleResponse<ExperimentQueriesResponse>(response);
}

export async function getExperimentConfigs(): Promise<ExperimentConfigsResponse> {
  const response = await fetch(`${API_BASE}/api/experiments/configs`);
  return handleResponse<ExperimentConfigsResponse>(response);
}

export async function validateExperimentConfig(yamlText: string): Promise<ExperimentConfigValidateResponse> {
  const response = await fetch(`${API_BASE}/api/experiments/configs/validate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ yaml_text: yamlText }),
  });
  return handleResponse<ExperimentConfigValidateResponse>(response);
}

export async function saveExperimentConfig(
  fileName: string,
  yamlText: string,
  overwrite = false,
): Promise<ExperimentConfigSaveResponse> {
  const response = await fetch(`${API_BASE}/api/experiments/configs`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ file_name: fileName, yaml_text: yamlText, overwrite }),
  });
  return handleResponse<ExperimentConfigSaveResponse>(response);
}

export async function runExperiment(payload: ExperimentRunRequest): Promise<ExperimentRunResponse> {
  const response = await fetch(`${API_BASE}/api/experiments/run`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });
  return handleResponse<ExperimentRunResponse>(response);
}

export type StreamEventHandler = (event: string, data: Record<string, unknown>) => void;

function parseSseBlock(block: string): { event: string; data: Record<string, unknown> } | null {
  let event = 'message';
  const dataLines: string[] = [];
  for (const line of block.split('\n')) {
    if (line.startsWith('event:')) {
      event = line.slice(6).trim();
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).trim());
    }
  }
  if (dataLines.length === 0) {
    return null;
  }
  try {
    return { event, data: JSON.parse(dataLines.join('\n')) as Record<string, unknown> };
  } catch {
    return null;
  }
}

export async function streamPipeline(
  payload: PipelineRequest,
  onEvent: StreamEventHandler,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${API_BASE}/api/pipelines/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    },
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok || !response.body) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const errorBody = (await response.json()) as { detail?: string };
      if (errorBody.detail) {
        detail = errorBody.detail;
      }
    } catch {
      // ignore — fall through with status-based detail
    }
    throw new Error(detail);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });

    let boundary = buffer.indexOf('\n\n');
    while (boundary !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const parsed = parseSseBlock(block);
      if (parsed) {
        onEvent(parsed.event, parsed.data);
      }
      boundary = buffer.indexOf('\n\n');
    }
  }
}
