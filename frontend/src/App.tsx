import {
  Alert,
  AppBar,
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  Checkbox,
  Chip,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Divider,
  Drawer,
  FormControlLabel,
  IconButton,
  LinearProgress,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  MenuItem,
  Paper,
  Radio,
  RadioGroup,
  Stack,
  Tab,
  Tabs,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  Toolbar,
  Tooltip,
  Typography,
} from '@mui/material';
import AccountTreeIcon from '@mui/icons-material/AccountTree';
import AddIcon from '@mui/icons-material/Add';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import CompareArrowsIcon from '@mui/icons-material/CompareArrows';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import HourglassEmptyIcon from '@mui/icons-material/HourglassEmpty';
import MenuIcon from '@mui/icons-material/Menu';
import PlayCircleIcon from '@mui/icons-material/PlayCircle';
import RefreshIcon from '@mui/icons-material/Refresh';
import ScienceIcon from '@mui/icons-material/Science';
import { ChangeEvent, ReactNode, useEffect, useMemo, useState } from 'react';

import {
  createPrompt,
  getCatalog,
  getExperimentConfigs,
  getExperimentDetail,
  getExperimentQueries,
  getExperiments,
  getPipelineTemplates,
  getPrompts,
  getSources,
  initializePipeline,
  previewPipeline,
  registerPublicRepo,
  registerSourcePath,
  runExperiment,
  runPipeline,
  saveExperimentConfig,
  streamPipeline,
  uploadSources,
  validateExperimentConfig,
} from './api';
import type {
  ChunkResult,
  ComponentCatalogResponse,
  ComponentGroup,
  ComponentStatus,
  ExperimentConfigItem,
  ExperimentDetailResponse,
  ExperimentListItem,
  ExperimentQueryItem,
  PipelinePlan,
  PipelineInitializeResponse,
  PipelinePreviewResponse,
  PipelineRequest,
  PipelineRunResponse,
  PipelineRunStep,
  PipelineSelection,
  PipelineTemplate,
  PromptTemplateItem,
  SourceRecord,
  StepStatus,
  Subcomponent,
} from './types';

type PageId = 'runner' | 'compare' | 'experiments' | 'pipelines';
type Notice = { severity: 'info' | 'success' | 'warning'; text: string };
type RunnerStep = 'setup' | 'query';
type ExperimentRunMode = 'existing' | 'new';

const SIDEBAR_WIDTH = 272;
const TOPBAR_HEIGHT = 76;

const EMPTY_SELECTION: PipelineSelection = {
  chunking: [],
  indexing: [],
  query: [],
  retrieval: [],
  ranking: [],
  context: [],
  generation: [],
  postprocessing: [],
  evaluation: [],
};

const PRIMARY_RETRIEVERS = new Set(['coarse_retriever', 'fine_retriever', 'hybrid_retriever']);
const EXTERNAL_FALLBACK_THRESHOLD = 3;

const NAV_ITEMS: Array<{ id: PageId; label: string; icon: ReactNode }> = [
  { id: 'runner', label: 'Pipeline Runner', icon: <PlayCircleIcon fontSize="small" /> },
  { id: 'compare', label: 'Compare Pipelines', icon: <CompareArrowsIcon fontSize="small" /> },
  { id: 'experiments', label: 'Experiments', icon: <ScienceIcon fontSize="small" /> },
  { id: 'pipelines', label: 'Pipeline Templates', icon: <AccountTreeIcon fontSize="small" /> },
];

function statusColor(status: ComponentStatus): 'success' | 'warning' | 'default' {
  if (status === 'ready') return 'success';
  if (status === 'experimental') return 'warning';
  return 'default';
}

function formatComponent(component: string | string[]): string {
  return Array.isArray(component) ? component.join(' + ') : component;
}

function formatMs(value?: number | null): string {
  if (value === undefined || value === null || Number.isNaN(value)) return 'n/a';
  if (value >= 1000) return `${(value / 1000).toFixed(1)}s`;
  return `${Math.round(value)}ms`;
}

function formatMetric(value: unknown): string {
  if (value === undefined || value === null) return 'n/a';
  if (typeof value === 'number') {
    if (Math.abs(value) >= 1000) return value.toFixed(0);
    return value.toFixed(3).replace(/0+$/, '').replace(/\.$/, '');
  }
  return String(value);
}

function metricPercent(value: unknown): number {
  if (typeof value !== 'number' || Number.isNaN(value)) return 0;
  const normalized = value > 1 ? value / 100 : value;
  return Math.max(0, Math.min(100, normalized * 100));
}

function safeJson(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

function templateSelection(template: PipelineTemplate, catalog: ComponentCatalogResponse | null): PipelineSelection {
  if (!catalog) return EMPTY_SELECTION;
  const byComponent = new Map<string, ComponentGroup>();
  catalog.groups.forEach((group) => {
    group.subcomponents.forEach((sub) => byComponent.set(sub.id, group));
  });

  const next: PipelineSelection = {
    chunking: [],
    indexing: [],
    query: [],
    retrieval: [],
    ranking: [],
    context: [],
    generation: [],
    postprocessing: [],
    evaluation: [],
  };

  template.components.forEach((component) => {
    const group = byComponent.get(component);
    if (!group) return;
    const key = group.id as keyof PipelineSelection;
    if (group.multi_select) {
      if (!next[key].includes(component)) next[key].push(component);
    } else if (next[key].length === 0) {
      next[key] = [component];
    }
  });

  catalog.groups.forEach((group) => {
    const key = group.id as keyof PipelineSelection;
    if (group.required && next[key].length === 0) {
      next[key] = [...group.default];
    }
  });

  return next;
}

function firstStepOutput(
  result: PipelineRunResponse | null,
  outputType: string,
): Record<string, unknown> | null {
  const step = (result?.steps ?? result?.result.steps ?? []).find((item) => item.output_type === outputType);
  return step?.output ?? null;
}

function chunkSource(chunk: ChunkResult): string {
  const source = chunk.metadata?.source ?? chunk.metadata?.path ?? chunk.metadata?.file_path;
  return source ? String(source) : chunk.id;
}

function stepIcon(status: StepStatus) {
  if (status === 'completed') return <CheckCircleIcon fontSize="small" />;
  if (status === 'error') return <ErrorOutlineIcon fontSize="small" />;
  return <HourglassEmptyIcon fontSize="small" />;
}

function PageHeader({
  title,
  description,
  actions,
}: {
  title: string;
  description: string;
  actions?: ReactNode;
}) {
  return (
    <Stack
      direction={{ xs: 'column', md: 'row' }}
      justifyContent="space-between"
      alignItems={{ xs: 'flex-start', md: 'flex-start' }}
      spacing={2}
      sx={{ mb: 3 }}
    >
      <Box>
        <Typography variant="h4" sx={{ lineHeight: 1.05 }}>
          {title}
        </Typography>
        <Typography color="text.secondary" sx={{ mt: 1, maxWidth: 860, lineHeight: 1.6 }}>
          {description}
        </Typography>
      </Box>
      {actions ? <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>{actions}</Stack> : null}
    </Stack>
  );
}

function MetricCard({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <Card>
      <CardContent>
        <Typography variant="caption" color="text.secondary" fontWeight={900}>
          {label}
        </Typography>
        <Typography sx={{ mt: 0.8, fontSize: 23, fontWeight: 900, lineHeight: 1.1 }}>
          {value}
        </Typography>
        {sub ? (
          <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
            {sub}
          </Typography>
        ) : null}
      </CardContent>
    </Card>
  );
}

function CodeBlock({ children }: { children: ReactNode }) {
  return (
    <Box
      component="pre"
      sx={{
        m: 0,
        p: 2,
        borderRadius: 2,
        bgcolor: '#0f172a',
        color: '#e2e8f0',
        overflow: 'auto',
        whiteSpace: 'pre-wrap',
        lineHeight: 1.55,
        fontSize: 12,
        maxHeight: 520,
      }}
    >
      {children}
    </Box>
  );
}

function EmptyState({ text }: { text: string }) {
  return (
    <Paper variant="outlined" sx={{ p: 2, bgcolor: '#f8fafc' }}>
      <Typography variant="body2" color="text.secondary">
        {text}
      </Typography>
    </Paper>
  );
}

function truncateText(text: string, limit = 420): string {
  if (text.length <= limit) return text;
  return `${text.slice(0, limit).trimEnd()}...`;
}

function renderInlineMarkdown(text: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  const pattern = /(\[[^\]]+\]\([^)]+\)|`[^`]+`|\*\*[^*]+\*\*|__[^_]+__|\*[^*]+\*|_[^_]+_)/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      nodes.push(text.slice(lastIndex, match.index));
    }

    const token = match[0];
    const key = `${match.index}-${token}`;
    const link = token.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
    if (link) {
      nodes.push(
        <Box
          key={key}
          component="a"
          href={link[2]}
          target="_blank"
          rel="noreferrer"
          sx={{ color: 'primary.main', fontWeight: 800, textDecoration: 'none' }}
        >
          {link[1]}
        </Box>,
      );
    } else if (token.startsWith('`')) {
      nodes.push(
        <Box
          key={key}
          component="code"
          sx={{
            px: 0.5,
            py: 0.15,
            borderRadius: 0.75,
            bgcolor: '#e2e8f0',
            color: '#0f172a',
            fontSize: '0.9em',
          }}
        >
          {token.slice(1, -1)}
        </Box>,
      );
    } else if (token.startsWith('**') || token.startsWith('__')) {
      nodes.push(<strong key={key}>{token.slice(2, -2)}</strong>);
    } else {
      nodes.push(<em key={key}>{token.slice(1, -1)}</em>);
    }

    lastIndex = match.index + token.length;
  }

  if (lastIndex < text.length) {
    nodes.push(text.slice(lastIndex));
  }

  return nodes;
}

function isMarkdownTableSeparator(line: string): boolean {
  return /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line);
}

function splitMarkdownTableLine(line: string): string[] {
  return line
    .trim()
    .replace(/^\|/, '')
    .replace(/\|$/, '')
    .split('|')
    .map((cell) => cell.trim());
}

function MarkdownViewer({ children, compact = false }: { children: string; compact?: boolean }) {
  if (!children.trim()) {
    return <Typography color="text.secondary">No content.</Typography>;
  }

  const lines = children.replace(/\r\n/g, '\n').split('\n');
  const elements: ReactNode[] = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index];
    const trimmed = line.trim();

    if (!trimmed) {
      index += 1;
      continue;
    }

    if (trimmed.startsWith('```')) {
      const codeLines: string[] = [];
      index += 1;
      while (index < lines.length && !lines[index].trim().startsWith('```')) {
        codeLines.push(lines[index]);
        index += 1;
      }
      index += 1;
      elements.push(<CodeBlock key={`code-${index}`}>{codeLines.join('\n')}</CodeBlock>);
      continue;
    }

    if (index + 1 < lines.length && line.includes('|') && isMarkdownTableSeparator(lines[index + 1])) {
      const header = splitMarkdownTableLine(line);
      const rows: string[][] = [];
      index += 2;
      while (index < lines.length && lines[index].includes('|') && lines[index].trim()) {
        rows.push(splitMarkdownTableLine(lines[index]));
        index += 1;
      }
      elements.push(
        <TableContainer key={`table-${index}`} component={Paper} variant="outlined" sx={{ my: compact ? 1 : 1.5 }}>
          <Table size="small">
            <TableHead>
              <TableRow>
                {header.map((cell, cellIndex) => (
                  <TableCell key={`${cell}-${cellIndex}`}>{renderInlineMarkdown(cell)}</TableCell>
                ))}
              </TableRow>
            </TableHead>
            <TableBody>
              {rows.map((row, rowIndex) => (
                <TableRow key={`row-${rowIndex}`}>
                  {header.map((_, cellIndex) => (
                    <TableCell key={`cell-${rowIndex}-${cellIndex}`}>
                      {renderInlineMarkdown(row[cellIndex] ?? '')}
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>,
      );
      continue;
    }

    const heading = trimmed.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      const level = heading[1].length;
      elements.push(
        <Typography
          key={`heading-${index}`}
          variant={level <= 2 ? 'h6' : 'subtitle1'}
          sx={{ mt: compact ? 1 : 1.5, mb: 0.75, fontWeight: 900 }}
        >
          {renderInlineMarkdown(heading[2])}
        </Typography>,
      );
      index += 1;
      continue;
    }

    if (/^(-{3,}|\*{3,})$/.test(trimmed)) {
      elements.push(<Divider key={`hr-${index}`} sx={{ my: compact ? 1 : 1.5 }} />);
      index += 1;
      continue;
    }

    if (trimmed.startsWith('>')) {
      const quoteLines: string[] = [];
      while (index < lines.length && lines[index].trim().startsWith('>')) {
        quoteLines.push(lines[index].trim().replace(/^>\s?/, ''));
        index += 1;
      }
      elements.push(
        <Paper
          key={`quote-${index}`}
          variant="outlined"
          sx={{ my: compact ? 1 : 1.5, p: 1.5, bgcolor: '#f8fafc', borderLeft: '4px solid #bfdbfe' }}
        >
          <Typography variant="body2" sx={{ color: '#334155', lineHeight: 1.65 }}>
            {renderInlineMarkdown(quoteLines.join(' '))}
          </Typography>
        </Paper>,
      );
      continue;
    }

    const listMatch = trimmed.match(/^(([-*])|\d+\.)\s+(.+)$/);
    if (listMatch) {
      const ordered = /^\d+\./.test(listMatch[1]);
      const items: string[] = [];
      while (index < lines.length) {
        const itemMatch = lines[index].trim().match(/^(([-*])|\d+\.)\s+(.+)$/);
        if (!itemMatch || (/^\d+\./.test(itemMatch[1]) !== ordered)) break;
        items.push(itemMatch[3]);
        index += 1;
      }
      elements.push(
        <Box
          key={`list-${index}`}
          component={ordered ? 'ol' : 'ul'}
          sx={{ my: compact ? 0.75 : 1, pl: 3, color: '#334155', lineHeight: 1.65 }}
        >
          {items.map((item, itemIndex) => (
            <li key={`${item}-${itemIndex}`}>{renderInlineMarkdown(item)}</li>
          ))}
        </Box>,
      );
      continue;
    }

    const paragraph: string[] = [];
    while (
      index < lines.length
      && lines[index].trim()
      && !lines[index].trim().startsWith('```')
      && !lines[index].trim().match(/^(#{1,4})\s+(.+)$/)
      && !lines[index].trim().match(/^(([-*])|\d+\.)\s+(.+)$/)
      && !lines[index].trim().startsWith('>')
      && !(index + 1 < lines.length && lines[index].includes('|') && isMarkdownTableSeparator(lines[index + 1]))
    ) {
      paragraph.push(lines[index].trim());
      index += 1;
    }
    elements.push(
      <Typography
        key={`paragraph-${index}`}
        variant={compact ? 'body2' : 'body1'}
        sx={{ my: compact ? 0.75 : 1, color: '#334155', lineHeight: compact ? 1.6 : 1.75 }}
      >
        {renderInlineMarkdown(paragraph.join(' '))}
      </Typography>,
    );
  }

  return <Box sx={{ '& > :first-of-type': { mt: 0 }, '& > :last-child': { mb: 0 } }}>{elements}</Box>;
}

export default function App() {
  const [activePage, setActivePage] = useState<PageId>('runner');
  const [mobileOpen, setMobileOpen] = useState(false);

  const [catalog, setCatalog] = useState<ComponentCatalogResponse | null>(null);
  const [selection, setSelection] = useState<PipelineSelection>(EMPTY_SELECTION);
  const [activeComponentTab, setActiveComponentTab] = useState<string>('chunking');

  const [sources, setSources] = useState<SourceRecord[]>([]);
  const [selectedSourceIds, setSelectedSourceIds] = useState<string[]>([]);
  const [pathInput, setPathInput] = useState('');
  const [repoUrlInput, setRepoUrlInput] = useState('');
  const [repoBranchInput, setRepoBranchInput] = useState('');

  const [templates, setTemplates] = useState<PipelineTemplate[]>([]);
  const [selectedTemplateId, setSelectedTemplateId] = useState<string>('');
  const [query, setQuery] = useState('What are key retrieval approaches for a RAG system?');
  const [topK, setTopK] = useState(5);
  const [templateName, setTemplateName] = useState('summarize.yaml');
  const [parserName, setParserName] = useState('Answer');

  const [prompts, setPrompts] = useState<PromptTemplateItem[]>([]);
  const [promptDialogOpen, setPromptDialogOpen] = useState(false);
  const [newPromptName, setNewPromptName] = useState('');
  const [newPromptTemplate, setNewPromptTemplate] = useState('');
  const [promptError, setPromptError] = useState('');
  const [savingPrompt, setSavingPrompt] = useState(false);

  const [preview, setPreview] = useState<PipelinePreviewResponse | null>(null);
  const [previewing, setPreviewing] = useState(false);
  const [previewError, setPreviewError] = useState('');
  const [runnerStep, setRunnerStep] = useState<RunnerStep>('setup');
  const [initializing, setInitializing] = useState(false);
  const [initialized, setInitialized] = useState<PipelineInitializeResponse | null>(null);
  const [initializedFingerprint, setInitializedFingerprint] = useState('');
  const [runResult, setRunResult] = useState<PipelineRunResponse | null>(null);
  const [liveSteps, setLiveSteps] = useState<PipelineRunStep[]>([]);
  const [runnerTab, setRunnerTab] = useState(0);
  const [streamingAnswer, setStreamingAnswer] = useState('');
  const [streaming, setStreaming] = useState(false);

  const [experiments, setExperiments] = useState<ExperimentListItem[]>([]);
  const [selectedExperimentKey, setSelectedExperimentKey] = useState('');
  const [experimentDetail, setExperimentDetail] = useState<ExperimentDetailResponse | null>(null);
  const [experimentQueries, setExperimentQueries] = useState<ExperimentQueryItem[]>([]);
  const [compareTab, setCompareTab] = useState(0);
  const [loadingExperiments, setLoadingExperiments] = useState(false);
  const [experimentRunOpen, setExperimentRunOpen] = useState(false);
  const [experimentRunMode, setExperimentRunMode] = useState<ExperimentRunMode>('existing');
  const [experimentConfigs, setExperimentConfigs] = useState<ExperimentConfigItem[]>([]);
  const [selectedExperimentConfig, setSelectedExperimentConfig] = useState('');
  const [experimentYamlText, setExperimentYamlText] = useState('');
  const [experimentConfigFileName, setExperimentConfigFileName] = useState('');
  const [experimentOverwrite, setExperimentOverwrite] = useState(false);
  const [experimentRunError, setExperimentRunError] = useState('');
  const [experimentRunning, setExperimentRunning] = useState(false);
  const [experimentConfigLoading, setExperimentConfigLoading] = useState(false);
  const [experimentConfigValidation, setExperimentConfigValidation] = useState<string[]>([]);

  const [loadingInitial, setLoadingInitial] = useState(true);
  const [loadingSources, setLoadingSources] = useState(true);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState<Notice | null>(null);

  useEffect(() => {
    const init = async () => {
      setLoadingInitial(true);
      setLoadingSources(true);
      setError('');
      try {
        const [catalogPayload, sourcesPayload, templatesPayload, promptsPayload, experimentsPayload] = await Promise.all([
          getCatalog(),
          getSources(),
          getPipelineTemplates(),
          getPrompts(),
          getExperiments(),
        ]);
        setCatalog(catalogPayload);
        setSelection({ ...EMPTY_SELECTION, ...catalogPayload.defaults });
        setSources(sourcesPayload.sources);
        setTemplates(templatesPayload.templates);
        setPrompts(promptsPayload.prompts);
        setExperiments(experimentsPayload.experiments);
        setActiveComponentTab(catalogPayload.groups[0]?.id ?? 'chunking');
        if (templatesPayload.templates.length > 0) {
          setSelectedTemplateId(templatesPayload.templates[0].id);
        }
        if (experimentsPayload.experiments.length > 0) {
          const first = experimentsPayload.experiments[0];
          setSelectedExperimentKey(`${first.name}::${first.run_id}`);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load Studio data.');
      } finally {
        setLoadingInitial(false);
        setLoadingSources(false);
      }
    };
    void init();
  }, []);

  useEffect(() => {
    if (!selectedExperimentKey) return;
    const [name, runId] = selectedExperimentKey.split('::');
    if (!name || !runId) return;
    const load = async () => {
      setLoadingExperiments(true);
      try {
        const [detail, queries] = await Promise.all([
          getExperimentDetail(name, runId),
          getExperimentQueries(name, runId, 25),
        ]);
        setExperimentDetail(detail);
        setExperimentQueries(queries.queries);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load experiment detail.');
      } finally {
        setLoadingExperiments(false);
      }
    };
    void load();
  }, [selectedExperimentKey]);

  const isSelectionComplete = useMemo(() => {
    if (!catalog) return false;
    return catalog.groups
      .filter((group) => group.required)
      .every((group) => (selection[group.id as keyof PipelineSelection] ?? []).length > 0);
  }, [catalog, selection]);

  const missingRequiredGroups = useMemo(() => {
    if (!catalog) return [] as string[];
    return catalog.groups
      .filter((group) => group.required && (selection[group.id as keyof PipelineSelection] ?? []).length === 0)
      .map((group) => group.label);
  }, [catalog, selection]);

  const isStreamingPipeline = selection.generation.includes('streaming_generator');
  const canInitialize = isSelectionComplete && selectedSourceIds.length > 0 && !working && !initializing && !!preview;

  const selectedTemplate = useMemo(
    () => templates.find((item) => item.id === selectedTemplateId) ?? null,
    [templates, selectedTemplateId],
  );

  const selectedPrompt = useMemo(
    () => prompts.find((item) => item.name === templateName) ?? null,
    [prompts, templateName],
  );

  const configFingerprint = useMemo(
    () => JSON.stringify({
      source_ids: selectedSourceIds,
      selection,
      top_k: topK,
      template_name: templateName,
      parser_name: parserName,
    }),
    [selectedSourceIds, selection, topK, templateName, parserName],
  );

  const canRun = Boolean(initialized?.initialization_id)
    && runnerStep === 'query'
    && initializedFingerprint === configFingerprint
    && query.trim().length > 0
    && !working
    && !initializing
    && !!preview;

  const buildPipelineRequest = (options?: { skipInitialization?: boolean }): PipelineRequest => {
    const skipInitialization = Boolean(options?.skipInitialization);
    return {
      query,
      source_ids: selectedSourceIds,
      selection,
      top_k: topK,
      template_name: templateName,
      parser_name: parserName,
      skip_initialization: skipInitialization,
      initialization_id: skipInitialization ? initialized?.initialization_id ?? null : null,
    };
  };

  useEffect(() => {
    if (!catalog || !isSelectionComplete) {
      setPreview(null);
      setPreviewError('');
      return;
    }
    const timer = setTimeout(() => {
      void refreshPreview();
    }, 300);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [catalog, isSelectionComplete, selection, topK, templateName, parserName]);

  useEffect(() => {
    if (!initialized || initializedFingerprint === configFingerprint) {
      return;
    }
    setInitialized(null);
    setInitializedFingerprint('');
    setRunResult(null);
    setLiveSteps([]);
    setStreamingAnswer('');
    setRunnerStep('setup');
    setNotice({
      severity: 'info',
      text: 'Pipeline configuration changed. Re-initialize before querying.',
    });
  }, [configFingerprint, initialized, initializedFingerprint]);

  const refreshPreview = async () => {
    setPreviewError('');
    setPreviewing(true);
    try {
      const payload = await previewPipeline(buildPipelineRequest());
      setPreview(payload);
    } catch (err) {
      setPreviewError(err instanceof Error ? err.message : 'Pipeline preview failed.');
    } finally {
      setPreviewing(false);
    }
  };

  const handleToggleSubcomponent = (
    groupId: keyof PipelineSelection,
    subcomponentId: string,
    checked: boolean,
  ) => {
    setSelection((previous) => {
      const current = previous[groupId] ?? [];
      let next = checked
        ? [...current, subcomponentId]
        : current.filter((item) => item !== subcomponentId);

      if (groupId === 'retrieval' && checked && PRIMARY_RETRIEVERS.has(subcomponentId)) {
        next = next.filter((item) => item === subcomponentId || !PRIMARY_RETRIEVERS.has(item));
      }

      return { ...previous, [groupId]: next };
    });
  };

  const handleSetSingleSubcomponent = (
    groupId: keyof PipelineSelection,
    subcomponentId: string,
  ) => {
    setSelection((previous) => ({ ...previous, [groupId]: [subcomponentId] }));
  };

  const handleSelectSource = (sourceId: string, checked: boolean) => {
    setSelectedSourceIds((previous) =>
      checked ? [...previous, sourceId] : previous.filter((item) => item !== sourceId),
    );
  };

  const reloadSources = async () => {
    setLoadingSources(true);
    try {
      const payload = await getSources();
      setSources(payload.sources);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to refresh sources.');
    } finally {
      setLoadingSources(false);
    }
  };

  const handleUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;
    setError('');
    setWorking(true);
    try {
      await uploadSources(Array.from(files));
      await reloadSources();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed.');
    } finally {
      setWorking(false);
      event.target.value = '';
    }
  };

  const handleRegisterPath = async () => {
    if (!pathInput.trim()) return;
    setError('');
    setWorking(true);
    try {
      await registerSourcePath(pathInput.trim());
      setPathInput('');
      await reloadSources();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Path registration failed.');
    } finally {
      setWorking(false);
    }
  };

  const handleRegisterRepo = async () => {
    if (!repoUrlInput.trim()) return;
    setError('');
    setWorking(true);
    try {
      await registerPublicRepo(repoUrlInput.trim(), repoBranchInput.trim());
      setRepoUrlInput('');
      setRepoBranchInput('');
      await reloadSources();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Repository registration failed.');
    } finally {
      setWorking(false);
    }
  };

  const handleRunComparison = async () => {
    if (!selectedExperimentKey) {
      setNotice({ severity: 'warning', text: 'Select an experiment run before comparing pipelines.' });
      return;
    }

    const [name, runId] = selectedExperimentKey.split('::');
    if (!name || !runId) {
      setNotice({ severity: 'warning', text: 'The selected experiment run is invalid.' });
      return;
    }

    setLoadingExperiments(true);
    setError('');
    try {
      const [detail, queries] = await Promise.all([
        getExperimentDetail(name, runId),
        getExperimentQueries(name, runId, 25),
      ]);
      setExperimentDetail(detail);
      setExperimentQueries(queries.queries);
      setCompareTab(0);
      setNotice({ severity: 'success', text: `Comparison refreshed for ${name} / ${runId}.` });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to run comparison.');
    } finally {
      setLoadingExperiments(false);
    }
  };

  const refreshExperiments = async (selectKey?: string) => {
    const payload = await getExperiments();
    setExperiments(payload.experiments);
    if (selectKey) {
      setSelectedExperimentKey(selectKey);
    } else if (!selectedExperimentKey && payload.experiments.length > 0) {
      const first = payload.experiments[0];
      setSelectedExperimentKey(`${first.name}::${first.run_id}`);
    }
  };

  const loadExperimentConfigs = async () => {
    setExperimentConfigLoading(true);
    try {
      const payload = await getExperimentConfigs();
      setExperimentConfigs(payload.configs);
      const preferred = payload.configs.find((config) => config.name === selectedExperiment?.name && config.valid);
      const firstValid = payload.configs.find((config) => config.valid);
      setSelectedExperimentConfig((previous) => previous || preferred?.file || firstValid?.file || '');
    } catch (err) {
      setExperimentRunError(err instanceof Error ? err.message : 'Failed to load experiment configs.');
    } finally {
      setExperimentConfigLoading(false);
    }
  };

  const handleNewExperiment = () => {
    setActivePage('experiments');
    setExperimentRunMode('existing');
    setExperimentRunError('');
    setExperimentConfigValidation([]);
    setExperimentRunOpen(true);
    void loadExperimentConfigs();
  };

  const handleExperimentYamlUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      setExperimentYamlText(await file.text());
      setExperimentConfigFileName(file.name.endsWith('.yaml') || file.name.endsWith('.yml') ? file.name : `${file.name}.yaml`);
      setExperimentConfigValidation([]);
      setExperimentRunError('');
    } catch (err) {
      setExperimentRunError(err instanceof Error ? err.message : 'Failed to read experiment config.');
    } finally {
      event.target.value = '';
    }
  };

  const handleValidateExperimentConfig = async () => {
    setExperimentRunError('');
    setExperimentConfigValidation([]);
    try {
      const payload = await validateExperimentConfig(experimentYamlText);
      if (!payload.valid) {
        setExperimentConfigValidation(payload.errors);
        return false;
      }
      setExperimentConfigValidation([]);
      setNotice({ severity: 'success', text: `Experiment config is valid: ${payload.config?.name ?? 'draft'}.` });
      return true;
    } catch (err) {
      setExperimentRunError(err instanceof Error ? err.message : 'Experiment config validation failed.');
      return false;
    }
  };

  const handleRunExperiment = async () => {
    setExperimentRunError('');
    setExperimentConfigValidation([]);
    setExperimentRunning(true);
    try {
      if (experimentRunMode === 'new') {
        const valid = await handleValidateExperimentConfig();
        if (!valid) return;
        if (experimentConfigFileName.trim()) {
          await saveExperimentConfig(experimentConfigFileName.trim(), experimentYamlText, experimentOverwrite);
        }
      }

      const result = await runExperiment(
        experimentRunMode === 'existing'
          ? { config_file: selectedExperimentConfig }
          : {
              yaml_text: experimentYamlText,
              save_as: experimentConfigFileName.trim() || null,
              overwrite: experimentOverwrite,
            },
      );
      const key = `${result.name}::${result.run_id}`;
      await refreshExperiments(key);
      setExperimentRunOpen(false);
      setNotice({
        severity: result.warnings.length ? 'warning' : 'success',
        text: result.warnings.length
          ? `Experiment completed with ${result.warnings.length} warning(s): ${result.name} / ${result.run_id}`
          : `Experiment completed: ${result.name} / ${result.run_id}`,
      });
    } catch (err) {
      setExperimentRunError(err instanceof Error ? err.message : 'Experiment run failed.');
    } finally {
      setExperimentRunning(false);
    }
  };

  const openPromptDialog = () => {
    setPromptError('');
    setNewPromptName('');
    setNewPromptTemplate('');
    setPromptDialogOpen(true);
  };

  const handleCreatePrompt = async () => {
    setPromptError('');
    setSavingPrompt(true);
    try {
      const { prompt } = await createPrompt(newPromptName.trim(), newPromptTemplate);
      setPrompts((previous) => {
        const others = previous.filter((item) => item.name !== prompt.name);
        return [...others, prompt].sort((a, b) => a.name.localeCompare(b.name));
      });
      setTemplateName(prompt.name);
      setPromptDialogOpen(false);
      setNewPromptName('');
      setNewPromptTemplate('');
      setNotice({ severity: 'success', text: `Prompt saved: ${prompt.name}` });
    } catch (err) {
      setPromptError(err instanceof Error ? err.message : 'Failed to save prompt.');
    } finally {
      setSavingPrompt(false);
    }
  };

  const handleNewPipeline = () => {
    setActivePage('runner');
    setRunnerStep('setup');
    setInitialized(null);
    setInitializedFingerprint('');
    setSelectedTemplateId('');
    setNotice({
      severity: 'info',
      text: 'New pipeline draft started. Adjust component selections in Advanced Config, then run it from the Pipeline Runner.',
    });
  };

  const mergeLiveStep = (step: PipelineRunStep) => {
    setLiveSteps((previous) => {
      const existing = previous.findIndex((item) => item.step_id === step.step_id);
      if (existing === -1) return [...previous, step];
      const next = [...previous];
      next[existing] = step;
      return next;
    });
  };

  const handleInitializePipeline = async () => {
    setError('');
    setNotice(null);
    setInitializing(true);
    setInitialized(null);
    setRunResult(null);
    setLiveSteps([]);
    setStreamingAnswer('');
    try {
      const payload = await initializePipeline(buildPipelineRequest());
      setInitialized(payload);
      setInitializedFingerprint(configFingerprint);
      setPreview({ plan: payload.plan, warnings: payload.warnings });
      setRunnerStep('query');
      setNotice({
        severity: 'success',
        text: payload.init_skipped
          ? 'Pipeline was already initialized from the current index cache. You can query it now.'
          : `Pipeline initialized with ${payload.document_count} loaded documents. You can query it now.`,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Pipeline initialization failed.');
      setRunnerStep('setup');
    } finally {
      setInitializing(false);
    }
  };

  const handleRun = async () => {
    setError('');
    setWorking(true);
    setRunResult(null);
    setLiveSteps([]);
    setRunnerTab(0);

    const handleRunError = (message: string) => {
      setError(message);
      const lower = message.toLowerCase();
      if (lower.includes('initializ') || lower.includes('configuration changed')) {
        setInitialized(null);
        setInitializedFingerprint('');
        setRunnerStep('setup');
        setNotice({
          severity: 'warning',
          text: 'Re-initialize the pipeline before running another query.',
        });
      }
    };

    if (isStreamingPipeline) {
      setStreaming(true);
      setStreamingAnswer('');
      let receivedPlan: PipelinePlan | null = preview?.plan ?? null;
      let receivedWarnings: string[] = preview?.warnings ?? [];
      let streamError = '';

      try {
        await streamPipeline(buildPipelineRequest({ skipInitialization: true }), (event, data) => {
          if (event === 'plan') {
            receivedPlan = data.plan as PipelinePlan;
            receivedWarnings = (data.warnings as string[]) ?? [];
            setPreview({ plan: receivedPlan, warnings: receivedWarnings });
          } else if (event === 'step') {
            mergeLiveStep(data.step as PipelineRunStep);
          } else if (event === 'token') {
            setStreamingAnswer((prev) => prev + String(data.piece ?? ''));
          } else if (event === 'done') {
            const result = data.result as PipelineRunResponse['result'];
            const steps = ((data.steps as PipelineRunStep[]) ?? result.steps ?? []);
            setRunResult({
              plan:
                receivedPlan ?? {
                  init_pipeline: [],
                  pipeline: [],
                  indexers: [],
                  selected_components: {},
                },
              warnings: receivedWarnings,
              result,
              run_id: result.run_id,
              pipeline_id: result.pipeline_id,
              pipeline_name: result.pipeline_name,
              status: result.status,
              query: result.query,
              steps,
            });
            setLiveSteps(steps);
          } else if (event === 'error') {
            streamError = String(data.detail ?? 'Streaming failed.');
          }
        });
        if (streamError) handleRunError(streamError);
      } catch (err) {
        handleRunError(err instanceof Error ? err.message : 'Streaming execution failed.');
      } finally {
        setStreaming(false);
        setWorking(false);
      }
      return;
    }

    try {
      const payload = await runPipeline(buildPipelineRequest({ skipInitialization: true }));
      setRunResult(payload);
      setLiveSteps(payload.steps ?? payload.result.steps ?? []);
      setPreview({ plan: payload.plan, warnings: payload.warnings });
    } catch (err) {
      handleRunError(err instanceof Error ? err.message : 'Pipeline execution failed.');
    } finally {
      setWorking(false);
    }
  };

  const applyTemplate = (template: PipelineTemplate) => {
    setSelectedTemplateId(template.id);
    setSelection(templateSelection(template, catalog));
    const promptStep = template.run_steps.find((step) => step.component === 'prompt_builder');
    const nextTemplate = typeof promptStep?.template_name === 'string' ? promptStep.template_name : templateName;
    const nextParser = typeof promptStep?.parser === 'string' ? promptStep.parser : parserName;
    setTemplateName(nextTemplate);
    setParserName(nextParser);
    setActivePage('runner');
    setRunnerStep('setup');
  };

  const renderSubcomponent = (group: ComponentGroup, sub: Subcomponent) => {
    const selected = (selection[group.id as keyof PipelineSelection] ?? []).includes(sub.id);
    const disabled = sub.status === 'not_implemented';
    const control = group.multi_select ? (
      <Checkbox
        checked={selected}
        disabled={disabled}
        onChange={(event) =>
          handleToggleSubcomponent(group.id as keyof PipelineSelection, sub.id, event.target.checked)
        }
      />
    ) : (
      <Checkbox
        checked={selected}
        disabled={disabled}
        onChange={() => handleSetSingleSubcomponent(group.id as keyof PipelineSelection, sub.id)}
      />
    );

    return (
      <Paper
        key={sub.id}
        variant="outlined"
        sx={{
          p: 1.5,
          bgcolor: selected ? '#eff6ff' : '#fff',
          borderColor: selected ? 'primary.main' : 'divider',
          opacity: disabled ? 0.55 : 1,
        }}
      >
        <Stack direction="row" spacing={1.5} alignItems="flex-start">
          <Tooltip title={disabled ? 'Stub component' : group.multi_select ? 'Toggle component' : 'Select component'}>
            <Box>{control}</Box>
          </Tooltip>
          <Box flex={1} minWidth={0}>
            <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
              <Typography fontWeight={900}>{sub.label}</Typography>
              <Chip
                size="small"
                label={sub.status.replace('_', ' ')}
                color={statusColor(sub.status)}
                variant={sub.status === 'ready' ? 'filled' : 'outlined'}
              />
            </Stack>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
              {sub.description}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              {sub.id}
            </Typography>
          </Box>
        </Stack>
      </Paper>
    );
  };

  const renderPlanFlow = (steps: PipelinePlan['pipeline']) => {
    if (steps.length === 0) return <Typography variant="body2" color="text.secondary">(no steps)</Typography>;
    return (
      <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
        {steps.map((step, index) => (
          <Chip
            key={`${step.name}-${index}`}
            size="small"
            color="primary"
            variant="outlined"
            label={formatComponent(step.component)}
          />
        ))}
      </Stack>
    );
  };

  const displaySteps = useMemo(() => {
    if (liveSteps.length > 0) return liveSteps;
    if (runResult?.steps?.length) return runResult.steps;
    if (runResult?.result.steps?.length) return runResult.result.steps;
    return (preview?.plan.pipeline ?? []).map((step, index) => ({
      step_id: `preview:${index}:${step.name}`,
      step_name: step.name,
      component: formatComponent(step.component),
      status: working ? 'running' as StepStatus : 'pending' as StepStatus,
      latency_ms: null,
      output_type: 'raw_json',
      output: {},
      summary: null,
      error: null,
    }));
  }, [liveSteps, preview, runResult, working]);

  const selectedExperiment = useMemo(
    () => experiments.find((item) => `${item.name}::${item.run_id}` === selectedExperimentKey) ?? null,
    [experiments, selectedExperimentKey],
  );

  const comparison = experimentDetail?.comparison ?? {};
  const comparisonMetrics = comparison.metrics ?? [];
  const variants = comparison.variants ?? {};
  const variantNames = Object.keys(variants);
  const qualityMetric = comparisonMetrics.find((metric) => metric === 'faithfulness')
    ?? comparisonMetrics.find((metric) => metric === 'answer_relevancy')
    ?? comparisonMetrics[0];
  const bestQuality = qualityMetric ? comparison.best?.[qualityMetric] : null;
  const fastest = comparison.best?.latency_ms ?? null;
  const selectedQuery = experimentQueries[0] ?? null;

  const nav = (
    <Box sx={{ p: 2 }}>
      <Typography variant="caption" color="text.secondary" fontWeight={900} sx={{ px: 1.5 }}>
        NAVIGATION
      </Typography>
      <List sx={{ mt: 1 }}>
        {NAV_ITEMS.map((item) => (
          <ListItem key={item.id} disablePadding sx={{ mb: 0.75 }}>
            <ListItemButton
              selected={activePage === item.id}
              onClick={() => {
                setActivePage(item.id);
                setMobileOpen(false);
              }}
              sx={{
                borderRadius: 2,
                '&.Mui-selected': {
                  bgcolor: '#dbeafe',
                  color: 'primary.main',
                  '& .MuiListItemIcon-root': { color: 'primary.main' },
                },
              }}
            >
              <ListItemIcon sx={{ minWidth: 38 }}>{item.icon}</ListItemIcon>
              <ListItemText primary={item.label} primaryTypographyProps={{ fontWeight: 900, fontSize: 14 }} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
      <Paper variant="outlined" sx={{ mt: 3, p: 2, bgcolor: '#f8fafc' }}>
        <Typography fontWeight={900} fontSize={13}>
          Recruiter signal
        </Typography>
        <Typography variant="caption" color="text.secondary" sx={{ mt: 0.75, display: 'block', lineHeight: 1.55 }}>
          Evaluation, comparison, traces, and markdown-aware outputs make this read like a RAG platform.
        </Typography>
      </Paper>
    </Box>
  );

  const renderRunner = () => {
    const answerText = runResult?.result.answer || streamingAnswer;
    const retrieved = runResult?.result.retrieved ?? [];
    const ranked = runResult?.result.ranked ?? [];
    const promptOutput = firstStepOutput(runResult, 'prompt');
    const promptText = String(promptOutput?.prompt ?? runResult?.result.prompt ?? '');
    const metricsOutput = firstStepOutput(runResult, 'metrics');
    const metrics = (metricsOutput?.metrics as Record<string, unknown> | undefined)
      ?? runResult?.result.evaluation
      ?? {};
    const initializedSteps = initialized?.steps ?? [];

    const setupSummary = (
      <Card>
        <CardHeader
          title="Initialize Pipeline"
          subheader="Build the selected indexes and prepare the pipeline before sending queries."
        />
        <CardContent>
          <Stack spacing={2}>
            {!isSelectionComplete ? (
              <Alert severity="info" variant="outlined">
                Pick required groups before initialization. Missing: {missingRequiredGroups.join(', ') || 'none'}.
              </Alert>
            ) : previewError ? (
              <Alert severity="error">{previewError}</Alert>
            ) : null}

            <Box>
              <Typography variant="caption" color="text.secondary" fontWeight={900}>INIT PIPELINE</Typography>
              <Box sx={{ mt: 1 }}>
                {preview ? renderPlanFlow(preview.plan.init_pipeline) : <EmptyState text="Waiting for pipeline preview." />}
              </Box>
            </Box>
            <Box>
              <Typography variant="caption" color="text.secondary" fontWeight={900}>QUERY PIPELINE</Typography>
              <Box sx={{ mt: 1 }}>
                {preview ? renderPlanFlow(preview.plan.pipeline) : <EmptyState text="Waiting for pipeline preview." />}
              </Box>
            </Box>

            {initialized ? (
              <Alert severity="success" variant="outlined">
                Initialized {initialized.document_count} documents from {initialized.source_count} source(s).
              </Alert>
            ) : null}

            {initializedSteps.length > 0 ? (
              <Stack spacing={1}>
                {initializedSteps.map((step) => (
                  <Paper key={step.step_id} variant="outlined" sx={{ p: 1.5, bgcolor: '#f8fafc' }}>
                    <Stack direction="row" spacing={1.5} alignItems="center">
                      <Box sx={{ color: 'success.main' }}>{stepIcon(step.status)}</Box>
                      <Box flex={1} minWidth={0}>
                        <Typography fontWeight={900}>{step.step_name}</Typography>
                        <Typography variant="caption" color="text.secondary">
                          {formatMs(step.latency_ms)} · {step.component}{step.summary ? ` · ${step.summary}` : ''}
                        </Typography>
                      </Box>
                      <Chip size="small" color="success" variant="outlined" label={step.status} />
                    </Stack>
                  </Paper>
                ))}
              </Stack>
            ) : null}

            {previewing ? <LinearProgress /> : null}
            {preview?.warnings.length ? (
              <Stack spacing={1}>
                {preview.warnings.map((warning) => <Alert key={warning} severity="warning" variant="outlined">{warning}</Alert>)}
              </Stack>
            ) : null}

            <Button
              variant="contained"
              size="large"
              startIcon={initializing ? <CircularProgress size={18} color="inherit" /> : <CheckCircleIcon />}
              disabled={!canInitialize}
              onClick={() => void handleInitializePipeline()}
            >
              {initializing ? 'Initializing...' : initialized ? 'Re-initialize Pipeline' : 'Initialize Pipeline'}
            </Button>
          </Stack>
        </CardContent>
      </Card>
    );

    const configPanel = (
      <Card>
        <CardHeader title="Pipeline Config" subheader="Select sources, template, components, and runtime options." />
        <CardContent>
          <Stack spacing={2}>
            <TextField
              select
              label="Template"
              value={selectedTemplateId}
              onChange={(event) => {
                const template = templates.find((item) => item.id === event.target.value);
                if (template) applyTemplate(template);
                if (!template) setSelectedTemplateId('');
              }}
              size="small"
            >
              <MenuItem value="">Custom Draft</MenuItem>
              {templates.map((template) => (
                <MenuItem key={template.id} value={template.id}>{template.name}</MenuItem>
              ))}
            </TextField>
            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
              {(selectedTemplate?.tags ?? ['Custom']).map((tag) => (
                <Chip key={tag} size="small" label={tag} color={tag === 'Baseline' ? 'default' : 'primary'} />
              ))}
              <Chip size="small" label={`top_k: ${topK}`} />
              {isStreamingPipeline ? <Chip size="small" label="Streaming" color="warning" /> : null}
            </Stack>
            <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '120px 1fr 1fr' }, gap: 1 }}>
              <TextField
                type="number"
                label="Top K"
                value={topK}
                onChange={(event) => setTopK(Number(event.target.value || 5))}
                inputProps={{ min: 1, max: 100 }}
                size="small"
              />
              <Stack direction="row" spacing={0.5} alignItems="center">
                <TextField
                  select
                  label="Prompt"
                  value={prompts.some((item) => item.name === templateName) ? templateName : ''}
                  onChange={(event) => setTemplateName(event.target.value)}
                  size="small"
                  fullWidth
                >
                  {templateName && !prompts.some((item) => item.name === templateName) ? (
                    <MenuItem value="">{templateName}</MenuItem>
                  ) : null}
                  {prompts.length === 0 ? (
                    <MenuItem value="" disabled>No prompts available</MenuItem>
                  ) : null}
                  {prompts.map((prompt) => (
                    <MenuItem key={prompt.name} value={prompt.name}>
                      {prompt.label} · {prompt.name}
                    </MenuItem>
                  ))}
                </TextField>
                <Tooltip title="Add new prompt">
                  <IconButton color="primary" onClick={openPromptDialog} size="small">
                    <AddIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Stack>
              <TextField
                label="Parser"
                value={parserName}
                onChange={(event) => setParserName(event.target.value)}
                size="small"
              />
            </Box>
            {selectedPrompt ? (
              <Box>
                <Stack
                  direction="row"
                  alignItems="center"
                  justifyContent="space-between"
                  spacing={1}
                  flexWrap="wrap"
                  useFlexGap
                  sx={{ mb: 0.75 }}
                >
                  <Typography variant="caption" color="text.secondary" fontWeight={900}>
                    RAW PROMPT · {selectedPrompt.name}
                  </Typography>
                  {selectedPrompt.variables.length ? (
                    <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap>
                      {selectedPrompt.variables.map((variable) => (
                        <Chip key={variable} size="small" variant="outlined" label={`{${variable}}`} />
                      ))}
                    </Stack>
                  ) : null}
                </Stack>
                <CodeBlock>{selectedPrompt.template}</CodeBlock>
              </Box>
            ) : null}
            <Divider />
            <Stack spacing={1.2}>
              <Stack direction="row" alignItems="center" justifyContent="space-between" spacing={1}>
                <Typography fontWeight={900}>Dataset / Sources</Typography>
                <Stack direction="row" spacing={1}>
                  <Button component="label" variant="outlined" size="small" startIcon={<CloudUploadIcon />} disabled={working || initializing}>
                    Upload
                    <input type="file" hidden multiple accept=".md,.markdown,.txt,.log" onChange={handleUpload} />
                  </Button>
                  <Button size="small" startIcon={<RefreshIcon />} onClick={() => void reloadSources()}>
                    Refresh
                  </Button>
                </Stack>
              </Stack>
              <Stack direction={{ xs: 'column', md: 'row' }} spacing={1}>
                <TextField
                  size="small"
                  label="Local dataset/source path"
                  value={pathInput}
                  onChange={(event) => setPathInput(event.target.value)}
                  fullWidth
                />
                <Button variant="outlined" onClick={handleRegisterPath} disabled={working || initializing || !pathInput.trim()}>
                  Add Path
                </Button>
              </Stack>
              <Stack direction={{ xs: 'column', md: 'row' }} spacing={1}>
                <TextField
                  size="small"
                  label="Public repo URL"
                  value={repoUrlInput}
                  onChange={(event) => setRepoUrlInput(event.target.value)}
                  placeholder="https://github.com/owner/repo.git"
                  fullWidth
                />
                <TextField
                  size="small"
                  label="Branch"
                  value={repoBranchInput}
                  onChange={(event) => setRepoBranchInput(event.target.value)}
                  placeholder="default"
                  sx={{ width: { xs: '100%', md: 140 } }}
                />
                <Button
                  variant="outlined"
                  onClick={handleRegisterRepo}
                  disabled={working || initializing || !repoUrlInput.trim()}
                  sx={{ minWidth: 110 }}
                >
                  Add Repo
                </Button>
              </Stack>
              {loadingSources ? (
                <Stack direction="row" spacing={1} alignItems="center">
                  <CircularProgress size={16} />
                  <Typography variant="body2">Loading sources...</Typography>
                </Stack>
              ) : sources.length === 0 ? (
                <EmptyState text="No sources registered yet." />
              ) : (
                <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 320 }}>
                  <Table size="small" stickyHeader>
                    <TableHead>
                      <TableRow>
                        <TableCell padding="checkbox">Use</TableCell>
                        <TableCell>Name</TableCell>
                        <TableCell>Type</TableCell>
                        <TableCell>Location</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {sources.map((source) => (
                        <TableRow
                          key={source.id}
                          hover
                          selected={selectedSourceIds.includes(source.id)}
                          onClick={() => handleSelectSource(source.id, !selectedSourceIds.includes(source.id))}
                          sx={{ cursor: 'pointer' }}
                        >
                          <TableCell padding="checkbox">
                            <Checkbox checked={selectedSourceIds.includes(source.id)} />
                          </TableCell>
                          <TableCell>
                            <Typography fontWeight={900}>{source.name}</Typography>
                            {source.commit_sha ? (
                              <Typography variant="caption" color="text.secondary">{source.commit_sha.slice(0, 8)}</Typography>
                            ) : null}
                          </TableCell>
                          <TableCell><Chip size="small" label={source.source_type} /></TableCell>
                          <TableCell sx={{ maxWidth: 460 }}>
                            <Typography variant="body2" noWrap>
                              {source.source_type === 'repository'
                                ? `${source.repo_url ?? source.path} · ${source.branch ?? 'default'}`
                                : source.path}
                            </Typography>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
              <Typography variant="caption" color="text.secondary">
                Selected: {selectedSourceIds.length} of {sources.length}
              </Typography>
            </Stack>
          </Stack>
        </CardContent>
      </Card>
    );

    const advancedPanel = (
      <Card>
        <CardHeader title="Steps & Components" subheader="Choose the pipeline steps before initialization." />
        <CardContent>
          {catalog ? (
            <Stack spacing={2}>
              <Tabs
                value={activeComponentTab}
                onChange={(_, value: string) => setActiveComponentTab(value)}
                variant="scrollable"
                scrollButtons="auto"
              >
                {catalog.groups.map((group) => (
                  <Tab key={group.id} value={group.id} label={group.label} />
                ))}
              </Tabs>
              {catalog.groups.map((group) => {
                if (group.id !== activeComponentTab) return null;
                return (
                  <Stack key={group.id} spacing={1.2}>
                    <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                      <Chip size="small" label={group.multi_select ? 'multi-select' : 'choose one'} />
                      <Chip size="small" label={group.required ? 'required' : 'optional'} color={group.required ? 'error' : 'default'} variant="outlined" />
                    </Stack>
                    <Typography variant="body2" color="text.secondary">{group.description}</Typography>
                    {group.id === 'retrieval'
                      && selection.retrieval.includes('external_retriever')
                      && selection.retrieval.some((id) => PRIMARY_RETRIEVERS.has(id)) ? (
                      <Alert severity="info" variant="outlined">
                        external_retriever runs when the primary returns fewer than {EXTERNAL_FALLBACK_THRESHOLD} chunks.
                      </Alert>
                    ) : null}
                    {group.subcomponents.map((sub) => renderSubcomponent(group, sub))}
                  </Stack>
                );
              })}
            </Stack>
          ) : (
            <EmptyState text="Component catalog unavailable." />
          )}
        </CardContent>
      </Card>
    );

    return (
      <Box>
        <PageHeader
          title="Pipeline Runner"
          description="Step 1 initializes sources, indexes, and selected pipeline config. Step 2 sends queries and shows intermediate plus final outputs."
          actions={
            runnerStep === 'setup' ? (
              <Button
                variant="contained"
                startIcon={initializing ? <CircularProgress size={18} color="inherit" /> : <CheckCircleIcon />}
                onClick={() => void handleInitializePipeline()}
                disabled={!canInitialize}
              >
                {initializing ? 'Initializing...' : initialized ? 'Re-initialize' : 'Initialize Pipeline'}
              </Button>
            ) : (
              <>
                <Button variant="outlined" onClick={() => setRunnerStep('setup')}>
                  Back to Setup
                </Button>
                <Button
                  variant="contained"
                  startIcon={working ? <CircularProgress size={18} color="inherit" /> : <PlayCircleIcon />}
                  onClick={handleRun}
                  disabled={!canRun}
                >
                  {streaming ? 'Streaming...' : working ? 'Running...' : 'Run Query'}
                </Button>
              </>
            )
          }
        />

        {error ? <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert> : null}

        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 2, mb: 2.5 }}>
          <Paper
            variant="outlined"
            sx={{ p: 2, bgcolor: runnerStep === 'setup' ? '#eff6ff' : '#fff', borderColor: runnerStep === 'setup' ? 'primary.main' : 'divider' }}
          >
            <Stack direction="row" spacing={1.5} alignItems="center">
              <Chip label="1" color={initialized ? 'success' : 'primary'} />
              <Box>
                <Typography fontWeight={900}>Select steps, sources, and initialize</Typography>
                <Typography variant="body2" color="text.secondary">
                  {initialized ? 'Initialized. You can still edit and re-initialize.' : 'Choose pipeline config and selected datasets/sources.'}
                </Typography>
              </Box>
            </Stack>
          </Paper>
          <Paper
            variant="outlined"
            sx={{ p: 2, bgcolor: runnerStep === 'query' ? '#eff6ff' : '#fff', borderColor: runnerStep === 'query' ? 'primary.main' : 'divider', opacity: initialized ? 1 : 0.65 }}
          >
            <Stack direction="row" spacing={1.5} alignItems="center">
              <Chip label="2" color={runnerStep === 'query' ? 'primary' : 'default'} />
              <Box>
                <Typography fontWeight={900}>Query and inspect results</Typography>
                <Typography variant="body2" color="text.secondary">
                  Pass a query through the initialized pipeline and inspect intermediate plus final outputs.
                </Typography>
              </Box>
            </Stack>
          </Paper>
        </Box>

        {runnerStep === 'setup' ? (
          <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', xl: 'minmax(0, 1.15fr) minmax(360px, 0.85fr)' }, gap: 2.5 }}>
            <Stack spacing={2.5}>
              {configPanel}
              {advancedPanel}
            </Stack>
            {setupSummary}
          </Box>
        ) : (
          <Stack spacing={2.5}>
            <Card>
              <CardHeader
                title="Initialized Pipeline"
                subheader="This query step uses the current initialized source/index state."
                action={<Button size="small" variant="outlined" onClick={() => setRunnerStep('setup')}>Edit Setup</Button>}
              />
              <CardContent>
                <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(4, minmax(0, 1fr))' }, gap: 1.5 }}>
                  <MetricCard label="Sources" value={initialized?.source_count ?? 0} sub={`${selectedSourceIds.length} selected`} />
                  <MetricCard label="Documents" value={initialized?.document_count ?? 0} sub={initialized?.init_skipped ? 'cache hit' : 'loaded'} />
                  <MetricCard label="Init Steps" value={initialized?.steps.length ?? 0} sub="completed setup steps" />
                  <MetricCard label="Pipeline" value={selectedTemplate?.name ?? 'Custom'} sub={isStreamingPipeline ? 'streaming generation' : 'standard generation'} />
                </Box>
              </CardContent>
            </Card>

            <Card>
              <CardHeader title="Query / Input" subheader="Question sent to the initialized pipeline." />
              <CardContent>
                <TextField
                  fullWidth
                  multiline
                  minRows={3}
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                />
                <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap sx={{ mt: 2 }}>
                  <Chip size="small" color="success" label="Pipeline initialized" />
                  <Chip size="small" color="primary" label={`${selectedSourceIds.length} selected sources`} />
                  <Chip size="small" label="Mode: Query Run" />
                </Stack>
                <Button
                  variant="contained"
                  startIcon={working ? <CircularProgress size={18} color="inherit" /> : <PlayCircleIcon />}
                  onClick={handleRun}
                  disabled={!canRun}
                  sx={{ mt: 2 }}
                >
                  {streaming ? 'Streaming...' : working ? 'Running...' : 'Run Query'}
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader title="Intermediate Progress" subheader="Query-time step outputs from the backend." />
              <CardContent>
                <Stack spacing={1.2}>
                  {displaySteps.length === 0 ? <EmptyState text="Run a query to see intermediate steps." /> : displaySteps.map((step) => (
                    <Paper key={step.step_id} variant="outlined" sx={{ p: 1.5, bgcolor: '#f8fafc' }}>
                      <Stack direction="row" spacing={1.5} alignItems="center">
                        <Box sx={{ color: step.status === 'completed' ? 'success.main' : step.status === 'error' ? 'error.main' : 'warning.main' }}>
                          {stepIcon(step.status)}
                        </Box>
                        <Box flex={1} minWidth={0}>
                          <Typography fontWeight={900}>{step.step_name}</Typography>
                          <Typography variant="caption" color="text.secondary">
                            {formatMs(step.latency_ms)} · {step.output_type}{step.summary ? ` · ${step.summary}` : ''}
                          </Typography>
                        </Box>
                        <Chip size="small" label={step.status} color={step.status === 'completed' ? 'success' : step.status === 'error' ? 'error' : 'warning'} variant="outlined" />
                      </Stack>
                    </Paper>
                  ))}
                </Stack>
              </CardContent>
            </Card>

            <Card>
              <CardHeader title="Result Explorer" subheader="Final and intermediate outputs for the query run." />
              <CardContent>
                <Tabs value={runnerTab} onChange={(_, value: number) => setRunnerTab(value)} variant="scrollable" scrollButtons="auto">
                  <Tab label="Answer" />
                  <Tab label="Retrieved Chunks" />
                  <Tab label="Prompt" />
                  <Tab label="Metrics" />
                  <Tab label="Trace" />
                  <Tab label="Raw JSON" />
                </Tabs>
                <Divider sx={{ mb: 2 }} />

                {runnerTab === 0 ? (
                  <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1.25fr 0.75fr' }, gap: 2 }}>
                    <Paper variant="outlined" sx={{ p: 2 }}>
                      <Typography variant="h6">Final Answer</Typography>
                      <Box sx={{ mt: 1 }}>
                        {answerText ? (
                          <MarkdownViewer>{answerText}</MarkdownViewer>
                        ) : (
                          <Typography color="text.secondary">
                            {streaming ? 'Waiting for first token...' : 'Run a query to generate an answer.'}
                          </Typography>
                        )}
                      </Box>
                    </Paper>
                    <Paper variant="outlined" sx={{ p: 2 }}>
                      <Typography variant="h6">Answer Metadata</Typography>
                      <Stack spacing={1.1} sx={{ mt: 1.5 }}>
                        <MetaRow label="Pipeline" value={runResult?.pipeline_name ?? runResult?.result.pipeline_name ?? 'n/a'} />
                        <MetaRow label="Run ID" value={runResult?.run_id ?? runResult?.result.run_id ?? 'n/a'} />
                        <MetaRow label="Retrieved" value={retrieved.length} />
                        <MetaRow label="Ranked" value={ranked.length} />
                        <MetaRow label="Latency" value={formatMs(displaySteps.reduce((sum, step) => sum + Number(step.latency_ms ?? 0), 0))} />
                      </Stack>
                    </Paper>
                  </Box>
                ) : null}

                {runnerTab === 1 ? (
                  retrieved.length === 0 ? <EmptyState text="No retrieved chunks yet." /> : (
                    <TableContainer component={Paper} variant="outlined">
                      <Table size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell>Rank</TableCell>
                            <TableCell>Source</TableCell>
                            <TableCell>Chunk</TableCell>
                            <TableCell>Score</TableCell>
                            <TableCell>Preview</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {retrieved.map((chunk, index) => (
                            <TableRow key={`${chunk.id}-${index}`}>
                              <TableCell>{index + 1}</TableCell>
                              <TableCell>{chunkSource(chunk)}</TableCell>
                              <TableCell><Chip size="small" label={chunk.id || `chunk_${index + 1}`} /></TableCell>
                              <TableCell>{chunk.score.toFixed(3)}</TableCell>
                              <TableCell sx={{ minWidth: 320 }}>
                                <MarkdownViewer compact>{truncateText(chunk.text, 520)}</MarkdownViewer>
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  )
                ) : null}

                {runnerTab === 2 ? (
                  promptText ? <CodeBlock>{promptText}</CodeBlock> : <EmptyState text="Prompt output appears after a run reaches prompt_builder." />
                ) : null}

                {runnerTab === 3 ? (
                  Object.keys(metrics).length === 0 ? <EmptyState text="No evaluation metrics were produced by this pipeline." /> : (
                    <Stack spacing={1.5}>
                      {Object.entries(metrics).map(([name, value]) => (
                        <MetricBar key={name} label={name} value={value} />
                      ))}
                    </Stack>
                  )
                ) : null}

                {runnerTab === 4 ? (
                  displaySteps.length === 0 ? <EmptyState text="Trace appears after a query run." /> : (
                    <Stack spacing={1}>
                      {displaySteps.map((step) => (
                        <Paper key={`trace-${step.step_id}`} variant="outlined" sx={{ p: 1.5, bgcolor: '#f8fafc' }}>
                          <Stack direction="row" justifyContent="space-between" spacing={2}>
                            <Typography fontWeight={900}>{step.component}</Typography>
                            <Chip size="small" color={step.status === 'completed' ? 'success' : 'warning'} label={formatMs(step.latency_ms)} />
                          </Stack>
                          <Typography variant="caption" color="text.secondary">{step.step_name} · {step.output_type}</Typography>
                        </Paper>
                      ))}
                    </Stack>
                  )
                ) : null}

                {runnerTab === 5 ? (
                  <CodeBlock>{safeJson(runResult ?? initialized ?? preview ?? { message: 'No run yet.' })}</CodeBlock>
                ) : null}
              </CardContent>
            </Card>
          </Stack>
        )}
      </Box>
    );
  };

  const renderCompare = () => (
    <Box>
      <PageHeader
        title="Compare Pipelines"
        description="Compare answers, retrieved context, metrics, latency, and component outputs across historical RAG experiments."
        actions={
          <Button
            variant="contained"
            startIcon={loadingExperiments ? <CircularProgress size={18} color="inherit" /> : <CompareArrowsIcon />}
            disabled={!selectedExperiment || loadingExperiments}
            onClick={() => void handleRunComparison()}
          >
            Run Comparison
          </Button>
        }
      />
      <Stack spacing={2.5}>
        <TextField
          select
          label="Experiment run"
          value={selectedExperimentKey}
          onChange={(event) => setSelectedExperimentKey(event.target.value)}
          sx={{ maxWidth: 520 }}
          size="small"
        >
          {experiments.map((experiment) => (
            <MenuItem key={`${experiment.name}::${experiment.run_id}`} value={`${experiment.name}::${experiment.run_id}`}>
              {experiment.name} · {experiment.run_id}
            </MenuItem>
          ))}
        </TextField>

        {loadingExperiments ? <LinearProgress /> : null}

        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(4, minmax(0, 1fr))' }, gap: 2 }}>
          <MetricCard label="Best Quality" value={bestQuality ?? 'n/a'} sub={qualityMetric ? `${qualityMetric}` : 'No metrics'} />
          <MetricCard label="Fastest" value={fastest ?? 'n/a'} sub="lowest latency_ms" />
          <MetricCard label="Variants" value={variantNames.length} sub={`${selectedExperiment?.queries ?? 0} queries`} />
          <MetricCard label="Best Tradeoff" value={comparison.best?.answer_relevancy ?? bestQuality ?? 'n/a'} sub="quality and speed scan" />
        </Box>

        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', xl: '1fr 1fr' }, gap: 2.5 }}>
          <Card>
            <CardHeader title="Pipeline Summary" subheader="High-level experiment comparison." />
            <CardContent>
              {variantNames.length === 0 ? <EmptyState text="No comparison data found." /> : (
                <TableContainer component={Paper} variant="outlined">
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Pipeline</TableCell>
                        {comparisonMetrics.map((metric) => <TableCell key={metric}>{metric}</TableCell>)}
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {variantNames.map((variant) => (
                        <TableRow key={variant}>
                          <TableCell>
                            <Typography fontWeight={900}>{variant}</Typography>
                            <Typography variant="caption" color="text.secondary">variant</Typography>
                          </TableCell>
                          {comparisonMetrics.map((metric) => {
                            const isBest = comparison.best?.[metric] === variant;
                            return (
                              <TableCell
                                key={`${variant}-${metric}`}
                                sx={{
                                  bgcolor: isBest ? '#dcfce7' : undefined,
                                  color: isBest ? '#065f46' : undefined,
                                  fontWeight: isBest ? 900 : undefined,
                                }}
                              >
                                {formatMetric(variants[variant]?.[metric]?.value)}
                              </TableCell>
                            );
                          })}
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader title="Quality vs Latency" subheader="Tradeoff view for pipeline selection." />
            <CardContent>
              <Stack spacing={1.6}>
                {variantNames.map((variant) => {
                  const quality = qualityMetric ? variants[variant]?.[qualityMetric]?.value : undefined;
                  const latency = variants[variant]?.latency_ms?.value;
                  return (
                    <Paper key={variant} variant="outlined" sx={{ p: 1.5 }}>
                      <Stack direction="row" justifyContent="space-between" spacing={1}>
                        <Typography fontWeight={900}>{variant}</Typography>
                        <Typography variant="caption" color="text.secondary">{formatMs(typeof latency === 'number' ? latency : null)}</Typography>
                      </Stack>
                      <MetricBar label={qualityMetric ?? 'quality'} value={quality} compact />
                    </Paper>
                  );
                })}
              </Stack>
            </CardContent>
          </Card>
        </Box>

        <Card>
          <CardHeader
            title="Query Deep Dive"
            subheader={selectedQuery ? `Query: ${selectedQuery.question}` : 'Select an experiment with recorded query runs.'}
            action={<Chip color="primary" label={`${experimentQueries.length} queries`} />}
          />
          <CardContent>
            <Tabs value={compareTab} onChange={(_, value: number) => setCompareTab(value)} variant="scrollable" scrollButtons="auto">
              <Tab label="Answer" />
              <Tab label="Retrieval" />
              <Tab label="Metrics" />
              <Tab label="Heatmap" />
              <Tab label="Component Matrix" />
            </Tabs>
            <Divider sx={{ mb: 2 }} />
            {!selectedQuery ? <EmptyState text="No query records found." /> : null}
            {selectedQuery && compareTab === 0 ? (
              <Stack spacing={2}>
                <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: 'repeat(3, minmax(0, 1fr))' }, gap: 2 }}>
                  {Object.entries(selectedQuery.variants).slice(0, 3).map(([variant, record]) => (
                    <Paper key={variant} variant="outlined" sx={{ p: 2 }}>
                      <Typography variant="h6">{variant}</Typography>
                      <Box sx={{ mt: 1 }}>
                        <MarkdownViewer>{record.answer || record.error || 'No answer recorded.'}</MarkdownViewer>
                      </Box>
                      <MetaRow label="Latency" value={formatMs(record.latency_ms)} />
                    </Paper>
                  ))}
                </Box>
                <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 2 }}>
                  <Alert severity="warning" variant="outlined">Compare answer omissions against the ground truth below.</Alert>
                  <Alert severity="success" variant="outlined">Use this view to spot grounding and specificity improvements.</Alert>
                </Box>
                {selectedQuery.ground_truth ? (
                  <Paper variant="outlined" sx={{ p: 2, bgcolor: '#f8fafc' }}>
                    <Typography fontWeight={900}>Ground truth</Typography>
                    <Box sx={{ mt: 1 }}>
                      <MarkdownViewer>{selectedQuery.ground_truth}</MarkdownViewer>
                    </Box>
                  </Paper>
                ) : null}
              </Stack>
            ) : null}

            {selectedQuery && compareTab === 1 ? (
              <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: 'repeat(3, minmax(0, 1fr))' }, gap: 2 }}>
                {Object.entries(selectedQuery.variants).slice(0, 3).map(([variant, record]) => (
                  <Paper key={variant} variant="outlined" sx={{ p: 2 }}>
                    <Typography variant="h6">{variant}</Typography>
                    <Stack spacing={1.2} sx={{ mt: 1.5 }}>
                      {(record.contexts ?? []).slice(0, 3).map((context, index) => (
                        <Paper key={`${variant}-${index}`} variant="outlined" sx={{ p: 1.5, bgcolor: '#f8fafc' }}>
                          <Stack direction="row" justifyContent="space-between">
                            <Typography fontWeight={900}>Rank {index + 1}</Typography>
                            <Chip size="small" label="context" />
                          </Stack>
                          <Box sx={{ mt: 1 }}>
                            <MarkdownViewer compact>{truncateText(context, 420)}</MarkdownViewer>
                          </Box>
                        </Paper>
                      ))}
                    </Stack>
                  </Paper>
                ))}
              </Box>
            ) : null}

            {compareTab === 2 ? (
              <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 2 }}>
                {comparisonMetrics.slice(0, 4).map((metric) => (
                  <Paper key={metric} variant="outlined" sx={{ p: 2 }}>
                    <Typography fontWeight={900}>{metric}</Typography>
                    <Stack spacing={1.2} sx={{ mt: 1.5 }}>
                      {variantNames.map((variant) => (
                        <MetricBar key={`${metric}-${variant}`} label={variant} value={variants[variant]?.[metric]?.value} compact />
                      ))}
                    </Stack>
                  </Paper>
                ))}
              </Box>
            ) : null}

            {compareTab === 3 ? (
              <TableContainer component={Paper} variant="outlined">
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Metric</TableCell>
                      {variantNames.map((variant) => <TableCell key={variant}>{variant}</TableCell>)}
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {comparisonMetrics.map((metric) => (
                      <TableRow key={metric}>
                        <TableCell>{metric}</TableCell>
                        {variantNames.map((variant) => {
                          const value = variants[variant]?.[metric]?.value;
                          const percent = metricPercent(value);
                          return (
                            <TableCell key={`${metric}-${variant}`}>
                              <Chip
                                label={formatMetric(value)}
                                color={percent >= 85 ? 'success' : percent >= 70 ? 'warning' : 'error'}
                                variant="outlined"
                              />
                            </TableCell>
                          );
                        })}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            ) : null}

            {compareTab === 4 ? (
              <TableContainer component={Paper} variant="outlined">
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Component</TableCell>
                      {variantNames.map((variant) => <TableCell key={variant}>{variant}</TableCell>)}
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {['retrieval', 'ranking', 'generation', 'evaluation', 'latency'].map((component) => (
                      <TableRow key={component}>
                        <TableCell><Typography fontWeight={900}>{component}</Typography></TableCell>
                        {variantNames.map((variant) => (
                          <TableCell key={`${component}-${variant}`}>
                            {component === 'latency'
                              ? formatMs(variants[variant]?.latency_ms?.value)
                              : <Button size="small" variant="outlined">View Output</Button>}
                          </TableCell>
                        ))}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            ) : null}
          </CardContent>
        </Card>
      </Stack>
    </Box>
  );

  const renderExperiments = () => {
    const selectedConfig = experimentConfigs.find((config) => config.file === selectedExperimentConfig);
    const canRunSelectedConfig = experimentRunMode === 'existing'
      ? Boolean(selectedExperimentConfig) && Boolean(selectedConfig?.valid)
      : experimentYamlText.trim().length > 0;

    return (
    <Box>
      <PageHeader
        title="Experiments"
        description="Track historical RAG experiments, compare versions, and inspect regressions over time."
        actions={<Button variant="contained" startIcon={<ScienceIcon />} onClick={handleNewExperiment}>Run Experiment</Button>}
      />
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', xl: '1fr 1fr' }, gap: 2.5 }}>
        <Card>
          <CardHeader title="Recent Experiments" subheader="Experiment-level history." />
          <CardContent>
            {experiments.length === 0 ? <EmptyState text="No experiments found in data/experiments." /> : (
              <TableContainer component={Paper} variant="outlined">
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Experiment</TableCell>
                      <TableCell>Pipelines</TableCell>
                      <TableCell>Queries</TableCell>
                      <TableCell>Best</TableCell>
                      <TableCell>Status</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {experiments.map((experiment) => (
                      <TableRow
                        key={`${experiment.name}-${experiment.run_id}`}
                        hover
                        selected={`${experiment.name}::${experiment.run_id}` === selectedExperimentKey}
                        onClick={() => setSelectedExperimentKey(`${experiment.name}::${experiment.run_id}`)}
                        sx={{ cursor: 'pointer' }}
                      >
                        <TableCell>
                          <Typography fontWeight={900}>{experiment.name}</Typography>
                          <Typography variant="caption" color="text.secondary">{experiment.run_id}</Typography>
                        </TableCell>
                        <TableCell>{experiment.variants}</TableCell>
                        <TableCell>{experiment.queries}</TableCell>
                        <TableCell>{experiment.best.faithfulness ?? experiment.best.answer_relevancy ?? 'n/a'}</TableCell>
                        <TableCell><Chip size="small" color={experiment.status === 'completed' ? 'success' : 'warning'} label={experiment.status} /></TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader title="Regression Alerts" subheader="Automatically detected quality changes." />
          <CardContent>
            <Stack spacing={1.5}>
              {bestQuality ? (
                <Alert severity="success" variant="outlined">
                  {bestQuality} leads {qualityMetric} in the selected run.
                </Alert>
              ) : (
                <Alert severity="info" variant="outlined">Select an experiment to compute quality leaders.</Alert>
              )}
              {fastest ? (
                <Alert severity="warning" variant="outlined">
                  {fastest} is fastest by latency_ms. Compare quality before choosing it.
                </Alert>
              ) : null}
              <Paper variant="outlined" sx={{ p: 2, bgcolor: '#f8fafc' }}>
                <Typography fontWeight={900}>Selected run</Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  {selectedExperiment ? `${selectedExperiment.name} / ${selectedExperiment.run_id}` : 'No experiment selected.'}
                </Typography>
              </Paper>
            </Stack>
          </CardContent>
        </Card>
      </Box>

      <Dialog open={experimentRunOpen} onClose={() => !experimentRunning && setExperimentRunOpen(false)} fullWidth maxWidth="md">
        <DialogTitle>Run Experiment</DialogTitle>
        <DialogContent dividers>
          <Stack spacing={2.2}>
            {experimentRunError ? <Alert severity="error">{experimentRunError}</Alert> : null}
            {experimentConfigValidation.length > 0 ? (
              <Stack spacing={1}>
                {experimentConfigValidation.map((message) => (
                  <Alert key={message} severity="error" variant="outlined">{message}</Alert>
                ))}
              </Stack>
            ) : null}

            <RadioGroup
              row
              value={experimentRunMode}
              onChange={(event) => setExperimentRunMode(event.target.value as ExperimentRunMode)}
            >
              <FormControlLabel value="existing" control={<Radio />} label="Run existing config" />
              <FormControlLabel value="new" control={<Radio />} label="Create/upload config" />
            </RadioGroup>

            {experimentRunMode === 'existing' ? (
              <Stack spacing={1.5}>
                {experimentConfigLoading ? <LinearProgress /> : null}
                <TextField
                  select
                  label="Experiment config"
                  value={selectedExperimentConfig}
                  onChange={(event) => setSelectedExperimentConfig(event.target.value)}
                  size="small"
                  fullWidth
                >
                  {experimentConfigs.map((config) => (
                    <MenuItem key={config.file} value={config.file} disabled={!config.valid}>
                      {config.file}{config.name ? ` · ${config.name}` : ''}{config.valid ? '' : ' · invalid'}
                    </MenuItem>
                  ))}
                </TextField>
                {selectedConfig ? (
                  <Paper variant="outlined" sx={{ p: 1.5, bgcolor: '#f8fafc' }}>
                    <Stack spacing={0.8}>
                      <MetaRow label="Dataset" value={selectedConfig.dataset ?? 'n/a'} />
                      <MetaRow label="Sources" value={selectedConfig.sources ?? 'n/a'} />
                      <MetaRow label="Variants" value={selectedConfig.variants.length} />
                      <MetaRow label="Metrics" value={selectedConfig.metrics.join(', ') || 'none'} />
                      {selectedConfig.error ? <Alert severity="error">{selectedConfig.error}</Alert> : null}
                    </Stack>
                  </Paper>
                ) : (
                  <EmptyState text="No valid experiment configs found." />
                )}
              </Stack>
            ) : (
              <Stack spacing={1.5}>
                <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1}>
                  <Button component="label" variant="outlined" startIcon={<CloudUploadIcon />} disabled={experimentRunning}>
                    Upload YAML
                    <input type="file" hidden accept=".yaml,.yml,text/yaml" onChange={(event) => void handleExperimentYamlUpload(event)} />
                  </Button>
                  <TextField
                    label="Save as"
                    value={experimentConfigFileName}
                    onChange={(event) => setExperimentConfigFileName(event.target.value)}
                    placeholder="my_experiment.yaml"
                    size="small"
                    fullWidth
                  />
                </Stack>
                <FormControlLabel
                  control={<Checkbox checked={experimentOverwrite} onChange={(event) => setExperimentOverwrite(event.target.checked)} />}
                  label="Overwrite existing config"
                />
                <TextField
                  label="Experiment YAML"
                  value={experimentYamlText}
                  onChange={(event) => {
                    setExperimentYamlText(event.target.value);
                    setExperimentConfigValidation([]);
                  }}
                  multiline
                  minRows={12}
                  fullWidth
                  inputProps={{ spellCheck: false }}
                />
                <Button
                  variant="outlined"
                  onClick={() => void handleValidateExperimentConfig()}
                  disabled={experimentRunning || !experimentYamlText.trim()}
                  sx={{ alignSelf: 'flex-start' }}
                >
                  Validate Config
                </Button>
              </Stack>
            )}
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setExperimentRunOpen(false)} disabled={experimentRunning}>Cancel</Button>
          <Button
            variant="contained"
            startIcon={experimentRunning ? <CircularProgress size={18} color="inherit" /> : <ScienceIcon />}
            onClick={() => void handleRunExperiment()}
            disabled={experimentRunning || !canRunSelectedConfig}
          >
            {experimentRunning ? 'Running...' : 'Run Experiment'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
    );
  };

  const renderPipelines = () => (
    <Box>
      <PageHeader
        title="Pipeline Templates"
        description="Reusable RAG configurations for single-run execution and experiments."
        actions={<Button variant="contained" startIcon={<AccountTreeIcon />} onClick={handleNewPipeline}>New Pipeline</Button>}
      />
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(2, minmax(0, 1fr))', xl: 'repeat(3, minmax(0, 1fr))' }, gap: 2.5 }}>
        {templates.map((template) => (
          <Card key={template.id}>
            <CardContent>
              <Stack spacing={1.5}>
                <Stack direction="row" justifyContent="space-between" spacing={1}>
                  <Typography variant="h6">{template.name}</Typography>
                  <Chip size="small" label={template.tags[0] ?? 'Template'} color={template.tags.includes('Hybrid') ? 'primary' : template.tags.includes('Reranker') ? 'success' : 'default'} />
                </Stack>
                <Typography variant="body2" color="text.secondary" sx={{ minHeight: 42 }}>
                  {template.description ?? 'Pipeline configuration from configs/pipeline.'}
                </Typography>
                <MetaRow label="Run steps" value={template.run_steps.length} />
                <MetaRow label="Init steps" value={template.init_steps.length} />
                <MetaRow label="Components" value={template.components.length} />
                <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                  {template.tags.map((tag) => <Chip key={tag} size="small" label={tag} variant="outlined" />)}
                </Stack>
                <Button variant="contained" size="small" startIcon={<PlayCircleIcon />} onClick={() => applyTemplate(template)}>
                  Run
                </Button>
              </Stack>
            </CardContent>
          </Card>
        ))}
      </Box>
    </Box>
  );

  const page = {
    runner: renderRunner,
    compare: renderCompare,
    experiments: renderExperiments,
    pipelines: renderPipelines,
  }[activePage];

  return (
    <Box>
      <AppBar
        position="fixed"
        elevation={0}
        color="inherit"
        sx={{
          height: TOPBAR_HEIGHT,
          borderBottom: '1px solid #e5e7eb',
          bgcolor: 'rgba(255, 255, 255, 0.94)',
          backdropFilter: 'blur(14px)',
          zIndex: (theme) => theme.zIndex.drawer + 1,
        }}
      >
        <Toolbar sx={{ minHeight: `${TOPBAR_HEIGHT}px !important`, gap: 2 }}>
          <IconButton sx={{ display: { xs: 'inline-flex', md: 'none' } }} onClick={() => setMobileOpen(true)}>
            <MenuIcon />
          </IconButton>
          <Box
            sx={{
              width: 44,
              height: 44,
              borderRadius: 2,
              display: 'grid',
              placeItems: 'center',
              color: '#fff',
              fontWeight: 900,
              background: 'linear-gradient(135deg, #2563eb, #7c3aed)',
              boxShadow: '0 10px 24px rgba(37, 99, 235, 0.28)',
            }}
          >
            R
          </Box>
          <Box flex={1} minWidth={0}>
            <Typography variant="h6" color="text.primary" sx={{ lineHeight: 1.1 }}>
              RAG Studio
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Pipeline execution, comparison, evaluation, and debugging UI
            </Typography>
          </Box>
          <Chip color="success" label="API-connected" sx={{ display: { xs: 'none', sm: 'inline-flex' } }} />
          <Chip color="primary" variant="outlined" label="No CDN" sx={{ display: { xs: 'none', sm: 'inline-flex' } }} />
        </Toolbar>
      </AppBar>

      <Box
        component="aside"
        sx={{
          position: 'fixed',
          top: TOPBAR_HEIGHT,
          bottom: 0,
          left: 0,
          width: SIDEBAR_WIDTH,
          bgcolor: '#fff',
          borderRight: '1px solid #e5e7eb',
          overflow: 'auto',
          display: { xs: 'none', md: 'block' },
        }}
      >
        {nav}
      </Box>
      <Drawer open={mobileOpen} onClose={() => setMobileOpen(false)} PaperProps={{ sx: { width: SIDEBAR_WIDTH } }}>
        <Toolbar />
        {nav}
      </Drawer>

      <Box
        component="main"
        sx={{
          ml: { xs: 0, md: `${SIDEBAR_WIDTH}px` },
          pt: `${TOPBAR_HEIGHT + 28}px`,
          px: { xs: 1.75, md: 3.5 },
          pb: 5,
          minHeight: '100vh',
        }}
      >
        {loadingInitial ? (
          <Stack direction="row" spacing={1.5} alignItems="center">
            <CircularProgress size={20} />
            <Typography>Loading RAG Studio...</Typography>
          </Stack>
        ) : (
          <>
            {notice ? (
              <Alert severity={notice.severity} onClose={() => setNotice(null)} sx={{ mb: 2 }}>
                {notice.text}
              </Alert>
            ) : null}
            {page()}
          </>
        )}
      </Box>

      <Dialog open={promptDialogOpen} onClose={() => !savingPrompt && setPromptDialogOpen(false)} fullWidth maxWidth="sm">
        <DialogTitle>Add Prompt</DialogTitle>
        <DialogContent dividers>
          <Stack spacing={2}>
            {promptError ? <Alert severity="error">{promptError}</Alert> : null}
            <TextField
              label="Prompt name"
              value={newPromptName}
              onChange={(event) => setNewPromptName(event.target.value)}
              placeholder="my_prompt.yaml"
              size="small"
              fullWidth
              helperText="Saved to components/generation/templates. '.yaml' is added automatically."
            />
            <TextField
              label="Prompt template"
              value={newPromptTemplate}
              onChange={(event) => setNewPromptTemplate(event.target.value)}
              placeholder={'Answer the query using the context.\n\nContext:\n{context}\n\nQuery:\n{query}'}
              multiline
              minRows={10}
              fullWidth
              inputProps={{ spellCheck: false }}
              helperText="Use {context} and {query} placeholders. Detected variables are saved automatically."
            />
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setPromptDialogOpen(false)} disabled={savingPrompt}>Cancel</Button>
          <Button
            variant="contained"
            startIcon={savingPrompt ? <CircularProgress size={18} color="inherit" /> : null}
            onClick={() => void handleCreatePrompt()}
            disabled={savingPrompt || !newPromptName.trim() || !newPromptTemplate.trim()}
          >
            {savingPrompt ? 'Saving...' : 'Save Prompt'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

function MetaRow({ label, value }: { label: string; value: ReactNode }) {
  return (
    <Stack direction="row" justifyContent="space-between" spacing={2} sx={{ py: 0.75, borderBottom: '1px dashed #e2e8f0' }}>
      <Typography variant="body2" color="text.secondary">{label}</Typography>
      <Typography variant="body2" fontWeight={900} textAlign="right">{value}</Typography>
    </Stack>
  );
}

function MetricBar({ label, value, compact = false }: { label: string; value: unknown; compact?: boolean }) {
  const percent = metricPercent(value);
  return (
    <Box>
      <Stack direction="row" spacing={1.5} alignItems="center">
        <Typography variant="body2" sx={{ width: compact ? 150 : 190 }} noWrap>{label}</Typography>
        <Box flex={1}>
          <LinearProgress
            variant="determinate"
            value={percent}
            sx={{
              height: compact ? 8 : 14,
              borderRadius: 999,
              bgcolor: '#e2e8f0',
              '& .MuiLinearProgress-bar': {
                borderRadius: 999,
                background: 'linear-gradient(90deg, #2563eb, #7c3aed)',
              },
            }}
          />
        </Box>
        <Typography variant="body2" fontWeight={900} sx={{ width: 58, textAlign: 'right' }}>
          {formatMetric(value)}
        </Typography>
      </Stack>
    </Box>
  );
}
