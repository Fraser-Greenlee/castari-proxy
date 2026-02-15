import { AsyncLocalStorage } from 'node:async_hooks';
import { query, type Options, type Query, type SDKUserMessage } from '@anthropic-ai/claude-agent-sdk';

export type Provider = 'anthropic' | 'openrouter' | 'vllm' | 'sglang';

export interface ProviderCredentials {
  apiKey?: string;
}

export interface CastariProviders {
  openrouter?: ProviderCredentials;
  anthropic?: ProviderCredentials;
  vllm?: ProviderCredentials;
  sglang?: ProviderCredentials;
}

export type ReasoningEffortPreference = 'auto' | 'low' | 'medium' | 'high' | 'max';
export type ReasoningSummaryPreference = 'auto' | 'concise' | 'detailed' | 'none';

export interface ReasoningConfig {
  effort?: 'low' | 'medium' | 'high' | 'max';
  maxTokens?: number;
  exclude?: boolean;
  summary?: ReasoningSummaryPreference;
}

export interface CastariMeta {
  app?: string;
  clientId?: string;
  deploymentId?: string;
  version?: string;
  reasoningEffort?: ReasoningEffortPreference;
  reasoningSummary?: ReasoningSummaryPreference;
}

export interface CastariOptions extends Options {
  meta?: CastariMeta;
  providers?: CastariProviders;
  reasoning?: ReasoningConfig;
  reasoningEffort?: ReasoningEffortPreference;
  reasoningSummary?: ReasoningSummaryPreference;
  subagentModel?: string;
}

interface InterceptorContext {
  provider: Provider;
  originalModel: string;
  wireModel: string;
  meta?: CastariMeta;
  reasoning?: ReasoningConfig;
  effectiveEffort?: ReasoningEffortPreference;
  effectiveSummary?: ReasoningSummaryPreference;
  workerToken?: string;
  resolvedMeta?: Record<string, string>;
}

const baseOrigins = new Set<string>();

const ctxStore = new AsyncLocalStorage<InterceptorContext | undefined>();
let interceptorInstalled = false;

const HDR_MODEL = 'x-castari-model';
const HDR_PROVIDER = 'x-castari-provider';
const HDR_WIRE_MODEL = 'x-castari-wire-model';

export function resolveProvider(model: string): Provider {
  const normalized = model.trim();
  if (!normalized) throw new Error('model must be a non-empty string');
  if (normalized.startsWith('vllm:') || normalized.startsWith('vllm/')) return 'vllm';
  if (normalized.startsWith('sglang:') || normalized.startsWith('sglang/')) return 'sglang';
  if (normalized.startsWith('or:') || normalized.startsWith('openrouter/')) return 'openrouter';
  if (normalized.startsWith('openai/')) return 'openrouter';
  if (normalized.startsWith('anthropic/')) return 'anthropic';
  if (normalized.startsWith('claude')) return 'anthropic';
  return 'anthropic';
}

export function resolveWireModel(model: string, provider: Provider, defaultVendor = 'openai'): string {
  if (provider === 'openrouter') {
    if (model.startsWith('or:')) {
      const slug = model.slice(3);
      // If vendor-qualified in or: prefix, pass through vendor/model directly
      if (slug.includes('/')) return slug;
      // Otherwise prefix with default vendor (configurable via env)
      return `${defaultVendor}/${slug}`;
    }
    if (model.startsWith('openrouter/')) return model.substring('openrouter/'.length);
    if (model.startsWith('openai/')) return model;
    return model;
  }
  if (provider === 'vllm' || provider === 'sglang') {
    const colonPrefix = `${provider}:`;
    if (model.startsWith(colonPrefix)) return model.slice(colonPrefix.length);
    const slashPrefix = `${provider}/`;
    if (model.startsWith(slashPrefix)) return model.slice(slashPrefix.length);
    return model;
  }
  return model;
}

export interface CastariInterceptorOptions { baseUrl?: string }

export function installCastariInterceptor(options?: CastariInterceptorOptions): void {
  registerBaseOrigins({ explicit: options?.baseUrl });
  ensureInterceptorInstalled();
}

export function queryCastari({
  prompt,
  options = {},
}: {
  prompt: string | AsyncIterable<SDKUserMessage>;
  options?: CastariOptions;
}): Query {
  const model = typeof options.model === 'string' ? options.model : undefined;
  if (!model) throw new Error('queryCastari requires options.model to be specified');

  const provider = resolveProvider(model);
  // Build effective env (per-call) first so default vendor can be sourced from env
  const inputEnv = options.env;
  const effectiveEnv: Record<string, string | undefined> = {
    ...getProcessEnv(),
    ...inputEnv,
  };
  const defaultVendor = (effectiveEnv.OPENROUTER_DEFAULT_VENDOR || 'openai').trim() || 'openai';
  const wireModel = resolveWireModel(model, provider, defaultVendor);

  const {
    meta,
    providers,
    reasoning,
    reasoningEffort,
    reasoningSummary,
    subagentModel,
    ...sdkOptions
  } = options;

  const baseUrl =
    effectiveEnv?.ANTHROPIC_BASE_URL ||
    effectiveEnv?.CASTARI_GATEWAY_URL;
  if (!baseUrl) throw new Error('Castari Worker base URL not set. Configure ANTHROPIC_BASE_URL or CASTARI_GATEWAY_URL.');

  registerBaseOrigins({ explicit: baseUrl });

  const credential = getCredentialsForProvider(provider, providers, effectiveEnv);
  if (!credential) {
    const envVarMap: Record<Provider, string> = {
      anthropic: 'ANTHROPIC_API_KEY',
      openrouter: 'OPENROUTER_API_KEY',
      vllm: 'VLLM_API_KEY',
      sglang: 'SGLANG_API_KEY',
    };
    throw new Error(`${envVarMap[provider]} is required for model ${model}`);
  }
  effectiveEnv.ANTHROPIC_API_KEY = credential;
  effectiveEnv.ANTHROPIC_BASE_URL = baseUrl;

  const effEffort =
    normalizeEffort(reasoning?.effort) ||
    normalizeEffort(reasoningEffort) ||
    normalizeEffort(meta?.reasoningEffort) ||
    normalizeEffort(effectiveEnv.CASTARI_REASONING_EFFORT);
  const effSummary =
    normalizeSummary(reasoning?.summary) ||
    normalizeSummary(reasoningSummary) ||
    normalizeSummary(meta?.reasoningSummary) ||
    normalizeSummary(effectiveEnv.CASTARI_REASONING_SUMMARY) ||
    (effEffort ? 'auto' : undefined);

  // Map SDK Options.maxThinkingTokens into reasoning.max_tokens when not explicitly set
  const fallbackMaxThinking = typeof (options as Options).maxThinkingTokens === 'number'
    ? (options as Options).maxThinkingTokens
    : undefined;

  const resolvedSubagentModel = resolveSubagentModel({
    requested: subagentModel ?? effectiveEnv.CASTARI_SUBAGENT_MODEL,
    fallback: model,
  });

  if (!effectiveEnv.CLAUDE_CODE_SUBAGENT_MODEL) {
    effectiveEnv.CLAUDE_CODE_SUBAGENT_MODEL = resolvedSubagentModel;
  }
  const globalEnv = getProcessEnv();
  if (globalEnv && !globalEnv.CLAUDE_CODE_SUBAGENT_MODEL) {
    globalEnv.CLAUDE_CODE_SUBAGENT_MODEL = resolvedSubagentModel;
  }

  ensureInterceptorInstalled();

  const workingOptions: Options = {
    ...sdkOptions,
    env: effectiveEnv,
  } as Options;

  // Build resolved meta (per-call) for x-client-meta
  const resolvedMeta = buildResolvedMeta({
    base: meta,
    env: effectiveEnv,
    provider,
    originalModel: model,
    effort: effEffort,
    summary: effSummary,
  });

  const ctx: InterceptorContext = {
    provider,
    originalModel: model,
    wireModel,
    meta,
    reasoning: normalizeReasoning(reasoning, effEffort, effSummary, fallbackMaxThinking),
    effectiveEffort: effEffort,
    effectiveSummary: effSummary,
    workerToken: effectiveEnv.X_WORKER_TOKEN,
    resolvedMeta,
  };

  return ctxStore.run(ctx, () => query({ prompt: prompt as any, options: workingOptions }));
}

function ensureInterceptorInstalled(): void {
  if (interceptorInstalled) return;
  const g = globalThis as unknown as { fetch?: typeof fetch };
  if (typeof g.fetch !== 'function') throw new Error('globalThis.fetch is not available in this environment');
  const originalFetch = g.fetch.bind(globalThis);

  g.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const request = input instanceof Request ? input : new Request(input, init);
    if (!(await shouldIntercept(request))) return originalFetch(request);

    const ctx = ctxStore.getStore();
    const headers = new Headers(request.headers);

    if (ctx) {
      headers.set(HDR_PROVIDER, ctx.provider);
      headers.set(HDR_MODEL, ctx.originalModel);
      headers.set(HDR_WIRE_MODEL, ctx.wireModel);

      const workerToken = ctx.workerToken;
      if (workerToken && !headers.has('x-worker-token')) headers.set('x-worker-token', workerToken);

      const meta = ctx.resolvedMeta;
      if (meta && !headers.has('x-client-meta')) headers.set('x-client-meta', JSON.stringify(meta));
    }

    let nextBody: string | undefined;
    const contentType = request.headers.get('content-type') || '';
    if (request.method.toUpperCase() === 'POST' && contentType.includes('application/json')) {
      try {
        const raw = await request.clone().text();
        if (raw) {
          const updated = injectReasoningIntoPayload(raw, ctx?.reasoning);
          if (updated !== null) nextBody = updated;
        }
      } catch {
        // no-op
      }
    }

    const nextReq = nextBody != null ? new Request(request, { headers, body: nextBody }) : new Request(request, { headers });
    return originalFetch(nextReq);
  };
  interceptorInstalled = true;
}

async function shouldIntercept(request: Request): Promise<boolean> {
  try {
    const url = new URL(request.url);
    if (request.method.toUpperCase() !== 'POST') return false;
    if (url.pathname !== '/v1/messages') return false;
    if (baseOrigins.size > 0 && !baseOrigins.has(url.origin)) return false;
    return true;
  } catch {
    return false;
  }
}

function registerBaseOrigins(options?: { env?: Record<string, string | undefined>; explicit?: string }): void {
  const candidates: Array<string | undefined> = [
    options?.explicit,
    options?.env?.ANTHROPIC_BASE_URL,
    getProcessEnv()?.ANTHROPIC_BASE_URL,
    getProcessEnv()?.CASTARI_GATEWAY_URL,
  ];
  for (const cand of candidates) addBaseOrigin(cand);
}

function addBaseOrigin(candidate?: string): void {
  if (!candidate) return;
  try {
    const url = new URL(candidate);
    baseOrigins.add(url.origin);
  } catch {
    // ignore
  }
}

function getCredentialsForProvider(
  provider: Provider,
  providers: CastariProviders | undefined,
  env: Record<string, string | undefined>,
): string | undefined {
  if (provider === 'openrouter') {
    return providers?.openrouter?.apiKey ?? env.OPENROUTER_API_KEY;
  }
  if (provider === 'vllm') {
    return providers?.vllm?.apiKey ?? env.VLLM_API_KEY ?? 'no-auth';
  }
  if (provider === 'sglang') {
    return providers?.sglang?.apiKey ?? env.SGLANG_API_KEY ?? 'no-auth';
  }
  return providers?.anthropic?.apiKey ?? env.ANTHROPIC_API_KEY;
}

function normalizeEffort(value: unknown): ReasoningEffortPreference | undefined {
  if (typeof value !== 'string') return undefined;
  const v = value.trim().toLowerCase();
  if (v === 'auto' || v === 'low' || v === 'medium' || v === 'high' || v === 'max') return v as ReasoningEffortPreference;
  return undefined;
}

function normalizeSummary(value: unknown): ReasoningSummaryPreference | undefined {
  if (typeof value !== 'string') return undefined;
  const v = value.trim().toLowerCase();
  if (v === 'auto' || v === 'concise' || v === 'detailed' || v === 'none') return v as ReasoningSummaryPreference;
  return undefined;
}

function normalizeReasoning(
  cfg: ReasoningConfig | undefined,
  eff: ReasoningEffortPreference | undefined,
  sum: ReasoningSummaryPreference | undefined,
  fallbackMaxTokens?: number,
): ReasoningConfig | undefined {
  const result: ReasoningConfig = {};
  if (cfg?.effort) result.effort = cfg.effort;
  else if (eff && eff !== 'auto') result.effort = eff as Exclude<ReasoningEffortPreference, 'auto'>;

  if (typeof cfg?.maxTokens === 'number') result.maxTokens = cfg.maxTokens;
  else if (typeof fallbackMaxTokens === 'number') result.maxTokens = fallbackMaxTokens;
  if (typeof cfg?.exclude === 'boolean') result.exclude = cfg.exclude;
  if (cfg?.summary) result.summary = cfg.summary;
  else if (sum && sum !== 'auto') result.summary = sum;

  return Object.keys(result).length ? result : undefined;
}

function injectReasoningIntoPayload(raw: string, reasoning?: ReasoningConfig): string | null {
  if (!reasoning) return null;
  try {
    const payload = JSON.parse(raw);
    if (!payload || typeof payload !== 'object' || Array.isArray(payload)) return null;
    const metadata = typeof payload.metadata === 'object' && payload.metadata !== null && !Array.isArray(payload.metadata)
      ? { ...payload.metadata }
      : {};
    const castari = typeof (metadata as any).castari === 'object' && (metadata as any).castari !== null && !Array.isArray((metadata as any).castari)
      ? { ...(metadata as any).castari }
      : {};

    const r: Record<string, unknown> = {};
    if (reasoning.effort) r.effort = reasoning.effort;
    if (typeof reasoning.maxTokens === 'number') r.max_tokens = reasoning.maxTokens;
    if (typeof reasoning.exclude === 'boolean') r.exclude = reasoning.exclude;
    if (reasoning.summary) r.summary = reasoning.summary;

    if (Object.keys(r).length > 0) {
      (castari as any).reasoning = r;
      (metadata as any).castari = castari;
      (payload as any).metadata = metadata;
      return JSON.stringify(payload);
    }
    return null;
  } catch {
    return null;
  }
}

function buildResolvedMeta(args: {
  base?: CastariMeta;
  env: Record<string, string | undefined>;
  provider: Provider;
  originalModel: string;
  effort?: ReasoningEffortPreference;
  summary?: ReasoningSummaryPreference;
}): Record<string, string> | undefined {
  const meta: Record<string, string | undefined> = {
    sdk: 'claude-agent',
    app: args.base?.app ?? args.env.CASTARI_APP,
    clientId: args.base?.clientId ?? args.env.CASTARI_CLIENT_ID,
    deploymentId: args.base?.deploymentId ?? args.env.CASTARI_DEPLOYMENT_ID,
    version: args.base?.version ?? args.env.CASTARI_VERSION,
    model: args.originalModel,
    provider: args.provider,
    reasoningEffort: args.base?.reasoningEffort ?? args.effort,
    reasoningSummary: args.base?.reasoningSummary ?? args.summary,
  };
  const cleaned: Record<string, string> = {};
  for (const [k, v] of Object.entries(meta)) {
    if (typeof v === 'string' && v.trim().length > 0) cleaned[k] = v;
  }
  return Object.keys(cleaned).length ? cleaned : undefined;
}

function getProcessEnv(): Record<string, string | undefined> | undefined {
  const g = globalThis as unknown as { process?: { env?: Record<string, string | undefined> } };
  return g.process?.env;
}

interface SubagentResolutionParams {
  requested?: string;
  fallback: string;
}

const CLAUDE_TIER_ALIASES: Record<string, string> = {
  haiku: 'claude-haiku-4-5-20251001',
  sonnet: 'claude-sonnet-4-5-20250929',
  opus: 'claude-opus-4-1-20240808',
};

function resolveSubagentModel(params: SubagentResolutionParams): string {
  const input = typeof params.requested === 'string' ? params.requested.trim() : '';
  if (!input) return params.fallback;
  const normalized = input.toLowerCase();
  if (normalized === 'inherit' || normalized === 'auto' || normalized === 'default') {
    return params.fallback;
  }
  const alias = CLAUDE_TIER_ALIASES[normalized];
  return alias ?? input;
}
