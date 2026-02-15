import {
  CastariHeaders,
  Provider,
  WorkerConfig,
  AnthropicRequest,
  AnthropicToolDefinition,
} from './types';
import { invalidRequest } from './errors';

const SERVER_TOOL_PATTERN = /(Tool_|Tool$)/i;
const WEB_SEARCH_NAMES = new Set([
  'websearch',
  'websearchtool',
  'webfetch',
  'webfetchtool',
]);
const SERVER_TOOL_ALIAS = new Set([
  'websearch',
  'webfetch',
  'codeexecution',
  'computeruse',
  'texteditor',
  'memorytool',
]);
const SERVER_TOOL_NAME_SET = new Set([
  'websearch',
  'websearchtool',
  'webfetch',
  'webfetchtool',
  'codeexecution',
  'codeexecutiontool',
  'computeruse',
  'computerusetool',
  'texteditor',
  'texteditortool',
  'memorytool',
]);

export interface ProviderResolution {
  provider: Provider;
  wireModel: string;
  originalModel: string;
}

export function resolveProvider(
  headers: CastariHeaders,
  body: AnthropicRequest,
  config: WorkerConfig,
): ProviderResolution {
  const originalModel = body.model;
  const provider = headers.provider ?? inferProviderFromModel(originalModel);
  if (!provider) throw invalidRequest(`Unable to infer provider for model ${originalModel}`);
  let wireModel: string;
  if (provider === 'openrouter') {
    wireModel = resolveOpenRouterModel(headers.wireModel ?? originalModel, config.defaultOpenRouterVendor);
  } else if (provider === 'vllm' || provider === 'sglang') {
    wireModel = resolveCustomProviderModel(headers.wireModel ?? originalModel, provider);
  } else {
    wireModel = originalModel;
  }
  return { provider, wireModel, originalModel };
}

function inferProviderFromModel(model: string): Provider | undefined {
  const normalized = model.toLowerCase();
  if (normalized.startsWith('vllm:') || normalized.startsWith('vllm/')) return 'vllm';
  if (normalized.startsWith('sglang:') || normalized.startsWith('sglang/')) return 'sglang';
  if (normalized.startsWith('claude') || normalized.startsWith('anthropic/')) return 'anthropic';
  if (normalized.startsWith('or:') || normalized.startsWith('openrouter/') || normalized.startsWith('openai/')) return 'openrouter';
  return 'anthropic';
}

function resolveOpenRouterModel(model: string, defaultVendor: string): string {
  if (model.startsWith('or:')) {
    const slug = model.slice(3);
    if (!slug) throw invalidRequest('OpenRouter model prefix "or:" must include a slug');
    if (slug.includes('/')) return slug;
    return `${defaultVendor}/${slug}`;
  }
  if (model.startsWith('openrouter/')) return model.substring('openrouter/'.length);
  return model;
}

function resolveCustomProviderModel(model: string, provider: 'vllm' | 'sglang'): string {
  const prefix = `${provider}:`;
  if (model.toLowerCase().startsWith(prefix)) {
    const slug = model.slice(prefix.length);
    if (!slug) throw invalidRequest(`${provider} model prefix "${prefix}" must include a model name`);
    return slug;
  }
  const slashPrefix = `${provider}/`;
  if (model.toLowerCase().startsWith(slashPrefix)) {
    const slug = model.slice(slashPrefix.length);
    if (!slug) throw invalidRequest(`${provider} model prefix "${slashPrefix}" must include a model name`);
    return slug;
  }
  return model;
}

export interface ServerToolEntry {
  label: string;
  kind: 'websearch' | 'other';
}

export function categorizeServerTools(tools?: AnthropicToolDefinition[]): ServerToolEntry[] {
  if (!Array.isArray(tools)) return [];
  const entries: ServerToolEntry[] = [];
  for (const tool of tools) {
    if (!isServerTool(tool)) continue;
    const label =
      (typeof tool.type === 'string' && tool.type) ||
      (typeof tool.name === 'string' && tool.name) ||
      'server_tool';
    entries.push({
      label,
      kind: isWebSearchTool(tool) ? 'websearch' : 'other',
    });
  }
  return entries;
}

export function detectServerTools(tools?: AnthropicToolDefinition[]): string[] {
  return categorizeServerTools(tools).map((entry) => entry.label);
}

export function isServerTool(tool: AnthropicToolDefinition | undefined): boolean {
  if (!tool || typeof tool !== 'object') return false;
  const type = typeof tool.type === 'string' ? tool.type : undefined;
  if (type && SERVER_TOOL_PATTERN.test(type)) return true;
  const name = typeof tool.name === 'string' ? tool.name : undefined;
  if (!name) return false;
  if (SERVER_TOOL_PATTERN.test(name)) return true;
  if (SERVER_TOOL_ALIAS.has(name.toLowerCase())) return true;
  return false;
}

export function isWebSearchTool(tool: AnthropicToolDefinition | undefined): boolean {
  if (!tool) return false;
  const type = typeof tool.type === 'string' ? tool.type : undefined;
  if (type && type.toLowerCase().includes('websearch')) return true;
  const name = typeof tool.name === 'string' ? tool.name : undefined;
  if (!name) return false;
  const lower = name.toLowerCase();
  if (lower.includes('websearch')) return true;
  if (WEB_SEARCH_NAMES.has(lower)) return true;
  return false;
}
