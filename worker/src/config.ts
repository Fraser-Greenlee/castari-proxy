import { McpBridgeMode, ServerToolsMode, WorkerConfig } from './types';

export interface Env {
  UPSTREAM_ANTHROPIC_BASE_URL?: string;
  UPSTREAM_OPENROUTER_BASE_URL?: string;
  UPSTREAM_VLLM_BASE_URL?: string;
  UPSTREAM_SGLANG_BASE_URL?: string;
  SERVER_TOOLS_MODE?: string;
  MCP_BRIDGE_MODE?: string;
  OPENROUTER_DEFAULT_VENDOR?: string;
}

const DEFAULT_ANTHROPIC_URL = 'https://api.anthropic.com';
const DEFAULT_OPENROUTER_URL = 'https://openrouter.ai/api';

export function resolveConfig(env: Env): WorkerConfig {
  const anthropicBaseUrl = normalizeBaseUrl(env.UPSTREAM_ANTHROPIC_BASE_URL ?? DEFAULT_ANTHROPIC_URL, '/v1/messages');
  const openRouterBaseUrl = normalizeBaseUrl(env.UPSTREAM_OPENROUTER_BASE_URL ?? DEFAULT_OPENROUTER_URL, '/v1/chat/completions');
  const vllmBaseUrl = env.UPSTREAM_VLLM_BASE_URL?.trim()
    ? normalizeBaseUrl(env.UPSTREAM_VLLM_BASE_URL, '/v1/chat/completions')
    : '';
  const sglangBaseUrl = env.UPSTREAM_SGLANG_BASE_URL?.trim()
    ? normalizeBaseUrl(env.UPSTREAM_SGLANG_BASE_URL, '/v1/chat/completions')
    : '';
  const serverToolsMode = normalizeServerToolsMode(env.SERVER_TOOLS_MODE);
  const mcpMode = normalizeMcpMode(env.MCP_BRIDGE_MODE);
  const defaultOpenRouterVendor = (env.OPENROUTER_DEFAULT_VENDOR?.trim() || 'openai').toLowerCase();

  return {
    anthropicBaseUrl,
    openRouterBaseUrl,
    vllmBaseUrl,
    sglangBaseUrl,
    serverToolsMode,
    mcpMode,
    defaultOpenRouterVendor,
  };
}

function normalizeBaseUrl(value: string, suffix: '/v1/messages' | '/v1/chat/completions'): string {
  const trimmed = value.replace(/\/$/, '');
  if (trimmed.endsWith(suffix)) return trimmed;
  return `${trimmed}${suffix}`;
}

function normalizeServerToolsMode(value?: string): ServerToolsMode {
  switch ((value ?? '').toLowerCase()) {
    case 'enforceanthropic':
    case 'enforce-anthropic':
      return 'enforceAnthropic';
    case 'emulate':
      return 'emulate';
    case 'error':
    default:
      return 'error';
  }
}

function normalizeMcpMode(value?: string): McpBridgeMode {
  switch ((value ?? '').toLowerCase()) {
    case 'http-sse':
      return 'http-sse';
    default:
      return 'off';
  }
}
