import { describe, expect, it } from 'vitest';
import { detectServerTools, resolveProvider } from '../src/provider';
import { resolveConfig } from '../src/config';

const config = resolveConfig({
  UPSTREAM_ANTHROPIC_BASE_URL: 'https://api.anthropic.com',
  UPSTREAM_OPENROUTER_BASE_URL: 'https://openrouter.ai/api',
  UPSTREAM_VLLM_BASE_URL: 'http://localhost:8000',
  UPSTREAM_SGLANG_BASE_URL: 'http://localhost:30000',
  SERVER_TOOLS_MODE: 'error',
  OPENROUTER_DEFAULT_VENDOR: 'openai',
});

describe('resolveProvider', () => {
  it('detects OpenRouter or: slugs and normalizes wire models', () => {
    const result = resolveProvider(
      { provider: 'openrouter', originalModel: 'or:gpt-5-mini', wireModel: undefined },
      { model: 'or:gpt-5-mini', messages: [], metadata: {}, tools: [] } as any,
      config,
    );
    expect(result.provider).toBe('openrouter');
    expect(result.wireModel).toBe('openai/gpt-5-mini');
  });

  it('infers provider when headers missing', () => {
    const result = resolveProvider(
      {},
      { model: 'claude-3-5', messages: [], metadata: {} } as any,
      config,
    );
    expect(result.provider).toBe('anthropic');
    expect(result.wireModel).toBe('claude-3-5');
  });

  it('detects vllm: prefix and strips it for wire model', () => {
    const result = resolveProvider(
      {},
      { model: 'vllm:meta-llama/Llama-3-70b', messages: [], metadata: {} } as any,
      config,
    );
    expect(result.provider).toBe('vllm');
    expect(result.wireModel).toBe('meta-llama/Llama-3-70b');
    expect(result.originalModel).toBe('vllm:meta-llama/Llama-3-70b');
  });

  it('detects vllm/ prefix and strips it for wire model', () => {
    const result = resolveProvider(
      {},
      { model: 'vllm/my-local-model', messages: [], metadata: {} } as any,
      config,
    );
    expect(result.provider).toBe('vllm');
    expect(result.wireModel).toBe('my-local-model');
  });

  it('detects sglang: prefix and strips it for wire model', () => {
    const result = resolveProvider(
      {},
      { model: 'sglang:deepseek-ai/DeepSeek-V3', messages: [], metadata: {} } as any,
      config,
    );
    expect(result.provider).toBe('sglang');
    expect(result.wireModel).toBe('deepseek-ai/DeepSeek-V3');
    expect(result.originalModel).toBe('sglang:deepseek-ai/DeepSeek-V3');
  });

  it('detects sglang/ prefix and strips it for wire model', () => {
    const result = resolveProvider(
      {},
      { model: 'sglang/my-model', messages: [], metadata: {} } as any,
      config,
    );
    expect(result.provider).toBe('sglang');
    expect(result.wireModel).toBe('my-model');
  });

  it('uses header provider override for vllm', () => {
    const result = resolveProvider(
      { provider: 'vllm', wireModel: 'custom-model' },
      { model: 'vllm:custom-model', messages: [], metadata: {} } as any,
      config,
    );
    expect(result.provider).toBe('vllm');
    expect(result.wireModel).toBe('custom-model');
  });
});

describe('detectServerTools', () => {
  it('flags Anthropic server tools based on type pattern', () => {
    const matches = detectServerTools([
      { name: 'my_tool', input_schema: {} },
      { type: 'WebSearchTool_20250305' },
    ] as any);
    expect(matches).toContain('WebSearchTool_20250305');
  });
});
