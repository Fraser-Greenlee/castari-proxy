import { resolveConfig, Env } from './config';
import { errorResponse, authenticationError, invalidRequest } from './errors';
import { normalizeCastariHeaders, readJsonBody, getHeader, randomId } from './utils';
import { categorizeServerTools, resolveProvider } from './provider';
import { buildOpenRouterRequest, mapOpenRouterResponse } from './translator';
import { streamOpenRouterToAnthropic } from './stream';
import { LogEntry, deriveTurnIndex, writeLogEntry, teeAndCollectStream } from './logger';
import {
  AnthropicRequest,
  AnthropicResponse,
  CastariMetadata,
  CastariReasoningConfig,
  Provider,
  WebSearchOptions,
  WorkerConfig,
} from './types';

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    try {
      if (new URL(request.url).pathname !== '/v1/messages' || request.method !== 'POST') {
        return new Response('Not found', { status: 404 });
      }

      const startTime = Date.now();
      const requestId = randomId('req');
      const sessionId = getHeader(request.headers, 'x-castari-session-id') ?? randomId('sess');
      const clientMetaRaw = getHeader(request.headers, 'x-client-meta');
      const clientMeta = clientMetaRaw ? safeParseJsonMeta(clientMetaRaw) : undefined;

      const config = resolveConfig(env);
      const headers = normalizeCastariHeaders(request.headers);
      const body = await readJsonBody<AnthropicRequest>(request.clone());
      const metadata = normalizeMetadata(body.metadata);
      const reasoning = metadata?.castari?.reasoning as CastariReasoningConfig | undefined;
      let webSearch = metadata?.castari?.web_search_options as WebSearchOptions | undefined;

      let { provider, wireModel, originalModel } = resolveProvider(headers, body, config);

      const turnIndex = deriveTurnIndex(body.messages);

      // vLLM and SGLang don't require auth; all other providers do.
      const optionalAuthProviders: Provider[] = ['vllm', 'sglang'];
      const apiKey = optionalAuthProviders.includes(provider)
        ? extractOptionalApiKey(request.headers)
        : extractApiKey(request.headers).value;

      const serverToolEntries = categorizeServerTools(body.tools);
      const webSearchTools = serverToolEntries.filter((entry) => entry.kind === 'websearch');
      const otherServerTools = serverToolEntries.filter((entry) => entry.kind === 'other');

      const isNonAnthropicProvider = provider !== 'anthropic';

      if (isNonAnthropicProvider && otherServerTools.length) {
        if (config.serverToolsMode === 'error') {
          throw invalidRequest('Server tools require Anthropic provider', {
            tools: otherServerTools.map((entry) => entry.label),
          });
        }
        if (config.serverToolsMode === 'enforceAnthropic') {
          provider = 'anthropic';
          wireModel = originalModel;
        }
        // emulate mode would be implemented when server backends are available
      }

      if (provider === 'openrouter') {
        const wantsWebSearch = webSearchTools.length > 0;
        if (wantsWebSearch && !webSearch) {
          webSearch = {};
        }
      }

      if (body.mcp_servers?.length && provider !== 'anthropic' && env.MCP_BRIDGE_MODE !== 'http-sse') {
        throw invalidRequest('MCP servers require Anthropic routing or http-sse bridge', { mode: env.MCP_BRIDGE_MODE ?? 'off' });
      }

      // Build the base log entry (response filled in per-path)
      const baseEntry: Omit<LogEntry, 'response' | 'latencyMs' | 'streaming'> = {
        requestId,
        sessionId,
        turnIndex,
        timestamp: new Date().toISOString(),
        provider,
        originalModel,
        wireModel,
        request: body,
        clientMeta,
      };

      // Helper to schedule a log write in the background
      const scheduleLog = (response: AnthropicResponse | null, streaming: boolean, error?: LogEntry['error']) => {
        const entry: LogEntry = {
          ...baseEntry,
          response,
          latencyMs: Date.now() - startTime,
          streaming,
          ...(error ? { error } : {}),
        };
        ctx.waitUntil(writeLogEntry(env.LOG_BUCKET, entry));
      };

      // Helper to handle a streaming response: tee the stream, log asynchronously
      const logStreamingResponse = (resp: Response): Response => {
        const { clientResponse, collected } = teeAndCollectStream(resp);
        ctx.waitUntil(
          collected.then((assembled) => scheduleLog(assembled, true)),
        );
        return clientResponse;
      };

      if (provider === 'anthropic') {
        const resp = await proxyAnthropic(body, request, apiKey!, config.anthropicBaseUrl);
        if (!resp.ok) {
          // Error responses are not streamed — log as-is
          const errorText = await resp.clone().text();
          scheduleLog(null, false, { status: resp.status, message: errorText.slice(0, 1000) });
          return resp;
        }
        return logStreamingResponse(resp);
      }

      if (provider === 'vllm' || provider === 'sglang') {
        const upstreamUrl = provider === 'vllm' ? config.vllmBaseUrl : config.sglangBaseUrl;
        if (!upstreamUrl) {
          throw invalidRequest(
            `${provider} provider is not configured. Set UPSTREAM_${provider.toUpperCase()}_BASE_URL.`,
          );
        }
        const resp = await handleOpenAICompatible({
          body,
          wireModel,
          originalModel,
          apiKey,
          upstreamUrl,
          providerName: provider,
        });
        if (body.stream) {
          return logStreamingResponse(resp);
        }
        // Non-streaming: parse the response body for logging, then return it
        const clonedBody = await resp.clone().json() as AnthropicResponse;
        scheduleLog(clonedBody, false);
        return resp;
      }

      const resp = await handleOpenRouter({
        body,
        wireModel,
        originalModel,
        apiKey: apiKey!,
        config,
        reasoning,
        webSearch,
      });
      if (body.stream) {
        return logStreamingResponse(resp);
      }
      const clonedBody = await resp.clone().json() as AnthropicResponse;
      scheduleLog(clonedBody, false);
      return resp;
    } catch (error) {
      return errorResponse(error);
    }
  },
};

function normalizeMetadata(metadata: AnthropicRequest['metadata']): CastariMetadata | undefined {
  if (!metadata || typeof metadata !== 'object') return undefined;
  return metadata as CastariMetadata;
}

function extractApiKey(headers: Headers): { value: string; type: 'x-api-key' | 'bearer' } {
  const auth = headers.get('authorization');
  if (auth && auth.toLowerCase().startsWith('bearer ')) {
    const token = auth.slice(7).trim();
    if (token) return { value: token, type: 'bearer' };
  }
  const key = headers.get('x-api-key');
  if (key) return { value: key, type: 'x-api-key' };
  throw authenticationError('Missing API key');
}

function extractOptionalApiKey(headers: Headers): string | undefined {
  const auth = headers.get('authorization');
  if (auth && auth.toLowerCase().startsWith('bearer ')) {
    const token = auth.slice(7).trim();
    if (token && token !== 'no-auth') return token;
  }
  const key = headers.get('x-api-key');
  if (key && key !== 'no-auth') return key;
  return undefined;
}

async function proxyAnthropic(
  body: AnthropicRequest,
  request: Request,
  apiKey: string,
  upstreamUrl: string,
): Promise<Response> {
  const upstreamResp = await fetch(upstreamUrl, {
    method: 'POST',
    headers: buildAnthropicHeaders(request.headers, apiKey),
    body: JSON.stringify(body),
  });
  if (!upstreamResp.ok) {
    const text = await upstreamResp.text();
    return new Response(text || JSON.stringify({ error: 'Anthropic upstream error' }), {
      status: upstreamResp.status,
      headers: {
        'content-type': upstreamResp.headers.get('content-type') ?? 'application/json',
      },
    });
  }
  return upstreamResp;
}

function buildAnthropicHeaders(original: Headers, apiKey: string): HeadersInit {
  const headers = new Headers();
  headers.set('content-type', 'application/json');
  headers.set('x-api-key', apiKey);
  const anthropicVersion = original.get('anthropic-version');
  if (anthropicVersion) headers.set('anthropic-version', anthropicVersion);
  return headers;
}

interface OpenRouterContext {
  body: AnthropicRequest;
  wireModel: string;
  originalModel: string;
  apiKey: string;
  config: WorkerConfig;
  reasoning?: CastariReasoningConfig;
  webSearch?: WebSearchOptions;
}

interface OpenAICompatibleContext {
  body: AnthropicRequest;
  wireModel: string;
  originalModel: string;
  apiKey?: string;
  upstreamUrl: string;
  providerName: string;
}

async function handleOpenAICompatible(ctx: OpenAICompatibleContext): Promise<Response> {
  const compatRequest = buildOpenRouterRequest(ctx.body, {
    wireModel: ctx.wireModel,
  });

  const headers: Record<string, string> = {
    'content-type': 'application/json',
  };
  if (ctx.apiKey) {
    headers.authorization = `Bearer ${ctx.apiKey}`;
  }

  const upstreamResp = await fetch(ctx.upstreamUrl, {
    method: 'POST',
    headers,
    body: JSON.stringify(compatRequest),
  });

  if (ctx.body.stream) {
    if (!upstreamResp.ok) {
      const payload = await upstreamResp.text();
      throw invalidRequest(`${ctx.providerName} streaming error`, { status: upstreamResp.status, body: payload });
    }
    return streamOpenRouterToAnthropic(upstreamResp, { originalModel: ctx.originalModel });
  }

  const json = await upstreamResp.json();
  if (!upstreamResp.ok) {
    throw invalidRequest(`${ctx.providerName} error`, { status: upstreamResp.status, body: json });
  }
  const responseBody: AnthropicResponse = mapOpenRouterResponse(json, ctx.originalModel);
  return new Response(JSON.stringify(responseBody), {
    status: 200,
    headers: {
      'content-type': 'application/json',
      'cache-control': 'no-store',
    },
  });
}

async function handleOpenRouter(ctx: OpenRouterContext): Promise<Response> {
  const openRouterRequest = buildOpenRouterRequest(ctx.body, {
    wireModel: ctx.wireModel,
    reasoning: ctx.reasoning,
    webSearch: ctx.webSearch,
  });

  const upstreamResp = await fetch(ctx.config.openRouterBaseUrl, {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      authorization: `Bearer ${ctx.apiKey}`,
    },
    body: JSON.stringify(openRouterRequest),
  });

  if (ctx.body.stream) {
    if (!upstreamResp.ok) {
      const payload = await upstreamResp.text();
      throw invalidRequest('OpenRouter streaming error', { status: upstreamResp.status, body: payload });
    }
    return streamOpenRouterToAnthropic(upstreamResp, { originalModel: ctx.originalModel });
  }

  const json = await upstreamResp.json();
  if (!upstreamResp.ok) {
    throw invalidRequest('OpenRouter error', { status: upstreamResp.status, body: json });
  }
  const responseBody: AnthropicResponse = mapOpenRouterResponse(json, ctx.originalModel);
  return new Response(JSON.stringify(responseBody), {
    status: 200,
    headers: {
      'content-type': 'application/json',
      'cache-control': 'no-store',
    },
  });
}

function safeParseJsonMeta(raw: string): Record<string, unknown> | undefined {
  try {
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
    return undefined;
  } catch {
    return undefined;
  }
}
