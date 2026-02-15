import {
  AnthropicContent,
  AnthropicMessage,
  AnthropicRequest,
  AnthropicResponse,
  Provider,
} from './types';
import { randomId } from './utils';

export interface LogEntry {
  requestId: string;
  sessionId: string;
  turnIndex: number;
  timestamp: string;
  provider: Provider;
  originalModel: string;
  wireModel: string;
  request: AnthropicRequest;
  response: AnthropicResponse | null;
  error?: { status: number; message: string };
  clientMeta?: Record<string, unknown>;
  latencyMs: number;
  streaming: boolean;
}

/**
 * Derives the turn index from the messages array by counting assistant messages.
 * In an agentic loop, each request appends the previous assistant response and
 * the tool results, so the number of assistant messages reflects the turn number.
 */
export function deriveTurnIndex(messages: AnthropicMessage[]): number {
  let count = 0;
  for (const msg of messages) {
    if (msg.role === 'assistant') count++;
  }
  return count;
}

/**
 * Constructs the R2 object key for a log entry.
 * Format: logs/{YYYY-MM-DD}/{sessionId}/{turnIndex}_{requestId}.json
 */
export function buildLogKey(entry: LogEntry): string {
  const date = entry.timestamp.slice(0, 10); // YYYY-MM-DD
  return `logs/${date}/${entry.sessionId}/${entry.turnIndex}_${entry.requestId}.json`;
}

/**
 * Writes a log entry to R2. Wrapped in try/catch so logging failures
 * never break the proxy response.
 */
export async function writeLogEntry(bucket: R2Bucket, entry: LogEntry): Promise<void> {
  try {
    const key = buildLogKey(entry);
    const body = JSON.stringify(entry);
    await bucket.put(key, body, {
      httpMetadata: { contentType: 'application/json' },
      customMetadata: {
        sessionId: entry.sessionId,
        provider: entry.provider,
        model: entry.originalModel,
        turnIndex: String(entry.turnIndex),
      },
    });
  } catch {
    // Logging must never break the proxy — silently swallow errors.
  }
}

/**
 * Result of tee-ing a streaming response for logging.
 */
export interface TeeResult {
  /** The response to send back to the client (unchanged stream). */
  clientResponse: Response;
  /** Resolves to the assembled AnthropicResponse once the stream ends. */
  collected: Promise<AnthropicResponse>;
}

// Internal block accumulator types
interface TextBlock {
  type: 'text';
  text: string;
}

interface ToolUseBlock {
  type: 'tool_use';
  id: string;
  name: string;
  inputBuffer: string;
}

type BlockState = TextBlock | ToolUseBlock;

/**
 * Tees a streaming SSE response so the client receives it unmodified,
 * while we consume the other branch in the background to reassemble
 * the complete AnthropicResponse for logging.
 *
 * Works with Anthropic-format SSE (both native Anthropic pass-through
 * and the translated output from streamOpenRouterToAnthropic).
 */
export function teeAndCollectStream(response: Response): TeeResult {
  if (!response.body) {
    return {
      clientResponse: response,
      collected: Promise.resolve(emptyResponse()),
    };
  }

  const [clientBranch, logBranch] = response.body.tee();

  const clientResponse = new Response(clientBranch, {
    status: response.status,
    headers: response.headers,
  });

  const collected = reassembleFromSSE(logBranch);

  return { clientResponse, collected };
}

/**
 * Reads an Anthropic-format SSE stream and reassembles the full response.
 */
async function reassembleFromSSE(stream: ReadableStream<Uint8Array>): Promise<AnthropicResponse> {
  const decoder = new TextDecoder();
  const reader = stream.getReader();

  let messageId = '';
  let model = '';
  let stopReason: string | null = null;
  let usage: AnthropicResponse['usage'] = undefined;
  const blocks: BlockState[] = [];
  let buffer = '';

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let boundary = buffer.indexOf('\n\n');
      while (boundary !== -1) {
        const rawEvent = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        processRawEvent(rawEvent.trim());
        boundary = buffer.indexOf('\n\n');
      }
    }
  } catch {
    // If the stream errors, return whatever we've accumulated so far.
  } finally {
    reader.releaseLock();
  }

  // Build final content array from accumulated blocks
  const content: AnthropicContent[] = blocks.map((block) => {
    if (block.type === 'text') {
      return { type: 'text' as const, text: block.text };
    }
    return {
      type: 'tool_use' as const,
      id: block.id,
      name: block.name,
      input: safeParseJson(block.inputBuffer),
    };
  });

  return {
    id: messageId || randomId('msg'),
    type: 'message',
    role: 'assistant',
    model,
    stop_reason: stopReason,
    stop_sequence: null,
    content,
    usage,
  };

  function processRawEvent(raw: string): void {
    if (!raw || raw.startsWith(':')) return;

    // Extract event type and data
    let eventType = '';
    let dataStr = '';
    for (const line of raw.split('\n')) {
      if (line.startsWith('event:')) {
        eventType = line.slice(6).trim();
      } else if (line.startsWith('data:')) {
        dataStr = line.slice(5).trim();
      }
    }

    if (!dataStr || dataStr === '[DONE]') return;

    let data: any;
    try {
      data = JSON.parse(dataStr);
    } catch {
      return;
    }

    switch (eventType || data?.type) {
      case 'message_start': {
        const msg = data.message;
        if (msg) {
          messageId = msg.id ?? messageId;
          model = msg.model ?? model;
        }
        break;
      }
      case 'content_block_start': {
        const block = data.content_block;
        if (block?.type === 'text') {
          blocks[data.index] = { type: 'text', text: block.text ?? '' };
        } else if (block?.type === 'tool_use') {
          blocks[data.index] = {
            type: 'tool_use',
            id: block.id ?? '',
            name: block.name ?? '',
            inputBuffer: '',
          };
        }
        break;
      }
      case 'content_block_delta': {
        const idx = data.index;
        const delta = data.delta;
        if (delta && blocks[idx]) {
          if (delta.type === 'text_delta' && blocks[idx].type === 'text') {
            (blocks[idx] as TextBlock).text += delta.text ?? '';
          } else if (delta.type === 'input_json_delta' && blocks[idx].type === 'tool_use') {
            (blocks[idx] as ToolUseBlock).inputBuffer += delta.partial_json ?? '';
          }
        }
        break;
      }
      case 'content_block_stop': {
        // Block finalized — nothing to do, it's already accumulated.
        break;
      }
      case 'message_delta': {
        if (data.delta?.stop_reason) {
          stopReason = data.delta.stop_reason;
        }
        if (data.delta?.usage ?? data.usage) {
          const u = data.delta?.usage ?? data.usage;
          usage = {
            input_tokens: u.input_tokens,
            output_tokens: u.output_tokens,
            reasoning_tokens: u.reasoning_tokens,
          };
        }
        break;
      }
      case 'message_stop': {
        if (data.stop_reason) {
          stopReason = data.stop_reason;
        }
        break;
      }
    }
  }
}

function emptyResponse(): AnthropicResponse {
  return {
    id: randomId('msg'),
    type: 'message',
    role: 'assistant',
    model: '',
    stop_reason: null,
    stop_sequence: null,
    content: [],
  };
}

function safeParseJson(value: string): unknown {
  if (!value) return {};
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}
