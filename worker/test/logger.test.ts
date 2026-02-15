import { describe, expect, it, vi } from 'vitest';
import {
  deriveTurnIndex,
  buildLogKey,
  writeLogEntry,
  teeAndCollectStream,
  LogEntry,
} from '../src/logger';
import { AnthropicMessage } from '../src/types';

// ---------------------------------------------------------------------------
// deriveTurnIndex
// ---------------------------------------------------------------------------

describe('deriveTurnIndex', () => {
  it('returns 0 for an initial request with only a user message', () => {
    const messages: AnthropicMessage[] = [
      { role: 'user', content: 'Hello' },
    ];
    expect(deriveTurnIndex(messages)).toBe(0);
  });

  it('returns 1 after one assistant turn', () => {
    const messages: AnthropicMessage[] = [
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: [{ type: 'text', text: 'Hi!' }] },
      { role: 'user', content: 'Follow up' },
    ];
    expect(deriveTurnIndex(messages)).toBe(1);
  });

  it('returns 3 for a multi-turn conversation', () => {
    const messages: AnthropicMessage[] = [
      { role: 'user', content: 'Turn 0' },
      { role: 'assistant', content: 'Response 0' },
      { role: 'user', content: 'Turn 1' },
      { role: 'assistant', content: 'Response 1' },
      { role: 'user', content: 'Turn 2' },
      { role: 'assistant', content: 'Response 2' },
      { role: 'user', content: 'Turn 3' },
    ];
    expect(deriveTurnIndex(messages)).toBe(3);
  });

  it('returns 0 for empty messages', () => {
    expect(deriveTurnIndex([])).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// buildLogKey
// ---------------------------------------------------------------------------

describe('buildLogKey', () => {
  it('constructs the expected R2 key path', () => {
    const entry = {
      requestId: 'req_abc123',
      sessionId: 'sess_xyz789',
      turnIndex: 2,
      timestamp: '2026-02-15T12:30:00.000Z',
    } as LogEntry;

    expect(buildLogKey(entry)).toBe('logs/2026-02-15/sess_xyz789/2_req_abc123.json');
  });
});

// ---------------------------------------------------------------------------
// writeLogEntry
// ---------------------------------------------------------------------------

describe('writeLogEntry', () => {
  function makeEntry(overrides: Partial<LogEntry> = {}): LogEntry {
    return {
      requestId: 'req_001',
      sessionId: 'sess_001',
      turnIndex: 0,
      timestamp: '2026-02-15T10:00:00.000Z',
      provider: 'anthropic',
      originalModel: 'claude-sonnet-4-5-20250929',
      wireModel: 'claude-sonnet-4-5-20250929',
      request: { model: 'claude-sonnet-4-5-20250929', messages: [{ role: 'user', content: 'Hi' }] } as any,
      response: {
        id: 'msg_001',
        type: 'message',
        role: 'assistant',
        model: 'claude-sonnet-4-5-20250929',
        stop_reason: 'end_turn',
        stop_sequence: null,
        content: [{ type: 'text', text: 'Hello!' }],
      },
      latencyMs: 150,
      streaming: false,
      ...overrides,
    };
  }

  it('writes to the correct key with metadata', async () => {
    const putMock = vi.fn().mockResolvedValue(undefined);
    const bucket = { put: putMock } as unknown as R2Bucket;

    await writeLogEntry(bucket, makeEntry());

    expect(putMock).toHaveBeenCalledOnce();
    const [key, body, options] = putMock.mock.calls[0];
    expect(key).toBe('logs/2026-02-15/sess_001/0_req_001.json');
    expect(JSON.parse(body)).toMatchObject({ requestId: 'req_001', sessionId: 'sess_001' });
    expect(options.customMetadata.sessionId).toBe('sess_001');
    expect(options.customMetadata.turnIndex).toBe('0');
  });

  it('does not throw when bucket.put fails', async () => {
    const bucket = {
      put: vi.fn().mockRejectedValue(new Error('R2 unavailable')),
    } as unknown as R2Bucket;

    // Should not throw
    await expect(writeLogEntry(bucket, makeEntry())).resolves.toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// teeAndCollectStream
// ---------------------------------------------------------------------------

describe('teeAndCollectStream', () => {
  /** Helper to build an Anthropic SSE stream from events. */
  function buildSSEStream(events: Array<{ event: string; data: unknown }>): ReadableStream<Uint8Array> {
    const encoder = new TextEncoder();
    return new ReadableStream({
      start(controller) {
        for (const { event, data } of events) {
          controller.enqueue(encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`));
        }
        controller.close();
      },
    });
  }

  it('reassembles a simple text response from SSE events', async () => {
    const events = [
      {
        event: 'message_start',
        data: {
          type: 'message_start',
          message: { id: 'msg_test', type: 'message', role: 'assistant', model: 'claude-3', content: [] },
        },
      },
      {
        event: 'content_block_start',
        data: { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } },
      },
      {
        event: 'content_block_delta',
        data: { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'Hello ' } },
      },
      {
        event: 'content_block_delta',
        data: { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'world!' } },
      },
      {
        event: 'content_block_stop',
        data: { type: 'content_block_stop', index: 0 },
      },
      {
        event: 'message_delta',
        data: {
          type: 'message_delta',
          delta: { stop_reason: 'end_turn', usage: { input_tokens: 10, output_tokens: 5 } },
        },
      },
      {
        event: 'message_stop',
        data: { type: 'message_stop' },
      },
    ];

    const response = new Response(buildSSEStream(events), {
      headers: { 'content-type': 'text/event-stream' },
    });

    const { clientResponse, collected } = teeAndCollectStream(response);

    // Client should get the full stream
    const clientText = await clientResponse.text();
    expect(clientText).toContain('Hello ');
    expect(clientText).toContain('world!');

    // Collected should be a reassembled AnthropicResponse
    const assembled = await collected;
    expect(assembled.id).toBe('msg_test');
    expect(assembled.model).toBe('claude-3');
    expect(assembled.stop_reason).toBe('end_turn');
    expect(assembled.content).toHaveLength(1);
    expect(assembled.content[0]).toEqual({ type: 'text', text: 'Hello world!' });
    expect(assembled.usage).toEqual({ input_tokens: 10, output_tokens: 5 });
  });

  it('reassembles a response with tool_use blocks', async () => {
    const events = [
      {
        event: 'message_start',
        data: {
          type: 'message_start',
          message: { id: 'msg_tools', type: 'message', role: 'assistant', model: 'claude-3', content: [] },
        },
      },
      {
        event: 'content_block_start',
        data: {
          type: 'content_block_start',
          index: 0,
          content_block: { type: 'text', text: '' },
        },
      },
      {
        event: 'content_block_delta',
        data: { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'Let me search.' } },
      },
      {
        event: 'content_block_stop',
        data: { type: 'content_block_stop', index: 0 },
      },
      {
        event: 'content_block_start',
        data: {
          type: 'content_block_start',
          index: 1,
          content_block: { type: 'tool_use', id: 'tool_1', name: 'search', input: {} },
        },
      },
      {
        event: 'content_block_delta',
        data: {
          type: 'content_block_delta',
          index: 1,
          delta: { type: 'input_json_delta', partial_json: '{"query":' },
        },
      },
      {
        event: 'content_block_delta',
        data: {
          type: 'content_block_delta',
          index: 1,
          delta: { type: 'input_json_delta', partial_json: '"hello"}' },
        },
      },
      {
        event: 'content_block_stop',
        data: { type: 'content_block_stop', index: 1 },
      },
      {
        event: 'message_delta',
        data: { type: 'message_delta', delta: { stop_reason: 'tool_use' } },
      },
      {
        event: 'message_stop',
        data: { type: 'message_stop' },
      },
    ];

    const response = new Response(buildSSEStream(events), {
      headers: { 'content-type': 'text/event-stream' },
    });

    const { collected } = teeAndCollectStream(response);
    const assembled = await collected;

    expect(assembled.id).toBe('msg_tools');
    expect(assembled.stop_reason).toBe('tool_use');
    expect(assembled.content).toHaveLength(2);
    expect(assembled.content[0]).toEqual({ type: 'text', text: 'Let me search.' });
    expect(assembled.content[1]).toEqual({
      type: 'tool_use',
      id: 'tool_1',
      name: 'search',
      input: { query: 'hello' },
    });
  });

  it('returns an empty response when response has no body', async () => {
    const response = new Response(null, { status: 200 });

    const { clientResponse, collected } = teeAndCollectStream(response);
    const assembled = await collected;

    expect(clientResponse.status).toBe(200);
    expect(assembled.content).toEqual([]);
    expect(assembled.type).toBe('message');
  });
});
