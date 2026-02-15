export type Provider = 'anthropic' | 'openrouter' | 'vllm' | 'sglang';

export type ServerToolsMode = 'error' | 'enforceAnthropic' | 'emulate';
export type McpBridgeMode = 'off' | 'http-sse';

export interface CastariMetadata {
  castari?: {
    reasoning?: CastariReasoningConfig;
    web_search_options?: WebSearchOptions;
  } & Record<string, unknown>;
  [key: string]: unknown;
}

export interface CastariReasoningConfig {
  effort?: 'low' | 'medium' | 'high' | 'max';
  max_tokens?: number;
  exclude?: boolean;
  summary?: 'auto' | 'concise' | 'detailed' | 'none';
}

export interface WebSearchOptions {
  engine?: 'native' | 'exa';
  max_results?: number;
  search_context_size?: 'low' | 'medium' | 'high';
}

export interface AnthropicContentText {
  type: 'text';
  text: string;
}

export interface AnthropicContentImage {
  type: 'image';
  source:
    | { type: 'url'; url: string; media_type?: string }
    | { type: 'base64'; media_type: string; data: string };
}

export interface AnthropicContentToolResult {
  type: 'tool_result';
  tool_use_id: string;
  content?: string | AnthropicContentText[];
  is_error?: boolean;
}

export interface AnthropicContentToolUse {
  type: 'tool_use';
  id: string;
  name: string;
  input: unknown;
}

export type AnthropicContent =
  | AnthropicContentText
  | AnthropicContentImage
  | AnthropicContentToolResult
  | AnthropicContentToolUse;

export interface AnthropicMessage {
  role: 'user' | 'assistant';
  content: AnthropicContent[] | string;
}

export interface AnthropicToolDefinition {
  name?: string;
  description?: string;
  input_schema?: Record<string, unknown>;
  type?: string;
  [key: string]: unknown;
}

export interface AnthropicRequest extends CastariMetadata {
  model: string;
  system?: string | AnthropicContentText[];
  messages: AnthropicMessage[];
  tools?: AnthropicToolDefinition[];
  tool_choice?:
    | 'auto'
    | 'none'
    | { type: 'tool'; name: string }
    | { type: 'any' };
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  stop_sequences?: string[];
  metadata?: CastariMetadata | Record<string, unknown>;
  stream?: boolean;
  mcp_servers?: unknown[];
}

export interface AnthropicResponse {
  id: string;
  type: 'message';
  role: 'assistant';
  model: string;
  stop_reason: string | null;
  stop_sequence: string | null;
  content: AnthropicContent[];
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
    reasoning_tokens?: number;
  };
}

export interface OpenRouterContentText {
  type: 'text';
  text: string;
}

export interface OpenRouterContentImage {
  type: 'image_url';
  image_url: { url: string; detail?: string };
}

export type OpenRouterMessageContent = string | (OpenRouterContentText | OpenRouterContentImage)[];

export interface OpenRouterMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: OpenRouterMessageContent;
  name?: string;
  tool_call_id?: string;
  tool_calls?: OpenRouterToolCall[];
}

export interface OpenRouterToolCallFunction {
  name: string;
  arguments?: string;
}

export interface OpenRouterToolCall {
  id: string;
  type: 'function';
  function: OpenRouterToolCallFunction;
}

export interface OpenRouterToolDefinition {
  type: 'function';
  function: {
    name: string;
    description?: string;
    parameters?: Record<string, unknown>;
  };
}

export interface OpenRouterRequest {
  model: string;
  messages: OpenRouterMessage[];
  tools?: OpenRouterToolDefinition[];
  tool_choice?: 'auto' | 'none' | { type: 'function'; function: { name: string } };
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  stop?: string[];
  stream?: boolean;
  reasoning?: CastariReasoningConfig;
  plugins?: Array<{ id: string; engine?: 'native' | 'exa'; max_results?: number; search_prompt?: string }>;
  web_search_options?: WebSearchOptions;
}

export interface OpenRouterChoice {
  finish_reason: string | null;
  delta?: {
    content?: string | null;
    tool_calls?: OpenRouterToolCall[];
  };
  message?: {
    role: 'assistant';
    content: string | null;
    tool_calls?: OpenRouterToolCall[];
    reasoning?: string | null;
  };
}

export interface OpenRouterResponse {
  id: string;
  choices: OpenRouterChoice[];
  model: string;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
    reasoning_tokens?: number;
  };
}

export interface ProviderAuth {
  type: 'x-api-key' | 'bearer';
  value: string;
}

export interface WorkerConfig {
  anthropicBaseUrl: string;
  openRouterBaseUrl: string;
  vllmBaseUrl: string;
  sglangBaseUrl: string;
  serverToolsMode: ServerToolsMode;
  mcpMode: McpBridgeMode;
  defaultOpenRouterVendor: string;
}

export interface CastariHeaders {
  provider?: Provider;
  originalModel?: string;
  wireModel?: string;
}
