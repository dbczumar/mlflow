/**
 * Types for Claude Agent integration.
 */

import type { ModelTrace } from '../model-trace-explorer/ModelTrace.types';

/**
 * Represents a message in the Claude chat.
 */
export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  isStreaming?: boolean;
}

/**
 * Request body for starting a trace analysis.
 */
export interface AnalyzeRequest {
  trace_context: string;
  prompt?: string;
  session_id?: string;
}

/**
 * Response from the analyze endpoint.
 */
export interface AnalyzeResponse {
  session_id: string;
  stream_url: string;
}

/**
 * Request body for sending a follow-up message.
 */
export interface MessageRequest {
  session_id: string;
  message: string;
}

/**
 * SSE event data types.
 */
export interface SSEMessageEvent {
  text: string;
}

export interface SSEDoneEvent {
  status: 'complete';
}

export interface SSEErrorEvent {
  error: string;
}

/**
 * Health check response.
 */
export interface HealthCheckResponse {
  status: string;
  claude_available: string;
  config_exists: string;
}

/**
 * Claude Agent context state.
 */
export interface ClaudeAgentState {
  isDrawerOpen: boolean;
  traceContext: ModelTrace | null;
  sessionId: string | null;
  messages: ChatMessage[];
  isStreaming: boolean;
  error: string | null;
  isClaudeAvailable: boolean | null;
}

/**
 * Claude Agent context actions.
 */
export interface ClaudeAgentActions {
  openDrawer: (trace: ModelTrace) => void;
  closeDrawer: () => void;
  sendMessage: (message: string) => void;
  startAnalysis: (prompt?: string) => Promise<void>;
  reset: () => void;
}

/**
 * Combined context type.
 */
export type ClaudeAgentContextType = ClaudeAgentState & ClaudeAgentActions;
