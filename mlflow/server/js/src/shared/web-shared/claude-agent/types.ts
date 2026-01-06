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

// ============================================================================
// Global Claude Context Types
// ============================================================================

/**
 * Context types that Claude can understand.
 */
export type ClaudeContextType =
  | 'trace'
  | 'run'
  | 'experiment'
  | 'traces-list'
  | 'runs-list'
  | 'sessions-list'
  | 'session'
  | 'issue'
  | 'model'
  | 'none';

/**
 * Context data that can be passed to Claude.
 */
export interface ClaudeContext {
  /** Type of context being viewed */
  type: ClaudeContextType;
  /** Human-readable summary (e.g., "Trace tr-abc123" or "Run xyz") */
  summary: string;
  /** The actual data - type depends on context type */
  data: ModelTrace | Record<string, unknown> | null;
  /** Navigation context */
  navigation?: {
    experimentId?: string;
    experimentName?: string;
    page?: string;
    filters?: {
      searchQuery?: string;
      status?: string[];
      tags?: Record<string, string>;
    };
  };
}

/**
 * Setup status for the assistant.
 */
export type AssistantSetupStatus = 'unknown' | 'not-configured' | 'configured';

/**
 * Global Claude Agent state.
 */
export interface GlobalClaudeAgentState {
  /** Whether the Claude panel is open */
  isPanelOpen: boolean;
  /** Current context (set by pages) */
  context: ClaudeContext;
  /** Session ID for conversation continuity */
  sessionId: string | null;
  /** Chat message history */
  messages: ChatMessage[];
  /** Whether a response is being streamed */
  isStreaming: boolean;
  /** Error message if any */
  error: string | null;
  /** Whether Claude backend is available */
  isClaudeAvailable: boolean | null;
  /** Current tool usage status (e.g., "Reading file...", "Searching...") */
  currentStatus: string | null;
  /** Setup status for the assistant */
  setupStatus: AssistantSetupStatus;
  /** Whether to show the setup wizard */
  showSetupWizard: boolean;
}

/**
 * Global Claude Agent actions.
 */
export interface GlobalClaudeAgentActions {
  /** Open the Claude panel */
  openPanel: () => void;
  /** Close the Claude panel */
  closePanel: () => void;
  /** Set the current context (called by pages) */
  setContext: (context: ClaudeContext) => void;
  /** Send a message to Claude */
  sendMessage: (message: string) => void;
  /** Start analysis with optional prompt */
  startAnalysis: (prompt?: string) => Promise<void>;
  /** Reset the conversation */
  reset: () => void;
  /** Complete the setup wizard */
  completeSetup: () => void;
  /** Show the setup wizard again */
  openSetup: () => void;
}

/**
 * Combined global context type.
 */
export type GlobalClaudeAgentContextType = GlobalClaudeAgentState & GlobalClaudeAgentActions;

// ============================================================================
// Legacy Types (for backward compatibility during transition)
// ============================================================================

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
  isClaudeTabActive: boolean;
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
  openClaudeTab: (trace: ModelTrace) => void;
  closeClaudeTab: () => void;
  sendMessage: (message: string) => void;
  startAnalysis: (prompt?: string) => Promise<void>;
  reset: () => void;
}

/**
 * Combined context type.
 */
export type ClaudeAgentContextType = ClaudeAgentState & ClaudeAgentActions;
