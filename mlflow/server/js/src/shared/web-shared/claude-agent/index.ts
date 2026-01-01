/**
 * Claude Agent integration for MLflow UI.
 *
 * This module provides components for AI-powered trace analysis
 * using Claude Code Agent SDK.
 */

// Context and Provider
export { ClaudeAgentProvider, useClaudeAgentContext, useClaudeAgentContextOptional } from './ClaudeAgentContext';

// Components
export { ClaudeAgentDrawer } from './ClaudeAgentDrawer';
export { ClaudeAgentChatPanel } from './ClaudeAgentChatPanel';

// Hooks
export { useClaudeAgent } from './hooks/useClaudeAgent';
export { useSSEStream } from './hooks/useSSEStream';

// Services
export * from './ClaudeAgentService';

// Utilities
export { serializeTraceContext, traceHasErrors } from './TraceContextSerializer';

// Types
export type {
  ChatMessage,
  AnalyzeRequest,
  AnalyzeResponse,
  MessageRequest,
  HealthCheckResponse,
  ClaudeAgentState,
  ClaudeAgentActions,
  ClaudeAgentContextType,
} from './types';
