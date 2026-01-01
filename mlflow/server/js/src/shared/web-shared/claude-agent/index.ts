/**
 * Claude Agent integration for MLflow UI.
 *
 * This module provides components for AI-powered trace analysis
 * using Claude Code Agent SDK.
 */

// Global Claude Context and Provider (New - recommended)
export { GlobalClaudeProvider, useGlobalClaude, useGlobalClaudeOptional } from './GlobalClaudeContext';
export { GlobalClaudeLayout } from './GlobalClaudeLayout';
export { GlobalClaudeChatPanel } from './GlobalClaudeChatPanel';
export { GlobalClaudeButton } from './GlobalClaudeButton';
export { serializeContext } from './ContextSerializer';

// Legacy Context and Provider (for backward compatibility)
export { ClaudeAgentProvider, useClaudeAgentContext, useClaudeAgentContextOptional } from './ClaudeAgentContext';

// Components
export { ClaudeAgentTabContent, ClaudeAgentDrawer } from './ClaudeAgentDrawer';
export { ClaudeAgentChatPanel } from './ClaudeAgentChatPanel';
export { AskClaudeButton } from './AskClaudeButton';

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
  // New global types
  ClaudeContextType,
  ClaudeContext,
  GlobalClaudeAgentState,
  GlobalClaudeAgentActions,
  GlobalClaudeAgentContextType,
} from './types';
