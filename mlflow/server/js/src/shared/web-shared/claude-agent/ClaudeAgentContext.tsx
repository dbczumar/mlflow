/**
 * React Context for Claude Agent state and actions.
 */

import { createContext, useContext, type ReactNode } from 'react';
import type { ClaudeAgentContextType } from './types';
import { useClaudeAgent } from './hooks/useClaudeAgent';

const ClaudeAgentContext = createContext<ClaudeAgentContextType | null>(null);

/**
 * Provider component for Claude Agent context.
 */
export const ClaudeAgentProvider = ({ children }: { children: ReactNode }) => {
  const claudeAgent = useClaudeAgent();

  return <ClaudeAgentContext.Provider value={claudeAgent}>{children}</ClaudeAgentContext.Provider>;
};

/**
 * Hook to access Claude Agent context.
 * Must be used within a ClaudeAgentProvider.
 */
export const useClaudeAgentContext = (): ClaudeAgentContextType => {
  const context = useContext(ClaudeAgentContext);
  if (!context) {
    throw new Error('useClaudeAgentContext must be used within a ClaudeAgentProvider');
  }
  return context;
};

/**
 * Optional hook that returns null if not within a provider.
 * Useful for components that may or may not be in a Claude Agent context.
 */
export const useClaudeAgentContextOptional = (): ClaudeAgentContextType | null => {
  return useContext(ClaudeAgentContext);
};
