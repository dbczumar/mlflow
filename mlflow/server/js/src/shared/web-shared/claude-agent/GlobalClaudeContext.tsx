/**
 * Global React Context for MLflow Assistant (formerly Claude Agent).
 * Provides AI assistant functionality accessible from anywhere in MLflow.
 */

import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from 'react';

import type { AssistantSetupStatus, ChatMessage, ClaudeContext, GlobalClaudeAgentContextType } from './types';
import { serializeContext } from './ContextSerializer';
import { startAnalysis, sendMessageStream, checkHealth } from './ClaudeAgentService';
import { useSSEStream } from './hooks/useSSEStream';
import { ExperimentKind } from '../../../experiment-tracking/constants';

const DEFAULT_CONTEXT: ClaudeContext = {
  type: 'none',
  summary: '',
  data: null,
};

// localStorage key for setup status (global or per-experiment)
const SETUP_STATUS_KEY_PREFIX = 'mlflow.assistant.setupStatus';

const GlobalClaudeContext = createContext<GlobalClaudeAgentContextType | null>(null);

const generateMessageId = (): string => {
  return `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
};

/**
 * Get localStorage key for setup status.
 * If experimentId provided, returns experiment-specific key.
 * Otherwise returns global key.
 */
const getSetupStatusKey = (experimentId?: string): string => {
  if (experimentId) {
    return `${SETUP_STATUS_KEY_PREFIX}.experiment.${experimentId}`;
  }
  return `${SETUP_STATUS_KEY_PREFIX}.global`;
};

/**
 * Load setup status from localStorage.
 * If experimentId provided, loads experiment-specific status.
 */
const loadSetupStatus = (experimentId?: string): AssistantSetupStatus => {
  try {
    const key = getSetupStatusKey(experimentId);
    const stored = localStorage.getItem(key);
    if (stored === 'configured') {
      return stored;
    }
  } catch {
    // localStorage not available
  }
  return 'unknown';
};

/**
 * Load selected backend from localStorage (global only).
 */
const loadSelectedBackend = (): string | null => {
  try {
    return localStorage.getItem('mlflow.assistant.selectedBackend.global');
  } catch {
    // localStorage not available
    return null;
  }
};

/**
 * Save setup status to localStorage.
 * If experimentId provided, saves experiment-specific status.
 */
const saveSetupStatus = (status: AssistantSetupStatus, experimentId?: string): void => {
  try {
    const key = getSetupStatusKey(experimentId);
    localStorage.setItem(key, status);
  } catch {
    // localStorage not available
  }
};

/**
 * Global MLflow Assistant Provider.
 * Wrap at the app root level (MlflowRootRoute) to enable assistant functionality everywhere.
 */
export const GlobalClaudeProvider = ({ children }: { children: ReactNode }) => {
  // Panel state
  const [isPanelOpen, setIsPanelOpen] = useState(false);

  // Context state (set by pages)
  const [context, setContextState] = useState<ClaudeContext>(DEFAULT_CONTEXT);

  // Chat state (persists across navigation)
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isClaudeAvailable, setIsClaudeAvailable] = useState<boolean | null>(null);
  const [currentStatus, setCurrentStatus] = useState<string | null>(null);

  // Setup state
  const [setupStatus, setSetupStatus] = useState<AssistantSetupStatus>(() => {
    return loadSetupStatus();
  });
  const [showSetupWizard, setShowSetupWizard] = useState(false);
  const [selectedBackend, setSelectedBackend] = useState<string | null>(() => {
    return loadSelectedBackend();
  });

  // Use ref to track current streaming message
  const streamingMessageRef = useRef<string>('');
  // Use ref to track previous experiment ID and kind for auto-opening panel
  const previousExperimentIdRef = useRef<string | undefined>(undefined);
  const previousExperimentKindRef = useRef<string | undefined>(undefined);
  // Track if this is the very first mount (before any useEffect runs)
  const isFirstMountRef = useRef<boolean>(true);

  // Ensure selectedBackend is always loaded from localStorage on mount
  // This is a safety net to handle edge cases like new tabs
  useEffect(() => {
    const backend = loadSelectedBackend();
    if (backend !== selectedBackend) {
      setSelectedBackend(backend);
    }
    const status = loadSetupStatus();
    if (status !== setupStatus) {
      setSetupStatus(status);
    }
  }, []); // Run once on mount

  const appendToStreamingMessage = useCallback((text: string) => {
    streamingMessageRef.current += text;
    setMessages((prev) => {
      const lastMessage = prev[prev.length - 1];
      if (lastMessage && lastMessage.role === 'assistant' && lastMessage.isStreaming) {
        return [...prev.slice(0, -1), { ...lastMessage, content: streamingMessageRef.current }];
      }
      return prev;
    });
  }, []);

  const finalizeStreamingMessage = useCallback(() => {
    setMessages((prev) => {
      const lastMessage = prev[prev.length - 1];
      if (lastMessage && lastMessage.role === 'assistant' && lastMessage.isStreaming) {
        return [...prev.slice(0, -1), { ...lastMessage, isStreaming: false }];
      }
      return prev;
    });
    streamingMessageRef.current = '';
    setIsStreaming(false);
    setCurrentStatus(null);
  }, []);

  const handleStatus = useCallback((status: string) => {
    setCurrentStatus(status);
  }, []);

  const handleStreamError = useCallback((errorMsg: string) => {
    setError(errorMsg);
    setIsStreaming(false);
    setCurrentStatus(null);
    setMessages((prev) => {
      const lastMessage = prev[prev.length - 1];
      if (lastMessage && lastMessage.role === 'assistant' && lastMessage.isStreaming) {
        return [...prev.slice(0, -1), { ...lastMessage, content: `Error: ${errorMsg}`, isStreaming: false }];
      }
      return prev;
    });
  }, []);

  const { connect: connectSSE, disconnect: disconnectSSE } = useSSEStream({
    onMessage: appendToStreamingMessage,
    onError: handleStreamError,
    onDone: finalizeStreamingMessage,
    onStatus: handleStatus,
  });

  // Check Claude availability and setup status when context changes
  useEffect(() => {
    const experimentId = context.navigation?.experimentId;
    const previousExperimentId = previousExperimentIdRef.current;

    const experimentKind = context.navigation?.experimentKind;
    const previousExperimentKind = previousExperimentKindRef.current;

    // Check if experiment context actually changed
    const experimentChanged = experimentId !== previousExperimentId || experimentKind !== previousExperimentKind;

    // Check if this is initial mount with an experiment (previousExperimentId is undefined but current is set)
    const isInitialMount = previousExperimentId === undefined && experimentId !== undefined;

    // Check if this is the very first mount (home page or any page on load)
    const isFirstMount = isFirstMountRef.current;

    // Check global status for button variant and backend availability
    const globalStatus = loadSetupStatus();

    // Check experiment-specific status for wizard visibility
    // For experiments: check if THIS experiment's wizard is complete
    // For non-experiments: use global status
    const experimentSpecificStatus = experimentId ? loadSetupStatus(experimentId) : globalStatus;

    // Check if this is a GenAI experiment
    const isGenAIExp =
      experimentKind === ExperimentKind.GENAI_DEVELOPMENT ||
      experimentKind === ExperimentKind.GENAI_DEVELOPMENT_INFERRED;

    // Process when experiment context changes OR on initial mount OR on first mount
    if (experimentChanged || isInitialMount || isFirstMount) {
      // Reset session when experiment changes (inline to avoid circular dependency)
      setSessionId(null);
      setMessages([]);
      setIsStreaming(false);
      setError(null);
      streamingMessageRef.current = '';
      disconnectSSE();

      // Always check Claude availability, regardless of localStorage status
      // This ensures we detect Claude on PATH even on fresh load
      // NOTE: Claude availability ≠ assistant configured
      // User must still complete setup wizard (provide code path, etc.)
      checkHealth()
        .then((health) => {
          const isAvailable = health.claude_available === 'true' || health.claude_available === 'True';
          setIsClaudeAvailable(isAvailable);
          // Setup status comes from localStorage (user's actual configuration)
          // Do NOT auto-mark as configured just because Claude is available!
          setSetupStatus(globalStatus);
          // Reload selected backend from localStorage
          const backend = loadSelectedBackend();
          setSelectedBackend(backend);
        })
        .catch(() => {
          // Backend not responding or Claude not available
          setIsClaudeAvailable(false);
          // Setup status still comes from localStorage
          setSetupStatus(globalStatus);
          // Reload selected backend from localStorage
          const backend = loadSelectedBackend();
          setSelectedBackend(backend);
        });

      // Auto-open panel when navigating to a NEW GenAI experiment (only if not configured)
      if (experimentId && experimentId !== previousExperimentId) {
        // Experiment ID changed - user navigated to a different experiment
        if (isGenAIExp) {
          if (experimentSpecificStatus !== 'configured') {
            // This GenAI experiment's wizard is not complete - auto-open panel with wizard
            setIsPanelOpen(true);
            setShowSetupWizard(true);
          } else if (globalStatus === 'configured') {
            // GenAI experiment AND assistant is configured - don't auto-open, just hide wizard
            setShowSetupWizard(false);
          }
        } else if (!isGenAIExp && globalStatus === 'configured') {
          // Non-GenAI experiment AND user is configured - hide wizard
          setShowSetupWizard(false);
        }
        // If not configured, keep wizard visible (don't hide it)
      } else if (experimentKind !== previousExperimentKind) {
        // ExperimentKind changed (data loaded)
        if (isGenAIExp && experimentId) {
          if (experimentSpecificStatus !== 'configured') {
            // Just learned it's a GenAI experiment that's not configured - auto-open with wizard
            setIsPanelOpen(true);
            setShowSetupWizard(true);
          } else if (globalStatus === 'configured') {
            // Just learned it's a GenAI experiment AND assistant is configured - don't auto-open, just hide wizard
            setShowSetupWizard(false);
          }
        } else if (!isGenAIExp && globalStatus === 'configured') {
          // Not a GenAI experiment AND user is configured - hide wizard
          setShowSetupWizard(false);
        }
        // If not configured, keep wizard visible (don't hide it)
      }
    }

    // Update refs with current experiment ID and kind
    previousExperimentIdRef.current = experimentId;
    previousExperimentKindRef.current = experimentKind;
    // Mark that first mount is complete
    isFirstMountRef.current = false;
  }, [context.navigation?.experimentId, context.navigation?.experimentKind, disconnectSSE]);

  // Actions
  const openPanel = useCallback(() => {
    setIsPanelOpen(true);
    setError(null);
    // Show setup wizard ONLY if assistant not configured
    const currentSetupStatus = loadSetupStatus();
    if (currentSetupStatus !== 'configured') {
      setShowSetupWizard(true);
    } else {
      // If configured, show chat (not wizard)
      setShowSetupWizard(false);
    }
  }, []);

  const closePanel = useCallback(() => {
    setIsPanelOpen(false);
    disconnectSSE();
  }, [disconnectSSE]);

  const setContext = useCallback((newContext: ClaudeContext) => {
    setContextState(newContext);

    // If switching to global context (no experiment), close wizard ONLY if configured
    if (!newContext.navigation?.experimentId) {
      const globalStatus = loadSetupStatus();
      if (globalStatus === 'configured') {
        setShowSetupWizard(false);
      }
      // If not configured, keep wizard visible
    }
  }, []);

  const reset = useCallback(() => {
    setSessionId(null);
    setMessages([]);
    setIsStreaming(false);
    setError(null);
    streamingMessageRef.current = '';
    disconnectSSE();
  }, [disconnectSSE]);

  const completeSetup = useCallback(() => {
    const experimentId = context.navigation?.experimentId;
    setSetupStatus('configured');
    // ALWAYS save to global when assistant backend is configured
    saveSetupStatus('configured');
    // ALSO save to experiment-specific if in an experiment
    if (experimentId) {
      saveSetupStatus('configured', experimentId);
    }
    // Reload selected backend from localStorage (updated by wizard)
    setSelectedBackend(loadSelectedBackend());
    setShowSetupWizard(false);
    setIsClaudeAvailable(true);
  }, [context.navigation?.experimentId]);

  const openSetup = useCallback(() => {
    setShowSetupWizard(true);
  }, []);

  const handleStartAnalysis = useCallback(
    async (prompt?: string) => {
      setError(null);
      setIsStreaming(true);

      // Add user message if prompt provided
      if (prompt) {
        setMessages((prev) => [
          ...prev,
          {
            id: generateMessageId(),
            role: 'user',
            content: prompt,
            timestamp: new Date(),
          },
        ]);
      }

      // Add streaming assistant message placeholder
      streamingMessageRef.current = '';
      setMessages((prev) => [
        ...prev,
        {
          id: generateMessageId(),
          role: 'assistant',
          content: '',
          timestamp: new Date(),
          isStreaming: true,
        },
      ]);

      try {
        const serializedContext = serializeContext(context);
        const response = await startAnalysis({
          trace_context: serializedContext,
          prompt,
          session_id: sessionId ?? undefined,
        });

        setSessionId(response.session_id);

        // Connect to SSE stream
        connectSSE(response.session_id);
      } catch (err) {
        handleStreamError(err instanceof Error ? err.message : 'Failed to start analysis');
      }
    },
    [context, sessionId, connectSSE, handleStreamError],
  );

  const handleSendMessage = useCallback(
    (message: string) => {
      if (!sessionId) {
        // No session yet - start analysis with the message as prompt
        handleStartAnalysis(message);
        return;
      }

      setError(null);
      setIsStreaming(true);

      // Add user message
      setMessages((prev) => [
        ...prev,
        {
          id: generateMessageId(),
          role: 'user',
          content: message,
          timestamp: new Date(),
        },
      ]);

      // Add streaming assistant message placeholder
      streamingMessageRef.current = '';
      setMessages((prev) => [
        ...prev,
        {
          id: generateMessageId(),
          role: 'assistant',
          content: '',
          timestamp: new Date(),
          isStreaming: true,
        },
      ]);

      // Send message and stream response
      sendMessageStream(
        { session_id: sessionId, message },
        appendToStreamingMessage,
        handleStreamError,
        finalizeStreamingMessage,
        handleStatus,
      );
    },
    [
      sessionId,
      handleStartAnalysis,
      appendToStreamingMessage,
      handleStreamError,
      finalizeStreamingMessage,
      handleStatus,
    ],
  );

  const value: GlobalClaudeAgentContextType = {
    // State
    isPanelOpen,
    context,
    sessionId,
    messages,
    isStreaming,
    error,
    isClaudeAvailable,
    currentStatus,
    setupStatus,
    showSetupWizard,
    selectedBackend,
    // Actions
    openPanel,
    closePanel,
    setContext,
    sendMessage: handleSendMessage,
    startAnalysis: handleStartAnalysis,
    reset,
    completeSetup,
    openSetup,
  };

  return <GlobalClaudeContext.Provider value={value}>{children}</GlobalClaudeContext.Provider>;
};

/**
 * Hook to access the global Claude context.
 * Must be used within a GlobalClaudeProvider.
 */
export const useGlobalClaude = (): GlobalClaudeAgentContextType => {
  const context = useContext(GlobalClaudeContext);
  if (!context) {
    throw new Error('useGlobalClaude must be used within a GlobalClaudeProvider');
  }
  return context;
};

/**
 * Optional hook that returns null if not within a provider.
 * Useful for components that may or may not be in a global Claude context.
 */
export const useGlobalClaudeOptional = (): GlobalClaudeAgentContextType | null => {
  return useContext(GlobalClaudeContext);
};
