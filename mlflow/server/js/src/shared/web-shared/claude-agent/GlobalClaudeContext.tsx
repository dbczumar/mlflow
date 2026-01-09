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
  const [setupStatus, setSetupStatus] = useState<AssistantSetupStatus>(loadSetupStatus);
  const [showSetupWizard, setShowSetupWizard] = useState(false);

  // Use ref to track current streaming message
  const streamingMessageRef = useRef<string>('');
  // Use ref to track previous experiment ID and kind for auto-opening panel
  const previousExperimentIdRef = useRef<string | undefined>(undefined);
  const previousExperimentKindRef = useRef<string | undefined>(undefined);

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

    // Process when experiment context changes OR on initial mount
    if (experimentChanged || isInitialMount) {
      // Reset session when experiment changes (inline to avoid circular dependency)
      setSessionId(null);
      setMessages([]);
      setIsStreaming(false);
      setError(null);
      streamingMessageRef.current = '';
      disconnectSSE();

      // Button variant and availability based on GLOBAL backend configuration
      if (globalStatus === 'configured') {
        checkHealth()
          .then((health) => {
            const isAvailable = health.claude_available === 'true' || health.claude_available === 'True';
            setIsClaudeAvailable(isAvailable);
            setSetupStatus('configured');
          })
          .catch(() => {
            setIsClaudeAvailable(false);
            setSetupStatus('configured'); // Keep configured status - user may just need to restart backend
          });
      } else {
        setSetupStatus('not-configured');
        setIsClaudeAvailable(false);
      }

      // Auto-open panel when navigating to a NEW GenAI experiment
      if (experimentId && experimentId !== previousExperimentId) {
        // Experiment ID changed - user navigated to a different experiment
        if (experimentSpecificStatus !== 'configured' && isGenAIExp) {
          // This GenAI experiment's wizard is not complete - auto-open panel
          setIsPanelOpen(true);
          setShowSetupWizard(true);
        } else if (!isGenAIExp) {
          // Non-GenAI experiment - make sure wizard is hidden
          setShowSetupWizard(false);
        }
      } else if (experimentKind !== previousExperimentKind) {
        // ExperimentKind changed (data loaded)
        if (isGenAIExp && experimentId && experimentSpecificStatus !== 'configured') {
          // Just learned it's a GenAI experiment that's not configured - auto-open
          setIsPanelOpen(true);
          setShowSetupWizard(true);
        } else if (!isGenAIExp) {
          // Not a GenAI experiment - hide wizard
          setShowSetupWizard(false);
        }
      }
    }

    // Update refs with current experiment ID and kind
    previousExperimentIdRef.current = experimentId;
    previousExperimentKindRef.current = experimentKind;
  }, [context.navigation?.experimentId, context.navigation?.experimentKind, disconnectSSE]);

  // Actions
  const openPanel = useCallback(() => {
    setIsPanelOpen(true);
    setError(null);
    // Show setup wizard ONLY for GenAI experiments
    // (The wizard itself will determine the appropriate starting step based on experiment state)
    setContextState((currentContext) => {
      const isGenAIExp =
        currentContext.navigation?.experimentKind &&
        (currentContext.navigation.experimentKind === ExperimentKind.GENAI_DEVELOPMENT ||
          currentContext.navigation.experimentKind === ExperimentKind.GENAI_DEVELOPMENT_INFERRED);

      if (isGenAIExp) {
        setShowSetupWizard(true);
      }
      return currentContext;
    });
  }, []);

  const closePanel = useCallback(() => {
    setIsPanelOpen(false);
    disconnectSSE();
  }, [disconnectSSE]);

  const setContext = useCallback((newContext: ClaudeContext) => {
    setContextState(newContext);

    // If switching to global context (no experiment), close the setup wizard
    if (!newContext.navigation?.experimentId) {
      setShowSetupWizard((prev) => {
        if (prev) {
          return false;
        }
        return prev;
      });
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
