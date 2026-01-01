/**
 * Main hook for Claude Agent functionality.
 */

import { useCallback, useEffect, useState, useRef } from 'react';
import type { ModelTrace } from '../../model-trace-explorer/ModelTrace.types';
import type { ChatMessage, ClaudeAgentState } from '../types';
import { serializeTraceContext } from '../TraceContextSerializer';
import { startAnalysis, sendMessageStream, checkHealth } from '../ClaudeAgentService';
import { useSSEStream } from './useSSEStream';

interface UseClaudeAgentReturn extends ClaudeAgentState {
  openClaudeTab: (trace: ModelTrace) => void;
  closeClaudeTab: () => void;
  sendMessage: (message: string) => void;
  startAnalysis: (prompt?: string) => Promise<void>;
  reset: () => void;
}

const generateMessageId = (): string => {
  return `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
};

export const useClaudeAgent = (): UseClaudeAgentReturn => {
  const [isClaudeTabActive, setIsClaudeTabActive] = useState(false);
  const [traceContext, setTraceContext] = useState<ModelTrace | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isClaudeAvailable, setIsClaudeAvailable] = useState<boolean | null>(null);

  // Use ref to track current streaming message
  const streamingMessageRef = useRef<string>('');

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
  }, []);

  const handleStreamError = useCallback((errorMsg: string) => {
    setError(errorMsg);
    setIsStreaming(false);
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
  });

  // Check Claude availability on mount
  useEffect(() => {
    checkHealth()
      .then((health) => {
        setIsClaudeAvailable(health.claude_available === 'true' || health.claude_available === 'True');
      })
      .catch(() => {
        setIsClaudeAvailable(false);
      });
  }, []);

  const openClaudeTab = useCallback((trace: ModelTrace) => {
    setTraceContext(trace);
    setIsClaudeTabActive(true);
    setError(null);
  }, []);

  const closeClaudeTab = useCallback(() => {
    setIsClaudeTabActive(false);
    disconnectSSE();
  }, [disconnectSSE]);

  const reset = useCallback(() => {
    setTraceContext(null);
    setSessionId(null);
    setMessages([]);
    setIsStreaming(false);
    setError(null);
    streamingMessageRef.current = '';
    disconnectSSE();
  }, [disconnectSSE]);

  const handleStartAnalysis = useCallback(
    async (prompt?: string) => {
      if (!traceContext) {
        setError('No trace context available');
        return;
      }

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
        const serializedTrace = serializeTraceContext(traceContext);
        const response = await startAnalysis({
          trace_context: serializedTrace,
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
    [traceContext, sessionId, connectSSE, handleStreamError],
  );

  const handleSendMessage = useCallback(
    (message: string) => {
      if (!sessionId) {
        setError('No active session');
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
      );
    },
    [sessionId, appendToStreamingMessage, handleStreamError, finalizeStreamingMessage],
  );

  return {
    isClaudeTabActive,
    traceContext,
    sessionId,
    messages,
    isStreaming,
    error,
    isClaudeAvailable,
    openClaudeTab,
    closeClaudeTab,
    sendMessage: handleSendMessage,
    startAnalysis: handleStartAnalysis,
    reset,
  };
};
