/**
 * Hook for managing SSE stream connections.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { createEventSource, parseSSEStream } from '../ClaudeAgentService';

interface UseSSEStreamOptions {
  onMessage?: (text: string) => void;
  onError?: (error: string) => void;
  onDone?: () => void;
  onStatus?: (status: string) => void;
}

interface UseSSEStreamReturn {
  isStreaming: boolean;
  error: string | null;
  connect: (sessionId: string) => void;
  disconnect: () => void;
}

export const useSSEStream = (options: UseSSEStreamOptions = {}): UseSSEStreamReturn => {
  const { onMessage, onError, onDone, onStatus } = options;
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  const disconnect = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    setIsStreaming(false);
  }, []);

  const connect = useCallback(
    (sessionId: string) => {
      // Close any existing connection
      disconnect();
      setError(null);
      setIsStreaming(true);

      const eventSource = createEventSource(sessionId);
      eventSourceRef.current = eventSource;

      parseSSEStream(
        eventSource,
        (text) => {
          onMessage?.(text);
        },
        (err) => {
          setError(err);
          setIsStreaming(false);
          onError?.(err);
        },
        () => {
          setIsStreaming(false);
          onDone?.();
        },
        (status) => {
          onStatus?.(status);
        },
      );
    },
    [disconnect, onMessage, onError, onDone, onStatus],
  );

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  return {
    isStreaming,
    error,
    connect,
    disconnect,
  };
};
