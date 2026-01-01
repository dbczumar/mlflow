/**
 * Service layer for Claude Agent API calls.
 */

import type { AnalyzeRequest, AnalyzeResponse, MessageRequest, HealthCheckResponse } from './types';

const API_BASE = '/api/claude-agent';

/**
 * Start a trace analysis session.
 */
export const startAnalysis = async (request: AnalyzeRequest): Promise<AnalyzeResponse> => {
  const response = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Failed to start analysis: ${error}`);
  }

  return response.json();
};

/**
 * Send a follow-up message.
 * Returns the SSE stream URL.
 */
export const sendMessage = async (request: MessageRequest): Promise<string> => {
  // The message endpoint returns a streaming response directly
  // We return the URL to connect to
  return `${API_BASE}/message`;
};

/**
 * Create an EventSource for streaming responses.
 */
export const createEventSource = (sessionId: string): EventSource => {
  return new EventSource(`${API_BASE}/stream/${sessionId}`);
};

/**
 * Send a message and get the response stream.
 * This uses fetch with streaming for POST requests.
 */
export const sendMessageStream = async (
  request: MessageRequest,
  onMessage: (text: string) => void,
  onError: (error: string) => void,
  onDone: () => void,
  onStatus?: (status: string) => void,
): Promise<void> => {
  try {
    const response = await fetch(`${API_BASE}/message`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response.text();
      onError(`Failed to send message: ${error}`);
      return;
    }

    if (!response.body) {
      onError('No response body');
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let reading = true;

    while (reading) {
      const { done, value } = await reader.read();

      if (done) {
        onDone();
        reading = false;
        continue;
      }

      buffer += decoder.decode(value, { stream: true });

      // Parse SSE events from buffer
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? ''; // Keep incomplete line in buffer

      for (const line of lines) {
        if (line.startsWith('event: ')) {
          // Event type line - we'll process the data on the next line
          continue;
        }
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          parseSSEData(data, onMessage, onError, onDone, onStatus);
        }
      }
    }
  } catch (error) {
    onError(error instanceof Error ? error.message : 'Unknown error');
  }
};

/**
 * Parse a single SSE data line.
 */
const parseSSEData = (
  data: string,
  onMessage: (text: string) => void,
  onError: (error: string) => void,
  onDone: () => void,
  onStatus?: (status: string) => void,
): void => {
  try {
    const parsed = JSON.parse(data);
    if ('text' in parsed) {
      onMessage(parsed.text);
    } else if ('error' in parsed) {
      onError(parsed.error);
    } else if ('status' in parsed) {
      if (parsed.status === 'complete') {
        onDone();
      } else {
        // Tool usage status message
        onStatus?.(parsed.status);
      }
    }
  } catch {
    // Non-JSON data, treat as plain text
    onMessage(data);
  }
};

/**
 * Check if Claude is available.
 */
export const checkHealth = async (): Promise<HealthCheckResponse> => {
  const response = await fetch(`${API_BASE}/health`);

  if (!response.ok) {
    throw new Error('Health check failed');
  }

  return response.json();
};

/**
 * Parse SSE stream from EventSource.
 */
export const parseSSEStream = (
  eventSource: EventSource,
  onMessage: (text: string) => void,
  onError: (error: string) => void,
  onDone: () => void,
  onStatus?: (status: string) => void,
): void => {
  eventSource.addEventListener('message', (event) => {
    try {
      const data = JSON.parse(event.data);
      if ('text' in data) {
        onMessage(data.text);
      }
    } catch {
      onMessage(event.data);
    }
  });

  eventSource.addEventListener('status', (event) => {
    try {
      const data = JSON.parse((event as MessageEvent).data);
      if ('status' in data && data.status !== 'complete') {
        onStatus?.(data.status);
      }
    } catch {
      // Ignore parse errors for status events
    }
  });

  eventSource.addEventListener('done', () => {
    onDone();
    eventSource.close();
  });

  eventSource.addEventListener('error', (event) => {
    if (eventSource.readyState === EventSource.CLOSED) {
      return;
    }
    try {
      const data = JSON.parse((event as MessageEvent).data);
      onError(data.error || 'Unknown error');
    } catch {
      onError('Connection error');
    }
    eventSource.close();
  });
};
