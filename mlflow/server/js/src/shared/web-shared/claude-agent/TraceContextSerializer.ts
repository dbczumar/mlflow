/**
 * Serializes ModelTrace data into a human-readable format for Claude analysis.
 */

import type {
  ModelTrace,
  ModelTraceSpan,
  ModelTraceSpanNode,
  ModelTraceEvent,
  ModelTraceInfoV3,
} from '../model-trace-explorer/ModelTrace.types';
import { isV3ModelTraceInfo } from '../model-trace-explorer/ModelTraceExplorer.utils';

/**
 * Format a timestamp to a readable string.
 */
const formatTimestamp = (timestamp: number | string): string => {
  if (typeof timestamp === 'string') {
    return timestamp;
  }
  // Nanoseconds to milliseconds conversion for V3 spans
  const ms = timestamp > 1e15 ? timestamp / 1e6 : timestamp;
  return new Date(ms).toISOString();
};

/**
 * Get duration in milliseconds from span.
 */
const getSpanDuration = (span: ModelTraceSpan): number => {
  if ('start_time_unix_nano' in span && 'end_time_unix_nano' in span) {
    // V3 format (nanoseconds)
    return (parseInt(span.end_time_unix_nano) - parseInt(span.start_time_unix_nano)) / 1e6;
  }
  // V2 format (milliseconds)
  return span.end_time - span.start_time;
};

/**
 * Get span ID from span.
 */
const getSpanId = (span: ModelTraceSpan): string => {
  if ('span_id' in span) {
    return span.span_id;
  }
  return span.context?.span_id ?? 'unknown';
};

/**
 * Get span status.
 */
const getSpanStatus = (span: ModelTraceSpan): { code: string; message?: string } => {
  if ('status' in span && typeof span.status === 'object' && span.status !== null && 'code' in span.status) {
    return span.status;
  }
  if ('status_code' in span) {
    return { code: span.status_code ?? 'OK', message: span.status_message ?? undefined };
  }
  if ('status' in span && typeof span.status === 'object' && span.status !== null) {
    const status = span.status as { description?: string };
    return { code: status.description ?? 'UNKNOWN' };
  }
  return { code: 'OK' };
};

/**
 * Get parent span ID.
 */
const getParentSpanId = (span: ModelTraceSpan): string | null => {
  return span.parent_id ?? span.parent_span_id ?? null;
};

/**
 * Extract exceptions from span events.
 */
const extractExceptions = (events?: ModelTraceEvent[]): { type: string; message: string; stacktrace?: string }[] => {
  if (!events) return [];

  return events
    .filter((event) => event.name === 'exception')
    .map((event) => ({
      type: event.attributes?.['exception.type'] ?? 'Unknown',
      message: event.attributes?.['exception.message'] ?? '',
      stacktrace: event.attributes?.['exception.stacktrace'],
    }));
};

/**
 * Serialize span inputs/outputs, handling various formats.
 */
const serializeValue = (value: unknown, maxLength = 2000): string => {
  if (value === undefined || value === null) {
    return 'null';
  }
  const str = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
  if (str.length > maxLength) {
    return str.substring(0, maxLength) + '... (truncated)';
  }
  return str;
};

/**
 * Serialize a single span to text.
 */
const serializeSpan = (span: ModelTraceSpan, indent = ''): string => {
  const spanId = getSpanId(span);
  const status = getSpanStatus(span);
  const duration = getSpanDuration(span);
  const parentId = getParentSpanId(span);
  const exceptions = extractExceptions(span.events);

  // Extract inputs/outputs from attributes
  const inputs = span.attributes?.['mlflow.spanInputs'] ?? span.inputs ?? null;
  const outputs = span.attributes?.['mlflow.spanOutputs'] ?? span.outputs ?? null;
  const spanType = span.attributes?.['mlflow.spanType'] ?? span.span_type ?? 'UNKNOWN';

  let text = `${indent}## Span: ${span.name}\n`;
  text += `${indent}- ID: ${spanId}\n`;
  text += `${indent}- Type: ${spanType}\n`;
  text += `${indent}- Status: ${status.code}${status.message ? ` (${status.message})` : ''}\n`;
  text += `${indent}- Duration: ${duration.toFixed(2)}ms\n`;
  if (parentId) {
    text += `${indent}- Parent: ${parentId}\n`;
  }

  // Add inputs
  if (inputs !== null) {
    text += `${indent}### Inputs\n`;
    text += `${indent}\`\`\`json\n${serializeValue(inputs)}\n${indent}\`\`\`\n`;
  }

  // Add outputs
  if (outputs !== null) {
    text += `${indent}### Outputs\n`;
    text += `${indent}\`\`\`json\n${serializeValue(outputs)}\n${indent}\`\`\`\n`;
  }

  // Add exceptions
  if (exceptions.length > 0) {
    text += `${indent}### Exceptions\n`;
    for (const exc of exceptions) {
      text += `${indent}- **${exc.type}**: ${exc.message}\n`;
      if (exc.stacktrace) {
        text += `${indent}\`\`\`\n${exc.stacktrace}\n${indent}\`\`\`\n`;
      }
    }
  }

  // Add relevant attributes (excluding already shown ones)
  const excludedAttrs = ['mlflow.spanInputs', 'mlflow.spanOutputs', 'mlflow.spanType'];
  const attrs = Object.entries(span.attributes ?? {}).filter(([key]) => !excludedAttrs.includes(key));

  if (attrs.length > 0) {
    text += `${indent}### Attributes\n`;
    for (const [key, value] of attrs) {
      text += `${indent}- ${key}: ${serializeValue(value, 200)}\n`;
    }
  }

  return text;
};

/**
 * Serialize trace info to text.
 */
const serializeTraceInfo = (info: ModelTrace['info']): string => {
  let text = '# Trace Information\n\n';

  if (isV3ModelTraceInfo(info)) {
    const v3Info = info as ModelTraceInfoV3;
    text += `- Trace ID: ${v3Info.trace_id}\n`;
    text += `- Status: ${v3Info.state}\n`;
    text += `- Request Time: ${v3Info.request_time}\n`;
    if (v3Info.execution_duration) {
      text += `- Duration: ${v3Info.execution_duration}\n`;
    }

    // Add tags
    if (Object.keys(v3Info.tags).length > 0) {
      text += '\n## Tags\n';
      for (const [key, value] of Object.entries(v3Info.tags)) {
        text += `- ${key}: ${value}\n`;
      }
    }

    // Add metadata
    if (v3Info.trace_metadata && Object.keys(v3Info.trace_metadata).length > 0) {
      text += '\n## Metadata\n';
      for (const [key, value] of Object.entries(v3Info.trace_metadata)) {
        text += `- ${key}: ${value}\n`;
      }
    }
  } else {
    // V2 format
    text += `- Request ID: ${info.request_id ?? 'unknown'}\n`;
    text += `- Status: ${info.status ?? 'unknown'}\n`;
    if (info.timestamp_ms) {
      text += `- Timestamp: ${new Date(info.timestamp_ms).toISOString()}\n`;
    }
    if (info.execution_time_ms) {
      text += `- Execution Time: ${info.execution_time_ms}ms\n`;
    }
  }

  return text;
};

/**
 * Build span hierarchy for better context.
 */
const buildSpanHierarchy = (spans: ModelTraceSpan[]): Map<string | null, ModelTraceSpan[]> => {
  const hierarchy = new Map<string | null, ModelTraceSpan[]>();

  for (const span of spans) {
    const parentId = getParentSpanId(span);
    const children = hierarchy.get(parentId) ?? [];
    children.push(span);
    hierarchy.set(parentId, children);
  }

  return hierarchy;
};

/**
 * Serialize spans recursively in hierarchy.
 */
const serializeSpansRecursive = (
  hierarchy: Map<string | null, ModelTraceSpan[]>,
  parentId: string | null,
  level = 0,
): string => {
  const children = hierarchy.get(parentId) ?? [];
  const indent = '  '.repeat(level);
  let text = '';

  for (const span of children) {
    text += serializeSpan(span, indent);
    text += '\n';
    // Recursively serialize children
    text += serializeSpansRecursive(hierarchy, getSpanId(span), level + 1);
  }

  return text;
};

/**
 * Main function to serialize a ModelTrace to a Claude-friendly format.
 */
export const serializeTraceContext = (trace: ModelTrace): string => {
  let text = '';

  // Add trace info
  text += serializeTraceInfo(trace.info);
  text += '\n---\n\n';

  // Add spans in hierarchical order
  text += '# Spans\n\n';

  const spans = trace.data?.spans ?? [];
  if (spans.length === 0) {
    text += 'No span data available.\n';
  } else {
    const hierarchy = buildSpanHierarchy(spans);
    text += serializeSpansRecursive(hierarchy, null);
  }

  return text;
};

/**
 * Check if trace has any errors.
 */
export const traceHasErrors = (trace: ModelTrace): boolean => {
  // Check trace-level status
  if (isV3ModelTraceInfo(trace.info)) {
    if ((trace.info as ModelTraceInfoV3).state === 'ERROR') {
      return true;
    }
  }

  // Check span-level errors
  const spans = trace.data?.spans ?? [];
  for (const span of spans) {
    const status = getSpanStatus(span);
    if (status.code === 'STATUS_CODE_ERROR' || status.code === 'ERROR') {
      return true;
    }
    // Check for exception events
    const exceptions = extractExceptions(span.events);
    if (exceptions.length > 0) {
      return true;
    }
  }

  return false;
};
