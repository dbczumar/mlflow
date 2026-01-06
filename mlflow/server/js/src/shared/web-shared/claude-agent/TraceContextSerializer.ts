/**
 * Serializes ModelTrace data into a human-readable format for Claude analysis.
 */

import type {
  ModelTrace,
  ModelTraceSpan,
  ModelTraceEvent,
  ModelTraceInfoV3,
  Assessment,
  FeedbackAssessment,
  ExpectationAssessment,
  IssueAssessment,
} from '../model-trace-explorer/ModelTrace.types';
import { isV3ModelTraceInfo } from '../model-trace-explorer/ModelTraceExplorer.utils';

/**
 * Get duration in milliseconds from span.
 */
const getSpanDuration = (span: ModelTraceSpan): number => {
  if ('start_time_unix_nano' in span && 'end_time_unix_nano' in span) {
    // V3 format (nanoseconds)
    return (parseInt(span.end_time_unix_nano, 10) - parseInt(span.start_time_unix_nano, 10)) / 1e6;
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
  if ('parent_span_id' in span) {
    return span.parent_span_id ?? null;
  }
  if ('parent_id' in span) {
    return span.parent_id ?? null;
  }
  return null;
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
 * Type guards for assessment types.
 */
const isFeedbackAssessment = (assessment: Assessment): assessment is FeedbackAssessment => {
  return 'feedback' in assessment;
};

const isExpectationAssessment = (assessment: Assessment): assessment is ExpectationAssessment => {
  return 'expectation' in assessment;
};

const isIssueAssessment = (assessment: Assessment): assessment is IssueAssessment => {
  return 'issue' in assessment;
};

/**
 * Get the value from an assessment, handling different assessment types.
 */
const getAssessmentValue = (assessment: Assessment): string => {
  if (isFeedbackAssessment(assessment)) {
    if (assessment.feedback.error) {
      return `ERROR - ${assessment.feedback.error.error_message ?? assessment.feedback.error.error_code}`;
    }
    const value = assessment.feedback.value;
    if (value === null || value === undefined) {
      return 'null';
    }
    if (Array.isArray(value)) {
      return JSON.stringify(value);
    }
    return String(value);
  }
  if (isExpectationAssessment(assessment)) {
    const expectation = assessment.expectation;
    if ('serialized_value' in expectation) {
      return expectation.serialized_value.value;
    }
    const value = expectation.value;
    if (value === null || value === undefined) {
      return 'null';
    }
    if (Array.isArray(value)) {
      return JSON.stringify(value);
    }
    return String(value);
  }
  if (isIssueAssessment(assessment)) {
    return assessment.issue.value ? 'true' : 'false';
  }
  return 'unknown';
};

/**
 * Get the assessment type label.
 */
const getAssessmentTypeLabel = (assessment: Assessment): string => {
  if (isFeedbackAssessment(assessment)) return 'feedback';
  if (isExpectationAssessment(assessment)) return 'expectation';
  if (isIssueAssessment(assessment)) return 'issue';
  return 'unknown';
};

/**
 * Serialize a single assessment to text.
 */
const serializeAssessment = (assessment: Assessment, indent = ''): string => {
  const sourceType = assessment.source.source_type;
  const sourceId = assessment.source.source_id;
  const value = getAssessmentValue(assessment);
  const typeLabel = getAssessmentTypeLabel(assessment);

  let text = `${indent}- **${assessment.assessment_name}** (${sourceType}: ${sourceId}): ${value}\n`;
  text += `${indent}  - Type: ${typeLabel}\n`;

  if (assessment.rationale) {
    // Truncate long rationales
    const rationale =
      assessment.rationale.length > 500
        ? assessment.rationale.substring(0, 500) + '... (truncated)'
        : assessment.rationale;
    text += `${indent}  - Rationale: ${rationale}\n`;
  }

  if (assessment.error) {
    text += `${indent}  - Error: ${assessment.error.error_message ?? assessment.error.error_code}\n`;
  }

  return text;
};

/**
 * Serialize multiple assessments to text.
 */
const serializeAssessments = (assessments: Assessment[], indent = '', heading = '## Assessments'): string => {
  if (!assessments || assessments.length === 0) {
    return '';
  }

  let text = `${indent}${heading}\n`;
  for (const assessment of assessments) {
    text += serializeAssessment(assessment, indent);
  }
  return text;
};

/**
 * Serialize a single span to text.
 * @param spanAssessments - Optional assessments specifically for this span
 */
const serializeSpan = (span: ModelTraceSpan, indent = '', spanAssessments?: Assessment[]): string => {
  const spanId = getSpanId(span);
  const status = getSpanStatus(span);
  const duration = getSpanDuration(span);
  const parentId = getParentSpanId(span);
  const exceptions = extractExceptions(span.events);

  // Extract inputs/outputs from attributes (V3 only has attributes, V2 may have direct props)
  const v2Inputs = 'inputs' in span ? span.inputs : null;
  const v2Outputs = 'outputs' in span ? span.outputs : null;
  const v2SpanType = 'span_type' in span ? span.span_type : null;

  const inputs = span.attributes?.['mlflow.spanInputs'] ?? v2Inputs ?? null;
  const outputs = span.attributes?.['mlflow.spanOutputs'] ?? v2Outputs ?? null;
  const spanType = span.attributes?.['mlflow.spanType'] ?? v2SpanType ?? 'UNKNOWN';

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

  // Add span-level assessments
  if (spanAssessments && spanAssessments.length > 0) {
    text += serializeAssessments(spanAssessments, indent, '### Assessments');
  }

  return text;
};

/**
 * Build a map of span_id to assessments for quick lookup.
 */
const buildSpanAssessmentsMap = (assessments: Assessment[]): Map<string, Assessment[]> => {
  const map = new Map<string, Assessment[]>();
  for (const assessment of assessments) {
    if (assessment.span_id) {
      const existing = map.get(assessment.span_id) ?? [];
      existing.push(assessment);
      map.set(assessment.span_id, existing);
    }
  }
  return map;
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

    // Add trace-level assessments (only those without span_id)
    if (v3Info.assessments && v3Info.assessments.length > 0) {
      const traceLevelAssessments = v3Info.assessments.filter((a) => !a.span_id);
      if (traceLevelAssessments.length > 0) {
        text += '\n' + serializeAssessments(traceLevelAssessments);
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
  spanAssessmentsMap: Map<string, Assessment[]>,
  parentId: string | null,
  level = 0,
): string => {
  const children = hierarchy.get(parentId) ?? [];
  const indent = '  '.repeat(level);
  let text = '';

  for (const span of children) {
    const spanId = getSpanId(span);
    const spanAssessments = spanAssessmentsMap.get(spanId);
    text += serializeSpan(span, indent, spanAssessments);
    text += '\n';
    // Recursively serialize children
    text += serializeSpansRecursive(hierarchy, spanAssessmentsMap, spanId, level + 1);
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

    // Build span assessments map for V3 traces
    let spanAssessmentsMap = new Map<string, Assessment[]>();
    if (isV3ModelTraceInfo(trace.info)) {
      const v3Info = trace.info as ModelTraceInfoV3;
      if (v3Info.assessments) {
        spanAssessmentsMap = buildSpanAssessmentsMap(v3Info.assessments);
      }
    }

    text += serializeSpansRecursive(hierarchy, spanAssessmentsMap, null);
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
