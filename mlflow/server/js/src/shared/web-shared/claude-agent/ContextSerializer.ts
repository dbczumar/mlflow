/**
 * Serializes ClaudeContext into a human-readable format for Claude analysis.
 * Supports multiple context types: trace, run, experiment, etc.
 */

import type { ModelTrace } from '../model-trace-explorer/ModelTrace.types';
import type { ClaudeContext } from './types';
import { serializeTraceContext } from './TraceContextSerializer';

/**
 * Serialize navigation context to text.
 */
const serializeNavigation = (navigation: ClaudeContext['navigation']): string => {
  if (!navigation) return '';

  let text = '\n## Navigation Context\n\n';

  if (navigation.experimentId) {
    text += `- Experiment ID: ${navigation.experimentId}\n`;
  }
  if (navigation.experimentName) {
    text += `- Experiment Name: ${navigation.experimentName}\n`;
  }
  if (navigation.page) {
    text += `- Current Page: ${navigation.page}\n`;
  }
  if (navigation.filters) {
    text += '\n### Active Filters\n';
    if (navigation.filters.searchQuery) {
      text += `- Search Query: "${navigation.filters.searchQuery}"\n`;
    }
    if (navigation.filters.status && navigation.filters.status.length > 0) {
      text += `- Status Filter: ${navigation.filters.status.join(', ')}\n`;
    }
    if (navigation.filters.tags && Object.keys(navigation.filters.tags).length > 0) {
      text += '- Tags:\n';
      for (const [key, value] of Object.entries(navigation.filters.tags)) {
        text += `  - ${key}: ${value}\n`;
      }
    }
  }

  return text;
};

/**
 * Serialize run context to text.
 */
const serializeRunContext = (data: Record<string, unknown>): string => {
  let text = '# Run Information\n\n';

  if (data['runInfo']) {
    const runInfo = data['runInfo'] as Record<string, unknown>;
    text += `- Run ID: ${runInfo['run_uuid'] ?? runInfo['run_id'] ?? 'unknown'}\n`;
    if (runInfo['run_name']) {
      text += `- Run Name: ${runInfo['run_name']}\n`;
    }
    if (runInfo['status']) {
      text += `- Status: ${runInfo['status']}\n`;
    }
    if (runInfo['start_time']) {
      text += `- Start Time: ${new Date(runInfo['start_time'] as number).toISOString()}\n`;
    }
    if (runInfo['end_time']) {
      text += `- End Time: ${new Date(runInfo['end_time'] as number).toISOString()}\n`;
    }
    if (runInfo['lifecycle_stage']) {
      text += `- Lifecycle Stage: ${runInfo['lifecycle_stage']}\n`;
    }
  }

  if (data['params'] && Array.isArray(data['params'])) {
    text += '\n## Parameters\n';
    for (const param of data['params']) {
      text += `- ${param.key}: ${param.value}\n`;
    }
  }

  if (data['metrics'] && Array.isArray(data['metrics'])) {
    text += '\n## Metrics\n';
    for (const metric of data['metrics']) {
      text += `- ${metric.key}: ${metric.value}\n`;
    }
  }

  if (data['tags'] && Array.isArray(data['tags'])) {
    const userTags = data['tags'].filter((tag: { key: string }) => !tag.key.startsWith('mlflow.'));
    if (userTags.length > 0) {
      text += '\n## Tags\n';
      for (const tag of userTags) {
        text += `- ${tag.key}: ${tag.value}\n`;
      }
    }
  }

  return text;
};

/**
 * Serialize experiment context to text.
 */
const serializeExperimentContext = (data: Record<string, unknown>): string => {
  let text = '# Experiment Information\n\n';

  if (data['experiment']) {
    const exp = data['experiment'] as Record<string, unknown>;
    text += `- Experiment ID: ${exp['experiment_id'] ?? 'unknown'}\n`;
    if (exp['name']) {
      text += `- Name: ${exp['name']}\n`;
    }
    if (exp['lifecycle_stage']) {
      text += `- Lifecycle Stage: ${exp['lifecycle_stage']}\n`;
    }
  }

  if (data['runCount'] !== undefined) {
    text += `- Total Runs: ${data['runCount']}\n`;
  }

  if (data['traceCount'] !== undefined) {
    text += `- Total Traces: ${data['traceCount']}\n`;
  }

  return text;
};

/**
 * Serialize traces list context to text.
 */
const serializeTracesListContext = (data: Record<string, unknown>): string => {
  let text = '# Traces List\n\n';

  if (data['totalTraces'] !== undefined) {
    text += `- Total Traces: ${data['totalTraces']}\n`;
  }

  if (data['errorCount'] !== undefined) {
    text += `- Error Count: ${data['errorCount']}\n`;
  }

  if (data['traces'] && Array.isArray(data['traces'])) {
    text += `- Displaying: ${data['traces'].length} traces\n`;
  }

  return text;
};

/**
 * Serialize runs list context to text.
 */
const serializeRunsListContext = (data: Record<string, unknown>): string => {
  let text = '# Runs List\n\n';

  if (data['totalRuns'] !== undefined) {
    text += `- Total Runs: ${data['totalRuns']}\n`;
  }

  if (data['runs'] && Array.isArray(data['runs'])) {
    text += `- Displaying: ${data['runs'].length} runs\n`;
  }

  return text;
};

/**
 * Serialize model context to text.
 */
const serializeModelContext = (data: Record<string, unknown>): string => {
  let text = '# Model Information\n\n';

  if (data['modelName']) {
    text += `- Model Name: ${data['modelName']}\n`;
  }
  if (data['version']) {
    text += `- Version: ${data['version']}\n`;
  }
  if (data['stage']) {
    text += `- Stage: ${data['stage']}\n`;
  }

  return text;
};

/**
 * Serialize sessions list context to text.
 */
const serializeSessionsListContext = (data: Record<string, unknown>): string => {
  let text = '# Chat Sessions List\n\n';
  text += 'User is viewing a list of chat sessions (multi-turn LLM conversations).\n\n';

  if (data['totalSessions'] !== undefined) {
    text += `- Total Sessions: ${data['totalSessions']}\n`;
  }

  if (data['sessions'] && Array.isArray(data['sessions'])) {
    text += `- Displaying: ${data['sessions'].length} sessions\n`;

    // Show summary of sessions
    const sessions = data['sessions'] as Array<Record<string, unknown>>;
    if (sessions.length > 0) {
      text += '\n## Session Summaries\n\n';
      for (const session of sessions.slice(0, 10)) {
        // Limit to first 10
        text += `### Session ${session['sessionId']}\n`;
        if (session['requestPreview']) {
          text += `- First message: "${session['requestPreview']}"\n`;
        }
        if (session['turns'] !== undefined) {
          text += `- Turns: ${session['turns']}\n`;
        }
        if (session['tokens'] !== undefined) {
          text += `- Total tokens: ${session['tokens']}\n`;
        }
        if (session['sessionDuration']) {
          text += `- Duration: ${session['sessionDuration']}\n`;
        }
        text += '\n';
      }
      if (sessions.length > 10) {
        text += `... and ${sessions.length - 10} more sessions\n`;
      }
    }
  }

  return text;
};

/**
 * Serialize single session context to text.
 */
const serializeSessionContext = (data: Record<string, unknown>): string => {
  let text = '# Chat Session Details\n\n';
  text += 'User is viewing a single chat session (multi-turn LLM conversation).\n\n';

  if (data['sessionId']) {
    text += `- Session ID: ${data['sessionId']}\n`;
  }

  if (data['turns'] !== undefined) {
    text += `- Total Turns: ${data['turns']}\n`;
  }

  if (data['totalTokens'] !== undefined) {
    text += `- Total Tokens: ${data['totalTokens']}\n`;
  }

  // Serialize conversation turns
  if (data['traces'] && Array.isArray(data['traces'])) {
    const traces = data['traces'] as Array<Record<string, unknown>>;
    text += `\n## Conversation (${traces.length} turns)\n\n`;

    for (let i = 0; i < traces.length; i++) {
      const trace = traces[i];
      text += `### Turn ${i + 1}\n`;

      // Include trace_id so Claude can fetch full details for this turn
      if (trace['trace_id']) {
        text += `- Trace ID: ${trace['trace_id']}\n`;
      }

      if (trace['request_time']) {
        text += `- Time: ${trace['request_time']}\n`;
      }
      if (trace['status']) {
        text += `- Status: ${trace['status']}\n`;
      }
      if (trace['execution_time_ms'] !== undefined) {
        text += `- Execution Time: ${trace['execution_time_ms']}ms\n`;
      }

      // Try to get request/response previews
      if (trace['request_preview']) {
        text += `- Request: "${trace['request_preview']}"\n`;
      }
      if (trace['response_preview']) {
        text += `- Response: "${trace['response_preview']}"\n`;
      }

      text += '\n';
    }
  }

  return text;
};

/**
 * Serialize issue context to text.
 */
const serializeIssueContext = (data: Record<string, unknown>): string => {
  let text = '# Issue Details\n\n';
  text += 'User is viewing an issue that tracks a quality problem found in LLM traces.\n\n';

  if (data['issue_id']) {
    text += `- Issue ID: ${data['issue_id']}\n`;
  }

  if (data['name']) {
    text += `- Name: ${data['name']}\n`;
  }

  if (data['state']) {
    text += `- State: ${data['state']}\n`;
  }

  if (data['description']) {
    text += `\n## Description\n\n${data['description']}\n`;
  }

  if (data['creation_time']) {
    text += `\n- Created: ${new Date(data['creation_time'] as number).toISOString()}\n`;
  }

  if (data['last_update_time']) {
    text += `- Last Updated: ${new Date(data['last_update_time'] as number).toISOString()}\n`;
  }

  // Tags if present
  if (data['tags'] && typeof data['tags'] === 'object') {
    const tags = data['tags'] as Record<string, string>;
    const tagEntries = Object.entries(tags);
    if (tagEntries.length > 0) {
      text += '\n## Tags\n\n';
      for (const [key, value] of tagEntries) {
        text += `- ${key}: ${value}\n`;
      }
    }
  }

  // Judge information if available
  if (data['judge']) {
    const judge = data['judge'] as Record<string, unknown>;
    text += '\n## Issue Judge (Scorer)\n\n';
    if (judge['scorer_name']) {
      text += `- Scorer Name: ${judge['scorer_name']}\n`;
    }
    if (judge['model']) {
      text += `- Model: ${judge['model']}\n`;
    }
    if (judge['prompt']) {
      text += `\n### Judge Prompt\n\n\`\`\`\n${judge['prompt']}\n\`\`\`\n`;
    }
  }

  // Linked traces count
  if (data['linkedTracesCount'] !== undefined) {
    text += `\n- Linked Traces: ${data['linkedTracesCount']}\n`;
  }

  // Linked evaluation runs
  if (data['linkedRuns'] && Array.isArray(data['linkedRuns'])) {
    const runs = data['linkedRuns'] as Array<Record<string, unknown>>;
    if (runs.length > 0) {
      text += `\n## Linked Evaluation Runs (${runs.length})\n\n`;
      for (const run of runs.slice(0, 5)) {
        // Limit to first 5
        text += `- Run: ${run['run_name'] || run['run_id']}\n`;
      }
      if (runs.length > 5) {
        text += `... and ${runs.length - 5} more runs\n`;
      }
    }
  }

  // Comments summary
  if (data['commentsCount'] !== undefined) {
    text += `\n- Comments: ${data['commentsCount']}\n`;
  }

  return text;
};

/**
 * Main function to serialize a ClaudeContext to a Claude-friendly format.
 */
export const serializeContext = (context: ClaudeContext): string => {
  let text = '';

  // Add type-specific serialization
  switch (context.type) {
    case 'trace':
      if (context.data) {
        text += serializeTraceContext(context.data as ModelTrace);
      }
      break;

    case 'run':
      if (context.data) {
        text += serializeRunContext(context.data as Record<string, unknown>);
      }
      break;

    case 'experiment':
      if (context.data) {
        text += serializeExperimentContext(context.data as Record<string, unknown>);
      }
      break;

    case 'traces-list':
      if (context.data) {
        text += serializeTracesListContext(context.data as Record<string, unknown>);
      }
      break;

    case 'runs-list':
      if (context.data) {
        text += serializeRunsListContext(context.data as Record<string, unknown>);
      }
      break;

    case 'model':
      if (context.data) {
        text += serializeModelContext(context.data as Record<string, unknown>);
      }
      break;

    case 'sessions-list':
      if (context.data) {
        text += serializeSessionsListContext(context.data as Record<string, unknown>);
      }
      break;

    case 'session':
      if (context.data) {
        text += serializeSessionContext(context.data as Record<string, unknown>);
      }
      break;

    case 'issue':
      if (context.data) {
        text += serializeIssueContext(context.data as Record<string, unknown>);
      }
      break;

    case 'none':
    default:
      text += '# MLflow Assistant\n\n';
      text += 'User is browsing MLflow. No specific item is selected.\n';
      text += `Current page URL: ${window.location.hash || '/'}\n`;
      text +=
        '\nYou can help with general MLflow questions, navigation, or wait for the user to select a specific trace, run, or experiment for detailed analysis.\n';
  }

  // Add navigation context if available
  if (context.navigation) {
    text += serializeNavigation(context.navigation);
  }

  // Add summary if not already included
  if (context.summary && !text.includes(context.summary)) {
    text = `# ${context.summary}\n\n` + text;
  }

  return text;
};
