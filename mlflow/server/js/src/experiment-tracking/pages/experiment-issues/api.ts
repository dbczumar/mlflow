import { fetchOrFail, getAjaxUrl } from '../../../common/utils/FetchUtils';
import type {
  Issue,
  IssueState,
  IssueComment,
  IssueJudge,
  SearchIssuesResponse,
  CreateIssueResponse,
  GetIssueResponse,
  UpdateIssueResponse,
  GetIssueLinkedRunsResponse,
  SearchIssueCommentsResponse,
  CreateIssueCommentResponse,
  GetIssueCommentResponse,
  UpdateIssueCommentResponse,
  CreateJudgeFromIssueResponse,
} from './types';

/**
 * Search issues for an experiment
 */
export async function searchIssues(
  experimentId: string,
  states?: IssueState[],
  maxResults?: number,
  pageToken?: string,
): Promise<SearchIssuesResponse> {
  const body: { [key: string]: unknown } = {
    experiment_id: experimentId,
  };
  if (states && states.length > 0) {
    body['states'] = states.map((s) => s.toUpperCase());
  }
  if (maxResults) {
    body['max_results'] = maxResults;
  }
  if (pageToken) {
    body['page_token'] = pageToken;
  }

  const res = await fetch(getAjaxUrl('ajax-api/3.0/mlflow/issues/search'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  // Handle 404 as empty result (API not implemented)
  if (res.status === 404) {
    console.warn('Issues API returned 404 - endpoint may not be implemented');
    return { issues: [] };
  }

  if (!res.ok) {
    const errorText = await res.text();
    console.error('Issues API error:', res.status, errorText);
    throw new Error(`Failed to search issues: ${res.status}`);
  }

  const data = await res.json();
  return {
    ...data,
    issues: (data.issues || []).map(normalizeIssue),
  };
}

/**
 * Get a single issue by ID
 */
export async function getIssue(issueId: string): Promise<GetIssueResponse> {
  const params = new URLSearchParams();
  params.append('issue_id', issueId);

  const res = await fetchOrFail(getAjaxUrl(`ajax-api/3.0/mlflow/issues/get?${params.toString()}`));
  const data = await res.json();
  return {
    issue: normalizeIssue(data.issue),
  };
}

/**
 * Create a new issue
 */
export async function createIssue(
  experimentId: string,
  name: string,
  description?: string,
  state?: IssueState,
  tags?: Record<string, string>,
): Promise<CreateIssueResponse> {
  const body: { [key: string]: unknown } = {
    experiment_id: experimentId,
    name,
  };
  if (description) {
    body['description'] = description;
  }
  if (state) {
    body['state'] = state.toUpperCase();
  }
  if (tags) {
    body['tags'] = tags;
  }

  const res = await fetchOrFail(getAjaxUrl('ajax-api/3.0/mlflow/issues/create'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  return {
    issue: normalizeIssue(data.issue),
  };
}

/**
 * Update an existing issue
 */
export async function updateIssue(
  issueId: string,
  updates: {
    name?: string;
    description?: string;
    state?: IssueState;
    tags?: Record<string, string>;
  },
): Promise<UpdateIssueResponse> {
  const body: { [key: string]: unknown } = {
    issue_id: issueId,
  };
  if (updates.name !== undefined) {
    body['name'] = updates.name;
  }
  if (updates.description !== undefined) {
    body['description'] = updates.description;
  }
  if (updates.state !== undefined) {
    body['state'] = updates.state.toUpperCase();
  }
  if (updates.tags !== undefined) {
    body['tags'] = updates.tags;
  }

  const res = await fetchOrFail(getAjaxUrl('ajax-api/3.0/mlflow/issues/update'), {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  return {
    issue: normalizeIssue(data.issue),
  };
}

/**
 * Delete an issue
 */
export async function deleteIssue(issueId: string): Promise<void> {
  const res = await fetchOrFail(getAjaxUrl('ajax-api/3.0/mlflow/issues/delete'), {
    method: 'DELETE',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      issue_id: issueId,
    }),
  });
  await res.json();
}

/**
 * Normalize issue state from backend (uppercase) to frontend (lowercase)
 */
function normalizeIssue(issue: Issue & { state?: string }): Issue {
  return {
    ...issue,
    state: (issue.state?.toLowerCase() || 'draft') as IssueState,
  };
}

/**
 * Get evaluation runs linked to an issue
 */
export async function getIssueLinkedRuns(issueId: string): Promise<GetIssueLinkedRunsResponse> {
  const params = new URLSearchParams();
  params.append('issue_id', issueId);

  const res = await fetchOrFail(getAjaxUrl(`ajax-api/3.0/mlflow/issues/linked-runs?${params.toString()}`));
  const data = await res.json();
  return {
    runs: data.runs || [],
    linked_runs: data.linked_runs || [],
  };
}

// ========== Issue Comments API ==========

/**
 * Search comments for an issue
 */
export async function searchIssueComments(
  issueId: string,
  maxResults?: number,
  pageToken?: string,
): Promise<SearchIssueCommentsResponse> {
  const body: { [key: string]: unknown } = {
    issue_id: issueId,
  };
  if (maxResults) {
    body['max_results'] = maxResults;
  }
  if (pageToken) {
    body['page_token'] = pageToken;
  }

  const res = await fetch(getAjaxUrl('ajax-api/3.0/mlflow/issues/comments/search'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  // Handle 404 as empty result (API not implemented)
  if (res.status === 404) {
    console.warn('Issue comments API returned 404 - endpoint may not be implemented');
    return { comments: [] };
  }

  if (!res.ok) {
    const errorText = await res.text();
    console.error('Issue comments API error:', res.status, errorText);
    throw new Error(`Failed to search issue comments: ${res.status}`);
  }

  const data = await res.json();
  return {
    ...data,
    comments: data.comments || [],
  };
}

/**
 * Get a single comment by ID
 */
export async function getIssueComment(commentId: string): Promise<GetIssueCommentResponse> {
  const params = new URLSearchParams();
  params.append('comment_id', commentId);

  const res = await fetchOrFail(getAjaxUrl(`ajax-api/3.0/mlflow/issues/comments/get?${params.toString()}`));
  const data = await res.json();
  return {
    comment: data.comment,
  };
}

/**
 * Create a new comment
 */
export async function createIssueComment(
  issueId: string,
  content: string,
  author?: string,
): Promise<CreateIssueCommentResponse> {
  const body: { [key: string]: unknown } = {
    issue_id: issueId,
    content,
  };
  if (author) {
    body['author'] = author;
  }

  const res = await fetchOrFail(getAjaxUrl('ajax-api/3.0/mlflow/issues/comments/create'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  return {
    comment: data.comment,
  };
}

/**
 * Update an existing comment
 */
export async function updateIssueComment(
  commentId: string,
  content: string,
): Promise<UpdateIssueCommentResponse> {
  const body: { [key: string]: unknown } = {
    comment_id: commentId,
    content,
  };

  const res = await fetchOrFail(getAjaxUrl('ajax-api/3.0/mlflow/issues/comments/update'), {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  return {
    comment: data.comment,
  };
}

/**
 * Delete a comment
 */
export async function deleteIssueComment(commentId: string): Promise<void> {
  const res = await fetchOrFail(getAjaxUrl('ajax-api/3.0/mlflow/issues/comments/delete'), {
    method: 'DELETE',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      comment_id: commentId,
    }),
  });
  await res.json();
}

// ========== Issue Judge API ==========

/**
 * Parse serialized scorer JSON to extract issue judge details
 */
function parseSerializedScorer(serializedScorer: string): {
  issue_id: string;
  issue_name: string;
  prompt: string;
  model: string;
  description?: string;
} | null {
  try {
    const parsed = JSON.parse(serializedScorer);
    const judgeData = parsed.instructions_judge_pydantic_data;

    if (!judgeData || !judgeData.is_issue_judge) {
      return null;
    }

    return {
      issue_id: judgeData.issue_id || '',
      issue_name: judgeData.issue_name || '',
      prompt: judgeData.system_prompt || '',
      model: judgeData.model || '',
      description: parsed.description,
    };
  } catch {
    return null;
  }
}

/**
 * Create a judge from an issue
 */
export async function createJudgeFromIssue(issueId: string): Promise<IssueJudge> {
  const res = await fetchOrFail(getAjaxUrl('ajax-api/3.0/mlflow/issues/create-judge'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      issue_id: issueId,
    }),
  });
  const data: CreateJudgeFromIssueResponse = await res.json();

  const parsedDetails = parseSerializedScorer(data.scorer.serialized_scorer);
  if (!parsedDetails) {
    throw new Error('Failed to parse judge scorer data');
  }

  return {
    scorer_id: data.scorer.scorer_id || '',
    scorer_name: data.scorer.scorer_name,
    scorer_version: data.scorer.scorer_version,
    experiment_id: data.scorer.experiment_id,
    creation_time: data.scorer.creation_time,
    ...parsedDetails,
  };
}

/**
 * Get judge for an issue by looking up scorers with matching issue_id
 */
export async function getJudgeForIssue(issueId: string, experimentId: string): Promise<IssueJudge | null> {
  const params = new URLSearchParams();
  params.append('experiment_id', experimentId);

  const res = await fetch(getAjaxUrl(`ajax-api/3.0/mlflow/scorers/list?${params.toString()}`));

  // Handle 404 as no scorers
  if (res.status === 404) {
    return null;
  }

  if (!res.ok) {
    const errorText = await res.text();
    console.error('Scorers API error:', res.status, errorText);
    return null;
  }

  const data = await res.json();
  const scorers = data.scorers || [];

  // Find the scorer that is an issue judge for this issue
  for (const scorer of scorers) {
    const parsedDetails = parseSerializedScorer(scorer.serialized_scorer);
    if (parsedDetails && parsedDetails.issue_id === issueId) {
      return {
        scorer_id: scorer.scorer_id || '',
        scorer_name: scorer.scorer_name,
        scorer_version: scorer.scorer_version,
        experiment_id: scorer.experiment_id,
        creation_time: scorer.creation_time,
        ...parsedDetails,
      };
    }
  }

  return null;
}
