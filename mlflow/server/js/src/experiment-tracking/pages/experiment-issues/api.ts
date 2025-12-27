import { fetchOrFail, getAjaxUrl } from '../../../common/utils/FetchUtils';
import type {
  Issue,
  IssueState,
  SearchIssuesResponse,
  CreateIssueResponse,
  GetIssueResponse,
  UpdateIssueResponse,
  GetIssueLinkedRunsResponse,
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
