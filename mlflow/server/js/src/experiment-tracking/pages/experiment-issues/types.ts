/**
 * Issue state enum matching backend proto IssueState
 */
export type IssueState = 'draft' | 'open' | 'closed';

/**
 * Issue entity matching backend proto Issue message
 */
export interface Issue {
  issue_id: string;
  experiment_id: string;
  name: string;
  description?: string;
  state: IssueState;
  creation_time?: number;
  last_update_time?: number;
  tags?: Record<string, string>;
}

/**
 * Response from searchIssues API
 */
export interface SearchIssuesResponse {
  issues: Issue[];
  next_page_token?: string;
}

/**
 * Response from createIssue API
 */
export interface CreateIssueResponse {
  issue: Issue;
}

/**
 * Response from getIssue API
 */
export interface GetIssueResponse {
  issue: Issue;
}

/**
 * Response from updateIssue API
 */
export interface UpdateIssueResponse {
  issue: Issue;
}

/**
 * Tab names for the issue detail panel
 */
export type IssueDetailTab = 'monitor' | 'traces' | 'evaluation-runs' | 'comments';

/**
 * Metric associated with a run
 */
export interface RunMetric {
  key: string;
  value: number;
}

/**
 * RunInfo matching backend proto RunInfo message (subset of fields we use)
 */
export interface LinkedRunInfo {
  run_id: string;
  run_name?: string;
  experiment_id: string;
  start_time?: number;
  end_time?: number;
  status?: string;
  lifecycle_stage?: string;
}

/**
 * Linked evaluation run with metrics
 */
export interface LinkedEvaluationRun {
  info: LinkedRunInfo;
  metrics: RunMetric[];
}

/**
 * Response from getIssueLinkedRuns API
 */
export interface GetIssueLinkedRunsResponse {
  runs: LinkedRunInfo[];
  linked_runs?: LinkedEvaluationRun[];
}
