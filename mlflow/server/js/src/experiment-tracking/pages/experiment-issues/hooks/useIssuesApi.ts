import { useQuery, useMutation, useQueryClient, type UseQueryResult } from '@databricks/web-shared/query-client';
import { UnknownError, type PredefinedError } from '@databricks/web-shared/errors';
import type { Issue, IssueState, SearchIssuesResponse, LinkedEvaluationRun } from '../types';
import { searchIssues, getIssue, createIssue, updateIssue, deleteIssue, getIssueLinkedRuns } from '../api';

const ISSUES_QUERY_KEY = 'mlflow.issues';

/**
 * Hook to search issues for an experiment.
 * Fetches all issues at once (no state filter) and filters client-side.
 */
export function useSearchIssues(experimentId?: string): UseQueryResult<Issue[], PredefinedError> {
  return useQuery<Issue[], PredefinedError>({
    queryKey: [ISSUES_QUERY_KEY, 'search', experimentId],
    queryFn: async () => {
      if (!experimentId) {
        throw new UnknownError('Experiment ID is required');
      }
      const response: SearchIssuesResponse = await searchIssues(experimentId);
      return response.issues || [];
    },
    enabled: !!experimentId,
    staleTime: 60 * 1000, // 1 minute - avoid frequent refetches
    refetchOnWindowFocus: false, // Don't refetch on window focus
    refetchOnMount: false, // Don't refetch when component remounts
  });
}

/**
 * Hook to get a single issue by ID
 */
export function useGetIssue(issueId?: string): UseQueryResult<Issue | null, PredefinedError> {
  return useQuery<Issue | null, PredefinedError>({
    queryKey: [ISSUES_QUERY_KEY, 'get', issueId],
    queryFn: async () => {
      if (!issueId) {
        return null;
      }
      const response = await getIssue(issueId);
      return response.issue;
    },
    enabled: !!issueId,
    staleTime: 30 * 1000, // 30 seconds
  });
}

/**
 * Hook to create a new issue
 */
export function useCreateIssueMutation() {
  const queryClient = useQueryClient();

  return useMutation<
    Issue,
    PredefinedError,
    {
      experimentId: string;
      name: string;
      description?: string;
      state?: IssueState;
      tags?: Record<string, string>;
    }
  >({
    mutationFn: async ({ experimentId, name, description, state, tags }) => {
      const response = await createIssue(experimentId, name, description, state, tags);
      return response.issue;
    },
    onSuccess: (_data, variables) => {
      // Invalidate the search query to refetch the list
      queryClient.invalidateQueries({
        queryKey: [ISSUES_QUERY_KEY, 'search', variables.experimentId],
      });
    },
  });
}

/**
 * Hook to update an existing issue
 */
export function useUpdateIssueMutation() {
  const queryClient = useQueryClient();

  return useMutation<
    Issue,
    PredefinedError,
    {
      issueId: string;
      experimentId: string;
      updates: {
        name?: string;
        description?: string;
        state?: IssueState;
        tags?: Record<string, string>;
      };
    }
  >({
    mutationFn: async ({ issueId, updates }) => {
      const response = await updateIssue(issueId, updates);
      return response.issue;
    },
    onSuccess: (data, variables) => {
      // Update the individual issue cache
      queryClient.setQueryData([ISSUES_QUERY_KEY, 'get', variables.issueId], data);
      // Invalidate the search query to refetch the list
      queryClient.invalidateQueries({
        queryKey: [ISSUES_QUERY_KEY, 'search', variables.experimentId],
      });
    },
  });
}

/**
 * Hook to delete an issue
 */
export function useDeleteIssueMutation() {
  const queryClient = useQueryClient();

  return useMutation<
    void,
    PredefinedError,
    {
      issueId: string;
      experimentId: string;
    }
  >({
    mutationFn: async ({ issueId }) => {
      await deleteIssue(issueId);
    },
    onSuccess: (_data, variables) => {
      // Remove the individual issue from cache
      queryClient.removeQueries({
        queryKey: [ISSUES_QUERY_KEY, 'get', variables.issueId],
      });
      // Invalidate the search query to refetch the list
      queryClient.invalidateQueries({
        queryKey: [ISSUES_QUERY_KEY, 'search', variables.experimentId],
      });
    },
  });
}

/**
 * Hook to get evaluation runs linked to an issue
 */
export function useGetLinkedEvaluationRuns(
  issueId?: string,
): UseQueryResult<LinkedEvaluationRun[], PredefinedError> {
  return useQuery<LinkedEvaluationRun[], PredefinedError>({
    queryKey: [ISSUES_QUERY_KEY, 'linked-runs', issueId],
    queryFn: async () => {
      if (!issueId) {
        throw new UnknownError('Issue ID is required');
      }
      const response = await getIssueLinkedRuns(issueId);

      // Use 'runs' for run info (always available) and extract metrics from 'linked_runs'
      const runs = response.runs || [];
      const linkedRuns = response.linked_runs || [];

      // Create a map of run_id to metrics from linked_runs
      const metricsMap = new Map<string, { key: string; value: number }[]>();
      for (const linkedRun of linkedRuns) {
        // Handle both snake_case and camelCase field names from proto serialization
        const runId = linkedRun.info?.run_id || (linkedRun.info as any)?.runId;
        const metrics = linkedRun.metrics || [];
        if (runId) {
          metricsMap.set(runId, metrics);
        }
      }

      // Combine run info with metrics
      return runs.map((run) => ({
        info: run,
        metrics: metricsMap.get(run.run_id) || [],
      }));
    },
    enabled: !!issueId,
    staleTime: 60 * 1000, // 1 minute
  });
}
