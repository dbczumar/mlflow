import { useQuery, useMutation, useQueryClient, type UseQueryResult } from '@databricks/web-shared/query-client';
import type { PredefinedError } from '@databricks/web-shared/errors';
import type { IssueJudge } from '../types';
import { createJudgeFromIssue, getJudgeForIssue } from '../api';

const JUDGE_QUERY_KEY = 'mlflow.issues.judge';

/**
 * Hook to get the judge for an issue
 */
export function useGetJudgeForIssue(
  issueId?: string,
  experimentId?: string,
): UseQueryResult<IssueJudge | null, PredefinedError> {
  return useQuery<IssueJudge | null, PredefinedError>({
    queryKey: [JUDGE_QUERY_KEY, 'get', issueId, experimentId],
    queryFn: async () => {
      if (!issueId || !experimentId) {
        return null;
      }
      return getJudgeForIssue(issueId, experimentId);
    },
    enabled: !!issueId && !!experimentId,
    staleTime: 60 * 1000, // 1 minute
    refetchOnWindowFocus: false,
  });
}

/**
 * Hook to create a judge from an issue
 */
export function useCreateJudgeMutation() {
  const queryClient = useQueryClient();

  return useMutation<
    IssueJudge,
    PredefinedError,
    {
      issueId: string;
      experimentId: string;
    }
  >({
    mutationFn: async ({ issueId }) => {
      return createJudgeFromIssue(issueId);
    },
    onSuccess: (data, variables) => {
      // Update the judge cache with the new judge
      queryClient.setQueryData([JUDGE_QUERY_KEY, 'get', variables.issueId, variables.experimentId], data);
    },
  });
}
