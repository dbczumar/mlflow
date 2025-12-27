import { useQuery, useMutation, useQueryClient, type UseQueryResult } from '@databricks/web-shared/query-client';
import { UnknownError, type PredefinedError } from '@databricks/web-shared/errors';
import type { IssueComment, SearchIssueCommentsResponse } from '../types';
import {
  searchIssueComments,
  getIssueComment,
  createIssueComment,
  updateIssueComment,
  deleteIssueComment,
} from '../api';

const ISSUE_COMMENTS_QUERY_KEY = 'mlflow.issue-comments';

/**
 * Hook to search comments for an issue.
 */
export function useSearchIssueComments(issueId?: string): UseQueryResult<IssueComment[], PredefinedError> {
  return useQuery<IssueComment[], PredefinedError>({
    queryKey: [ISSUE_COMMENTS_QUERY_KEY, 'search', issueId],
    queryFn: async () => {
      if (!issueId) {
        throw new UnknownError('Issue ID is required');
      }
      const response: SearchIssueCommentsResponse = await searchIssueComments(issueId);
      return response.comments || [];
    },
    enabled: !!issueId,
    staleTime: 30 * 1000, // 30 seconds
    refetchOnWindowFocus: false,
  });
}

/**
 * Hook to get a single comment by ID
 */
export function useGetIssueComment(commentId?: string): UseQueryResult<IssueComment | null, PredefinedError> {
  return useQuery<IssueComment | null, PredefinedError>({
    queryKey: [ISSUE_COMMENTS_QUERY_KEY, 'get', commentId],
    queryFn: async () => {
      if (!commentId) {
        return null;
      }
      const response = await getIssueComment(commentId);
      return response.comment;
    },
    enabled: !!commentId,
    staleTime: 30 * 1000,
  });
}

/**
 * Hook to create a new comment
 */
export function useCreateIssueCommentMutation() {
  const queryClient = useQueryClient();

  return useMutation<
    IssueComment,
    PredefinedError,
    {
      issueId: string;
      content: string;
      author?: string;
    }
  >({
    mutationFn: async ({ issueId, content, author }) => {
      const response = await createIssueComment(issueId, content, author);
      return response.comment;
    },
    onSuccess: (_data, variables) => {
      // Invalidate the search query to refetch the list
      queryClient.invalidateQueries({
        queryKey: [ISSUE_COMMENTS_QUERY_KEY, 'search', variables.issueId],
      });
    },
  });
}

/**
 * Hook to update an existing comment
 */
export function useUpdateIssueCommentMutation() {
  const queryClient = useQueryClient();

  return useMutation<
    IssueComment,
    PredefinedError,
    {
      commentId: string;
      issueId: string;
      content: string;
    }
  >({
    mutationFn: async ({ commentId, content }) => {
      const response = await updateIssueComment(commentId, content);
      return response.comment;
    },
    onSuccess: (data, variables) => {
      // Update the individual comment cache
      queryClient.setQueryData([ISSUE_COMMENTS_QUERY_KEY, 'get', variables.commentId], data);
      // Invalidate the search query to refetch the list
      queryClient.invalidateQueries({
        queryKey: [ISSUE_COMMENTS_QUERY_KEY, 'search', variables.issueId],
      });
    },
  });
}

/**
 * Hook to delete a comment
 */
export function useDeleteIssueCommentMutation() {
  const queryClient = useQueryClient();

  return useMutation<
    void,
    PredefinedError,
    {
      commentId: string;
      issueId: string;
    }
  >({
    mutationFn: async ({ commentId }) => {
      await deleteIssueComment(commentId);
    },
    onSuccess: (_data, variables) => {
      // Remove the individual comment from cache
      queryClient.removeQueries({
        queryKey: [ISSUE_COMMENTS_QUERY_KEY, 'get', variables.commentId],
      });
      // Invalidate the search query to refetch the list
      queryClient.invalidateQueries({
        queryKey: [ISSUE_COMMENTS_QUERY_KEY, 'search', variables.issueId],
      });
    },
  });
}
