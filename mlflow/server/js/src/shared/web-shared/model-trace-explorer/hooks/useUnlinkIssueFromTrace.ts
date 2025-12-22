import type { IssueAssessment } from '../ModelTrace.types';
import { useDeleteAssessment } from './useDeleteAssessment';

export const useUnlinkIssueFromTrace = ({
  issueAssessment,
  onSuccess,
  onError,
}: {
  issueAssessment?: IssueAssessment;
  onSuccess?: () => void;
  onError?: (error: any) => void;
}) => {
  const { deleteAssessmentMutation, isLoading } = useDeleteAssessment({
    assessment: issueAssessment,
    onSuccess,
    onError,
    skip: !issueAssessment,
  });

  return {
    unlinkIssueFromTrace: deleteAssessmentMutation,
    isLoading,
  };
};
