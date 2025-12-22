import type { Issue } from '../../../../experiment-tracking/pages/experiment-issues/types';
import type { IssueAssessment } from '../ModelTrace.types';
import { useCreateAssessment } from './useCreateAssessment';

const ISSUE_NAME_METADATA_KEY = 'mlflow.issue.name';
const ISSUE_ID_METADATA_KEY = 'mlflow.issue.id';

export const useLinkIssueToTrace = ({
  traceId,
  spanId,
  onSuccess,
  onError,
}: {
  traceId: string;
  spanId?: string;
  onSuccess?: () => void;
  onError?: (error: any) => void;
}) => {
  const { createAssessmentMutation, isLoading } = useCreateAssessment({
    traceId,
    onSuccess,
    onError,
  });

  const linkIssueToTrace = (issue: Issue) => {
    // Create an IssueAssessment payload - we need to cast since the API accepts union types
    const issueAssessmentPayload: Omit<IssueAssessment, 'assessment_id' | 'create_time' | 'last_update_time'> = {
      assessment_name: issue.name,
      trace_id: traceId,
      ...(spanId && { span_id: spanId }),
      source: {
        source_type: 'HUMAN',
        source_id: 'user',
      },
      issue: {
        value: true,
      },
      metadata: {
        [ISSUE_NAME_METADATA_KEY]: issue.name,
        [ISSUE_ID_METADATA_KEY]: issue.issue_id,
      },
    };

    createAssessmentMutation({
      assessment: issueAssessmentPayload as any,
    });
  };

  return {
    linkIssueToTrace,
    isLoading,
  };
};
