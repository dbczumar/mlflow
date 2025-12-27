import { useMemo } from 'react';

import { CloseIcon, Button, Tooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import type { Assessment, IssueAssessment } from '../ModelTrace.types';
import { useSearchIssues } from '../../../../experiment-tracking/pages/experiment-issues/hooks/useIssuesApi';
import { IssueCard } from './IssueCard';
import { IssueSelector } from './IssueSelector';
import { ModelTraceExplorerBadge } from '../ModelTraceExplorerBadge';

const ISSUE_ID_METADATA_KEY = 'mlflow.issue.id';

const isIssueAssessment = (assessment: Assessment): assessment is IssueAssessment => {
  return 'issue' in assessment && assessment.issue !== undefined;
};

const getIssueIdFromAssessment = (assessment: IssueAssessment): string | undefined => {
  return assessment.metadata?.[ISSUE_ID_METADATA_KEY];
};

export interface IssuesSectionProps {
  traceId: string;
  experimentId: string;
  assessments: Assessment[];
  activeSpanId?: string;
  onClose?: () => void;
}

export const IssuesSection = ({ traceId, experimentId, assessments, activeSpanId, onClose }: IssuesSectionProps) => {
  const { theme } = useDesignSystemTheme();

  // Filter to only valid issue assessments with value=true,
  // then deduplicate by issue ID (keeping only the latest assessment for each issue ID)
  // Note: issue_id is stored as assessment_name in the backend
  const issueAssessments = useMemo(() => {
    const validIssueAssessments = assessments.filter(
      (assessment): assessment is IssueAssessment =>
        isIssueAssessment(assessment) && assessment.valid !== false && assessment.issue?.value !== false,
    );

    // Group assessments by issue ID (assessment_name) and keep only the latest one for each issue
    const issueAssessmentsByIssueId = new Map<string, IssueAssessment>();

    validIssueAssessments.forEach((assessment) => {
      // issue_id is stored as assessment_name
      const issueId = assessment.assessment_name;

      const existing = issueAssessmentsByIssueId.get(issueId);
      if (!existing) {
        issueAssessmentsByIssueId.set(issueId, assessment);
      } else {
        // Keep the one with the later last_update_time (or create_time as fallback)
        const existingTime = new Date(existing.last_update_time || existing.create_time).getTime();
        const newTime = new Date(assessment.last_update_time || assessment.create_time).getTime();
        if (newTime > existingTime) {
          issueAssessmentsByIssueId.set(issueId, assessment);
        }
      }
    });

    return Array.from(issueAssessmentsByIssueId.values());
  }, [assessments]);

  // Get linked issue IDs for filtering in the selector
  const linkedIssueIds = useMemo(() => {
    return issueAssessments
      .map((assessment) => getIssueIdFromAssessment(assessment))
      .filter((id): id is string => id !== undefined);
  }, [issueAssessments]);

  // Fetch all issues to get full details (name, description)
  const { data: allIssues = [] } = useSearchIssues(experimentId);

  // Create a map of issue details by ID
  const issueDetailsMap = useMemo(() => {
    const map = new Map<string, typeof allIssues[0]>();
    allIssues.forEach((issue) => {
      map.set(issue.issue_id, issue);
    });
    return map;
  }, [allIssues]);

  const issueCount = issueAssessments.length;

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        marginBottom: theme.spacing.md,
        paddingBottom: theme.spacing.md,
        borderBottom: `1px solid ${theme.colors.border}`,
      }}
    >
      {/* Header */}
      <div
        css={{
          display: 'flex',
          flexDirection: 'row',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: theme.spacing.sm,
        }}
      >
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
          <Typography.Text bold>
            <FormattedMessage defaultMessage="Issues" description="Label for the issues section header" />
          </Typography.Text>
          {issueCount > 0 && <ModelTraceExplorerBadge count={issueCount} />}
        </div>
        {onClose && (
          <Tooltip
            componentId="shared.model-trace-explorer.close-issues-section-tooltip"
            content={
              <FormattedMessage
                defaultMessage="Hide issues"
                description="Tooltip for a button that closes the issues section"
              />
            }
          >
            <Button
              data-testid="close-issues-section-button"
              componentId="shared.model-trace-explorer.close-issues-section"
              size="small"
              icon={<CloseIcon />}
              onClick={onClose}
            />
          </Tooltip>
        )}
      </div>

      {/* Issue cards */}
      {issueAssessments.length > 0 && (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm, marginBottom: theme.spacing.sm }}>
          {issueAssessments.map((issueAssessment) => {
            const issueId = getIssueIdFromAssessment(issueAssessment);
            const issueDetails = issueId ? issueDetailsMap.get(issueId) : undefined;

            return (
              <IssueCard
                key={issueAssessment.assessment_id}
                issueAssessment={issueAssessment}
                issueDetails={issueDetails}
                experimentId={experimentId}
              />
            );
          })}
        </div>
      )}

      {/* Issue selector */}
      <IssueSelector
        experimentId={experimentId}
        traceId={traceId}
        spanId={activeSpanId}
        linkedIssueIds={linkedIssueIds}
      />
    </div>
  );
};
