import { useDesignSystemTheme, Typography } from '@databricks/design-system';
import type { Assessment, IssueAssessment, ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { NullCell } from './NullCell';
import { StackedComponents } from './StackedComponents';

const ISSUE_NAME_METADATA_KEY = 'mlflow.issue.name';
const MAX_VISIBLE_ISSUES = 3;
const MAX_ISSUE_NAME_LENGTH = 15;

const RAINBOW_GRADIENT = 'linear-gradient(90deg, #64B5F6, #BA68C8, #E57373)';

const truncateIssueName = (name: string): string => {
  if (name.length <= MAX_ISSUE_NAME_LENGTH) {
    return name;
  }
  return `${name.slice(0, MAX_ISSUE_NAME_LENGTH)}...`;
};

const isIssueAssessment = (assessment: Assessment): assessment is IssueAssessment => {
  return 'issue' in assessment && assessment.issue !== undefined;
};

const getIssueName = (assessment: IssueAssessment): string => {
  return assessment.metadata?.[ISSUE_NAME_METADATA_KEY] || assessment.assessment_name;
};

interface IssueBadgeProps {
  name: string;
}

const IssueBadge = ({ name }: IssueBadgeProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <span
      css={{
        display: 'inline-flex',
        alignItems: 'center',
        padding: `1px 6px`,
        borderRadius: 4,
        background: RAINBOW_GRADIENT,
        fontSize: theme.typography.fontSizeSm,
        color: theme.colors.textPrimary,
        whiteSpace: 'nowrap',
        maxWidth: '100%',
        position: 'relative',
        // Create gradient border effect using background-clip
        '&::before': {
          content: '""',
          position: 'absolute',
          inset: 1,
          borderRadius: 3,
          backgroundColor: theme.colors.backgroundPrimary,
        },
      }}
      title={name}
    >
      <span css={{ position: 'relative', zIndex: 1 }}>{truncateIssueName(name)}</span>
    </span>
  );
};

const getIssueAssessments = (traceInfo?: ModelTraceInfoV3): IssueAssessment[] => {
  if (!traceInfo?.assessments) {
    return [];
  }
  return traceInfo.assessments.filter(
    (assessment): assessment is IssueAssessment => isIssueAssessment(assessment) && assessment.valid !== false,
  );
};

interface IssuesBadgeListProps {
  traceInfo?: ModelTraceInfoV3;
  isComparing: boolean;
}

const IssuesBadgeList = ({ traceInfo, isComparing }: IssuesBadgeListProps) => {
  const { theme } = useDesignSystemTheme();
  const issueAssessments = getIssueAssessments(traceInfo);

  if (issueAssessments.length === 0) {
    return <NullCell isComparing={isComparing} />;
  }

  const visibleIssues = issueAssessments.slice(0, MAX_VISIBLE_ISSUES);
  const remainingCount = issueAssessments.length - MAX_VISIBLE_ISSUES;

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        flexWrap: 'wrap',
        columnGap: theme.spacing.xs,
        rowGap: theme.spacing.xs,
      }}
    >
      {visibleIssues.map((issue) => (
        <IssueBadge key={issue.assessment_id} name={getIssueName(issue)} />
      ))}
      {remainingCount > 0 && (
        <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
          + {remainingCount} {remainingCount === 1 ? 'issue' : 'issues'}
        </Typography.Text>
      )}
    </div>
  );
};

export interface IssuesCellRendererProps {
  currentTraceInfo?: ModelTraceInfoV3;
  otherTraceInfo?: ModelTraceInfoV3;
  isComparing: boolean;
}

export const IssuesCellRenderer = ({ currentTraceInfo, otherTraceInfo, isComparing }: IssuesCellRendererProps) => {
  return (
    <StackedComponents
      first={<IssuesBadgeList traceInfo={currentTraceInfo} isComparing={isComparing} />}
      second={isComparing && <IssuesBadgeList traceInfo={otherTraceInfo} isComparing={isComparing} />}
    />
  );
};
