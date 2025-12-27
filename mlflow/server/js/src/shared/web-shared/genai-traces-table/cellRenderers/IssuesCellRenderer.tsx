import { useDesignSystemTheme, Typography, CheckCircleIcon, WarningIcon, XCircleIcon } from '@databricks/design-system';
import type { Assessment, IssueAssessment, ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { NullCell } from './NullCell';
import { StackedComponents } from './StackedComponents';

const ISSUE_NAME_METADATA_KEY = 'mlflow.issue.name';
const MAX_VISIBLE_ISSUES = 3;
const MAX_ISSUE_NAME_LENGTH = 15;

const RAINBOW_GRADIENT = 'linear-gradient(90deg, #64B5F6, #BA68C8, #E57373)';

// Colors for issue status indicators
const ISSUE_DETECTED_COLOR = '#E57373'; // Red/orange for detected issues
const NO_ISSUE_COLOR = '#66BB6A'; // Green for no issues

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
    (assessment): assessment is IssueAssessment =>
      isIssueAssessment(assessment) && assessment.valid !== false && assessment.issue?.value !== false,
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

/**
 * Converts a value to boolean, handling both boolean and string types.
 * This is needed because the backend may return "true"/"false" strings.
 */
const toBoolean = (value: unknown): boolean => {
  if (typeof value === 'boolean') {
    return value;
  }
  if (typeof value === 'string') {
    return value.toLowerCase() === 'true';
  }
  return Boolean(value);
};

/**
 * Finds a specific issue assessment by name in the trace info.
 * Returns the issue value (true = detected, false = not detected) or undefined if not found.
 */
const findIssueByName = (traceInfo: ModelTraceInfoV3 | undefined, issueName: string): boolean | undefined => {
  if (!traceInfo?.assessments) {
    return undefined;
  }

  for (const assessment of traceInfo.assessments) {
    if (isIssueAssessment(assessment)) {
      const name = getIssueName(assessment);
      if (name === issueName) {
        // Return the actual boolean value (true = issue detected, false = no issue)
        // Handle both boolean and string values from the backend
        const value = assessment.issue?.value;
        return value !== undefined ? toBoolean(value) : undefined;
      }
    }
  }

  return undefined;
};

interface SingleIssueValueProps {
  issueDetected: boolean | undefined;
  isComparing: boolean;
}

// Background colors matching assessment tag styling
const PASS_TAG_BACKGROUND = '#02B30214'; // tagBackgroundLime

/**
 * Renders a single issue value (Pass/Fail) with colored background tag
 * - No issue detected (false) → Pass (green)
 * - Issue detected (true) → Fail (red)
 */
const SingleIssueValue = ({ issueDetected, isComparing }: SingleIssueValueProps) => {
  const { theme } = useDesignSystemTheme();

  if (issueDetected === undefined) {
    return <NullCell isComparing={isComparing} />;
  }

  // Issue detected = Fail (red), No issue = Pass (green)
  const isPass = !issueDetected;
  const backgroundColor = isPass
    ? PASS_TAG_BACKGROUND
    : theme.isDarkMode
    ? theme.colors.red200
    : theme.colors.red200;
  const textColor = isPass
    ? theme.isDarkMode
      ? theme.colors.green400
      : theme.colors.green600
    : theme.isDarkMode
    ? theme.colors.red400
    : theme.colors.red600;
  const iconColor = textColor;

  return (
    <div
      css={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: theme.spacing.sm,
        padding: '0 8px',
        height: 20,
        width: 'fit-content',
        borderRadius: theme.legacyBorders.borderRadiusMd,
        backgroundColor,
        fontSize: theme.typography.fontSizeSm,
        svg: { width: 12, height: 12 },
      }}
    >
      {isPass ? (
        <CheckCircleIcon css={{ color: iconColor }} />
      ) : (
        <XCircleIcon css={{ color: iconColor }} />
      )}
      <span css={{ color: textColor }}>{isPass ? 'Pass' : 'Fail'}</span>
    </div>
  );
};

export interface SingleIssueCellRendererProps {
  currentTraceInfo?: ModelTraceInfoV3;
  otherTraceInfo?: ModelTraceInfoV3;
  isComparing: boolean;
  issueName: string;
}

/**
 * Renders a single issue column cell showing True/False for the issue detection status.
 * Used in expanded view where each issue has its own column.
 */
export const SingleIssueCellRenderer = ({
  currentTraceInfo,
  otherTraceInfo,
  isComparing,
  issueName,
}: SingleIssueCellRendererProps) => {
  const currentIssueDetected = findIssueByName(currentTraceInfo, issueName);
  const otherIssueDetected = isComparing ? findIssueByName(otherTraceInfo, issueName) : undefined;

  return (
    <StackedComponents
      first={<SingleIssueValue issueDetected={currentIssueDetected} isComparing={isComparing} />}
      second={isComparing && <SingleIssueValue issueDetected={otherIssueDetected} isComparing={isComparing} />}
    />
  );
};
