import { useState } from 'react';

import {
  Button,
  ChevronDownIcon,
  ChevronRightIcon,
  NewWindowIcon,
  Tooltip,
  TrashIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { GenAIMarkdownRenderer } from '@databricks/web-shared/genai-markdown-renderer';
import type { IssueAssessment } from '../ModelTrace.types';
import type { Issue } from '../../../../experiment-tracking/pages/experiment-issues/types';
import { useUnlinkIssueFromTrace } from '../hooks/useUnlinkIssueFromTrace';
import { getSourceIcon } from './utils';
import { AssessmentSourceName } from './AssessmentSourceName';
import { timeSinceStr } from './AssessmentsPane.utils';

const RAINBOW_GRADIENT = 'linear-gradient(90deg, #64B5F6, #BA68C8, #E57373)';
const ISSUE_NAME_METADATA_KEY = 'mlflow.issue.name';

const getIssueName = (assessment: IssueAssessment): string => {
  return assessment.metadata?.[ISSUE_NAME_METADATA_KEY] || assessment.assessment_name;
};

const truncateDescription = (description: string | undefined, maxLength = 100): string => {
  if (!description) {
    return '';
  }
  // Get first line
  const firstLine = description.split('\n')[0];
  if (firstLine.length <= maxLength) {
    return firstLine;
  }
  return `${firstLine.slice(0, maxLength)}...`;
};

export interface IssueCardProps {
  issueAssessment: IssueAssessment;
  issueDetails?: Issue;
  experimentId: string;
  onUnlinked?: () => void;
}

export const IssueCard = ({ issueAssessment, issueDetails, experimentId, onUnlinked }: IssueCardProps) => {
  const { theme } = useDesignSystemTheme();
  const [isExpanded, setIsExpanded] = useState(false);

  const { unlinkIssueFromTrace, isLoading: isUnlinking } = useUnlinkIssueFromTrace({
    issueAssessment,
    onSuccess: onUnlinked,
  });

  const issueName = getIssueName(issueAssessment);
  const description = issueDetails?.description;
  const rationale = issueAssessment.rationale;
  const hasDescription = Boolean(description);
  const hasRationale = Boolean(rationale);
  // Card is expandable if there's description or rationale to show
  const isExpandable = hasDescription || hasRationale;

  const SourceIcon = getSourceIcon(issueAssessment.source);

  const handleOpenInNewTab = () => {
    // Navigate to issue detail page in new tab
    // MLflow uses hash-based routing, so URL format is: /static-files#/experiments/{id}/issues?issueId={id}
    const issueId = issueDetails?.issue_id || issueAssessment.metadata?.['mlflow.issue.id'];
    const basePath = window.location.pathname; // e.g., /static-files
    const hashRoute = `/experiments/${experimentId}/issues${issueId ? `?issueId=${issueId}` : ''}`;
    const url = `${basePath}#${hashRoute}`;
    window.open(url, '_blank');
  };

  const handleUnlink = () => {
    unlinkIssueFromTrace();
  };

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        border: `1px solid transparent`,
        borderRadius: theme.borders.borderRadiusMd,
        background: RAINBOW_GRADIENT,
        padding: 1, // Border width
      }}
    >
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          backgroundColor: theme.colors.backgroundPrimary,
          borderRadius: theme.borders.borderRadiusMd - 1,
          padding: theme.spacing.sm,
        }}
      >
        {/* Header row with title and actions */}
        <div
          css={{
            display: 'flex',
            flexDirection: 'row',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: theme.spacing.sm,
          }}
        >
          <div
            css={{
              display: 'flex',
              alignItems: 'center',
              gap: theme.spacing.xs,
              flex: 1,
              minWidth: 0,
              cursor: isExpandable ? 'pointer' : 'default',
            }}
            onClick={() => isExpandable && setIsExpanded(!isExpanded)}
          >
            {isExpandable && (
              <span css={{ flexShrink: 0, display: 'flex', alignItems: 'center' }}>
                {isExpanded ? (
                  <ChevronDownIcon css={{ fontSize: 12, color: theme.colors.textSecondary }} />
                ) : (
                  <ChevronRightIcon css={{ fontSize: 12, color: theme.colors.textSecondary }} />
                )}
              </span>
            )}
            <Typography.Text bold css={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {issueName}
            </Typography.Text>
          </div>

          {/* Action buttons */}
          <div css={{ display: 'flex', gap: theme.spacing.xs, flexShrink: 0 }}>
            <Tooltip
              content={
                <FormattedMessage
                  defaultMessage="Open issue in new tab"
                  description="Tooltip for opening issue in new tab"
                />
              }
              componentId="shared.model-trace-explorer.issue-open-new-tab-tooltip"
            >
              <Button
                componentId="shared.model-trace-explorer.issue-open-new-tab"
                size="small"
                icon={<NewWindowIcon css={{ fontSize: 12 }} />}
                onClick={handleOpenInNewTab}
              />
            </Tooltip>
            <Tooltip
              content={
                <FormattedMessage
                  defaultMessage="Unlink issue from trace"
                  description="Tooltip for unlinking issue from trace"
                />
              }
              componentId="shared.model-trace-explorer.issue-unlink-tooltip"
            >
              <Button
                componentId="shared.model-trace-explorer.issue-unlink"
                size="small"
                icon={<TrashIcon css={{ fontSize: 12 }} />}
                onClick={handleUnlink}
                loading={isUnlinking}
              />
            </Tooltip>
          </div>
        </div>

        {/* Collapsed view: truncated description */}
        {!isExpanded && hasDescription && (
          <Typography.Text
            color="secondary"
            css={{
              marginTop: theme.spacing.xs,
              fontSize: theme.typography.fontSizeSm,
              overflow: 'hidden',
              whiteSpace: 'nowrap',
              textOverflow: 'ellipsis',
              maxWidth: '100%',
            }}
          >
            {truncateDescription(description)}
          </Typography.Text>
        )}

        {/* Expanded view: full details */}
        {isExpanded && (
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm, marginTop: theme.spacing.sm }}>
            {/* Description */}
            {hasDescription && (
              <div>
                <Typography.Text color="secondary" size="sm" bold>
                  <FormattedMessage
                    defaultMessage="Description"
                    description="Label for issue description in expanded issue card"
                  />
                </Typography.Text>
                <Typography.Text
                  color="secondary"
                  size="sm"
                  css={{ display: 'block', marginTop: theme.spacing.xs, whiteSpace: 'pre-wrap' }}
                >
                  {description}
                </Typography.Text>
              </div>
            )}

            {/* Rationale */}
            {hasRationale && (
              <div>
                <Typography.Text color="secondary" size="sm" bold>
                  <FormattedMessage
                    defaultMessage="Rationale"
                    description="Label for rationale explaining why issue was linked to trace"
                  />
                </Typography.Text>
                <div
                  css={{
                    marginTop: theme.spacing.xs,
                    color: theme.colors.textSecondary,
                    fontSize: theme.typography.fontSizeSm,
                  }}
                >
                  <GenAIMarkdownRenderer>{rationale ?? ''}</GenAIMarkdownRenderer>
                </div>
              </div>
            )}

            {/* Source and timestamp - matching AssessmentItemHeader style */}
            <div css={{ display: 'flex', flexDirection: 'row', alignItems: 'center' }}>
              <SourceIcon
                size={theme.typography.fontSizeSm}
                css={{
                  padding: 2,
                  backgroundColor: theme.colors.actionIconBackgroundHover,
                  borderRadius: theme.borders.borderRadiusFull,
                }}
              />
              <AssessmentSourceName source={issueAssessment.source} />
              <Typography.Text
                color="secondary"
                size="sm"
                css={{
                  marginLeft: 'auto',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  textWrap: 'nowrap',
                }}
              >
                {timeSinceStr(new Date(issueAssessment.create_time))}
              </Typography.Text>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
