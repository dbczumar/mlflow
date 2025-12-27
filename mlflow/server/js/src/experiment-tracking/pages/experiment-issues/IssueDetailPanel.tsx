import { useState, useEffect, useMemo } from 'react';
import {
  useDesignSystemTheme,
  Typography,
  Button,
  Tabs,
  Empty,
  DropdownMenu,
  Tooltip,
  PencilIcon,
  ShareIcon,
  CopyIcon,
  OverflowIcon,
  ChartLineIcon,
  CheckIcon,
  Spinner,
  NewWindowIcon,
  Tag,
  Table,
  TableRow,
  TableCell,
  TableHeader,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { Link } from '../../../common/utils/RoutingUtils';
import Routes from '../../routes';
import type { Issue, IssueState, IssueDetailTab } from './types';
import { useUpdateIssueMutation, useGetLinkedEvaluationRuns } from './hooks/useIssuesApi';
import { TracesView } from '../../components/traces/TracesView';
import { ExperimentViewTracesTableColumns } from '../../components/traces/TracesView.utils';
import Utils from '../../../common/utils/Utils';

interface IssueDetailPanelProps {
  issue: Issue | null;
  experimentId: string;
  onIssueUpdated: () => void;
}

const useStateConfig = () => {
  const { theme } = useDesignSystemTheme();
  return {
    open: { label: 'OPEN', bgColor: theme.colors.blue100, textColor: theme.colors.blue600 },
    closed: { label: 'CLOSED', bgColor: theme.colors.green100, textColor: theme.colors.green600 },
    draft: { label: 'DRAFT', bgColor: theme.colors.grey100, textColor: theme.colors.grey600 },
  };
};

const StateSelector = ({
  issue,
  experimentId,
  onIssueUpdated,
}: {
  issue: Issue;
  experimentId: string;
  onIssueUpdated: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const stateConfig = useStateConfig();
  const updateMutation = useUpdateIssueMutation();

  const handleStateChange = async (newState: IssueState) => {
    if (newState !== issue.state) {
      await updateMutation.mutateAsync({
        issueId: issue.issue_id,
        experimentId,
        updates: { state: newState },
      });
      onIssueUpdated();
    }
  };

  const { label, bgColor, textColor } = stateConfig[issue.state];

  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <button
          css={{
            display: 'inline-flex',
            alignItems: 'center',
            padding: '2px 8px',
            borderRadius: theme.borders.borderRadiusMd,
            backgroundColor: bgColor,
            color: textColor,
            fontWeight: 600,
            fontSize: theme.typography.fontSizeSm,
            border: 'none',
            cursor: 'pointer',
            '&:hover': {
              opacity: 0.8,
            },
          }}
        >
          {label}
        </button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Content>
        {(['open', 'closed', 'draft'] as IssueState[]).map((state) => (
          <DropdownMenu.Item
            key={state}
            componentId={`mlflow.issues.state-${state}`}
            onClick={() => handleStateChange(state)}
          >
            <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
              <span
                css={{
                  display: 'inline-block',
                  width: 8,
                  height: 8,
                  borderRadius: '50%',
                  backgroundColor: stateConfig[state].textColor,
                }}
              />
              {stateConfig[state].label}
              {issue.state === state && <CheckIcon css={{ marginLeft: 'auto', color: theme.colors.textSecondary }} />}
            </div>
          </DropdownMenu.Item>
        ))}
      </DropdownMenu.Content>
    </DropdownMenu.Root>
  );
};

const IssueIdPill = ({ issueId }: { issueId: string }) => {
  const { theme } = useDesignSystemTheme();
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(issueId);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <Tooltip
      componentId="mlflow.issues.id-pill-tooltip"
      content={copied ? 'Copied!' : `Click to copy: ${issueId}`}
    >
      <button
        onClick={handleCopy}
        css={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: theme.spacing.xs,
          padding: `2px ${theme.spacing.sm}px`,
          borderRadius: theme.borders.borderRadiusMd,
          backgroundColor: theme.colors.tagDefault,
          color: theme.colors.textSecondary,
          fontSize: theme.typography.fontSizeSm,
          fontFamily: 'monospace',
          border: 'none',
          cursor: 'pointer',
          '&:hover': {
            backgroundColor: theme.colors.tagHover,
          },
        }}
      >
        {copied ? <CheckIcon css={{ width: 12, height: 12 }} /> : <CopyIcon css={{ width: 12, height: 12 }} />}
        {issueId.slice(0, 8)}
      </button>
    </Tooltip>
  );
};

const DetailHeader = ({
  issue,
  experimentId,
  onIssueUpdated,
}: {
  issue: Issue;
  experimentId: string;
  onIssueUpdated: () => void;
}) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: theme.spacing.md,
      }}
    >
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <IssueIdPill issueId={issue.issue_id} />
        <StateSelector issue={issue} experimentId={experimentId} onIssueUpdated={onIssueUpdated} />
      </div>
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
        <Button componentId="mlflow.issues.share-button" icon={<ShareIcon />} type="tertiary" aria-label="Share" />
        <Button
          componentId="mlflow.issues.more-button"
          icon={<OverflowIcon />}
          type="tertiary"
          aria-label="More options"
        />
      </div>
    </div>
  );
};

const DetailTitle = ({
  issue,
  experimentId,
  onIssueUpdated,
}: {
  issue: Issue;
  experimentId: string;
  onIssueUpdated: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const [isEditing, setIsEditing] = useState(false);
  const [editValue, setEditValue] = useState(issue.name);
  const updateMutation = useUpdateIssueMutation();

  // Sync editValue when issue changes
  useEffect(() => {
    setEditValue(issue.name);
  }, [issue.issue_id, issue.name]);

  const handleSave = async () => {
    if (editValue.trim() && editValue !== issue.name) {
      await updateMutation.mutateAsync({
        issueId: issue.issue_id,
        experimentId,
        updates: { name: editValue.trim() },
      });
      onIssueUpdated();
    }
    setIsEditing(false);
  };

  if (isEditing) {
    return (
      <input
        type="text"
        value={editValue}
        onChange={(e) => setEditValue(e.target.value)}
        onBlur={handleSave}
        onKeyDown={(e) => {
          if (e.key === 'Enter') handleSave();
          if (e.key === 'Escape') {
            setEditValue(issue.name);
            setIsEditing(false);
          }
        }}
        autoFocus
        css={{
          fontSize: theme.typography.fontSizeLg,
          fontWeight: 600,
          border: `1px solid ${theme.colors.actionPrimaryBackgroundDefault}`,
          borderRadius: theme.borders.borderRadiusMd,
          padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
          width: '100%',
          marginBottom: theme.spacing.md,
        }}
      />
    );
  }

  return (
    <div
      css={{
        marginBottom: theme.spacing.md,
        cursor: 'pointer',
        '&:hover': { color: theme.colors.actionPrimaryBackgroundDefault },
      }}
      onClick={() => setIsEditing(true)}
    >
      <Typography.Title level={3}>{issue.name}</Typography.Title>
    </div>
  );
};

const DetailDescription = ({
  issue,
  experimentId,
  onIssueUpdated,
}: {
  issue: Issue;
  experimentId: string;
  onIssueUpdated: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const [isEditing, setIsEditing] = useState(false);
  const [editValue, setEditValue] = useState(issue.description || '');

  // Sync editValue when issue changes
  useEffect(() => {
    setEditValue(issue.description || '');
  }, [issue.issue_id, issue.description]);
  const updateMutation = useUpdateIssueMutation();

  const handleSave = async () => {
    if (editValue !== issue.description) {
      await updateMutation.mutateAsync({
        issueId: issue.issue_id,
        experimentId,
        updates: { description: editValue },
      });
      onIssueUpdated();
    }
    setIsEditing(false);
  };

  return (
    <div css={{ marginBottom: theme.spacing.lg }}>
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: theme.spacing.xs,
        }}
      >
        <Typography.Text bold>
          <FormattedMessage defaultMessage="Description" description="Label for issue description section" />
        </Typography.Text>
        <Button
          componentId="mlflow.issues.edit-description-button"
          icon={<PencilIcon />}
          type="tertiary"
          size="small"
          onClick={() => setIsEditing(true)}
        />
      </div>
      {isEditing ? (
        <textarea
          value={editValue}
          onChange={(e) => setEditValue(e.target.value)}
          onBlur={handleSave}
          onKeyDown={(e) => {
            if (e.key === 'Escape') {
              setEditValue(issue.description || '');
              setIsEditing(false);
            }
          }}
          autoFocus
          css={{
            width: '100%',
            minHeight: 80,
            border: `1px solid ${theme.colors.actionPrimaryBackgroundDefault}`,
            borderRadius: theme.borders.borderRadiusMd,
            padding: theme.spacing.sm,
            fontSize: theme.typography.fontSizeBase,
            resize: 'vertical',
          }}
        />
      ) : (
        <Typography.Text color={issue.description ? 'primary' : 'secondary'}>
          {issue.description || (
            <FormattedMessage
              defaultMessage="No description provided"
              description="Placeholder when issue has no description"
            />
          )}
        </Typography.Text>
      )}
    </div>
  );
};

const MonitorTabContent = () => {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ padding: theme.spacing.md }}>
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: theme.spacing.md,
        }}
      >
        <Typography.Text bold>
          <FormattedMessage
            defaultMessage="Judge results (last 14 days)"
            description="Title for judge results chart section"
          />
        </Typography.Text>
        <Button componentId="mlflow.issues.view-judge-button" type="tertiary">
          <FormattedMessage defaultMessage="View Judge" description="Button to view judge details" />
        </Button>
      </div>
      <Empty
        image={<ChartLineIcon css={{ fontSize: 48, color: theme.colors.textSecondary }} />}
        description={
          <FormattedMessage
            defaultMessage="Judge results chart coming soon"
            description="Placeholder for judge results chart"
          />
        }
      />
    </div>
  );
};

const TracesTabContent = ({ issue, experimentId }: { issue: Issue; experimentId: string }) => {
  // Filter traces by issue assessment: issue.<issue_id> = 'true'
  const issueFilter = useMemo(() => {
    return `issue.\`${issue.issue_id}\` = 'true'`;
  }, [issue.issue_id]);

  return (
    <div css={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <TracesView
        experimentIds={[experimentId]}
        fixedFilter={issueFilter}
        baseComponentId="mlflow.issue_page.traces"
        disabledColumns={[ExperimentViewTracesTableColumns.runName, ExperimentViewTracesTableColumns.tags]}
      />
    </div>
  );
};

const formatTime = (timestamp?: number): string => {
  if (!timestamp) return '-';
  return new Date(timestamp).toLocaleString();
};

/**
 * Get pass rate from run metrics for display
 */
const getPassRateDisplay = (
  metrics: { key: string; value: number }[] | undefined,
  issueId: string,
): { passCount: number; failCount: number; passRate: number } | null => {
  if (!metrics || metrics.length === 0) {
    return null;
  }

  // Metrics are named: {issue_id}/pass_count, {issue_id}/fail_count, etc.
  const passCountMetric = metrics.find((m) => m.key === `${issueId}/pass_count`);
  const failCountMetric = metrics.find((m) => m.key === `${issueId}/fail_count`);

  if (passCountMetric === undefined || failCountMetric === undefined) {
    return null;
  }

  const passCount = passCountMetric.value;
  const failCount = failCountMetric.value;
  const total = passCount + failCount;
  const passRate = total > 0 ? passCount / total : 0;

  return { passCount, failCount, passRate };
};

const PassRateBadge = ({
  passCount,
  failCount,
}: {
  passCount: number;
  failCount: number;
}) => {
  const { theme } = useDesignSystemTheme();
  const total = passCount + failCount;
  const passRate = total > 0 ? passCount / total : 0;
  const passPercentage = Math.round(passRate * 100);

  return (
    <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
      {/* Bar chart */}
      <div
        css={{
          display: 'flex',
          width: 120,
          height: 8,
          borderRadius: 4,
          overflow: 'hidden',
          backgroundColor: theme.colors.grey200,
        }}
      >
        {/* Green (pass) portion */}
        <div
          css={{
            width: `${passRate * 100}%`,
            backgroundColor: theme.colors.green400,
          }}
        />
        {/* Red (fail) portion */}
        <div
          css={{
            width: `${(1 - passRate) * 100}%`,
            backgroundColor: theme.colors.red400,
          }}
        />
      </div>
      {/* Count text with percentage */}
      <span css={{ color: theme.colors.textSecondary, fontSize: theme.typography.fontSizeSm }}>
        {passCount} / {total} ({passPercentage}%)
      </span>
    </div>
  );
};

type SortColumn = 'name' | 'created' | 'passRate';
type SortDirection = 'asc' | 'desc';

const EvaluationRunsTabContent = ({ issue, experimentId }: { issue: Issue; experimentId: string }) => {
  const { theme } = useDesignSystemTheme();
  const { data: runs, isLoading, error } = useGetLinkedEvaluationRuns(issue.issue_id);

  // Default sort by created date, newest first
  const [sortColumn, setSortColumn] = useState<SortColumn>('created');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');

  const handleToggleSort = (column: SortColumn) => {
    if (sortColumn === column) {
      // Toggle direction if same column
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      // New column, default to descending
      setSortColumn(column);
      setSortDirection('desc');
    }
  };

  const getSortDirection = (column: SortColumn): 'asc' | 'desc' | undefined => {
    return sortColumn === column ? sortDirection : undefined;
  };

  // Sort runs based on current sort state
  const sortedRuns = useMemo(() => {
    if (!runs) return [];

    return [...runs].sort((a, b) => {
      let comparison = 0;

      switch (sortColumn) {
        case 'name': {
          const nameA = a.info.run_name || a.info.run_id;
          const nameB = b.info.run_name || b.info.run_id;
          comparison = nameA.localeCompare(nameB);
          break;
        }
        case 'created': {
          const timeA = a.info.start_time || 0;
          const timeB = b.info.start_time || 0;
          comparison = timeA - timeB;
          break;
        }
        case 'passRate': {
          const passRateA = getPassRateDisplay(a.metrics, issue.issue_id)?.passRate ?? -1;
          const passRateB = getPassRateDisplay(b.metrics, issue.issue_id)?.passRate ?? -1;
          comparison = passRateA - passRateB;
          break;
        }
      }

      return sortDirection === 'asc' ? comparison : -comparison;
    });
  }, [runs, sortColumn, sortDirection, issue.issue_id]);

  if (isLoading) {
    return (
      <div css={{ padding: theme.spacing.md, display: 'flex', justifyContent: 'center' }}>
        <Spinner />
      </div>
    );
  }

  if (error) {
    return (
      <div css={{ padding: theme.spacing.md }}>
        <Empty
          description={
            <FormattedMessage
              defaultMessage="Failed to load evaluation runs"
              description="Error message when loading evaluation runs fails"
            />
          }
        />
      </div>
    );
  }

  if (!runs || runs.length === 0) {
    return (
      <div css={{ padding: theme.spacing.md }}>
        <Empty
          description={
            <FormattedMessage
              defaultMessage="No evaluation runs have been linked to this issue yet"
              description="Empty state when no evaluation runs are linked"
            />
          }
        />
      </div>
    );
  }

  return (
    <div css={{ padding: theme.spacing.md }}>
      <Table>
        <TableRow isHeader>
          <TableHeader
            componentId="mlflow.issues.linked-runs.header.name"
            sortable
            sortDirection={getSortDirection('name')}
            onToggleSort={() => handleToggleSort('name')}
            css={{ cursor: 'pointer' }}
          >
            <FormattedMessage defaultMessage="Run Name" description="Column header for run name" />
          </TableHeader>
          <TableHeader
            componentId="mlflow.issues.linked-runs.header.created"
            sortable
            sortDirection={getSortDirection('created')}
            onToggleSort={() => handleToggleSort('created')}
            css={{ cursor: 'pointer' }}
          >
            <FormattedMessage defaultMessage="Created" description="Column header for creation time" />
          </TableHeader>
          <TableHeader componentId="mlflow.issues.linked-runs.header.dataset">
            <FormattedMessage defaultMessage="Dataset" description="Column header for dataset" />
          </TableHeader>
          <TableHeader
            componentId="mlflow.issues.linked-runs.header.pass-rate"
            sortable
            sortDirection={getSortDirection('passRate')}
            onToggleSort={() => handleToggleSort('passRate')}
            css={{ cursor: 'pointer' }}
          >
            <FormattedMessage defaultMessage="Pass Rate" description="Column header for pass rate" />
          </TableHeader>
        </TableRow>
        {sortedRuns.map((run) => {
          const passRateData = getPassRateDisplay(run.metrics, issue.issue_id);
          // Link to Evaluation Runs tab with this run selected
          const evalRunsUrl = `${Routes.getExperimentPageRoute(experimentId)}/evaluation-runs?selectedRunUuid=${run.info.run_id}`;

          return (
            <TableRow key={run.info.run_id}>
              <TableCell>
                <Link
                  to={evalRunsUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.xs }}
                >
                  {run.info.run_name || run.info.run_id.slice(0, 8)}
                  <NewWindowIcon css={{ fontSize: 12 }} />
                </Link>
              </TableCell>
              <TableCell>
                {run.info.start_time ? (
                  <Tooltip
                    componentId="mlflow.issues.linked-runs.created-tooltip"
                    content={formatTime(run.info.start_time)}
                  >
                    <span>{Utils.timeSinceStr(new Date(run.info.start_time))}</span>
                  </Tooltip>
                ) : (
                  '-'
                )}
              </TableCell>
              <TableCell>
                <span css={{ color: theme.colors.textSecondary }}>-</span>
              </TableCell>
              <TableCell>
                {passRateData ? (
                  <PassRateBadge passCount={passRateData.passCount} failCount={passRateData.failCount} />
                ) : (
                  '-'
                )}
              </TableCell>
            </TableRow>
          );
        })}
      </Table>
    </div>
  );
};

const CommentsTabContent = () => {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ padding: theme.spacing.md }}>
      <Empty
        description={
          <FormattedMessage defaultMessage="Comments coming soon" description="Placeholder for comments section" />
        }
      />
    </div>
  );
};

export const IssueDetailPanel = ({ issue, experimentId, onIssueUpdated }: IssueDetailPanelProps) => {
  const { theme } = useDesignSystemTheme();
  const [activeTab, setActiveTab] = useState<IssueDetailTab>('traces');

  if (!issue) {
    return (
      <div
        css={{
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: theme.spacing.lg,
        }}
      >
        <Empty
          description={
            <FormattedMessage
              defaultMessage="Select an issue to view details"
              description="Placeholder when no issue is selected"
            />
          }
        />
      </div>
    );
  }

  return (
    <div
      css={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        padding: theme.spacing.md,
        minWidth: 0,
        overflow: 'auto',
      }}
    >
      <DetailHeader issue={issue} experimentId={experimentId} onIssueUpdated={onIssueUpdated} />
      <DetailTitle issue={issue} experimentId={experimentId} onIssueUpdated={onIssueUpdated} />
      <DetailDescription issue={issue} experimentId={experimentId} onIssueUpdated={onIssueUpdated} />

      <Tabs.Root
        componentId="mlflow.issues.detail-tabs"
        value={activeTab}
        onValueChange={(value) => setActiveTab(value as IssueDetailTab)}
      >
        <Tabs.List>
          <Tabs.Trigger value="traces">
            <FormattedMessage defaultMessage="Traces" description="Tab label for traces section" />
          </Tabs.Trigger>
          <Tabs.Trigger value="monitor">
            <FormattedMessage defaultMessage="Monitor" description="Tab label for monitor section" />
          </Tabs.Trigger>
          <Tabs.Trigger value="evaluation-runs">
            <FormattedMessage defaultMessage="Evaluation Runs" description="Tab label for evaluation runs section" />
          </Tabs.Trigger>
          <Tabs.Trigger value="comments">
            <FormattedMessage defaultMessage="Comments" description="Tab label for comments section" />
          </Tabs.Trigger>
        </Tabs.List>
        <Tabs.Content value="traces">
          <TracesTabContent issue={issue} experimentId={experimentId} />
        </Tabs.Content>
        <Tabs.Content value="monitor">
          <MonitorTabContent />
        </Tabs.Content>
        <Tabs.Content value="evaluation-runs">
          <EvaluationRunsTabContent issue={issue} experimentId={experimentId} />
        </Tabs.Content>
        <Tabs.Content value="comments">
          <CommentsTabContent />
        </Tabs.Content>
      </Tabs.Root>
    </div>
  );
};
