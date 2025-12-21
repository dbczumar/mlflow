import { useState, useEffect } from 'react';
import {
  useDesignSystemTheme,
  Typography,
  Button,
  Tabs,
  Empty,
  DropdownMenu,
  PencilIcon,
  ShareIcon,
  CopyIcon,
  OverflowIcon,
  ChartLineIcon,
  CheckIcon,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import type { Issue, IssueState, IssueDetailTab } from './types';
import { useUpdateIssueMutation } from './hooks/useIssuesApi';

interface IssueDetailPanelProps {
  issue: Issue | null;
  experimentId: string;
  onIssueUpdated: () => void;
  issueNumber?: number;
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
              {issue.state === state && (
                <CheckIcon css={{ marginLeft: 'auto', color: theme.colors.textSecondary }} />
              )}
            </div>
          </DropdownMenu.Item>
        ))}
      </DropdownMenu.Content>
    </DropdownMenu.Root>
  );
};

const DetailHeader = ({
  issue,
  issueNumber,
  experimentId,
  onIssueUpdated,
}: {
  issue: Issue;
  issueNumber: number;
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
        <Typography.Text color="secondary">Issue #{issueNumber}</Typography.Text>
        <StateSelector issue={issue} experimentId={experimentId} onIssueUpdated={onIssueUpdated} />
      </div>
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
        <Button componentId="mlflow.issues.share-button" icon={<ShareIcon />} type="tertiary" aria-label="Share" />
        <Button componentId="mlflow.issues.copy-button" icon={<CopyIcon />} type="tertiary" aria-label="Copy" />
        <Button componentId="mlflow.issues.more-button" icon={<OverflowIcon />} type="tertiary" aria-label="More options" />
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

const TracesTabContent = () => {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ padding: theme.spacing.md }}>
      <Empty
        description={
          <FormattedMessage
            defaultMessage="Linked traces will appear here"
            description="Placeholder for linked traces section"
          />
        }
      />
    </div>
  );
};

const EvaluationRunsTabContent = () => {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ padding: theme.spacing.md }}>
      <Empty
        description={
          <FormattedMessage
            defaultMessage="Linked evaluation runs will appear here"
            description="Placeholder for linked evaluation runs section"
          />
        }
      />
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

export const IssueDetailPanel = ({ issue, experimentId, onIssueUpdated, issueNumber }: IssueDetailPanelProps) => {
  const { theme } = useDesignSystemTheme();
  const [activeTab, setActiveTab] = useState<IssueDetailTab>('monitor');

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
      <DetailHeader
        issue={issue}
        issueNumber={issueNumber || 1}
        experimentId={experimentId}
        onIssueUpdated={onIssueUpdated}
      />
      <DetailTitle issue={issue} experimentId={experimentId} onIssueUpdated={onIssueUpdated} />
      <DetailDescription issue={issue} experimentId={experimentId} onIssueUpdated={onIssueUpdated} />

      <Tabs.Root
        componentId="mlflow.issues.detail-tabs"
        value={activeTab}
        onValueChange={(value) => setActiveTab(value as IssueDetailTab)}
      >
        <Tabs.List>
          <Tabs.Trigger value="monitor">
            <FormattedMessage defaultMessage="Monitor" description="Tab label for monitor section" />
          </Tabs.Trigger>
          <Tabs.Trigger value="traces">
            <FormattedMessage defaultMessage="Traces" description="Tab label for traces section" />
          </Tabs.Trigger>
          <Tabs.Trigger value="evaluation-runs">
            <FormattedMessage defaultMessage="Evaluation Runs" description="Tab label for evaluation runs section" />
          </Tabs.Trigger>
          <Tabs.Trigger value="comments">
            <FormattedMessage defaultMessage="Comments" description="Tab label for comments section" />
          </Tabs.Trigger>
        </Tabs.List>
        <Tabs.Content value="monitor">
          <MonitorTabContent />
        </Tabs.Content>
        <Tabs.Content value="traces">
          <TracesTabContent />
        </Tabs.Content>
        <Tabs.Content value="evaluation-runs">
          <EvaluationRunsTabContent />
        </Tabs.Content>
        <Tabs.Content value="comments">
          <CommentsTabContent />
        </Tabs.Content>
      </Tabs.Root>
    </div>
  );
};
