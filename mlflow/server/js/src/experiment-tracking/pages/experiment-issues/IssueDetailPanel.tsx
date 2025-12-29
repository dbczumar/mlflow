import { useState, useEffect, useMemo, useCallback } from 'react';
import { GenAIMarkdownRenderer } from '../../../shared/web-shared/genai-markdown-renderer';
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
  CheckIcon,
  Spinner,
  TrashIcon,
  NewWindowIcon,
  Table,
  TableRow,
  TableCell,
  TableHeader,
  PlayIcon,
  FormUI,
  Input,
  Checkbox,
  Tag,
  SparkleDoubleIcon,
  Popover,
  SearchIcon,
  ChevronDownIcon,
  CalendarEventIcon,
  CodeIcon,
  ParagraphSkeleton,
  DangerIcon,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import type { Issue, IssueState, IssueDetailTab, IssueComment, IssueJudge } from './types';
import { useUpdateIssueMutation, useGetLinkedEvaluationRuns } from './hooks/useIssuesApi';
import {
  useSearchIssueComments,
  useCreateIssueCommentMutation,
  useUpdateIssueCommentMutation,
  useDeleteIssueCommentMutation,
} from './hooks/useIssueCommentsApi';
import { useGetJudgeForIssue, useCreateJudgeMutation } from './hooks/useJudgeApi';
import Routes from '../../routes';
import { Link } from '../../../common/utils/RoutingUtils';
import Utils from '../../../common/utils/Utils';
import {
  GenAiTracesMarkdownConverterProvider,
  GenAITracesTableBodyContainer,
  GenAITracesTableToolbar,
  GenAITracesTableProvider,
  useSearchMlflowTraces,
  useSelectedColumns,
  useMlflowTracesTableMetadata,
  useFilters,
  useTableSort,
  TracesTableColumnType,
  TracesTableColumnGroup,
  REQUEST_TIME_COLUMN_ID,
  EXECUTION_DURATION_COLUMN_ID,
  TRACE_ID_COLUMN_ID,
  RESPONSE_COLUMN_ID,
  createTraceLocationForExperiment,
  type TracesTableColumn,
  type TraceActions,
} from '@databricks/web-shared/genai-traces-table';
import { useMarkdownConverter } from '@mlflow/mlflow/src/common/utils/MarkdownUtils';
import { getTrace as getTraceV3 } from '@mlflow/mlflow/src/experiment-tracking/utils/TraceUtils';
import { useEditExperimentTraceTags } from '../../components/traces/hooks/useEditExperimentTraceTags';
import { useQueryClient } from '@databricks/web-shared/query-client';
import { invalidateMlflowSearchTracesCache, getTracesTagKeys } from '@databricks/web-shared/genai-traces-table';
import { useGetDeleteTracesAction } from '../../components/experiment-page/components/traces-v3/hooks/useGetDeleteTracesAction';


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

const JudgeEmptyState = ({
  issueId,
  experimentId,
  onJudgeCreated,
}: {
  issueId: string;
  experimentId: string;
  onJudgeCreated: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const createJudgeMutation = useCreateJudgeMutation();

  const handleCreateJudge = async () => {
    try {
      await createJudgeMutation.mutateAsync({ issueId, experimentId });
      onJudgeCreated();
    } catch (error) {
      console.error('Failed to create judge:', error);
    }
  };

  return (
    <div css={{ padding: theme.spacing.lg }}>
      <Empty
        image={<PlayIcon css={{ fontSize: 48, color: theme.colors.textSecondary }} />}
        title={
          <FormattedMessage
            defaultMessage="No Judge Created"
            description="Empty state title when no judge exists for the issue"
          />
        }
        description={
          <FormattedMessage
            defaultMessage="Create an LLM-as-a-Judge to automatically detect this issue on new traces. The judge will evaluate incoming traces and flag those that match this issue."
            description="Empty state description for judge tab"
          />
        }
      />
      <div css={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginTop: theme.spacing.md }}>
        <Button
          componentId="mlflow.issues.create-judge-button"
          type="primary"
          onClick={handleCreateJudge}
          loading={createJudgeMutation.isLoading}
        >
          {createJudgeMutation.isLoading ? (
            <FormattedMessage defaultMessage="Creating Judge..." description="Button text while creating judge" />
          ) : (
            <FormattedMessage defaultMessage="Create Judge" description="Button to create a new judge for the issue" />
          )}
        </Button>
        {createJudgeMutation.isLoading && (
          <Typography.Text color="secondary" css={{ marginTop: theme.spacing.sm }}>
            <FormattedMessage
              defaultMessage="Crafting LLM Judge for the issue... this may take a minute"
              description="Loading message while creating judge"
            />
          </Typography.Text>
        )}
      </div>
      {createJudgeMutation.isError && (
        <div css={{ textAlign: 'center', marginTop: theme.spacing.md, color: theme.colors.textValidationDanger }}>
          <FormattedMessage
            defaultMessage="Failed to create judge. Please try again."
            description="Error message when judge creation fails"
          />
        </div>
      )}
    </div>
  );
};

type RunJudgeTab = 'quick-scan' | 'batch-evaluation' | 'schedule';

const RunJudgeDropdown = ({ judge, experimentId }: { judge: IssueJudge; experimentId: string }) => {
  const { theme } = useDesignSystemTheme();
  const [isOpen, setIsOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<RunJudgeTab>('quick-scan');
  const [traceSearch, setTraceSearch] = useState('');
  const [copied, setCopied] = useState(false);

  const codeSnippet = `import mlflow

# Load the judge scorer
scorer = mlflow.genai.scorers.get_scorer(
    experiment_id="${experimentId}",
    scorer_name="${judge.scorer_name}",
)

# Run evaluation on your traces
results = mlflow.genai.evaluate(
    data=traces,
    scorers=[scorer],
)

# View the evaluation results
results.tables["eval_results"]`;

  const handleCopyCode = async () => {
    await navigator.clipboard.writeText(codeSnippet);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <Popover.Root componentId="mlflow.issues.judge.run-popover" open={isOpen} onOpenChange={setIsOpen}>
      <Popover.Trigger asChild>
        <Button
          componentId="mlflow.issues.judge.run-judge-button"
          type="primary"
          icon={<PlayIcon />}
          endIcon={<ChevronDownIcon />}
        >
          <FormattedMessage defaultMessage="Run Judge" description="Button to run the judge" />
        </Button>
      </Popover.Trigger>
      <Popover.Content align="end" css={{ width: 480, padding: 0 }}>
        <div css={{ padding: theme.spacing.md }}>
          <Typography.Title level={4} css={{ marginBottom: theme.spacing.sm }}>
            <FormattedMessage defaultMessage="Run Judge" description="Run judge dropdown title" />
          </Typography.Title>
          <Tabs.Root
            componentId="mlflow.issues.judge.run-tabs"
            value={activeTab}
            onValueChange={(value) => setActiveTab(value as RunJudgeTab)}
          >
            <Tabs.List>
              <Tabs.Trigger value="quick-scan">
                <span css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                  <SearchIcon />
                  <FormattedMessage defaultMessage="Quick Scan" description="Tab for quick scan" />
                </span>
              </Tabs.Trigger>
              <Tabs.Trigger value="batch-evaluation">
                <span css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                  <CodeIcon />
                  <FormattedMessage defaultMessage="Batch Evaluation" description="Tab for batch evaluation code" />
                </span>
              </Tabs.Trigger>
              <Tabs.Trigger value="schedule">
                <span css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                  <CalendarEventIcon />
                  <FormattedMessage defaultMessage="Schedule" description="Tab for scheduling judge" />
                </span>
              </Tabs.Trigger>
            </Tabs.List>

            <Tabs.Content value="quick-scan" css={{ paddingTop: theme.spacing.md }}>
              <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.sm }}>
                <FormattedMessage
                  defaultMessage="Search and select traces to quickly run the judge on."
                  description="Quick scan description"
                />
              </Typography.Text>
              <Input
                componentId="mlflow.issues.judge.trace-search"
                value={traceSearch}
                onChange={(e) => setTraceSearch(e.target.value)}
                placeholder="Search traces by request ID or content..."
                prefix={<SearchIcon />}
                css={{ marginBottom: theme.spacing.sm }}
              />
              <div
                css={{
                  border: `1px solid ${theme.colors.border}`,
                  borderRadius: theme.borders.borderRadiusMd,
                  padding: theme.spacing.md,
                  minHeight: 150,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <Typography.Text color="secondary">
                  <FormattedMessage
                    defaultMessage="Search for traces to evaluate"
                    description="Empty state for trace search"
                  />
                </Typography.Text>
              </div>
              <div css={{ display: 'flex', justifyContent: 'flex-end', marginTop: theme.spacing.md }}>
                <Button componentId="mlflow.issues.judge.run-quick-scan" type="primary" disabled>
                  <FormattedMessage defaultMessage="Run on Selected" description="Button to run judge on selected traces" />
                </Button>
              </div>
            </Tabs.Content>

            <Tabs.Content value="batch-evaluation" css={{ paddingTop: theme.spacing.md }}>
              <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.sm }}>
                <FormattedMessage
                  defaultMessage="Use this code snippet to run the judge on a batch of traces programmatically."
                  description="Batch evaluation description"
                />
              </Typography.Text>
              <div
                css={{
                  position: 'relative',
                  border: `1px solid ${theme.colors.border}`,
                  borderRadius: theme.borders.borderRadiusMd,
                  backgroundColor: theme.colors.backgroundSecondary,
                }}
              >
                <Button
                  componentId="mlflow.issues.judge.copy-code"
                  type="tertiary"
                  size="small"
                  icon={copied ? <CheckIcon /> : <CopyIcon />}
                  onClick={handleCopyCode}
                  css={{ position: 'absolute', top: theme.spacing.xs, right: theme.spacing.xs }}
                >
                  {copied ? (
                    <FormattedMessage defaultMessage="Copied!" description="Copied confirmation" />
                  ) : (
                    <FormattedMessage defaultMessage="Copy" description="Copy button" />
                  )}
                </Button>
                <pre
                  css={{
                    padding: theme.spacing.md,
                    margin: 0,
                    fontSize: theme.typography.fontSizeSm,
                    fontFamily: 'monospace',
                    overflow: 'auto',
                    maxHeight: 250,
                  }}
                >
                  {codeSnippet}
                </pre>
              </div>
            </Tabs.Content>

            <Tabs.Content value="schedule" css={{ paddingTop: theme.spacing.md }}>
              <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.sm }}>
                <FormattedMessage
                  defaultMessage="Configure the judge to automatically run on new traces."
                  description="Schedule description"
                />
              </Typography.Text>
              <div css={{ marginBottom: theme.spacing.md }}>
                <Checkbox componentId="mlflow.issues.judge.auto-evaluate" isChecked={false} onChange={() => {}}>
                  <FormattedMessage
                    defaultMessage="Automatically evaluate future traces using this judge"
                    description="Auto evaluate checkbox"
                  />
                </Checkbox>
              </div>
              <div css={{ marginBottom: theme.spacing.md }}>
                <FormUI.Label>
                  <FormattedMessage defaultMessage="Sample Rate" description="Sample rate label" />
                </FormUI.Label>
                <Input
                  componentId="mlflow.issues.judge.sample-rate"
                  type="number"
                  value="100"
                  disabled
                  suffix="%"
                  css={{ width: 120, marginTop: theme.spacing.xs }}
                />
                <FormUI.Hint>
                  <FormattedMessage
                    defaultMessage="Percentage of traces to evaluate (1-100%)"
                    description="Sample rate hint"
                  />
                </FormUI.Hint>
              </div>
              <div css={{ display: 'flex', justifyContent: 'flex-end', marginTop: theme.spacing.md }}>
                <Button componentId="mlflow.issues.judge.save-schedule" type="primary" disabled>
                  <FormattedMessage defaultMessage="Save Schedule" description="Save schedule button" />
                </Button>
              </div>
            </Tabs.Content>
          </Tabs.Root>
        </div>
      </Popover.Content>
    </Popover.Root>
  );
};

const JudgeDetails = ({ judge, experimentId }: { judge: IssueJudge; experimentId: string }) => {
  const { theme } = useDesignSystemTheme();

  // Editable state
  const [name, setName] = useState(judge.scorer_name);
  const [useDefaultModel, setUseDefaultModel] = useState(true);
  const [model, setModel] = useState(judge.model || 'openai:/gpt-4o-mini');
  const [prompt, setPrompt] = useState(judge.prompt);
  const [isEditingPrompt, setIsEditingPrompt] = useState(false);

  // Track if there are unsaved changes
  const hasChanges =
    name !== judge.scorer_name ||
    prompt !== judge.prompt ||
    model !== (judge.model || 'openai:/gpt-4o-mini');

  const cardStyles = {
    border: `1px solid ${theme.colors.border}`,
    borderRadius: theme.borders.borderRadiusMd,
    padding: theme.spacing.md,
    marginBottom: theme.spacing.md,
  };

  const handleSave = () => {
    // TODO: Implement save API call
    console.log('Saving judge:', { name, model, prompt });
    setIsEditingPrompt(false);
  };

  const handleCancel = () => {
    setName(judge.scorer_name);
    setModel(judge.model || 'openai:/gpt-4o-mini');
    setPrompt(judge.prompt);
    setIsEditingPrompt(false);
  };

  // Calculate line numbers for prompt
  const promptLines = (isEditingPrompt ? prompt : judge.prompt).split('\n');

  return (
    <div css={{ padding: theme.spacing.md }}>
      {/* Header with LLM-as-a-judge tag on the left and Run Judge on the right */}
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: theme.spacing.sm,
        }}
      >
        <Tag componentId="mlflow.issues.judge.type-tag" color="purple" icon={<SparkleDoubleIcon />}>
          <FormattedMessage defaultMessage="LLM-as-a-judge" description="Label indicating this is an LLM judge" />
        </Tag>
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
          {hasChanges && (
            <>
              <Button componentId="mlflow.issues.judge.cancel-button" type="tertiary" size="small" onClick={handleCancel}>
                <FormattedMessage defaultMessage="Cancel" description="Cancel button" />
              </Button>
              <Button componentId="mlflow.issues.judge.save-button" type="primary" size="small" onClick={handleSave}>
                <FormattedMessage defaultMessage="Save Changes" description="Save button" />
              </Button>
            </>
          )}
          <RunJudgeDropdown judge={judge} experimentId={experimentId} />
        </div>
      </div>
      {/* Description */}
      <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.md }}>
        <FormattedMessage
          defaultMessage="This judge automatically evaluates new traces and flags those that match this issue."
          description="Description of what the judge does"
        />
      </Typography.Text>

      {/* Name Section */}
      <div css={{ marginBottom: theme.spacing.md }}>
        <FormUI.Label>
          <FormattedMessage defaultMessage="Name" description="Section header for judge name" />
        </FormUI.Label>
        <Input
          componentId="mlflow.issues.judge.name-input"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Enter judge name"
          css={{ marginTop: theme.spacing.xs }}
        />
      </div>

      {/* Model Card */}
      <div css={cardStyles}>
        <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.sm }}>
          <FormattedMessage defaultMessage="Model" description="Section header for judge model" />
        </Typography.Text>
        <Checkbox
          componentId="mlflow.issues.judge.use-default-model-checkbox"
          isChecked={useDefaultModel}
          onChange={(checked) => setUseDefaultModel(checked)}
        >
          <FormattedMessage
            defaultMessage="Use default evaluation model"
            description="Checkbox label for using default model"
          />
        </Checkbox>
        <div css={{ marginLeft: theme.spacing.lg, marginTop: theme.spacing.xs }}>
          {useDefaultModel ? (
            <Typography.Text color="secondary">
              <FormattedMessage
                defaultMessage="Current default model: {model}"
                description="Display current default model"
                values={{ model }}
              />
            </Typography.Text>
          ) : (
            <Input
              componentId="mlflow.issues.judge.model-input"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              placeholder="Enter model name (e.g., openai:/gpt-4o)"
              css={{ marginTop: theme.spacing.xs }}
            />
          )}
        </div>
      </div>

      {/* Prompt Card */}
      <div css={cardStyles}>
        <div css={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: theme.spacing.sm }}>
          <Typography.Text bold>
            <FormattedMessage defaultMessage="Prompt" description="Section header for judge prompt" />
          </Typography.Text>
          {!isEditingPrompt && (
            <Button
              componentId="mlflow.issues.judge.edit-prompt-button"
              icon={<PencilIcon />}
              type="tertiary"
              size="small"
              onClick={() => setIsEditingPrompt(true)}
            >
              <FormattedMessage defaultMessage="Edit" description="Edit button" />
            </Button>
          )}
        </div>
        <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.sm }}>
          <FormattedMessage
            defaultMessage="LLM-as-a-Judge evaluation prompt. You can customize the prompt."
            description="Hint text for prompt section"
          />
        </Typography.Text>
        <div
          css={{
            border: `1px solid ${theme.colors.border}`,
            borderRadius: theme.borders.borderRadiusMd,
            maxHeight: 400,
            overflow: 'auto',
            display: 'flex',
          }}
        >
          {/* Line numbers - sticky to left */}
          <div
            css={{
              padding: theme.spacing.sm,
              backgroundColor: theme.colors.backgroundSecondary,
              borderRight: `1px solid ${theme.colors.border}`,
              color: theme.colors.textSecondary,
              fontFamily: 'monospace',
              fontSize: theme.typography.fontSizeSm,
              lineHeight: '20px',
              userSelect: 'none',
              textAlign: 'right',
              minWidth: 40,
              flexShrink: 0,
              position: 'sticky',
              left: 0,
            }}
          >
            {promptLines.map((_, index) => (
              <div key={index}>{index + 1}</div>
            ))}
          </div>
          {/* Prompt content */}
          {isEditingPrompt ? (
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              css={{
                flex: 1,
                padding: theme.spacing.sm,
                fontFamily: 'monospace',
                fontSize: theme.typography.fontSizeSm,
                lineHeight: '20px',
                border: 'none',
                outline: 'none',
                resize: 'none',
                minHeight: 200,
              }}
            />
          ) : (
            <div
              css={{
                flex: 1,
                padding: theme.spacing.sm,
                fontFamily: 'monospace',
                fontSize: theme.typography.fontSizeSm,
                lineHeight: '20px',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
              }}
            >
              {judge.prompt}
            </div>
          )}
        </div>
        {isEditingPrompt && (
          <div css={{ display: 'flex', justifyContent: 'flex-end', gap: theme.spacing.sm, marginTop: theme.spacing.sm }}>
            <Button
              componentId="mlflow.issues.judge.cancel-prompt-button"
              type="tertiary"
              size="small"
              onClick={() => {
                setPrompt(judge.prompt);
                setIsEditingPrompt(false);
              }}
            >
              <FormattedMessage defaultMessage="Cancel" description="Cancel button" />
            </Button>
            <Button
              componentId="mlflow.issues.judge.done-prompt-button"
              type="primary"
              size="small"
              onClick={() => setIsEditingPrompt(false)}
            >
              <FormattedMessage defaultMessage="Done" description="Done editing button" />
            </Button>
          </div>
        )}
      </div>
    </div>
  );
};

const JudgeTabContent = ({ issue, experimentId }: { issue: Issue; experimentId: string }) => {
  const { theme } = useDesignSystemTheme();
  const { data: judge, isLoading, error, refetch } = useGetJudgeForIssue(issue.issue_id, experimentId);

  if (isLoading) {
    return (
      <div css={{ padding: theme.spacing.lg, display: 'flex', justifyContent: 'center' }}>
        <Spinner />
      </div>
    );
  }

  if (error) {
    return (
      <div css={{ padding: theme.spacing.lg }}>
        <Empty
          description={
            <FormattedMessage
              defaultMessage="Failed to load judge information"
              description="Error message when loading judge fails"
            />
          }
        />
      </div>
    );
  }

  if (!judge) {
    return <JudgeEmptyState issueId={issue.issue_id} experimentId={experimentId} onJudgeCreated={() => refetch()} />;
  }

  return <JudgeDetails judge={judge} experimentId={experimentId} />;
};

/**
 * Helper function to check if a trace has an assessment for a specific issue.
 * The issue_id is stored as assessment_name in the backend.
 */
const traceHasIssueAssessment = (
  trace: import('@databricks/web-shared/model-trace-explorer').ModelTraceInfoV3,
  issueId: string,
): boolean => {
  if (!trace.assessments) {
    return false;
  }
  return trace.assessments.some(
    (assessment) =>
      'issue' in assessment &&
      assessment.issue !== undefined &&
      assessment.assessment_name === issueId &&
      assessment.valid !== false,
  );
};

// Column ID for request/inputs (not exported from genai-traces-table)
const INPUTS_COLUMN_ID = 'request';
// Issue column ID prefix - matches the one used in genai-traces-table for pass/fail rendering
const ISSUE_COLUMN_ID_PREFIX = 'issue_';

const TracesTabContent = ({ issue, experimentId }: { issue: Issue; experimentId: string }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const makeHtmlFromMarkdown = useMarkdownConverter();
  const queryClient = useQueryClient();

  const traceSearchLocations = useMemo(
    () => [createTraceLocationForExperiment(experimentId)],
    [experimentId],
  );

  // Get metadata
  const {
    assessmentInfos,
    allColumns: baseColumns,
    isLoading: isMetadataLoading,
    error: metadataError,
    tableFilterOptions,
  } = useMlflowTracesTableMetadata({
    locations: traceSearchLocations,
    disabled: false,
  });

  // Create a custom column for this specific issue (shows Pass/Fail)
  // The column ID must use issue.name because the cell renderer extracts the name
  // from the column ID and uses it to look up the assessment via getIssueName()
  // Using INFO group to avoid showing the "Issues (x/y)" group header
  const issueColumnId = ISSUE_COLUMN_ID_PREFIX + issue.name;
  const issueColumn: TracesTableColumn = useMemo(
    () => ({
      id: issueColumnId,
      label: 'Judge',
      type: TracesTableColumnType.TRACE_INFO,
      group: TracesTableColumnGroup.INFO,
      issueName: issue.name,
    }),
    [issueColumnId, issue.name],
  );

  // Add the issue column to allColumns, positioned before execution_duration
  // Remove the generic "issues" column since we only want to show this specific issue
  const allColumns = useMemo(() => {
    const filteredColumns = baseColumns.filter((col) => col.id !== 'issues');

    // Find the execution duration column and insert our issue column before it
    const executionDurationIndex = filteredColumns.findIndex((col) => col.id === EXECUTION_DURATION_COLUMN_ID);
    if (executionDurationIndex !== -1) {
      const result = [...filteredColumns];
      result.splice(executionDurationIndex, 0, issueColumn);
      return result;
    }

    // Fallback: find response column and insert after it
    const responseIndex = filteredColumns.findIndex((col) => col.id === RESPONSE_COLUMN_ID);
    if (responseIndex !== -1) {
      const result = [...filteredColumns];
      result.splice(responseIndex + 1, 0, issueColumn);
      return result;
    }

    // Last fallback: append to the end
    return [...filteredColumns, issueColumn];
  }, [baseColumns, issueColumn]);

  // Setup table states
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [filters, setFilters] = useFilters();

  // Default columns in order: trace_id, request, response, issue (pass/fail), execution_duration, request_time
  const defaultSelectedColumns = useCallback(
    (cols: TracesTableColumn[]) => {
      const columnOrder = [
        TRACE_ID_COLUMN_ID,
        INPUTS_COLUMN_ID,
        RESPONSE_COLUMN_ID,
        issueColumnId,
        EXECUTION_DURATION_COLUMN_ID,
        REQUEST_TIME_COLUMN_ID,
      ];

      const filteredCols = cols.filter((col) => columnOrder.includes(col.id));
      return filteredCols.sort((a, b) => {
        const aIndex = columnOrder.indexOf(a.id);
        const bIndex = columnOrder.indexOf(b.id);
        return aIndex - bIndex;
      });
    },
    [issueColumnId],
  );

  const { selectedColumns, toggleColumns, setSelectedColumns } = useSelectedColumns(
    experimentId,
    allColumns,
    defaultSelectedColumns,
  );

  const [tableSort, setTableSort] = useTableSort(selectedColumns, {
    key: REQUEST_TIME_COLUMN_ID,
    type: TracesTableColumnType.TRACE_INFO,
    asc: false,
  });

  // Get traces data
  const {
    data: allTraceInfos,
    isLoading: traceInfosLoading,
    error: traceInfosError,
  } = useSearchMlflowTraces({
    locations: traceSearchLocations,
    searchQuery,
    filters,
    tableSort,
    disabled: false,
  });

  // Filter traces client-side to only show traces that have an assessment for this specific issue
  // This is necessary because the backend doesn't support OR conditions for issue filters
  const traceInfos = useMemo(() => {
    if (!allTraceInfos) {
      return undefined;
    }
    return allTraceInfos.filter((trace) => traceHasIssueAssessment(trace, issue.issue_id));
  }, [allTraceInfos, issue.issue_id]);

  const { showEditTagsModalForTrace, EditTagsModal } = useEditExperimentTraceTags({
    onSuccess: () => invalidateMlflowSearchTracesCache({ queryClient }),
    existingTagKeys: getTracesTagKeys(traceInfos || []),
  });

  const deleteTracesAction = useGetDeleteTracesAction({ traceSearchLocations });

  const traceActions: TraceActions = useMemo(() => {
    return {
      deleteTracesAction,
      editTags: {
        showEditTagsModalForTrace,
        EditTagsModal,
      },
    };
  }, [deleteTracesAction, showEditTagsModalForTrace, EditTagsModal]);

  const isTableLoading = traceInfosLoading || isMetadataLoading;
  const tableError = traceInfosError || metadataError;

  const countInfo = useMemo(() => {
    return {
      currentCount: traceInfos?.length,
      logCountLoading: traceInfosLoading,
      totalCount: traceInfos?.length ?? 0,
      maxAllowedCount: 10000,
    };
  }, [traceInfos, traceInfosLoading]);

  return (
    <GenAITracesTableProvider>
      <div
        css={{
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
      >
        <GenAITracesTableToolbar
          experimentId={experimentId}
          searchQuery={searchQuery}
          setSearchQuery={setSearchQuery}
          filters={filters}
          setFilters={setFilters}
          assessmentInfos={assessmentInfos}
          traceInfos={traceInfos}
          tableFilterOptions={tableFilterOptions}
          countInfo={countInfo}
          traceActions={traceActions}
          tableSort={tableSort}
          setTableSort={setTableSort}
          allColumns={allColumns}
          selectedColumns={selectedColumns}
          toggleColumns={toggleColumns}
          setSelectedColumns={setSelectedColumns}
          isMetadataLoading={isMetadataLoading}
          metadataError={metadataError}
          usesV4APIs={true}
        />
        <div css={{ flex: 1, overflow: 'hidden', display: 'flex' }}>
          {isTableLoading ? (
            <div
              css={{
                display: 'flex',
                flexDirection: 'column',
                width: '100%',
                gap: theme.spacing.sm,
                padding: theme.spacing.md,
              }}
            >
              {[...Array(10).keys()].map((i) => (
                <ParagraphSkeleton label="Loading..." key={i} seed={`s-${i}`} />
              ))}
            </div>
          ) : tableError ? (
            <div
              css={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                width: '100%',
                height: '100%',
              }}
            >
              <Empty
                image={<DangerIcon />}
                title={intl.formatMessage({
                  defaultMessage: 'Fetching traces failed',
                  description: 'Issue detail > traces tab > error state title',
                })}
                description={tableError.message}
              />
            </div>
          ) : (
            <GenAiTracesMarkdownConverterProvider makeHtml={makeHtmlFromMarkdown}>
              <GenAITracesTableBodyContainer
                experimentId={experimentId}
                allColumns={allColumns}
                currentTraceInfoV3={traceInfos || []}
                currentRunDisplayName=""
                getTrace={getTraceV3}
                assessmentInfos={assessmentInfos}
                setFilters={setFilters}
                filters={filters}
                selectedColumns={selectedColumns}
                tableSort={tableSort}
                onTraceTagsEdit={showEditTagsModalForTrace}
                displayLoadingOverlay={false}
              />
            </GenAiTracesMarkdownConverterProvider>
          )}
        </div>
      </div>
    </GenAITracesTableProvider>
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

const formatRelativeTime = (timestamp?: number): string => {
  if (!timestamp) return '';
  const now = Date.now();
  const diff = now - timestamp;
  const seconds = Math.floor(diff / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (days > 0) return `${days} day${days > 1 ? 's' : ''} ago`;
  if (hours > 0) return `${hours} hour${hours > 1 ? 's' : ''} ago`;
  if (minutes > 0) return `${minutes} minute${minutes > 1 ? 's' : ''} ago`;
  return 'just now';
};

const CommentItem = ({
  comment,
  issueId,
  onUpdated,
}: {
  comment: IssueComment;
  issueId: string;
  onUpdated: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const [isEditing, setIsEditing] = useState(false);
  const [editContent, setEditContent] = useState(comment.content);
  const updateMutation = useUpdateIssueCommentMutation();
  const deleteMutation = useDeleteIssueCommentMutation();

  const handleSave = async () => {
    if (editContent.trim() && editContent !== comment.content) {
      await updateMutation.mutateAsync({
        commentId: comment.comment_id,
        issueId,
        content: editContent.trim(),
      });
      onUpdated();
    }
    setIsEditing(false);
  };

  const handleDelete = async () => {
    await deleteMutation.mutateAsync({
      commentId: comment.comment_id,
      issueId,
    });
    onUpdated();
  };

  return (
    <div
      css={{
        display: 'flex',
        padding: theme.spacing.md,
        borderBottom: `1px solid ${theme.colors.border}`,
        '&:last-child': { borderBottom: 'none' },
      }}
    >
      {/* Avatar column - fixed width for alignment */}
      <div
        css={{
          width: 24,
          height: 24,
          borderRadius: '50%',
          backgroundColor: theme.colors.grey200,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: theme.colors.grey500,
          fontSize: 12,
          fontWeight: 500,
          marginRight: theme.spacing.sm,
        }}
      >
        {(comment.author || 'U').charAt(0).toUpperCase()}
      </div>

      {/* Content column */}
      <div css={{ flex: 1, minWidth: 0 }}>
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            marginBottom: theme.spacing.md,
          }}
        >
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
            <Typography.Text bold>{comment.author || 'User'}</Typography.Text>
            <Typography.Text color="secondary" size="sm">
              {formatRelativeTime(comment.creation_time)}
            </Typography.Text>
            {comment.last_update_time && comment.last_update_time !== comment.creation_time && (
              <Typography.Text color="secondary" size="sm">
                (edited)
              </Typography.Text>
            )}
          </div>
          {!isEditing && (
            <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
              <Button
                componentId="mlflow.issues.edit-comment-button"
                icon={<PencilIcon />}
                type="tertiary"
                size="small"
                onClick={() => {
                  setEditContent(comment.content);
                  setIsEditing(true);
                }}
                aria-label="Edit comment"
              />
              <Button
                componentId="mlflow.issues.delete-comment-button"
                icon={<TrashIcon />}
                type="tertiary"
                size="small"
                onClick={handleDelete}
                loading={deleteMutation.isLoading}
                aria-label="Delete comment"
              />
            </div>
          )}
        </div>
        {isEditing ? (
          <div>
            <textarea
              value={editContent}
              onChange={(e) => setEditContent(e.target.value)}
              css={{
                width: '100%',
                minHeight: 60,
                border: `1px solid ${theme.colors.actionPrimaryBackgroundDefault}`,
                borderRadius: theme.borders.borderRadiusMd,
                padding: theme.spacing.sm,
                fontSize: theme.typography.fontSizeBase,
                resize: 'vertical',
                marginBottom: theme.spacing.sm,
              }}
            />
            <div css={{ display: 'flex', gap: theme.spacing.sm }}>
              <Button
                componentId="mlflow.issues.save-comment-button"
                type="primary"
                size="small"
                onClick={handleSave}
                loading={updateMutation.isLoading}
              >
                <FormattedMessage defaultMessage="Save" description="Save button label" />
              </Button>
              <Button
                componentId="mlflow.issues.cancel-edit-comment-button"
                type="tertiary"
                size="small"
                onClick={() => setIsEditing(false)}
              >
                <FormattedMessage defaultMessage="Cancel" description="Cancel button label" />
              </Button>
            </div>
          </div>
        ) : (
          <div>
            <GenAIMarkdownRenderer>{comment.content}</GenAIMarkdownRenderer>
          </div>
        )}
      </div>
    </div>
  );
};

const CommentsTabContent = ({ issue }: { issue: Issue }) => {
  const { theme } = useDesignSystemTheme();
  const [newComment, setNewComment] = useState('');
  const { data: comments, isLoading, refetch } = useSearchIssueComments(issue.issue_id);
  const createMutation = useCreateIssueCommentMutation();

  const handleAddComment = async () => {
    if (newComment.trim()) {
      await createMutation.mutateAsync({
        issueId: issue.issue_id,
        content: newComment.trim(),
      });
      setNewComment('');
      refetch();
    }
  };

  if (isLoading) {
    return (
      <div
        css={{
          padding: theme.spacing.lg,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <Spinner />
      </div>
    );
  }

  return (
    <div css={{ padding: theme.spacing.md }}>
      {/* New comment input */}
      <div css={{ marginBottom: theme.spacing.lg }}>
        <Typography.Text bold css={{ marginBottom: theme.spacing.xs, display: 'block' }}>
          <FormattedMessage defaultMessage="Add a comment" description="Label for new comment input" />
        </Typography.Text>
        <textarea
          value={newComment}
          onChange={(e) => setNewComment(e.target.value)}
          placeholder="Write a comment..."
          css={{
            width: '100%',
            minHeight: 80,
            border: `1px solid ${theme.colors.border}`,
            borderRadius: theme.borders.borderRadiusMd,
            padding: theme.spacing.sm,
            fontSize: theme.typography.fontSizeBase,
            resize: 'vertical',
            marginBottom: theme.spacing.sm,
            '&:focus': {
              borderColor: theme.colors.actionPrimaryBackgroundDefault,
              outline: 'none',
            },
          }}
        />
        <Button
          componentId="mlflow.issues.add-comment-button"
          type="primary"
          onClick={handleAddComment}
          disabled={!newComment.trim()}
          loading={createMutation.isLoading}
        >
          <FormattedMessage defaultMessage="Add Comment" description="Button to add a new comment" />
        </Button>
      </div>

      {/* Comments list */}
      <div>
        <Typography.Text bold css={{ marginBottom: theme.spacing.sm, display: 'block' }}>
          <FormattedMessage
            defaultMessage="Comments ({count})"
            description="Comments section header with count"
            values={{ count: comments?.length || 0 }}
          />
        </Typography.Text>
        {comments && comments.length > 0 ? (
          <div>
            {comments.map((comment) => (
              <CommentItem
                key={comment.comment_id}
                comment={comment}
                issueId={issue.issue_id}
                onUpdated={() => refetch()}
              />
            ))}
          </div>
        ) : (
          <Empty
            description={
              <FormattedMessage
                defaultMessage="No comments yet. Be the first to add one!"
                description="Empty state for comments"
              />
            }
          />
        )}
      </div>
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
          <Tabs.Trigger value="judge">
            <FormattedMessage defaultMessage="Judge" description="Tab label for judge section" />
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
        <Tabs.Content value="judge">
          <JudgeTabContent issue={issue} experimentId={experimentId} />
        </Tabs.Content>
        <Tabs.Content value="evaluation-runs">
          <EvaluationRunsTabContent issue={issue} experimentId={experimentId} />
        </Tabs.Content>
        <Tabs.Content value="comments">
          <CommentsTabContent issue={issue} />
        </Tabs.Content>
      </Tabs.Root>
    </div>
  );
};
