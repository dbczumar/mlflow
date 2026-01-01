import { useState, useMemo, useEffect, useRef } from 'react';
import { useDesignSystemTheme, Empty } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { ErrorBoundary } from 'react-error-boundary';
import { useParams, useSearchParams } from '../../../common/utils/RoutingUtils';
import { useSearchIssues } from './hooks/useIssuesApi';
import type { Issue, IssueState } from './types';
import { IssuesListPanel } from './IssuesListPanel';
import { IssueDetailPanel } from './IssueDetailPanel';
import { useGlobalClaudeOptional } from '@mlflow/mlflow/src/shared/web-shared/claude-agent';

const ErrorFallback = ({ error }: { error?: Error }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        alignItems: 'center',
        justifyContent: 'center',
        padding: theme.spacing.lg,
      }}
    >
      <Empty
        title={
          <FormattedMessage
            defaultMessage="Unable to load issues"
            description="Error message when issues page fails to load"
          />
        }
        description={
          error ? (
            <span>{error.message}</span>
          ) : (
            <FormattedMessage
              defaultMessage="We encountered an issue loading the issues interface. Please refresh the page or contact support if the problem persists."
              description="Error description for issues page loading failure"
            />
          )
        }
      />
    </div>
  );
};

const ExperimentIssuesPageContent = ({ experimentId }: { experimentId: string }) => {
  const { theme } = useDesignSystemTheme();
  const [searchParams] = useSearchParams();
  const [selectedIssueId, setSelectedIssueId] = useState<string | null>(null);
  const [activeStateTab, setActiveStateTab] = useState<IssueState>('open');
  const [searchQuery, setSearchQuery] = useState('');
  const [urlIssueHandled, setUrlIssueHandled] = useState(false);

  // Fetch all issues for the experiment (we'll filter client-side for state tabs)
  const { data: allIssues = [], isLoading, refetch } = useSearchIssues(experimentId);

  // Get issueId from URL query params (e.g., ?issueId=xxx)
  const issueIdFromUrl = searchParams.get('issueId');

  // Filter issues by state and search query
  const filteredIssues = useMemo(() => {
    return allIssues.filter((issue) => {
      // Filter by state
      if (issue.state !== activeStateTab) {
        return false;
      }
      // Filter by search query
      if (searchQuery) {
        const query = searchQuery.toLowerCase();
        return issue.name.toLowerCase().includes(query) || issue.description?.toLowerCase().includes(query);
      }
      return true;
    });
  }, [allIssues, activeStateTab, searchQuery]);

  // Find the selected issue
  const selectedIssue = useMemo(() => {
    return allIssues.find((issue) => issue.issue_id === selectedIssueId) || null;
  }, [allIssues, selectedIssueId]);

  // Set Claude context for issue page
  const globalClaude = useGlobalClaudeOptional();
  const setContext = globalClaude?.setContext;
  const lastContextKeyRef = useRef<string | null>(null);

  useEffect(() => {
    if (!setContext || isLoading) return;

    const contextKey = selectedIssue
      ? `issue-${selectedIssue.issue_id}`
      : `issues-list-${experimentId}-${filteredIssues.length}`;

    if (lastContextKeyRef.current !== contextKey) {
      lastContextKeyRef.current = contextKey;

      if (selectedIssue) {
        // Set context for the selected issue
        setContext({
          type: 'issue',
          summary: `Issue: ${selectedIssue.name}`,
          data: {
            issue_id: selectedIssue.issue_id,
            name: selectedIssue.name,
            state: selectedIssue.state,
            description: selectedIssue.description,
            creation_time: selectedIssue.creation_time,
            last_update_time: selectedIssue.last_update_time,
            tags: selectedIssue.tags,
          },
          navigation: {
            experimentId,
            page: 'issue-detail',
          },
        });
      } else {
        // No issue selected, set context for issues list
        setContext({
          type: 'none',
          summary: `${filteredIssues.length} Issues`,
          data: null,
          navigation: {
            experimentId,
            page: 'issues',
          },
        });
      }
    }
  }, [setContext, selectedIssue, experimentId, filteredIssues.length, isLoading]);

  // Handle initial issue selection from URL query param (runs once when data loads)
  useEffect(() => {
    if (urlIssueHandled || allIssues.length === 0 || !issueIdFromUrl) {
      return;
    }

    const targetIssue = allIssues.find((issue: Issue) => issue.issue_id === issueIdFromUrl);
    if (targetIssue) {
      setSelectedIssueId(targetIssue.issue_id);
      setActiveStateTab(targetIssue.state);
    }
    setUrlIssueHandled(true);
  }, [allIssues, issueIdFromUrl, urlIssueHandled]);

  // Auto-select first issue when filtered list changes (only when no URL param or after it's handled)
  useEffect(() => {
    // Skip if we're still waiting to handle URL param
    if (issueIdFromUrl && !urlIssueHandled) {
      return;
    }

    if (filteredIssues.length > 0) {
      // Only auto-select if current selection is not in the filtered list
      if (!selectedIssueId || !filteredIssues.find((i) => i.issue_id === selectedIssueId)) {
        setSelectedIssueId(filteredIssues[0].issue_id);
      }
    } else if (selectedIssueId !== null) {
      // No issues in filtered list, clear selection
      setSelectedIssueId(null);
    }
  }, [filteredIssues, selectedIssueId, issueIdFromUrl, urlIssueHandled]);

  return (
    <div
      css={{
        display: 'flex',
        height: '100%',
        minHeight: 0,
        gap: theme.spacing.md,
      }}
    >
      <IssuesListPanel
        issues={filteredIssues}
        selectedIssueId={selectedIssueId}
        onSelectIssue={setSelectedIssueId}
        activeStateTab={activeStateTab}
        onStateTabChange={setActiveStateTab}
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
        isLoading={isLoading}
      />
      <IssueDetailPanel issue={selectedIssue} experimentId={experimentId} onIssueUpdated={refetch} />
    </div>
  );
};

const ExperimentIssuesPage = () => {
  const { experimentId } = useParams();

  return (
    <ErrorBoundary FallbackComponent={ErrorFallback}>
      {experimentId ? <ExperimentIssuesPageContent experimentId={experimentId} /> : null}
    </ErrorBoundary>
  );
};

export default ExperimentIssuesPage;
