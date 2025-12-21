import { useState, useMemo, useEffect } from 'react';
import { useDesignSystemTheme, Empty } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { ErrorBoundary } from 'react-error-boundary';
import { useParams } from '../../../common/utils/RoutingUtils';
import { useSearchIssues } from './hooks/useIssuesApi';
import type { IssueState } from './types';
import { IssuesListPanel } from './IssuesListPanel';
import { IssueDetailPanel } from './IssueDetailPanel';

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
  const [selectedIssueId, setSelectedIssueId] = useState<string | null>(null);
  const [activeStateTab, setActiveStateTab] = useState<IssueState>('open');
  const [searchQuery, setSearchQuery] = useState('');

  // Fetch all issues for the experiment (we'll filter client-side for state tabs)
  const { data: allIssues = [], isLoading, refetch } = useSearchIssues(experimentId);

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

  // Auto-select first issue when filtered list changes
  useEffect(() => {
    if (filteredIssues.length > 0 && !filteredIssues.find((i) => i.issue_id === selectedIssueId)) {
      setSelectedIssueId(filteredIssues[0].issue_id);
    } else if (filteredIssues.length === 0) {
      setSelectedIssueId(null);
    }
  }, [filteredIssues, selectedIssueId]);

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
      <IssueDetailPanel
        issue={selectedIssue}
        experimentId={experimentId}
        onIssueUpdated={refetch}
        issueNumber={selectedIssue ? allIssues.findIndex((i) => i.issue_id === selectedIssue.issue_id) + 1 : undefined}
      />
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
