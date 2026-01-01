import React from 'react';
import {
  useDesignSystemTheme,
  Input,
  Button,
  TitleSkeleton,
  Typography,
  SearchIcon,
  FilterIcon,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import type { Issue, IssueState } from './types';
import { IssueCard } from './IssueCard';

const LIST_PANEL_WIDTH = 380;

interface IssuesListPanelProps {
  issues: Issue[];
  selectedIssueId: string | null;
  onSelectIssue: (issueId: string) => void;
  activeStateTab: IssueState;
  onStateTabChange: (state: IssueState) => void;
  searchQuery: string;
  onSearchChange: (query: string) => void;
  isLoading: boolean;
}

const StateTab = ({ label, isActive, onClick }: { label: React.ReactNode; isActive: boolean; onClick: () => void }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <button
      onClick={onClick}
      css={{
        display: 'flex',
        alignItems: 'center',
        padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
        border: 'none',
        borderRadius: theme.borders.borderRadiusSm,
        background: isActive ? theme.colors.actionDefaultBackgroundPress : 'transparent',
        color: isActive ? theme.colors.actionPrimaryBackgroundDefault : theme.colors.textSecondary,
        cursor: 'pointer',
        fontWeight: isActive ? 600 : 400,
        fontSize: theme.typography.fontSizeSm,
        transition: 'background 0.15s ease',
        '&:hover': {
          background: isActive ? theme.colors.actionDefaultBackgroundPress : theme.colors.actionDefaultBackgroundHover,
        },
      }}
    >
      {label}
    </button>
  );
};

export const IssuesListPanel = ({
  issues,
  selectedIssueId,
  onSelectIssue,
  activeStateTab,
  onStateTabChange,
  searchQuery,
  onSearchChange,
  isLoading,
}: IssuesListPanelProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  return (
    <div
      css={{
        width: LIST_PANEL_WIDTH,
        minWidth: LIST_PANEL_WIDTH,
        display: 'flex',
        flexDirection: 'column',
        borderRight: `1px solid ${theme.colors.border}`,
        height: '100%',
        minHeight: 0,
      }}
    >
      {/* State tabs */}
      <div
        css={{
          display: 'flex',
          gap: theme.spacing.xs,
          paddingBottom: theme.spacing.sm,
          borderBottom: `1px solid ${theme.colors.border}`,
        }}
      >
        <StateTab
          label={<FormattedMessage defaultMessage="Open" description="Label for Open issues tab" />}
          isActive={activeStateTab === 'open'}
          onClick={() => onStateTabChange('open')}
        />
        <StateTab
          label={<FormattedMessage defaultMessage="Closed" description="Label for Closed issues tab" />}
          isActive={activeStateTab === 'closed'}
          onClick={() => onStateTabChange('closed')}
        />
        <StateTab
          label={<FormattedMessage defaultMessage="Drafts" description="Label for Draft issues tab" />}
          isActive={activeStateTab === 'draft'}
          onClick={() => onStateTabChange('draft')}
        />
      </div>

      {/* Search and filter */}
      <div
        css={{
          display: 'flex',
          gap: theme.spacing.sm,
          padding: `${theme.spacing.sm}px 0`,
        }}
      >
        <Input
          componentId="mlflow.issues.search-input"
          prefix={<SearchIcon />}
          placeholder={intl.formatMessage({
            defaultMessage: 'Search issues',
            description: 'Placeholder for issue search input',
          })}
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          css={{ flex: 1 }}
        />
        <Button componentId="mlflow.issues.filter-button" icon={<FilterIcon />} type="tertiary" aria-label="Filter" />
      </div>

      {/* Issue list */}
      <div
        css={{
          flex: 1,
          overflow: 'auto',
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.xs,
        }}
      >
        {isLoading ? (
          <>
            <TitleSkeleton css={{ width: '100%' }} />
            <TitleSkeleton css={{ width: '100%' }} />
            <TitleSkeleton css={{ width: '100%' }} />
          </>
        ) : issues.length === 0 ? (
          <div
            css={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              padding: theme.spacing.lg,
            }}
          >
            <Typography.Text color="secondary">
              <FormattedMessage
                defaultMessage="No {state} issues"
                description="Message when no issues match the filter"
                values={{ state: activeStateTab }}
              />
            </Typography.Text>
          </div>
        ) : (
          issues.map((issue) => (
            <IssueCard
              key={issue.issue_id}
              issue={issue}
              isSelected={issue.issue_id === selectedIssueId}
              onClick={() => onSelectIssue(issue.issue_id)}
            />
          ))
        )}
      </div>
    </div>
  );
};
