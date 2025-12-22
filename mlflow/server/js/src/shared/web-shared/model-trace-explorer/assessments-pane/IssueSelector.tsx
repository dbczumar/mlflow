import { useEffect, useMemo, useState } from 'react';

import {
  TypeaheadComboboxInput,
  TypeaheadComboboxMenu,
  TypeaheadComboboxMenuItem,
  TypeaheadComboboxRoot,
  Typography,
  useComboboxState,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useIntl } from '@databricks/i18n';

import type { Issue } from '../../../../experiment-tracking/pages/experiment-issues/types';
import { useSearchIssues } from '../../../../experiment-tracking/pages/experiment-issues/hooks/useIssuesApi';
import { useLinkIssueToTrace } from '../hooks/useLinkIssueToTrace';

export interface IssueSelectorProps {
  experimentId: string;
  traceId: string;
  spanId?: string;
  linkedIssueIds: string[];
  onIssueLinked?: () => void;
}

export const IssueSelector = ({ experimentId, traceId, spanId, linkedIssueIds, onIssueLinked }: IssueSelectorProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  // Fetch all open issues for the experiment
  const { data: allIssues = [], isLoading: isLoadingIssues } = useSearchIssues(experimentId);

  // Filter to only open issues that are not already linked
  const availableIssues = useMemo(() => {
    return allIssues.filter((issue) => issue.state === 'open' && !linkedIssueIds.includes(issue.issue_id));
  }, [allIssues, linkedIssueIds]);

  const [inputValue, setInputValue] = useState('');
  const [filteredItems, setFilteredItems] = useState<(Issue | null)[]>(availableIssues);
  const [selectedItem, setSelectedItem] = useState<Issue | null>(null);

  const { linkIssueToTrace, isLoading: isLinking } = useLinkIssueToTrace({
    traceId,
    spanId,
    onSuccess: () => {
      setSelectedItem(null);
      setInputValue('');
      onIssueLinked?.();
    },
  });

  const handleSelectIssue = (issue: Issue | null) => {
    if (issue) {
      linkIssueToTrace(issue);
    }
    setSelectedItem(null);
  };

  const comboboxState = useComboboxState<Issue | null>({
    componentId: 'shared.model-trace-explorer.issue-selector-typeahead',
    allItems: availableIssues,
    items: filteredItems,
    setItems: setFilteredItems,
    multiSelect: false,
    setInputValue: (value) => {
      setInputValue(value);
    },
    itemToString: (item) => item?.name ?? '',
    matcher: (item, query) => {
      const lowerQuery = query.toLowerCase();
      return (
        (item?.name?.toLowerCase().includes(lowerQuery) ?? false) ||
        (item?.description?.toLowerCase().includes(lowerQuery) ?? false)
      );
    },
    formValue: selectedItem,
    formOnChange: handleSelectIssue,
    preventUnsetOnBlur: true,
  });

  // Reset filtered items when available issues change
  useEffect(() => {
    setFilteredItems(availableIssues);
  }, [availableIssues]);

  const isDisabled = isLoadingIssues || isLinking;

  return (
    <div css={{ marginTop: theme.spacing.sm }}>
      <Typography.Text color="secondary" css={{ marginBottom: theme.spacing.xs, display: 'block' }}>
        {intl.formatMessage({
          defaultMessage: 'Select or create issue',
          description: 'Label for the issue selector in trace detail',
        })}
      </Typography.Text>
      <TypeaheadComboboxRoot
        id="shared.model-trace-explorer.issue-selector-typeahead"
        comboboxState={comboboxState}
        onKeyDown={(e) => {
          // Prevent navigation while typing
          if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
            e.stopPropagation();
          }
        }}
      >
        <TypeaheadComboboxInput
          data-testid="issue-selector-input"
          placeholder={intl.formatMessage({
            defaultMessage: 'Start typing issue name...',
            description: 'Placeholder for the issue selector input',
          })}
          comboboxState={comboboxState}
          formOnChange={handleSelectIssue}
          disabled={isDisabled}
          allowClear
          showComboboxToggleButton
        />
        <TypeaheadComboboxMenu comboboxState={comboboxState}>
          {filteredItems.length === 0 ? (
            <div css={{ padding: theme.spacing.sm, textAlign: 'center' }}>
              <Typography.Text color="secondary">
                {availableIssues.length === 0
                  ? intl.formatMessage({
                      defaultMessage: 'No open issues available',
                      description: 'Message when no open issues are available to link',
                    })
                  : intl.formatMessage({
                      defaultMessage: 'No matching issues found',
                      description: 'Message when no issues match the search query',
                    })}
              </Typography.Text>
            </div>
          ) : (
            filteredItems.map((issue, index) =>
              issue ? (
                <TypeaheadComboboxMenuItem
                  data-testid={`issue-selector-item-${issue.issue_id}`}
                  key={issue.issue_id}
                  item={issue}
                  index={index}
                  comboboxState={comboboxState}
                >
                  <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                    <Typography.Text bold css={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {issue.name}
                    </Typography.Text>
                    {issue.description && (
                      <Typography.Text
                        color="secondary"
                        css={{
                          fontSize: theme.typography.fontSizeSm,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                          maxWidth: 250,
                        }}
                      >
                        {issue.description.split('\n')[0]}
                      </Typography.Text>
                    )}
                  </div>
                </TypeaheadComboboxMenuItem>
              ) : null,
            )
          )}
        </TypeaheadComboboxMenu>
      </TypeaheadComboboxRoot>
    </div>
  );
};
