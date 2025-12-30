import type { HeaderContext } from '@tanstack/react-table';

import { HoverCard, useDesignSystemTheme, ExpandLessIcon, ExpandMoreIcon, Tooltip } from '@databricks/design-system';
import { useIntl } from '@databricks/i18n';

import { TracesTableColumnGroup, TracesTableColumnGroupToLabelMap, type EvalTraceComparisonEntry } from '../types';

type HeaderCellRendererMeta = {
  groupId: TracesTableColumnGroup;
  visibleCount: number;
  totalCount: number;
  enableGrouping?: boolean;
  issuesExpanded?: boolean;
  onToggleIssuesExpanded?: () => void;
};

export const HeaderCellRenderer = (props: HeaderContext<EvalTraceComparisonEntry, unknown>) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { groupId, visibleCount, totalCount, enableGrouping, issuesExpanded, onToggleIssuesExpanded } = props.column
    .columnDef.meta as HeaderCellRendererMeta;

  if (!enableGrouping) {
    return TracesTableColumnGroupToLabelMap[groupId as TracesTableColumnGroup];
  }

  const groupName = TracesTableColumnGroupToLabelMap[groupId as TracesTableColumnGroup];
  const isIssueGroup = groupId === TracesTableColumnGroup.ISSUE;

  // Toggle button for Issues group
  const toggleButton = isIssueGroup && onToggleIssuesExpanded && (
    <Tooltip
      componentId="issues-group-toggle-button-tooltip"
      content={
        issuesExpanded
          ? intl.formatMessage({
              defaultMessage: 'Collapse to compact view',
              description: 'Tooltip for collapsing issues to compact view',
            })
          : intl.formatMessage({
              defaultMessage: 'Expand to show individual issues as columns',
              description: 'Tooltip for expanding issues to individual columns',
            })
      }
    >
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          width: 20,
          height: 20,
          borderRadius: theme.legacyBorders.borderRadiusMd,
          cursor: 'pointer',
          color: theme.colors.textSecondary,
          ':hover': {
            backgroundColor: theme.colors.actionTertiaryBackgroundHover,
            color: theme.colors.textPrimary,
          },
        }}
        onClick={(e) => {
          e.stopPropagation();
          onToggleIssuesExpanded();
        }}
      >
        {issuesExpanded ? (
          <ExpandLessIcon css={{ fontSize: 14, transform: 'rotate(90deg)' }} />
        ) : (
          <ExpandMoreIcon css={{ fontSize: 14, transform: 'rotate(-90deg)' }} />
        )}
      </div>
    </Tooltip>
  );

  // For ISSUE group when collapsed, show total count only (e.g., "(5)")
  // For ISSUE group when expanded or other groups, show visible/total (e.g., "(3/5)")
  const renderCount = () => {
    if (groupId === TracesTableColumnGroup.INFO) {
      return null;
    }

    // ISSUE group in collapsed mode: show just total count
    if (isIssueGroup && !issuesExpanded) {
      if (totalCount === 0) {
        return null; // Don't show count when no issues
      }
      return (
        <div
          css={{
            color: theme.colors.textSecondary,
            fontWeight: 'normal',
          }}
        >
          ({totalCount})
        </div>
      );
    }

    // Normal case: show visible/total
    if (visibleCount === totalCount) {
      return (
        <div
          css={{
            color: theme.colors.textSecondary,
            fontWeight: 'normal',
          }}
        >
          ({visibleCount}/{totalCount})
        </div>
      );
    }

    return (
      <HoverCard
        trigger={
          <div
            css={{
              color: theme.colors.textSecondary,
              ':hover': {
                textDecoration: 'underline',
              },
              fontWeight: 'normal',
            }}
          >
            ({visibleCount}/{totalCount})
          </div>
        }
        content={intl.formatMessage(
          {
            defaultMessage: 'Showing {visibleCount} out of {totalCount} {groupName}. Select columns to view more.',
            description: 'Tooltip for the group column header',
          },
          {
            visibleCount,
            totalCount,
            groupName,
          },
        )}
        align="start"
      />
    );
  };

  return (
    <div
      css={{
        height: '100%',
        width: '100%',
        display: 'flex',
        overflow: 'hidden',
        alignItems: 'center',
        gap: theme.spacing.sm,
      }}
    >
      <div>{groupName}</div>
      {renderCount()}
      {toggleButton}
    </div>
  );
};
