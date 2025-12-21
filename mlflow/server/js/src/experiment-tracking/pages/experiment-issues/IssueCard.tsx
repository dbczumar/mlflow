import React from 'react';
import { useDesignSystemTheme, Typography, DragIcon } from '@databricks/design-system';
import type { Issue } from './types';

const RAINBOW_GRADIENT = 'linear-gradient(90deg, #64B5F6, #BA68C8, #E57373)';
const MAX_DESCRIPTION_LENGTH = 80;

interface IssueCardProps {
  issue: Issue;
  isSelected: boolean;
  onClick: () => void;
}

const IssueBadge = ({ name }: { name: string }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <span
      css={{
        display: 'inline-flex',
        alignItems: 'center',
        padding: `2px 8px`,
        borderRadius: theme.borders.borderRadiusMd,
        background: RAINBOW_GRADIENT,
        fontSize: theme.typography.fontSizeSm,
        color: theme.colors.textPrimary,
        whiteSpace: 'nowrap',
        maxWidth: '100%',
        position: 'relative',
        fontWeight: 500,
        '&::before': {
          content: '""',
          position: 'absolute',
          inset: 1,
          borderRadius: theme.borders.borderRadiusMd - 1,
          backgroundColor: theme.colors.backgroundPrimary,
        },
      }}
    >
      <span css={{ position: 'relative', zIndex: 1 }}>{name}</span>
    </span>
  );
};

const StateIcon = ({ state }: { state: Issue['state'] }) => {
  const { theme } = useDesignSystemTheme();

  const colors = {
    open: theme.colors.blue400,
    closed: theme.colors.green400,
    draft: theme.colors.grey400,
  };

  return (
    <div
      css={{
        width: 16,
        height: 16,
        borderRadius: '50%',
        border: `2px solid ${colors[state]}`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexShrink: 0,
      }}
    >
      {state === 'closed' && (
        <div
          css={{
            width: 8,
            height: 8,
            borderRadius: '50%',
            backgroundColor: colors[state],
          }}
        />
      )}
    </div>
  );
};

export const IssueCard = ({ issue, isSelected, onClick }: IssueCardProps) => {
  const { theme } = useDesignSystemTheme();

  const truncatedDescription =
    issue.description && issue.description.length > MAX_DESCRIPTION_LENGTH
      ? `${issue.description.substring(0, MAX_DESCRIPTION_LENGTH)}...`
      : issue.description;

  return (
    <div
      onClick={onClick}
      css={{
        display: 'flex',
        gap: theme.spacing.sm,
        padding: theme.spacing.sm,
        borderRadius: theme.borders.borderRadiusMd,
        cursor: 'pointer',
        backgroundColor: isSelected ? theme.colors.actionDefaultBackgroundPress : 'transparent',
        border: `1px solid ${isSelected ? theme.colors.actionPrimaryBackgroundDefault : 'transparent'}`,
        transition: 'background 0.15s ease, border 0.15s ease',
        '&:hover': {
          backgroundColor: isSelected
            ? theme.colors.actionDefaultBackgroundPress
            : theme.colors.actionDefaultBackgroundHover,
        },
      }}
    >
      <StateIcon state={issue.state} />
      <div css={{ flex: 1, minWidth: 0 }}>
        <div css={{ marginBottom: theme.spacing.xs }}>
          <IssueBadge name={issue.name} />
        </div>
        {truncatedDescription && (
          <Typography.Text
            color="secondary"
            css={{
              fontSize: theme.typography.fontSizeSm,
              display: '-webkit-box',
              WebkitLineClamp: 2,
              WebkitBoxOrient: 'vertical',
              overflow: 'hidden',
            }}
          >
            {truncatedDescription}
          </Typography.Text>
        )}
      </div>
      <DragIcon
        css={{
          color: theme.colors.grey400,
          flexShrink: 0,
          cursor: 'grab',
        }}
      />
    </div>
  );
};
