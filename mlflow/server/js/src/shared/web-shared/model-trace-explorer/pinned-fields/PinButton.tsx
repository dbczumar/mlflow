import { PinFillIcon, PinIcon, Tooltip, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

export interface PinButtonProps {
  isPinned: boolean;
  onToggle: () => void;
}

export const PinButton = ({ isPinned, onToggle }: PinButtonProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <Tooltip
      componentId="shared.model-trace-explorer.pin-button-tooltip"
      content={
        isPinned ? (
          <FormattedMessage defaultMessage="Unpin from top" description="Tooltip for unpin button in trace explorer" />
        ) : (
          <FormattedMessage defaultMessage="Pin to top" description="Tooltip for pin button in trace explorer" />
        )
      }
    >
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          onToggle();
        }}
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: theme.spacing.xs,
          border: 'none',
          background: 'transparent',
          cursor: 'pointer',
          borderRadius: theme.borders.borderRadiusSm,
          color: isPinned ? theme.colors.actionPrimaryBackgroundDefault : theme.colors.textSecondary,
          transition: 'background 0.15s ease-in-out, opacity 0.15s ease-in-out',
          ':hover': {
            background: theme.colors.actionTertiaryBackgroundHover,
          },
        }}
        aria-label={isPinned ? 'Unpin from top' : 'Pin to top'}
      >
        {isPinned ? <PinFillIcon /> : <PinIcon />}
      </button>
    </Tooltip>
  );
};
