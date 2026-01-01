/**
 * Global floating Ask Claude button component.
 * Positioned at bottom-right of the screen, always visible.
 */

import { Button, Tag, Tooltip, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { useGlobalClaudeOptional } from './GlobalClaudeContext';
import ClaudeLogo from '../../../common/static/logos/claude.svg';

const COMPONENT_ID = 'mlflow.global_claude.button';

// Rainbow gradient for modern button styling (matches IssueCard)
const RAINBOW_GRADIENT = 'linear-gradient(90deg, #64B5F6, #BA68C8, #E57373)';

/**
 * Get a short label for the context type.
 */
const getContextLabel = (type: string): string | null => {
  switch (type) {
    case 'trace':
      return 'Trace';
    case 'run':
      return 'Run';
    case 'experiment':
      return 'Experiment';
    case 'traces-list':
      return 'Traces';
    case 'runs-list':
      return 'Runs';
    case 'sessions-list':
      return 'Sessions';
    case 'session':
      return 'Session';
    case 'issue':
      return 'Issue';
    case 'model':
      return 'Model';
    default:
      return null;
  }
};

/**
 * Global floating button that opens the Claude assistant panel.
 * - Positioned at bottom-right corner
 * - Shows current context type as a badge
 * - Hidden when Claude panel is already open
 * - Hidden if Claude is not available
 */
export const GlobalClaudeButton = () => {
  const { theme } = useDesignSystemTheme();
  const globalClaude = useGlobalClaudeOptional();

  // Don't render if not within provider context
  if (!globalClaude) {
    return null;
  }

  const { openPanel, isPanelOpen, isClaudeAvailable, context } = globalClaude;

  // Don't render if Claude is not available or panel is already open
  if (!isClaudeAvailable || isPanelOpen) {
    return null;
  }

  const contextLabel = getContextLabel(context.type);

  const handleClick = () => {
    openPanel();
  };

  const borderRadius = 24; // More rounded for modern look

  return (
    <div
      css={{
        position: 'fixed',
        bottom: theme.spacing.lg,
        right: theme.spacing.lg,
        zIndex: 2200,
      }}
    >
      <Tooltip
        componentId={`${COMPONENT_ID}.tooltip`}
        content={
          <FormattedMessage defaultMessage="Ask Claude for help" description="Tooltip for global Ask Claude button" />
        }
      >
        {/* Rainbow gradient border wrapper */}
        <div
          css={{
            background: RAINBOW_GRADIENT,
            borderRadius: borderRadius,
            padding: 2, // Border width
            boxShadow: '0 4px 16px rgba(0, 0, 0, 0.15)',
            cursor: 'pointer',
            transition: 'box-shadow 0.2s ease, transform 0.2s ease',
            '&:hover': {
              boxShadow: '0 6px 24px rgba(0, 0, 0, 0.2)',
              transform: 'translateY(-2px)',
            },
          }}
          onClick={handleClick}
        >
          <Button
            componentId={COMPONENT_ID}
            icon={<img src={ClaudeLogo} width={20} height={20} alt="" aria-hidden css={{ display: 'block' }} />}
            css={{
              backgroundColor: '#ffffff !important',
              border: 'none !important',
              borderRadius: borderRadius - 2,
              padding: `${theme.spacing.md}px ${theme.spacing.lg}px`,
              height: 'auto',
              minHeight: 48,
              fontSize: theme.typography.fontSizeMd,
              display: 'flex',
              alignItems: 'center',
              gap: theme.spacing.sm,
              '&:hover': {
                backgroundColor: '#fafafa !important',
              },
            }}
          >
            <FormattedMessage defaultMessage="Ask Claude" description="Label for global Ask Claude button" />
            {contextLabel && (
              <Tag componentId={`${COMPONENT_ID}.context_tag`} color="turquoise" css={{ marginLeft: theme.spacing.xs }}>
                {contextLabel}
              </Tag>
            )}
          </Button>
        </div>
      </Tooltip>
    </div>
  );
};
