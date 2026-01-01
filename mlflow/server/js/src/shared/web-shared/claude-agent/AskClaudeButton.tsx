/**
 * Floating Ask Claude button component for triggering trace analysis.
 * Positioned at bottom-right of the trace explorer, above all other content.
 */

import { Button, Tooltip, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import type { ModelTrace } from '../model-trace-explorer/ModelTrace.types';
import { useClaudeAgentContextOptional } from './ClaudeAgentContext';
import ClaudeLogo from '../../../common/static/logos/claude.svg';

const COMPONENT_ID = 'mlflow.trace.ask_claude.button';

interface AskClaudeButtonProps {
  trace: ModelTrace;
}

/**
 * Floating button that opens the Claude Agent tab for trace analysis.
 * Positioned at bottom-right corner, above assessment pane and other content.
 * Only renders if within a ClaudeAgentProvider context and Claude tab is not active.
 */
export const AskClaudeButton = ({ trace }: AskClaudeButtonProps) => {
  const { theme } = useDesignSystemTheme();
  const claudeAgent = useClaudeAgentContextOptional();

  // Don't render if not within provider context
  if (!claudeAgent) {
    return null;
  }

  const { openClaudeTab, isClaudeTabActive, isClaudeAvailable } = claudeAgent;

  // Don't render if Claude is not available or tab is already active
  if (!isClaudeAvailable || isClaudeTabActive) {
    return null;
  }

  const handleClick = () => {
    openClaudeTab(trace);
  };

  return (
    <div
      css={{
        position: 'absolute',
        bottom: theme.spacing.lg,
        right: theme.spacing.lg,
        zIndex: 1000,
      }}
    >
      <Tooltip
        componentId={`${COMPONENT_ID}.tooltip`}
        content={
          <FormattedMessage
            defaultMessage="Analyze this trace with Claude AI"
            description="Tooltip for Ask Claude button"
          />
        }
      >
        <Button
          componentId={COMPONENT_ID}
          onClick={handleClick}
          icon={<img src={ClaudeLogo} width={20} height={20} alt="" aria-hidden css={{ display: 'block' }} />}
          css={{
            backgroundColor: '#ffffff !important',
            border: `1px solid ${theme.colors.border}`,
            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)',
            padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
            height: 'auto',
            fontSize: theme.typography.fontSizeMd,
            '&:hover': {
              backgroundColor: '#f5f5f5 !important',
              border: `1px solid ${theme.colors.border}`,
            },
          }}
        >
          <FormattedMessage defaultMessage="Ask Claude" description="Label for Ask Claude button" />
        </Button>
      </Tooltip>
    </div>
  );
};
