/**
 * Ask Claude button component for triggering trace analysis.
 */

import { Button, SparkleIcon, Tooltip, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import type { ModelTrace } from '../model-trace-explorer/ModelTrace.types';
import { useClaudeAgentContextOptional } from './ClaudeAgentContext';

const COMPONENT_ID = 'mlflow.trace.ask_claude.button';

interface AskClaudeButtonProps {
  trace: ModelTrace;
}

/**
 * Button that opens the Claude Agent tab for trace analysis.
 * Only renders if within a ClaudeAgentProvider context.
 */
export const AskClaudeButton = ({ trace }: AskClaudeButtonProps) => {
  const { theme } = useDesignSystemTheme();
  const claudeAgent = useClaudeAgentContextOptional();

  // Don't render if not within provider context
  if (!claudeAgent) {
    return null;
  }

  const { openClaudeTab, isClaudeTabActive } = claudeAgent;

  const handleClick = () => {
    openClaudeTab(trace);
  };

  return (
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
        size="small"
        onClick={handleClick}
        disabled={isClaudeTabActive}
        icon={<SparkleIcon css={{ color: theme.colors.purple }} />}
      >
        <FormattedMessage defaultMessage="Ask Claude" description="Label for Ask Claude button" />
      </Button>
    </Tooltip>
  );
};
