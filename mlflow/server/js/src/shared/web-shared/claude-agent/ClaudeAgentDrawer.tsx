/**
 * Tab content component for Claude Agent interaction.
 * This component is displayed as a tab in the trace explorer.
 */

import { Button, useDesignSystemTheme, CloseIcon, SparkleIcon } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { useClaudeAgentContext } from './ClaudeAgentContext';
import { ClaudeAgentChatPanel } from './ClaudeAgentChatPanel';

const COMPONENT_ID = 'mlflow.trace.claude_agent.tab_content';

/**
 * Tab content component for Claude Agent.
 * Displays the chat interface within the trace explorer tabs.
 */
export const ClaudeAgentTabContent = () => {
  const { theme } = useDesignSystemTheme();
  const { closeClaudeTab, reset } = useClaudeAgentContext();

  const handleClose = () => {
    closeClaudeTab();
    // Reset state after closing
    setTimeout(() => {
      reset();
    }, 100);
  };

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: theme.spacing.md,
          borderBottom: `1px solid ${theme.colors.border}`,
          flexShrink: 0,
        }}
      >
        <span
          css={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: theme.spacing.sm,
            fontWeight: theme.typography.typographyBoldFontWeight,
          }}
        >
          <SparkleIcon css={{ color: theme.colors.purple }} />
          <FormattedMessage defaultMessage="Ask Claude" description="Title for the Claude Agent tab" />
        </span>
        <Button
          componentId={`${COMPONENT_ID}.close`}
          size="small"
          icon={<CloseIcon />}
          onClick={handleClose}
          aria-label="Close Claude chat"
        />
      </div>

      {/* Chat panel */}
      <div css={{ flex: 1, minHeight: 0 }}>
        <ClaudeAgentChatPanel />
      </div>
    </div>
  );
};

// Keep old export name for backwards compatibility during transition
export const ClaudeAgentDrawer = ClaudeAgentTabContent;
