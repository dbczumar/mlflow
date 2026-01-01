/**
 * Tab content component for Claude Agent interaction.
 * This component is displayed as a tab in the trace explorer.
 */

import { Button, useDesignSystemTheme, CloseIcon } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { useClaudeAgentContext } from './ClaudeAgentContext';
import { ClaudeAgentChatPanel } from './ClaudeAgentChatPanel';
import ClaudeLogo from '../../../common/static/logos/claude.svg';

const COMPONENT_ID = 'mlflow.trace.claude_agent.tab_content';

/**
 * Tab content component for Claude Agent.
 * Displays the chat interface within the trace explorer tabs.
 */
export const ClaudeAgentTabContent = () => {
  const { theme } = useDesignSystemTheme();
  const { closeClaudeTab } = useClaudeAgentContext();

  const handleClose = () => {
    closeClaudeTab();
    // Note: We don't reset the session here to preserve conversation history
    // The session can be explicitly reset via the reset button if needed
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
          <img src={ClaudeLogo} width={18} height={18} alt="" aria-hidden css={{ display: 'block' }} />
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
