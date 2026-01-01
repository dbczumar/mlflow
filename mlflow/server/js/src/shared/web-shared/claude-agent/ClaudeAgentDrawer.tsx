/**
 * Drawer component for Claude Agent interaction.
 */

import { Drawer, useDesignSystemTheme, SparkleIcon } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { useClaudeAgentContext } from './ClaudeAgentContext';
import { ClaudeAgentChatPanel } from './ClaudeAgentChatPanel';

const COMPONENT_ID = 'mlflow.trace.claude_agent.drawer';

/**
 * Main drawer component for Claude Agent.
 */
export const ClaudeAgentDrawer = () => {
  const { theme } = useDesignSystemTheme();
  const { isDrawerOpen, closeDrawer, reset } = useClaudeAgentContext();

  const handleOpenChange = (open: boolean) => {
    if (!open) {
      closeDrawer();
      // Reset state after drawer closes
      setTimeout(() => {
        reset();
      }, 300);
    }
  };

  return (
    <Drawer.Root modal open={isDrawerOpen} onOpenChange={handleOpenChange}>
      <Drawer.Content
        componentId={COMPONENT_ID}
        width="50vw"
        title={
          <span
            css={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: theme.spacing.sm,
            }}
          >
            <span
              css={{
                borderRadius: theme.borders.borderRadiusSm,
                background: theme.colors.actionDefaultBackgroundHover,
                padding: theme.spacing.xs,
                color: theme.colors.purple,
                height: 'min-content',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <SparkleIcon />
            </span>
            <FormattedMessage defaultMessage="Ask Claude" description="Title for the Claude Agent drawer" />
          </span>
        }
        expandContentToFullHeight
      >
        <div
          css={{
            height: '100%',
            marginLeft: -theme.spacing.lg,
            marginRight: -theme.spacing.lg,
            marginBottom: -theme.spacing.lg,
          }}
        >
          <ClaudeAgentChatPanel />
        </div>
      </Drawer.Content>
    </Drawer.Root>
  );
};
