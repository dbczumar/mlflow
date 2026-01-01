/**
 * Global Claude Chat Panel component.
 * Displays the chat interface for the global Claude assistant.
 */

import { Button, RefreshIcon, SidebarCollapseIcon, Tag, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { useGlobalClaude } from './GlobalClaudeContext';
import { ClaudeAgentChatPanel } from './ClaudeAgentChatPanel';
import ClaudeLogo from '../../../common/static/logos/claude.svg';

const COMPONENT_ID = 'mlflow.global_claude.chat_panel';

/**
 * Get a display label for the context type.
 */
const getContextTypeLabel = (type: string): string => {
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
      return '';
  }
};

/**
 * Global Claude Chat Panel.
 * Shows the chat header with context badge and the chat interface.
 */
export const GlobalClaudeChatPanel = () => {
  const { theme } = useDesignSystemTheme();
  const { closePanel, context, reset } = useGlobalClaude();

  const contextLabel = getContextTypeLabel(context.type);

  const handleClose = () => {
    closePanel();
  };

  const handleReset = () => {
    reset();
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
          <FormattedMessage defaultMessage="Ask Claude" description="Title for the global Claude chat panel" />
          {contextLabel && (
            <Tag componentId={`${COMPONENT_ID}.context_tag`} color="turquoise">
              {contextLabel}
            </Tag>
          )}
        </span>
        <div css={{ display: 'flex', gap: theme.spacing.xs }}>
          <Button
            componentId={`${COMPONENT_ID}.reset`}
            size="small"
            icon={<RefreshIcon />}
            onClick={handleReset}
            aria-label="Reset conversation"
          />
          <Button
            componentId={`${COMPONENT_ID}.close`}
            size="small"
            icon={<SidebarCollapseIcon css={{ transform: 'rotate(180deg)' }} />}
            onClick={handleClose}
            aria-label="Minimize Claude chat"
          />
        </div>
      </div>

      {/* Context summary */}
      {context.summary && (
        <div
          css={{
            padding: `${theme.spacing.xs}px ${theme.spacing.md}px`,
            backgroundColor: theme.colors.backgroundSecondary,
            borderBottom: `1px solid ${theme.colors.border}`,
            fontSize: theme.typography.fontSizeSm,
            color: theme.colors.textSecondary,
          }}
        >
          {context.summary}
        </div>
      )}

      {/* Chat panel */}
      <div css={{ flex: 1, minHeight: 0 }}>
        <ChatPanelGlobal />
      </div>
    </div>
  );
};

/**
 * Chat panel wrapper for global mode.
 */
const ChatPanelGlobal = () => {
  const { messages, isStreaming, error, currentStatus, sendMessage } = useGlobalClaude();

  return (
    <ClaudeAgentChatPanel
      messages={messages}
      isStreaming={isStreaming}
      error={error}
      currentStatus={currentStatus}
      onSendMessage={sendMessage}
      isGlobalMode
    />
  );
};
