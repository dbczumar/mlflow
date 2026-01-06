/**
 * Global MLflow Assistant Chat Panel component.
 * Displays the chat interface for the MLflow AI assistant.
 */

import {
  Button,
  GearIcon,
  RefreshIcon,
  SidebarCollapseIcon,
  Tag,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { useGlobalClaude } from './GlobalClaudeContext';
import { ClaudeAgentChatPanel } from './ClaudeAgentChatPanel';
import { OnboardingWizard } from './OnboardingWizard';
import ClaudeLogo from '../../../common/static/logos/claude.svg';
import AssistantSparklesLogo from '../../../common/static/logos/assistant-sparkles.svg';

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
 * Global MLflow Assistant Chat Panel.
 * Shows the chat header with context badge and the chat interface.
 * Displays setup wizard when configuration is needed.
 */
export const GlobalClaudeChatPanel = () => {
  const { theme } = useDesignSystemTheme();
  const { closePanel, context, reset, showSetupWizard, completeSetup, openSetup, setupStatus } = useGlobalClaude();

  const isConfigured = setupStatus === 'configured';

  const contextLabel = getContextTypeLabel(context.type);

  const handleClose = () => {
    closePanel();
  };

  const handleReset = () => {
    reset();
  };

  // Show onboarding wizard if needed
  if (showSetupWizard) {
    const currentExperimentId = context.navigation?.experimentId;

    return (
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          height: '100%',
          overflow: 'hidden',
        }}
      >
        {/* Header for onboarding */}
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
            <img src={AssistantSparklesLogo} width={18} height={18} alt="" aria-hidden />
            <FormattedMessage defaultMessage="MLflow GenAI Setup" description="Title for the onboarding wizard panel" />
          </span>
          <Button
            componentId={`${COMPONENT_ID}.close`}
            size="small"
            icon={<SidebarCollapseIcon css={{ transform: 'rotate(180deg)' }} />}
            onClick={handleClose}
            aria-label="Close assistant"
          />
        </div>

        {/* Onboarding wizard */}
        <div css={{ flex: 1, minHeight: 0, overflow: 'hidden' }}>
          <OnboardingWizard
            onComplete={completeSetup}
            currentExperimentId={currentExperimentId}
            assistantAlreadyConfigured={isConfigured}
          />
        </div>
      </div>
    );
  }

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
          {isConfigured ? (
            <>
              <img src={ClaudeLogo} width={18} height={18} alt="" aria-hidden />
              <FormattedMessage
                defaultMessage="Assistant (Claude)"
                description="Title for the Assistant panel when configured with Claude"
              />
            </>
          ) : (
            <>
              <img src={AssistantSparklesLogo} width={18} height={18} alt="" aria-hidden />
              <FormattedMessage defaultMessage="Assistant" description="Title for the Assistant panel" />
            </>
          )}
          {contextLabel && (
            <Tag componentId={`${COMPONENT_ID}.context_tag`} color="turquoise">
              {contextLabel}
            </Tag>
          )}
        </span>
        <div css={{ display: 'flex', gap: theme.spacing.xs }}>
          {/* Settings button - allows re-running setup */}
          <Button
            componentId={`${COMPONENT_ID}.settings`}
            size="small"
            icon={<GearIcon />}
            onClick={openSetup}
            aria-label="Assistant settings"
          />
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
            aria-label="Close assistant"
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
