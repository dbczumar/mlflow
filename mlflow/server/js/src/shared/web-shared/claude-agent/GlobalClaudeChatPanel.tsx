/**
 * Global MLflow Assistant Chat Panel component.
 * Displays the chat interface for the MLflow AI assistant.
 */

import { useState } from 'react';
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
import { AssistantBackendStep } from './onboarding/AssistantBackendStep';
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
 * Extract experiment ID from the current URL hash.
 * Returns the experiment ID if found in the URL, otherwise undefined.
 */
const getExperimentIdFromUrl = (): string | undefined => {
  const hash = window.location.hash;
  // Match patterns like #/experiments/123/... or #/experiments/123
  const match = hash.match(/#\/experiments\/([^/]+)/);
  return match ? match[1] : undefined;
};

/**
 * Global MLflow Assistant Chat Panel.
 * Shows the chat header with context badge and the chat interface.
 * Displays setup wizard when configuration is needed.
 */
export const GlobalClaudeChatPanel = () => {
  const { theme } = useDesignSystemTheme();
  const { closePanel, context, reset, showSetupWizard, completeSetup, openSetup, setupStatus } = useGlobalClaude();
  const [showBackendConfig, setShowBackendConfig] = useState(false);

  const isConfigured = setupStatus === 'configured';

  const contextLabel = getContextTypeLabel(context.type);

  const handleClose = () => {
    closePanel();
  };

  const handleReset = () => {
    reset();
  };

  const handleOpenSettings = () => {
    if (isConfigured) {
      // If already configured, show backend configuration
      setShowBackendConfig(true);
    } else {
      // If not configured, show the full onboarding wizard
      openSetup();
    }
  };

  const handleBackendConfigured = () => {
    setShowBackendConfig(false);
    completeSetup();
  };

  const handleCloseBackendConfig = () => {
    setShowBackendConfig(false);
  };

  // Show assistant backend configuration if requested
  if (showBackendConfig) {
    return (
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          height: '100%',
          overflow: 'hidden',
        }}
      >
        {/* Header for backend config */}
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
            <FormattedMessage
              defaultMessage="Assistant Configuration"
              description="Title for assistant backend configuration"
            />
          </span>
          <Button
            componentId={`${COMPONENT_ID}.close_backend_config`}
            size="small"
            icon={<SidebarCollapseIcon css={{ transform: 'rotate(180deg)' }} />}
            onClick={handleCloseBackendConfig}
            aria-label="Close configuration"
          />
        </div>

        {/* Assistant backend configuration */}
        <div css={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
          <AssistantBackendStep onConfigured={handleBackendConfigured} />
        </div>
      </div>
    );
  }

  // Show onboarding wizard if needed
  if (showSetupWizard) {
    // Prefer experiment ID from URL (source of truth) over context (which may be stale)
    const currentExperimentId = getExperimentIdFromUrl() || context.navigation?.experimentId;

    return (
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          height: '100%',
          overflow: 'hidden',
        }}
      >
        {/* DEBUG: Show experimentId being used */}
        <div css={{ padding: theme.spacing.sm, backgroundColor: '#ff0', color: '#000', fontSize: '12px' }}>
          DEBUG: URL ExpID={getExperimentIdFromUrl()} | Context ExpID={context.navigation?.experimentId} | Using=
          {currentExperimentId}
        </div>
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
            {currentExperimentId ? (
              <FormattedMessage
                defaultMessage="GenAI Experiment Setup"
                description="Title for the onboarding wizard panel when in experiment"
              />
            ) : (
              <FormattedMessage
                defaultMessage="MLflow Setup"
                description="Title for the onboarding wizard panel when not in experiment"
              />
            )}
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
          {/* Settings button - allows re-running setup or configuring backend */}
          <Button
            componentId={`${COMPONENT_ID}.settings`}
            size="small"
            icon={<GearIcon />}
            onClick={handleOpenSettings}
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
