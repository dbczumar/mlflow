/**
 * Set up the AI assistant backend (Claude Code).
 * Can be used standalone or embedded in InstrumentationStep.
 */

import { useCallback, useState } from 'react';
import {
  Button,
  CheckCircleIcon,
  DangerIcon,
  Input,
  Spinner,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { checkHealth } from '../ClaudeAgentService';
import ClaudeLogo from '../../../../common/static/logos/claude.svg';

// Save setup status to localStorage (global)
const saveGlobalSetupStatus = (backendId: string): void => {
  try {
    localStorage.setItem('mlflow.assistant.setupStatus.global', 'configured');
    localStorage.setItem('mlflow.assistant.selectedBackend.global', backendId);
  } catch {
    // localStorage not available
  }
};

const COMPONENT_ID_PREFIX = 'mlflow.onboarding.assistant';

/**
 * Available backend options for the assistant.
 */
interface BackendOption {
  id: string;
  name: string;
  description: string;
  logo: string;
  installCommand: string;
  authCommand: string;
  docsUrl: string;
}

const BACKEND_OPTIONS: BackendOption[] = [
  {
    id: 'claude-code',
    name: 'Claude Code',
    description: "Powered by Anthropic's Claude with advanced code analysis and reasoning capabilities.",
    logo: ClaudeLogo,
    installCommand: 'npm install -g @anthropic-ai/claude-code',
    authCommand: 'claude auth login',
    docsUrl: 'https://docs.anthropic.com/en/docs/claude-code',
  },
];

type BackendSetupStep = 'select-backend' | 'install';

interface AssistantBackendStepProps {
  /** Called when assistant is successfully configured. Receives the backend ID. */
  onConfigured?: (backendId: string) => void;
  /** Called when user skips setup. If not provided, skip button is not shown. */
  onSkip?: () => void;
}

/**
 * Configure the AI assistant backend.
 * Can be used embedded in InstrumentationStep (with onConfigured callback).
 */
export const AssistantBackendStep = ({
  onConfigured,
  onSkip,
}: AssistantBackendStepProps) => {
  const { theme } = useDesignSystemTheme();

  const [currentSubStep, setCurrentSubStep] = useState<BackendSetupStep>('select-backend');
  const [selectedBackend, setSelectedBackend] = useState<BackendOption | null>(null);
  const [isVerifying, setIsVerifying] = useState(false);
  const [verificationError, setVerificationError] = useState<string | null>(null);
  const [verificationSuccess, setVerificationSuccess] = useState(false);

  const handleBackendSelect = useCallback((backend: BackendOption) => {
    setSelectedBackend(backend);
    setCurrentSubStep('install');
  }, []);

  const handleVerify = useCallback(async () => {
    setIsVerifying(true);
    setVerificationError(null);
    setVerificationSuccess(false);

    try {
      const health = await checkHealth();
      const isAvailable = health.claude_available === 'true' || health.claude_available === 'True';

      if (isAvailable) {
        setVerificationSuccess(true);
        // Save global state immediately so button shows Claude variant everywhere
        const backendId = selectedBackend?.id || 'claude-code';
        saveGlobalSetupStatus(backendId);
        // Call the callback after showing success, passing the backend ID
        setTimeout(() => {
          onConfigured?.(backendId);
        }, 1500);
      } else {
        setVerificationError('Claude CLI is installed but not authenticated. Please run the authentication command.');
      }
    } catch {
      setVerificationError(
        'Could not connect to the assistant backend. Please ensure the CLI is installed and try again.',
      );
    } finally {
      setIsVerifying(false);
    }
  }, [onConfigured, selectedBackend]);

  const handleBack = useCallback(() => {
    setCurrentSubStep('select-backend');
    setSelectedBackend(null);
    setVerificationError(null);
    setVerificationSuccess(false);
  }, []);

  return (
    <div css={{ padding: theme.spacing.lg }}>
      {/* Sub-step 1: Select Backend */}
      {currentSubStep === 'select-backend' && (
        <div>
          <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.md }}>
            <FormattedMessage defaultMessage="Choose an assistant backend:" description="Label for backend selection" />
          </Typography.Text>

          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md, marginBottom: theme.spacing.lg }}>
            {BACKEND_OPTIONS.map((backend) => (
              <button
                key={backend.id}
                onClick={() => handleBackendSelect(backend)}
                css={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: theme.spacing.md,
                  padding: theme.spacing.lg,
                  backgroundColor: theme.colors.backgroundSecondary,
                  border: `1px solid ${theme.colors.border}`,
                  borderRadius: theme.borders.borderRadiusLg,
                  cursor: 'pointer',
                  textAlign: 'left',
                  transition: 'border-color 0.2s, background-color 0.2s',
                  '&:hover': {
                    borderColor: theme.colors.actionPrimaryBackgroundDefault,
                    backgroundColor: theme.colors.backgroundPrimary,
                  },
                }}
              >
                <img src={backend.logo} width={32} height={32} alt="" aria-hidden css={{ flexShrink: 0 }} />
                <div>
                  <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.xs }}>
                    {backend.name}
                  </Typography.Text>
                  <Typography.Text color="secondary" size="sm">
                    {backend.description}
                  </Typography.Text>
                </div>
              </button>
            ))}
          </div>

          {/* Skip button for select-backend sub-step */}
          {onSkip && (
            <div css={{ display: 'flex', justifyContent: 'flex-end' }}>
              <Button componentId={`${COMPONENT_ID_PREFIX}.skip_select`} size="small" onClick={onSkip} css={{ opacity: 0.7 }}>
                <FormattedMessage defaultMessage="Skip this step" description="Skip backend selection button" />
              </Button>
            </div>
          )}
        </div>
      )}

      {/* Sub-step 2: Installation Instructions */}
      {currentSubStep === 'install' && selectedBackend && (
        <div>
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, marginBottom: theme.spacing.lg }}>
            <img src={selectedBackend.logo} width={24} height={24} alt="" aria-hidden />
            <Typography.Text bold>{selectedBackend.name}</Typography.Text>
          </div>

          <div css={{ marginBottom: theme.spacing.lg }}>
            <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.sm }}>
              <FormattedMessage defaultMessage="Step 1: Install the CLI" description="Install step label" />
            </Typography.Text>
            <code
              css={{
                display: 'block',
                padding: theme.spacing.md,
                backgroundColor: theme.colors.backgroundSecondary,
                borderRadius: theme.borders.borderRadiusMd,
                fontFamily: 'monospace',
                fontSize: theme.typography.fontSizeSm,
                overflowX: 'auto',
              }}
            >
              {selectedBackend.installCommand}
            </code>
          </div>

          <div css={{ marginBottom: theme.spacing.lg }}>
            <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.sm }}>
              <FormattedMessage defaultMessage="Step 2: Authenticate" description="Auth step label" />
            </Typography.Text>
            <code
              css={{
                display: 'block',
                padding: theme.spacing.md,
                backgroundColor: theme.colors.backgroundSecondary,
                borderRadius: theme.borders.borderRadiusMd,
                fontFamily: 'monospace',
                fontSize: theme.typography.fontSizeSm,
                overflowX: 'auto',
              }}
            >
              {selectedBackend.authCommand}
            </code>
            <Typography.Text size="sm" color="secondary" css={{ display: 'block', marginTop: theme.spacing.sm }}>
              <FormattedMessage
                defaultMessage="This will open a browser window to authenticate with your account."
                description="Auth step help text"
              />
            </Typography.Text>
          </div>

          <div css={{ marginBottom: theme.spacing.lg }}>
            <a
              href={selectedBackend.docsUrl}
              target="_blank"
              rel="noopener noreferrer"
              css={{
                color: theme.colors.actionPrimaryBackgroundDefault,
                textDecoration: 'none',
                fontSize: theme.typography.fontSizeSm,
                '&:hover': { textDecoration: 'underline' },
              }}
            >
              <FormattedMessage defaultMessage="View full documentation" description="Link to backend documentation" />
            </a>
          </div>

          {/* Verification section */}
          <div
            css={{
              padding: theme.spacing.lg,
              backgroundColor: theme.colors.backgroundSecondary,
              borderRadius: theme.borders.borderRadiusLg,
              marginBottom: theme.spacing.lg,
            }}
          >
            <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.md }}>
              <FormattedMessage defaultMessage="Step 3: Verify Setup" description="Verify step label" />
            </Typography.Text>

            {verificationSuccess ? (
              <div
                css={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: theme.spacing.sm,
                  color: theme.colors.textValidationSuccess,
                }}
              >
                <CheckCircleIcon />
                <Typography.Text>
                  <FormattedMessage
                    defaultMessage="Setup complete! Continuing to next step..."
                    description="Setup success message"
                  />
                </Typography.Text>
              </div>
            ) : verificationError ? (
              <div css={{ marginBottom: theme.spacing.md }}>
                <div
                  css={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: theme.spacing.sm,
                    color: theme.colors.textValidationDanger,
                    marginBottom: theme.spacing.sm,
                  }}
                >
                  <DangerIcon css={{ flexShrink: 0, marginTop: 2 }} />
                  <Typography.Text>{verificationError}</Typography.Text>
                </div>
                <Button
                  componentId={`${COMPONENT_ID_PREFIX}.retry_verify`}
                  onClick={handleVerify}
                  disabled={isVerifying}
                >
                  {isVerifying ? (
                    <Spinner size="small" />
                  ) : (
                    <FormattedMessage defaultMessage="Try Again" description="Retry verification button" />
                  )}
                </Button>
              </div>
            ) : (
              <Button
                componentId={`${COMPONENT_ID_PREFIX}.verify`}
                type="primary"
                onClick={handleVerify}
                disabled={isVerifying}
              >
                {isVerifying ? (
                  <>
                    <Spinner size="small" css={{ marginRight: theme.spacing.sm }} />
                    <FormattedMessage defaultMessage="Checking..." description="Verification in progress" />
                  </>
                ) : (
                  <FormattedMessage defaultMessage="Check Setup" description="Verify setup button" />
                )}
              </Button>
            )}
          </div>

          {/* Navigation */}
          <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Button componentId={`${COMPONENT_ID_PREFIX}.back`} onClick={handleBack}>
              <FormattedMessage defaultMessage="Back" description="Back button" />
            </Button>
            {onSkip && (
              <Button componentId={`${COMPONENT_ID_PREFIX}.skip`} size="small" onClick={onSkip} css={{ opacity: 0.7 }}>
                <FormattedMessage defaultMessage="Skip this step" description="Skip button" />
              </Button>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
