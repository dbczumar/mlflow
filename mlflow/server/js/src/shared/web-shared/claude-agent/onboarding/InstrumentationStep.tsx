/**
 * Step 4: Instrument your application with MLflow tracing.
 * Two options:
 * - Let the assistant instrument directly (sends task to Claude) - requires assistant setup
 * - Copy instructions for Claude Code CLI
 */

import { useCallback, useState } from 'react';
import {
  Alert,
  Button,
  CheckCircleIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  CopyIcon,
  Input,
  Spinner,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { useOnboarding } from '../OnboardingWizard';
import { useGlobalClaudeOptional } from '../GlobalClaudeContext';
import { AssistantBackendStep } from './AssistantBackendStep';
import { CopyButton } from '../../../building_blocks/CopyButton';

const COMPONENT_ID_PREFIX = 'mlflow.onboarding.instrumentation';

type InstrumentationMethod = 'assistant-direct' | 'copy-instructions' | 'manual';

/**
 * Generate the Claude prompt for instrumenting an application.
 */
const generateInstrumentationPrompt = (codePath: string, trackingUri: string, experimentName: string): string => {
  return `Add MLflow tracing to my code at: ${codePath}. Set tracking URI to ${trackingUri} and experiment to "${experimentName}".`;
};

/**
 * Step 4: Add tracing to the user's application.
 */
export const InstrumentationStep = () => {
  const { theme } = useDesignSystemTheme();
  const { goToNextStep, updateState, state } = useOnboarding();
  const globalClaude = useGlobalClaudeOptional();

  const [selectedMethod, setSelectedMethod] = useState<InstrumentationMethod | null>(
    state.instrumentationMethod || null,
  );
  const [codePath, setCodePath] = useState(state.codePath || '');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisComplete, setAnalysisComplete] = useState(state.instrumentationApplied);
  const [showAssistantSetup, setShowAssistantSetup] = useState(false);

  // Get tracking URI and experiment name from context
  // In dev mode (port 3000), use the actual MLflow backend port (5000)
  const trackingUri = window.location.port === '3000'
    ? `${window.location.protocol}//${window.location.hostname}:5000`
    : window.location.origin;
  const experimentName = globalClaude?.context?.navigation?.experimentName || 'Default';

  // Check if assistant is configured
  const isAssistantConfigured = state.assistantConfigured || globalClaude?.isClaudeAvailable;

  const handleMethodSelect = useCallback(
    (method: InstrumentationMethod) => {
      // If user wants assistant to do it but assistant is not configured, show setup
      if (method === 'assistant-direct' && !isAssistantConfigured) {
        setShowAssistantSetup(true);
        return;
      }

      // For assistant-direct, immediately send message and continue
      if (method === 'assistant-direct' && globalClaude && state.codePath) {
        const prompt = generateInstrumentationPrompt(state.codePath, trackingUri, experimentName);

        globalClaude.sendMessage(prompt);

        updateState({
          instrumentationMethod: method,
          instrumentationApplied: true,
        });

        // Continue to next step
        goToNextStep();
        return;
      }

      setSelectedMethod(method);
      updateState({ instrumentationMethod: method });
    },
    [isAssistantConfigured, updateState, globalClaude, state.codePath, trackingUri, experimentName, goToNextStep],
  );

  const handleAssistantConfigured = useCallback((backendId: string) => {
    updateState({ assistantConfigured: true, selectedBackend: backendId });
    setShowAssistantSetup(false);
    setSelectedMethod('assistant-direct');
    updateState({ instrumentationMethod: 'assistant-direct' });
  }, [updateState]);

  const handleBackFromAssistantSetup = useCallback(() => {
    setShowAssistantSetup(false);
  }, []);

  const handleAnalyzeAndInstrument = useCallback(async () => {
    if (!codePath.trim()) return;

    setIsAnalyzing(true);

    // Send the instrumentation request to Claude via the assistant
    if (globalClaude) {
      const prompt = generateInstrumentationPrompt(codePath, trackingUri, experimentName);

      globalClaude.sendMessage(prompt);
    }

    // Simulate analysis completion (in real implementation, we'd wait for Claude's response)
    setTimeout(() => {
      setIsAnalyzing(false);
      setAnalysisComplete(true);
      updateState({
        codePath,
        instrumentationApplied: true,
      });
    }, 2000);
  }, [codePath, experimentName, globalClaude, trackingUri, updateState]);

  const handleSkipToNext = useCallback(() => {
    updateState({ instrumentationMethod: selectedMethod });
    goToNextStep();
  }, [goToNextStep, selectedMethod, updateState]);

  const handleContinue = useCallback(() => {
    updateState({
      instrumentationMethod: selectedMethod,
      codePath,
      instrumentationApplied: analysisComplete,
    });
    goToNextStep();
  }, [analysisComplete, codePath, goToNextStep, selectedMethod, updateState]);

  return (
    <div css={{ padding: theme.spacing.lg }}>
      {/* Show assistant setup when user wants to use assistant but it's not configured */}
      {showAssistantSetup && (
        <div>
          <Alert
            type="info"
            closable={false}
            componentId={`${COMPONENT_ID_PREFIX}.assistant_required`}
            css={{ marginBottom: theme.spacing.lg }}
            message={
              <FormattedMessage
                defaultMessage="To let the assistant instrument your code, you need to set up the AI assistant first."
                description="Message explaining assistant setup is needed"
              />
            }
          />

          <AssistantBackendStep
            onConfigured={handleAssistantConfigured}
          />

          <Button
            componentId={`${COMPONENT_ID_PREFIX}.back_from_assistant`}
            onClick={handleBackFromAssistantSetup}
            icon={<ChevronLeftIcon />}
            css={{ marginTop: theme.spacing.lg }}
          >
            <FormattedMessage defaultMessage="Back to options" description="Back button" />
          </Button>
        </div>
      )}

      {/* Method selection */}
      {!selectedMethod && !showAssistantSetup && (
        <div>
          <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.md }}>
            <FormattedMessage
              defaultMessage="How would you like to add tracing?"
              description="Label for instrumentation method selection"
            />
          </Typography.Text>

          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
            {/* Option A: Let Assistant Do It - only show if code path was configured */}
            {isAssistantConfigured && state.codePath && (
              <button
                onClick={() => handleMethodSelect('assistant-direct')}
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
                <div
                  css={{
                    width: 40,
                    height: 40,
                    borderRadius: theme.borders.borderRadiusMd,
                    backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    flexShrink: 0,
                    fontSize: 20,
                  }}
                >
                  <span role="img" aria-hidden>
                    ✨
                  </span>
                </div>
                <div css={{ flex: 1 }}>
                  <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
                    <Typography.Text bold>
                      <FormattedMessage
                        defaultMessage="Automatically Add Tracing"
                        description="Option to let assistant instrument code"
                      />
                    </Typography.Text>
                    <span
                      css={{
                        fontSize: theme.typography.fontSizeSm,
                        padding: `2px ${theme.spacing.xs}px`,
                        backgroundColor: theme.colors.tagTurquoise,
                        borderRadius: theme.borders.borderRadiusSm,
                        color: theme.colors.textPrimary,
                      }}
                    >
                      <FormattedMessage defaultMessage="Recommended" description="Recommended label" />
                    </span>
                  </div>
                  <Typography.Text color="secondary" size="sm">
                    <FormattedMessage
                      defaultMessage="The assistant will automatically analyze your code and add MLflow tracing."
                      description="Description for assistant instrumentation"
                    />
                  </Typography.Text>
                </div>
                <ChevronRightIcon css={{ color: theme.colors.textSecondary, flexShrink: 0 }} />
              </button>
            )}

            {/* Option B: Copy Instructions - only show if Claude Code is the selected backend */}
            {state.selectedBackend === 'claude-code' && (
              <button
              onClick={() => handleMethodSelect('copy-instructions')}
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
              <div
                css={{
                  width: 40,
                  height: 40,
                  borderRadius: theme.borders.borderRadiusMd,
                  backgroundColor: theme.colors.backgroundSecondary,
                  border: `1px solid ${theme.colors.border}`,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flexShrink: 0,
                }}
              >
                <CopyIcon />
              </div>
              <div css={{ flex: 1 }}>
                <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.xs }}>
                  <FormattedMessage
                    defaultMessage="Get Instructions for Claude Code"
                    description="Option to copy instructions"
                  />
                </Typography.Text>
                <Typography.Text color="secondary" size="sm">
                  <FormattedMessage
                    defaultMessage="Copy instructions to use with Claude Code CLI in your terminal."
                    description="Description for copy instructions option"
                  />
                </Typography.Text>
              </div>
              <ChevronRightIcon css={{ color: theme.colors.textSecondary, flexShrink: 0 }} />
            </button>
            )}

            {/* Option C: Manual / Read the Docs - always available */}
            <button
              onClick={() => handleMethodSelect('manual')}
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
              <div
                css={{
                  width: 40,
                  height: 40,
                  borderRadius: theme.borders.borderRadiusMd,
                  backgroundColor: theme.colors.backgroundSecondary,
                  border: `1px solid ${theme.colors.border}`,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flexShrink: 0,
                }}
              >
                📖
              </div>
              <div css={{ flex: 1 }}>
                <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.xs }}>
                  <FormattedMessage
                    defaultMessage="Set Up Manually"
                    description="Option to manually set up tracing"
                  />
                </Typography.Text>
                <Typography.Text color="secondary" size="sm">
                  <FormattedMessage
                    defaultMessage="Follow the MLflow documentation to manually add MLflow tracing to your application."
                    description="Description for manual setup option"
                  />
                </Typography.Text>
              </div>
              <ChevronRightIcon css={{ color: theme.colors.textSecondary, flexShrink: 0 }} />
            </button>
          </div>

          {/* Skip button */}
          <div css={{ display: 'flex', justifyContent: 'flex-end', marginTop: theme.spacing.lg }}>
            <Button
              componentId={`${COMPONENT_ID_PREFIX}.skip_method_selection`}
              size="small"
              onClick={handleSkipToNext}
              css={{ opacity: 0.7 }}
            >
              <FormattedMessage defaultMessage="Skip this step" description="Skip button" />
            </Button>
          </div>
        </div>
      )}

      {/* Assistant Direct Method */}
      {selectedMethod === 'assistant-direct' && (
        <div>
          {analysisComplete ? (
            <div
              css={{
                display: 'flex',
                alignItems: 'center',
                gap: theme.spacing.sm,
                padding: theme.spacing.md,
                backgroundColor: theme.colors.backgroundSecondary,
                borderRadius: theme.borders.borderRadiusMd,
                marginBottom: theme.spacing.lg,
                color: theme.colors.textValidationSuccess,
              }}
            >
              <CheckCircleIcon />
              <Typography.Text>
                <FormattedMessage
                  defaultMessage="Instrumentation request sent to assistant. Check the assistant panel for progress."
                  description="Success message after sending instrumentation request"
                />
              </Typography.Text>
            </div>
          ) : (
            <Button
              componentId={`${COMPONENT_ID_PREFIX}.analyze`}
              type="primary"
              onClick={handleAnalyzeAndInstrument}
              disabled={!codePath.trim() || isAnalyzing}
              css={{ marginBottom: theme.spacing.lg }}
            >
              {isAnalyzing ? (
                <>
                  <Spinner size="small" css={{ marginRight: theme.spacing.sm }} />
                  <FormattedMessage defaultMessage="Sending to Assistant..." description="Analyzing button text" />
                </>
              ) : (
                <FormattedMessage defaultMessage="Analyze & Instrument" description="Analyze button text" />
              )}
            </Button>
          )}

          <div css={{ display: 'flex', justifyContent: 'space-between' }}>
            <Button componentId={`${COMPONENT_ID_PREFIX}.back`} onClick={() => setSelectedMethod(null)}>
              <FormattedMessage defaultMessage="Back" description="Back button" />
            </Button>
            <Button componentId={`${COMPONENT_ID_PREFIX}.continue`} type="primary" onClick={handleContinue}>
              <FormattedMessage defaultMessage="Continue" description="Continue button" />
            </Button>
          </div>
        </div>
      )}

      {/* Copy Instructions Method */}
      {selectedMethod === 'copy-instructions' && (
        <div>
          <div
            css={{
              padding: theme.spacing.lg,
              backgroundColor: theme.colors.backgroundSecondary,
              borderRadius: theme.borders.borderRadiusLg,
              marginBottom: theme.spacing.lg,
            }}
          >
            <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.md }}>
              <FormattedMessage
                defaultMessage="Run this command in your project directory:"
                description="Instructions header"
              />
            </Typography.Text>

            <div css={{ position: 'relative' }}>
              <div
                css={{
                  position: 'absolute',
                  top: theme.spacing.sm,
                  right: theme.spacing.sm,
                  zIndex: 1,
                }}
              >
                <CopyButton
                  showLabel={false}
                  icon={<CopyIcon />}
                  copyText={`claude "Add MLflow tracing to my GenAI app. Set tracking URI to ${trackingUri} and experiment to '${experimentName}'. Enable autologging for any supported frameworks."`}
                />
              </div>
              <code
                css={{
                  display: 'block',
                  padding: theme.spacing.md,
                  paddingRight: theme.spacing.lg * 3,
                  backgroundColor: theme.colors.backgroundPrimary,
                  borderRadius: theme.borders.borderRadiusMd,
                  border: `1px solid ${theme.colors.border}`,
                  fontFamily: 'monospace',
                  fontSize: theme.typography.fontSizeSm,
                  overflowX: 'auto',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                }}
              >
                {`claude "Add MLflow tracing to my GenAI app. Set tracking URI to ${trackingUri} and experiment to '${experimentName}'. Enable autologging for any supported frameworks."`}
              </code>
            </div>
          </div>

          <div css={{ display: 'flex', justifyContent: 'space-between' }}>
            <Button componentId={`${COMPONENT_ID_PREFIX}.back_copy`} onClick={() => setSelectedMethod(null)}>
              <FormattedMessage defaultMessage="Back" description="Back button" />
            </Button>
            <Button componentId={`${COMPONENT_ID_PREFIX}.skip`} type="primary" onClick={handleSkipToNext}>
              <FormattedMessage defaultMessage="Continue" description="Continue button" />
            </Button>
          </div>
        </div>
      )}

      {/* Manual / Docs Method */}
      {selectedMethod === 'manual' && (
        <div>
          <div
            css={{
              padding: theme.spacing.lg,
              backgroundColor: theme.colors.backgroundSecondary,
              borderRadius: theme.borders.borderRadiusLg,
              marginBottom: theme.spacing.lg,
            }}
          >
            <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.md }}>
              <FormattedMessage
                defaultMessage="Follow the MLflow Tracing documentation"
                description="Manual setup heading"
              />
            </Typography.Text>

            <Typography.Text css={{ display: 'block', marginBottom: theme.spacing.md }}>
              <FormattedMessage
                defaultMessage="Our documentation provides step-by-step instructions for adding MLflow tracing to your application:"
                description="Manual setup description"
              />
            </Typography.Text>

            <ul css={{ marginBottom: theme.spacing.md, paddingLeft: theme.spacing.lg }}>
              <li css={{ marginBottom: theme.spacing.sm }}>
                <Typography.Text>
                  <a
                    href="https://mlflow.org/docs/latest/llms/tracing/index.html"
                    target="_blank"
                    rel="noopener noreferrer"
                    css={{
                      color: theme.colors.actionPrimaryBackgroundDefault,
                      textDecoration: 'underline',
                      '&:hover': { opacity: 0.8 },
                    }}
                  >
                    <FormattedMessage
                      defaultMessage="MLflow Tracing Overview"
                      description="Link to tracing overview docs"
                    />
                  </a>
                </Typography.Text>
              </li>
              <li css={{ marginBottom: theme.spacing.sm }}>
                <Typography.Text>
                  <a
                    href="https://mlflow.org/docs/latest/llms/tracing/index.html#automatic-tracing"
                    target="_blank"
                    rel="noopener noreferrer"
                    css={{
                      color: theme.colors.actionPrimaryBackgroundDefault,
                      textDecoration: 'underline',
                      '&:hover': { opacity: 0.8 },
                    }}
                  >
                    <FormattedMessage
                      defaultMessage="Automatic Tracing (OpenAI, Anthropic, LangChain, etc.)"
                      description="Link to autologging docs"
                    />
                  </a>
                </Typography.Text>
              </li>
              <li>
                <Typography.Text>
                  <a
                    href="https://mlflow.org/docs/latest/llms/tracing/index.html#manual-tracing"
                    target="_blank"
                    rel="noopener noreferrer"
                    css={{
                      color: theme.colors.actionPrimaryBackgroundDefault,
                      textDecoration: 'underline',
                      '&:hover': { opacity: 0.8 },
                    }}
                  >
                    <FormattedMessage
                      defaultMessage="Manual Tracing with @mlflow.trace decorator"
                      description="Link to manual tracing docs"
                    />
                  </a>
                </Typography.Text>
              </li>
            </ul>

            <Alert
              type="info"
              closable={false}
              componentId={`${COMPONENT_ID_PREFIX}.manual_info`}
              message={
                <FormattedMessage
                  defaultMessage="Once you've added tracing to your code and run your application, traces will appear in this experiment automatically."
                  description="Info about traces appearing after setup"
                />
              }
            />
          </div>

          <div css={{ display: 'flex', justifyContent: 'space-between' }}>
            <Button componentId={`${COMPONENT_ID_PREFIX}.back_manual`} onClick={() => setSelectedMethod(null)}>
              <FormattedMessage defaultMessage="Back" description="Back button" />
            </Button>
            <Button componentId={`${COMPONENT_ID_PREFIX}.continue_manual`} type="primary" onClick={handleSkipToNext}>
              <FormattedMessage defaultMessage="Continue" description="Continue button" />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
};
