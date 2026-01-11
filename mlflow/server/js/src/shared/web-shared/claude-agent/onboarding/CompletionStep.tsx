/**
 * Step 5: Completion - Experiment setup complete! Show summary and next steps.
 */

import { Button, CheckCircleIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { useOnboarding } from '../OnboardingWizard';

const COMPONENT_ID_PREFIX = 'mlflow.onboarding.completion';

/**
 * Step 5: Completion screen with summary and next steps.
 */
export const CompletionStep = () => {
  const { theme } = useDesignSystemTheme();
  const { state, completeOnboarding } = useOnboarding();

  const isAssistantConfigured = state.assistantConfigured;
  const isChatbotUseCase = state.useCase === 'chatbot';

  return (
    <div
      css={{
        padding: theme.spacing.lg,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        textAlign: 'center',
      }}
    >
      {/* Success Icon */}
      <div
        css={{
          width: 80,
          height: 80,
          borderRadius: '50%',
          backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          marginBottom: theme.spacing.lg,
        }}
      >
        <CheckCircleIcon css={{ fontSize: 40, color: theme.colors.white }} />
      </div>

      {/* Title */}
      <Typography.Title level={2} css={{ marginBottom: theme.spacing.lg * 2 }}>
        <FormattedMessage defaultMessage="Experiment Setup Complete" description="Completion title" />
      </Typography.Title>

      {/* Next Steps */}
      <div
        css={{
          width: '100%',
          maxWidth: 600,
          padding: theme.spacing.lg,
          backgroundColor: theme.colors.backgroundSecondary,
          borderRadius: theme.borders.borderRadiusLg,
          marginBottom: theme.spacing.lg * 2,
          textAlign: 'left',
        }}
      >
        <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.md }}>
          <FormattedMessage defaultMessage="What's next:" description="Next steps header" />
        </Typography.Text>

        <ul
          css={{
            margin: 0,
            paddingLeft: theme.spacing.lg,
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.sm,
          }}
        >
          <li>
            <Typography.Text color="secondary">
              <FormattedMessage defaultMessage="Run your agent or application to generate traces" description="Next step 1" />
            </Typography.Text>
          </li>
          <li>
            <Typography.Text color="secondary">
              <FormattedMessage
                defaultMessage="View traces, judge results, and feedback in the Traces tab"
                description="Next step 2"
              />
            </Typography.Text>
          </li>
          {isChatbotUseCase && (
            <li>
              <Typography.Text color="secondary">
                <FormattedMessage
                  defaultMessage="Analyze multi-turn conversations in the Sessions tab"
                  description="Next step for chatbot - sessions"
                />
              </Typography.Text>
            </li>
          )}
          {isAssistantConfigured && (
            <li>
              <Typography.Text color="secondary">
                <FormattedMessage
                  defaultMessage="Ask the Assistant for help analyzing issues"
                  description="Next step 3"
                />
              </Typography.Text>
            </li>
          )}
        </ul>
      </div>

      {/* Action Buttons */}
      <div css={{ display: 'flex', gap: theme.spacing.md }}>
        {isAssistantConfigured ? (
          <Button componentId={`${COMPONENT_ID_PREFIX}.open_assistant`} type="primary" onClick={completeOnboarding}>
            <FormattedMessage defaultMessage="Open Assistant" description="Open assistant button" />
          </Button>
        ) : (
          <Button componentId={`${COMPONENT_ID_PREFIX}.done`} type="primary" onClick={completeOnboarding}>
            <FormattedMessage defaultMessage="Done" description="Done button" />
          </Button>
        )}
      </div>
    </div>
  );
};
