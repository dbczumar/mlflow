/**
 * Step 5: Completion - All set! Show summary and next steps.
 */

import { Button, CheckCircleIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { useOnboarding } from '../OnboardingWizard';
import { useGlobalClaude } from '../GlobalClaudeContext';

const COMPONENT_ID_PREFIX = 'mlflow.onboarding.completion';

/**
 * Step 5: Completion screen with summary and next steps.
 */
export const CompletionStep = () => {
  const { theme } = useDesignSystemTheme();
  const { state, completeOnboarding } = useOnboarding();
  const { closePanel } = useGlobalClaude();

  const enabledScorersCount = state.selectedScorers.filter((s) => s.enabled).length;

  const handleGoBack = () => {
    completeOnboarding();
    closePanel();
  };

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
      <Typography.Title level={2} css={{ marginBottom: theme.spacing.sm }}>
        <FormattedMessage defaultMessage="You're all set!" description="Completion title" />
      </Typography.Title>

      <Typography.Text color="secondary" css={{ marginBottom: theme.spacing.lg * 2 }}>
        <FormattedMessage
          defaultMessage="Your MLflow GenAI setup is complete. Here's what's configured:"
          description="Completion subtitle"
        />
      </Typography.Text>

      {/* Summary */}
      <div
        css={{
          width: '100%',
          maxWidth: 400,
          marginBottom: theme.spacing.lg * 2,
        }}
      >
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.md,
            textAlign: 'left',
          }}
        >
          {/* Assistant Configured */}
          <div
            css={{
              display: 'flex',
              alignItems: 'center',
              gap: theme.spacing.md,
              padding: theme.spacing.md,
              backgroundColor: theme.colors.backgroundSecondary,
              borderRadius: theme.borders.borderRadiusMd,
            }}
          >
            <CheckCircleIcon css={{ color: theme.colors.textValidationSuccess, flexShrink: 0 }} />
            <Typography.Text>
              <FormattedMessage
                defaultMessage="AI Assistant configured"
                description="Assistant configured summary item"
              />
            </Typography.Text>
          </div>

          {/* Tracing Set Up */}
          <div
            css={{
              display: 'flex',
              alignItems: 'center',
              gap: theme.spacing.md,
              padding: theme.spacing.md,
              backgroundColor: theme.colors.backgroundSecondary,
              borderRadius: theme.borders.borderRadiusMd,
            }}
          >
            <CheckCircleIcon css={{ color: theme.colors.textValidationSuccess, flexShrink: 0 }} />
            <Typography.Text>
              <FormattedMessage
                defaultMessage="Tracing instrumentation ready"
                description="Tracing configured summary item"
              />
            </Typography.Text>
          </div>

          {/* Online Scoring Enabled */}
          <div
            css={{
              display: 'flex',
              alignItems: 'center',
              gap: theme.spacing.md,
              padding: theme.spacing.md,
              backgroundColor: theme.colors.backgroundSecondary,
              borderRadius: theme.borders.borderRadiusMd,
            }}
          >
            <CheckCircleIcon css={{ color: theme.colors.textValidationSuccess, flexShrink: 0 }} />
            <Typography.Text>
              <FormattedMessage
                defaultMessage="{count} scorers running online"
                description="Scorers configured summary item"
                values={{ count: enabledScorersCount }}
              />
            </Typography.Text>
          </div>
        </div>
      </div>

      {/* Next Steps */}
      <div
        css={{
          width: '100%',
          maxWidth: 400,
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
              <FormattedMessage defaultMessage="Run your app to generate traces" description="Next step 1" />
            </Typography.Text>
          </li>
          <li>
            <Typography.Text color="secondary">
              <FormattedMessage defaultMessage="View traces and scores in the Traces tab" description="Next step 2" />
            </Typography.Text>
          </li>
          <li>
            <Typography.Text color="secondary">
              <FormattedMessage
                defaultMessage="Ask the Assistant for help analyzing issues"
                description="Next step 3"
              />
            </Typography.Text>
          </li>
        </ul>
      </div>

      {/* Action Buttons */}
      <div css={{ display: 'flex', gap: theme.spacing.md }}>
        <Button componentId={`${COMPONENT_ID_PREFIX}.go_back`} onClick={handleGoBack}>
          <FormattedMessage defaultMessage="Go Back" description="Go back button" />
        </Button>
        <Button componentId={`${COMPONENT_ID_PREFIX}.open_assistant`} type="primary" onClick={completeOnboarding}>
          <FormattedMessage defaultMessage="Open Assistant" description="Open assistant button" />
        </Button>
      </div>
    </div>
  );
};
