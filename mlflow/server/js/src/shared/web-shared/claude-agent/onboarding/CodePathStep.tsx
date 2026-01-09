/**
 * Code path selection step for Claude Code setup.
 * Only shown when user selects Claude Code as the assistant backend.
 */

import { useCallback } from 'react';
import { Button, Input, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { useOnboarding } from '../OnboardingWizard';

const COMPONENT_ID_PREFIX = 'mlflow.onboarding.code_path';

/**
 * Step for configuring code path for Claude Code assistant.
 * Allows users to specify the path to their code for context-aware assistance.
 */
export const CodePathStep = () => {
  const { theme } = useDesignSystemTheme();
  const { state, updateState, goToNextStep } = useOnboarding();

  const handleCodePathChange = useCallback(
    (newCodePath: string) => {
      updateState({ codePath: newCodePath });
    },
    [updateState],
  );

  const handleContinue = useCallback(() => {
    goToNextStep();
  }, [goToNextStep]);

  const handleSkip = useCallback(() => {
    // Clear any entered code path and skip to next step
    updateState({ codePath: undefined });
    goToNextStep();
  }, [updateState, goToNextStep]);

  return (
    <div css={{ padding: theme.spacing.lg }}>
      <div css={{ marginBottom: theme.spacing.lg }}>
        <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.sm }}>
          <FormattedMessage defaultMessage="Code Path" description="Code path input label" />
        </Typography.Text>
        <Input
          componentId={`${COMPONENT_ID_PREFIX}.input`}
          placeholder="/path/to/your/agent.py or /path/to/your/project"
          value={state.codePath || ''}
          onChange={(e) => handleCodePathChange(e.target.value)}
        />
      </div>

      <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Button
          componentId={`${COMPONENT_ID_PREFIX}.continue`}
          type="primary"
          onClick={handleContinue}
        >
          <FormattedMessage defaultMessage="Continue" description="Continue button" />
        </Button>
        <Button
          componentId={`${COMPONENT_ID_PREFIX}.skip`}
          size="small"
          onClick={handleSkip}
          css={{ opacity: 0.7 }}
        >
          <FormattedMessage defaultMessage="Skip this step" description="Skip button" />
        </Button>
      </div>
    </div>
  );
};
