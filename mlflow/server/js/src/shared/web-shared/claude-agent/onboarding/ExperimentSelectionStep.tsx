/**
 * Step: Select or create an experiment (only if not already in one).
 * Guides users to either select an existing experiment from the experiments page,
 * or create a new one via the CreateExperimentModal.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Button,
  CheckCircleIcon,
  ChevronRightIcon,
  FolderIcon,
  PlusCircleIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { useNavigate } from '../../../../common/utils/RoutingUtils';

import { useOnboarding } from '../OnboardingWizard';
import { useGlobalClaudeOptional } from '../GlobalClaudeContext';
import { CreateExperimentModal } from '../../../../experiment-tracking/components/modals/CreateExperimentModal';
import Routes from '../../../../experiment-tracking/routes';
import { searchTracesV4 } from '../../model-trace-explorer/api';

const COMPONENT_ID_PREFIX = 'mlflow.onboarding.experiment';

/**
 * Step: Select or create an experiment.
 * Shown only if user is not already in an experiment context.
 */
export const ExperimentSelectionStep = () => {
  const { theme } = useDesignSystemTheme();
  const { goToNextStep, updateState } = useOnboarding();
  const globalClaude = useGlobalClaudeOptional();
  const navigate = useNavigate();

  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [experimentSelected, setExperimentSelected] = useState(false);
  const [hasTraces, setHasTraces] = useState<boolean | null>(null);
  const [checkingTraces, setCheckingTraces] = useState(false);
  const hasAutoAdvancedRef = useRef(false);

  // Check if user is already in an experiment
  const currentExperimentId = globalClaude?.context?.navigation?.experimentId;
  const currentExperimentName = globalClaude?.context?.navigation?.experimentName;

  // Check if current experiment has traces
  useEffect(() => {
    if (!currentExperimentId || checkingTraces || hasTraces !== null) {
      return;
    }

    const checkForTraces = async () => {
      setCheckingTraces(true);
      try {
        const traces = await searchTracesV4({
          locations: [
            {
              type: 'MLFLOW_EXPERIMENT',
              mlflow_experiment: { experiment_id: currentExperimentId },
            },
          ],
        });
        setHasTraces(traces.length > 0);
      } catch {
        setHasTraces(false);
      } finally {
        setCheckingTraces(false);
      }
    };

    checkForTraces();
  }, [currentExperimentId, checkingTraces, hasTraces]);

  // Auto-advance to next step ONLY if experiment has traces
  useEffect(() => {
    if (!currentExperimentId || hasAutoAdvancedRef.current) {
      return;
    }

    // Still checking for traces - wait for result
    if (hasTraces === null) {
      return;
    }

    // Only auto-advance if experiment has traces
    if (hasTraces === true) {
      hasAutoAdvancedRef.current = true;
      updateState({ experimentSelected: true });
      // Small delay to show the experiment was detected
      setTimeout(() => {
        goToNextStep();
      }, 800);
    }
  }, [currentExperimentId, hasTraces, goToNextStep, updateState]);

  const handleSelectExisting = useCallback(() => {
    // Navigate to experiments page so user can select one
    navigate(Routes.experimentsObservatoryRoute);
    setExperimentSelected(true);
  }, [navigate]);

  const handleCreateNew = useCallback(() => {
    setIsCreateModalOpen(true);
  }, []);

  const handleCloseCreateModal = useCallback(() => {
    setIsCreateModalOpen(false);
  }, []);

  const handleExperimentCreated = useCallback(() => {
    setIsCreateModalOpen(false);
    setExperimentSelected(true);
    // Modal automatically navigates to the new experiment
  }, []);

  const handleContinue = useCallback(() => {
    updateState({ experimentSelected: true });
    goToNextStep();
  }, [goToNextStep, updateState]);

  return (
    <div css={{ padding: theme.spacing.lg }}>
      {/* If already in an experiment with traces, show detection message (will auto-advance) */}
      {currentExperimentId && hasTraces === true && (
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            gap: theme.spacing.sm,
            padding: theme.spacing.md,
            backgroundColor: theme.colors.backgroundSecondary,
            borderRadius: theme.borders.borderRadiusMd,
            color: theme.colors.textValidationSuccess,
          }}
        >
          <CheckCircleIcon />
          <div>
            <Typography.Text bold>
              <FormattedMessage
                defaultMessage="Experiment with traces detected! Advancing..."
                description="Message when experiment with traces is detected"
              />
            </Typography.Text>
            <Typography.Text color="secondary" size="sm" css={{ display: 'block' }}>
              {currentExperimentName || currentExperimentId}
            </Typography.Text>
          </div>
        </div>
      )}

      {/* If already in an experiment without traces, show confirmation and allow continue */}
      {currentExperimentId && hasTraces === false && (
        <div>
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
            <div>
              <Typography.Text bold>
                <FormattedMessage
                  defaultMessage="Experiment Selected"
                  description="Confirmation that user is in an experiment"
                />
              </Typography.Text>
              <Typography.Text color="secondary" size="sm" css={{ display: 'block' }}>
                {currentExperimentName || currentExperimentId}
              </Typography.Text>
            </div>
          </div>

          <Button componentId={`${COMPONENT_ID_PREFIX}.continue`} type="primary" onClick={handleContinue}>
            <FormattedMessage defaultMessage="Continue" description="Continue button" />
          </Button>
        </div>
      )}

      {/* If checking for traces, show loading state */}
      {currentExperimentId && hasTraces === null && (
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            gap: theme.spacing.sm,
            padding: theme.spacing.md,
            backgroundColor: theme.colors.backgroundSecondary,
            borderRadius: theme.borders.borderRadiusMd,
          }}
        >
          <Typography.Text>
            <FormattedMessage
              defaultMessage="Checking for traces in {experimentName}..."
              description="Checking for traces message"
              values={{ experimentName: currentExperimentName || currentExperimentId }}
            />
          </Typography.Text>
        </div>
      )}

      {/* If not in an experiment, show selection options */}
      {!currentExperimentId && (
        <div>
          <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.md }}>
            <FormattedMessage
              defaultMessage="Choose how to set up your experiment:"
              description="Label for experiment selection method"
            />
          </Typography.Text>

          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
            {/* Option A: Select Existing Experiment */}
            <button
              onClick={handleSelectExisting}
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
                <FolderIcon />
              </div>
              <div css={{ flex: 1 }}>
                <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.xs }}>
                  <FormattedMessage
                    defaultMessage="Select Existing Experiment"
                    description="Option to select existing experiment"
                  />
                </Typography.Text>
                <Typography.Text color="secondary" size="sm">
                  <FormattedMessage
                    defaultMessage="Choose from your existing experiments to track your traces."
                    description="Description for select existing experiment option"
                  />
                </Typography.Text>
              </div>
              <ChevronRightIcon css={{ color: theme.colors.textSecondary, flexShrink: 0 }} />
            </button>

            {/* Option B: Create New Experiment */}
            <button
              onClick={handleCreateNew}
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
                  color: theme.colors.actionPrimaryTextDefault,
                }}
              >
                <PlusCircleIcon />
              </div>
              <div css={{ flex: 1 }}>
                <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
                  <Typography.Text bold>
                    <FormattedMessage
                      defaultMessage="Create New Experiment"
                      description="Option to create new experiment"
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
                    defaultMessage="Create a new experiment to organize your GenAI traces."
                    description="Description for create new experiment option"
                  />
                </Typography.Text>
              </div>
              <ChevronRightIcon css={{ color: theme.colors.textSecondary, flexShrink: 0 }} />
            </button>
          </div>

          {/* Show message if user selected to navigate to experiments page */}
          {experimentSelected && !currentExperimentId && (
            <div
              css={{
                marginTop: theme.spacing.lg,
                padding: theme.spacing.md,
                backgroundColor: theme.colors.tagTurquoise,
                borderRadius: theme.borders.borderRadiusMd,
              }}
            >
              <Typography.Text size="sm" bold css={{ display: 'block', marginBottom: theme.spacing.xs }}>
                <FormattedMessage
                  defaultMessage="Select an experiment to continue"
                  description="Message after navigating to experiments page"
                />
              </Typography.Text>
              <Typography.Text size="sm">
                <FormattedMessage
                  defaultMessage="Once you've selected an experiment from the list, the wizard will automatically advance to the next step."
                  description="Instructions for selecting experiment"
                />
              </Typography.Text>
            </div>
          )}

          {/* Show detecting message when experiment is detected */}
          {experimentSelected && currentExperimentId && (
            <div
              css={{
                marginTop: theme.spacing.lg,
                display: 'flex',
                alignItems: 'center',
                gap: theme.spacing.sm,
                padding: theme.spacing.md,
                backgroundColor: theme.colors.backgroundSecondary,
                borderRadius: theme.borders.borderRadiusMd,
                color: theme.colors.textValidationSuccess,
              }}
            >
              <CheckCircleIcon />
              <div>
                <Typography.Text bold>
                  <FormattedMessage
                    defaultMessage="Experiment detected! Advancing to next step..."
                    description="Message when experiment is detected"
                  />
                </Typography.Text>
                <Typography.Text color="secondary" size="sm" css={{ display: 'block' }}>
                  {currentExperimentName || currentExperimentId}
                </Typography.Text>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Create Experiment Modal */}
      <CreateExperimentModal
        isOpen={isCreateModalOpen}
        onClose={handleCloseCreateModal}
        onExperimentCreated={handleExperimentCreated}
      />
    </div>
  );
};
