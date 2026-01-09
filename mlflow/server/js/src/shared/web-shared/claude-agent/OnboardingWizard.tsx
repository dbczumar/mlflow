/**
 * Multi-step onboarding wizard for MLflow GenAI.
 * Guides users through:
 * 1. Selecting or creating an experiment
 * 2. Defining their use case for judge recommendations
 * 3. Configuring LLM judges for evaluation
 * 4. Instrumenting their application with tracing
 */

import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from 'react';
import {
  Button,
  ChevronLeftIcon,
  ChevronRightIcon,
  Typography,
  useDesignSystemTheme,
  importantify,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { ExperimentSelectionStep } from './onboarding/ExperimentSelectionStep';
import { UseCaseStep } from './onboarding/UseCaseStep';
import { ScorerSelectionStep } from './onboarding/ScorerSelectionStep';
import { AssistantBackendStep } from './onboarding/AssistantBackendStep';
import { InstrumentationStep } from './onboarding/InstrumentationStep';
import { CompletionStep } from './onboarding/CompletionStep';
import { listScheduledScorers } from '../../../experiment-tracking/pages/experiment-scorers/api';
import { searchTracesV4 } from '../model-trace-explorer/api';

const COMPONENT_ID_PREFIX = 'mlflow.onboarding';

/**
 * Onboarding wizard steps.
 */
export type OnboardingStep =
  | 'experiment-selection'
  | 'use-case'
  | 'scorer-selection'
  | 'assistant-backend'
  | 'instrumentation'
  | 'completion';

/**
 * Use case types for scorer recommendations.
 */
export type UseCaseType =
  | 'chatbot'
  | 'rag'
  | 'summarization'
  | 'extraction'
  | 'content'
  | 'code'
  | 'classification'
  | 'other';

/**
 * Scorer configuration.
 */
export interface ScorerConfig {
  id: string;
  type: 'builtin' | 'guidelines';
  name: string;
  description: string;
  builtinType?: string;
  guidelines?: string;
  enabled: boolean;
}

/**
 * Onboarding state shared across all steps.
 */
export interface OnboardingState {
  // Step 1: Experiment Selection
  experimentSelected: boolean;

  // Step 2: Use Case
  useCase: UseCaseType | null;
  useCaseDescription?: string;

  // Step 3: Scorer Selection
  selectedScorers: ScorerConfig[];
  samplingMode: 'all' | 'sample';
  samplingRate: number;
  onlineScoringEnabled: boolean;
  judgeEndpointName?: string;

  // Step 4: Instrumentation
  instrumentationMethod: 'assistant-direct' | 'copy-instructions' | 'manual' | null;
  codePath?: string;
  detectedFrameworks?: string[];
  instrumentationApplied: boolean;
  tracingVerified: boolean;

  // Assistant configuration (used conditionally in instrumentation step)
  assistantConfigured: boolean;

  // Judge configuration status (used to skip scorer selection if already configured)
  judgesConfigured: boolean;

  // Completion
  completedAt: Date | null;

  // Track which steps have been visited (for enabling forward navigation)
  visitedSteps: OnboardingStep[];
}

/**
 * Context for sharing onboarding state and actions.
 */
interface OnboardingContextType {
  state: OnboardingState;
  currentStep: OnboardingStep;
  isCheckingInitialStep: boolean;
  currentExperimentId?: string;
  goToStep: (step: OnboardingStep) => void;
  goToNextStep: () => void;
  goToPreviousStep: () => void;
  updateState: (updates: Partial<OnboardingState>) => void;
  completeOnboarding: () => void;
}

const OnboardingContext = createContext<OnboardingContextType | null>(null);

/**
 * Hook to access onboarding context.
 */
export const useOnboarding = (): OnboardingContextType => {
  const context = useContext(OnboardingContext);
  if (!context) {
    throw new Error('useOnboarding must be used within OnboardingWizard');
  }
  return context;
};

/**
 * Step order for navigation.
 */
const STEP_ORDER: OnboardingStep[] = [
  'experiment-selection',
  'use-case',
  'scorer-selection',
  'assistant-backend',
  'instrumentation',
  'completion',
];

/**
 * Step metadata for display.
 */
const STEP_INFO: Record<OnboardingStep, { title: string; subtitle: string }> = {
  'experiment-selection': {
    title: 'Select Experiment',
    subtitle: 'Select an experiment to build an agent or model',
  },
  'use-case': {
    title: 'Select Use Case',
    subtitle: 'Tell us about your agent or application, so that MLflow can help fix issues and make improvements',
  },
  'scorer-selection': {
    title: 'Configure Judges',
    subtitle: 'Select LLM judges that will automatically identify issues in your agent or application',
  },
  'assistant-backend': {
    title: 'Configure Assistant',
    subtitle: 'Set up the AI assistant backend to help with tracing.',
  },
  instrumentation: {
    title: 'Add Tracing',
    subtitle: 'Instrument your GenAI application to capture traces.',
  },
  completion: {
    title: 'All Set!',
    subtitle: "You're ready to start monitoring your GenAI application.",
  },
};

/**
 * Initial onboarding state.
 */
const INITIAL_STATE: OnboardingState = {
  experimentSelected: false,
  useCase: null,
  selectedScorers: [],
  samplingMode: 'all',
  samplingRate: 25,
  onlineScoringEnabled: false,
  instrumentationMethod: null,
  instrumentationApplied: false,
  tracingVerified: false,
  assistantConfigured: false,
  judgesConfigured: false,
  completedAt: null,
  visitedSteps: [],
};

/**
 * Get localStorage key for wizard step.
 */
const getWizardStepKey = (experimentId?: string): string => {
  if (experimentId) {
    return `mlflow.assistant.wizardStep.experiment.${experimentId}`;
  }
  return 'mlflow.assistant.wizardStep.global';
};

/**
 * Load saved wizard step from localStorage.
 */
const loadWizardStep = (experimentId?: string): OnboardingStep | null => {
  try {
    const key = getWizardStepKey(experimentId);
    const stored = localStorage.getItem(key);
    if (stored && STEP_ORDER.includes(stored as OnboardingStep)) {
      return stored as OnboardingStep;
    }
  } catch {
    // localStorage not available
  }
  return null;
};

/**
 * Save wizard step to localStorage.
 */
const saveWizardStep = (step: OnboardingStep, experimentId?: string): void => {
  try {
    const key = getWizardStepKey(experimentId);
    localStorage.setItem(key, step);
  } catch {
    // localStorage not available
  }
};

/**
 * Get localStorage key for wizard state.
 */
const getWizardStateKey = (experimentId?: string): string => {
  if (experimentId) {
    return `mlflow.assistant.wizardState.experiment.${experimentId}`;
  }
  return 'mlflow.assistant.wizardState.global';
};

/**
 * Load saved wizard state from localStorage.
 */
const loadWizardState = (experimentId?: string): Partial<OnboardingState> | null => {
  try {
    const key = getWizardStateKey(experimentId);
    const stored = localStorage.getItem(key);
    if (stored) {
      return JSON.parse(stored);
    }
  } catch {
    // localStorage not available or invalid JSON
  }
  return null;
};

/**
 * Save wizard state to localStorage.
 */
const saveWizardState = (state: OnboardingState, experimentId?: string): void => {
  try {
    const key = getWizardStateKey(experimentId);
    localStorage.setItem(key, JSON.stringify(state));
  } catch {
    // localStorage not available
  }
};

interface OnboardingWizardProps {
  /** Called when onboarding is complete */
  onComplete: () => void;
  /** Current experiment ID if viewing an experiment */
  currentExperimentId?: string;
  /** Whether assistant is already configured (for instrumentation step) */
  assistantAlreadyConfigured?: boolean;
}

/**
 * Determine the appropriate initial step based on current context.
 * Returns step and whether judges are configured.
 */
const determineInitialStep = async (
  experimentId?: string,
  assistantConfigured?: boolean,
): Promise<{ step: OnboardingStep; judgesConfigured: boolean }> => {
  // If no experiment, start with assistant-backend (for home page setup)
  if (!experimentId) {
    // When opening from home page, go directly to assistant backend setup
    // (skip experiment selection since user is not in an experiment context)
    return { step: 'assistant-backend', judgesConfigured: false };
  }

  // Check if experiment has judges configured
  try {
    const judgesResponse = await listScheduledScorers(experimentId);
    const hasJudges = judgesResponse.scorers && judgesResponse.scorers.length > 0;

    // If no judges, skip experiment selection and go to use case
    if (!hasJudges) {
      return { step: 'use-case', judgesConfigured: false };
    }

    // Check if experiment has traces
    const tracesResponse = await searchTracesV4({
      locations: [
        {
          type: 'MLFLOW_EXPERIMENT',
          mlflow_experiment: { experiment_id: experimentId },
        },
      ],
    });
    const hasTraces = tracesResponse.length > 0;

    // If has judges but no traces, skip to instrumentation
    if (!hasTraces) {
      return { step: 'instrumentation', judgesConfigured: true };
    }

    // If has both judges and traces, everything is set up!
    return { step: 'completion', judgesConfigured: true };
  } catch (error) {
    console.error('[OnboardingWizard] Error determining initial step:', error);
    // On error, default to experiment selection (or assistant-backend if not configured)
    if (!assistantConfigured) {
      return { step: 'assistant-backend', judgesConfigured: false };
    }
    return { step: 'experiment-selection', judgesConfigured: false };
  }
};

/**
 * Multi-step onboarding wizard for MLflow GenAI.
 */
export const OnboardingWizard = ({
  onComplete,
  currentExperimentId,
  assistantAlreadyConfigured = false,
}: OnboardingWizardProps) => {
  const { theme } = useDesignSystemTheme();

  const [currentStep, setCurrentStep] = useState<OnboardingStep>('experiment-selection');
  const [isCheckingInitialStep, setIsCheckingInitialStep] = useState(true);

  const [state, setState] = useState<OnboardingState>(() => {
    // Load saved state from localStorage if available
    const savedState = loadWizardState(currentExperimentId);
    return {
      ...INITIAL_STATE,
      ...savedState,
      assistantConfigured: assistantAlreadyConfigured,
      // If we have an experimentId, mark it as selected
      experimentSelected: Boolean(currentExperimentId),
    };
  });

  // Determine initial step based on current context
  useEffect(() => {
    const checkInitialStep = async () => {
      // Set checking flag at the start to prevent race conditions with child component effects
      setIsCheckingInitialStep(true);

      // Load saved state ONLY for the current experiment (note: useState initializer only runs on first mount,
      // so we need to reload when experimentId changes)
      // IMPORTANT: Only load state if it's for THIS specific experiment to prevent cross-experiment contamination
      const savedState = loadWizardState(currentExperimentId);
      if (savedState && currentExperimentId) {
        // Verify the saved state is actually for this experiment before applying it
        setState((prev) => ({
          ...INITIAL_STATE,
          ...savedState,
          assistantConfigured: assistantAlreadyConfigured,
          experimentSelected: Boolean(currentExperimentId),
        }));
      } else if (currentExperimentId) {
        // New experiment with no saved state - reset to initial
        setState((prev) => ({
          ...INITIAL_STATE,
          assistantConfigured: assistantAlreadyConfigured,
          experimentSelected: Boolean(currentExperimentId),
        }));
      }

      // Check for saved step (note: useState initializer only runs on first mount,
      // so we need to check again when experimentId changes)
      const savedStep = loadWizardStep(currentExperimentId);
      if (savedStep) {
        // If saved step is assistant-backend but assistant is already configured, skip it
        if (savedStep === 'assistant-backend' && assistantAlreadyConfigured) {
          const result = await determineInitialStep(currentExperimentId, assistantAlreadyConfigured);
          setCurrentStep(result.step);
          setState((prev) => ({ ...prev, judgesConfigured: result.judgesConfigured }));
        } else {
          setCurrentStep(savedStep);
        }
        setIsCheckingInitialStep(false);
        return;
      }

      // No saved step - determine based on current state
      const result = await determineInitialStep(currentExperimentId, assistantAlreadyConfigured);
      setCurrentStep(result.step);
      setState((prev) => ({ ...prev, judgesConfigured: result.judgesConfigured }));
      setIsCheckingInitialStep(false);
    };

    checkInitialStep();
  }, [currentExperimentId, assistantAlreadyConfigured]);

  // Save current step to localStorage whenever it changes
  // NOTE: We DON'T include currentExperimentId in dependencies to avoid saving the old step
  // to a new experiment key when navigating between experiments
  useEffect(() => {
    if (!isCheckingInitialStep) {
      saveWizardStep(currentStep, currentExperimentId);
    }
  }, [currentStep, isCheckingInitialStep, currentExperimentId]);

  // Save wizard state to localStorage whenever it changes
  useEffect(() => {
    if (!isCheckingInitialStep) {
      saveWizardState(state, currentExperimentId);
    }
  }, [state, isCheckingInitialStep, currentExperimentId]);

  // Track visited steps for forward navigation
  useEffect(() => {
    setState((prev) => {
      if (!prev.visitedSteps.includes(currentStep)) {
        return { ...prev, visitedSteps: [...prev.visitedSteps, currentStep] };
      }
      return prev;
    });
  }, [currentStep]);

  const goToStep = useCallback((step: OnboardingStep) => {
    setCurrentStep(step);
  }, []);

  const completeOnboarding = useCallback(() => {
    setState((prev) => ({ ...prev, completedAt: new Date() }));
    onComplete();
  }, [onComplete]);

  const goToNextStep = useCallback(() => {
    const currentIndex = STEP_ORDER.indexOf(currentStep);
    if (currentIndex < STEP_ORDER.length - 1) {
      let nextIndex = currentIndex + 1;
      let nextStep = STEP_ORDER[nextIndex];

      // Skip completed steps and experiment-specific steps when not in experiment
      while (nextIndex < STEP_ORDER.length) {
        if (nextStep === 'experiment-selection' && state.experimentSelected) {
          nextIndex += 1;
          nextStep = STEP_ORDER[nextIndex];
        } else if (nextStep === 'assistant-backend' && state.assistantConfigured) {
          nextIndex += 1;
          nextStep = STEP_ORDER[nextIndex];
        } else if ((nextStep === 'use-case' || nextStep === 'scorer-selection') && state.judgesConfigured) {
          nextIndex += 1;
          nextStep = STEP_ORDER[nextIndex];
        } else if (
          !currentExperimentId &&
          (nextStep === 'use-case' ||
            nextStep === 'scorer-selection' ||
            nextStep === 'instrumentation' ||
            nextStep === 'completion')
        ) {
          // Skip experiment-specific steps when not in an experiment (including completion)
          // When skipping completion from home page, just complete the onboarding
          if (nextStep === 'completion') {
            completeOnboarding();
            return;
          }
          nextIndex += 1;
          if (nextIndex < STEP_ORDER.length) {
            nextStep = STEP_ORDER[nextIndex];
          }
        } else {
          break;
        }
      }

      // When advancing to completion step (in an experiment context), mark setup as complete
      if (nextStep === 'completion') {
        completeOnboarding();
      }

      setCurrentStep(nextStep);
    }
  }, [
    currentStep,
    currentExperimentId,
    state.assistantConfigured,
    state.experimentSelected,
    state.judgesConfigured,
    completeOnboarding,
  ]);

  const goToPreviousStep = useCallback(() => {
    const currentIndex = STEP_ORDER.indexOf(currentStep);
    if (currentIndex > 0) {
      let prevIndex = currentIndex - 1;
      let prevStep = STEP_ORDER[prevIndex];

      // Skip completed steps and experiment-specific steps when not in experiment
      while (prevIndex >= 0) {
        if (prevStep === 'experiment-selection' && state.experimentSelected) {
          prevIndex -= 1;
          if (prevIndex >= 0) {
            prevStep = STEP_ORDER[prevIndex];
          }
        } else if (prevStep === 'assistant-backend' && state.assistantConfigured) {
          prevIndex -= 1;
          if (prevIndex >= 0) {
            prevStep = STEP_ORDER[prevIndex];
          }
        } else if ((prevStep === 'use-case' || prevStep === 'scorer-selection') && state.judgesConfigured) {
          prevIndex -= 1;
          if (prevIndex >= 0) {
            prevStep = STEP_ORDER[prevIndex];
          }
        } else if (
          !currentExperimentId &&
          (prevStep === 'use-case' ||
            prevStep === 'scorer-selection' ||
            prevStep === 'instrumentation' ||
            prevStep === 'completion')
        ) {
          // Skip experiment-specific steps when not in an experiment (including completion)
          prevIndex -= 1;
          if (prevIndex >= 0) {
            prevStep = STEP_ORDER[prevIndex];
          }
        } else {
          break;
        }
      }

      setCurrentStep(prevStep);
    }
  }, [currentStep, currentExperimentId, state.assistantConfigured, state.experimentSelected, state.judgesConfigured]);

  const updateState = useCallback((updates: Partial<OnboardingState>) => {
    setState((prev) => ({ ...prev, ...updates }));
  }, []);

  const contextValue: OnboardingContextType = {
    state,
    currentStep,
    isCheckingInitialStep,
    currentExperimentId,
    goToStep,
    goToNextStep,
    goToPreviousStep,
    updateState,
    completeOnboarding,
  };

  const stepInfo = STEP_INFO[currentStep];
  const currentStepIndex = STEP_ORDER.indexOf(currentStep);

  // Calculate visible steps (skip completed steps and experiment-only steps when not in experiment)
  const visibleSteps = STEP_ORDER.filter((step) => {
    if (step === 'experiment-selection' && state.experimentSelected) {
      return false; // Skip experiment selection if already in an experiment
    }
    if (step === 'assistant-backend' && state.assistantConfigured) {
      return false; // Skip assistant-backend if configured
    }
    if ((step === 'use-case' || step === 'scorer-selection') && state.judgesConfigured) {
      return false; // Skip judge setup steps if configured
    }
    // Skip experiment-specific steps when not in an experiment (including completion)
    if (
      !currentExperimentId &&
      (step === 'use-case' || step === 'scorer-selection' || step === 'instrumentation' || step === 'completion')
    ) {
      return false;
    }
    return true;
  });

  // Find current step position in visible steps
  const currentVisibleStepIndex = visibleSteps.indexOf(currentStep);
  const showBackButton = currentVisibleStepIndex > 0 && currentStep !== 'completion';
  const showForwardButton = currentVisibleStepIndex < visibleSteps.length - 1 && currentStep !== 'completion';

  // Disable forward button if:
  // 1. On experiment selection and no experiment selected, OR
  // 2. The next step hasn't been visited yet (can't skip ahead)
  const nextVisibleStep = visibleSteps[currentVisibleStepIndex + 1];
  const isForwardButtonDisabled =
    (currentStep === 'experiment-selection' && !state.experimentSelected) ||
    (nextVisibleStep && !state.visitedSteps.includes(nextVisibleStep));

  // Show loading state while checking initial step
  if (isCheckingInitialStep) {
    return (
      <OnboardingContext.Provider value={contextValue}>
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            height: '100%',
            padding: theme.spacing.lg,
          }}
        >
          <Typography.Text color="secondary">
            <FormattedMessage
              defaultMessage="Checking setup status..."
              description="Loading message while determining onboarding step"
            />
          </Typography.Text>
        </div>
      </OnboardingContext.Provider>
    );
  }

  return (
    <OnboardingContext.Provider value={contextValue}>
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          height: '100%',
          overflow: 'hidden',
        }}
      >
        {/* Header with step indicator */}
        <div
          css={{
            padding: theme.spacing.lg,
            borderBottom: `1px solid ${theme.colors.border}`,
            flexShrink: 0,
          }}
        >
          {/* Navigation buttons */}
          {(showBackButton || showForwardButton) && (
            <div
              css={{
                display: 'flex',
                justifyContent: 'space-between',
                marginBottom: theme.spacing.md,
              }}
            >
              {showBackButton ? (
                <button
                  onClick={goToPreviousStep}
                  aria-label="Previous step"
                  css={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    width: 32,
                    height: 32,
                    padding: 0,
                    backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
                    border: `1px solid ${theme.colors.actionPrimaryBackgroundDefault}`,
                    borderRadius: theme.borders.borderRadiusMd,
                    color: theme.colors.white,
                    cursor: 'pointer',
                    '&:hover': {
                      backgroundColor: theme.colors.actionPrimaryBackgroundHover,
                      borderColor: theme.colors.actionPrimaryBackgroundHover,
                    },
                  }}
                >
                  <ChevronLeftIcon css={{ color: theme.colors.white }} />
                </button>
              ) : (
                <div />
              )}
              {showForwardButton && (
                <button
                  onClick={goToNextStep}
                  disabled={isForwardButtonDisabled}
                  aria-label="Next step"
                  css={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    width: 32,
                    height: 32,
                    padding: 0,
                    backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
                    border: `1px solid ${theme.colors.actionPrimaryBackgroundDefault}`,
                    borderRadius: theme.borders.borderRadiusMd,
                    color: theme.colors.white,
                    cursor: 'pointer',
                    '&:hover:not(:disabled)': {
                      backgroundColor: theme.colors.actionPrimaryBackgroundHover,
                      borderColor: theme.colors.actionPrimaryBackgroundHover,
                    },
                    '&:disabled': {
                      opacity: 0.4,
                      cursor: 'not-allowed',
                    },
                  }}
                >
                  <ChevronRightIcon css={{ color: theme.colors.white }} />
                </button>
              )}
            </div>
          )}

          {/* Step progress bar */}
          <div
            css={{
              display: 'flex',
              gap: theme.spacing.xs,
              marginBottom: theme.spacing.lg,
            }}
          >
            {visibleSteps.map((step, index) => (
              <div
                key={step}
                css={{
                  flex: 1,
                  height: 4,
                  borderRadius: 2,
                  backgroundColor:
                    index <= currentVisibleStepIndex
                      ? theme.colors.actionPrimaryBackgroundDefault
                      : theme.colors.backgroundSecondary,
                  transition: 'background-color 0.2s',
                }}
              />
            ))}
          </div>

          {/* Step title and subtitle */}
          <Typography.Title level={3} css={{ marginBottom: theme.spacing.xs }}>
            {stepInfo.title}
          </Typography.Title>
          <Typography.Text color="secondary">{stepInfo.subtitle}</Typography.Text>
        </div>

        {/* Step content */}
        <div css={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
          {!isCheckingInitialStep && <StepContent step={currentStep} currentExperimentId={currentExperimentId} />}
        </div>
      </div>
    </OnboardingContext.Provider>
  );
};

/**
 * Renders the content for the current step.
 */
interface StepContentProps {
  step: OnboardingStep;
  currentExperimentId?: string;
}

const StepContent = ({ step, currentExperimentId }: StepContentProps) => {
  const { goToNextStep, updateState, state } = useOnboarding();

  const handleAssistantConfigured = () => {
    updateState({ assistantConfigured: true });
    goToNextStep();
  };

  const handleAssistantSkipped = () => {
    // User skipped assistant setup - continue without marking as configured
    goToNextStep();
  };

  const handleCodePathChange = (codePath: string) => {
    updateState({ codePath });
  };

  switch (step) {
    case 'experiment-selection':
      return <ExperimentSelectionStep />;
    case 'use-case':
      return <UseCaseStep />;
    case 'scorer-selection':
      return <ScorerSelectionStep />;
    case 'assistant-backend':
      // Only allow skipping assistant setup when in an experiment
      return (
        <AssistantBackendStep
          onConfigured={handleAssistantConfigured}
          onSkip={currentExperimentId ? handleAssistantSkipped : undefined}
          codePath={state.codePath}
          onCodePathChange={handleCodePathChange}
        />
      );
    case 'instrumentation':
      return <InstrumentationStep />;
    case 'completion':
      return <CompletionStep />;
    default:
      return null;
  }
};
