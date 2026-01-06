/**
 * Multi-step onboarding wizard for MLflow GenAI.
 * Guides users through:
 * 1. Selecting or creating an experiment
 * 2. Defining their use case for judge recommendations
 * 3. Configuring LLM judges for evaluation
 * 4. Instrumenting their application with tracing
 */

import { createContext, useCallback, useContext, useState, type ReactNode } from 'react';
import { Button, ChevronLeftIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { ExperimentSelectionStep } from './onboarding/ExperimentSelectionStep';
import { UseCaseStep } from './onboarding/UseCaseStep';
import { ScorerSelectionStep } from './onboarding/ScorerSelectionStep';
import { InstrumentationStep } from './onboarding/InstrumentationStep';
import { CompletionStep } from './onboarding/CompletionStep';

const COMPONENT_ID_PREFIX = 'mlflow.onboarding';

/**
 * Onboarding wizard steps.
 * Note: 'assistant-backend' is not in the main flow but can be shown conditionally
 * within the instrumentation step if user chooses "Let Assistant Do It".
 */
export type OnboardingStep =
  | 'experiment-selection'
  | 'use-case'
  | 'scorer-selection'
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
  instrumentationMethod: 'assistant-direct' | 'copy-instructions' | null;
  codePath?: string;
  detectedFrameworks?: string[];
  instrumentationApplied: boolean;
  tracingVerified: boolean;

  // Assistant configuration (used conditionally in instrumentation step)
  assistantConfigured: boolean;

  // Completion
  completedAt: Date | null;
}

/**
 * Context for sharing onboarding state and actions.
 */
interface OnboardingContextType {
  state: OnboardingState;
  currentStep: OnboardingStep;
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
  'instrumentation',
  'completion',
];

/**
 * Step metadata for display.
 */
const STEP_INFO: Record<OnboardingStep, { title: string; subtitle: string }> = {
  'experiment-selection': {
    title: 'Select Experiment',
    subtitle: 'Choose or create an experiment to organize your traces.',
  },
  'use-case': {
    title: 'Select Use Case',
    subtitle: 'Tell us about your application so we can recommend the right judges.',
  },
  'scorer-selection': {
    title: 'Configure Judges',
    subtitle: 'Select LLM judges to automatically evaluate your traces.',
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
  completedAt: null,
};

interface OnboardingWizardProps {
  /** Called when onboarding is complete */
  onComplete: () => void;
  /** Initial step to show (defaults to 'experiment-selection') */
  initialStep?: OnboardingStep;
  /** Whether assistant is already configured (for instrumentation step) */
  assistantAlreadyConfigured?: boolean;
}

/**
 * Multi-step onboarding wizard for MLflow GenAI.
 */
export const OnboardingWizard = ({
  onComplete,
  initialStep = 'experiment-selection',
  assistantAlreadyConfigured = false,
}: OnboardingWizardProps) => {
  const { theme } = useDesignSystemTheme();

  const [currentStep, setCurrentStep] = useState<OnboardingStep>(initialStep);

  const [state, setState] = useState<OnboardingState>({
    ...INITIAL_STATE,
    assistantConfigured: assistantAlreadyConfigured,
  });

  const goToStep = useCallback((step: OnboardingStep) => {
    setCurrentStep(step);
  }, []);

  const goToNextStep = useCallback(() => {
    const currentIndex = STEP_ORDER.indexOf(currentStep);
    if (currentIndex < STEP_ORDER.length - 1) {
      setCurrentStep(STEP_ORDER[currentIndex + 1]);
    }
  }, [currentStep]);

  const goToPreviousStep = useCallback(() => {
    const currentIndex = STEP_ORDER.indexOf(currentStep);
    if (currentIndex > 0) {
      setCurrentStep(STEP_ORDER[currentIndex - 1]);
    }
  }, [currentStep]);

  const updateState = useCallback((updates: Partial<OnboardingState>) => {
    setState((prev) => ({ ...prev, ...updates }));
  }, []);

  const completeOnboarding = useCallback(() => {
    setState((prev) => ({ ...prev, completedAt: new Date() }));
    onComplete();
  }, [onComplete]);

  const contextValue: OnboardingContextType = {
    state,
    currentStep,
    goToStep,
    goToNextStep,
    goToPreviousStep,
    updateState,
    completeOnboarding,
  };

  const stepInfo = STEP_INFO[currentStep];
  const currentStepIndex = STEP_ORDER.indexOf(currentStep);
  const showBackButton = currentStepIndex > 0 && currentStep !== 'completion';

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
          {/* Back button and step counter */}
          <div
            css={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              marginBottom: theme.spacing.md,
            }}
          >
            {showBackButton ? (
              <Button
                componentId={`${COMPONENT_ID_PREFIX}.back`}
                size="small"
                icon={<ChevronLeftIcon />}
                onClick={goToPreviousStep}
              >
                <FormattedMessage defaultMessage="Back" description="Back button" />
              </Button>
            ) : (
              <div />
            )}
            <Typography.Text color="secondary" size="sm">
              <FormattedMessage
                defaultMessage="Step {current} of {total}"
                description="Step counter"
                values={{ current: currentStepIndex + 1, total: STEP_ORDER.length }}
              />
            </Typography.Text>
          </div>

          {/* Step progress bar */}
          <div
            css={{
              display: 'flex',
              gap: theme.spacing.xs,
              marginBottom: theme.spacing.lg,
            }}
          >
            {STEP_ORDER.map((step, index) => (
              <div
                key={step}
                css={{
                  flex: 1,
                  height: 4,
                  borderRadius: 2,
                  backgroundColor:
                    index <= currentStepIndex
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
          <StepContent step={currentStep} />
        </div>
      </div>
    </OnboardingContext.Provider>
  );
};

/**
 * Renders the content for the current step.
 */
const StepContent = ({ step }: { step: OnboardingStep }) => {
  switch (step) {
    case 'experiment-selection':
      return <ExperimentSelectionStep />;
    case 'use-case':
      return <UseCaseStep />;
    case 'scorer-selection':
      return <ScorerSelectionStep />;
    case 'instrumentation':
      return <InstrumentationStep />;
    case 'completion':
      return <CompletionStep />;
    default:
      return null;
  }
};
