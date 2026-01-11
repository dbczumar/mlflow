/**
 * Multi-step onboarding wizard for MLflow GenAI.
 * Guides users through:
 * 1. Selecting or creating an experiment
 * 2. Defining their use case for judge recommendations
 * 3. Configuring LLM judges for evaluation
 * 4. Instrumenting their application with tracing
 */

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
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
import { CodePathStep } from './onboarding/CodePathStep';
import { InstrumentationStep } from './onboarding/InstrumentationStep';
import { CompletionStep } from './onboarding/CompletionStep';
import { listScheduledScorers } from '../../../experiment-tracking/pages/experiment-scorers/api';
import { searchTracesV4 } from '../model-trace-explorer/api';
import { ExperimentKind } from '../../../experiment-tracking/constants';

const COMPONENT_ID_PREFIX = 'mlflow.onboarding';

/**
 * Onboarding wizard steps.
 */
export type OnboardingStep =
  | 'experiment-selection'
  | 'use-case'
  | 'scorer-selection'
  | 'assistant-backend'
  | 'code-path-selection'
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

  // Step 4: Assistant Backend
  selectedBackend: string | null;
  assistantConfigured: boolean;

  // Step 5: Code Path (conditional - only for Claude Code)
  codePath?: string;

  // Step 6: Instrumentation
  instrumentationMethod: 'assistant-direct' | 'copy-instructions' | 'manual' | null;
  detectedFrameworks?: string[];
  instrumentationApplied: boolean;
  tracingVerified: boolean;

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
 * Context type for determining which steps to show.
 */
type WizardContext = 'home' | 'genai-experiment' | 'ml-experiment';

/**
 * Determine wizard context based on experiment ID and kind.
 */
const determineContext = (experimentId?: string, experimentKind?: string): WizardContext => {
  if (!experimentId) {
    return 'home';
  }
  const isGenAI =
    experimentKind === ExperimentKind.GENAI_DEVELOPMENT || experimentKind === ExperimentKind.GENAI_DEVELOPMENT_INFERRED;
  return isGenAI ? 'genai-experiment' : 'ml-experiment';
};

/**
 * Global steps that apply to all contexts.
 */
const GLOBAL_STEPS: OnboardingStep[] = ['experiment-selection', 'assistant-backend'];

/**
 * GenAI experiment-specific steps (in addition to global steps).
 */
const GENAI_STEPS: OnboardingStep[] = [
  'use-case',
  'scorer-selection',
  'assistant-backend',
  'instrumentation',
  'completion',
];

/**
 * ML experiment-specific steps (currently same as global).
 */
const ML_STEPS: OnboardingStep[] = ['assistant-backend'];

/**
 * Build step order based on wizard context and current state.
 * Local steps always incorporate global steps.
 * Code path step is always included (for progress bar visibility) but skipped during navigation if not needed.
 */
const buildStepOrder = (context: WizardContext, selectedBackend?: string | null): OnboardingStep[] => {
  let baseSteps: OnboardingStep[];

  switch (context) {
    case 'home':
      baseSteps = GLOBAL_STEPS;
      break;
    case 'genai-experiment':
      baseSteps = GENAI_STEPS;
      break;
    case 'ml-experiment':
      baseSteps = ML_STEPS;
      break;
    default:
      baseSteps = GLOBAL_STEPS;
  }

  // Always insert code-path-selection step after assistant-backend for consistent progress bar
  const assistantBackendIndex = baseSteps.indexOf('assistant-backend');
  if (assistantBackendIndex !== -1) {
    // Insert code-path-selection after assistant-backend
    const withCodePath = [...baseSteps];
    withCodePath.splice(assistantBackendIndex + 1, 0, 'code-path-selection');
    return withCodePath;
  }

  return baseSteps;
};

/**
 * Full step order (for backward compatibility with saved state).
 * NOTE: This should only be used for checking if saved step is valid.
 * Use buildStepOrder() for actual navigation.
 */
const ALL_STEPS: OnboardingStep[] = [
  'experiment-selection',
  'use-case',
  'scorer-selection',
  'assistant-backend',
  'code-path-selection',
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
  'code-path-selection': {
    title: 'Link Your Codebase',
    subtitle: 'Specify the path to your agent or application code for more accurate issue detection and detailed improvement recommendations.',
  },
  instrumentation: {
    title: 'Add Tracing',
    subtitle: 'Add tracing to your agent or application to identify and debug issues',
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
  selectedBackend: null,
  assistantConfigured: false,
  instrumentationMethod: null,
  instrumentationApplied: false,
  tracingVerified: false,
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
    if (stored && ALL_STEPS.includes(stored as OnboardingStep)) {
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

/**
 * Load global selected backend from localStorage.
 */
const loadGlobalSelectedBackend = (): string | null => {
  try {
    return localStorage.getItem('mlflow.assistant.selectedBackend.global');
  } catch {
    return null;
  }
};

interface OnboardingWizardProps {
  /** Called when onboarding is complete */
  onComplete: () => void;
  /** Current experiment ID if viewing an experiment */
  currentExperimentId?: string;
  /** Experiment kind (genai_development, machine_learning, etc.) */
  currentExperimentKind?: string;
  /** Whether assistant is already configured (for instrumentation step) */
  assistantAlreadyConfigured?: boolean;
}

/**
 * Determine the appropriate initial step based on current context.
 * Returns step and whether judges are configured.
 */
const determineInitialStep = async (
  experimentId?: string,
  experimentKind?: string,
  assistantConfigured?: boolean,
): Promise<{ step: OnboardingStep; judgesConfigured: boolean }> => {
  // Determine wizard context and build appropriate step order
  const context = determineContext(experimentId, experimentKind);
  // Load global backend if assistant is already configured
  const selectedBackend = assistantConfigured ? loadGlobalSelectedBackend() : null;
  const stepOrder = buildStepOrder(context, selectedBackend);

  // If no experiment (home page), start with first step in order
  if (!experimentId) {
    return { step: stepOrder[0], judgesConfigured: false };
  }

  // For GenAI experiments, check configuration status
  if (context === 'genai-experiment') {
    try {
      // Check if experiment has judges configured
      const judgesResponse = await listScheduledScorers(experimentId);
      const hasJudges = judgesResponse.scorers && judgesResponse.scorers.length > 0;

      // If no judges, start at first GenAI-specific step (use-case)
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
      // On error, start with first step in order
      return { step: stepOrder[0], judgesConfigured: false };
    }
  }

  // For ML experiments, start with first step (assistant-backend)
  return { step: stepOrder[0], judgesConfigured: false };
};

/**
 * Multi-step onboarding wizard for MLflow GenAI.
 */
export const OnboardingWizard = ({
  onComplete,
  currentExperimentId,
  currentExperimentKind,
  assistantAlreadyConfigured = false,
}: OnboardingWizardProps) => {
  const { theme } = useDesignSystemTheme();

  const [currentStep, setCurrentStep] = useState<OnboardingStep>('experiment-selection');
  const [isCheckingInitialStep, setIsCheckingInitialStep] = useState(true);

  const [state, setState] = useState<OnboardingState>(() => {
    // Load saved state from localStorage if available
    const savedState = loadWizardState(currentExperimentId);
    // Load globally selected backend if assistant is already configured
    const globalBackend = assistantAlreadyConfigured ? loadGlobalSelectedBackend() : null;
    return {
      ...INITIAL_STATE,
      ...savedState,
      assistantConfigured: assistantAlreadyConfigured,
      // Use global backend if available and not overridden by experiment-specific state
      selectedBackend: savedState?.selectedBackend || globalBackend,
      // If we have an experimentId, mark it as selected
      experimentSelected: Boolean(currentExperimentId),
    };
  });

  // Determine wizard context based on current experiment
  const wizardContext = useMemo(
    () => determineContext(currentExperimentId, currentExperimentKind),
    [currentExperimentId, currentExperimentKind],
  );

  // Build step order based on context (global vs GenAI vs ML) and selected backend
  const stepOrder = useMemo(() => buildStepOrder(wizardContext, state.selectedBackend), [wizardContext, state.selectedBackend]);

  // Track if we've initialized the step for current experiment to prevent re-initialization
  const initializedExperimentRef = useRef<{ experimentId?: string; experimentKind?: string } | null>(null);

  // Track current experimentId and experimentKind in refs for save logic
  const currentExperimentIdRef = useRef(currentExperimentId);
  const currentExperimentKindRef = useRef(currentExperimentKind);

  // Update refs when props change
  useEffect(() => {
    currentExperimentIdRef.current = currentExperimentId;
    currentExperimentKindRef.current = currentExperimentKind;
  }, [currentExperimentId, currentExperimentKind]);

  // Determine initial step based on current context
  useEffect(() => {
    // CRITICAL: If we're in an experiment context and experimentKind hasn't loaded yet, WAIT!
    // Don't initialize until we know the experimentKind, otherwise we'll validate the saved step
    // against the wrong stepOrder (ML vs GenAI) and reject a valid saved step.
    if (currentExperimentId && currentExperimentKind === undefined) {
      return;
    }

    // Skip if we've already initialized for this experiment+kind combination
    // (prevents re-running when stepOrder changes after experimentKind loads)
    if (
      initializedExperimentRef.current &&
      initializedExperimentRef.current.experimentId === currentExperimentId &&
      initializedExperimentRef.current.experimentKind === currentExperimentKind &&
      currentExperimentKind !== undefined // Only skip if we have a known experimentKind
    ) {
      return;
    }

    const checkInitialStep = async () => {
      // Set checking flag at the start to prevent race conditions with child component effects
      setIsCheckingInitialStep(true);

      // Load saved state ONLY for the current experiment (note: useState initializer only runs on first mount,
      // so we need to reload when experimentId changes)
      // IMPORTANT: Only load state if it's for THIS specific experiment to prevent cross-experiment contamination
      const savedState = loadWizardState(currentExperimentId);
      // Load globally selected backend if assistant is already configured
      const globalBackend = assistantAlreadyConfigured ? loadGlobalSelectedBackend() : null;
      if (savedState && currentExperimentId) {
        // Verify the saved state is actually for this experiment before applying it
        setState((prev) => ({
          ...INITIAL_STATE,
          ...savedState,
          assistantConfigured: assistantAlreadyConfigured,
          // Use global backend if available and not overridden by experiment-specific state
          selectedBackend: savedState.selectedBackend || globalBackend,
          experimentSelected: Boolean(currentExperimentId),
        }));
      } else if (currentExperimentId) {
        // New experiment with no saved state - reset to initial with global backend
        setState((prev) => ({
          ...INITIAL_STATE,
          assistantConfigured: assistantAlreadyConfigured,
          selectedBackend: globalBackend,
          experimentSelected: Boolean(currentExperimentId),
        }));
      }

      // Check for saved step (note: useState initializer only runs on first mount,
      // so we need to check again when experimentId changes)
      // Note: If this is first kind load and step was 'assistant-backend', we cleared it above
      const savedStep = loadWizardStep(currentExperimentId);
      if (savedStep) {
        // Validate that saved step is in the current step order
        if (!stepOrder.includes(savedStep)) {
          // Saved step is not valid for current context - determine new step
          const result = await determineInitialStep(
            currentExperimentId,
            currentExperimentKind,
            assistantAlreadyConfigured,
          );
          setCurrentStep(result.step);
          setState((prev) => ({ ...prev, judgesConfigured: result.judgesConfigured }));
        } else if (savedStep === 'assistant-backend' && assistantAlreadyConfigured) {
          // If saved step is assistant-backend but assistant is already configured, skip it
          const result = await determineInitialStep(
            currentExperimentId,
            currentExperimentKind,
            assistantAlreadyConfigured,
          );
          setCurrentStep(result.step);
          setState((prev) => ({ ...prev, judgesConfigured: result.judgesConfigured }));
        } else {
          setCurrentStep(savedStep);
        }
        setIsCheckingInitialStep(false);
        // Mark as initialized (track experimentKind even if undefined)
        initializedExperimentRef.current = { experimentId: currentExperimentId, experimentKind: currentExperimentKind };
        return;
      }

      // No saved step - determine based on current state
      const result = await determineInitialStep(currentExperimentId, currentExperimentKind, assistantAlreadyConfigured);
      setCurrentStep(result.step);
      setState((prev) => ({ ...prev, judgesConfigured: result.judgesConfigured }));
      setIsCheckingInitialStep(false);
      // Mark as initialized (track experimentKind even if undefined)
      initializedExperimentRef.current = { experimentId: currentExperimentId, experimentKind: currentExperimentKind };
    };

    checkInitialStep();
  }, [currentExperimentId, currentExperimentKind, assistantAlreadyConfigured, stepOrder]);

  // Save current step to localStorage whenever it changes
  // NOTE: We DON'T include currentExperimentId or currentExperimentKind in dependencies
  // to avoid saving during context transitions (navigating between experiments or when experimentKind loads)
  // The effect only runs when currentStep actually changes or when isCheckingInitialStep changes
  // IMPORTANT: Don't save step for an experiment until we know its experimentKind (prevents race condition)
  useEffect(() => {
    if (!isCheckingInitialStep) {
      // Use refs to get current values (not stale closure values)
      const expId = currentExperimentIdRef.current;
      const expKind = currentExperimentKindRef.current;
      // Only save if either: no experiment (home), or experimentKind is known
      const shouldSave = !expId || expKind !== undefined;
      if (shouldSave) {
        saveWizardStep(currentStep, expId);
      }
    }
  }, [currentStep, isCheckingInitialStep]);

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
    const currentIndex = stepOrder.indexOf(currentStep);
    if (currentIndex < stepOrder.length - 1) {
      let nextIndex = currentIndex + 1;
      let nextStep = stepOrder[nextIndex];

      // Skip already completed steps or conditional steps
      while (nextIndex < stepOrder.length) {
        if (nextStep === 'experiment-selection' && state.experimentSelected) {
          nextIndex += 1;
          nextStep = stepOrder[nextIndex];
        } else if (nextStep === 'assistant-backend' && state.assistantConfigured) {
          nextIndex += 1;
          nextStep = stepOrder[nextIndex];
        } else if ((nextStep === 'use-case' || nextStep === 'scorer-selection') && state.judgesConfigured) {
          nextIndex += 1;
          nextStep = stepOrder[nextIndex];
        } else if (nextStep === 'code-path-selection' && state.selectedBackend !== 'claude-code') {
          // Skip code path step if Claude Code is not selected
          nextIndex += 1;
          nextStep = stepOrder[nextIndex];
        } else {
          break;
        }
      }

      // Navigate to next step (including completion step)
      if (nextIndex >= stepOrder.length) {
        // Reached end of steps without completion step (e.g., home page, ML experiment)
        completeOnboarding();
      } else {
        setCurrentStep(nextStep);
      }
    } else {
      // At last step - complete onboarding
      completeOnboarding();
    }
  }, [
    currentStep,
    stepOrder,
    state.assistantConfigured,
    state.experimentSelected,
    state.judgesConfigured,
    state.selectedBackend,
    completeOnboarding,
  ]);

  const goToPreviousStep = useCallback(() => {
    const currentIndex = stepOrder.indexOf(currentStep);
    if (currentIndex > 0) {
      let prevIndex = currentIndex - 1;
      let prevStep = stepOrder[prevIndex];

      // Skip already completed steps or conditional steps
      while (prevIndex >= 0) {
        if (prevStep === 'experiment-selection' && state.experimentSelected) {
          prevIndex -= 1;
          if (prevIndex >= 0) {
            prevStep = stepOrder[prevIndex];
          }
        } else if (prevStep === 'assistant-backend' && state.assistantConfigured) {
          prevIndex -= 1;
          if (prevIndex >= 0) {
            prevStep = stepOrder[prevIndex];
          }
        } else if ((prevStep === 'use-case' || prevStep === 'scorer-selection') && state.judgesConfigured) {
          prevIndex -= 1;
          if (prevIndex >= 0) {
            prevStep = stepOrder[prevIndex];
          }
        } else if (prevStep === 'code-path-selection' && state.selectedBackend !== 'claude-code') {
          // Skip code path step if Claude Code is not selected
          prevIndex -= 1;
          if (prevIndex >= 0) {
            prevStep = stepOrder[prevIndex];
          }
        } else {
          break;
        }
      }

      if (prevIndex >= 0) {
        setCurrentStep(prevStep);
      }
    }
  }, [currentStep, stepOrder, state.assistantConfigured, state.experimentSelected, state.judgesConfigured, state.selectedBackend]);

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
  const currentStepIndex = stepOrder.indexOf(currentStep);

  // Calculate steps to show in progress bar (ALL steps for visual continuity)
  const progressSteps = stepOrder.filter((step) => {
    // Only filter out experiment-selection if already in an experiment
    if (step === 'experiment-selection' && state.experimentSelected) {
      return false;
    }
    // Don't filter judge steps or assistant/code-path - show them all for progress tracking
    return true;
  });

  // Calculate visible steps for navigation (skip already completed steps)
  const visibleSteps = stepOrder.filter((step) => {
    if (step === 'experiment-selection' && state.experimentSelected) {
      return false; // Skip experiment selection if already in an experiment
    }
    if (step === 'assistant-backend' && state.assistantConfigured) {
      return false; // Skip assistant-backend if configured
    }
    if ((step === 'use-case' || step === 'scorer-selection') && state.judgesConfigured) {
      return false; // Skip judge setup steps if configured
    }
    return true;
  });

  // Find current step position in visible steps (for navigation)
  const currentVisibleStepIndex = visibleSteps.indexOf(currentStep);

  // Find current step position in progress steps (for progress bar)
  const currentProgressStepIndex = progressSteps.indexOf(currentStep);
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
              // Add top margin when navigation buttons are hidden (e.g., completion step)
              marginTop: showBackButton || showForwardButton ? 0 : theme.spacing.md + 32, // 32px = button height
            }}
          >
            {progressSteps.map((step, index) => (
              <div
                key={step}
                css={{
                  flex: 1,
                  height: 4,
                  borderRadius: 2,
                  backgroundColor:
                    index <= currentProgressStepIndex
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
          {!isCheckingInitialStep && (
            <StepContent
              step={currentStep}
              currentExperimentId={currentExperimentId}
              currentExperimentKind={currentExperimentKind}
            />
          )}
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
  currentExperimentKind?: string;
}

const StepContent = ({ step, currentExperimentId, currentExperimentKind }: StepContentProps) => {
  const { goToNextStep, goToStep, updateState, state, currentStep } = useOnboarding();

  const handleAssistantConfigured = (backendId: string) => {
    updateState({ assistantConfigured: true, selectedBackend: backendId });

    // Calculate what the step order will be AFTER the state update
    // This ensures we navigate to the correct next step immediately
    const wizardContext = determineContext(currentExperimentId, currentExperimentKind);
    const newStepOrder = buildStepOrder(wizardContext, backendId);
    const currentIndex = newStepOrder.indexOf(currentStep);
    if (currentIndex < newStepOrder.length - 1) {
      goToStep(newStepOrder[currentIndex + 1]);
    }
  };

  const handleAssistantSkipped = () => {
    // User skipped assistant setup - mark selectedBackend as null
    updateState({ selectedBackend: null });
    goToNextStep();
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
        />
      );
    case 'code-path-selection':
      return <CodePathStep />;
    case 'instrumentation':
      return <InstrumentationStep />;
    case 'completion':
      return <CompletionStep />;
    default:
      return null;
  }
};
