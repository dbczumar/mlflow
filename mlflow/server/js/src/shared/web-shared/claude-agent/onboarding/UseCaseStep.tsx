/**
 * Step 3: Select your use case to get recommended scorers.
 */

import { useCallback, useState } from 'react';
import { Button, Input, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { useOnboarding, type ScorerConfig, type UseCaseType } from '../OnboardingWizard';

const COMPONENT_ID_PREFIX = 'mlflow.onboarding.usecase';

/**
 * Use case options with icons and descriptions.
 */
interface UseCaseOption {
  id: UseCaseType;
  icon: string;
  label: string;
  description: string;
}

const USE_CASE_OPTIONS: UseCaseOption[] = [
  {
    id: 'chatbot',
    icon: '💬',
    label: 'Chatbot',
    description: 'Multi-turn conversational assistant with dialogue history',
  },
  {
    id: 'rag',
    icon: '🔍',
    label: 'RAG / Q&A',
    description: 'Single-turn question answering over documents or knowledge bases',
  },
  {
    id: 'summarization',
    icon: '📋',
    label: 'Summarization',
    description: 'Condensing documents, articles, or long-form content',
  },
  {
    id: 'extraction',
    icon: '🔎',
    label: 'Information Extraction',
    description: 'Extracting structured data, entities, or facts from text',
  },
  {
    id: 'content',
    icon: '📝',
    label: 'Content Generation',
    description: 'Writing, creative content, or text generation',
  },
  {
    id: 'classification',
    icon: '🏷️',
    label: 'Classification',
    description: 'Categorization, sentiment analysis, or intent detection',
  },
  {
    id: 'code',
    icon: '💻',
    label: 'Coding',
    description: 'Code writing, review, or programming assistance',
  },
  {
    id: 'other',
    icon: '🎯',
    label: 'Other',
    description: 'Describe your use case for tailored recommendations',
  },
];

/**
 * Pre-configured scorers for each use case.
 */
const USE_CASE_SCORERS: Record<UseCaseType, ScorerConfig[]> = {
  chatbot: [
    {
      id: 'frustration',
      type: 'builtin',
      name: 'User Frustration',
      description: 'Detects signs of user frustration or dissatisfaction',
      builtinType: 'user_frustration',
      enabled: true,
    },
    {
      id: 'completeness',
      type: 'builtin',
      name: 'Conversation Completeness',
      description: 'Evaluates if the conversation fully addresses user needs',
      builtinType: 'conversation_completeness',
      enabled: true,
    },
    {
      id: 'tool_correctness',
      type: 'builtin',
      name: 'Tool Call Correctness',
      description: 'Evaluates if tools are used correctly',
      builtinType: 'tool_call_correctness',
      enabled: true,
    },
    {
      id: 'safety',
      type: 'builtin',
      name: 'Safety',
      description: 'Detects harmful, offensive, or inappropriate content',
      builtinType: 'safety',
      enabled: true,
    },
  ],
  rag: [
    {
      id: 'correctness',
      type: 'builtin',
      name: 'Correctness',
      description: 'Evaluates factual accuracy of responses',
      builtinType: 'correctness',
      enabled: true,
    },
    {
      id: 'groundedness',
      type: 'builtin',
      name: 'Retrieval Groundedness',
      description: 'Checks if response is grounded in retrieved sources',
      builtinType: 'retrieval_groundedness',
      enabled: true,
    },
    {
      id: 'retrieval_relevance',
      type: 'builtin',
      name: 'Retrieval Relevance',
      description: 'Evaluates if retrieved documents are relevant',
      builtinType: 'retrieval_relevance',
      enabled: true,
    },
    {
      id: 'sufficiency',
      type: 'builtin',
      name: 'Retrieval Sufficiency',
      description: 'Checks if retrieved context is sufficient to answer',
      builtinType: 'retrieval_sufficiency',
      enabled: true,
    },
  ],
  summarization: [
    {
      id: 'fluency',
      type: 'builtin',
      name: 'Fluency',
      description: 'Evaluates language quality and readability',
      builtinType: 'fluency',
      enabled: true,
    },
    {
      id: 'correctness',
      type: 'builtin',
      name: 'Correctness',
      description: 'Evaluates factual accuracy of the summary',
      builtinType: 'correctness',
      enabled: true,
    },
    {
      id: 'groundedness',
      type: 'builtin',
      name: 'Groundedness',
      description: 'Checks if summary is grounded in the source content',
      builtinType: 'retrieval_groundedness',
      enabled: true,
    },
    {
      id: 'safety',
      type: 'builtin',
      name: 'Safety',
      description: 'Detects harmful or inappropriate content',
      builtinType: 'safety',
      enabled: true,
    },
  ],
  extraction: [
    {
      id: 'correctness',
      type: 'builtin',
      name: 'Correctness',
      description: 'Evaluates accuracy of extracted information',
      builtinType: 'correctness',
      enabled: true,
    },
    {
      id: 'groundedness',
      type: 'builtin',
      name: 'Groundedness',
      description: 'Checks if extracted data is grounded in source',
      builtinType: 'retrieval_groundedness',
      enabled: true,
    },
    {
      id: 'guidelines',
      type: 'guidelines',
      name: 'Extraction Schema',
      description: 'Custom guidelines for extraction format and completeness',
      guidelines: 'Extracted data should match the expected schema and include all required fields.',
      enabled: true,
    },
  ],
  content: [
    {
      id: 'fluency',
      type: 'builtin',
      name: 'Fluency',
      description: 'Evaluates language quality and readability',
      builtinType: 'fluency',
      enabled: true,
    },
    {
      id: 'safety',
      type: 'builtin',
      name: 'Safety',
      description: 'Detects harmful or inappropriate content',
      builtinType: 'safety',
      enabled: true,
    },
    {
      id: 'guidelines',
      type: 'guidelines',
      name: 'Content Guidelines',
      description: 'Custom guidelines for your content standards',
      guidelines: 'Content should be engaging, accurate, and appropriate for the target audience.',
      enabled: true,
    },
  ],
  code: [
    {
      id: 'correctness',
      type: 'builtin',
      name: 'Correctness',
      description: 'Evaluates if generated code is correct',
      builtinType: 'correctness',
      enabled: true,
    },
    {
      id: 'safety',
      type: 'builtin',
      name: 'Safety',
      description: 'Detects potentially malicious or insecure code',
      builtinType: 'safety',
      enabled: true,
    },
    {
      id: 'guidelines',
      type: 'guidelines',
      name: 'Coding Standards',
      description: 'Custom guidelines for code quality',
      guidelines: 'Code should follow best practices, be well-documented, and handle errors appropriately.',
      enabled: true,
    },
  ],
  classification: [
    {
      id: 'correctness',
      type: 'builtin',
      name: 'Correctness',
      description: 'Evaluates accuracy of classification',
      builtinType: 'correctness',
      enabled: true,
    },
    {
      id: 'guidelines',
      type: 'guidelines',
      name: 'Classification Schema',
      description: 'Custom guidelines for valid categories and classification rules',
      guidelines: 'Classifications should match one of the predefined categories and follow the labeling guidelines.',
      enabled: true,
    },
  ],
  other: [
    {
      id: 'safety',
      type: 'builtin',
      name: 'Safety',
      description: 'Detects harmful or inappropriate content',
      builtinType: 'safety',
      enabled: true,
    },
    {
      id: 'correctness',
      type: 'builtin',
      name: 'Correctness',
      description: 'Evaluates factual accuracy',
      builtinType: 'correctness',
      enabled: true,
    },
  ],
};

/**
 * Step 3: Select use case for scorer recommendations.
 */
export const UseCaseStep = () => {
  const { theme } = useDesignSystemTheme();
  const { goToNextStep, updateState, state } = useOnboarding();

  const [selectedUseCase, setSelectedUseCase] = useState<UseCaseType | null>(state.useCase);
  const [otherDescription, setOtherDescription] = useState(state.useCaseDescription || '');

  const handleUseCaseSelect = useCallback(
    (useCase: UseCaseType) => {
      setSelectedUseCase(useCase);

      if (useCase !== 'other') {
        // Pre-populate scorers based on use case
        const recommendedScorers = USE_CASE_SCORERS[useCase];
        updateState({
          useCase,
          selectedScorers: recommendedScorers,
        });
        goToNextStep();
      }
    },
    [goToNextStep, updateState],
  );

  const handleOtherContinue = useCallback(() => {
    // For "other", we use a base set of scorers
    // In a real implementation, we could ask Claude to recommend scorers based on description
    const recommendedScorers = USE_CASE_SCORERS.other;
    updateState({
      useCase: 'other',
      useCaseDescription: otherDescription,
      selectedScorers: recommendedScorers,
    });
    goToNextStep();
  }, [goToNextStep, otherDescription, updateState]);

  return (
    <div css={{ padding: theme.spacing.lg }}>
      {/* Use case selection */}
      {selectedUseCase !== 'other' && (
        <div>
          <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.md }}>
            <FormattedMessage
              defaultMessage="What kind of agent are you building?"
              description="Label for use case selection"
            />
          </Typography.Text>

          <div
            css={{
              display: 'grid',
              gridTemplateColumns: 'repeat(2, 1fr)',
              gap: theme.spacing.md,
            }}
          >
            {USE_CASE_OPTIONS.map((option) => (
              <button
                key={option.id}
                onClick={() => handleUseCaseSelect(option.id)}
                css={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  gap: theme.spacing.sm,
                  padding: theme.spacing.lg,
                  backgroundColor: theme.colors.backgroundSecondary,
                  border: `1px solid ${theme.colors.border}`,
                  borderRadius: theme.borders.borderRadiusLg,
                  cursor: 'pointer',
                  textAlign: 'center',
                  transition: 'border-color 0.2s, background-color 0.2s',
                  '&:hover': {
                    borderColor: theme.colors.actionPrimaryBackgroundDefault,
                    backgroundColor: theme.colors.backgroundPrimary,
                  },
                }}
              >
                <span css={{ fontSize: 32 }} role="img" aria-hidden>
                  {option.icon}
                </span>
                <Typography.Text bold>{option.label}</Typography.Text>
                <Typography.Text color="secondary" size="sm">
                  {option.description}
                </Typography.Text>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Other use case description */}
      {selectedUseCase === 'other' && (
        <div>
          <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.md }}>
            <FormattedMessage
              defaultMessage="Describe your application"
              description="Label for other use case description"
            />
          </Typography.Text>

          <Input
            componentId={`${COMPONENT_ID_PREFIX}.other_description`}
            placeholder="My app helps users draft legal contracts based on their requirements. It needs to be accurate and use proper legal terminology."
            value={otherDescription}
            onChange={(e) => setOtherDescription(e.target.value)}
            css={{ marginBottom: theme.spacing.lg }}
          />

          <Typography.Text size="sm" color="secondary" css={{ display: 'block', marginBottom: theme.spacing.lg }}>
            <FormattedMessage
              defaultMessage="We'll recommend scorers based on your description. You can customize them in the next step."
              description="Help text for other use case"
            />
          </Typography.Text>

          <div css={{ display: 'flex', justifyContent: 'space-between' }}>
            <Button componentId={`${COMPONENT_ID_PREFIX}.back`} onClick={() => setSelectedUseCase(null)}>
              <FormattedMessage defaultMessage="Back" description="Back button" />
            </Button>
            <Button
              componentId={`${COMPONENT_ID_PREFIX}.continue`}
              type="primary"
              onClick={handleOtherContinue}
              disabled={!otherDescription.trim()}
            >
              <FormattedMessage defaultMessage="Get Recommendations" description="Continue button" />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
};
