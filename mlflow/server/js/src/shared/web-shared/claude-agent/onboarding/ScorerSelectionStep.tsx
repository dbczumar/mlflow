/**
 * Step 4: Review and customize recommended scorers, configure online scoring.
 */

import { useCallback, useState } from 'react';
import {
  Button,
  Checkbox,
  Input,
  PlusIcon,
  Spinner,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { useOnboarding, type ScorerConfig } from '../OnboardingWizard';
import { EndpointSelector } from '../../../../experiment-tracking/components/EndpointSelector';

const COMPONENT_ID_PREFIX = 'mlflow.onboarding.scorers';

/**
 * Available built-in scorers that can be added.
 */
const AVAILABLE_SCORERS: ScorerConfig[] = [
  {
    id: 'safety',
    type: 'builtin',
    name: 'Safety',
    description: 'Detects harmful, offensive, or inappropriate content',
    builtinType: 'safety',
    enabled: false,
  },
  {
    id: 'correctness',
    type: 'builtin',
    name: 'Correctness',
    description: 'Evaluates factual accuracy of responses',
    builtinType: 'correctness',
    enabled: false,
  },
  {
    id: 'fluency',
    type: 'builtin',
    name: 'Fluency',
    description: 'Evaluates language quality and readability',
    builtinType: 'fluency',
    enabled: false,
  },
  {
    id: 'relevance',
    type: 'builtin',
    name: 'Relevance to Query',
    description: "Checks if response addresses the user's question",
    builtinType: 'relevance_to_query',
    enabled: false,
  },
  {
    id: 'groundedness',
    type: 'builtin',
    name: 'Retrieval Groundedness',
    description: 'Checks if response is grounded in retrieved sources',
    builtinType: 'retrieval_groundedness',
    enabled: false,
  },
  {
    id: 'tool_correctness',
    type: 'builtin',
    name: 'Tool Call Correctness',
    description: 'Evaluates if tools are used correctly',
    builtinType: 'tool_call_correctness',
    enabled: false,
  },
  {
    id: 'frustration',
    type: 'builtin',
    name: 'User Frustration',
    description: 'Detects signs of user frustration or dissatisfaction',
    builtinType: 'user_frustration',
    enabled: false,
  },
];

/**
 * Step 4: Review and configure scorers for online scoring.
 */
export const ScorerSelectionStep = () => {
  const { theme } = useDesignSystemTheme();
  const { goToNextStep, updateState, state } = useOnboarding();

  const [scorers, setScorers] = useState<ScorerConfig[]>(state.selectedScorers);
  const [showAddScorer, setShowAddScorer] = useState(false);
  const [samplingMode, setSamplingMode] = useState<'all' | 'sample'>(state.samplingMode || 'all');
  const [samplingRate, setSamplingRate] = useState(state.samplingRate || 25);
  const [isEnabling, setIsEnabling] = useState(false);
  const [selectedEndpoint, setSelectedEndpoint] = useState<string | undefined>(state.judgeEndpointName);

  const handleToggleScorer = useCallback((scorerId: string) => {
    setScorers((prev) => prev.map((s) => (s.id === scorerId ? { ...s, enabled: !s.enabled } : s)));
  }, []);

  const handleAddScorer = useCallback(
    (scorer: ScorerConfig) => {
      // Check if already added
      if (scorers.some((s) => s.id === scorer.id)) {
        return;
      }
      setScorers((prev) => [...prev, { ...scorer, enabled: true }]);
      setShowAddScorer(false);
    },
    [scorers],
  );

  const handleEndpointSelect = useCallback((endpointName: string) => {
    setSelectedEndpoint(endpointName);
  }, []);

  const handleEnableOnlineScoring = useCallback(async () => {
    setIsEnabling(true);

    // In a real implementation, this would call the API to create scheduled scorers
    // For now, we simulate the API call
    await new Promise((resolve) => setTimeout(resolve, 1500));

    updateState({
      selectedScorers: scorers,
      samplingMode,
      samplingRate,
      onlineScoringEnabled: true,
      judgeEndpointName: selectedEndpoint,
    });

    setIsEnabling(false);
    goToNextStep();
  }, [goToNextStep, samplingMode, samplingRate, scorers, selectedEndpoint, updateState]);

  const enabledScorers = scorers.filter((s) => s.enabled);
  const availableToAdd = AVAILABLE_SCORERS.filter((available) => !scorers.some((s) => s.id === available.id));

  const isFormValid = enabledScorers.length > 0 && !!selectedEndpoint;

  return (
    <div css={{ padding: theme.spacing.lg }}>
      {/* Endpoint Selection */}
      <div css={{ marginBottom: theme.spacing.lg }}>
        <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.sm }}>
          <FormattedMessage defaultMessage="Judge Endpoint" description="Label for endpoint selection" />
        </Typography.Text>
        <Typography.Text color="secondary" size="sm" css={{ display: 'block', marginBottom: theme.spacing.md }}>
          <FormattedMessage
            defaultMessage="Select or create an LLM endpoint that will power your judges."
            description="Help text for endpoint selection"
          />
        </Typography.Text>
        <EndpointSelector
          currentEndpointName={selectedEndpoint}
          onEndpointSelect={handleEndpointSelect}
          componentIdPrefix={`${COMPONENT_ID_PREFIX}.endpoint`}
          placeholder="Select or create an endpoint..."
        />
      </div>

      {/* Recommended Judges */}
      <div css={{ marginBottom: theme.spacing.lg }}>
        <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.md }}>
          <FormattedMessage defaultMessage="Recommended Judges" description="Label for recommended judges section" />
        </Typography.Text>

        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
          {scorers.map((scorer) => (
            <div
              key={scorer.id}
              css={{
                display: 'flex',
                alignItems: 'flex-start',
                gap: theme.spacing.md,
                padding: theme.spacing.md,
                backgroundColor: theme.colors.backgroundSecondary,
                borderRadius: theme.borders.borderRadiusMd,
                border: `1px solid ${
                  scorer.enabled ? theme.colors.actionPrimaryBackgroundDefault : theme.colors.border
                }`,
              }}
            >
              <Checkbox
                componentId={`${COMPONENT_ID_PREFIX}.scorer_${scorer.id}`}
                isChecked={scorer.enabled}
                onChange={() => handleToggleScorer(scorer.id)}
                css={{ marginTop: 2 }}
              />
              <div css={{ flex: 1 }}>
                <Typography.Text bold css={{ display: 'block' }}>
                  {scorer.name}
                </Typography.Text>
                <Typography.Text color="secondary" size="sm">
                  {scorer.description}
                </Typography.Text>
                {scorer.type === 'guidelines' && scorer.guidelines && (
                  <div
                    css={{
                      marginTop: theme.spacing.sm,
                      padding: theme.spacing.sm,
                      backgroundColor: theme.colors.backgroundPrimary,
                      borderRadius: theme.borders.borderRadiusSm,
                      fontSize: theme.typography.fontSizeSm,
                      fontStyle: 'italic',
                    }}
                  >
                    "{scorer.guidelines}"
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Add More Judges */}
      <div css={{ marginBottom: theme.spacing.lg }}>
        <Button
          componentId={`${COMPONENT_ID_PREFIX}.add_scorer`}
          size="small"
          icon={<PlusIcon />}
          onClick={() => setShowAddScorer(!showAddScorer)}
        >
          <FormattedMessage defaultMessage="Add Judge" description="Add judge button" />
        </Button>

        {/* Add Judge Dropdown */}
        {showAddScorer && availableToAdd.length > 0 && (
          <div
            css={{
              marginTop: theme.spacing.sm,
              padding: theme.spacing.md,
              backgroundColor: theme.colors.backgroundSecondary,
              borderRadius: theme.borders.borderRadiusMd,
              border: `1px solid ${theme.colors.border}`,
            }}
          >
            <Typography.Text size="sm" bold css={{ display: 'block', marginBottom: theme.spacing.sm }}>
              <FormattedMessage defaultMessage="Available Judges:" description="Available judges label" />
            </Typography.Text>
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
              {availableToAdd.map((scorer) => (
                <button
                  key={scorer.id}
                  onClick={() => handleAddScorer(scorer)}
                  css={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: theme.spacing.sm,
                    backgroundColor: theme.colors.backgroundPrimary,
                    border: `1px solid ${theme.colors.border}`,
                    borderRadius: theme.borders.borderRadiusSm,
                    cursor: 'pointer',
                    textAlign: 'left',
                    '&:hover': {
                      borderColor: theme.colors.actionPrimaryBackgroundDefault,
                    },
                  }}
                >
                  <div>
                    <Typography.Text size="sm" bold>
                      {scorer.name}
                    </Typography.Text>
                    <Typography.Text size="sm" color="secondary" css={{ display: 'block' }}>
                      {scorer.description}
                    </Typography.Text>
                  </div>
                  <PlusIcon css={{ color: theme.colors.textSecondary }} />
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Sampling Configuration */}
      <div
        css={{
          padding: theme.spacing.lg,
          backgroundColor: theme.colors.backgroundSecondary,
          borderRadius: theme.borders.borderRadiusLg,
          marginBottom: theme.spacing.lg,
        }}
      >
        <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.md }}>
          <FormattedMessage defaultMessage="Scope" description="Sampling configuration label" />
        </Typography.Text>

        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
          <label
            css={{
              display: 'flex',
              alignItems: 'center',
              gap: theme.spacing.sm,
              cursor: 'pointer',
            }}
          >
            <input
              type="radio"
              name="samplingMode"
              checked={samplingMode === 'all'}
              onChange={() => setSamplingMode('all')}
            />
            <Typography.Text>
              <FormattedMessage defaultMessage="All traces" description="All traces option" />
            </Typography.Text>
          </label>

          <label
            css={{
              display: 'flex',
              alignItems: 'center',
              gap: theme.spacing.sm,
              cursor: 'pointer',
            }}
          >
            <input
              type="radio"
              name="samplingMode"
              checked={samplingMode === 'sample'}
              onChange={() => setSamplingMode('sample')}
            />
            <Typography.Text>
              <FormattedMessage
                defaultMessage="Sample (recommended for high-volume apps)"
                description="Sample option"
              />
            </Typography.Text>
          </label>

          {samplingMode === 'sample' && (
            <div css={{ marginLeft: theme.spacing.lg, display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
              <Input
                componentId={`${COMPONENT_ID_PREFIX}.sampling_rate`}
                type="number"
                min={1}
                max={100}
                value={samplingRate}
                onChange={(e) => setSamplingRate(parseInt(e.target.value, 10) || 25)}
                css={{ width: 80 }}
              />
              <Typography.Text>
                <FormattedMessage defaultMessage="% of traces" description="Sampling rate label" />
              </Typography.Text>
            </div>
          )}
        </div>
      </div>

      {/* Summary */}
      <div
        css={{
          padding: theme.spacing.md,
          backgroundColor: theme.colors.backgroundSecondary,
          borderRadius: theme.borders.borderRadiusMd,
          marginBottom: theme.spacing.lg,
          border: `1px solid ${theme.colors.border}`,
        }}
      >
        <Typography.Text color="secondary" size="sm">
          {selectedEndpoint ? (
            <FormattedMessage
              defaultMessage="{count} {count, plural, one {judge} other {judges}} using {endpoint} will evaluate {scope} of your traces."
              description="Summary of online scoring configuration with endpoint"
              values={{
                count: enabledScorers.length,
                endpoint: selectedEndpoint,
                scope: samplingMode === 'all' ? 'all' : `${samplingRate}%`,
              }}
            />
          ) : (
            <FormattedMessage
              defaultMessage="Select an endpoint to power your judges."
              description="Summary when no endpoint selected"
            />
          )}
        </Typography.Text>
      </div>

      {/* Enable Button */}
      <Button
        componentId={`${COMPONENT_ID_PREFIX}.enable`}
        type="primary"
        onClick={handleEnableOnlineScoring}
        disabled={!isFormValid || isEnabling}
        css={{ width: '100%' }}
      >
        {isEnabling ? (
          <>
            <Spinner size="small" css={{ marginRight: theme.spacing.sm }} />
            <FormattedMessage defaultMessage="Enabling Online Scoring..." description="Enabling button text" />
          </>
        ) : (
          <FormattedMessage defaultMessage="Enable Online Scoring" description="Enable button text" />
        )}
      </Button>
    </div>
  );
};
