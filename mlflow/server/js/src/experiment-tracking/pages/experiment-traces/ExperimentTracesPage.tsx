import { useEffect, useMemo } from 'react';
import invariant from 'invariant';
import { useParams } from '../../../common/utils/RoutingUtils';
import { ExperimentViewTraces } from '../../components/experiment-page/components/ExperimentViewTraces';
import { useGlobalClaudeOptional } from '../../../shared/web-shared/claude-agent/GlobalClaudeContext';
import { useExperiments } from '../../components/experiment-page/hooks/useExperiments';
import { getExperimentKindFromTags } from '../../utils/ExperimentKindUtils';

const ExperimentTracesPage = () => {
  const { experimentId } = useParams();
  invariant(experimentId, 'Experiment ID must be defined');

  const experimentIds = useMemo(() => [experimentId], [experimentId]);

  // Get experiment data to extract experimentKind
  const experiments = useExperiments([experimentId]);
  const experiment = experiments[0];
  const experimentKind = experiment ? getExperimentKindFromTags(experiment.tags) : undefined;

  const globalClaude = useGlobalClaudeOptional();
  const setClaudeContext = globalClaude?.setContext;

  // Update global Claude context when viewing an experiment's traces
  useEffect(() => {
    if (setClaudeContext && experimentId) {
      console.log('[ExperimentTracesPage] Setting context with experimentId:', experimentId);
      setClaudeContext({
        type: 'experiment',
        summary: `Experiment ${experimentId} traces`,
        data: null,
        navigation: {
          experimentId,
          experimentKind,
        },
      });
    }
  }, [setClaudeContext, experimentId, experimentKind]);

  return <ExperimentViewTraces experimentIds={experimentIds} />;
};

export default ExperimentTracesPage;
