import { useEffect, useMemo } from 'react';
import invariant from 'invariant';
import { useParams } from '../../../common/utils/RoutingUtils';
import { ExperimentViewTraces } from '../../components/experiment-page/components/ExperimentViewTraces';
import { useGlobalClaudeOptional } from '../../../shared/web-shared/claude-agent/GlobalClaudeContext';

const ExperimentTracesPage = () => {
  const { experimentId } = useParams();
  invariant(experimentId, 'Experiment ID must be defined');

  const experimentIds = useMemo(() => [experimentId], [experimentId]);
  const globalClaude = useGlobalClaudeOptional();

  // Update global Claude context when viewing an experiment's traces
  useEffect(() => {
    if (globalClaude && experimentId) {
      console.log('[ExperimentTracesPage] Setting context with experimentId:', experimentId);
      globalClaude.setContext({
        type: 'experiment',
        summary: `Experiment ${experimentId} traces`,
        data: null,
        navigation: {
          experimentId,
        },
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [experimentId]);

  return <ExperimentViewTraces experimentIds={experimentIds} />;
};

export default ExperimentTracesPage;
