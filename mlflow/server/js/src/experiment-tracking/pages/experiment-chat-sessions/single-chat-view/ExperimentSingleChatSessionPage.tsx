import ErrorUtils from '@mlflow/mlflow/src/common/utils/ErrorUtils';
import { withErrorBoundary } from '@mlflow/mlflow/src/common/utils/withErrorBoundary';
import { FormattedMessage } from '@mlflow/mlflow/src/i18n/i18n';
import type { GetTraceFunction } from '@databricks/web-shared/genai-traces-table';
import {
  createTraceLocationForExperiment,
  createTraceLocationForUCSchema,
  doesTraceSupportV4API,
  useGetTraces,
  useSearchMlflowTraces,
} from '@databricks/web-shared/genai-traces-table';
import { useGlobalClaudeOptional } from '@databricks/web-shared/claude-agent';

import { useParams, useLocation } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import invariant from 'invariant';
import { useGetExperimentQuery } from '../../../hooks/useExperimentQuery';
import { useCallback, useMemo, useRef, useState, useEffect } from 'react';
import { TracesV3Toolbar } from '../../../components/experiment-page/components/traces-v3/TracesV3Toolbar';
import type { ModelTrace } from '@databricks/web-shared/model-trace-explorer';
import {
  getModelTraceId,
  isV3ModelTraceInfo,
  ModelTraceExplorer,
  ModelTraceExplorerUpdateTraceContextProvider,
  shouldUseTracesV4API,
} from '@databricks/web-shared/model-trace-explorer';
import {
  ExperimentSingleChatSessionSidebar,
  ExperimentSingleChatSessionSidebarSkeleton,
} from './ExperimentSingleChatSessionSidebar';
import { getTrace as getTraceV3 } from '@mlflow/mlflow/src/experiment-tracking/utils/TraceUtils';
import { getChatSessionsFilter } from '../utils';
import {
  ExperimentSingleChatConversation,
  ExperimentSingleChatConversationSkeleton,
} from './ExperimentSingleChatConversation';
import { Drawer, useDesignSystemTheme } from '@databricks/design-system';
import { SELECTED_TRACE_ID_QUERY_PARAM } from '../../../constants';

const ContextProviders = ({ children }: { children: React.ReactNode }) => {
  return <>{children}</>;
};

const ExperimentSingleChatSessionPageImpl = () => {
  const { theme } = useDesignSystemTheme();
  const { experimentId, sessionId } = useParams();
  const location = useLocation();
  const [selectedTurnIndex, setSelectedTurnIndex] = useState<number | null>(null);
  const [selectedTrace, setSelectedTrace] = useState<ModelTrace | null>(null);
  const chatRefs = useRef<{ [traceId: string]: HTMLDivElement }>({});

  invariant(experimentId, 'Experiment ID must be defined');
  invariant(sessionId, 'Session ID must be defined');

  const selectedTraceIdFromUrl = useMemo(() => {
    const searchParams = new URLSearchParams(location.search);
    return searchParams.get(SELECTED_TRACE_ID_QUERY_PARAM);
  }, [location.search]);

  const { loading: isLoadingExperiment } = useGetExperimentQuery({
    experimentId,
    options: {
      fetchPolicy: 'cache-only',
    },
  });

  const traceSearchLocations = useMemo(
    () => {
      return [createTraceLocationForExperiment(experimentId)];
    },
    // prettier-ignore
    [
      experimentId,
    ],
  );

  const filters = useMemo(() => getChatSessionsFilter({ sessionId }), [sessionId]);

  const { data: traceInfos, isLoading: isLoadingTraceInfos } = useSearchMlflowTraces({
    locations: traceSearchLocations,
    filters,
    disabled: false,
  });

  const sortedTraceInfos = useMemo(() => {
    return traceInfos?.sort((a, b) => new Date(a.request_time).getTime() - new Date(b.request_time).getTime());
  }, [traceInfos]);

  const getTrace = getTraceV3;
  const getAssessmentTitle = useCallback((assessmentName: string) => assessmentName, []);
  const {
    data: traces,
    isLoading: isLoadingTraceDatas,
    invalidateSingleTraceQuery,
  } = useGetTraces(getTrace, sortedTraceInfos);

  useEffect(() => {
    if (selectedTraceIdFromUrl && traces && traces.length > 0 && !isLoadingTraceDatas) {
      const traceIndex = traces.findIndex((trace) => getModelTraceId(trace) === selectedTraceIdFromUrl);
      if (traceIndex !== -1) {
        setSelectedTurnIndex(traceIndex);
        const traceRef = chatRefs.current[selectedTraceIdFromUrl];
        if (traceRef) {
          traceRef.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
      }
    }
  }, [selectedTraceIdFromUrl, traces, isLoadingTraceDatas]);

  // Set Claude context for single session
  const globalClaude = useGlobalClaudeOptional();
  const setContext = globalClaude?.setContext;
  const lastContextKeyRef = useRef<string | null>(null);

  useEffect(() => {
    const contextKey = `session-${sessionId}-${traces?.length ?? 0}`;
    if (setContext && traces && !isLoadingTraceDatas && lastContextKeyRef.current !== contextKey) {
      lastContextKeyRef.current = contextKey;

      setContext({
        type: 'session',
        summary: `Session ${sessionId} (${traces.length} turns)`,
        data: {
          sessionId,
          turns: traces.length,
          traces: traces.map((trace) => {
            const info = trace.info;
            // Check if it's a V3 trace info (has request_preview)
            const isV3 = 'request_preview' in info;
            return {
              trace_id: getModelTraceId(trace),
              request_time: isV3 ? (info as { request_time?: string }).request_time : undefined,
              status: isV3 ? (info as { state?: string }).state : (info as { status?: string }).status,
              request_preview: isV3 ? (info as { request_preview?: string }).request_preview : undefined,
              response_preview: isV3 ? (info as { response_preview?: string }).response_preview : undefined,
            };
          }),
        },
        navigation: {
          experimentId,
          page: 'session-detail',
        },
      });
    }
  }, [setContext, sessionId, experimentId, traces, isLoadingTraceDatas]);

  return (
    <div css={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
      <TracesV3Toolbar
        // prettier-ignore
        viewState="single-chat-session"
        sessionId={sessionId}
      />
      {isLoadingTraceDatas || isLoadingTraceInfos ? (
        <div css={{ display: 'flex', flex: 1, minHeight: 0 }}>
          <ExperimentSingleChatSessionSidebarSkeleton />
          <ExperimentSingleChatConversationSkeleton />
        </div>
      ) : (
        <div css={{ display: 'flex', flex: 1, minHeight: 0 }}>
          <ExperimentSingleChatSessionSidebar
            traces={traces ?? []}
            selectedTurnIndex={selectedTurnIndex}
            setSelectedTurnIndex={setSelectedTurnIndex}
            setSelectedTrace={setSelectedTrace}
            chatRefs={chatRefs}
          />
          <ExperimentSingleChatConversation
            traces={traces ?? []}
            selectedTurnIndex={selectedTurnIndex}
            setSelectedTurnIndex={setSelectedTurnIndex}
            setSelectedTrace={setSelectedTrace}
            chatRefs={chatRefs}
            getAssessmentTitle={getAssessmentTitle}
          />
        </div>
      )}
      <Drawer.Root
        open={selectedTrace !== null}
        onOpenChange={(open) => {
          if (!open) {
            setSelectedTrace(null);
          }
        }}
      >
        <Drawer.Content
          componentId="mlflow.experiment.chat-session.trace-drawer"
          title={selectedTrace ? getModelTraceId(selectedTrace) : ''}
          width="90vw"
          expandContentToFullHeight
        >
          <div
            css={{
              height: '100%',
              marginLeft: -theme.spacing.lg,
              marginRight: -theme.spacing.lg,
              marginBottom: -theme.spacing.lg,
            }}
          >
            <ContextProviders // prettier-ignore
            >
              {selectedTrace && <ModelTraceExplorer modelTrace={selectedTrace} collapseAssessmentPane="force-open" />}
            </ContextProviders>
          </div>
        </Drawer.Content>
      </Drawer.Root>
    </div>
  );
};

const ExperimentSingleChatSessionPage = withErrorBoundary(
  ErrorUtils.mlflowServices.CHAT_SESSIONS,
  ExperimentSingleChatSessionPageImpl,
  <FormattedMessage
    defaultMessage="An error occurred while rendering the chat session."
    description="Generic error message for uncaught errors when rendering a single chat session in MLflow experiment page"
  />,
);

export default ExperimentSingleChatSessionPage;
