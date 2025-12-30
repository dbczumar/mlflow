import { type ModelTraceInfo, type ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { MlflowService } from '../../../sdk/MlflowService';
import { EXPERIMENT_TRACES_SORTABLE_COLUMNS, getTraceInfoRunId } from '../TracesView.utils';
import { ViewType } from '../../../sdk/MlflowEnums';
import { first, uniq, values } from 'lodash';
import type { RunEntity } from '../../../types';
import { postJson } from '../../../../common/utils/FetchUtils';

// A filter expression used to filter traces by run ID
const RUN_ID_FILTER_EXPRESSION = 'request_metadata.`mlflow.sourceRun`';
const LOGGED_MODEL_ID_FILTER_EXPRESSION = 'request_metadata.`mlflow.modelId`';

const createRunIdsFilterExpression = (runUuids: string[]) => {
  const runIdsInQuotes = runUuids.map((runId: any) => `'${runId}'`);
  return `run_id IN (${runIdsInQuotes.join(',')})`;
};

/**
 * Utility function that fetches run names for traces.
 */
const fetchRunNamesForTraces = async (experimentIds: string[], traces: ModelTraceInfo[]) => {
  const traceIdToRunIdMap = traces.reduce<Record<string, string>>((acc, trace) => {
    const traceId = trace.request_id;
    const runId = getTraceInfoRunId(trace);
    if (!traceId || !runId) {
      return acc;
    }
    return { ...acc, [traceId]: runId };
  }, {});

  const runUuids = uniq(values(traceIdToRunIdMap));
  if (runUuids.length < 1) {
    return {};
  }
  const runResponse = (await MlflowService.searchRuns({
    experiment_ids: experimentIds,
    filter: createRunIdsFilterExpression(runUuids),
    run_view_type: ViewType.ALL,
  })) as { runs?: RunEntity[] };

  const runs = runResponse.runs;

  const runIdsToRunNames = (runs || []).reduce<Record<string, string>>((acc, run) => {
    return { ...acc, [run.info.runUuid]: run.info.runName };
  }, {});

  const traceIdsToRunNames = traces.reduce<Record<string, string>>((acc, trace) => {
    const traceId = trace.request_id;
    if (!traceId) {
      return acc;
    }
    const runId = traceIdToRunIdMap[traceId];

    return { ...acc, [traceId]: runIdsToRunNames[runId] || runId };
  }, {});

  return traceIdsToRunNames;
};

// Extended type that includes both v2 and v3 fields, plus run name
export interface ModelTraceInfoWithRunName extends ModelTraceInfo {
  runName?: string;
  // Include assessments from ModelTraceInfoV3 for issue status display
  assessments?: ModelTraceInfoV3['assessments'];
}

export const useExperimentTraces = ({
  experimentIds,
  sorting,
  filter = '',
  runUuid,
  loggedModelId,
}: {
  experimentIds: string[];
  sorting: {
    id: string;
    desc: boolean;
  }[];
  filter?: string;
  runUuid?: string;
  loggedModelId?: string;
}) => {
  const [traces, setTraces] = useState<ModelTraceInfoWithRunName[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<Error | undefined>(undefined);

  // Backend currently only supports ordering by timestamp
  const orderByString = useMemo(() => {
    const firstOrderByColumn = first(sorting);
    if (firstOrderByColumn && EXPERIMENT_TRACES_SORTABLE_COLUMNS.includes(firstOrderByColumn.id)) {
      return `${firstOrderByColumn.id} ${firstOrderByColumn.desc ? 'DESC' : 'ASC'}`;
    }
    return 'timestamp_ms DESC';
  }, [sorting]);

  const filterString = useMemo(() => {
    if (!runUuid && !loggedModelId) {
      return filter;
    }

    if (loggedModelId) {
      if (filter) {
        return `${filter} AND ${LOGGED_MODEL_ID_FILTER_EXPRESSION}='${loggedModelId}'`;
      }
      return `${LOGGED_MODEL_ID_FILTER_EXPRESSION}='${loggedModelId}'`;
    }

    if (filter) {
      return `${filter} AND ${RUN_ID_FILTER_EXPRESSION}='${runUuid}'`;
    }

    return `${RUN_ID_FILTER_EXPRESSION}='${runUuid}'`;
  }, [filter, runUuid, loggedModelId]);

  const [pageTokens, setPageTokens] = useState<Record<string, string | undefined>>({ 0: undefined });
  const [currentPage, setCurrentPage] = useState(0);
  const currentPageToken = pageTokens[currentPage];

  const fetchTraces = useCallback(
    async ({
      experimentIds,
      currentPage = 0,
      pageToken,
      silent,
      orderByString = '',
      filterString = '',
    }: {
      experimentIds: string[];
      currentPage?: number;
      pageToken?: string;
      filterString?: string;
      orderByString?: string;
      silent?: boolean;
    }) => {
      if (!silent) {
        setLoading(true);
      }
      setError(undefined);

      try {
        // Use v3 search API to get traces with assessments
        const locations = experimentIds.map((id) => ({
          type: 'MLFLOW_EXPERIMENT' as const,
          mlflow_experiment: { experiment_id: id },
        }));

        const requestData = {
          locations,
          filter: filterString || undefined,
          max_results: 25,
          order_by: orderByString ? [orderByString] : undefined,
          page_token: pageToken,
        };

        const response = (await postJson({
          relativeUrl: 'ajax-api/3.0/mlflow/traces/search',
          data: requestData,
        })) as { traces?: ModelTraceInfoV3[]; next_page_token?: string };

        if (!response.traces) {
          setTraces([]);
          return;
        }

        // Convert v3 traces to v2 format with run names, preserving assessments
        const v3Traces = response.traces;
        const v2StyleTraces: ModelTraceInfo[] = v3Traces.map((trace) => ({
          request_id: trace.trace_id,
          experiment_id:
            trace.trace_location?.type === 'MLFLOW_EXPERIMENT'
              ? trace.trace_location.mlflow_experiment?.experiment_id
              : undefined,
          timestamp_ms: trace.request_time ? new Date(trace.request_time).getTime() : undefined,
          execution_time_ms: trace.execution_duration ? parseFloat(trace.execution_duration) * 1000 : undefined,
          status: trace.state === 'OK' ? 'OK' : trace.state === 'ERROR' ? 'ERROR' : 'UNSET',
          tags: trace.tags ? Object.entries(trace.tags).map(([key, value]) => ({ key, value })) : [],
          request_metadata: trace.trace_metadata
            ? Object.entries(trace.trace_metadata).map(([key, value]) => ({ key, value }))
            : [],
        }));

        const runNamesForTraces = await fetchRunNamesForTraces(experimentIds, v2StyleTraces);
        const tracesWithRunNames: ModelTraceInfoWithRunName[] = v2StyleTraces.map((trace, index) => {
          const traceId = trace.request_id;
          const v3Trace = v3Traces[index];
          if (!traceId) {
            return { ...trace, assessments: v3Trace.assessments };
          }
          const runName = runNamesForTraces[traceId];
          return { ...trace, runName, assessments: v3Trace.assessments };
        });

        setTraces(tracesWithRunNames);
        setPageTokens((prevPages) => {
          return { ...prevPages, [currentPage + 1]: response.next_page_token };
        });
      } catch (e: any) {
        setError(e);
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  const hasNextPage = !loading && pageTokens[currentPage + 1] !== undefined;
  const hasPreviousPage = !loading && (currentPage === 1 || pageTokens[currentPage - 1] !== undefined);

  useEffect(() => {
    fetchTraces({ experimentIds, filterString, orderByString });
  }, [fetchTraces, filterString, experimentIds, orderByString]);

  const reset = useCallback(() => {
    setTraces([]);
    setPageTokens({ 0: undefined });
    setCurrentPage(0);
    fetchTraces({ experimentIds });
  }, [fetchTraces, experimentIds]);

  const fetchNextPage = useCallback(() => {
    setCurrentPage((prevPage) => prevPage + 1);
    fetchTraces({
      experimentIds,
      currentPage: currentPage + 1,
      pageToken: pageTokens[currentPage + 1],
      filterString,
      orderByString,
    });
  }, [experimentIds, currentPage, fetchTraces, pageTokens, filterString, orderByString]);

  const fetchPrevPage = useCallback(() => {
    setCurrentPage((prevPage) => prevPage - 1);
    fetchTraces({
      experimentIds,
      currentPage: currentPage - 1,
      pageToken: pageTokens[currentPage - 1],
      filterString,
      orderByString,
    });
  }, [experimentIds, currentPage, fetchTraces, pageTokens, filterString, orderByString]);

  const refreshCurrentPage = useCallback(
    (silent = false) => {
      return fetchTraces({
        experimentIds,
        currentPage,
        pageToken: currentPageToken,
        silent,
        filterString,
        orderByString,
      });
    },
    [experimentIds, currentPage, fetchTraces, currentPageToken, filterString, orderByString],
  );

  return {
    traces,
    loading,
    error,
    hasNextPage,
    hasPreviousPage,
    fetchNextPage,
    fetchPrevPage,
    refreshCurrentPage,
    reset,
  };
};
