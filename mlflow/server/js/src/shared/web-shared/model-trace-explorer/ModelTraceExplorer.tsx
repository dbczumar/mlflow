import { useEffect, useRef, useState } from 'react';
import { ErrorBoundary } from 'react-error-boundary';
import { useDesignSystemTheme } from '@databricks/design-system';

import { getLargeTraceDisplaySizeThreshold, shouldBlockLargeTraceDisplay } from './FeatureUtils';
import type { ModelTrace } from './ModelTrace.types';
import { getModelTraceId, getModelTraceSize } from './ModelTraceExplorer.utils';
import { ModelTraceExplorerErrorState } from './ModelTraceExplorerErrorState';
import { ModelTraceExplorerGenericErrorState } from './ModelTraceExplorerGenericErrorState';
import { ModelTraceExplorerTraceTooLargeView } from './ModelTraceExplorerTraceTooLargeView';
import { ModelTraceExplorerViewStateProvider } from './ModelTraceExplorerViewStateContext';
import { ModelTraceHeaderDetails } from './ModelTraceHeaderDetails';
import { useGetModelTraceInfo } from './hooks/useGetModelTraceInfo';
import { useTraceCachedActions } from './hooks/useTraceCachedActions';
import { ModelTraceExplorerContent } from './ModelTraceExplorerContent';
import { ModelTraceExplorerComparisonView } from './ModelTraceExplorerComparisonView';
import { ClaudeAgentProvider, ClaudeAgentTabContent, useClaudeAgentContextOptional } from '../claude-agent';

const ContextProviders = ({ children }: { traceId: string; children: React.ReactNode }) => {
  return <ErrorBoundary fallbackRender={ModelTraceExplorerErrorState}>{children}</ErrorBoundary>;
};

export const ModelTraceExplorerImpl = ({
  modelTrace: initialModelTrace,
  className,
  initialActiveView,
  selectedSpanId,
  onSelectSpan,
  collapseAssessmentPane,
  isInComparisonView,
}: {
  modelTrace: ModelTrace;
  className?: string;
  initialActiveView?: 'summary' | 'detail';
  selectedSpanId?: string;
  onSelectSpan?: (selectedSpanId?: string) => void;
  /**
   * If set to `false`, the assessments pane will be expanded if there are any assessments.
   * If set to `'force-open'`, the assessments pane will be expanded regardless of whether there are any assessments.
   */
  collapseAssessmentPane?: boolean | 'force-open';
  isInComparisonView?: boolean;
}) => {
  const [modelTrace, setModelTrace] = useState(initialModelTrace);
  const [forceDisplay, setForceDisplay] = useState(false);
  const traceId = getModelTraceId(initialModelTrace);
  // older traces don't have a size, so we default to 0 to always display them
  const size = getModelTraceSize(initialModelTrace) ?? 0;
  // always displayable if the feature flag is disabled
  const isDisplayable = shouldBlockLargeTraceDisplay() ? size < getLargeTraceDisplaySizeThreshold() : true;
  const spanLength = initialModelTrace.data?.spans?.length ?? 0;
  const [assessmentsPaneEnabled, setAssessmentsPaneEnabled] = useState(traceId.startsWith('tr-'));
  const [isMountingTrace, setIsMountingTrace] = useState(true);

  const { isFetching } = useGetModelTraceInfo({
    traceId,
    setModelTrace,
    setAssessmentsPaneEnabled,
    enabled: isDisplayable,
  });

  const isTraceInitialLoading = isMountingTrace && isFetching;

  useEffect(() => {
    setModelTrace(initialModelTrace);
    setIsMountingTrace(true);
    // reset the model trace when the traceId changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [traceId, spanLength]);

  useEffect(() => {
    if (isMountingTrace && !isFetching) {
      setIsMountingTrace(false);
    }
  }, [isMountingTrace, isFetching]);

  const resetActionCache = useTraceCachedActions((state) => state.resetCache);

  // Reset the cache each time a trace explorer is mounted
  useEffect(() => {
    resetActionCache();
  }, [resetActionCache]);

  if (!isDisplayable && !forceDisplay) {
    return <ModelTraceExplorerTraceTooLargeView traceId={traceId} setForceDisplay={setForceDisplay} />;
  }

  return (
    <ClaudeAgentProvider>
      <ModelTraceExplorerInner
        traceId={traceId}
        modelTrace={modelTrace}
        initialActiveView={initialActiveView}
        selectedSpanId={selectedSpanId}
        onSelectSpan={onSelectSpan}
        assessmentsPaneEnabled={assessmentsPaneEnabled}
        isInComparisonView={isInComparisonView}
        collapseAssessmentPane={collapseAssessmentPane}
        isTraceInitialLoading={isTraceInitialLoading}
        className={className}
      />
    </ClaudeAgentProvider>
  );
};

/**
 * Inner component that has access to Claude context.
 */
const ModelTraceExplorerInner = ({
  traceId,
  modelTrace,
  initialActiveView,
  selectedSpanId,
  onSelectSpan,
  assessmentsPaneEnabled,
  isInComparisonView,
  collapseAssessmentPane,
  isTraceInitialLoading,
  className,
}: {
  traceId: string;
  modelTrace: ModelTrace;
  initialActiveView?: 'summary' | 'detail';
  selectedSpanId?: string;
  onSelectSpan?: (selectedSpanId?: string) => void;
  assessmentsPaneEnabled: boolean;
  isInComparisonView?: boolean;
  collapseAssessmentPane?: boolean | 'force-open';
  isTraceInitialLoading: boolean;
  className?: string;
}) => {
  const { theme } = useDesignSystemTheme();
  const claudeAgent = useClaudeAgentContextOptional();
  const isClaudeTabActive = claudeAgent?.isClaudeTabActive ?? false;
  const isClaudeAvailable = claudeAgent?.isClaudeAvailable ?? false;
  const hasAutoOpenedRef = useRef(false);

  // Auto-open Claude panel when it's available and configured
  useEffect(() => {
    if (isClaudeAvailable && !hasAutoOpenedRef.current && claudeAgent?.openClaudeTab) {
      hasAutoOpenedRef.current = true;
      claudeAgent.openClaudeTab(modelTrace);
    }
  }, [isClaudeAvailable, claudeAgent, modelTrace]);

  return (
    <ContextProviders traceId={traceId}>
      <ModelTraceExplorerViewStateProvider
        modelTrace={modelTrace}
        initialActiveView={initialActiveView}
        selectedSpanIdOnRender={selectedSpanId}
        assessmentsPaneEnabled={assessmentsPaneEnabled}
        isInComparisonView={isInComparisonView}
        initialAssessmentsPaneCollapsed={collapseAssessmentPane}
        isTraceInitialLoading={isTraceInitialLoading}
      >
        <ModelTraceHeaderDetails modelTraceInfo={modelTrace.info} modelTrace={modelTrace} />
        <div
          css={{
            display: 'flex',
            flex: 1,
            minHeight: 0,
            overflow: 'hidden',
          }}
        >
          {/* Main trace content - takes 2/3 when Claude is open, full width otherwise */}
          <div
            css={{
              flex: isClaudeTabActive ? '0 0 66%' : 1,
              minWidth: 0,
              overflow: 'hidden',
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            {isInComparisonView ? (
              <ModelTraceExplorerComparisonView modelTraceInfo={modelTrace.info} />
            ) : (
              <ModelTraceExplorerContent
                modelTraceInfo={modelTrace.info}
                className={className}
                selectedSpanId={selectedSpanId}
                onSelectSpan={onSelectSpan}
              />
            )}
          </div>

          {/* Claude panel on the right - 1/3 width */}
          {isClaudeTabActive && (
            <div
              css={{
                flex: '0 0 34%',
                minWidth: 0,
                borderLeft: `1px solid ${theme.colors.border}`,
                display: 'flex',
                flexDirection: 'column',
                overflow: 'hidden',
              }}
            >
              <ClaudeAgentTabContent />
            </div>
          )}
        </div>
      </ModelTraceExplorerViewStateProvider>
    </ContextProviders>
  );
};

export const ModelTraceExplorer = ModelTraceExplorerImpl;
