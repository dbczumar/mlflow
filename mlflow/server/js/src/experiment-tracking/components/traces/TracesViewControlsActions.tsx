import { Button, useDesignSystemTheme } from '@databricks/design-system';
import React, { useCallback, useMemo, useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { TracesViewDeleteTraceModal } from './TracesViewDeleteTraceModal';
import { GenAITraceComparisonModal } from '@mlflow/mlflow/src/shared/web-shared/genai-traces-table/components/GenAITraceComparisonModal';

export const TracesViewControlsActions = ({
  experimentIds,
  rowSelection,
  setRowSelection,
  refreshTraces,
  baseComponentId,
}: {
  experimentIds: string[];
  rowSelection: { [id: string]: boolean };
  setRowSelection: (rowSelection: { [id: string]: boolean }) => void;
  refreshTraces: () => void;
  baseComponentId: string;
}) => {
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [isCompareModalOpen, setIsCompareModalOpen] = useState(false);
  const { theme } = useDesignSystemTheme();

  const openDeleteModal = useCallback(() => {
    setIsDeleteModalOpen(true);
  }, []);

  const closeDeleteModal = useCallback(() => {
    setIsDeleteModalOpen(false);
  }, []);

  const openCompareModal = useCallback(() => {
    setIsCompareModalOpen(true);
  }, []);

  const closeCompareModal = useCallback(() => {
    setIsCompareModalOpen(false);
  }, []);

  const selectedTraceIds = useMemo(() => {
    return Object.entries(rowSelection)
      .filter(([, isSelected]) => isSelected)
      .map(([id]) => id);
  }, [rowSelection]);

  const canCompare = selectedTraceIds.length >= 2 && selectedTraceIds.length <= 3;

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'row',
        alignItems: 'center',
        gap: theme.spacing.sm,
      }}
    >
      <Button
        componentId={`${baseComponentId}.traces_table.compare_traces`}
        onClick={openCompareModal}
        disabled={!canCompare}
      >
        <FormattedMessage
          defaultMessage="Compare"
          description="Experiment page > traces view controls > Compare button"
        />
      </Button>
      <Button componentId={`${baseComponentId}.traces_table.delete_traces`} onClick={openDeleteModal} danger>
        <FormattedMessage
          defaultMessage="Delete"
          description="Experiment page > traces view controls > Delete button"
        />
      </Button>
      <TracesViewDeleteTraceModal
        experimentIds={experimentIds}
        visible={isDeleteModalOpen}
        rowSelection={rowSelection}
        handleClose={closeDeleteModal}
        refreshTraces={refreshTraces}
        setRowSelection={setRowSelection}
      />
      {isCompareModalOpen && (
        <GenAITraceComparisonModal traceIds={selectedTraceIds} onClose={closeCompareModal} />
      )}
    </div>
  );
};
