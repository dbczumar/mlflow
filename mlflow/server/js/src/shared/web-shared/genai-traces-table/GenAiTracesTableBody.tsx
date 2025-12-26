import { getCoreRowModel, getSortedRowModel } from '@tanstack/react-table';
import type { RowSelectionState, OnChangeFn, ColumnDef, Row } from '@tanstack/react-table';
import { useVirtualizer } from '@tanstack/react-virtual';
import { isNil } from 'lodash';
import React, { useEffect, useMemo, useState } from 'react';

import { Empty, SearchIcon, Spinner, Table, useDesignSystemTheme } from '@databricks/design-system';
import { useIntl } from '@databricks/i18n';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { useReactTable_unverifiedWithReact18 as useReactTable } from '@databricks/web-shared/react-table';

import { GenAITracesTableContext } from './GenAITracesTableContext';
import { sortColumns, sortGroupedColumns } from './GenAiTracesTable.utils';
import { getColumnConfig } from './GenAiTracesTableBody.utils';
import { MemoizedGenAiTracesTableBodyRows } from './GenAiTracesTableBodyRows';
import { GenAiTracesTableHeader } from './GenAiTracesTableHeader';
import { HeaderCellRenderer } from './cellRenderers/HeaderCellRenderer';
import { GenAiEvaluationTracesReviewModal } from './components/GenAiEvaluationTracesReviewModal';
import type { GetTraceFunction } from './hooks/useGetTrace';
import {
  REQUEST_TIME_COLUMN_ID,
  SESSION_COLUMN_ID,
  SERVER_SORTABLE_INFO_COLUMNS,
  ISSUES_COLUMN_ID,
  getUniqueIssueNames,
  generateIssueColumns,
} from './hooks/useTableColumns';
import {
  type EvaluationsOverviewTableSort,
  TracesTableColumnType,
  type AssessmentAggregates,
  type AssessmentFilter,
  type AssessmentInfo,
  type AssessmentValueType,
  type EvalTraceComparisonEntry,
  type SaveAssessmentsQuery,
  type TracesTableColumn,
  TracesTableColumnGroup,
} from './types';
import { getAssessmentAggregates } from './utils/AggregationUtils';
import { escapeCssSpecialCharacters } from './utils/DisplayUtils';
import { getRowIdFromEvaluation } from './utils/TraceUtils';

export const GenAiTracesTableBody = React.memo(
  ({
    experimentId,
    selectedColumns,
    evaluations,
    selectedEvaluationId,
    selectedAssessmentInfos,
    assessmentInfos,
    assessmentFilters,
    tableSort,
    onChangeEvaluationId,
    getRunColor,
    runUuid,
    runDisplayName,
    compareToRunUuid,
    compareToRunDisplayName,
    rowSelection,
    setRowSelection,
    exportToEvalsInstanceEnabled = false,
    getTrace,
    toggleAssessmentFilter,
    saveAssessmentsQuery,
    disableAssessmentTooltips,
    onTraceTagsEdit,
    enableRowSelection,
    enableGrouping = false,
    allColumns,
    displayLoadingOverlay,
    defaultIssuesExpanded = false,
  }: {
    experimentId: string;
    selectedColumns: TracesTableColumn[];
    evaluations: EvalTraceComparisonEntry[];
    selectedEvaluationId: string | undefined;
    selectedAssessmentInfos: AssessmentInfo[];
    assessmentInfos: AssessmentInfo[];
    assessmentFilters: AssessmentFilter[];
    tableSort: EvaluationsOverviewTableSort | undefined;
    onChangeEvaluationId: (evaluationId: string | undefined) => void;
    getRunColor?: (runUuid: string) => string;
    // Current run
    runUuid?: string;
    runDisplayName?: string;
    // Other run
    compareToRunUuid?: string;
    compareToRunDisplayName?: string;
    rowSelection?: RowSelectionState;
    setRowSelection?: OnChangeFn<RowSelectionState>;
    exportToEvalsInstanceEnabled?: boolean;
    getTrace?: GetTraceFunction;
    toggleAssessmentFilter: (
      assessmentName: string,
      filterValue: AssessmentValueType,
      run: string,
      filterType?: AssessmentFilter['filterType'],
    ) => void;
    saveAssessmentsQuery?: SaveAssessmentsQuery;
    disableAssessmentTooltips?: boolean;
    onTraceTagsEdit?: (trace: ModelTraceInfoV3) => void;
    enableRowSelection?: boolean;
    enableGrouping?: boolean;
    allColumns: TracesTableColumn[];
    displayLoadingOverlay?: boolean;
    /** Whether to show issues in expanded view by default */
    defaultIssuesExpanded?: boolean;
  }) => {
    const intl = useIntl();
    const { theme } = useDesignSystemTheme();
    const [collapsedHeader, setCollapsedHeader] = useState(false);
    const [issuesExpanded, setIssuesExpanded] = useState(defaultIssuesExpanded);

    const isComparing = !isNil(compareToRunUuid);

    const evaluationInputs = selectedColumns.filter((col) => col.type === TracesTableColumnType.INPUT);

    // Get unique issue names from evaluations for expanded view
    const uniqueIssueNames = useMemo(() => {
      const allResults = evaluations.flatMap((e) => [e.currentRunValue, e.otherRunValue].filter(Boolean));
      return getUniqueIssueNames(allResults as any[]);
    }, [evaluations]);

    // Generate issue columns for expanded view
    const issueColumns = useMemo(() => generateIssueColumns(uniqueIssueNames), [uniqueIssueNames]);

    // Toggle handler for issues expanded state
    const onToggleIssuesExpanded = React.useCallback(() => {
      setIssuesExpanded((prev) => !prev);
    }, []);

    const { columns, columnsToRender } = useMemo(() => {
      // When issuesExpanded is true, we need to replace the ISSUES_COLUMN_ID with individual issue columns
      // or add an ISSUE group with individual columns in grouped mode
      const getColumnsForRendering = (): TracesTableColumn[] => {
        // Replace the single Issues column with individual issue columns
        const result: TracesTableColumn[] = [];
        for (const col of selectedColumns) {
          if (col.id === ISSUES_COLUMN_ID) {
            if (!issuesExpanded) {
              // In compact mode, hide the second row label (use empty string)
              result.push({
                ...col,
                label: '',
              });
            } else if (issueColumns.length > 0) {
              // Replace with individual issue columns
              result.push(...issueColumns);
            } else {
              // Show "No Issues" column when no issues exist
              result.push({
                ...col,
                label: intl.formatMessage({
                  defaultMessage: 'No Issues',
                  description: 'Column label shown when no issues are logged in expanded view',
                }),
                isEmptyState: true,
              });
            }
          } else {
            result.push(col);
          }
        }
        return result;
      };

      const columnsToRender = getColumnsForRendering();

      if (!enableGrouping) {
        // Return flat columns without grouping
        const columnsList = columnsToRender.map((col) =>
          getColumnConfig(col, {
            evaluationInputs,
            isComparing,
            theme,
            intl,
            experimentId,
            onChangeEvaluationId,
            onTraceTagsEdit,
          }),
        );

        return { columns: sortColumns(columnsList, columnsToRender), columnsToRender };
      }

      // Create a map of group IDs to their column arrays
      const groupColumns = new Map<TracesTableColumnGroup, ColumnDef<EvalTraceComparisonEntry>[]>();
      const sortedGroupedColumns = sortGroupedColumns(columnsToRender, isComparing);

      sortedGroupedColumns.forEach((col) => {
        // Get the group for this column, defaulting to 'Info' if not specified
        const groupId = col.group || TracesTableColumnGroup.INFO;

        // Initialize the group's columns array if it doesn't exist
        if (!groupColumns.has(groupId)) {
          groupColumns.set(groupId, []);
        }

        // or branch of this should never get hit
        (groupColumns.get(groupId) || []).push(
          getColumnConfig(col, {
            evaluationInputs,
            isComparing,
            theme,
            intl,
            experimentId,
            onChangeEvaluationId,
            onTraceTagsEdit,
          }),
        );
      });

      // Convert the map to an array of column groups and sort columns within each group
      const topLevelColumns: ColumnDef<EvalTraceComparisonEntry>[] = Array.from(groupColumns.entries()).map(
        ([groupId, columns]) => {
          // Calculate totalCount - for ISSUE group, always show the actual number of unique issues
          let totalCount: number;
          if (groupId === TracesTableColumnGroup.ISSUE) {
            // Always show the actual number of unique issues
            totalCount = uniqueIssueNames.length;
          } else {
            totalCount = allColumns.filter((col) => col.group === groupId).length;
          }

          return {
            header: HeaderCellRenderer,
            meta: {
              groupId,
              visibleCount: columns.length,
              totalCount,
              enableGrouping,
              // Pass issues toggle state and handler for the ISSUE group
              ...(groupId === TracesTableColumnGroup.ISSUE && {
                issuesExpanded,
                onToggleIssuesExpanded,
              }),
            },
            id: `${groupId}-group`,
            columns,
          };
        },
      );

      return { columns: topLevelColumns, columnsToRender };
    }, [
      selectedColumns,
      evaluationInputs,
      isComparing,
      onChangeEvaluationId,
      theme,
      intl,
      experimentId,
      onTraceTagsEdit,
      enableGrouping,
      allColumns,
      issuesExpanded,
      issueColumns,
      onToggleIssuesExpanded,
      uniqueIssueNames,
    ]);

    const { setTable, setSelectedRowIds } = React.useContext(GenAITracesTableContext);

    const table = useReactTable<EvalTraceComparisonEntry & { multiline?: boolean }>(
      'js/packages/web-shared/src/genai-traces-table/GenAiTracesTableBody.tsx',
      {
        data: evaluations,
        columns,
        getCoreRowModel: getCoreRowModel(),
        getSortedRowModel: getSortedRowModel(),
        enableColumnResizing: true,
        columnResizeMode: 'onChange',
        enableRowSelection,
        enableMultiSort: true,
        state: {
          rowSelection,
        },
        meta: {
          getRunColor,
        },
        onRowSelectionChange: setRowSelection,
        getRowId: (row) => getRowIdFromEvaluation(row.currentRunValue),
      },
    );

    // Need to check if rowSelection is undefined, otherwise getIsAllRowsSelected throws an error
    const allRowSelected = rowSelection !== undefined && table.getIsAllRowsSelected();
    const someRowSelected = table.getIsSomeRowsSelected();

    useEffect(() => {
      setTable(table);

      return () => setTable(undefined);
    }, [table, setTable]);

    useEffect(() => {
      if (enableRowSelection) {
        setSelectedRowIds(table.getSelectedRowModel().rows.map((r) => r.id));
      }
    }, [table, rowSelection, setSelectedRowIds, enableRowSelection]);

    // When the table is empty.
    const emptyDescription = intl.formatMessage({
      defaultMessage: ' No traces found. Try clearing your active filters to see more traces.',
      description: 'Text displayed when no traces are found in the trace review page',
    });

    const emptyComponent = <Empty description={emptyDescription} image={<SearchIcon />} />;
    const isEmpty = (): boolean => table.getRowModel().rows.length === 0;

    // Updating sorting when the prop changes.
    useEffect(() => {
      // Only do client-side sorting for columns that are not supported by the server.
      if (!tableSort || SERVER_SORTABLE_INFO_COLUMNS.includes(tableSort.key)) {
        table.setSorting([]);
        return;
      }

      if (tableSort.key === SESSION_COLUMN_ID) {
        table.setSorting([
          {
            id: tableSort.key,
            desc: !tableSort.asc,
          },
          {
            id: REQUEST_TIME_COLUMN_ID,
            desc: false,
          },
        ]);
      } else {
        table.setSorting([
          {
            id: tableSort.key,
            desc: !tableSort.asc,
          },
        ]);
      }
    }, [tableSort, table]);

    const { rows } = table.getRowModel();

    // The virtualizer needs to know the scrollable container element
    const tableContainerRef = React.useRef<HTMLDivElement>(null);

    const rowVirtualizer = useVirtualizer({
      count: rows.length,
      estimateSize: () => 120, // estimate row height for accurate scrollbar dragging
      getScrollElement: () => tableContainerRef.current,
      measureElement:
        typeof window !== 'undefined' && navigator.userAgent.indexOf('Firefox') === -1
          ? (element) => element?.getBoundingClientRect().height
          : undefined,
      overscan: 10,
    });

    const virtualItems = rowVirtualizer.getVirtualItems();
    const tableHeaderGroups = table.getHeaderGroups();

    /**
     * Instead of calling `column.getSize()` on every render for every header
     * and especially every data cell (very expensive),
     * we will calculate all column sizes at once at the root table level in a useMemo
     * and pass the column sizes down as CSS variables to the <table> element.
     */
    const { columnSizeVars, tableWidth } = useMemo(() => {
      const colSizes: { [key: string]: string } = {};
      tableHeaderGroups.forEach((headerGroup) => {
        headerGroup.headers.forEach((header) => {
          colSizes[`--header-${escapeCssSpecialCharacters(header.column.id)}-size`] = header.getSize() + 'px';
        });
      });

      // Calculate the total width of the table by adding up the width of all columns.
      let tableWidth = 0;
      if (rows.length > 0) {
        const row = rows[0] as Row<EvalTraceComparisonEntry>;
        const cells = row.getVisibleCells();
        cells.forEach((cell) => {
          colSizes[`--col-${escapeCssSpecialCharacters(cell.column.id)}-size`] = cell.column.getSize() + 'px';
          tableWidth += cell.column.getSize();
        });
      }

      return { columnSizeVars: colSizes, tableWidth: tableWidth + 'px' };
    }, [tableHeaderGroups, rows]);

    // Compute assessment aggregates.
    const assessmentNameToAggregates = useMemo(() => {
      const result: Record<string, AssessmentAggregates> = {};
      for (const assessmentInfo of selectedAssessmentInfos) {
        result[assessmentInfo.name] = getAssessmentAggregates(assessmentInfo, evaluations, assessmentFilters);
      }
      return result;
    }, [selectedAssessmentInfos, evaluations, assessmentFilters]);

    return (
      <>
        <div
          className="container"
          ref={tableContainerRef}
          css={{
            height: '100%',
            position: 'relative',
            overflowY: 'auto',
            overflowX: 'auto',
            minWidth: '100%',
            width: tableWidth,
          }}
        >
          <Table
            css={{
              width: '100%',
              ...columnSizeVars, // Define column sizes on the <table> element
            }}
            empty={isEmpty() ? emptyComponent : undefined}
            someRowsSelected={enableRowSelection ? someRowSelected || allRowSelected : undefined}
          >
            <GenAiTracesTableHeader
              headerGroups={table.getHeaderGroups()}
              enableRowSelection={enableRowSelection}
              enableGrouping={enableGrouping}
              selectedAssessmentInfos={selectedAssessmentInfos}
              assessmentNameToAggregates={assessmentNameToAggregates}
              assessmentFilters={assessmentFilters}
              toggleAssessmentFilter={toggleAssessmentFilter}
              runDisplayName={runDisplayName}
              compareToRunUuid={compareToRunUuid}
              compareToRunDisplayName={compareToRunDisplayName}
              disableAssessmentTooltips={disableAssessmentTooltips}
              collapsedHeader={collapsedHeader}
              setCollapsedHeader={setCollapsedHeader}
              isComparing={isComparing}
              allRowSelected={allRowSelected}
              someRowSelected={someRowSelected}
              toggleAllRowsSelectedHandler={table.getToggleAllRowsSelectedHandler}
              setColumnSizing={table.setColumnSizing}
            />

            <MemoizedGenAiTracesTableBodyRows
              rows={rows}
              isComparing={isComparing}
              enableRowSelection={enableRowSelection}
              virtualItems={virtualItems}
              virtualizerTotalSize={rowVirtualizer.getTotalSize()}
              virtualizerMeasureElement={rowVirtualizer.measureElement}
              rowSelectionState={rowSelection}
              selectedColumns={columnsToRender}
            />
          </Table>
        </div>
        {displayLoadingOverlay && (
          <div
            css={{
              position: 'absolute',
              inset: 0,
              backgroundColor: theme.colors.backgroundPrimary,
              opacity: 0.75,
              pointerEvents: 'none',
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
            }}
          >
            <Spinner size="large" />
          </div>
        )}
        {selectedEvaluationId && (
          <GenAiEvaluationTracesReviewModal
            experimentId={experimentId}
            runUuid={runUuid}
            runDisplayName={runDisplayName}
            otherRunDisplayName={compareToRunDisplayName}
            evaluations={rows.map((row) => row.original)}
            selectedEvaluationId={selectedEvaluationId}
            onChangeEvaluationId={onChangeEvaluationId}
            exportToEvalsInstanceEnabled={exportToEvalsInstanceEnabled}
            assessmentInfos={assessmentInfos}
            getTrace={getTrace}
            saveAssessmentsQuery={saveAssessmentsQuery}
          />
        )}
      </>
    );
  },
);
