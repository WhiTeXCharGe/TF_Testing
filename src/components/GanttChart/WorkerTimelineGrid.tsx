import { memo, useEffect, useMemo, useRef, useState } from 'react';
import { WorkerTimelineRow, HeaderMonthGroup, WorkerSegment, FlightStint } from './workerViewModel';
import { UI } from '../../config/uiText';
import { getRangeOverlayGeom } from '../../utils/dateUtils';

const HEADER_HEIGHT = 26;
const ROW_HEIGHT = 34;
const DATE_CELL_WIDTH = 22;
const PLAN_RANGE_BG = 'rgba(66, 165, 245, 0.14)';
const PLAN_RANGE_CELL_BG = '#cfe8fc';
const PLAN_RANGE_ARROW_COLOR = '#1e88e5';

type WorkerFilterKey = 'id' | 'company' | 'name' | 'manager' | 'remarks';

// Always-visible columns
const ALWAYS_COLUMNS: Array<{ key: WorkerFilterKey; label: string; width: number; align?: 'left' | 'center' }> = [
  { key: 'company', label: UI.workerGridCompany, width: 110, align: 'left' },
  { key: 'id', label: UI.workerGridId, width: 52, align: 'left' },
  { key: 'name', label: UI.workerGridName, width: 80, align: 'left' },
];

// Collapsible meta columns (責任者, 備考欄)
const COLLAPSIBLE_META: Array<{ key: WorkerFilterKey; label: string; width: number; align?: 'left' | 'center' }> = [
  { key: 'manager', label: UI.workerGridManager, width: 78, align: 'center' },
  { key: 'remarks', label: UI.workerGridRemarks, width: 82, align: 'left' },
];

type ExtraColKey = 'workType' | 'assignedDuties' | 'visa' | 'overseasDriving';
type ExtraDescField = '業務形態' | 'VISA' | '海外運転';
const EXTRA_COL_DESC_FIELD: Partial<Record<ExtraColKey, ExtraDescField>> = {
  workType: '業務形態',
  visa: 'VISA',
  overseasDriving: '海外運転',
};
const EXTRA_COLUMNS: Array<{ key: ExtraColKey; label: string; width: number }> = [
  { key: 'workType', label: UI.extraColWorkType, width: 96 },
  { key: 'assignedDuties', label: UI.extraColAssignedDuties, width: 140 },
  { key: 'visa', label: UI.extraColVisa, width: 70 },
  { key: 'overseasDriving', label: UI.extraColOverseasDriving, width: 92 },
];
const TOGGLE_COL_WIDTH = 22;

const DOW_JA = UI.dowLabels;

type DragMode = 'move' | 'resize-start' | 'resize-end';

interface DragState {
  assignmentIndex: number;
  mode: DragMode;
  originRow: number;
  originWorkerId: string;
  originStartIndex: number;
  originEndIndex: number;
  startX: number;
  startY: number;
}

interface DragPreview {
  assignmentIndex: number;
  workerId: string;
  startIndex: number;
  endIndex: number;
}

export interface BarDragCommit {
  assignmentIndex: number;
  newWorkerId: string;
  newStartIndex: number;
  newEndIndex: number;
  originStartIndex: number;
  originEndIndex: number;
  mode: DragMode;
}

interface UnavailDragState {
  workerId: string;
  originStartDate: string;
  originEndDate: string;
  originStartIndex: number;
  originEndIndex: number;
  mode: 'move' | 'resize-start' | 'resize-end';
  startX: number;
}

interface UnavailDragPreview {
  workerId: string;
  startIndex: number;
  endIndex: number;
}

interface Props {
  dates: string[];
  rows: WorkerTimelineRow[];
  monthGroups: HeaderMonthGroup[];
  planRangeStart: string;
  planRangeEnd: string;
  selectedAssignmentIndex: number | null;
  violationAssignmentIndices: Set<number>;
  onSelectAssignment: (index: number | null) => void;
  onSelectUnavailable: (workerId: string, startDate: string, endDate: string) => void;
  onBarCommit: (commit: BarDragCommit) => void;
  onUnavailableDragCommit: (workerId: string, oldStartDate: string, oldEndDate: string, newStartDate: string, newEndDate: string) => void;
  metaFilterValues: Record<WorkerFilterKey, string[]>;
  metaFilterOptions: Record<WorkerFilterKey, string[]>;
  onToggleMetaFilter: (key: WorkerFilterKey, value: string) => void;
  selectedDateForCellFilter: string;
  onSelectedDateForCellFilterChange: (date: string) => void;
  selectedDateTaskValues: string[];
  selectedDateTaskOptions: string[];
  dateTaskOptionsByDate: Record<string, string[]>;
  onToggleSelectedDateTask: (taskName: string) => void;
  onClearMetaFilter: (key: WorkerFilterKey) => void;
  onClearSelectedDateTask: () => void;
  onChangeRemarks: (workerId: string, value: string) => void;
  onChangeDescField: (workerId: string, field: ExtraDescField, value: string) => void;
  extraFilterValues: Record<ExtraColKey, string[]>;
  extraFilterOptions: Record<ExtraColKey, string[]>;
  onToggleExtraFilter: (key: ExtraColKey, value: string) => void;
  onClearExtraFilter: (key: ExtraColKey) => void;
  /** When non-null, only bars whose assignmentIndex is in this set are highlighted; others are dimmed. */
  highlightedAssignmentIndices: Set<number> | null;
  /** When non-empty, bars whose label includes this string are additionally highlighted. */
  highlightBarName: string;
  regionColorMap: Map<string, string>;
  regionNameMap: Map<string, string>;
  showFlightStints: boolean;
  scrollToSelectedAssignment: boolean;
  onClearScrollToAssignment: () => void;
}

export const WorkerTimelineGrid = memo(function WorkerTimelineGrid({
  dates,
  rows,
  monthGroups,
  planRangeStart,
  planRangeEnd,
  selectedAssignmentIndex,
  violationAssignmentIndices,
  onSelectAssignment,
  onSelectUnavailable,
  onBarCommit,
  metaFilterValues,
  metaFilterOptions,
  onToggleMetaFilter,
  selectedDateForCellFilter,
  onSelectedDateForCellFilterChange,
  selectedDateTaskValues,
  selectedDateTaskOptions,
  dateTaskOptionsByDate,
  onToggleSelectedDateTask,
  onClearMetaFilter,
  onClearSelectedDateTask,
  onChangeRemarks,
  onChangeDescField,
  onUnavailableDragCommit,
  highlightedAssignmentIndices,
  highlightBarName,
  extraFilterValues,
  extraFilterOptions,
  onToggleExtraFilter,
  onClearExtraFilter,
  regionColorMap,
  regionNameMap,
  showFlightStints,
  scrollToSelectedAssignment,
  onClearScrollToAssignment,
}: Props) {
  // highlight mode is active when any global filter (non-date) is set
  const highlightModeActive = highlightedAssignmentIndices !== null || !!highlightBarName;
  const [isExtraExpanded, setIsExtraExpanded] = useState(false);
  const leftBodyRef = useRef<HTMLDivElement>(null);
  const rightScrollRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<DragState | null>(null);
  const dragPreviewRef = useRef<DragPreview | null>(null);
  const [dragPreview, setDragPreview] = useState<DragPreview | null>(null);
  const unavailDragRef = useRef<UnavailDragState | null>(null);
  const unavailDragPreviewRef = useRef<UnavailDragPreview | null>(null);
  const [unavailDragPreview, setUnavailDragPreview] = useState<UnavailDragPreview | null>(null);

  // Scroll timeline to show the selected assignment only when triggered from constraint dialog
  useEffect(() => {
    if (!scrollToSelectedAssignment || selectedAssignmentIndex === null || !rightScrollRef.current) return;
    for (let rowIndex = 0; rowIndex < rows.length; rowIndex++) {
      const seg = rows[rowIndex].segments.find(s => s.assignmentIndex === selectedAssignmentIndex);
      if (!seg) continue;
      const scrollLeft = seg.startIndex * DATE_CELL_WIDTH - 120;
      const scrollTop  = rowIndex * ROW_HEIGHT - 80;
      rightScrollRef.current.scrollTo({ left: Math.max(0, scrollLeft), top: Math.max(0, scrollTop), behavior: 'smooth' });
      if (leftBodyRef.current) leftBodyRef.current.scrollTop = Math.max(0, scrollTop);
      break;
    }
    onClearScrollToAssignment();
  }, [scrollToSelectedAssignment, selectedAssignmentIndex]); // eslint-disable-line react-hooks/exhaustive-deps

  const totalMetaWidth = useMemo(
    () => ALWAYS_COLUMNS.reduce((acc, c) => acc + c.width, 0)
      + TOGGLE_COL_WIDTH
      + (isExtraExpanded
        ? COLLAPSIBLE_META.reduce((acc, c) => acc + c.width, 0)
          + EXTRA_COLUMNS.reduce((acc, c) => acc + c.width, 0)
        : 0),
    [isExtraExpanded],
  );
  const timelineWidth = dates.length * DATE_CELL_WIDTH;
  const planRangeGeom = useMemo(
    () => getRangeOverlayGeom(dates, planRangeStart, planRangeEnd, DATE_CELL_WIDTH),
    [dates, planRangeStart, planRangeEnd],
  );

  const onScroll = () => {
    if (!leftBodyRef.current || !rightScrollRef.current) return;
    leftBodyRef.current.scrollTop = rightScrollRef.current.scrollTop;
  };

  const onLeftBodyWheel = (e: React.WheelEvent<HTMLDivElement>) => {
    if (!rightScrollRef.current) return;
    rightScrollRef.current.scrollTop += e.deltaY;
    if (Math.abs(e.deltaX) > 0) {
      rightScrollRef.current.scrollLeft += e.deltaX;
    }
    e.preventDefault();
  };

  const startDrag = (
    e: React.MouseEvent<HTMLButtonElement>,
    segment: WorkerSegment,
    row: WorkerTimelineRow,
    rowIndex: number,
    mode: DragMode,
  ) => {
    if (segment.assignmentIndex === undefined) return;
    e.preventDefault();
    e.stopPropagation();

    dragRef.current = {
      assignmentIndex: segment.assignmentIndex,
      mode,
      originRow: rowIndex,
      originWorkerId: row.workerId,
      originStartIndex: segment.startIndex,
      originEndIndex: segment.endIndex,
      startX: e.clientX,
      startY: e.clientY,
    };

    const initialPreview: DragPreview = {
      assignmentIndex: segment.assignmentIndex,
      workerId: row.workerId,
      startIndex: segment.startIndex,
      endIndex: segment.endIndex,
    };
    setDragPreview(initialPreview);
    dragPreviewRef.current = initialPreview;

    const onMouseMove = (ev: MouseEvent) => {
      const drag = dragRef.current;
      if (!drag) return;

      const dx = ev.clientX - drag.startX;
      const dy = ev.clientY - drag.startY;
      const dayDelta = Math.round(dx / DATE_CELL_WIDTH);
      const rowDelta = Math.round(dy / ROW_HEIGHT);

      const lastIndex = dates.length - 1;
      let nextStart = drag.originStartIndex;
      let nextEnd = drag.originEndIndex;
      let nextWorker = drag.originWorkerId;

      if (drag.mode === 'move') {
        const span = drag.originEndIndex - drag.originStartIndex;
        const unclampedStart = drag.originStartIndex + dayDelta;
        const clampedStart = Math.max(0, Math.min(unclampedStart, Math.max(0, lastIndex - span)));
        nextStart = clampedStart;
        nextEnd = clampedStart + span;

        const targetRow = Math.max(0, Math.min(rows.length - 1, drag.originRow + rowDelta));
        nextWorker = rows[targetRow]?.workerId ?? drag.originWorkerId;
      } else if (drag.mode === 'resize-start') {
        const clamped = Math.max(0, Math.min(drag.originStartIndex + dayDelta, drag.originEndIndex));
        nextStart = clamped;
      } else {
        const clamped = Math.max(drag.originStartIndex, Math.min(drag.originEndIndex + dayDelta, lastIndex));
        nextEnd = clamped;
      }

      const preview: DragPreview = {
        assignmentIndex: drag.assignmentIndex,
        workerId: nextWorker,
        startIndex: nextStart,
        endIndex: nextEnd,
      };
      setDragPreview(preview);
      dragPreviewRef.current = preview;
    };

    const onMouseUp = () => {
      const drag = dragRef.current;
      const preview = dragPreviewRef.current;
      if (drag && preview) {
        const changed =
          preview.workerId !== drag.originWorkerId ||
          preview.startIndex !== drag.originStartIndex ||
          preview.endIndex !== drag.originEndIndex;

        if (changed) {
          onBarCommit({
            assignmentIndex: drag.assignmentIndex,
            newWorkerId: preview.workerId,
            newStartIndex: preview.startIndex,
            newEndIndex: preview.endIndex,
            originStartIndex: drag.originStartIndex,
            originEndIndex: drag.originEndIndex,
            mode: drag.mode,
          });
        }
      }

      dragRef.current = null;
      dragPreviewRef.current = null;
      setDragPreview(null);
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    };

    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
  };

  const startUnavailDrag = (
    e: React.MouseEvent<HTMLButtonElement>,
    workerId: string,
    segStartIndex: number,
    segEndIndex: number,
    mode: 'move' | 'resize-start' | 'resize-end',
  ) => {
    e.preventDefault();
    e.stopPropagation();

    const span = segEndIndex - segStartIndex;
    unavailDragRef.current = {
      workerId,
      originStartDate: dates[segStartIndex] ?? '',
      originEndDate: dates[segEndIndex] ?? '',
      originStartIndex: segStartIndex,
      originEndIndex: segEndIndex,
      mode,
      startX: e.clientX,
    };
    const initialPreview: UnavailDragPreview = { workerId, startIndex: segStartIndex, endIndex: segEndIndex };
    setUnavailDragPreview(initialPreview);
    unavailDragPreviewRef.current = initialPreview;

    const onMouseMove = (ev: MouseEvent) => {
      const drag = unavailDragRef.current;
      if (!drag) return;
      const dayDelta = Math.round((ev.clientX - drag.startX) / DATE_CELL_WIDTH);
      const last = dates.length - 1;
      let nextStart = drag.originStartIndex;
      let nextEnd = drag.originEndIndex;
      if (drag.mode === 'move') {
        const clamped = Math.max(0, Math.min(drag.originStartIndex + dayDelta, last - span));
        nextStart = clamped;
        nextEnd = clamped + span;
      } else if (drag.mode === 'resize-start') {
        nextStart = Math.max(0, Math.min(drag.originStartIndex + dayDelta, drag.originEndIndex));
      } else {
        nextEnd = Math.max(drag.originStartIndex, Math.min(drag.originEndIndex + dayDelta, last));
      }
      const preview: UnavailDragPreview = { workerId: drag.workerId, startIndex: nextStart, endIndex: nextEnd };
      setUnavailDragPreview(preview);
      unavailDragPreviewRef.current = preview;
    };

    const onMouseUp = () => {
      const drag = unavailDragRef.current;
      const preview = unavailDragPreviewRef.current;
      if (drag && preview && (preview.startIndex !== drag.originStartIndex || preview.endIndex !== drag.originEndIndex)) {
        const newStartDate = dates[preview.startIndex];
        const newEndDate = dates[preview.endIndex];
        if (newStartDate && newEndDate) {
          onUnavailableDragCommit(drag.workerId, drag.originStartDate, drag.originEndDate, newStartDate, newEndDate);
        }
      }
      unavailDragRef.current = null;
      unavailDragPreviewRef.current = null;
      setUnavailDragPreview(null);
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    };

    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
  };

  return (
    <div style={{ display: 'flex', flex: 1, minHeight: 0, overflow: 'hidden', borderTop: '1px solid #d4dde8' }}>
      <div
        style={{
          width: totalMetaWidth,
          flexShrink: 0,
          borderRight: '1px solid #c9d5e3',
          background: '#f7fafc',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
      >
        <div style={{ position: 'sticky', top: 0, zIndex: 5, background: '#f2f6fb' }}>
          <div style={{ display: 'flex', height: HEADER_HEIGHT * 3, borderBottom: '1px solid #c9d5e3' }}>
            {/* Always-visible columns */}
            {ALWAYS_COLUMNS.map(column => (
              <div
                key={column.key}
                style={{
                  width: column.width,
                  minWidth: column.width,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  gap: 4,
                  padding: '0 6px',
                  fontWeight: 700,
                  fontSize: 12,
                  color: '#1e334b',
                  borderRight: '1px solid #dde5ef',
                  whiteSpace: 'nowrap',
                }}
              >
                <span>{column.label}</span>
                <PopupMultiSelect
                  options={metaFilterOptions[column.key]}
                  selected={metaFilterValues[column.key]}
                  onToggle={value => onToggleMetaFilter(column.key, value)}
                  onClearAll={() => onClearMetaFilter(column.key)}
                />
              </div>
            ))}
            {/* Collapsible columns: 責任者, 備考欄, extra */}
            {isExtraExpanded && COLLAPSIBLE_META.map(column => (
              <div
                key={column.key}
                style={{
                  width: column.width,
                  minWidth: column.width,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  gap: 4,
                  padding: '0 6px',
                  fontWeight: 700,
                  fontSize: 12,
                  color: '#1e334b',
                  borderRight: '1px solid #dde5ef',
                  whiteSpace: 'nowrap',
                  background: '#eef4fb',
                }}
              >
                <span>{column.label}</span>
                <PopupMultiSelect
                  options={metaFilterOptions[column.key]}
                  selected={metaFilterValues[column.key]}
                  onToggle={value => onToggleMetaFilter(column.key, value)}
                  onClearAll={() => onClearMetaFilter(column.key)}
                />
              </div>
            ))}
            {isExtraExpanded && EXTRA_COLUMNS.map(col => (
              <div
                key={col.key}
                style={{
                  width: col.width,
                  minWidth: col.width,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  gap: 4,
                  padding: '0 6px',
                  fontWeight: 700,
                  fontSize: 12,
                  color: '#1e334b',
                  borderRight: '1px solid #dde5ef',
                  whiteSpace: 'nowrap',
                  background: '#eef4fb',
                }}
              >
                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{col.label}</span>
                <PopupMultiSelect
                  options={extraFilterOptions[col.key]}
                  selected={extraFilterValues[col.key]}
                  onToggle={value => onToggleExtraFilter(col.key, value)}
                  onClearAll={() => onClearExtraFilter(col.key)}
                />
              </div>
            ))}
            {/* Toggle button — always rightmost */}
            <div
              onClick={() => setIsExtraExpanded(v => !v)}
              title={isExtraExpanded ? UI.collapseColumnsTitle : UI.expandColumnsTitle}
              style={{
                width: TOGGLE_COL_WIDTH,
                minWidth: TOGGLE_COL_WIDTH,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                borderRight: '1px solid #dde5ef',
                cursor: 'pointer',
                userSelect: 'none',
                fontSize: 11,
                color: '#607d8b',
              }}
            >
              {isExtraExpanded ? '◀' : '▶'}
            </div>
          </div>
        </div>

        <div ref={leftBodyRef} onWheel={onLeftBodyWheel} style={{ flex: 1, overflow: 'hidden' }}>
          {rows.map((row, rowIndex) => (
            <div
              key={row.workerId}
              style={{
                display: 'flex',
                height: ROW_HEIGHT,
                background: rowIndex % 2 === 0 ? '#ffffff' : '#f9fbfd',
                borderBottom: '1px solid #ecf1f7',
              }}
            >
              {/* Always-visible cells */}
              {ALWAYS_COLUMNS.map(col => (
                <div
                  key={`${row.workerId}_${col.key}`}
                  style={{
                    width: col.width,
                    minWidth: col.width,
                    padding: col.align === 'center' ? 0 : '0 8px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: col.align === 'center' ? 'center' : 'flex-start',
                    overflow: 'hidden',
                    whiteSpace: 'nowrap',
                    borderRight: '1px solid #edf2f8',
                    color: '#25384f',
                    fontSize: 12,
                    fontFamily: 'Meiryo, sans-serif',
                  }}
                >
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                    title={row.meta[col.key]}>
                    {row.meta[col.key]}
                  </span>
                </div>
              ))}
              {/* Collapsible: 責任者, 備考欄 */}
              {isExtraExpanded && COLLAPSIBLE_META.map(col => (
                <div
                  key={`${row.workerId}_${col.key}`}
                  style={{
                    width: col.width,
                    minWidth: col.width,
                    padding: col.key === 'remarks' ? '0 4px' : col.align === 'center' ? 0 : '0 8px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: col.align === 'center' ? 'center' : 'flex-start',
                    overflow: 'hidden',
                    whiteSpace: 'nowrap',
                    borderRight: '1px solid #edf2f8',
                    color: '#25384f',
                    fontSize: 12,
                    fontFamily: 'Meiryo, sans-serif',
                  }}
                >
                  {col.key === 'remarks' ? (
                    <input
                      type="text"
                      defaultValue={row.meta.remarks}
                      onBlur={e => onChangeRemarks(row.workerId, e.target.value)}
                      onKeyDown={e => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur(); }}
                      onClick={e => e.stopPropagation()}
                      style={{
                        width: '100%',
                        border: 'none',
                        background: 'transparent',
                        fontSize: 12,
                        fontFamily: 'Meiryo, sans-serif',
                        color: '#25384f',
                        outline: 'none',
                        cursor: 'text',
                      }}
                      title={UI.editInPlaceTitle}
                    />
                  ) : (
                    <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                      title={row.meta[col.key]}>
                      {row.meta[col.key]}
                    </span>
                  )}
                </div>
              ))}
              {/* Collapsible: extra columns */}
              {isExtraExpanded && EXTRA_COLUMNS.map(col => {
                const descField = EXTRA_COL_DESC_FIELD[col.key];
                return (
                  <div
                    key={`${row.workerId}_extra_${col.key}`}
                    style={{
                      width: col.width,
                      minWidth: col.width,
                      padding: descField ? '0 4px' : '0 6px',
                      display: 'flex',
                      alignItems: 'center',
                      overflow: 'hidden',
                      whiteSpace: 'nowrap',
                      borderRight: '1px solid #edf2f8',
                      color: '#25384f',
                      fontSize: 11,
                      fontFamily: 'Meiryo, sans-serif',
                    }}
                  >
                    {descField ? (
                      <input
                        type="text"
                        defaultValue={row.meta[col.key]}
                        onBlur={e => onChangeDescField(row.workerId, descField, e.target.value)}
                        onKeyDown={e => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur(); }}
                        onClick={e => e.stopPropagation()}
                        style={{
                          width: '100%',
                          border: 'none',
                          background: 'transparent',
                          fontSize: 11,
                          fontFamily: 'Meiryo, sans-serif',
                          color: '#25384f',
                          outline: 'none',
                          cursor: 'text',
                        }}
                        title={UI.editInPlaceTitle}
                      />
                    ) : (
                      <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                        title={row.meta[col.key]}>
                        {row.meta[col.key]}
                      </span>
                    )}
                  </div>
                );
              })}
              {/* Toggle column body cell — always rightmost */}
              <div
                style={{
                  width: TOGGLE_COL_WIDTH,
                  minWidth: TOGGLE_COL_WIDTH,
                  borderRight: '1px solid #edf2f8',
                }}
              />
            </div>
          ))}
        </div>
      </div>

      <div
        ref={rightScrollRef}
        onScroll={onScroll}
        style={{
          flex: 1,
          minWidth: 0,
          minHeight: 0,
          height: '100%',
          maxHeight: '100%',
          overflowX: 'auto',
          overflowY: 'auto',
          overscrollBehavior: 'contain',
          WebkitOverflowScrolling: 'touch',
          touchAction: 'pan-x pan-y',
          scrollbarGutter: 'stable both-edges',
        }}
      >
        <div style={{ minWidth: timelineWidth, minHeight: HEADER_HEIGHT * 3 + rows.length * ROW_HEIGHT }}>
          <div style={{ position: 'sticky', top: 0, zIndex: 4, background: '#f4f8fc', borderBottom: '1px solid #c9d5e3' }}>
            <div style={{ display: 'flex', height: HEADER_HEIGHT }}>
              {monthGroups.map(group => (
                <div
                  key={`${group.label}_${group.startIndex}`}
                  style={{
                    width: group.span * DATE_CELL_WIDTH,
                    minWidth: group.span * DATE_CELL_WIDTH,
                    borderRight: '1px solid #d7e1ed',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: '#18324f',
                    fontWeight: 700,
                    fontSize: 12,
                  }}
                >
                  {group.label}
                </div>
              ))}
            </div>

            <div style={{ display: 'flex', height: HEADER_HEIGHT }}>
              {dates.map(date => {
                const [yyyy, mm, dd] = date.split('-');
                const inRange = date >= planRangeStart && date <= planRangeEnd;
                const isBoundary = date === planRangeStart || date === planRangeEnd;
                return (
                  <div
                    key={`d_${date}`}
                    title={`${yyyy}/${mm}/${dd}`}
                    style={{
                      width: DATE_CELL_WIDTH,
                      minWidth: DATE_CELL_WIDTH,
                      borderRight: '1px solid #e4ebf4',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: '#2a3f56',
                      fontSize: 10,
                      position: 'relative',
                      background: inRange ? PLAN_RANGE_CELL_BG : undefined,
                    }}
                  >
                    {isBoundary && (
                      <span style={{ position: 'absolute', top: -1, fontSize: 8, lineHeight: 1, color: PLAN_RANGE_ARROW_COLOR, pointerEvents: 'none' }}>▼</span>
                    )}
                    {Number(dd)}
                  </div>
                );
              })}
            </div>

            <div style={{ display: 'flex', height: HEADER_HEIGHT }}>
              {dates.map(date => {
                const dow = DOW_JA[new Date(`${date}T00:00:00`).getDay()] ?? '';
                const isWeekend = dow === '土' || dow === '日';
                const inRange = date >= planRangeStart && date <= planRangeEnd;
                const isDateFilterActive = selectedDateForCellFilter === date && selectedDateTaskValues.length > 0;
                return (
                  <div
                    key={`w_${date}`}
                    style={{
                      width: DATE_CELL_WIDTH,
                      minWidth: DATE_CELL_WIDTH,
                      borderRight: '1px solid #e4ebf4',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: isWeekend ? '#b54747' : '#2a3f56',
                      background: inRange ? PLAN_RANGE_CELL_BG : (isWeekend ? '#fff5f5' : '#f8fbff'),
                      fontSize: 10,
                      position: 'relative',
                    }}
                  >
                    {dow}
                    <div style={{ position: 'absolute', right: 1, top: 1 }}>
                      <PopupMultiSelect
                        options={dateTaskOptionsByDate[date] ?? []}
                        selected={selectedDateForCellFilter === date ? selectedDateTaskValues : []}
                        onOpen={() => onSelectedDateForCellFilterChange(date)}
                        onClearAll={() => {
                          onSelectedDateForCellFilterChange(date);
                          onClearSelectedDateTask();
                        }}
                        onToggle={task => {
                          if (selectedDateForCellFilter !== date) {
                            onSelectedDateForCellFilterChange(date);
                          }
                          onToggleSelectedDateTask(task);
                        }}
                        isActive={isDateFilterActive}
                        compact
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {rows.map((row, rowIndex) => (
            <div
              key={`timeline_${row.workerId}`}
              style={{
                position: 'relative',
                height: ROW_HEIGHT,
                borderBottom: '1px solid #ecf1f7',
                backgroundColor: rowIndex % 2 === 0 ? '#ffffff' : '#f9fbfd',
                backgroundImage: `repeating-linear-gradient(to right, transparent, transparent ${DATE_CELL_WIDTH - 1}px, #edf2f8 ${DATE_CELL_WIDTH - 1}px, #edf2f8 ${DATE_CELL_WIDTH}px)`,
              }}
            >
              {planRangeGeom && (
                <div style={{ position: 'absolute', left: planRangeGeom.left, top: 0, width: planRangeGeom.width, height: ROW_HEIGHT, background: PLAN_RANGE_BG, pointerEvents: 'none' }} />
              )}

              {row.segments.map(segment => {
                const width = Math.max(4, (segment.endIndex - segment.startIndex + 1) * DATE_CELL_WIDTH - 1);
                const left = segment.startIndex * DATE_CELL_WIDTH;
                const isSelected = segment.assignmentIndex === selectedAssignmentIndex;
                const hasViolation = segment.assignmentIndex !== undefined && violationAssignmentIndices.has(segment.assignmentIndex);
                const inDrag = dragPreview?.assignmentIndex === segment.assignmentIndex;

                // Highlight logic: a bar is highlighted when it matches the active global filter.
                const matchesAssignment = segment.assignmentIndex !== undefined
                  && (highlightedAssignmentIndices === null || highlightedAssignmentIndices.has(segment.assignmentIndex));
                const matchesBarName = !highlightBarName
                  || segment.label.toLowerCase().includes(highlightBarName.toLowerCase());
                const isHighlighted = highlightModeActive && matchesAssignment && matchesBarName && segment.kind !== 'unavailable';
                const isDimmed = false;

                return (
                  <button
                    key={`${row.workerId}_${segment.startIndex}_${segment.endIndex}_${segment.kind}_${segment.assignmentIndex ?? 'na'}`}
                    type="button"
                    onClick={e => {
                      e.stopPropagation();
                      if (segment.kind === 'unavailable') {
                        onSelectUnavailable(row.workerId, dates[segment.startIndex], dates[segment.endIndex]);
                        return;
                      }
                      if (segment.assignmentIndex === undefined) return;
                      onSelectAssignment(segment.assignmentIndex === selectedAssignmentIndex ? null : segment.assignmentIndex);
                    }}
                    onMouseDown={e => {
                      const target = e.target as HTMLElement;
                      const handle = target.closest('[data-handle]')?.getAttribute('data-handle');
                      if (segment.kind === 'unavailable') {
                        const umode = handle === 'start' ? 'resize-start' : handle === 'end' ? 'resize-end' : 'move';
                        startUnavailDrag(e, row.workerId, segment.startIndex, segment.endIndex, umode);
                        return;
                      }
                      const mode: DragMode = handle === 'start' ? 'resize-start' : handle === 'end' ? 'resize-end' : 'move';
                      startDrag(e, segment, row, rowIndex, mode);
                    }}
                    style={{
                      position: 'absolute',
                      left,
                      width,
                      top: 4,
                      height: ROW_HEIGHT - 8,
                      borderRadius: (segment.planFlexibility === 'Fixed' || segment.kind === 'unavailable') ? 0 : 8,
                      border: isSelected
                        ? '2px solid #145da0'
                        : hasViolation
                        ? '2px solid #c62828'
                        : isHighlighted
                        ? '2px solid #1565c0'
                        : '1px solid rgba(0,0,0,0.1)',
                      backgroundColor: segment.color,
                      color: segment.textColor,
                      padding: '0 6px',
                      textAlign: 'left',
                      whiteSpace: 'nowrap',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      fontSize: 10,
                      cursor: 'grab',
                      opacity: inDrag || (unavailDragPreview?.workerId === row.workerId && segment.kind === 'unavailable' && segment.startIndex >= (unavailDragPreview?.startIndex ?? -1) && segment.endIndex <= (unavailDragPreview?.endIndex ?? -1))
                        ? 0.3
                        : isDimmed ? 0.3 : 1,
                      boxShadow: isHighlighted
                        ? '0 0 0 2px rgba(21,101,192,0.45), 0 1px 3px rgba(0,0,0,0.2)'
                        : '0 1px 2px rgba(0,0,0,0.15)',
                    }}
                    title={segment.label}
                  >
                    <span
                      data-handle="start"
                      style={{
                        position: 'absolute',
                        left: 0,
                        top: 0,
                        bottom: 0,
                        width: 6,
                        cursor: 'w-resize',
                      }}
                    />
                    {segment.label}
                    <span
                      data-handle="end"
                      style={{
                        position: 'absolute',
                        right: 0,
                        top: 0,
                        bottom: 0,
                        width: 6,
                        cursor: 'e-resize',
                      }}
                    />
                  </button>
                );
              })}

              {dragPreview && dragPreview.workerId === row.workerId && (
                <div
                  style={{
                    position: 'absolute',
                    left: dragPreview.startIndex * DATE_CELL_WIDTH,
                    width: Math.max(4, (dragPreview.endIndex - dragPreview.startIndex + 1) * DATE_CELL_WIDTH - 1),
                    top: 4,
                    height: ROW_HEIGHT - 8,
                    borderRadius: 8,
                    border: '2px dashed #1565c0',
                    background: 'rgba(21, 101, 192, 0.15)',
                    pointerEvents: 'none',
                  }}
                />
              )}
              {unavailDragPreview && unavailDragPreview.workerId === row.workerId && (
                <div
                  style={{
                    position: 'absolute',
                    left: unavailDragPreview.startIndex * DATE_CELL_WIDTH,
                    width: Math.max(4, (unavailDragPreview.endIndex - unavailDragPreview.startIndex + 1) * DATE_CELL_WIDTH - 1),
                    top: 4,
                    height: ROW_HEIGHT - 8,
                    borderRadius: 8,
                    border: '2px dashed #888',
                    background: 'rgba(150,150,150,0.25)',
                    pointerEvents: 'none',
                  }}
                />
              )}

              {showFlightStints && row.flightStints.map((stint, si) => (
                <FlightStintBar
                  key={`${row.workerId}_stint_${si}`}
                  stint={stint}
                  dates={dates}
                  color={regionColorMap.get(stint.regionId) ?? '#90A4AE'}
                  regionName={regionNameMap.get(stint.regionId) ?? stint.regionId}
                />
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
});

function FlightStintBar({
  stint,
  dates,
  color,
  regionName,
}: {
  stint: FlightStint;
  dates: string[];
  color: string;
  regionName: string;
}) {
  const [hovered, setHovered] = useState(false);
  const lastDateInView = dates[dates.length - 1] ?? '';
  const firstDateInView = dates[0] ?? '';

  const fiDate = stint.flightInDate < firstDateInView ? firstDateInView : stint.flightInDate;
  const foDate = stint.flightOutDate > lastDateInView ? lastDateInView : stint.flightOutDate;

  const fiIdx = dates.indexOf(fiDate);
  const foIdx = dates.indexOf(foDate);

  // If both dates are outside the visible range, don't render
  if (fiIdx === -1 && foIdx === -1) return null;

  const startIdx = fiIdx !== -1 ? fiIdx : 0;
  const endIdx = foIdx !== -1 ? foIdx : dates.length - 1;

  const left = startIdx * DATE_CELL_WIDTH;
  const width = (endIdx - startIdx + 1) * DATE_CELL_WIDTH;

  const showFI = stint.flightInDate >= firstDateInView && fiIdx !== -1;
  const showFO = stint.flightOutDate <= lastDateInView && foIdx !== -1;
  const sameDate = stint.flightInDate === stint.flightOutDate;

  const tooltipText = `${regionName}  ${stint.flightInDate} → ${stint.flightOutDate}`;

  return (
    <div
      style={{
        position: 'absolute',
        left,
        width,
        bottom: 1,
        height: 6,
        borderRadius: 3,
        backgroundColor: color,
        opacity: 0.85,
        pointerEvents: 'auto',
        cursor: 'default',
        zIndex: 1,
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      {showFI && (
        <span style={{
          position: 'absolute',
          left: 1,
          top: -13,
          fontSize: 8,
          fontWeight: 700,
          color,
          whiteSpace: 'nowrap',
          pointerEvents: 'none',
          lineHeight: 1,
        }}>
          {sameDate ? 'FI/FO' : 'FI'}
        </span>
      )}
      {showFO && !sameDate && (
        <span style={{
          position: 'absolute',
          right: 1,
          top: -13,
          fontSize: 8,
          fontWeight: 700,
          color,
          whiteSpace: 'nowrap',
          pointerEvents: 'none',
          lineHeight: 1,
        }}>
          FO
        </span>
      )}
      {hovered && (
        <div style={{
          position: 'absolute',
          bottom: 10,
          left: '50%',
          transform: 'translateX(-50%)',
          background: 'rgba(30,40,55,0.92)',
          color: '#fff',
          fontSize: 10,
          padding: '3px 7px',
          borderRadius: 4,
          whiteSpace: 'nowrap',
          pointerEvents: 'none',
          zIndex: 100,
          boxShadow: '0 2px 6px rgba(0,0,0,0.3)',
        }}>
          {tooltipText}
        </div>
      )}
    </div>
  );
}

function PopupMultiSelect({
  options,
  selected,
  onToggle,
  onClearAll,
  onOpen,
  compact = false,
  isActive,
}: {
  options: string[];
  selected: string[];
  onToggle: (value: string) => void;
  onClearAll?: () => void;
  onOpen?: () => void;
  compact?: boolean;
  isActive?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState('');
  const buttonRef = useRef<HTMLButtonElement>(null);
  const popupRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const [popupPos, setPopupPos] = useState<{ left: number; top: number }>({ left: 0, top: 0 });
  const selectedSet = useMemo(() => new Set(selected), [selected]);

  const filteredOptions = useMemo(() => {
    if (!search.trim()) return options;
    const q = search.toLowerCase();
    return options.filter(o => o.toLowerCase().includes(q));
  }, [options, search]);

  useEffect(() => {
    if (!open || !buttonRef.current) return;

    const rect = buttonRef.current.getBoundingClientRect();
    const popupWidth = compact ? 200 : 240;
    const popupHeight = 320;

    let left = rect.left;
    let top = rect.bottom + 4;

    if (left + popupWidth > window.innerWidth - 8) left = window.innerWidth - popupWidth - 8;
    if (left < 8) left = 8;
    if (top + popupHeight > window.innerHeight - 8) top = Math.max(8, rect.top - popupHeight - 4);

    setPopupPos({ left, top });
    setTimeout(() => searchRef.current?.focus(), 10);

    const onDocClick = (ev: MouseEvent) => {
      const t = ev.target as Node;
      if (!popupRef.current?.contains(t) && !buttonRef.current?.contains(t)) {
        setOpen(false);
        setSearch('');
      }
    };

    document.addEventListener('mousedown', onDocClick);
    return () => document.removeEventListener('mousedown', onDocClick);
  }, [open, compact]);

  const filterActive = isActive || (selected?.length ?? 0) > 0;

  return (
    <>
      <button
        ref={buttonRef}
        type="button"
        onClick={() => {
          const next = !open;
          setOpen(next);
          if (next) onOpen?.();
        }}
        style={{
          border: 'none',
          background: 'transparent',
          borderRadius: 3,
          cursor: 'pointer',
          fontSize: compact ? 9 : 11,
          lineHeight: 1,
          padding: compact ? '0 2px' : '1px 4px',
          color: '#1e334b',
          fontWeight: 400,
          display: 'inline-flex',
          alignItems: 'center',
          gap: 2,
        }}
        title={UI.filterTitle}
      >
        <span style={{ display: 'inline-block', transform: filterActive ? 'rotate(180deg)' : 'none' }}>▾</span>
      </button>
      {open && (
        <div
          ref={popupRef}
          style={{
            position: 'fixed',
            left: popupPos.left,
            top: popupPos.top,
            zIndex: 1000,
            width: compact ? 200 : 240,
            border: '1px solid #c6d3e2',
            borderRadius: 6,
            background: '#ffffff',
            boxShadow: '0 8px 18px rgba(0,0,0,0.12)',
            overflow: 'hidden',
          }}
        >
          <div style={{ padding: '6px 8px', borderBottom: '1px solid #eee' }}>
            <input
              ref={searchRef}
              type="text"
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder={UI.filterSearchPlaceholder}
              style={{
                width: '100%', padding: '3px 6px', border: '1px solid #c0d0e0',
                borderRadius: 3, fontSize: 11, fontFamily: 'Meiryo, sans-serif',
                boxSizing: 'border-box', outline: 'none',
              }}
            />
          </div>
          <div style={{ padding: '6px 8px', borderBottom: '1px solid #eee' }}>
            <button
              type="button"
              onClick={() => onClearAll?.()}
              style={{
                width: '100%', textAlign: 'left', border: 'none', background: '#f5f8fc',
                color: '#28405a', padding: '3px 6px', borderRadius: 4, cursor: 'pointer', fontSize: 11,
              }}
            >
              {UI.filterClearAll}
            </button>
          </div>
          <div style={{ maxHeight: 220, overflowY: 'auto', padding: '4px 8px' }}>
            {filteredOptions.length === 0 && <div style={{ fontSize: 11, color: '#8a9aac', padding: '4px 0' }}>{UI.filterNoValues}</div>}
            {filteredOptions.map(option => (
              <label
                key={option}
                style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3, fontSize: 11, color: '#30455e', cursor: 'pointer' }}
              >
                <input type="checkbox" checked={selectedSet.has(option)} onChange={() => onToggle(option)} />
                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{option}</span>
              </label>
            ))}
          </div>
        </div>
      )}
    </>
  );
}