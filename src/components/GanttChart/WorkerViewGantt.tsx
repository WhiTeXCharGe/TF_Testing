import { useMemo } from 'react';
import { useAppContext } from '../../context/AppContext';
import { WorkerTimelineGrid, BarDragCommit } from './WorkerTimelineGrid';
import { buildWorkerTimelineModel } from './workerViewModel';
import { addDays, formatDate } from '../../utils/dateUtils';

type WorkerFilterKey = 'company' | 'name' | 'manager' | 'remarks';

interface Props {
  dates: string[];
}

// Build lookup: workerId → { wfTaskIds, phaseIds, fabIds, regionIds }
function buildWorkerAssignmentIndex(
  schedule: import('../../types/schedule').ScheduleData,
  fabToRegion: Map<string, string>,
): Map<string, { wfTaskIds: Set<string>; phaseIds: Set<string>; fabIds: Set<string>; regionIds: Set<string> }> {
  const opToWf = new Map<string, string>();
  const opToPhase = new Map<string, string>();
  const wfFab = new Map<string, string>();
  const wfRegion = new Map<string, string>();

  for (const wt of schedule.workflowTaskList) {
    wfFab.set(wt.id, wt.fab ?? '');
    wfRegion.set(wt.id, wt.region ?? (wt.fab ? (fabToRegion.get(wt.fab) ?? '') : ''));
    for (const pt of wt.phaseTaskList) {
      for (const ot of pt.operationTaskList) {
        opToWf.set(ot.id, wt.id);
        opToPhase.set(ot.id, pt.phase);
      }
    }
  }

  const idx = new Map<string, { wfTaskIds: Set<string>; phaseIds: Set<string>; fabIds: Set<string>; regionIds: Set<string> }>();
  for (const a of schedule.assignmentList) {
    if (!idx.has(a.worker)) idx.set(a.worker, { wfTaskIds: new Set(), phaseIds: new Set(), fabIds: new Set(), regionIds: new Set() });
    const e = idx.get(a.worker)!;
    const wfId = opToWf.get(a.operationTask) ?? '';
    if (wfId) {
      e.wfTaskIds.add(wfId);
      const fab = wfFab.get(wfId) ?? '';
      const region = wfRegion.get(wfId) ?? '';
      if (fab) e.fabIds.add(fab);
      if (region) e.regionIds.add(region);
    }
    const phase = opToPhase.get(a.operationTask) ?? '';
    if (phase) e.phaseIds.add(phase);
  }
  return idx;
}

export function WorkerViewGantt({ dates }: Props) {
  const { state, dispatch } = useAppContext();
  const {
    schedule, envConfig, selectedAssignmentIndex, violations, workerViewFilter,
    workerColumnFilter, workerDateCellFilter,
  } = state;

  // Convert array-based context state to Sets for filter logic.
  const filters = useMemo(() => ({
    company: new Set(workerColumnFilter.company),
    name:    new Set(workerColumnFilter.name),
    manager: new Set(workerColumnFilter.manager),
    remarks: new Set(workerColumnFilter.remarks),
  }), [workerColumnFilter]);

  const selectedDateForCellFilter = workerDateCellFilter.date;
  const selectedDateTasks = useMemo(() => new Set(workerDateCellFilter.tasks), [workerDateCellFilter.tasks]);

  const dateIndex = useMemo(
    () => new Map(dates.map((d, i) => [d, i])),
    [dates],
  );

  const model = useMemo(() => {
    if (!schedule || !envConfig || dates.length === 0) {
      return { rows: [], monthGroups: [], dateWorkOptions: {} };
    }
    return buildWorkerTimelineModel(envConfig, schedule, dates, formatDate(new Date()));
  }, [schedule, envConfig, dates]);

  const violationIndices = useMemo(
    () => new Set(violations.flatMap(v => v.assignmentIndices)),
    [violations],
  );

  const metaFilterOptions = useMemo(() => {
    const getUnique = (key: WorkerFilterKey) => {
      const set = new Set<string>();
      model.rows.forEach(row => {
        const value = row.meta[key];
        if (value) set.add(value);
      });
      return [...set].sort((a, b) => a.localeCompare(b, 'ja'));
    };
    return {
      company: getUnique('company'),
      name: getUnique('name'),
      manager: getUnique('manager'),
      remarks: getUnique('remarks'),
    };
  }, [model.rows]);

  // Column-level filter (company/name/manager/remarks + cell date filter)
  const columnFilteredRows = useMemo(() => {
    if (model.rows.length === 0) return [];
    const dateCol = selectedDateForCellFilter ? dateIndex.get(selectedDateForCellFilter) : undefined;
    return model.rows.filter(row => {
      if (filters.company.size > 0 && !filters.company.has(row.meta.company)) return false;
      if (filters.name.size > 0 && !filters.name.has(row.meta.name)) return false;
      if (filters.manager.size > 0 && !filters.manager.has(row.meta.manager)) return false;
      if (filters.remarks.size > 0 && !filters.remarks.has(row.meta.remarks)) return false;
      if (selectedDateForCellFilter && selectedDateTasks.size > 0 && dateCol !== undefined) {
        const cell = row.dayCells[dateCol];
        if (!cell || cell.kind !== 'work' || !cell.moduleName || !selectedDateTasks.has(cell.moduleName)) return false;
      }
      return true;
    });
  }, [model.rows, dateIndex, filters, selectedDateForCellFilter, selectedDateTasks]);

  // Global top-bar filter (barName, moduleIds, phaseIds, fabIds, regionIds)
  const filteredRows = useMemo(() => {
    const { barName, moduleIds, phaseIds, fabIds, regionIds } = workerViewFilter;
    const noFilter = !barName && !moduleIds.length && !phaseIds.length && !fabIds.length && !regionIds.length;
    if (noFilter) return columnFilteredRows;

    const fabToRegion = new Map(envConfig?.fabList.map(f => [f.id, f.region ?? '']) ?? []);
    const workerIdx = schedule ? buildWorkerAssignmentIndex(schedule, fabToRegion) : new Map();

    return columnFilteredRows.filter(row => {
      if (barName) {
        const q = barName.toLowerCase();
        if (!row.segments.some(s => s.label.toLowerCase().includes(q))) return false;
      }
      const entry = workerIdx.get(row.workerId);
      if (moduleIds.length > 0) {
        if (!entry || !moduleIds.some(id => entry.wfTaskIds.has(id))) return false;
      }
      if (phaseIds.length > 0) {
        if (!entry || !phaseIds.some(id => entry.phaseIds.has(id))) return false;
      }
      if (fabIds.length > 0) {
        if (!entry || !fabIds.some(id => entry.fabIds.has(id))) return false;
      }
      if (regionIds.length > 0) {
        if (!entry || !regionIds.some(id => entry.regionIds.has(id))) return false;
      }
      return true;
    });
  }, [columnFilteredRows, workerViewFilter, schedule, envConfig]);

  // ── Highlight: assignments that specifically match the active global filter ──
  // null = no highlight mode (no global filter active).
  const highlightedAssignmentIndices = useMemo<Set<number> | null>(() => {
    const { barName, moduleIds, phaseIds, fabIds, regionIds } = workerViewFilter;
    const hasFilter = barName || moduleIds.length > 0 || phaseIds.length > 0 || fabIds.length > 0 || regionIds.length > 0;
    if (!hasFilter || !schedule) return null;

    const fabToRegion = new Map(envConfig?.fabList.map(f => [f.id, f.region ?? '']) ?? []);
    const opToWf    = new Map<string, string>();
    const opToPhase = new Map<string, string>();
    const wfFab     = new Map<string, string>();
    const wfRegion  = new Map<string, string>();

    for (const wt of schedule.workflowTaskList) {
      wfFab.set(wt.id, wt.fab ?? '');
      wfRegion.set(wt.id, wt.region ?? (wt.fab ? (fabToRegion.get(wt.fab) ?? '') : ''));
      for (const pt of wt.phaseTaskList) {
        for (const ot of pt.operationTaskList) {
          opToWf.set(ot.id, wt.id);
          opToPhase.set(ot.id, pt.phase);
        }
      }
    }

    const result = new Set<number>();
    schedule.assignmentList.forEach((a, idx) => {
      const wfId    = opToWf.get(a.operationTask) ?? '';
      const phaseId = opToPhase.get(a.operationTask) ?? '';
      const fabId   = wfFab.get(wfId) ?? '';
      const regionId = wfRegion.get(wfId) ?? '';
      if (moduleIds.length > 0  && !moduleIds.includes(wfId))    return;
      if (phaseIds.length > 0   && !phaseIds.includes(phaseId))  return;
      if (fabIds.length > 0     && !fabIds.includes(fabId))      return;
      if (regionIds.length > 0  && !regionIds.includes(regionId)) return;
      result.add(idx);
    });
    return result;
  }, [workerViewFilter, schedule, envConfig]);

  const highlightBarName = workerViewFilter.barName;

  // ── Column filter callbacks (now dispatch to context) ─────────────────────

  const toggleMetaFilter = (key: WorkerFilterKey, value: string) => {
    const current = workerColumnFilter[key];
    const next = current.includes(value)
      ? current.filter(v => v !== value)
      : [...current, value];
    dispatch({ type: 'SET_WORKER_COLUMN_FILTER', payload: { [key]: next } });
  };

  const clearMetaFilter = (key: WorkerFilterKey) => {
    dispatch({ type: 'SET_WORKER_COLUMN_FILTER', payload: { [key]: [] } });
  };

  const handleBarCommit = (commit: BarDragCommit) => {
    if (!schedule || dates.length === 0) return;
    const assignment = schedule.assignmentList[commit.assignmentIndex];
    if (!assignment) return;

    const assignmentStartIdx = dateIndex.get(assignment.startDate);
    const assignmentEndIdx = dateIndex.get(assignment.endDate);
    if (assignmentStartIdx === undefined || assignmentEndIdx === undefined) return;

    if (commit.mode === 'move') {
      // Use the segment's original position to compute delta — fixes split-bar drag bug
      const deltaDays = commit.newStartIndex - commit.originStartIndex;
      const newAssignmentStartIdx = Math.max(0, Math.min(dates.length - 1, assignmentStartIdx + deltaDays));
      const newAssignmentEndIdx = Math.max(0, Math.min(dates.length - 1, assignmentEndIdx + deltaDays));
      if (newAssignmentStartIdx > newAssignmentEndIdx) return;

      const shiftedWorkDateList = assignment.workDateList.map(wd => ({
        ...wd,
        date: addDays(wd.date, deltaDays),
      }));

      dispatch({
        type: 'UPDATE_ASSIGNMENT',
        payload: {
          index: commit.assignmentIndex,
          updates: {
            worker: commit.newWorkerId,
            startDate: dates[newAssignmentStartIdx],
            endDate: dates[newAssignmentEndIdx],
            workDateList: shiftedWorkDateList.length > 0 ? shiftedWorkDateList : [{ date: dates[newAssignmentStartIdx], hour: 8 }],
          },
        },
      });
    } else {
      // resize: update the boundary that was resized, keep the other from the assignment
      const newStartDate = commit.mode === 'resize-start' ? dates[commit.newStartIndex] : assignment.startDate;
      const newEndDate = commit.mode === 'resize-end' ? dates[commit.newEndIndex] : assignment.endDate;
      const newStartIdx = dateIndex.get(newStartDate) ?? 0;
      const newEndIdx = dateIndex.get(newEndDate) ?? dates.length - 1;
      if (newStartIdx > newEndIdx) return;

      const filteredWorkDateList = assignment.workDateList.filter(wd => {
        const idx = dateIndex.get(wd.date);
        return idx !== undefined && idx >= newStartIdx && idx <= newEndIdx;
      });

      dispatch({
        type: 'UPDATE_ASSIGNMENT',
        payload: {
          index: commit.assignmentIndex,
          updates: {
            worker: commit.newWorkerId,
            startDate: newStartDate,
            endDate: newEndDate,
            workDateList: filteredWorkDateList.length > 0 ? filteredWorkDateList : [{ date: newStartDate, hour: 8 }],
          },
        },
      });
    }

    dispatch({ type: 'SELECT_ASSIGNMENT', payload: commit.assignmentIndex });
  };

  return (
    <WorkerTimelineGrid
      dates={dates}
      rows={filteredRows}
      monthGroups={model.monthGroups}
      selectedAssignmentIndex={selectedAssignmentIndex}
      violationAssignmentIndices={violationIndices}
      onSelectAssignment={index => dispatch({ type: 'SELECT_ASSIGNMENT', payload: index })}
      onSelectUnavailable={(workerId, startDate, endDate) =>
        dispatch({ type: 'SELECT_UNAVAILABLE', payload: { workerId, startDate, endDate } })
      }
      onBarCommit={handleBarCommit}
      metaFilterValues={{
        company: workerColumnFilter.company,
        name:    workerColumnFilter.name,
        manager: workerColumnFilter.manager,
        remarks: workerColumnFilter.remarks,
      }}
      metaFilterOptions={metaFilterOptions}
      onToggleMetaFilter={toggleMetaFilter}
      selectedDateForCellFilter={selectedDateForCellFilter}
      onSelectedDateForCellFilterChange={date => {
        dispatch({
          type: 'SET_WORKER_DATE_CELL_FILTER',
          payload: { date, tasks: date !== selectedDateForCellFilter ? [] : workerDateCellFilter.tasks },
        });
      }}
      selectedDateTaskValues={[...selectedDateTasks]}
      selectedDateTaskOptions={selectedDateForCellFilter ? (model.dateWorkOptions[selectedDateForCellFilter] ?? []) : []}
      dateTaskOptionsByDate={model.dateWorkOptions}
      onToggleSelectedDateTask={value => {
        const current = workerDateCellFilter.tasks;
        const next = current.includes(value)
          ? current.filter(t => t !== value)
          : [...current, value];
        dispatch({ type: 'SET_WORKER_DATE_CELL_FILTER', payload: { date: selectedDateForCellFilter, tasks: next } });
      }}
      onClearMetaFilter={clearMetaFilter}
      onClearSelectedDateTask={() =>
        dispatch({ type: 'SET_WORKER_DATE_CELL_FILTER', payload: { date: selectedDateForCellFilter, tasks: [] } })
      }
      onChangeRemarks={(workerId, value) =>
        dispatch({ type: 'UPDATE_WORKER_DEFINITION', payload: { workerId, definition: value } })
      }
      onUnavailableDragCommit={(workerId, oldStartDate, oldEndDate, newStartDate, newEndDate) => {
        if (oldStartDate === oldEndDate && newStartDate === newEndDate) {
          dispatch({ type: 'MOVE_UNAVAILABLE_DATE', payload: { workerId, oldDate: oldStartDate, newDate: newStartDate } });
        } else {
          dispatch({ type: 'RESIZE_UNAVAILABLE_RANGE', payload: { workerId, oldStartDate, oldEndDate, newStartDate, newEndDate } });
        }
      }}
      highlightedAssignmentIndices={highlightedAssignmentIndices}
      highlightBarName={highlightBarName}
    />
  );
}
