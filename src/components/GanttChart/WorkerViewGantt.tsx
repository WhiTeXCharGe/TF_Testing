import { useMemo, useState } from 'react';
import { useAppContext } from '../../context/AppContext';
import { WorkerTimelineGrid, BarDragCommit } from './WorkerTimelineGrid';
import { buildWorkerTimelineModel } from './workerViewModel';
import { addDays, formatDate } from '../../utils/dateUtils';

type WorkerFilterKey = 'company' | 'name' | 'manager' | 'remarks';
type WorkerFilters = Record<WorkerFilterKey, Set<string>>;

interface Props {
  dates: string[];
}

function cloneFilters(src: WorkerFilters): WorkerFilters {
  return {
    company: new Set(src.company),
    name: new Set(src.name),
    manager: new Set(src.manager),
    remarks: new Set(src.remarks),
  };
}

const EMPTY_FILTERS: WorkerFilters = {
  company: new Set<string>(),
  name: new Set<string>(),
  manager: new Set<string>(),
  remarks: new Set<string>(),
};

export function WorkerViewGantt({ dates }: Props) {
  const { state, dispatch } = useAppContext();
  const { schedule, envConfig, selectedAssignmentIndex, violations } = state;

  const [filters, setFilters] = useState<WorkerFilters>(EMPTY_FILTERS);
  const [selectedDateForCellFilter, setSelectedDateForCellFilter] = useState('');
  const [selectedDateTasks, setSelectedDateTasks] = useState<Set<string>>(new Set());

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

  const filteredRows = useMemo(() => {
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

  const toggleMetaFilter = (key: WorkerFilterKey, value: string) => {
    setFilters(prev => {
      const next = cloneFilters(prev);
      if (next[key].has(value)) next[key].delete(value);
      else next[key].add(value);
      return next;
    });
  };

  const clearMetaFilter = (key: WorkerFilterKey) => {
    setFilters(prev => {
      const next = cloneFilters(prev);
      next[key].clear();
      return next;
    });
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
        company: [...filters.company],
        name: [...filters.name],
        manager: [...filters.manager],
        remarks: [...filters.remarks],
      }}
      metaFilterOptions={metaFilterOptions}
      onToggleMetaFilter={toggleMetaFilter}
      selectedDateForCellFilter={selectedDateForCellFilter}
      onSelectedDateForCellFilterChange={value => {
        setSelectedDateForCellFilter(prev => {
          if (prev !== value) setSelectedDateTasks(new Set());
          return value;
        });
      }}
      selectedDateTaskValues={[...selectedDateTasks]}
      selectedDateTaskOptions={selectedDateForCellFilter ? (model.dateWorkOptions[selectedDateForCellFilter] ?? []) : []}
      dateTaskOptionsByDate={model.dateWorkOptions}
      onToggleSelectedDateTask={value => {
        setSelectedDateTasks(prev => {
          const next = new Set(prev);
          if (next.has(value)) next.delete(value);
          else next.add(value);
          return next;
        });
      }}
      onClearMetaFilter={clearMetaFilter}
      onClearSelectedDateTask={() => setSelectedDateTasks(new Set())}
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
    />
  );
}
