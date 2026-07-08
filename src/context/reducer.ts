import { AppState, ActionType, DEFAULT_WORKER_VIEW_FILTER, DEFAULT_MODULE_VIEW_FILTER, DEFAULT_WORKER_COLUMN_FILTER } from '../types/appState';
import { ScheduleData, PlanFlexibility } from '../types/schedule';
import { EnvConfig } from '../types/envConfig';
import { MAX_UNDO_STACK } from '../config/appConfig';
import { generateDateRange } from '../utils/dateUtils';

function mergeById<T extends { id: string }>(existing: T[], incoming: T[]): T[] {
  const existingIds = new Set(existing.map(x => x.id));
  return [...existing, ...incoming.filter(x => !existingIds.has(x.id))];
}

function pushUndo(state: AppState): ScheduleData[] {
  if (!state.schedule) return state.undoStack;
  return [...state.undoStack, state.schedule].slice(-MAX_UNDO_STACK);
}

const DOW_NAMES = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];

function getWeekdayName(dateStr: string): string {
  const d = new Date(`${dateStr}T00:00:00`);
  return DOW_NAMES[d.getDay()] ?? '';
}

function isWeeklyDate(worker: { unavailableDates: Array<{ weekly?: { weekdays: string[] }; single?: { days: string[] } }> }, date: string): boolean {
  const targetDay = getWeekdayName(date).toLowerCase();
  return worker.unavailableDates.some(e =>
    e.weekly?.weekdays.some(wd => wd.trim().toLowerCase() === targetDay || wd.trim().toLowerCase().startsWith(targetDay.slice(0, 3))),
  );
}

function removeWeekdayFromWorker<T extends { unavailableDates: Array<{ weekly?: { weekdays: string[] }; single?: { days: string[] } }> }>(worker: T, date: string): T {
  const targetDay = getWeekdayName(date).toLowerCase();
  const newDates = worker.unavailableDates
    .map(entry => {
      if (!entry.weekly) return entry;
      const filtered = entry.weekly.weekdays.filter(wd => {
        const norm = wd.trim().toLowerCase();
        return norm !== targetDay && !norm.startsWith(targetDay.slice(0, 3));
      });
      if (filtered.length === entry.weekly.weekdays.length) return entry;
      return filtered.length > 0 ? { ...entry, weekly: { weekdays: filtered } } : null;
    })
    .filter((e): e is NonNullable<typeof e> => e !== null);
  return { ...worker, unavailableDates: newDates };
}

export function reducer(state: AppState, action: ActionType): AppState {
  switch (action.type) {

    case 'LOAD_FILES':
      return {
        ...state,
        envConfig: action.payload.envConfig,
        schedule: action.payload.schedule,
        currentEnvPath: action.payload.envPath,
        currentSchedulePath: action.payload.schedulePath,
        undoStack: [],
        redoStack: [],
        violations: [],
        selectedAssignmentIndex: null,
        workerViewFilter: { ...DEFAULT_WORKER_VIEW_FILTER },
        moduleViewFilter: { ...DEFAULT_MODULE_VIEW_FILTER },
        workerColumnFilter: { ...DEFAULT_WORKER_COLUMN_FILTER },
        workerDateCellFilter: { date: '', tasks: [] },
      };

    case 'SET_SCHEDULE':
      return {
        ...state,
        schedule: action.payload,
        undoStack: pushUndo(state),
        redoStack: [],
      };

    case 'SWITCH_VIEW':
      return { ...state, currentView: action.payload, selectedAssignmentIndex: null };

    case 'SET_VIOLATIONS':
      return { ...state, violations: action.payload };

    case 'UNDO': {
      if (state.undoStack.length === 0 || !state.schedule) return state;
      const prev = state.undoStack[state.undoStack.length - 1];
      return {
        ...state,
        schedule: prev,
        undoStack: state.undoStack.slice(0, -1),
        redoStack: [...state.redoStack, state.schedule].slice(-MAX_UNDO_STACK),
      };
    }

    case 'REDO': {
      if (state.redoStack.length === 0) return state;
      const next = state.redoStack[state.redoStack.length - 1];
      return {
        ...state,
        schedule: next,
        undoStack: state.schedule ? [...state.undoStack, state.schedule].slice(-MAX_UNDO_STACK) : state.undoStack,
        redoStack: state.redoStack.slice(0, -1),
      };
    }

    case 'SELECT_ASSIGNMENT':
      return { ...state, selectedAssignmentIndex: action.payload, selectedUnavailableInfo: null };

    case 'SELECT_UNAVAILABLE':
      return {
        ...state,
        selectedUnavailableInfo: action.payload
          ? { workerId: action.payload.workerId, startDate: action.payload.startDate, endDate: action.payload.endDate }
          : null,
        selectedAssignmentIndex: null,
      };

    case 'DELETE_UNAVAILABLE_DATE': {
      if (!state.envConfig) return state;
      const { workerId, date } = action.payload;
      const workerList = state.envConfig.workerList.map(w => {
        if (w.id !== workerId) return w;
        if (isWeeklyDate(w, date)) {
          return removeWeekdayFromWorker(w, date);
        }
        const newDates = w.unavailableDates
          .map(entry => {
            if (!entry.single) return entry;
            const filtered = entry.single.days.filter(d => d.replace(/\//g, '-') !== date);
            return filtered.length > 0 ? { ...entry, single: { days: filtered } } : null;
          })
          .filter((e): e is NonNullable<typeof e> => e !== null);
        return { ...w, unavailableDates: newDates };
      });
      return { ...state, envConfig: { ...state.envConfig, workerList }, selectedUnavailableInfo: null };
    }

    case 'DELETE_UNAVAILABLE_RANGE': {
      if (!state.envConfig) return state;
      const { workerId, startDate, endDate } = action.payload;
      const rangeDates = new Set(generateDateRange(startDate, endDate));
      const workerList = state.envConfig.workerList.map(w => {
        if (w.id !== workerId) return w;
        const newDates = w.unavailableDates
          .map(entry => {
            if (entry.weekly) {
              const filtered = entry.weekly.weekdays.filter(wd => {
                const key = wd.trim().toLowerCase();
                const dowMap: Record<string, number> = { sun: 0, sunday: 0, mon: 1, monday: 1, tue: 2, tuesday: 2, wed: 3, wednesday: 3, thu: 4, thursday: 4, fri: 5, friday: 5, sat: 6, saturday: 6 };
                const dow = dowMap[key] ?? dowMap[key.slice(0, 3)] ?? -1;
                if (dow < 0) return true;
                return ![...rangeDates].some(d => new Date(`${d}T00:00:00`).getDay() === dow);
              });
              return filtered.length > 0 ? { ...entry, weekly: { weekdays: filtered } } : null;
            }
            if (entry.single) {
              const filtered = entry.single.days.filter(d => !rangeDates.has(d.replace(/\//g, '-')));
              return filtered.length > 0 ? { ...entry, single: { days: filtered } } : null;
            }
            return entry;
          })
          .filter((e): e is NonNullable<typeof e> => e !== null);
        return { ...w, unavailableDates: newDates };
      });
      return { ...state, envConfig: { ...state.envConfig, workerList }, selectedUnavailableInfo: null };
    }

    case 'MOVE_UNAVAILABLE_DATE': {
      if (!state.envConfig) return state;
      const { workerId, oldDate, newDate } = action.payload;
      if (oldDate === newDate) return state;
      const workerList = state.envConfig.workerList.map(w => {
        if (w.id !== workerId) return w;
        // Remove oldDate from either weekly or single
        let removed: typeof w.unavailableDates;
        if (isWeeklyDate(w, oldDate)) {
          removed = removeWeekdayFromWorker(w, oldDate).unavailableDates;
        } else {
          removed = w.unavailableDates
            .map(entry => {
              if (!entry.single) return entry;
              const filtered = entry.single.days.filter(d => d.replace(/\//g, '-') !== oldDate);
              return filtered.length > 0 ? { ...entry, single: { days: filtered } } : null;
            })
            .filter((e): e is NonNullable<typeof e> => e !== null);
        }
        // Add newDate as a single entry
        const existingSingle = removed.find(e => e.single);
        if (existingSingle?.single) {
          return {
            ...w,
            unavailableDates: removed.map(e =>
              e === existingSingle ? { ...e, single: { days: [...(e.single?.days ?? []), newDate] } } : e,
            ),
          };
        }
        return { ...w, unavailableDates: [...removed, { single: { days: [newDate] } }] };
      });
      return {
        ...state,
        envConfig: { ...state.envConfig, workerList },
        selectedUnavailableInfo: { workerId, startDate: newDate, endDate: newDate },
      };
    }

    case 'UPDATE_WORKER_DEFINITION': {
      if (!state.envConfig) return state;
      const { workerId, definition } = action.payload;
      const workerList = state.envConfig.workerList.map(w => {
        if (w.id !== workerId) return w;
        return { ...w, description: { ...w.description, '備考': definition } };
      });
      return { ...state, envConfig: { ...state.envConfig, workerList } };
    }

    case 'UPDATE_WORKFLOW_TASK_COLOR': {
      if (!state.schedule) return state;
      const { workflowTaskId, colorCode } = action.payload;
      const workflowTaskList = state.schedule.workflowTaskList.map(wt =>
        wt.id === workflowTaskId ? { ...wt, colorCode } : wt,
      );
      const newSchedule: ScheduleData = { ...state.schedule, workflowTaskList };
      return { ...state, schedule: newSchedule, undoStack: pushUndo(state), redoStack: [] };
    }

    case 'UPDATE_OPERATION_TASK_COLOR': {
      if (!state.schedule) return state;
      const { operationTaskId, colorCode } = action.payload;
      const workflowTaskList = state.schedule.workflowTaskList.map(wt => ({
        ...wt,
        phaseTaskList: wt.phaseTaskList.map(pt => ({
          ...pt,
          operationTaskList: pt.operationTaskList.map(ot =>
            ot.id === operationTaskId ? { ...ot, colorCode } : ot,
          ),
        })),
      }));
      const newSchedule: ScheduleData = { ...state.schedule, workflowTaskList };
      return { ...state, schedule: newSchedule, undoStack: pushUndo(state), redoStack: [] };
    }

    case 'TOGGLE_DEVICE': {
      const newExpanded = new Set(state.expandedDeviceIds);
      if (newExpanded.has(action.payload)) newExpanded.delete(action.payload);
      else newExpanded.add(action.payload);
      return { ...state, expandedDeviceIds: newExpanded };
    }

    case 'SET_WORKER_VIEW_FILTER':
      return { ...state, workerViewFilter: { ...state.workerViewFilter, ...action.payload } };

    case 'SET_MODULE_VIEW_FILTER':
      return { ...state, moduleViewFilter: { ...state.moduleViewFilter, ...action.payload } };

    case 'SET_WORKER_COLUMN_FILTER':
      return { ...state, workerColumnFilter: { ...state.workerColumnFilter, ...action.payload } };

    case 'SET_WORKER_DATE_CELL_FILTER':
      return { ...state, workerDateCellFilter: action.payload };

    case 'CLEAR_ALL_WORKER_FILTERS':
      return {
        ...state,
        workerViewFilter: { ...DEFAULT_WORKER_VIEW_FILTER },
        moduleViewFilter: { ...DEFAULT_MODULE_VIEW_FILTER },
        workerColumnFilter: { ...DEFAULT_WORKER_COLUMN_FILTER },
        workerDateCellFilter: { date: '', tasks: [] },
      };

    case 'ADD_ASSIGNMENT': {
      if (!state.schedule) return state;
      const newSchedule: ScheduleData = {
        ...state.schedule,
        assignmentList: [...state.schedule.assignmentList, action.payload],
      };
      return { ...state, schedule: newSchedule, undoStack: pushUndo(state), redoStack: [] };
    }

    case 'UPDATE_ASSIGNMENT': {
      if (!state.schedule) return state;
      const { index, updates } = action.payload;
      if (index < 0 || index >= state.schedule.assignmentList.length) return state;
      const newList = [...state.schedule.assignmentList];
      newList[index] = { ...newList[index], ...updates };
      const newSchedule: ScheduleData = { ...state.schedule, assignmentList: newList };
      return { ...state, schedule: newSchedule, undoStack: pushUndo(state), redoStack: [] };
    }

    case 'UPDATE_OPERATION_TASK': {
      if (!state.schedule) return state;
      const { workflowTaskId, phaseTaskId, operationTaskId, updates } = action.payload;
      const workflowTaskList = state.schedule.workflowTaskList.map(wt => {
        if (wt.id !== workflowTaskId) return wt;
        return {
          ...wt,
          phaseTaskList: wt.phaseTaskList.map(pt => {
            if (pt.id !== phaseTaskId) return pt;
            return {
              ...pt,
              operationTaskList: pt.operationTaskList.map(ot =>
                ot.id === operationTaskId ? { ...ot, ...updates } : ot,
              ),
            };
          }),
        };
      });
      const newSchedule: ScheduleData = { ...state.schedule, workflowTaskList };
      return { ...state, schedule: newSchedule, undoStack: pushUndo(state), redoStack: [] };
    }

    case 'UPDATE_PHASE_TASK': {
      if (!state.schedule) return state;
      const { workflowTaskId, phaseTaskId, updates } = action.payload;
      const workflowTaskList = state.schedule.workflowTaskList.map(wt => {
        if (wt.id !== workflowTaskId) return wt;
        return {
          ...wt,
          phaseTaskList: wt.phaseTaskList.map(pt =>
            pt.id === phaseTaskId ? { ...pt, ...updates } : pt,
          ),
        };
      });
      const newSchedule: ScheduleData = { ...state.schedule, workflowTaskList };
      return { ...state, schedule: newSchedule, undoStack: pushUndo(state), redoStack: [] };
    }

    case 'DELETE_ASSIGNMENT': {
      if (!state.schedule) return state;
      const newList = state.schedule.assignmentList.filter((_, i) => i !== action.payload);
      const newSchedule: ScheduleData = { ...state.schedule, assignmentList: newList };
      return {
        ...state,
        schedule: newSchedule,
        undoStack: pushUndo(state),
        redoStack: [],
        selectedAssignmentIndex: null,
      };
    }

    case 'BULK_UPDATE_FLEXIBILITY': {
      if (!state.schedule) return state;
      const { flexibility, target, targetDate } = action.payload;
      const flex = flexibility as PlanFlexibility;
      const newList = state.schedule.assignmentList.map((a, i) => {
        if (target === 'selected' && i !== state.selectedAssignmentIndex) return a;
        if (targetDate && a.startDate > targetDate) return a;
        return { ...a, planFlexibility: flex };
      });
      const newSchedule: ScheduleData = { ...state.schedule, assignmentList: newList };
      return { ...state, schedule: newSchedule, undoStack: pushUndo(state), redoStack: [] };
    }

    case 'SET_ERROR':
      return { ...state, errorMessage: action.payload };

    case 'OPEN_TASK_ADD_DIALOG':
      return { ...state, isTaskAddDialogOpen: true };

    case 'CLOSE_TASK_ADD_DIALOG':
      return { ...state, isTaskAddDialogOpen: false };

    case 'OPEN_FILE_DIALOG':
      return { ...state, isFileOpenDialogOpen: true };

    case 'CLOSE_FILE_DIALOG':
      return { ...state, isFileOpenDialogOpen: false };

    case 'OPEN_NEW_SCHEDULE_DIALOG':
      return { ...state, isNewScheduleDialogOpen: true };

    case 'CLOSE_NEW_SCHEDULE_DIALOG':
      return { ...state, isNewScheduleDialogOpen: false };

    case 'ADD_WORKFLOW_TASKS': {
      if (!state.schedule) return state;
      const existingIds = new Set(state.schedule.workflowTaskList.map(wt => wt.id));
      const toAdd = action.payload.filter(wt => !existingIds.has(wt.id));
      if (toAdd.length === 0) return state;
      const newSchedule: ScheduleData = {
        ...state.schedule,
        workflowTaskList: [...state.schedule.workflowTaskList, ...toAdd],
      };
      return { ...state, schedule: newSchedule, undoStack: pushUndo(state), redoStack: [] };
    }

    case 'MERGE_DATA': {
      const { schedule: incomingSched, envConfig: incomingEnv } = action.payload;
      let newSchedule = state.schedule;
      let newEnvConfig = state.envConfig;

      if (incomingSched && state.schedule) {
        const existingWtIds = new Set(state.schedule.workflowTaskList.map(wt => wt.id));
        const newWts = incomingSched.workflowTaskList.filter(wt => !existingWtIds.has(wt.id));
        newSchedule = {
          ...state.schedule,
          workflowTaskList: [...state.schedule.workflowTaskList, ...newWts],
          assignmentList: [...state.schedule.assignmentList, ...incomingSched.assignmentList],
        };
      }

      if (incomingEnv && state.envConfig) {
        newEnvConfig = {
          workflowList: mergeById(state.envConfig.workflowList, incomingEnv.workflowList),
          fabList: mergeById(state.envConfig.fabList, incomingEnv.fabList),
          regionList: mergeById(state.envConfig.regionList, incomingEnv.regionList),
          customerCompanyList: mergeById(state.envConfig.customerCompanyList, incomingEnv.customerCompanyList),
          workerCompanyList: mergeById(state.envConfig.workerCompanyList, incomingEnv.workerCompanyList),
          workerList: mergeById(state.envConfig.workerList, incomingEnv.workerList),
          transiteDayMap: state.envConfig.transiteDayMap,
        };
      }

      const scheduleChanged = newSchedule !== state.schedule;
      return {
        ...state,
        schedule: newSchedule,
        envConfig: newEnvConfig,
        undoStack: scheduleChanged ? pushUndo(state) : state.undoStack,
        redoStack: scheduleChanged ? [] : state.redoStack,
      };
    }

    case 'SAVE_PATHS':
      return {
        ...state,
        currentEnvPath: action.payload.envPath ?? state.currentEnvPath,
        currentSchedulePath: action.payload.schedulePath ?? state.currentSchedulePath,
      };

    case 'ADD_UNAVAILABLE_DATES': {
      if (!state.envConfig) return state;
      let workerList = [...state.envConfig.workerList];
      for (const entry of action.payload) {
        if (entry.dates.length === 0) continue;
        workerList = workerList.map(w => {
          if (w.id !== entry.workerId) return w;
          const existingSingle = w.unavailableDates.find(e => e.single);
          if (existingSingle?.single) {
            const existingSet = new Set(existingSingle.single.days.map(d => d.replace(/\//g, '-')));
            const toAdd = entry.dates.filter(d => !existingSet.has(d));
            if (toAdd.length === 0) return w;
            return {
              ...w,
              unavailableDates: w.unavailableDates.map(e =>
                e === existingSingle ? { ...e, single: { days: [...(e.single?.days ?? []), ...toAdd] } } : e,
              ),
            };
          }
          return { ...w, unavailableDates: [...w.unavailableDates, { single: { days: entry.dates } }] };
        });
      }
      return { ...state, envConfig: { ...state.envConfig, workerList } };
    }

    case 'RESIZE_UNAVAILABLE_RANGE': {
      if (!state.envConfig) return state;
      const { workerId, oldStartDate, oldEndDate, newStartDate, newEndDate } = action.payload;
      const oldSet = new Set(generateDateRange(oldStartDate, oldEndDate));
      const newDates = generateDateRange(newStartDate, newEndDate);
      const newSet = new Set(newDates);
      const removedDates = new Set([...oldSet].filter(d => !newSet.has(d)));
      const addedDates = newDates.filter(d => !oldSet.has(d));

      const workerList = state.envConfig.workerList.map(w => {
        if (w.id !== workerId) return w;
        // Remove old dates (both single and weekly-generated)
        let unavailDates = w.unavailableDates
          .map(entry => {
            if (entry.weekly) {
              // Remove weekdays that are entirely within removed dates
              const filtered = entry.weekly.weekdays.filter(wd => {
                const key = wd.trim().toLowerCase();
                const map: Record<string, number> = { sun: 0, sunday: 0, mon: 1, monday: 1, tue: 2, tuesday: 2, wed: 3, wednesday: 3, thu: 4, thursday: 4, fri: 5, friday: 5, sat: 6, saturday: 6 };
                const dow = map[key] ?? map[key.slice(0, 3)] ?? -1;
                if (dow < 0) return true;
                // Remove weekday if ALL its occurrences in old range are being removed
                const occurrences = [...oldSet].filter(d => new Date(`${d}T00:00:00`).getDay() === dow);
                return !occurrences.every(d => removedDates.has(d));
              });
              return filtered.length > 0 ? { ...entry, weekly: { weekdays: filtered } } : null;
            }
            if (entry.single) {
              const filtered = entry.single.days.filter(d => !removedDates.has(d.replace(/\//g, '-')));
              return filtered.length > 0 ? { ...entry, single: { days: filtered } } : null;
            }
            return entry;
          })
          .filter((e): e is NonNullable<typeof e> => e !== null);
        // Add new dates as single
        if (addedDates.length > 0) {
          const existingSingle = unavailDates.find(e => e.single);
          if (existingSingle?.single) {
            unavailDates = unavailDates.map(e =>
              e === existingSingle ? { ...e, single: { days: [...(e.single?.days ?? []), ...addedDates] } } : e,
            );
          } else {
            unavailDates = [...unavailDates, { single: { days: addedDates } }];
          }
        }
        return { ...w, unavailableDates: unavailDates };
      });
      return { ...state, envConfig: { ...state.envConfig, workerList } };
    }

    case 'OPEN_CONSTRAINT_DIALOG':
      return { ...state, isConstraintDialogOpen: true };

    case 'CLOSE_CONSTRAINT_DIALOG':
      return { ...state, isConstraintDialogOpen: false };

    case 'SET_CONSTRAINT_CHECKING':
      return { ...state, isConstraintChecking: action.payload };

    case 'SET_BACKEND_VIOLATIONS':
      return {
        ...state,
        backendViolations: action.payload.violations,
        constraintCheckedAt: action.payload.checkedAt,
        isConstraintChecking: false,
        isConstraintDialogOpen: true,
      };

    default:
      return state;
  }
}