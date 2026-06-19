import { AppState, ActionType } from '../types/appState';
import { ScheduleData, PlanFlexibility } from '../types/schedule';
import { EnvConfig } from '../types/envConfig';
import { MAX_UNDO_STACK } from '../config/appConfig';

function mergeById<T extends { id: string }>(existing: T[], incoming: T[]): T[] {
  const existingIds = new Set(existing.map(x => x.id));
  return [...existing, ...incoming.filter(x => !existingIds.has(x.id))];
}

function pushUndo(state: AppState): ScheduleData[] {
  if (!state.schedule) return state.undoStack;
  return [...state.undoStack, state.schedule].slice(-MAX_UNDO_STACK);
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
      return { ...state, selectedAssignmentIndex: action.payload };

    case 'TOGGLE_DEVICE': {
      const newExpanded = new Set(state.expandedDeviceIds);
      if (newExpanded.has(action.payload)) newExpanded.delete(action.payload);
      else newExpanded.add(action.payload);
      return { ...state, expandedDeviceIds: newExpanded };
    }

    case 'SET_SEARCH_QUERY':
      return { ...state, searchQuery: action.payload };

    case 'SET_DISPLAY_PERIOD':
      return { ...state, displayStartDate: action.payload.startDate, displayEndDate: action.payload.endDate };

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

    default:
      return state;
  }
}
