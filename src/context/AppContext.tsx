import { createContext, useContext, useReducer, Dispatch, ReactNode } from 'react';
import { AppState, ActionType, DEFAULT_WORKER_VIEW_FILTER, DEFAULT_MODULE_VIEW_FILTER, DEFAULT_WORKER_COLUMN_FILTER } from '../types/appState';
import { reducer } from './reducer';

const initialState: AppState = {
  envConfig: null,
  schedule: null,
  currentView: 'worker',
  violations: [],
  undoStack: [],
  redoStack: [],
  selectedAssignmentIndex: null,
  selectedUnavailableInfo: null,
  expandedDeviceIds: new Set(),
  workerViewFilter: { ...DEFAULT_WORKER_VIEW_FILTER },
  moduleViewFilter: { ...DEFAULT_MODULE_VIEW_FILTER },
  workerColumnFilter: { ...DEFAULT_WORKER_COLUMN_FILTER },
  workerDateCellFilter: { date: '', tasks: [] },
  currentEnvPath: null,
  currentSchedulePath: null,
  errorMessage: null,
  isTaskAddDialogOpen: false,
  isFileOpenDialogOpen: false,
  isNewScheduleDialogOpen: false,
};

interface ContextType {
  state: AppState;
  dispatch: Dispatch<ActionType>;
}

const AppContext = createContext<ContextType | undefined>(undefined);

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  return <AppContext.Provider value={{ state, dispatch }}>{children}</AppContext.Provider>;
}

export function useAppContext(): ContextType {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useAppContext must be used within AppProvider');
  return ctx;
}
