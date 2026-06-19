import { ScheduleData } from './schedule';
import { EnvConfig } from './envConfig';

export type ViewMode = 'device' | 'worker';

export interface Violation {
  type: 'OVERLAP' | 'WORKER_UNAVAILABLE' | 'PHASE_OVERRUN';
  assignmentIndices: number[];
  message: string;
}

export interface SearchQuery {
  keyword: string;
  mode: 'device' | 'worker' | '';
}

export interface AppState {
  envConfig: EnvConfig | null;
  schedule: ScheduleData | null;
  currentView: ViewMode;
  violations: Violation[];
  undoStack: ScheduleData[];
  redoStack: ScheduleData[];
  selectedAssignmentIndex: number | null;
  expandedDeviceIds: Set<string>;
  searchQuery: SearchQuery;
  displayStartDate: string | null;
  displayEndDate: string | null;
  currentEnvPath: string | null;
  currentSchedulePath: string | null;
  errorMessage: string | null;
  isTaskAddDialogOpen: boolean;
  isFileOpenDialogOpen: boolean;
  isNewScheduleDialogOpen: boolean;
}

export type ActionType =
  | { type: 'LOAD_FILES'; payload: { envConfig: EnvConfig; schedule: ScheduleData; envPath: string; schedulePath: string } }
  | { type: 'SET_SCHEDULE'; payload: ScheduleData }
  | { type: 'SWITCH_VIEW'; payload: ViewMode }
  | { type: 'SET_VIOLATIONS'; payload: Violation[] }
  | { type: 'UNDO' }
  | { type: 'REDO' }
  | { type: 'SELECT_ASSIGNMENT'; payload: number | null }
  | { type: 'TOGGLE_DEVICE'; payload: string }
  | { type: 'SET_SEARCH_QUERY'; payload: SearchQuery }
  | { type: 'SET_DISPLAY_PERIOD'; payload: { startDate: string; endDate: string } }
  | { type: 'ADD_ASSIGNMENT'; payload: ScheduleData['assignmentList'][0] }
  | { type: 'UPDATE_ASSIGNMENT'; payload: { index: number; updates: Partial<ScheduleData['assignmentList'][0]> } }
  | { type: 'DELETE_ASSIGNMENT'; payload: number }
  | { type: 'BULK_UPDATE_FLEXIBILITY'; payload: { flexibility: string; target: 'all' | 'selected'; targetDate?: string } }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'OPEN_TASK_ADD_DIALOG' }
  | { type: 'CLOSE_TASK_ADD_DIALOG' }
  | { type: 'OPEN_FILE_DIALOG' }
  | { type: 'CLOSE_FILE_DIALOG' }
  | { type: 'OPEN_NEW_SCHEDULE_DIALOG' }
  | { type: 'CLOSE_NEW_SCHEDULE_DIALOG' }
  | { type: 'ADD_WORKFLOW_TASKS'; payload: ScheduleData['workflowTaskList'] }
  | { type: 'MERGE_DATA'; payload: { schedule?: ScheduleData; envConfig?: EnvConfig } }
  | { type: 'SAVE_PATHS'; payload: { envPath?: string; schedulePath?: string } };
