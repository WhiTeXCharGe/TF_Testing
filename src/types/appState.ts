import { ScheduleData, PhaseTask } from './schedule';
import { EnvConfig } from './envConfig';

export type ViewMode = 'device' | 'worker';

export interface Violation {
  type:
    | 'WORKER_UNAVAILABLE'
    | 'PHASE_OVERRUN'
    | 'WORK_HOUR_RANGE'
    | 'SKILL_MISMATCH'
    | 'REGION_SUITABILITY'
    | 'COMPANY_SUITABILITY'
    // backend-only
    | 'OVERLAP'
    | 'TASK_WORKER_COUNT'
    | 'PHASE_SEQUENCE'
    | 'WORKLOAD_TOTAL'
    | 'RESPONSIBLE_WORKER'
    | 'TRAVEL_DAYS'
    | 'OVERTIME'
    | 'STAY_DURATION';
  assignmentIndices: number[];
  message: string;
  date?: string;
  severity?: 'error' | 'warning';
}

export interface WorkerViewFilter {
  barName: string;       // free-text search in bar label
  moduleIds: string[];   // 装置 (製番) workflowTask IDs
  phaseIds: string[];    // 工程 phase IDs from EnvConfig
  fabIds: string[];
  regionIds: string[];
  startDate: string | null;
  endDate: string | null;
}

export interface ModuleViewFilter {
  workerIds: string[];
  fabIds: string[];
  regionIds: string[];
  startDate: string | null;
  endDate: string | null;
}

export const DEFAULT_WORKER_VIEW_FILTER: WorkerViewFilter = {
  barName: '', moduleIds: [], phaseIds: [], fabIds: [], regionIds: [],
  startDate: null, endDate: null,
};

export const DEFAULT_MODULE_VIEW_FILTER: ModuleViewFilter = {
  workerIds: [], fabIds: [], regionIds: [],
  startDate: null, endDate: null,
};

/** Column-level filter (company/name/manager/remarks + extra expanded columns). */
export interface WorkerColumnFilter {
  id: string[];
  company: string[];
  name: string[];
  manager: string[];
  remarks: string[];
  workType: string[];
  assignedDuties: string[];
  visa: string[];
  overseasDriving: string[];
}

export const DEFAULT_WORKER_COLUMN_FILTER: WorkerColumnFilter = {
  id: [], company: [], name: [], manager: [], remarks: [],
  workType: [], assignedDuties: [], visa: [], overseasDriving: [],
};

/** Date-cell filter (clicking a date column to filter by task name). */
export interface WorkerDateCellFilter {
  date: string;
  tasks: string[];
}

export interface AppState {
  envConfig: EnvConfig | null;
  schedule: ScheduleData | null;
  currentView: ViewMode;
  violations: Violation[];
  undoStack: ScheduleData[];
  redoStack: ScheduleData[];
  selectedAssignmentIndex: number | null;
  selectedUnavailableInfo: { workerId: string; startDate: string; endDate: string } | null;
  expandedDeviceIds: Set<string>;
  workerViewFilter: WorkerViewFilter;
  moduleViewFilter: ModuleViewFilter;
  workerColumnFilter: WorkerColumnFilter;
  workerDateCellFilter: WorkerDateCellFilter;
  currentEnvPath: string | null;
  currentSchedulePath: string | null;
  // Snapshots of schedule/envConfig as of the last successful save (or load).
  // Compared by reference against the live schedule/envConfig to detect
  // unsaved changes — every mutating reducer action produces a new object
  // reference, so a mismatch here means "dirty".
  savedScheduleRef: ScheduleData | null;
  savedEnvConfigRef: EnvConfig | null;
  errorMessage: string | null;
  isTaskAddDialogOpen: boolean;
  isFileOpenDialogOpen: boolean;
  isNewScheduleDialogOpen: boolean;
  isSendToSchedulerDialogOpen: boolean;
  // Backend constraint check
  isConstraintDialogOpen: boolean;
  isConstraintChecking: boolean;
  backendViolations: Violation[];
  constraintCheckedAt: string | null;
  showFlightStints: boolean;
  scrollToSelectedAssignment: boolean;
  // Live View sharing (fast read-only broadcast, see services/viewBroadcastService.ts).
  // Not a collaboration session — one-way, host-to-viewers, no editing on the viewer side.
  isSharingLiveView: boolean;
  liveViewShareLink: string | null;
  isShareViewDialogOpen: boolean;
  viewConnectionStatus: 'disconnected' | 'connecting' | 'connected';
}

export type ActionType =
  | { type: 'LOAD_FILES'; payload: { envConfig: EnvConfig; schedule: ScheduleData; envPath: string; schedulePath: string } }
  | { type: 'SET_SCHEDULE'; payload: ScheduleData }
  | { type: 'UPDATE_PLAN_RANGE'; payload: { startDate: string; endDate: string } }
  | { type: 'SWITCH_VIEW'; payload: ViewMode }
  | { type: 'SET_VIOLATIONS'; payload: Violation[] }
  | { type: 'UNDO' }
  | { type: 'REDO' }
  | { type: 'SELECT_ASSIGNMENT'; payload: number | null }
  | { type: 'TOGGLE_DEVICE'; payload: string }
  | { type: 'SET_WORKER_VIEW_FILTER'; payload: Partial<WorkerViewFilter> }
  | { type: 'SET_MODULE_VIEW_FILTER'; payload: Partial<ModuleViewFilter> }
  | { type: 'ADD_ASSIGNMENT'; payload: ScheduleData['assignmentList'][0] }
  | { type: 'UPDATE_ASSIGNMENT'; payload: { index: number; updates: Partial<ScheduleData['assignmentList'][0]> } }
  | { type: 'UPDATE_PHASE_TASK'; payload: { workflowTaskId: string; phaseTaskId: string; updates: Partial<PhaseTask> } }
  | { type: 'UPDATE_OPERATION_TASK'; payload: { workflowTaskId: string; phaseTaskId: string; operationTaskId: string; updates: Partial<import('./schedule').OperationTask> } }
  | { type: 'DELETE_ASSIGNMENT'; payload: number }
  | { type: 'BULK_UPDATE_FLEXIBILITY'; payload: { flexibility: string; target: 'all' | 'selected'; targetDate?: string } }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'OPEN_TASK_ADD_DIALOG' }
  | { type: 'CLOSE_TASK_ADD_DIALOG' }
  | { type: 'OPEN_FILE_DIALOG' }
  | { type: 'CLOSE_FILE_DIALOG' }
  | { type: 'OPEN_NEW_SCHEDULE_DIALOG' }
  | { type: 'CLOSE_NEW_SCHEDULE_DIALOG' }
  | { type: 'OPEN_SEND_TO_SCHEDULER_DIALOG' }
  | { type: 'CLOSE_SEND_TO_SCHEDULER_DIALOG' }
  | { type: 'ADD_WORKFLOW_TASKS'; payload: ScheduleData['workflowTaskList'] }
  | { type: 'MERGE_DATA'; payload: { schedule?: ScheduleData; envConfig?: EnvConfig } }
  | { type: 'SAVE_PATHS'; payload: { envPath?: string; schedulePath?: string } }
  | { type: 'MARK_SAVED' }
  | { type: 'SELECT_UNAVAILABLE'; payload: { workerId: string; startDate: string; endDate: string } | null }
  | { type: 'DELETE_UNAVAILABLE_DATE'; payload: { workerId: string; date: string } }
  | { type: 'DELETE_UNAVAILABLE_RANGE'; payload: { workerId: string; startDate: string; endDate: string } }
  | { type: 'MOVE_UNAVAILABLE_DATE'; payload: { workerId: string; oldDate: string; newDate: string } }
  | { type: 'UPDATE_OPERATION_TASK_COLOR'; payload: { operationTaskId: string; colorCode: string } }
  | { type: 'UPDATE_WORKFLOW_TASK_COLOR'; payload: { workflowTaskId: string; colorCode: string } }
  | { type: 'UPDATE_WORKER_DEFINITION'; payload: { workerId: string; definition: string } }
  | { type: 'ADD_UNAVAILABLE_DATES'; payload: Array<{ workerId: string; dates: string[] }> }
  | { type: 'RESIZE_UNAVAILABLE_RANGE'; payload: { workerId: string; oldStartDate: string; oldEndDate: string; newStartDate: string; newEndDate: string } }
  | { type: 'SET_WORKER_COLUMN_FILTER'; payload: Partial<WorkerColumnFilter> }
  | { type: 'SET_WORKER_DATE_CELL_FILTER'; payload: WorkerDateCellFilter }
  | { type: 'UPDATE_WORKER_DESC_FIELD'; payload: { workerId: string; field: '業務形態' | 'VISA' | '海外運転'; value: string } }
  | { type: 'CLEAR_ALL_WORKER_FILTERS' }
  // Backend constraint check
  | { type: 'OPEN_CONSTRAINT_DIALOG' }
  | { type: 'CLOSE_CONSTRAINT_DIALOG' }
  | { type: 'SET_CONSTRAINT_CHECKING'; payload: boolean }
  | { type: 'SET_BACKEND_VIOLATIONS'; payload: { violations: Violation[]; checkedAt: string } }
  | { type: 'TOGGLE_FLIGHT_STINTS' }
  | { type: 'SELECT_ASSIGNMENT_AND_SCROLL'; payload: number }
  | { type: 'CLEAR_SCROLL_TO_ASSIGNMENT' }
  // Live View sharing
  | { type: 'SET_SHARING_LIVE_VIEW'; payload: boolean }
  | { type: 'SET_LIVE_VIEW_SHARE_LINK'; payload: string | null }
  | { type: 'OPEN_SHARE_VIEW_DIALOG' }
  | { type: 'CLOSE_SHARE_VIEW_DIALOG' }
  | { type: 'SET_VIEW_CONNECTION_STATUS'; payload: 'disconnected' | 'connecting' | 'connected' }
  | { type: 'SET_VIEW_STATE'; payload: { schedule: ScheduleData; envConfig: EnvConfig; currentView: ViewMode } };