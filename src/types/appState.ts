import { ScheduleData, PhaseTask } from './schedule';
import { EnvConfig } from './envConfig';

export type ViewMode = 'device' | 'worker';

// One recorded entry per edit YOU made (never someone else's — inbound
// remote actions never produce one; see AppContext.tsx's dispatch wrapper
// and GanttChartEditor_LiveCollabEdit_UndoConflictDesign20260904.md §2-3).
// A discriminated union because each action type's "what changed" shape is
// different; src/context/undoEntries.ts is the only place that interprets
// these.
export type UndoEntry =
  | { kind: 'fieldPatch'; type: ActionType['type']; idPayload: Record<string, unknown>; fieldsBefore: Record<string, unknown>; fieldsAfter: Record<string, unknown> }
  | { kind: 'assignmentPatch'; id: string; fieldsBefore: Record<string, unknown>; fieldsAfter: Record<string, unknown> }
  | { kind: 'assignmentAdd'; id: string; added: ScheduleData['assignmentList'][0] }
  | { kind: 'assignmentDelete'; id: string; deleted: ScheduleData['assignmentList'][0] }
  | { kind: 'planRange'; before: { startDate: string; endDate: string }; after: { startDate: string; endDate: string } }
  | { kind: 'workerUnavailable'; workerId: string; before: unknown[]; after: unknown[] }
  | { kind: 'bulkFlex'; changes: { assignmentId: string; before: string; after: string }[] }
  | { kind: 'addWorkflowTasks'; addedIds: string[] }
  | { kind: 'mergeData'; addedWorkflowTaskIds: string[]; addedAssignmentIds: string[]; addedEnvConfigIds: Record<string, string[]> };

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

export type SessionRole = 'edit' | 'view';
export type SessionConnectionStatus = 'disconnected' | 'connecting' | 'connected';

// Server-side lifecycle status of a session (ACA1/ACA2). 'lock' = live but
// read-only for everyone; 'close' = not running (re-open reactivates it).
export type SessionStatus = 'open' | 'lock' | 'close';

export interface SessionParticipant {
  id: string;
  name: string;
  role: SessionRole;
}

/** One row of the online-session list from ACA1's GET /api/sessions. */
export interface SessionSummary {
  id: string;
  name: string;
  status: SessionStatus;
  createdAt: number;
  lastActivityAt: number;
  /** When someone last joined; the list is sorted on this (most recent first). */
  lastJoinAt: number | null;
  participantCount: number | null;
}

/** Which online-session dialog is open (all mutually exclusive). */
export type SessionDialogKind = 'join' | 'create' | 'info';

export interface SessionBaseline {
  schedule: ScheduleData;
  envConfig: EnvConfig;
  currentView: ViewMode;
}

export interface SessionState {
  id: string;
  name: string;
  role: SessionRole;
  connectionStatus: SessionConnectionStatus;
  participants: SessionParticipant[];
  // Live server-side status pushed via sync-init / session-status.
  status: SessionStatus;
  // Present only for the participant who created the session — gates the
  // lock/unlock and delete controls. Never sent to other participants.
  ownerToken?: string;
}

export interface AppState {
  envConfig: EnvConfig | null;
  schedule: ScheduleData | null;
  currentView: ViewMode;
  violations: Violation[];
  myPendingUndo: UndoEntry[];
  myPendingRedo: UndoEntry[];
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
  // Live collaboration session (see services/collabService.ts). null when not
  // in a session — solo editing/viewing is unaffected either way.
  session: SessionState | null;
  // Which online-session dialog is open (join / create / info), or null.
  sessionDialog: SessionDialogKind | null;
}

export type ActionType =
  | { type: 'LOAD_FILES'; payload: { envConfig: EnvConfig; schedule: ScheduleData; envPath: string; schedulePath: string } }
  | { type: 'SET_SCHEDULE'; payload: ScheduleData }
  | { type: 'UPDATE_PLAN_RANGE'; payload: { startDate: string; endDate: string } }
  | { type: 'SWITCH_VIEW'; payload: ViewMode }
  | { type: 'SET_VIOLATIONS'; payload: Violation[] }
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
  // Live collaboration session
  | { type: 'SET_SESSION'; payload: SessionState | null }
  | { type: 'SET_SESSION_BASELINE'; payload: SessionBaseline }
  | { type: 'SET_SESSION_CONNECTION_STATUS'; payload: SessionConnectionStatus }
  | { type: 'SET_SESSION_STATUS'; payload: SessionStatus }
  | { type: 'SET_SESSION_PARTICIPANTS'; payload: SessionParticipant[] }
  | { type: 'OPEN_SESSION_DIALOG'; payload: SessionDialogKind }
  | { type: 'CLOSE_SESSION_DIALOG' }
  | { type: 'SET_SESSION_NAME'; payload: string }
  // Internal to the undo/redo mechanism (src/context/undoEntries.ts,
  // AppContext.tsx's dispatch wrapper) — never dispatched directly by UI
  // code. UNDO/REDO tokens (dispatched by UndoRedoButtons.tsx and
  // useKeyboardShortcuts.ts) are intercepted by the wrapper before they
  // reach the reducer at all; see Task 5.
  | { type: 'RECORD_UNDO_ENTRY'; payload: UndoEntry }
  | { type: 'CONSUME_UNDO_ENTRY' }
  | { type: 'CONSUME_REDO_ENTRY' }
  | { type: 'RESTORE_ASSIGNMENT_FIELDS'; payload: { assignmentId: string; updates: Partial<ScheduleData['assignmentList'][0]> }[] }
  | { type: 'REMOVE_WORKFLOW_TASKS_BY_ID'; payload: string[] }
  | { type: 'RESTORE_WORKER_UNAVAILABLE_DATES'; payload: { workerId: string; unavailableDates: unknown[] } }
  | { type: 'REVERT_MERGE'; payload: { workflowTaskIds: string[]; assignmentIds: string[]; envConfigIds: Record<string, string[]> } };