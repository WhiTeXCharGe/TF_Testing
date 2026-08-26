import { reducer } from '../../context/reducer';
import { AppState, DEFAULT_WORKER_VIEW_FILTER, DEFAULT_MODULE_VIEW_FILTER } from '../../types/appState';
import { ScheduleData } from '../../types/schedule';
import { EnvConfig } from '../../types/envConfig';

// ── Minimal fixtures ──────────────────────────────────────────────────────────

const EMPTY_SCHEDULE: ScheduleData = {
  planRange: { startDate: '2025-09-01', endDate: '2025-09-30' },
  workflowTaskList: [
    {
      id: 'wt001', name: 'Module A', workflow: 'wf_std', fab: 'fab1',
      phaseTaskList: [
        {
          id: 'wt001_p0', phase: 'p1', startDate: '2025-09-01', endDate: '2025-09-15',
          operationTaskList: [
            { id: 'wt001_p0_o0', operation: 'op1', workloadHours: 80 },
          ],
        },
      ],
    },
  ],
  assignmentList: [
    {
      worker: 'w001', operationTask: 'wt001_p0_o0',
      startDate: '2025-09-01', endDate: '2025-09-05',
      planFlexibility: 'Flexible', workDateList: [],
    },
  ],
};

const EMPTY_ENV: EnvConfig = {
  workflowList: [],
  fabList: [{ id: 'fab1', name: 'Fab 1', unavailableDates: [] }],
  regionList: [],
  customerCompanyList: [],
  workerCompanyList: [],
  workerList: [{ id: 'w001', name: 'Worker One', unavailableDates: [] }],
  transiteDayMap: [],
};

const BASE_STATE: AppState = {
  envConfig: EMPTY_ENV,
  schedule: EMPTY_SCHEDULE,
  currentView: 'worker',
  violations: [],
  undoStack: [],
  redoStack: [],
  selectedAssignmentIndex: null,
  selectedUnavailableInfo: null,
  expandedDeviceIds: new Set(),
  workerViewFilter: { ...DEFAULT_WORKER_VIEW_FILTER },
  moduleViewFilter: { ...DEFAULT_MODULE_VIEW_FILTER },
  currentEnvPath: null,
  currentSchedulePath: null,
  errorMessage: null,
  isTaskAddDialogOpen: false,
  isFileOpenDialogOpen: false,
  isNewScheduleDialogOpen: false,
  isSendToSchedulerDialogOpen: false,
  session: null,
  isSessionDialogOpen: false,
};

// ── SWITCH_VIEW ───────────────────────────────────────────────────────────────

describe('SWITCH_VIEW', () => {
  it('switches to device view', () => {
    const next = reducer(BASE_STATE, { type: 'SWITCH_VIEW', payload: 'device' });
    expect(next.currentView).toBe('device');
  });
  it('clears selectedAssignmentIndex on view switch', () => {
    const state = { ...BASE_STATE, selectedAssignmentIndex: 0 };
    const next = reducer(state, { type: 'SWITCH_VIEW', payload: 'device' });
    expect(next.selectedAssignmentIndex).toBeNull();
  });
});

// ── SELECT_ASSIGNMENT ─────────────────────────────────────────────────────────

describe('SELECT_ASSIGNMENT', () => {
  it('sets selectedAssignmentIndex', () => {
    const next = reducer(BASE_STATE, { type: 'SELECT_ASSIGNMENT', payload: 0 });
    expect(next.selectedAssignmentIndex).toBe(0);
  });
  it('clears selectedAssignmentIndex on null', () => {
    const state = { ...BASE_STATE, selectedAssignmentIndex: 0 };
    const next = reducer(state, { type: 'SELECT_ASSIGNMENT', payload: null });
    expect(next.selectedAssignmentIndex).toBeNull();
  });
  it('clears selectedUnavailableInfo when selecting an assignment', () => {
    const state = { ...BASE_STATE, selectedUnavailableInfo: { workerId: 'w1', startDate: '2025-01-01', endDate: '2025-01-01' } };
    const next = reducer(state, { type: 'SELECT_ASSIGNMENT', payload: 0 });
    expect(next.selectedUnavailableInfo).toBeNull();
  });
});

// ── UPDATE_ASSIGNMENT ─────────────────────────────────────────────────────────

describe('UPDATE_ASSIGNMENT', () => {
  it('updates an assignment field', () => {
    const next = reducer(BASE_STATE, {
      type: 'UPDATE_ASSIGNMENT',
      payload: { index: 0, updates: { endDate: '2025-09-10' } },
    });
    expect(next.schedule!.assignmentList[0].endDate).toBe('2025-09-10');
  });
  it('pushes to undoStack', () => {
    const next = reducer(BASE_STATE, {
      type: 'UPDATE_ASSIGNMENT',
      payload: { index: 0, updates: { endDate: '2025-09-10' } },
    });
    expect(next.undoStack).toHaveLength(1);
  });
  it('clears redoStack', () => {
    const state = { ...BASE_STATE, redoStack: [EMPTY_SCHEDULE] };
    const next = reducer(state, {
      type: 'UPDATE_ASSIGNMENT',
      payload: { index: 0, updates: { worker: 'w002' } },
    });
    expect(next.redoStack).toHaveLength(0);
  });
  it('does nothing for out-of-bounds index', () => {
    const next = reducer(BASE_STATE, {
      type: 'UPDATE_ASSIGNMENT',
      payload: { index: 99, updates: { worker: 'w002' } },
    });
    expect(next.schedule!.assignmentList[0].worker).toBe('w001');
  });
});

// ── DELETE_ASSIGNMENT ─────────────────────────────────────────────────────────

describe('DELETE_ASSIGNMENT', () => {
  it('removes the assignment at the given index', () => {
    const next = reducer(BASE_STATE, { type: 'DELETE_ASSIGNMENT', payload: 0 });
    expect(next.schedule!.assignmentList).toHaveLength(0);
  });
  it('clears selectedAssignmentIndex', () => {
    const state = { ...BASE_STATE, selectedAssignmentIndex: 0 };
    const next = reducer(state, { type: 'DELETE_ASSIGNMENT', payload: 0 });
    expect(next.selectedAssignmentIndex).toBeNull();
  });
  it('pushes to undoStack', () => {
    const next = reducer(BASE_STATE, { type: 'DELETE_ASSIGNMENT', payload: 0 });
    expect(next.undoStack).toHaveLength(1);
  });
});

// ── UNDO / REDO ───────────────────────────────────────────────────────────────

describe('UNDO / REDO', () => {
  it('UNDO restores previous schedule', () => {
    const state = {
      ...BASE_STATE,
      undoStack: [{ ...EMPTY_SCHEDULE, planRange: { startDate: '2024-01-01', endDate: '2024-12-31' } }],
    };
    const next = reducer(state, { type: 'UNDO' });
    expect(next.schedule!.planRange.startDate).toBe('2024-01-01');
  });
  it('UNDO does nothing when undoStack is empty', () => {
    const next = reducer(BASE_STATE, { type: 'UNDO' });
    expect(next).toBe(BASE_STATE);
  });
  it('REDO does nothing when redoStack is empty', () => {
    const next = reducer(BASE_STATE, { type: 'REDO' });
    expect(next).toBe(BASE_STATE);
  });
  it('UNDO / REDO round-trip preserves schedule', () => {
    const afterUpdate = reducer(BASE_STATE, {
      type: 'UPDATE_ASSIGNMENT',
      payload: { index: 0, updates: { endDate: '2025-09-20' } },
    });
    const afterUndo = reducer(afterUpdate, { type: 'UNDO' });
    expect(afterUndo.schedule!.assignmentList[0].endDate).toBe('2025-09-05');
    const afterRedo = reducer(afterUndo, { type: 'REDO' });
    expect(afterRedo.schedule!.assignmentList[0].endDate).toBe('2025-09-20');
  });
});

// ── SET_WORKER_VIEW_FILTER ────────────────────────────────────────────────────

describe('SET_WORKER_VIEW_FILTER', () => {
  it('merges partial filter', () => {
    const next = reducer(BASE_STATE, {
      type: 'SET_WORKER_VIEW_FILTER',
      payload: { barName: 'Module A' },
    });
    expect(next.workerViewFilter.barName).toBe('Module A');
    expect(next.workerViewFilter.moduleIds).toEqual([]);
  });
  it('sets moduleIds', () => {
    const next = reducer(BASE_STATE, {
      type: 'SET_WORKER_VIEW_FILTER',
      payload: { moduleIds: ['wt001', 'wt002'] },
    });
    expect(next.workerViewFilter.moduleIds).toEqual(['wt001', 'wt002']);
  });
  it('sets date range', () => {
    const next = reducer(BASE_STATE, {
      type: 'SET_WORKER_VIEW_FILTER',
      payload: { startDate: '2025-09-01', endDate: '2025-09-15' },
    });
    expect(next.workerViewFilter.startDate).toBe('2025-09-01');
    expect(next.workerViewFilter.endDate).toBe('2025-09-15');
  });
});

// ── SET_MODULE_VIEW_FILTER ────────────────────────────────────────────────────

describe('SET_MODULE_VIEW_FILTER', () => {
  it('merges partial filter', () => {
    const next = reducer(BASE_STATE, {
      type: 'SET_MODULE_VIEW_FILTER',
      payload: { workerIds: ['w001'] },
    });
    expect(next.moduleViewFilter.workerIds).toEqual(['w001']);
    expect(next.moduleViewFilter.fabIds).toEqual([]);
  });
});

// ── Dialog open/close ─────────────────────────────────────────────────────────

describe('Dialog actions', () => {
  it('OPEN_TASK_ADD_DIALOG sets isTaskAddDialogOpen true', () => {
    const next = reducer(BASE_STATE, { type: 'OPEN_TASK_ADD_DIALOG' });
    expect(next.isTaskAddDialogOpen).toBe(true);
  });
  it('CLOSE_TASK_ADD_DIALOG sets isTaskAddDialogOpen false', () => {
    const state = { ...BASE_STATE, isTaskAddDialogOpen: true };
    const next = reducer(state, { type: 'CLOSE_TASK_ADD_DIALOG' });
    expect(next.isTaskAddDialogOpen).toBe(false);
  });
  it('OPEN_FILE_DIALOG sets isFileOpenDialogOpen true', () => {
    const next = reducer(BASE_STATE, { type: 'OPEN_FILE_DIALOG' });
    expect(next.isFileOpenDialogOpen).toBe(true);
  });
  it('OPEN_NEW_SCHEDULE_DIALOG sets isNewScheduleDialogOpen true', () => {
    const next = reducer(BASE_STATE, { type: 'OPEN_NEW_SCHEDULE_DIALOG' });
    expect(next.isNewScheduleDialogOpen).toBe(true);
  });
});

// ── SET_ERROR ─────────────────────────────────────────────────────────────────

describe('SET_ERROR', () => {
  it('sets errorMessage', () => {
    const next = reducer(BASE_STATE, { type: 'SET_ERROR', payload: 'Something went wrong' });
    expect(next.errorMessage).toBe('Something went wrong');
  });
  it('clears errorMessage with null', () => {
    const state = { ...BASE_STATE, errorMessage: 'error' };
    const next = reducer(state, { type: 'SET_ERROR', payload: null });
    expect(next.errorMessage).toBeNull();
  });
});

// ── ADD_WORKFLOW_TASKS ────────────────────────────────────────────────────────

describe('ADD_WORKFLOW_TASKS', () => {
  it('adds new workflow tasks', () => {
    const newTask = { id: 'wt002', name: 'Module B', workflow: 'wf_std', phaseTaskList: [] };
    const next = reducer(BASE_STATE, { type: 'ADD_WORKFLOW_TASKS', payload: [newTask] });
    expect(next.schedule!.workflowTaskList).toHaveLength(2);
    expect(next.schedule!.workflowTaskList[1].id).toBe('wt002');
  });
  it('ignores duplicate IDs', () => {
    const dupTask = { id: 'wt001', name: 'Module A Dup', workflow: 'wf_std', phaseTaskList: [] };
    const next = reducer(BASE_STATE, { type: 'ADD_WORKFLOW_TASKS', payload: [dupTask] });
    expect(next.schedule!.workflowTaskList).toHaveLength(1);
  });
});

// ── Live collaboration session ─────────────────────────────────────────────

describe('SET_SESSION', () => {
  it('sets the session', () => {
    const session = { id: 's1', role: 'edit' as const, connectionStatus: 'connecting' as const, participants: [] };
    const next = reducer(BASE_STATE, { type: 'SET_SESSION', payload: session });
    expect(next.session).toEqual(session);
  });

  it('clears the session', () => {
    const state = { ...BASE_STATE, session: { id: 's1', role: 'edit' as const, connectionStatus: 'connected' as const, participants: [] } };
    const next = reducer(state, { type: 'SET_SESSION', payload: null });
    expect(next.session).toBeNull();
  });
});

describe('SET_SESSION_BASELINE', () => {
  it('replaces schedule/envConfig/currentView, resets undo/redo and selection', () => {
    const state = {
      ...BASE_STATE,
      undoStack: [EMPTY_SCHEDULE],
      redoStack: [EMPTY_SCHEDULE],
      selectedAssignmentIndex: 0,
      selectedUnavailableInfo: { workerId: 'w001', startDate: '2025-09-01', endDate: '2025-09-02' },
    };
    const newSchedule = { ...EMPTY_SCHEDULE, planRange: { startDate: '2026-01-01', endDate: '2026-01-31' } };
    const next = reducer(state, { type: 'SET_SESSION_BASELINE', payload: { schedule: newSchedule, envConfig: EMPTY_ENV, currentView: 'device' } });
    expect(next.schedule).toBe(newSchedule);
    expect(next.currentView).toBe('device');
    expect(next.undoStack).toEqual([]);
    expect(next.redoStack).toEqual([]);
    expect(next.selectedAssignmentIndex).toBeNull();
    expect(next.selectedUnavailableInfo).toBeNull();
  });
});

describe('SET_SESSION_CONNECTION_STATUS', () => {
  it('updates connectionStatus on an existing session', () => {
    const state = { ...BASE_STATE, session: { id: 's1', role: 'edit' as const, connectionStatus: 'connecting' as const, participants: [] } };
    const next = reducer(state, { type: 'SET_SESSION_CONNECTION_STATUS', payload: 'connected' });
    expect(next.session?.connectionStatus).toBe('connected');
  });

  it('is a no-op when there is no session', () => {
    const next = reducer(BASE_STATE, { type: 'SET_SESSION_CONNECTION_STATUS', payload: 'connected' });
    expect(next.session).toBeNull();
  });
});

describe('SET_SESSION_PARTICIPANTS', () => {
  it('updates participants on an existing session', () => {
    const state = { ...BASE_STATE, session: { id: 's1', role: 'edit' as const, connectionStatus: 'connected' as const, participants: [] } };
    const participants = [{ id: 'p1', name: 'Alice', role: 'edit' as const }];
    const next = reducer(state, { type: 'SET_SESSION_PARTICIPANTS', payload: participants });
    expect(next.session?.participants).toEqual(participants);
  });
});

describe('OPEN_SESSION_DIALOG / CLOSE_SESSION_DIALOG', () => {
  it('opens and closes the session dialog', () => {
    const opened = reducer(BASE_STATE, { type: 'OPEN_SESSION_DIALOG' });
    expect(opened.isSessionDialogOpen).toBe(true);
    const closed = reducer(opened, { type: 'CLOSE_SESSION_DIALOG' });
    expect(closed.isSessionDialogOpen).toBe(false);
  });
});