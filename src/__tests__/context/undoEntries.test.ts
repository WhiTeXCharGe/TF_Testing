import { captureUndoEntry, hasConflict, buildRevertAction } from '../../context/undoEntries';
import { AppState, DEFAULT_WORKER_VIEW_FILTER, DEFAULT_MODULE_VIEW_FILTER, DEFAULT_WORKER_COLUMN_FILTER } from '../../types/appState';
import { ScheduleData } from '../../types/schedule';
import { EnvConfig } from '../../types/envConfig';

const SCHEDULE: ScheduleData = {
  planRange: { startDate: '2025-09-01', endDate: '2025-09-30' },
  workflowTaskList: [
    {
      id: 'wt1', workflow: 'wf', colorCode: 'blue',
      phaseTaskList: [
        {
          id: 'pt1', phase: 'p1', startDate: '2025-09-01', endDate: '2025-09-15',
          operationTaskList: [{ id: 'ot1', operation: 'op1', workloadHours: 10, colorCode: 'red' }],
        },
      ],
    },
  ],
  assignmentList: [
    { _id: 'a1', worker: 'w1', operationTask: 'ot1', startDate: '2025-09-01', endDate: '2025-09-05', planFlexibility: 'Flexible', workDateList: [] },
  ],
};

const ENV: EnvConfig = {
  workflowList: [], fabList: [], regionList: [], customerCompanyList: [], workerCompanyList: [],
  workerList: [{ id: 'w1', name: 'Worker One', description: { '備考': 'old note' }, unavailableDates: [] }],
  transiteDayMap: [],
};

const STATE: AppState = {
  envConfig: ENV, schedule: SCHEDULE, currentView: 'worker', violations: [],
  myPendingUndo: [], myPendingRedo: [],
  selectedAssignmentIndex: null, selectedUnavailableInfo: null, expandedDeviceIds: new Set(),
  workerViewFilter: { ...DEFAULT_WORKER_VIEW_FILTER }, moduleViewFilter: { ...DEFAULT_MODULE_VIEW_FILTER },
  workerColumnFilter: { ...DEFAULT_WORKER_COLUMN_FILTER }, workerDateCellFilter: { date: '', tasks: [] },
  currentEnvPath: null, currentSchedulePath: null, savedScheduleRef: null, savedEnvConfigRef: null,
  errorMessage: null, isTaskAddDialogOpen: false, isFileOpenDialogOpen: false, isNewScheduleDialogOpen: false,
  isSendToSchedulerDialogOpen: false, isConstraintDialogOpen: false, isConstraintChecking: false,
  backendViolations: [], constraintCheckedAt: null, showFlightStints: false, scrollToSelectedAssignment: false,
  session: null, sessionDialog: null,
};

describe('UPDATE_OPERATION_TASK_COLOR', () => {
  const payload = { operationTaskId: 'ot1', colorCode: 'green' };

  it('captures the old and new color', () => {
    const entry = captureUndoEntry('UPDATE_OPERATION_TASK_COLOR', payload, STATE);
    expect(entry).toEqual({
      kind: 'fieldPatch', type: 'UPDATE_OPERATION_TASK_COLOR',
      idPayload: { operationTaskId: 'ot1' },
      fieldsBefore: { colorCode: 'red' }, fieldsAfter: { colorCode: 'green' },
    });
  });

  it('is safe to undo when nothing has changed since', () => {
    const entry = captureUndoEntry('UPDATE_OPERATION_TASK_COLOR', payload, STATE)!;
    const after = { ...STATE, schedule: { ...SCHEDULE, workflowTaskList: [{ ...SCHEDULE.workflowTaskList[0], phaseTaskList: [{ ...SCHEDULE.workflowTaskList[0].phaseTaskList[0], operationTaskList: [{ ...SCHEDULE.workflowTaskList[0].phaseTaskList[0].operationTaskList[0], colorCode: 'green' }] }] }] } };
    expect(hasConflict(entry, after, 'undo')).toBe(false);
    const revert = buildRevertAction(entry, after, 'undo');
    expect(revert).toEqual({ type: 'UPDATE_OPERATION_TASK_COLOR', payload: { operationTaskId: 'ot1', colorCode: 'red' } });
  });

  it('blocks undo when someone else changed the color since', () => {
    const entry = captureUndoEntry('UPDATE_OPERATION_TASK_COLOR', payload, STATE)!;
    const touchedByOther = { ...STATE, schedule: { ...SCHEDULE, workflowTaskList: [{ ...SCHEDULE.workflowTaskList[0], phaseTaskList: [{ ...SCHEDULE.workflowTaskList[0].phaseTaskList[0], operationTaskList: [{ ...SCHEDULE.workflowTaskList[0].phaseTaskList[0].operationTaskList[0], colorCode: 'purple' }] }] }] } };
    expect(hasConflict(entry, touchedByOther, 'undo')).toBe(true);
  });
});

describe('UPDATE_WORKER_DESC_FIELD', () => {
  it('captures and reverts a description field', () => {
    const payload = { workerId: 'w1', field: '業務形態' as const, value: 'A' };
    const entry = captureUndoEntry('UPDATE_WORKER_DESC_FIELD', payload, STATE)!;
    expect(entry).toEqual({
      kind: 'fieldPatch', type: 'UPDATE_WORKER_DESC_FIELD',
      idPayload: { workerId: 'w1', field: '業務形態' },
      fieldsBefore: { '業務形態': undefined }, fieldsAfter: { '業務形態': 'A' },
    });
    // Apply it (STATE itself is left untouched — this is what "after" looks like).
    const after: AppState = { ...STATE, envConfig: { ...ENV, workerList: [{ ...ENV.workerList[0], description: { ...ENV.workerList[0].description, '業務形態': 'A' } }] } };
    expect(hasConflict(entry, after, 'undo')).toBe(false);
    const revert = buildRevertAction(entry, after, 'undo');
    expect(revert).toEqual({ type: 'UPDATE_WORKER_DESC_FIELD', payload: { workerId: 'w1', field: '業務形態', value: undefined } });
    // And blocked if someone else set it to something different since.
    const touchedByOther: AppState = { ...STATE, envConfig: { ...ENV, workerList: [{ ...ENV.workerList[0], description: { ...ENV.workerList[0].description, '業務形態': 'B' } }] } };
    expect(hasConflict(entry, touchedByOther, 'undo')).toBe(true);
  });
});

describe('UPDATE_ASSIGNMENT (index-based, keyed by _id)', () => {
  const payload = { index: 0, updates: { startDate: '2025-09-10' } };

  it('captures using the assignment _id, not the index', () => {
    const entry = captureUndoEntry('UPDATE_ASSIGNMENT', payload, STATE);
    expect(entry).toEqual({ kind: 'assignmentPatch', id: 'a1', fieldsBefore: { startDate: '2025-09-01' }, fieldsAfter: { startDate: '2025-09-10' } });
  });

  it('resolves the CURRENT index at revert time, even if it moved', () => {
    const entry = captureUndoEntry('UPDATE_ASSIGNMENT', payload, STATE)!;
    // Simulate another assignment having been inserted at index 0 in the meantime — a1 is now at index 1.
    const shifted: AppState = {
      ...STATE,
      schedule: {
        ...SCHEDULE,
        assignmentList: [
          { _id: 'a2', worker: 'w2', operationTask: 'ot1', startDate: '2025-09-01', endDate: '2025-09-02', planFlexibility: 'Flexible', workDateList: [] },
          { ...SCHEDULE.assignmentList[0], startDate: '2025-09-10' },
        ],
      },
    };
    expect(hasConflict(entry, shifted, 'undo')).toBe(false);
    const revert = buildRevertAction(entry, shifted, 'undo');
    expect(revert).toEqual({ type: 'UPDATE_ASSIGNMENT', payload: { index: 1, updates: { startDate: '2025-09-01' } } });
  });

  it('blocks when someone else changed the same field since', () => {
    const entry = captureUndoEntry('UPDATE_ASSIGNMENT', payload, STATE)!;
    const touched: AppState = { ...STATE, schedule: { ...SCHEDULE, assignmentList: [{ ...SCHEDULE.assignmentList[0], startDate: '2025-09-20' }] } };
    expect(hasConflict(entry, touched, 'undo')).toBe(true);
  });
});

describe('ADD_ASSIGNMENT / DELETE_ASSIGNMENT', () => {
  const newAssignment = { _id: 'a2', worker: 'w1', operationTask: 'ot1', startDate: '2025-09-06', endDate: '2025-09-07', planFlexibility: 'Flexible' as const, workDateList: [] };

  it('ADD_ASSIGNMENT undo (delete) is safe when untouched, blocked if edited since', () => {
    const entry = captureUndoEntry('ADD_ASSIGNMENT', newAssignment, STATE)!;
    const after: AppState = { ...STATE, schedule: { ...SCHEDULE, assignmentList: [...SCHEDULE.assignmentList, newAssignment] } };
    expect(hasConflict(entry, after, 'undo')).toBe(false);
    expect(buildRevertAction(entry, after, 'undo')).toEqual({ type: 'DELETE_ASSIGNMENT', payload: 1 });

    const editedSince = { ...STATE, schedule: { ...SCHEDULE, assignmentList: [...SCHEDULE.assignmentList, { ...newAssignment, startDate: '2025-09-08' }] } };
    expect(hasConflict(entry, editedSince, 'undo')).toBe(true);
  });

  it('DELETE_ASSIGNMENT undo (re-add) is always safe; redo (delete again) is blocked if edited since being restored', () => {
    const entry = captureUndoEntry('DELETE_ASSIGNMENT', 0, STATE)!;
    expect(entry).toEqual({ kind: 'assignmentDelete', id: 'a1', deleted: SCHEDULE.assignmentList[0] });
    const afterUndo: AppState = { ...STATE, schedule: { ...SCHEDULE, assignmentList: [...SCHEDULE.assignmentList] } };
    expect(hasConflict(entry, afterUndo, 'redo')).toBe(false);
    const touchedAfterUndo = { ...STATE, schedule: { ...SCHEDULE, assignmentList: [{ ...SCHEDULE.assignmentList[0], startDate: '2025-09-09' }] } };
    expect(hasConflict(entry, touchedAfterUndo, 'redo')).toBe(true);
  });
});

describe('UPDATE_PLAN_RANGE', () => {
  it('captures, checks conflict, and reverts', () => {
    const payload = { startDate: '2025-10-01', endDate: '2025-10-31' };
    const entry = captureUndoEntry('UPDATE_PLAN_RANGE', payload, STATE)!;
    expect(entry).toEqual({ kind: 'planRange', before: SCHEDULE.planRange, after: payload });
    const after: AppState = { ...STATE, schedule: { ...SCHEDULE, planRange: payload } };
    expect(hasConflict(entry, after, 'undo')).toBe(false);
    expect(buildRevertAction(entry, after, 'undo')).toEqual({ type: 'UPDATE_PLAN_RANGE', payload: SCHEDULE.planRange });
  });
});

describe('unavailable-date actions (whole-field snapshot per worker)', () => {
  it('MOVE_UNAVAILABLE_DATE captures the worker’s full unavailableDates before/after', () => {
    const payload = { workerId: 'w1', oldDate: '2025-09-01', newDate: '2025-09-02' };
    const withDate: AppState = { ...STATE, envConfig: { ...ENV, workerList: [{ ...ENV.workerList[0], unavailableDates: [{ single: { days: ['2025-09-01'] } }] }] } };
    const entry = captureUndoEntry('MOVE_UNAVAILABLE_DATE', payload, withDate)!;
    expect(entry.kind).toBe('workerUnavailable');
    if (entry.kind === 'workerUnavailable') {
      expect(entry.workerId).toBe('w1');
      expect(entry.before).toEqual([{ single: { days: ['2025-09-01'] } }]);
      expect(entry.after).toEqual([{ single: { days: ['2025-09-02'] } }]);
    }
  });

  it('blocks undo if someone else touched the same worker’s unavailable dates since', () => {
    const payload = { workerId: 'w1', oldDate: '2025-09-01', newDate: '2025-09-02' };
    const withDate: AppState = { ...STATE, envConfig: { ...ENV, workerList: [{ ...ENV.workerList[0], unavailableDates: [{ single: { days: ['2025-09-01'] } }] }] } };
    const entry = captureUndoEntry('MOVE_UNAVAILABLE_DATE', payload, withDate)!;
    const after: AppState = { ...withDate, envConfig: { ...ENV, workerList: [{ ...ENV.workerList[0], unavailableDates: [{ single: { days: ['2025-09-02'] } }] }] } };
    expect(hasConflict(entry, after, 'undo')).toBe(false);
    const touchedByOther: AppState = { ...withDate, envConfig: { ...ENV, workerList: [{ ...ENV.workerList[0], unavailableDates: [{ single: { days: ['2025-09-03'] } }] }] } };
    expect(hasConflict(entry, touchedByOther, 'undo')).toBe(true);
    const revert = buildRevertAction(entry, after, 'undo');
    expect(revert).toEqual({ type: 'RESTORE_WORKER_UNAVAILABLE_DATES', payload: { workerId: 'w1', unavailableDates: [{ single: { days: ['2025-09-01'] } }] } });
  });
});

describe('BULK_UPDATE_FLEXIBILITY', () => {
  it('captures old flexibility per affected assignment and reverts them together', () => {
    const twoAssignments: ScheduleData = { ...SCHEDULE, assignmentList: [
      { ...SCHEDULE.assignmentList[0] },
      { _id: 'a2', worker: 'w1', operationTask: 'ot1', startDate: '2025-09-06', endDate: '2025-09-07', planFlexibility: 'Fixed', workDateList: [] },
    ] };
    const state: AppState = { ...STATE, schedule: twoAssignments };
    const payload = { flexibility: 'Reluctant', target: 'all' as const };
    const entry = captureUndoEntry('BULK_UPDATE_FLEXIBILITY', payload, state)!;
    expect(entry).toEqual({ kind: 'bulkFlex', changes: [
      { assignmentId: 'a1', before: 'Flexible', after: 'Reluctant' },
      { assignmentId: 'a2', before: 'Fixed', after: 'Reluctant' },
    ] });
    const after: AppState = { ...state, schedule: { ...twoAssignments, assignmentList: twoAssignments.assignmentList.map(a => ({ ...a, planFlexibility: 'Reluctant' })) } };
    expect(hasConflict(entry, after, 'undo')).toBe(false);
    expect(buildRevertAction(entry, after, 'undo')).toEqual({
      type: 'RESTORE_ASSIGNMENT_FIELDS',
      payload: [
        { assignmentId: 'a1', updates: { planFlexibility: 'Flexible' } },
        { assignmentId: 'a2', updates: { planFlexibility: 'Fixed' } },
      ],
    });
  });

  it('blocks the whole undo if any one of the affected assignments was touched by someone else', () => {
    const twoAssignments: ScheduleData = { ...SCHEDULE, assignmentList: [
      { ...SCHEDULE.assignmentList[0] },
      { _id: 'a2', worker: 'w1', operationTask: 'ot1', startDate: '2025-09-06', endDate: '2025-09-07', planFlexibility: 'Fixed', workDateList: [] },
    ] };
    const state: AppState = { ...STATE, schedule: twoAssignments };
    const entry = captureUndoEntry('BULK_UPDATE_FLEXIBILITY', { flexibility: 'Reluctant', target: 'all' as const }, state)!;
    const partiallyTouched: AppState = { ...state, schedule: { ...twoAssignments, assignmentList: [
      { ...twoAssignments.assignmentList[0], planFlexibility: 'Reluctant' },
      { ...twoAssignments.assignmentList[1], planFlexibility: 'Fixed' }, // someone reverted this one already / never got the bulk update
    ] } };
    expect(hasConflict(entry, partiallyTouched, 'undo')).toBe(true);
  });
});

describe('ADD_WORKFLOW_TASKS', () => {
  it('captures exactly the newly-added ids (dedup already applied by the reducer)', () => {
    const payload = [{ id: 'wt2', workflow: 'wf2', phaseTaskList: [] }, { id: 'wt1', workflow: 'wf', phaseTaskList: [] }]; // wt1 already exists -> deduped
    const entry = captureUndoEntry('ADD_WORKFLOW_TASKS', payload, STATE)!;
    expect(entry).toEqual({ kind: 'addWorkflowTasks', addedIds: ['wt2'] });
  });

  it('reverts by removing exactly those ids', () => {
    const payload = [{ id: 'wt2', workflow: 'wf2', phaseTaskList: [] }];
    const entry = captureUndoEntry('ADD_WORKFLOW_TASKS', payload, STATE)!;
    const after: AppState = { ...STATE, schedule: { ...SCHEDULE, workflowTaskList: [...SCHEDULE.workflowTaskList, payload[0]] } };
    expect(hasConflict(entry, after, 'undo')).toBe(false);
    expect(buildRevertAction(entry, after, 'undo')).toEqual({ type: 'REMOVE_WORKFLOW_TASKS_BY_ID', payload: ['wt2'] });
  });
});

describe('MERGE_DATA', () => {
  it('captures newly-added ids across schedule and envConfig', () => {
    const payload = {
      schedule: { ...SCHEDULE, workflowTaskList: [{ id: 'wt2', workflow: 'wf2', phaseTaskList: [] }], assignmentList: [{ _id: 'a2', worker: 'w2', operationTask: 'ot1', startDate: '2025-09-06', endDate: '2025-09-07', planFlexibility: 'Flexible' as const, workDateList: [] }] },
      envConfig: { ...ENV, workerList: [{ id: 'w2', name: 'Worker Two', unavailableDates: [] }] },
    };
    const entry = captureUndoEntry('MERGE_DATA', payload, STATE)!;
    expect(entry.kind).toBe('mergeData');
    if (entry.kind === 'mergeData') {
      expect(entry.addedWorkflowTaskIds).toEqual(['wt2']);
      expect(entry.addedAssignmentIds).toEqual(['a2']);
      expect(entry.addedEnvConfigIds.workerList).toEqual(['w2']);
    }
  });
});
