// src/context/undoEntries.ts
//
// Captures, conflict-checks, and reverses your own edits for the
// own-action, conflict-aware undo/redo mechanism — see
// GanttChartEditor_LiveCollabEdit_UndoConflictDesign20260904.md. This file
// (plus its Task 4 extension) is the only place that knows how to
// interpret an UndoEntry; AppContext.tsx's dispatch wrapper (Task 5) just
// calls these three functions.
import { reducer } from './reducer';
import { AppState, ActionType, UndoEntry } from '../types/appState';
import { Assignment } from '../types/schedule';

function deepEqual(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

// ── Field-patch group: every action that targets one object by a stable id
// and patches one or more named fields on it (colors, phase/operation task
// fields, worker description fields). Table-driven so adding a new
// field-patch action type never needs a new switch branch in the three
// exported functions below — only a new row here.
interface FieldPatchDef {
  idPayload: (payload: any) => Record<string, unknown>;
  find: (state: AppState, idPayload: Record<string, unknown>) => Record<string, unknown> | undefined;
  toUpdates: (payload: any) => Record<string, unknown>;
  toActionPayload: (idPayload: Record<string, unknown>, updates: Record<string, unknown>) => unknown;
}

function findOperationTask(state: AppState, operationTaskId: string) {
  for (const wt of state.schedule?.workflowTaskList ?? []) {
    for (const pt of wt.phaseTaskList) {
      const ot = pt.operationTaskList.find(o => o.id === operationTaskId);
      if (ot) return ot as unknown as Record<string, unknown>;
    }
  }
  return undefined;
}

const FIELD_PATCH_DEFS: Partial<Record<ActionType['type'], FieldPatchDef>> = {
  UPDATE_PHASE_TASK: {
    idPayload: p => ({ workflowTaskId: p.workflowTaskId, phaseTaskId: p.phaseTaskId }),
    find: (s, id) => s.schedule?.workflowTaskList.find(wt => wt.id === id.workflowTaskId)
      ?.phaseTaskList.find(pt => pt.id === id.phaseTaskId) as unknown as Record<string, unknown> | undefined,
    toUpdates: p => p.updates,
    toActionPayload: (id, updates) => ({ ...id, updates }),
  },
  UPDATE_OPERATION_TASK: {
    idPayload: p => ({ workflowTaskId: p.workflowTaskId, phaseTaskId: p.phaseTaskId, operationTaskId: p.operationTaskId }),
    find: (s, id) => s.schedule?.workflowTaskList.find(wt => wt.id === id.workflowTaskId)
      ?.phaseTaskList.find(pt => pt.id === id.phaseTaskId)
      ?.operationTaskList.find(ot => ot.id === id.operationTaskId) as unknown as Record<string, unknown> | undefined,
    toUpdates: p => p.updates,
    toActionPayload: (id, updates) => ({ ...id, updates }),
  },
  UPDATE_OPERATION_TASK_COLOR: {
    idPayload: p => ({ operationTaskId: p.operationTaskId }),
    find: (s, id) => findOperationTask(s, id.operationTaskId as string),
    toUpdates: p => ({ colorCode: p.colorCode }),
    toActionPayload: (id, updates) => ({ ...id, colorCode: updates.colorCode }),
  },
  UPDATE_WORKFLOW_TASK_COLOR: {
    idPayload: p => ({ workflowTaskId: p.workflowTaskId }),
    find: (s, id) => s.schedule?.workflowTaskList.find(wt => wt.id === id.workflowTaskId) as unknown as Record<string, unknown> | undefined,
    toUpdates: p => ({ colorCode: p.colorCode }),
    toActionPayload: (id, updates) => ({ ...id, colorCode: updates.colorCode }),
  },
  UPDATE_WORKER_DEFINITION: {
    idPayload: p => ({ workerId: p.workerId }),
    find: (s, id) => s.envConfig?.workerList.find(w => w.id === id.workerId)?.description as Record<string, unknown> | undefined,
    toUpdates: p => ({ '備考': p.definition }),
    toActionPayload: (id, updates) => ({ ...id, definition: updates['備考'] }),
  },
  UPDATE_WORKER_DESC_FIELD: {
    idPayload: p => ({ workerId: p.workerId, field: p.field }),
    find: (s, id) => s.envConfig?.workerList.find(w => w.id === id.workerId)?.description as Record<string, unknown> | undefined,
    toUpdates: p => ({ [p.field]: p.value }),
    toActionPayload: (id, updates) => ({ workerId: id.workerId, field: id.field, value: updates[id.field as string] }),
  },
};

function findAssignment(state: AppState, id: string): { assignment: Assignment; index: number } | undefined {
  const index = state.schedule?.assignmentList.findIndex(a => a._id === id) ?? -1;
  if (index < 0) return undefined;
  return { assignment: state.schedule!.assignmentList[index], index };
}

export function captureUndoEntry(type: ActionType['type'], payload: unknown, before: AppState): UndoEntry | null {
  const fieldPatchDef = FIELD_PATCH_DEFS[type];
  if (fieldPatchDef) {
    const idPayload = fieldPatchDef.idPayload(payload);
    const target = fieldPatchDef.find(before, idPayload);
    if (!target) return null;
    const updates = fieldPatchDef.toUpdates(payload);
    const fieldsBefore: Record<string, unknown> = {};
    for (const key of Object.keys(updates)) fieldsBefore[key] = target[key];
    return { kind: 'fieldPatch', type, idPayload, fieldsBefore, fieldsAfter: { ...updates } };
  }

  switch (type) {
    case 'UPDATE_ASSIGNMENT': {
      const p = payload as { index: number; updates: Record<string, unknown> };
      const a = before.schedule?.assignmentList[p.index];
      if (!a?._id) return null;
      const fieldsBefore: Record<string, unknown> = {};
      for (const key of Object.keys(p.updates)) fieldsBefore[key] = (a as unknown as Record<string, unknown>)[key];
      return { kind: 'assignmentPatch', id: a._id, fieldsBefore, fieldsAfter: { ...p.updates } };
    }
    case 'DELETE_ASSIGNMENT': {
      const index = payload as number;
      const a = before.schedule?.assignmentList[index];
      if (!a?._id) return null;
      return { kind: 'assignmentDelete', id: a._id, deleted: a };
    }
    case 'ADD_ASSIGNMENT': {
      const a = payload as Assignment;
      if (!a._id) return null;
      return { kind: 'assignmentAdd', id: a._id, added: a };
    }
    case 'UPDATE_PLAN_RANGE': {
      if (!before.schedule) return null;
      return { kind: 'planRange', before: before.schedule.planRange, after: payload as { startDate: string; endDate: string } };
    }
    default:
      return null; // Task 4 adds more cases here.
  }
}

export function hasConflict(entry: UndoEntry, current: AppState, direction: 'undo' | 'redo'): boolean {
  switch (entry.kind) {
    case 'fieldPatch': {
      const def = FIELD_PATCH_DEFS[entry.type];
      if (!def) return true;
      const target = def.find(current, entry.idPayload);
      if (!target) return true;
      const expected = direction === 'undo' ? entry.fieldsAfter : entry.fieldsBefore;
      return Object.keys(expected).some(key => !deepEqual(target[key], expected[key]));
    }
    case 'assignmentPatch': {
      const found = findAssignment(current, entry.id);
      if (!found) return true;
      const expected = direction === 'undo' ? entry.fieldsAfter : entry.fieldsBefore;
      return Object.keys(expected).some(key => !deepEqual((found.assignment as unknown as Record<string, unknown>)[key], expected[key]));
    }
    case 'assignmentAdd': {
      const found = findAssignment(current, entry.id);
      if (direction === 'undo') return !found || !deepEqual(found.assignment, entry.added);
      return false; // redo = re-add; nothing to conflict with once it's gone
    }
    case 'assignmentDelete': {
      if (direction === 'undo') return false; // undo = re-add; always safe, ids are never reused
      const found = findAssignment(current, entry.id);
      return !found || !deepEqual(found.assignment, entry.deleted);
    }
    case 'planRange': {
      if (!current.schedule) return true;
      const expected = direction === 'undo' ? entry.after : entry.before;
      return !deepEqual(current.schedule.planRange, expected);
    }
    default:
      return true; // Task 4 adds more cases here; unknown kinds are conservatively blocked, never silently applied.
  }
}

export function buildRevertAction(entry: UndoEntry, current: AppState, direction: 'undo' | 'redo'): ActionType | null {
  switch (entry.kind) {
    case 'fieldPatch': {
      const def = FIELD_PATCH_DEFS[entry.type];
      if (!def) return null;
      const values = direction === 'undo' ? entry.fieldsBefore : entry.fieldsAfter;
      return { type: entry.type, payload: def.toActionPayload(entry.idPayload, values) } as ActionType;
    }
    case 'assignmentPatch': {
      const found = findAssignment(current, entry.id);
      if (!found) return null;
      const values = direction === 'undo' ? entry.fieldsBefore : entry.fieldsAfter;
      return { type: 'UPDATE_ASSIGNMENT', payload: { index: found.index, updates: values } };
    }
    case 'assignmentAdd': {
      if (direction === 'undo') {
        const found = findAssignment(current, entry.id);
        return found ? { type: 'DELETE_ASSIGNMENT', payload: found.index } : null;
      }
      return { type: 'ADD_ASSIGNMENT', payload: entry.added };
    }
    case 'assignmentDelete': {
      if (direction === 'undo') return { type: 'ADD_ASSIGNMENT', payload: entry.deleted };
      const found = findAssignment(current, entry.id);
      return found ? { type: 'DELETE_ASSIGNMENT', payload: found.index } : null;
    }
    case 'planRange': {
      if (!current.schedule) return null;
      return { type: 'UPDATE_PLAN_RANGE', payload: direction === 'undo' ? entry.before : entry.after };
    }
    default:
      return null; // Task 4 adds more cases here.
  }
}

// Exported so AppContext.tsx (Task 5) can compute "what would this action
// change" via the same oracle-call trick used internally here, without
// duplicating reducer logic — used for the compound-action capture helpers
// Task 4 adds (BULK_UPDATE_FLEXIBILITY, ADD_WORKFLOW_TASKS, MERGE_DATA,
// and the unavailable-date group), which need to diff before/after rather
// than read a single field.
export function computeAfter(state: AppState, action: ActionType): AppState {
  return reducer(state, action);
}
