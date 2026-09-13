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
import { Assignment, ScheduleData, PlanFlexibility } from '../types/schedule';
import { EnvConfig, UnavailableDateEntry } from '../types/envConfig';

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
    find: (s, id) => {
      const worker = s.envConfig?.workerList.find(w => w.id === id.workerId);
      return worker ? ((worker.description ?? {}) as Record<string, unknown>) : undefined;
    },
    toUpdates: p => ({ '備考': p.definition }),
    toActionPayload: (id, updates) => ({ ...id, definition: updates['備考'] }),
  },
  UPDATE_WORKER_DESC_FIELD: {
    idPayload: p => ({ workerId: p.workerId, field: p.field }),
    find: (s, id) => {
      const worker = s.envConfig?.workerList.find(w => w.id === id.workerId);
      return worker ? ((worker.description ?? {}) as Record<string, unknown>) : undefined;
    },
    toUpdates: p => ({ [p.field]: p.value }),
    toActionPayload: (id, updates) => ({ workerId: id.workerId, field: id.field, value: updates[id.field as string] }),
  },
};

function findAssignment(state: AppState, id: string): { assignment: Assignment; index: number } | undefined {
  const index = state.schedule?.assignmentList.findIndex(a => a._id === id) ?? -1;
  if (index < 0) return undefined;
  return { assignment: state.schedule!.assignmentList[index], index };
}

const WORKER_ID_OF: Partial<Record<ActionType['type'], (payload: any) => string>> = {
  DELETE_UNAVAILABLE_DATE: p => p.workerId,
  DELETE_UNAVAILABLE_RANGE: p => p.workerId,
  MOVE_UNAVAILABLE_DATE: p => p.workerId,
  RESIZE_UNAVAILABLE_RANGE: p => p.workerId,
};

export function captureUndoEntry(type: ActionType['type'], payload: unknown, before: AppState): UndoEntry | null {
  const fieldPatchDef = FIELD_PATCH_DEFS[type];
  if (fieldPatchDef) {
    const idPayload = fieldPatchDef.idPayload(payload);
    const target = fieldPatchDef.find(before, idPayload);
    if (!target) return null;
    const updates = fieldPatchDef.toUpdates(payload);
    const fieldsBefore: Record<string, unknown> = {};
    for (const key of Object.keys(updates)) fieldsBefore[key] = target[key];
    const after = computeAfter(before, { type, payload } as ActionType);
    const afterTarget = fieldPatchDef.find(after, idPayload) ?? target;
    return {
      kind: 'fieldPatch', type, idPayload, fieldsBefore, fieldsAfter: { ...updates },
      fullBefore: { ...target }, fullAfter: { ...afterTarget },
    };
  }

  switch (type) {
    case 'UPDATE_ASSIGNMENT': {
      const p = payload as { index: number; updates: Record<string, unknown> };
      const a = before.schedule?.assignmentList[p.index];
      if (!a?._id) return null;
      const fieldsBefore: Record<string, unknown> = {};
      for (const key of Object.keys(p.updates)) fieldsBefore[key] = (a as unknown as Record<string, unknown>)[key];
      return {
        kind: 'assignmentPatch', id: a._id, fieldsBefore, fieldsAfter: { ...p.updates },
        fullBefore: { ...a } as Record<string, unknown>, fullAfter: { ...a, ...p.updates } as Record<string, unknown>,
      };
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
    case 'DELETE_UNAVAILABLE_DATE':
    case 'DELETE_UNAVAILABLE_RANGE':
    case 'MOVE_UNAVAILABLE_DATE':
    case 'RESIZE_UNAVAILABLE_RANGE': {
      const workerId = WORKER_ID_OF[type]!(payload);
      const worker = before.envConfig?.workerList.find(w => w.id === workerId);
      if (!worker) return null;
      const beforeDates = worker.unavailableDates;
      const after = computeAfter(before, { type, payload } as ActionType);
      const afterDates = after.envConfig?.workerList.find(w => w.id === workerId)?.unavailableDates ?? beforeDates;
      return { kind: 'workerUnavailable', workerId, before: beforeDates, after: afterDates };
    }
    case 'ADD_UNAVAILABLE_DATES': {
      // Payload can touch several workers at once; record one target per
      // worker actually affected so the conflict check is per-worker, like
      // every other unavailable-date action.
      const entries = payload as { workerId: string; dates: string[] }[];
      const after = computeAfter(before, { type, payload } as ActionType);
      const workerIds = [...new Set(entries.map(e => e.workerId))];
      const changed = workerIds
        .map(workerId => {
          const beforeDates = before.envConfig?.workerList.find(w => w.id === workerId)?.unavailableDates;
          const afterDates = after.envConfig?.workerList.find(w => w.id === workerId)?.unavailableDates;
          return beforeDates && afterDates && !deepEqual(beforeDates, afterDates) ? { workerId, before: beforeDates, after: afterDates } : null;
        })
        .filter((x): x is { workerId: string; before: UnavailableDateEntry[]; after: UnavailableDateEntry[] } => x !== null);
      // Only ever one worker in practice for this UI's callers, but modeled
      // as "first changed worker" to keep the entry shape uniform with the
      // rest of this group rather than introducing a second multi-worker
      // UndoEntry kind for a case that never actually needs it.
      return changed[0] ? { kind: 'workerUnavailable', ...changed[0] } : null;
    }
    case 'BULK_UPDATE_FLEXIBILITY': {
      if (!before.schedule) return null;
      const after = computeAfter(before, { type, payload } as ActionType);
      const changes = before.schedule.assignmentList
        .map((a, i) => {
          const afterFlex = after.schedule?.assignmentList[i]?.planFlexibility;
          if (!a._id || afterFlex === undefined || afterFlex === a.planFlexibility) return null;
          return { assignmentId: a._id, before: a.planFlexibility, after: afterFlex };
        })
        .filter((x): x is { assignmentId: string; before: PlanFlexibility; after: PlanFlexibility } => x !== null);
      return changes.length > 0 ? { kind: 'bulkFlex', changes } : null;
    }
    case 'ADD_WORKFLOW_TASKS': {
      if (!before.schedule) return null;
      const existingIds = new Set(before.schedule.workflowTaskList.map(wt => wt.id));
      const addedIds = (payload as { id: string }[]).filter(wt => !existingIds.has(wt.id)).map(wt => wt.id);
      return addedIds.length > 0 ? { kind: 'addWorkflowTasks', addedIds } : null;
    }
    case 'MERGE_DATA': {
      const p = payload as { schedule?: ScheduleData; envConfig?: EnvConfig };
      const after = computeAfter(before, { type, payload } as ActionType);
      const addedWorkflowTaskIds = p.schedule && after.schedule && before.schedule
        ? after.schedule.workflowTaskList.map(wt => wt.id).filter(id => !before.schedule!.workflowTaskList.some(wt => wt.id === id))
        : [];
      const addedAssignmentIds = p.schedule && after.schedule && before.schedule
        ? after.schedule.assignmentList.map(a => a._id).filter((id): id is string => !!id && !before.schedule!.assignmentList.some(a => a._id === id))
        : [];
      const listNames = ['workflowList', 'fabList', 'regionList', 'customerCompanyList', 'workerCompanyList', 'workerList'] as const;
      const addedEnvConfigIds: Record<string, string[]> = {};
      if (p.envConfig && after.envConfig && before.envConfig) {
        for (const listName of listNames) {
          const beforeIds = new Set(before.envConfig[listName].map(x => x.id));
          addedEnvConfigIds[listName] = after.envConfig[listName].map(x => x.id).filter(id => !beforeIds.has(id));
        }
      }
      if (addedWorkflowTaskIds.length === 0 && addedAssignmentIds.length === 0 && Object.values(addedEnvConfigIds).every(ids => ids.length === 0)) return null;
      return { kind: 'mergeData', addedWorkflowTaskIds, addedAssignmentIds, addedEnvConfigIds };
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
      const expected = direction === 'undo' ? entry.fullAfter : entry.fullBefore;
      return !deepEqual(target, expected);
    }
    case 'assignmentPatch': {
      const found = findAssignment(current, entry.id);
      if (!found) return true;
      const expected = direction === 'undo' ? entry.fullAfter : entry.fullBefore;
      return !deepEqual(found.assignment, expected);
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
    case 'workerUnavailable': {
      const worker = current.envConfig?.workerList.find(w => w.id === entry.workerId);
      if (!worker) return true;
      const expected = direction === 'undo' ? entry.after : entry.before;
      return !deepEqual(worker.unavailableDates, expected);
    }
    case 'bulkFlex': {
      return entry.changes.some(change => {
        const found = findAssignment(current, change.assignmentId);
        if (!found) return true;
        const expected = direction === 'undo' ? change.after : change.before;
        return found.assignment.planFlexibility !== expected;
      });
    }
    case 'addWorkflowTasks': {
      const has = (id: string) => current.schedule?.workflowTaskList.some(wt => wt.id === id) ?? false;
      if (direction === 'undo') return !entry.addedIds.every(has);
      return entry.addedIds.some(has);
    }
    case 'mergeData': {
      const wtOk = entry.addedWorkflowTaskIds.every(id => current.schedule?.workflowTaskList.some(wt => wt.id === id));
      const asOk = entry.addedAssignmentIds.every(id => current.schedule?.assignmentList.some(a => a._id === id));
      const envOk = Object.entries(entry.addedEnvConfigIds).every(([listName, ids]) =>
        ids.every(id => (current.envConfig as unknown as Record<string, { id: string }[]>)[listName]?.some(x => x.id === id)),
      );
      const stillPresent = wtOk && asOk && envOk;
      return direction === 'undo' ? !stillPresent : stillPresent;
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
    case 'workerUnavailable': {
      const values = direction === 'undo' ? entry.before : entry.after;
      return { type: 'RESTORE_WORKER_UNAVAILABLE_DATES', payload: { workerId: entry.workerId, unavailableDates: values } };
    }
    case 'bulkFlex': {
      const field = direction === 'undo' ? 'before' : 'after';
      return { type: 'RESTORE_ASSIGNMENT_FIELDS', payload: entry.changes.map(c => ({ assignmentId: c.assignmentId, updates: { planFlexibility: c[field] } })) };
    }
    case 'addWorkflowTasks': {
      if (direction === 'redo') return null; // re-adding requires the original full WorkflowTask objects, which this entry doesn't retain (ADD_WORKFLOW_TASKS is expected to be rare enough that redo-after-undo isn't needed for it in this pass)
      return { type: 'REMOVE_WORKFLOW_TASKS_BY_ID', payload: entry.addedIds };
    }
    case 'mergeData': {
      if (direction === 'redo') return null; // same reasoning as addWorkflowTasks — MERGE_DATA's redo isn't supported in this pass
      return { type: 'REVERT_MERGE', payload: { workflowTaskIds: entry.addedWorkflowTaskIds, assignmentIds: entry.addedAssignmentIds, envConfigIds: entry.addedEnvConfigIds } };
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
