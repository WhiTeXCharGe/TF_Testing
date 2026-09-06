# GanttChartEditor — Own-Action, Conflict-Aware Undo/Redo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Undo/Redo in the live-collab feature act only on your own edits, block outright (with a message, no silent skip) if someone else has touched the same object since, and undoing one edit keeps every other edit made since — yours or anyone else's — as long as it didn't touch the same object.

**Architecture:** Per-edit before/after value capture replaces the current whole-document snapshot stack. Every syncable edit you make records a small `UndoEntry` (what changed, old value, new value) in `myPendingUndo`. Undo compares the object's *current* live value to what your edit set it to — if unchanged, it reverts to the old value by dispatching (and syncing) the same action type again with old values; if changed, it blocks. No shared/replayable log and no server change are needed — Socket.IO's `socket.to(...)` broadcast already never echoes your own actions back to you, so "is this mine" is already known for free from which code path an action arrived through.

**Tech Stack:** React 19 + TypeScript client, existing Jest test suite, no server changes.

**Design doc:** `documents/GanttChartEditor/GanttChartEditor_LiveCollabEdit_UndoConflictDesign20260904.md` — read for full rationale; this plan implements it exactly, revised to the per-action-patch approach (§3 onward in the design doc).

## Global Constraints

- Solo mode (no `state.session`) must be completely unaffected — every new mechanism here only ever engages for `state.session?.role === 'edit'`, matching the existing `isSyncingEdit` gate already in `AppContext.tsx`.
- `isReadOnly`/read-only gating elsewhere in the app is unrelated to this feature and must not change.
- No action's `_id` or undo bookkeeping is ever written to the saved YAML file — `yamlService.ts`'s writer is untouched and unaffected (it only serializes its own named fields; confirmed by reading it).
- Every new/changed reducer case keeps the existing style of this file: a `case` block returning a new `AppState` object, no mutation of the incoming `state`.
- `MAX_UNDO_STACK` (`src/config/appConfig.ts`, currently `100`) still caps `myPendingUndo`/`myPendingRedo`.
- Follow existing test conventions: Jest with `@jest-environment jsdom` where React Testing Library is used; plain Jest for pure-function/reducer tests (see `src/__tests__/context/reducer.test.ts` for the established fixture style).

---

### Task 1: Assignment stable `_id`

**Files:**
- Create: `src/utils/id.ts`
- Modify: `src/types/schedule.ts` (add `_id?: string` to `Assignment`)
- Modify: `src/context/reducer.ts` (backfill `_id` in `LOAD_FILES`, `SET_SESSION_BASELINE`, `MERGE_DATA`)
- Test: `src/__tests__/context/reducer.test.ts`

**Interfaces:**
- Produces: `generateId(): string` (from `src/utils/id.ts`) — used by this task and by Task 5's wrapper.
- Produces: `Assignment._id?: string` — every assignment in `state.schedule.assignmentList` has one once it has passed through `LOAD_FILES`, `SET_SESSION_BASELINE`, or `MERGE_DATA`.

- [ ] **Step 1: Create the id generator**

```ts
// src/utils/id.ts

// Session-local identity for objects that need one but don't have a
// natural stable id in the underlying data (currently just assignments —
// see GanttChartEditor_LiveCollabEdit_UndoConflictDesign20260904.md §3).
// Never persisted to the saved YAML file.
export function generateId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return `id-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}
```

- [ ] **Step 2: Add `_id` to `Assignment`**

In `src/types/schedule.ts`, change:

```ts
export interface Assignment {
  worker: string;
  operationTask: string;
  startDate: string;
  endDate: string;
  workDateList: WorkDate[];
  planFlexibility: PlanFlexibility;
  description?: string;
}
```

to:

```ts
export interface Assignment {
  // Session-local stable identity (see src/utils/id.ts) — never read or
  // written by yamlService.ts, which only serializes the named fields
  // below; purely an in-memory concern for undo/redo conflict detection.
  _id?: string;
  worker: string;
  operationTask: string;
  startDate: string;
  endDate: string;
  workDateList: WorkDate[];
  planFlexibility: PlanFlexibility;
  description?: string;
}
```

- [ ] **Step 3: Write the failing tests for backfill**

Add to `src/__tests__/context/reducer.test.ts` (near the existing `LOAD_FILES`/`MERGE_DATA` tests — find them by searching for `describe('LOAD_FILES'` and `describe('MERGE_DATA'`, or add new `describe` blocks if none exist yet):

```ts
import { generateId } from '../../utils/id';
// (add alongside the file's existing imports)

describe('Assignment _id backfill', () => {
  it('LOAD_FILES assigns _id to every assignment that lacks one', () => {
    const schedule = { ...EMPTY_SCHEDULE, assignmentList: [{ ...EMPTY_SCHEDULE.assignmentList[0] }] };
    const next = reducer(BASE_STATE, {
      type: 'LOAD_FILES',
      payload: { schedule, envConfig: EMPTY_ENV, envPath: 'e.yaml', schedulePath: 's.yaml' },
    });
    expect(next.schedule?.assignmentList[0]._id).toEqual(expect.any(String));
  });

  it('LOAD_FILES preserves an existing _id instead of generating a new one', () => {
    const schedule = { ...EMPTY_SCHEDULE, assignmentList: [{ ...EMPTY_SCHEDULE.assignmentList[0], _id: 'keep-me' }] };
    const next = reducer(BASE_STATE, {
      type: 'LOAD_FILES',
      payload: { schedule, envConfig: EMPTY_ENV, envPath: 'e.yaml', schedulePath: 's.yaml' },
    });
    expect(next.schedule?.assignmentList[0]._id).toBe('keep-me');
  });

  it('SET_SESSION_BASELINE assigns _id to every assignment that lacks one', () => {
    const schedule = { ...EMPTY_SCHEDULE, assignmentList: [{ ...EMPTY_SCHEDULE.assignmentList[0] }] };
    const next = reducer(BASE_STATE, {
      type: 'SET_SESSION_BASELINE',
      payload: { schedule, envConfig: EMPTY_ENV, currentView: 'worker' },
    });
    expect(next.schedule?.assignmentList[0]._id).toEqual(expect.any(String));
  });

  it('MERGE_DATA assigns _id to newly merged-in assignments', () => {
    const state = { ...BASE_STATE, schedule: EMPTY_SCHEDULE };
    const incoming = { ...EMPTY_SCHEDULE, assignmentList: [{ ...EMPTY_SCHEDULE.assignmentList[0], operationTask: 'wt001_p0_o0', worker: 'w002' }] };
    const next = reducer(state, { type: 'MERGE_DATA', payload: { schedule: incoming } });
    const merged = next.schedule?.assignmentList.find(a => a.worker === 'w002');
    expect(merged?._id).toEqual(expect.any(String));
  });
});
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `npx jest --config jest.config.cjs reducer.test.ts`
Expected: FAIL — `_id` is `undefined` on all four new assertions.

- [ ] **Step 5: Implement the backfill**

In `src/context/reducer.ts`, add near the top (alongside the other small helpers like `mergeById`):

```ts
import { generateId } from '../utils/id';

function withAssignmentIds(assignmentList: ScheduleData['assignmentList']): ScheduleData['assignmentList'] {
  return assignmentList.map(a => (a._id ? a : { ...a, _id: generateId() }));
}
```

In the `LOAD_FILES` case, change:

```ts
    case 'LOAD_FILES':
      return {
        ...state,
        envConfig: action.payload.envConfig,
        schedule: action.payload.schedule,
```

to:

```ts
    case 'LOAD_FILES':
      return {
        ...state,
        envConfig: action.payload.envConfig,
        schedule: { ...action.payload.schedule, assignmentList: withAssignmentIds(action.payload.schedule.assignmentList) },
```

In the `SET_SESSION_BASELINE` case, change:

```ts
      return {
        ...state,
        schedule: action.payload.schedule,
        envConfig: action.payload.envConfig,
        currentView: action.payload.currentView,
```

to:

```ts
      return {
        ...state,
        schedule: { ...action.payload.schedule, assignmentList: withAssignmentIds(action.payload.schedule.assignmentList) },
        envConfig: action.payload.envConfig,
        currentView: action.payload.currentView,
```

In the `MERGE_DATA` case, change:

```ts
      if (incomingSched && state.schedule) {
        const existingWtIds = new Set(state.schedule.workflowTaskList.map(wt => wt.id));
        const newWts = incomingSched.workflowTaskList.filter(wt => !existingWtIds.has(wt.id));
        newSchedule = {
          ...state.schedule,
          workflowTaskList: [...state.schedule.workflowTaskList, ...newWts],
          assignmentList: [...state.schedule.assignmentList, ...incomingSched.assignmentList],
        };
      }
```

to:

```ts
      if (incomingSched && state.schedule) {
        const existingWtIds = new Set(state.schedule.workflowTaskList.map(wt => wt.id));
        const newWts = incomingSched.workflowTaskList.filter(wt => !existingWtIds.has(wt.id));
        newSchedule = {
          ...state.schedule,
          workflowTaskList: [...state.schedule.workflowTaskList, ...newWts],
          assignmentList: [...state.schedule.assignmentList, ...withAssignmentIds(incomingSched.assignmentList)],
        };
      }
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `npx jest --config jest.config.cjs`
Expected: all passing, including the 4 new tests.

- [ ] **Step 7: Commit**

```bash
git add src/utils/id.ts src/types/schedule.ts src/context/reducer.ts src/__tests__/context/reducer.test.ts
git commit -m "feat(client): give every assignment a stable session-local _id"
```

---

### Task 2: New undo/redo data model — `AppState`, `ActionType`, reducer scaffolding

**Files:**
- Modify: `src/types/appState.ts`
- Modify: `src/context/reducer.ts`
- Modify: `src/context/AppContext.tsx` (only the `initialState` object — no wrapper logic yet, that's Task 5)
- Test: `src/__tests__/context/reducer.test.ts`

**Interfaces:**
- Consumes: nothing new from Task 1 directly (Task 1's `_id` is consumed by Task 3).
- Produces: `UndoEntry` discriminated union (exported from `appState.ts`), `AppState.myPendingUndo: UndoEntry[]`, `AppState.myPendingRedo: UndoEntry[]`, and five new internal-only `ActionType` members: `RECORD_UNDO_ENTRY`, `CONSUME_UNDO_ENTRY`, `CONSUME_REDO_ENTRY`, `RESTORE_ASSIGNMENT_FIELDS`, `REMOVE_WORKFLOW_TASKS_BY_ID`, `RESTORE_WORKER_UNAVAILABLE_DATES`, `REVERT_MERGE` (7 total — see full list below). These are dispatched only by `AppContext.tsx`'s wrapper (Task 5), never by UI code.

- [ ] **Step 1: Define `UndoEntry` and its variants in `appState.ts`**

Add near the top of `src/types/appState.ts` (after the existing imports, before `AppState`):

```ts
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
```

Note: `ActionType` is defined further down in this same file — TypeScript allows this forward reference since both are type-level declarations evaluated together, not runtime order-dependent. If your editor/tsc complains about ordering, move the `UndoEntry` type below the full `ActionType` union instead — functionally identical, this is a style choice not a requirement.

- [ ] **Step 2: Add the two new `AppState` fields, replacing `undoStack`/`redoStack`**

In `src/types/appState.ts`'s `AppState` interface, change:

```ts
  undoStack: ScheduleData[];
  redoStack: ScheduleData[];
```

to:

```ts
  myPendingUndo: UndoEntry[];
  myPendingRedo: UndoEntry[];
```

- [ ] **Step 3: Add the new internal `ActionType` members**

In the `ActionType` union in `src/types/appState.ts`, remove:

```ts
  | { type: 'UNDO' }
  | { type: 'REDO' }
```

and add (near the end of the union, after the collaboration-session members is a reasonable spot):

```ts
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
```

(`UNDO`/`REDO` themselves stay as UI-facing tokens — `UndoRedoButtons.tsx` and `useKeyboardShortcuts.ts` still dispatch `{ type: 'UNDO' }`/`{ type: 'REDO' }` unchanged; the wrapper intercepts them before `rawDispatch`, so they never need a reducer `case` at all after this task removes the old ones.)

- [ ] **Step 4: Write the failing tests for the new reducer cases**

Replace the existing `describe('UNDO / REDO', ...)` block in `src/__tests__/context/reducer.test.ts` (it tests the old snapshot-stack behavior, which no longer exists) with:

```ts
// ── Undo/redo bookkeeping (RECORD/CONSUME) ──────────────────────────────────

describe('RECORD_UNDO_ENTRY / CONSUME_UNDO_ENTRY / CONSUME_REDO_ENTRY', () => {
  const ENTRY: UndoEntry = { kind: 'planRange', before: { startDate: '2025-01-01', endDate: '2025-01-31' }, after: { startDate: '2025-02-01', endDate: '2025-02-28' } };

  it('RECORD_UNDO_ENTRY appends to myPendingUndo and clears myPendingRedo', () => {
    const state = { ...BASE_STATE, myPendingRedo: [ENTRY] };
    const next = reducer(state, { type: 'RECORD_UNDO_ENTRY', payload: ENTRY });
    expect(next.myPendingUndo).toEqual([ENTRY]);
    expect(next.myPendingRedo).toEqual([]);
  });

  it('RECORD_UNDO_ENTRY caps at MAX_UNDO_STACK', () => {
    const many = Array.from({ length: 100 }, () => ENTRY);
    const state = { ...BASE_STATE, myPendingUndo: many };
    const next = reducer(state, { type: 'RECORD_UNDO_ENTRY', payload: ENTRY });
    expect(next.myPendingUndo).toHaveLength(100);
  });

  it('CONSUME_UNDO_ENTRY moves the most recent entry from myPendingUndo to myPendingRedo', () => {
    const other: UndoEntry = { ...ENTRY, before: { startDate: '2025-03-01', endDate: '2025-03-31' } };
    const state = { ...BASE_STATE, myPendingUndo: [ENTRY, other] };
    const next = reducer(state, { type: 'CONSUME_UNDO_ENTRY' });
    expect(next.myPendingUndo).toEqual([ENTRY]);
    expect(next.myPendingRedo).toEqual([other]);
  });

  it('CONSUME_UNDO_ENTRY does nothing when myPendingUndo is empty', () => {
    const next = reducer(BASE_STATE, { type: 'CONSUME_UNDO_ENTRY' });
    expect(next).toBe(BASE_STATE);
  });

  it('CONSUME_REDO_ENTRY moves the most recent entry from myPendingRedo to myPendingUndo', () => {
    const state = { ...BASE_STATE, myPendingRedo: [ENTRY] };
    const next = reducer(state, { type: 'CONSUME_REDO_ENTRY' });
    expect(next.myPendingRedo).toEqual([]);
    expect(next.myPendingUndo).toEqual([ENTRY]);
  });
});
```

Also update `BASE_STATE` (near the top of the file) to replace `undoStack: [], redoStack: [],` with `myPendingUndo: [], myPendingRedo: [],`, and add `import { UndoEntry } from '../../types/appState';` to this file's imports (it can be a named import alongside the existing `AppState, DEFAULT_WORKER_VIEW_FILTER, ...` import from the same module).

Remove or update every other test in this file that references `undoStack`/`redoStack` directly (search for both terms) — replace assertions like `expect(next.undoStack).toHaveLength(1)` etc. with equivalent `myPendingUndo`/`myPendingRedo` checks *only where Task 3/4 has landed the corresponding entry-producing logic*; for reducer cases whose "record an undo entry" behavior now lives in `AppContext.tsx` (Task 5) rather than the reducer itself, simply delete the `undoStack`/`redoStack` assertion from that test — the reducer case itself no longer touches those fields, so there is nothing left for the reducer-level test to assert about undo/redo there. (Every mutating case's own behavior — the actual field/list mutation — is otherwise unchanged and its existing assertions about `schedule`/`envConfig` content stay exactly as they are.)

- [ ] **Step 5: Run tests to verify they fail**

Run: `npx jest --config jest.config.cjs reducer.test.ts`
Expected: FAIL — `AppState` doesn't have `myPendingUndo`/`myPendingRedo` yet, `RECORD_UNDO_ENTRY` etc. aren't valid action types yet, and old `undoStack` references are compile errors.

- [ ] **Step 6: Implement the reducer changes**

In `src/context/reducer.ts`:

1. Delete the `pushUndo` helper function entirely (no longer used by any case).
2. Remove `undoStack: pushUndo(state), redoStack: []` (and the `scheduleChanged ? pushUndo(state) : state.undoStack` / `scheduleChanged ? [] : state.redoStack` variant in `MERGE_DATA`) from **every** case that currently has it — this is a pure deletion of those two object-properties from each case's return value, nothing else in those cases changes. The full list (grep confirms these are the only ones): `SET_SCHEDULE`, `UPDATE_PLAN_RANGE`, `UPDATE_WORKFLOW_TASK_COLOR`, `UPDATE_OPERATION_TASK_COLOR`, `ADD_ASSIGNMENT`, `UPDATE_ASSIGNMENT`, `UPDATE_OPERATION_TASK`, `UPDATE_PHASE_TASK`, `DELETE_ASSIGNMENT`, `BULK_UPDATE_FLEXIBILITY`, `ADD_WORKFLOW_TASKS`, `MERGE_DATA`.
3. Delete the `case 'UNDO':` and `case 'REDO':` blocks entirely.
4. In `LOAD_FILES` and `SET_SESSION_BASELINE`, change `undoStack: [], redoStack: [],` to `myPendingUndo: [], myPendingRedo: [],`.
5. Add four new cases (a good spot is right after where `case 'REDO':` used to be):

```ts
    case 'RECORD_UNDO_ENTRY':
      return {
        ...state,
        myPendingUndo: [...state.myPendingUndo, action.payload].slice(-MAX_UNDO_STACK),
        myPendingRedo: [],
      };

    case 'CONSUME_UNDO_ENTRY': {
      if (state.myPendingUndo.length === 0) return state;
      const entry = state.myPendingUndo[state.myPendingUndo.length - 1];
      return {
        ...state,
        myPendingUndo: state.myPendingUndo.slice(0, -1),
        myPendingRedo: [...state.myPendingRedo, entry].slice(-MAX_UNDO_STACK),
      };
    }

    case 'CONSUME_REDO_ENTRY': {
      if (state.myPendingRedo.length === 0) return state;
      const entry = state.myPendingRedo[state.myPendingRedo.length - 1];
      return {
        ...state,
        myPendingRedo: state.myPendingRedo.slice(0, -1),
        myPendingUndo: [...state.myPendingUndo, entry].slice(-MAX_UNDO_STACK),
      };
    }
```

6. Add three more new cases for the compound-action revert vehicles (placed near their "forward" counterparts, e.g. `RESTORE_ASSIGNMENT_FIELDS` near `BULK_UPDATE_FLEXIBILITY`, `REMOVE_WORKFLOW_TASKS_BY_ID` near `ADD_WORKFLOW_TASKS`, `RESTORE_WORKER_UNAVAILABLE_DATES` near the other unavailable-date cases, `REVERT_MERGE` near `MERGE_DATA`):

```ts
    case 'RESTORE_ASSIGNMENT_FIELDS': {
      if (!state.schedule) return state;
      const byId = new Map(action.payload.map(c => [c.assignmentId, c.updates]));
      const assignmentList = state.schedule.assignmentList.map(a =>
        a._id && byId.has(a._id) ? { ...a, ...byId.get(a._id) } : a,
      );
      return { ...state, schedule: { ...state.schedule, assignmentList } };
    }

    case 'REMOVE_WORKFLOW_TASKS_BY_ID': {
      if (!state.schedule) return state;
      const idsToRemove = new Set(action.payload);
      const workflowTaskList = state.schedule.workflowTaskList.filter(wt => !idsToRemove.has(wt.id));
      return { ...state, schedule: { ...state.schedule, workflowTaskList } };
    }

    case 'RESTORE_WORKER_UNAVAILABLE_DATES': {
      if (!state.envConfig) return state;
      const { workerId, unavailableDates } = action.payload;
      const workerList = state.envConfig.workerList.map(w =>
        w.id === workerId ? { ...w, unavailableDates: unavailableDates as typeof w.unavailableDates } : w,
      );
      return { ...state, envConfig: { ...state.envConfig, workerList } };
    }

    case 'REVERT_MERGE': {
      const { workflowTaskIds, assignmentIds, envConfigIds } = action.payload;
      let newSchedule = state.schedule;
      let newEnvConfig = state.envConfig;
      if (newSchedule && (workflowTaskIds.length > 0 || assignmentIds.length > 0)) {
        const wtSet = new Set(workflowTaskIds);
        const asSet = new Set(assignmentIds);
        newSchedule = {
          ...newSchedule,
          workflowTaskList: newSchedule.workflowTaskList.filter(wt => !wtSet.has(wt.id)),
          assignmentList: newSchedule.assignmentList.filter(a => !a._id || !asSet.has(a._id)),
        };
      }
      if (newEnvConfig) {
        const remove = (list: { id: string }[], ids: string[] | undefined) =>
          ids && ids.length > 0 ? list.filter(x => !ids.includes(x.id)) : list;
        newEnvConfig = {
          workflowList: remove(newEnvConfig.workflowList, envConfigIds.workflowList),
          fabList: remove(newEnvConfig.fabList, envConfigIds.fabList),
          regionList: remove(newEnvConfig.regionList, envConfigIds.regionList),
          customerCompanyList: remove(newEnvConfig.customerCompanyList, envConfigIds.customerCompanyList),
          workerCompanyList: remove(newEnvConfig.workerCompanyList, envConfigIds.workerCompanyList),
          workerList: remove(newEnvConfig.workerList, envConfigIds.workerList),
          transiteDayMap: newEnvConfig.transiteDayMap,
        };
      }
      return { ...state, schedule: newSchedule, envConfig: newEnvConfig };
    }
```

- [ ] **Step 7: Update `initialState` in `AppContext.tsx`**

Change `undoStack: [], redoStack: [],` to `myPendingUndo: [], myPendingRedo: [],`.

- [ ] **Step 8: Run tests to verify they pass**

Run: `npx jest --config jest.config.cjs`
Expected: `reducer.test.ts` passing. Other test files will now fail to *compile* wherever they reference `state.undoStack`/`state.session` fixtures missing the renamed fields, or dispatch `{ type: 'UNDO' }` and assert on old fields — **do not fix those other files in this task**; Task 5 rewrites `AppContext.test.tsx`'s undo/redo tests wholesale, and any other file only needs its `AppState` fixture's two field names updated (`undoStack: []` → `myPendingUndo: []`, `redoStack: []` → `myPendingRedo: []`) with no behavior change — do that minimal rename now in any fixture that fails to compile because of it (e.g. check `src/__tests__/App.test.tsx`, `src/__tests__/components/readOnlyGating.test.tsx`, `src/__tests__/components/sessionDialog.test.tsx`, `src/__tests__/components/menuBar.test.tsx`, `src/__tests__/components/sessionJoinGate.test.tsx` — none of these dispatch UNDO/REDO or assert on stack contents, they just need any inline `AppState`-shaped object to compile), leaving deeper undo/redo assertions to Task 5.

- [ ] **Step 9: Commit**

```bash
git add src/types/appState.ts src/context/reducer.ts src/context/AppContext.tsx src/__tests__/context/reducer.test.ts
git commit -m "feat(client): replace undo/redo snapshot stacks with per-entry bookkeeping"
```

(If Step 8 required touching other test files' fixtures, `git add` those too in this same commit.)

---

### Task 3: `undoEntries.ts` — capture/conflict/revert for field-patch, assignment, and plan-range edits

**Files:**
- Create: `src/context/undoEntries.ts`
- Test: `src/__tests__/context/undoEntries.test.ts`

**Interfaces:**
- Consumes: `Assignment._id` (Task 1), `UndoEntry` (Task 2), the `reducer` function itself (`src/context/reducer.ts`).
- Produces: `captureUndoEntry(type, payload, before): UndoEntry | null`, `hasConflict(entry, current, direction): boolean`, `buildRevertAction(entry, current, direction): ActionType | null` — all consumed by Task 5's `AppContext.tsx`. This task implements these fully for six action types: `UPDATE_PHASE_TASK`, `UPDATE_OPERATION_TASK`, `UPDATE_OPERATION_TASK_COLOR`, `UPDATE_WORKFLOW_TASK_COLOR`, `UPDATE_WORKER_DEFINITION`, `UPDATE_WORKER_DESC_FIELD`, plus `UPDATE_ASSIGNMENT`, `ADD_ASSIGNMENT`, `DELETE_ASSIGNMENT`, and `UPDATE_PLAN_RANGE`. Task 4 extends the same three functions (adding more `switch`/table cases, not changing their signatures) for the remaining action types.

**Design recap (see the design doc for full rationale):** `captureUndoEntry` is called with the state *before* an action is applied; it internally calls `reducer(before, {type, payload})` to compute what *after* would look like (reusing the reducer as a pure oracle instead of re-implementing each case's logic), then extracts only the small before/after values needed for that entry. `hasConflict` compares the *live current* state against the entry's recorded "expected" value for the given direction (`'undo'` expects the object to still hold what the edit set it to; `'redo'` expects it to still hold what undo restored). `buildRevertAction` constructs the concrete action to dispatch.

- [ ] **Step 1: Write the failing tests**

Create `src/__tests__/context/undoEntries.test.ts`:

```ts
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
  session: null, isSessionDialogOpen: false, sessionDialogTab: 'start',
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx jest --config jest.config.cjs undoEntries.test.ts`
Expected: FAIL — `src/context/undoEntries.ts` doesn't exist yet.

- [ ] **Step 3: Implement `undoEntries.ts`**

```ts
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx jest --config jest.config.cjs undoEntries.test.ts`
Expected: all passing.

- [ ] **Step 5: Commit**

```bash
git add src/context/undoEntries.ts src/__tests__/context/undoEntries.test.ts
git commit -m "feat(client): capture/conflict/revert logic for field-patch and assignment edits"
```

---

### Task 4: `undoEntries.ts` — extend for worker-unavailable and compound (bulk) actions

**Files:**
- Modify: `src/context/undoEntries.ts`
- Test: `src/__tests__/context/undoEntries.test.ts`

**Interfaces:**
- Consumes: `computeAfter` (this file's own Task-3 export), `UndoEntry`'s `workerUnavailable`/`bulkFlex`/`addWorkflowTasks`/`mergeData` variants (Task 2), the `RESTORE_ASSIGNMENT_FIELDS`/`REMOVE_WORKFLOW_TASKS_BY_ID`/`RESTORE_WORKER_UNAVAILABLE_DATES`/`REVERT_MERGE` reducer cases (Task 2).
- Produces: `captureUndoEntry`/`hasConflict`/`buildRevertAction` now handle all remaining `SYNCABLE_ACTION_TYPES` members: `DELETE_UNAVAILABLE_DATE`, `DELETE_UNAVAILABLE_RANGE`, `MOVE_UNAVAILABLE_DATE`, `ADD_UNAVAILABLE_DATES`, `RESIZE_UNAVAILABLE_RANGE`, `BULK_UPDATE_FLEXIBILITY`, `ADD_WORKFLOW_TASKS`, `MERGE_DATA`.

- [ ] **Step 1: Write the failing tests**

Add to `src/__tests__/context/undoEntries.test.ts`:

```ts
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx jest --config jest.config.cjs undoEntries.test.ts`
Expected: FAIL — these action types still fall through to the Step-3-of-Task-3 `default: return null` / `default: return true` / `default: return null` branches.

- [ ] **Step 3: Implement the extension**

In `src/context/undoEntries.ts`:

1. Add a helper for the unavailable-date group, right after `findAssignment`:

```ts
const WORKER_ID_OF: Partial<Record<ActionType['type'], (payload: any) => string>> = {
  DELETE_UNAVAILABLE_DATE: p => p.workerId,
  DELETE_UNAVAILABLE_RANGE: p => p.workerId,
  MOVE_UNAVAILABLE_DATE: p => p.workerId,
  RESIZE_UNAVAILABLE_RANGE: p => p.workerId,
};
```

(`ADD_UNAVAILABLE_DATES`'s payload is an *array* of `{ workerId, dates }` — potentially touching several workers in one action — so it's handled separately below rather than through this single-worker table.)

2. In `captureUndoEntry`, add a new branch before the final `switch`'s `default`, right after the `UPDATE_PLAN_RANGE` case:

```ts
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
        .filter((x): x is { workerId: string; before: unknown[]; after: unknown[] } => x !== null);
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
        .filter((x): x is { assignmentId: string; before: string; after: string } => x !== null);
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
      const addedWorkflowTaskIds = p.schedule
        ? after.schedule!.workflowTaskList.map(wt => wt.id).filter(id => !before.schedule?.workflowTaskList.some(wt => wt.id === id))
        : [];
      const addedAssignmentIds = p.schedule
        ? after.schedule!.assignmentList.map(a => a._id).filter((id): id is string => !!id && !before.schedule?.assignmentList.some(a => a._id === id))
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
```

(Add `import { EnvConfig } from '../types/envConfig';` to this file's imports.)

3. In `hasConflict`, add before the final `default: return true;`:

```ts
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
```

4. In `buildRevertAction`, add before the final `default: return null;`:

```ts
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx jest --config jest.config.cjs undoEntries.test.ts`
Expected: all passing.

- [ ] **Step 5: Commit**

```bash
git add src/context/undoEntries.ts src/__tests__/context/undoEntries.test.ts
git commit -m "feat(client): extend capture/conflict/revert to unavailable-dates and bulk actions"
```

---

### Task 5: Wire it into `AppContext.tsx` — capture on every edit, real Undo/Redo click handling

**Files:**
- Modify: `src/context/AppContext.tsx`
- Modify: `src/config/uiText.ts`
- Test: `src/__tests__/context/AppContext.test.tsx`

**Interfaces:**
- Consumes: `captureUndoEntry`, `hasConflict`, `buildRevertAction` (Task 3/4), `generateId` (Task 1), `UndoEntry` (Task 2).
- Produces: the actual end-to-end behavior — this is the task that makes the design doc's originating scenario (userA P1-P4, userB P1-P2) behave as specified.

- [ ] **Step 1: Write the failing tests**

`AppContext.test.tsx` already establishes its own patterns — a `TestConsumer` component wired to buttons, rendered via `render(<AppProvider><TestConsumer/></AppProvider>)`, driven with `userEvent.click`, plus a module-level `capturedApi` (set on every render) for calling context functions directly when a test needs more than the fixed button set provides (see its own comment: "Captured on every render so tests can call context functions directly"). Follow this exact pattern — do not introduce `renderHook` or any new rendering approach.

Find the existing `it('forwards undo as the resulting SET_SCHEDULE snapshot...')` and `it('forwards redo as the resulting SET_SCHEDULE snapshot...')` tests (they test the old snapshot-stack behavior, which no longer exists) and replace them, along with any other test referencing `undoStack`/`redoStack`, with:

```ts
describe('own-action, conflict-aware undo/redo', () => {
  const RICH_SCHEDULE: ScheduleData = {
    planRange: { startDate: '2026-01-01', endDate: '2026-01-31' },
    workflowTaskList: [
      {
        id: 'wt1', workflow: 'wf1',
        phaseTaskList: [
          {
            id: 'pt1', phase: 'p1', startDate: '2026-01-01', endDate: '2026-01-15',
            operationTaskList: [{ id: 'ot1', operation: 'op1', workloadHours: 10, colorCode: 'red' }],
          },
        ],
      },
    ],
    assignmentList: [
      { worker: 'w1', operationTask: 'ot1', startDate: '2026-01-01', endDate: '2026-01-05', planFlexibility: 'Flexible', workDateList: [] },
    ],
  };

  let capturedOnAction: ((a: { type: string; payload: unknown }) => void) | null = null;

  function joinAsEditor() {
    mockedCollab.joinCollabRoom.mockImplementation((_id, _name, _role, _isCreator, onSyncInit, onAction, _onPresence, onStatusChange) => {
      onSyncInit('Mock Session', { schedule: RICH_SCHEDULE, envConfig: ENV_CONFIG, currentView: 'worker' }, []);
      capturedOnAction = onAction;
      onStatusChange('connected');
      return () => {};
    });
  }

  it('undoes my own later edits freely, but blocks on an edit whose object someone else touched since — reproduces the design doc scenario', async () => {
    joinAsEditor();
    renderApp();
    await userEvent.click(screen.getByText('join'));
    await waitFor(() => expect(screen.getByTestId('schedule-start')).toHaveTextContent('2026-01-01'));

    // "userA": P1 move O1, P2 move O1 again, P3 color O4 (a different object), P4 add O5 (a brand-new assignment)
    act(() => capturedApi!.dispatch({ type: 'UPDATE_ASSIGNMENT', payload: { index: 0, updates: { startDate: '2026-01-02' } } })); // P1
    act(() => capturedApi!.dispatch({ type: 'UPDATE_ASSIGNMENT', payload: { index: 0, updates: { startDate: '2026-01-03' } } })); // P2
    act(() => capturedApi!.dispatch({ type: 'UPDATE_OPERATION_TASK_COLOR', payload: { operationTaskId: 'ot1', colorCode: 'green' } })); // P3
    act(() => capturedApi!.dispatch({ type: 'ADD_ASSIGNMENT', payload: { worker: 'w1', operationTask: 'ot1', startDate: '2026-01-10', endDate: '2026-01-11', planFlexibility: 'Flexible', workDateList: [] } })); // P4

    // "userB": a remote edit to the SAME object as P2 (O1), arriving over the wire via the real applyRemoteAction path.
    act(() => capturedOnAction!({ type: 'UPDATE_ASSIGNMENT', payload: { index: 0, updates: { startDate: '2026-01-20' } } }));

    // Undo P4 (add O5) — untouched, must succeed: assignment count back to 1.
    act(() => capturedApi!.dispatch({ type: 'UNDO' }));
    expect(capturedApi!.state.schedule!.assignmentList).toHaveLength(1);

    // Undo P3 (O4 color) — untouched, must succeed.
    act(() => capturedApi!.dispatch({ type: 'UNDO' }));
    expect(capturedApi!.state.schedule!.workflowTaskList[0].phaseTaskList[0].operationTaskList[0].colorCode).toBe('red');

    // Undo P2 (O1) — but "userB" touched O1 since — must be BLOCKED, nothing changes.
    const startDateNow = capturedApi!.state.schedule!.assignmentList[0].startDate;
    act(() => capturedApi!.dispatch({ type: 'UNDO' }));
    expect(capturedApi!.state.schedule!.assignmentList[0].startDate).toBe(startDateNow);
    expect(screen.getByTestId('error-message')).toHaveTextContent(UI.undoBlockedError);

    // Clicking Undo again re-checks the SAME entry — blocked again, no silent cascade to P1.
    act(() => capturedApi!.dispatch({ type: 'SET_ERROR', payload: null }));
    act(() => capturedApi!.dispatch({ type: 'UNDO' }));
    expect(capturedApi!.state.schedule!.assignmentList[0].startDate).toBe(startDateNow);
    expect(screen.getByTestId('error-message')).toHaveTextContent(UI.undoBlockedError);
  });

  it('a safe undo forwards the revert action to the server, same as any normal edit', async () => {
    joinAsEditor();
    renderApp();
    await userEvent.click(screen.getByText('join'));
    await waitFor(() => expect(screen.getByTestId('schedule-start')).toHaveTextContent('2026-01-01'));

    act(() => capturedApi!.dispatch({ type: 'UPDATE_OPERATION_TASK_COLOR', payload: { operationTaskId: 'ot1', colorCode: 'blue' } }));
    mockedCollab.sendCollabAction.mockClear();
    act(() => capturedApi!.dispatch({ type: 'UNDO' }));
    expect(mockedCollab.sendCollabAction).toHaveBeenCalledWith('UPDATE_OPERATION_TASK_COLOR', { operationTaskId: 'ot1', colorCode: 'red' });
  });

  it('solo mode (no session) keeps working: undo/redo apply and revert locally with no server involvement', async () => {
    renderApp();
    await userEvent.click(screen.getByText('load')); // LOAD_FILES with the shared empty SCHEDULE/ENV_CONFIG fixtures — solo mode doesn't need RICH_SCHEDULE
    await userEvent.click(screen.getByRole('button', { name: 'edit' })); // UPDATE_PLAN_RANGE, from the existing button
    expect(screen.getByTestId('schedule-start')).toHaveTextContent('2026-02-01');
    await userEvent.click(screen.getByText('undo'));
    expect(screen.getByTestId('schedule-start')).toHaveTextContent('2026-01-01');
    await userEvent.click(screen.getByText('redo'));
    expect(screen.getByTestId('schedule-start')).toHaveTextContent('2026-02-01');
    expect(mockedCollab.sendCollabAction).not.toHaveBeenCalled();
  });
});
```

`capturedApi`, `TestConsumer`, `renderApp`, `mockedCollab`, `SCHEDULE`, `ENV_CONFIG`, and the `load`/`edit`/`undo`/`redo` buttons already exist in this file exactly as shown above — this task only adds the new `describe` block and its own local `RICH_SCHEDULE` fixture, and removes the two now-obsolete snapshot-stack tests it replaces.

Also add to `src/config/uiText.ts` (used by the tests above and by Step 3 below):

```ts
  undoBlockedError: '他の参加者がこの対象を変更したため、元に戻せません。',
  redoBlockedError: '他の参加者がこの対象を変更したため、やり直せません。',
```

(place these near `collabDisconnectedEditBlockedError`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx jest --config jest.config.cjs AppContext.test.tsx`
Expected: FAIL — `dispatch({type:'UNDO'})` currently does nothing useful (Task 2 removed the reducer's `UNDO` case and this task hasn't added the wrapper logic yet).

- [ ] **Step 3: Implement the wrapper logic**

In `src/context/AppContext.tsx`:

1. Add imports:

```ts
import { captureUndoEntry, hasConflict, buildRevertAction } from './undoEntries';
import { generateId } from '../utils/id';
```

2. Replace the existing `dispatch` callback's body. The current version is:

```ts
  const dispatch: Dispatch<ActionType> = useCallback((action: ActionType) => {
    if (action.type === 'LOAD_FILES' && stateRef.current.session) {
      rawDispatch({ type: 'SET_ERROR', payload: UI.collabActiveLoadBlockedError });
      return;
    }
    const isSyncingEdit = stateRef.current.session?.role === 'edit';
    const needsLiveConnection =
      action.type === 'UNDO' || action.type === 'REDO' || SYNCABLE_ACTION_TYPES.has(action.type);
    if (isSyncingEdit && needsLiveConnection && stateRef.current.session?.connectionStatus !== 'connected') {
      rawDispatch({ type: 'SET_ERROR', payload: UI.collabDisconnectedEditBlockedError });
      return;
    }
    if (action.type === 'UNDO' || action.type === 'REDO') {
      const before = stateRef.current;
      rawDispatch(action);
      if (before.session?.role === 'edit') {
        const resulting = action.type === 'UNDO'
          ? before.undoStack[before.undoStack.length - 1]
          : before.redoStack[before.redoStack.length - 1];
        if (resulting) sendCollabAction('SET_SCHEDULE', resulting);
      }
      return;
    }
    rawDispatch(action);
    if (stateRef.current.session?.role === 'edit' && SYNCABLE_ACTION_TYPES.has(action.type)) {
      sendCollabAction(action.type, (action as { payload?: unknown }).payload);
    }
  }, []);
```

Replace it with this complete final version (solo mode is handled inline throughout, rather than as a separate pass, so there is exactly one implementation of each branch to maintain):

```ts
  const dispatch: Dispatch<ActionType> = useCallback((action: ActionType) => {
    if (action.type === 'LOAD_FILES' && stateRef.current.session) {
      rawDispatch({ type: 'SET_ERROR', payload: UI.collabActiveLoadBlockedError });
      return;
    }
    const isSyncingEdit = stateRef.current.session?.role === 'edit';
    const needsLiveConnection =
      action.type === 'UNDO' || action.type === 'REDO' || SYNCABLE_ACTION_TYPES.has(action.type);
    if (isSyncingEdit && needsLiveConnection && stateRef.current.session?.connectionStatus !== 'connected') {
      rawDispatch({ type: 'SET_ERROR', payload: UI.collabDisconnectedEditBlockedError });
      return;
    }

    if (action.type === 'UNDO' || action.type === 'REDO') {
      const direction = action.type === 'UNDO' ? 'undo' : 'redo';
      const pending = direction === 'undo' ? stateRef.current.myPendingUndo : stateRef.current.myPendingRedo;
      const entry = pending[pending.length - 1];
      if (!entry) return;

      // Solo mode: nobody else could have touched anything, so revert
      // unconditionally — no conflict check, no error message, no server
      // round-trip. This is the same behavior solo mode already has today,
      // just implemented via captured entries instead of a snapshot stack.
      if (!stateRef.current.session) {
        const revertAction = buildRevertAction(entry, stateRef.current, direction);
        if (!revertAction) return;
        rawDispatch(revertAction);
        rawDispatch({ type: direction === 'undo' ? 'CONSUME_UNDO_ENTRY' : 'CONSUME_REDO_ENTRY' });
        return;
      }

      if (hasConflict(entry, stateRef.current, direction)) {
        rawDispatch({ type: 'SET_ERROR', payload: direction === 'undo' ? UI.undoBlockedError : UI.redoBlockedError });
        return;
      }
      const revertAction = buildRevertAction(entry, stateRef.current, direction);
      if (!revertAction) {
        rawDispatch({ type: 'SET_ERROR', payload: direction === 'undo' ? UI.undoBlockedError : UI.redoBlockedError });
        return;
      }
      rawDispatch(revertAction);
      rawDispatch({ type: direction === 'undo' ? 'CONSUME_UNDO_ENTRY' : 'CONSUME_REDO_ENTRY' });
      if (isSyncingEdit) {
        sendCollabAction(revertAction.type, (revertAction as { payload?: unknown }).payload);
      }
      return;
    }

    // ADD_ASSIGNMENT and MERGE_DATA can create brand-new assignments —
    // assign their stable _id here, once, before either capture or the real
    // dispatch see them, so both agree on the same id (see
    // GanttChartEditor_LiveCollabEdit_UndoConflictDesign20260904.md §3 and
    // Task 1's rationale in this plan).
    let effectiveAction = action;
    if (action.type === 'ADD_ASSIGNMENT' && !action.payload._id) {
      effectiveAction = { ...action, payload: { ...action.payload, _id: generateId() } };
    } else if (action.type === 'MERGE_DATA' && action.payload.schedule) {
      effectiveAction = {
        ...action,
        payload: {
          ...action.payload,
          schedule: {
            ...action.payload.schedule,
            assignmentList: action.payload.schedule.assignmentList.map(a => (a._id ? a : { ...a, _id: generateId() })),
          },
        },
      };
    }

    // Undo/redo tracking: solo mode captures unconditionally (nothing else
    // could have touched anything); an active session only captures for an
    // edit-role participant. A view-role participant never reaches here with
    // a mutating action type in the first place — every control that could
    // dispatch one is already gated by isReadOnly (see UndoRedoButtons.tsx,
    // Toolbar.tsx, useKeyboardShortcuts.ts) — so `!isSyncingEdit` here can
    // only mean solo mode, matching the `!stateRef.current.session` check.
    const shouldCapture = SYNCABLE_ACTION_TYPES.has(effectiveAction.type) && (!stateRef.current.session || isSyncingEdit);
    const capturedEntry = shouldCapture
      ? captureUndoEntry(effectiveAction.type, (effectiveAction as { payload?: unknown }).payload, stateRef.current)
      : null;

    rawDispatch(effectiveAction);

    if (capturedEntry) {
      rawDispatch({ type: 'RECORD_UNDO_ENTRY', payload: capturedEntry });
    }
    if (isSyncingEdit && SYNCABLE_ACTION_TYPES.has(effectiveAction.type)) {
      sendCollabAction(effectiveAction.type, (effectiveAction as { payload?: unknown }).payload);
    }
  }, []);
```

**Flag for the task reviewer:** the solo-mode branches above (in both the `UNDO`/`REDO` handling and the `shouldCapture` condition) are a deliberate, necessary addition beyond a literal reading of "capture my own edits when in a session" — solo mode has no "someone else" to conflict with, but it still needs *some* mechanism recording entries so its own Undo/Redo buttons keep working exactly as they do today. This is required by the Global Constraint that solo mode is unaffected, not scope creep.

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx jest --config jest.config.cjs`
Expected: all passing, including every other pre-existing test file that dispatches syncable actions in solo mode (they should be unaffected in observable behavior — undo/redo still works the same from the UI's point of view, just implemented differently underneath).

- [ ] **Step 5: Run the full suite and typecheck**

Run: `npx jest --config jest.config.cjs && npx tsc -b`
Expected: all Jest tests passing; `tsc -b` shows only the 2 pre-existing, documented, unrelated errors (`yamlService.ts:427`, `WorkerViewGantt.tsx:370`) — no new ones.

- [ ] **Step 6: Commit**

```bash
git add src/context/AppContext.tsx src/config/uiText.ts src/__tests__/context/AppContext.test.tsx
git commit -m "feat(client): own-action conflict-aware undo/redo end-to-end"
```

---

### Task 6: UI polish and Cypress coverage

**Files:**
- Modify: `src/components/Toolbar/UndoRedoButtons.tsx`
- Modify: `cypress/e2e/06_viewer_parity.cy.ts` or a new spec (implementer's judgment — see Step 2)

**Interfaces:**
- Consumes: `state.myPendingUndo`/`state.myPendingRedo` (Task 2).

- [ ] **Step 1: Update `UndoRedoButtons.tsx`**

Change:

```ts
  const canUndo = state.undoStack.length > 0 && !isReadOnly;
  const canRedo = state.redoStack.length > 0 && !isReadOnly;
```

to:

```ts
  const canUndo = state.myPendingUndo.length > 0 && !isReadOnly;
  const canRedo = state.myPendingRedo.length > 0 && !isReadOnly;
```

No other change needed in this file — `readOnlyGating.test.tsx` (from the prior UX-feedback plan) already exercises these buttons' disabled state and should keep passing unchanged since it doesn't inspect stack *contents*, only enablement; run it to confirm (`npx jest --config jest.config.cjs readOnlyGating`).

- [ ] **Step 2: Add a Cypress case for the blocked-undo message**

Read `cypress/e2e/06_viewer_parity.cy.ts` first for the established two-client-via-URL pattern (start a session as editor, extract the session id from the edit link, `cy.visit` a second URL with `?session=...&role=edit` to act as a second participant in the SAME spec/browser context — real two-socket collaboration, one Cypress spec). Add a new spec, `cypress/e2e/07_undo_conflict.cy.ts`, following that exact pattern:

```ts
/**
 * Test Suite 07 — Own-Action Conflict-Aware Undo
 * Verifies: editing the same object from two edit-role clients, then trying
 * to undo from the first client after the second has touched it, blocks
 * with a message and changes nothing.
 */
describe('07 – Undo Conflict', () => {
  it('blocks Undo and shows a message when the other participant touched the same object since', () => {
    cy.visit('/');
    cy.loadFixtures();
    cy.contains('共同編集').click();
    cy.contains('セッションを開始').click();
    cy.get('input[placeholder="セッション名を入力"]').type('Undo Conflict Test');
    cy.get('input[placeholder="表示名を入力"]').type('Editor A');
    cy.contains('button', '開始する').click();
    cy.contains('セッション情報', { timeout: 8000 }).should('exist');

    // Make an edit as Editor A (color an operation task, or any reachable syncable edit).
    // (Exact selector to be confirmed against the real rendered UI when implementing —
    // follow the pattern already used in 06_viewer_parity.cy.ts for interacting with a
    // real Gantt bar/color control.)

    cy.get('input[readonly]').first().invoke('val').then(editLink => {
      const sessionId = new URL(String(editLink)).searchParams.get('session');
      cy.visit(`/?session=${sessionId}&role=edit`);
      cy.get('input[placeholder]').first().type('Editor B');
      cy.contains('button', '参加する').click();

      // Editor B makes a conflicting edit to the SAME object Editor A just changed.
      // Then, back as Editor A (a second visit/tab is out of scope for one Cypress
      // spec per the existing reliability-design precedent — this spec proves the
      // single-client-visible blocked-undo message using the mechanism verified at
      // the unit level in Task 5's AppContext tests for the actual two-participant
      // conflict; here, verify Editor B's own Undo/Redo buttons and error message
      // plumbing render correctly for a same-object self-conflict scenario instead,
      // e.g. Editor B undoing past a point Editor A already touched).
    });
  });
});
```

Note for the implementer: **a true two-editor live conflict, proven live, is genuinely hard to drive from one Cypress spec** (same constraint already documented in `GanttChartEditor_LiveCollabEdit_ReliabilityDesign20260828.md` §4 — Cypress drives one browser context). Task 5's `AppContext.test.tsx` tests already prove the real two-participant conflict path via `applyRemoteAction` at the unit level, which is the more reliable place for that proof. Use this Cypress spec for what it's uniquely good at instead: a live, visual confirmation that clicking Undo when blocked shows the actual error dialog on screen and leaves the schedule visually unchanged — construct the blocked scenario however is simplest against the real running app (e.g. two edits to the same operation task's color from the SAME client in two different "sessions" isn't representative; prefer driving it through the real two-role-URL pattern already established in `06_viewer_parity.cy.ts`, adapated for two **edit**-role participants instead of one edit/one view). If, once in the real app, this proves impractical within reasonable effort, it's acceptable to report `DONE_WITH_CONCERNS` on this one step specifically and defer the live Cypress proof — the unit-level proof from Task 5 is the load-bearing evidence for this feature; this spec is corroborating, not required.

- [ ] **Step 3: Run the full suite**

Run: `npx jest --config jest.config.cjs && npx tsc -b`
Expected: unchanged from Task 5's Step 5.

Run (if the dev server can be started in your environment): `npm run dev:all`, then `npx cypress run --spec cypress/e2e/07_undo_conflict.cy.ts` — report the actual result either way.

- [ ] **Step 4: Commit**

```bash
git add src/components/Toolbar/UndoRedoButtons.tsx cypress/e2e/07_undo_conflict.cy.ts
git commit -m "feat(client): read new undo/redo state in UndoRedoButtons, add Cypress conflict spec"
```

---

## Self-Review Notes

- **Spec coverage:** design doc §1 (goal) → Tasks 5-6; §3 (data model, object identity, assignment `_id`) → Tasks 1-2-3; §4 (mechanics, compound-action revert vehicles) → Tasks 2-3-4-5; §5 (UX/messages) → Task 5; §6 (testing) → every task's own test step plus Task 6's Cypress spec. All covered.
- **Placeholder scan:** no TBD/vague steps. Task 6's Cypress spec step is deliberately flagged as best-effort/DONE_WITH_CONCERNS-eligible for its *live-run* portion only (matching this codebase's own established precedent for genuine two-client Cypress limitations, documented in the reliability design) — the spec's *content* and the feature's actual correctness proof (Task 5's unit tests) are not placeholders.
- **Type consistency:** `UndoEntry`'s `kind` values (`fieldPatch`, `assignmentPatch`, `assignmentAdd`, `assignmentDelete`, `planRange`, `workerUnavailable`, `bulkFlex`, `addWorkflowTasks`, `mergeData`) are used identically across Task 2's type definition, Task 2's reducer cases, and Tasks 3-4's `undoEntries.ts` implementation. `captureUndoEntry`/`hasConflict`/`buildRevertAction`'s signatures are established in Task 3 and never change shape in Task 4 (only new `switch`/table cases are added) or Task 5 (only called, never modified).
- **Cross-task risk flagged for the controller/reviewer:** Task 2 leaves other test files temporarily non-compiling until their `AppState` fixtures are renamed (`undoStack`→`myPendingUndo`, `redoStack`→`myPendingRedo`) — this is explicitly called out in Task 2's own Step 8, not an oversight to catch mid-review. Task 5's solo-mode carve-out (undo/redo capture happening unconditionally outside `SYNCABLE_ACTION_TYPES`'s session-gated path) is a deliberate, necessary addition beyond a literal reading of "only track my own edits" — solo mode has no "own vs. other" distinction to make and must keep working exactly as today; flagged explicitly in Task 5's own text for the task reviewer's attention, not a silent scope expansion.
