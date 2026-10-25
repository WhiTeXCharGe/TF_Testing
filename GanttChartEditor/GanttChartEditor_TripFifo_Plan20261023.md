# GanttChartEditor — `trip_list` (FI/FO) Implementation Plan

> **For agentic workers:** execute task by task (subagent-driven-development). Read the design first:
> `documents/GanttChartEditor/GanttChartEditor_TripFifo_Design20261023.md`. Backend reference (read, do not change):
> `web/Timefold/src/main/java/com/yourorg/scheduler/trip/TripDeriver.java`, `.../src/test/java/com/yourorg/scheduler/TripDeriverTest.java`,
> `documents/Timefold/Timefold_TripList_Design20261021.md`; sample `web/Timefold/work/output/result_Schedule_5module.yaml`.

**Goal:** `trip_list` in `Schedule.yaml` is the main FIFO: read, shown, edited (right-click popup, drag ends), saved, synced;
tasks without a trip get an automatic trip; stay limits count FI→FO.

**Architecture:** stored `ScheduleData.tripList` + derived (pure, uncovered days only) = `effectiveTrips`. New syncable
actions `UPSERT_TRIP`/`DELETE_TRIP`, undo kind `trip`. Constraint server counts stays from the trips the client sends.

**Tech stack:** React 19 + TypeScript (Vite), Jest (client, `npx jest --config jest.config.cjs`), Vitest (server, `cd server && npm run test`),
`npx tsc -b`. Install with `npm install --legacy-peer-deps` (root and `server/`).

## Global Constraints

- Work only in this worktree. Never touch `Test_data/*` in the main checkout. Do not push.
- Internal dates are `YYYY-MM-DD`; the YAML uses `YYYY/MM/DD` (use the existing `toYD` / `normalizeDate`).
- A trip needs `startDate < endDate`; equal dates inside one trip are invalid.
- Trip `_id` is deterministic and identical on every client: `trip:<worker>:<region>:<startDate>`, clash suffix `~2`, `~3`
  (in list order). Never use random ids for trips. Derived (automatic) trips use `auto:<worker>:<region>:<start>` and are never stored.
- Automatic generation never edits or deletes a stored trip.
- Editing dates never changes `planFlexibility`. UI toggles only `Fixed`/`Flexible`; any other value reads as `Flexible`.
- Read-only participants (use `isSessionReadOnly(state)` from `src/lib/sessionReadOnly.ts`) cannot change trips.
- Solo mode must keep working exactly as before (undo/redo, saving). `tsc -b` must not gain new errors versus the baseline
  recorded at the start (run it first and write the baseline in your report). Jest must stay green.
- User-visible text goes in `src/config/uiText.ts` (Japanese).
- Reducer cases return new state objects, never mutate.

---

### Task 1: Types, YAML read/write of stored trips, deterministic ids

**Files:** modify `src/types/schedule.ts`, `src/services/yamlService.ts`, `src/context/reducer.ts`; create `src/utils/tripIds.ts`;
tests `src/__tests__/services/yamlService.test.ts` (extend), `src/__tests__/utils/tripIds.test.ts`, `src/__tests__/context/reducer.test.ts` (extend).

**Produces:** `Trip` type; `ScheduleData.tripList?: Trip[]`; `tripIdFor(existingIds: Iterable<string>, worker, region, startDate): string`;
`withTripIds(trips: Trip[]): Trip[]`; `parseScheduleYaml` fills `tripList`; `stringifyScheduleYaml` writes `data.tripList` (stored trips only for now).

- [ ] Add to `schedule.ts`:
```ts
export interface Trip {
  _id: string;            // session-local, deterministic (see src/utils/tripIds.ts); never written to YAML
  worker: string;
  startDate: string;      // FI
  endDate: string;        // FO
  region: string;
  planFlexibility: PlanFlexibility;   // UI uses 'Fixed' | 'Flexible'
  description?: string;
}
// in ScheduleData:  tripList?: Trip[];
```
- [ ] `tripIds.ts`: `tripIdFor` returns `trip:${worker}:${region}:${startDate}` or the same with `~2`, `~3`… so it is not in `existingIds`.
  `withTripIds(trips)` returns trips with `_id` filled by calling `tripIdFor` in list order against the ids assigned so far (keep an existing `_id`
  only if it is not already used; otherwise reassign). Tests: plain id, clash suffixes, order independence of equal input giving equal output.
- [ ] `parseScheduleYaml`: read `data.trip_list`. Accept nested (`- worker: w1` + `trips: [...]`) and flat (`worker` on each item). Per trip read
  `start_date`, `end_date`, `region`, `plan_flexibility` (fallback `flexibility`), `description`. Dates through `normalizeDate`.
  Flexibility: `'Fixed'` if the text equals `Fixed` (case-insensitive), `'Reluctant'` not allowed here, anything else `'Flexible'`.
  Skip a trip when `worker`, `region`, `start_date` or `end_date` is missing, a date is unparseable, or `start >= end`. Result goes
  through `withTripIds` and is sorted by worker (input order) then start date. If there is no `trip_list` (or it is empty) leave `tripList` undefined.
- [ ] `stringifyScheduleYaml`: after the assignment list write, only when `data.tripList?.length`:
```
  trip_list:
  - worker: w1
    trips:
    - start_date: 2025/09/07
      end_date: 2025/09/13
      region: r2
      plan_flexibility: Flexible
      description: ''
```
  Group by worker keeping first-seen worker order, trips by start date. Use `ys(...)` for description exactly as assignments do.
  Round trip test with a trimmed excerpt of `result_Schedule_5module.yaml` (4–5 workers incl. one with 3 trips) plus a flat-form test and the skip rules.
- [ ] `reducer.ts`: in `LOAD_FILES` and `SET_SESSION_BASELINE` apply `withTripIds` to `schedule.tripList` when present (keep
  the existing single-object `schedule`/`savedScheduleRef` sharing fix: compute the schedule once and use the same reference for both).
  Tests: ids assigned, `schedule === savedScheduleRef` still true.
- [ ] Run Jest + `tsc -b`. Commit: `feat(client): read and write trip_list, deterministic trip ids`.

---

### Task 2: Trip derivation (`effectiveTrips`)

**Files:** create `src/utils/trips.ts`, `src/__tests__/utils/trips.test.ts`.

**Consumes:** `Trip`, `tripIdFor` (Task 1). **Produces:**
```ts
export interface WorkedDay { worker: string; date: string; region: string }
export function workedDays(envConfig: EnvConfig, schedule: ScheduleData): WorkedDay[];
export function deriveTrips(days: WorkedDay[], stored: Trip[], intervalOf: (regionId: string) => number): Trip[]; // derived only, ids "auto:..."
export function effectiveTrips(envConfig: EnvConfig, schedule: ScheduleData): Trip[];   // stored + derived, sorted by worker order then start
export function tripStayDays(t: { startDate: string; endDate: string }): number;       // inclusive: FO - FI + 1
```
- [ ] `workedDays`: region resolution exactly like `buildWorkerTimelineModel` in `workerViewModel.ts` (misc task → `wt.region`; otherwise fab → `fab.region`;
  none → skip). A worked day = date with `hour > 0` in `workDateList`; if `workDateList` is empty use every date from `startDate` to `endDate`.
- [ ] `deriveTrips`: port the `TripDeriver.derive` algorithm for the **non-cut** path, with "covered" meaning inside any `stored` trip of that
  worker (both ends included, any flexibility). Read `TripDeriver.java` (all 190 lines) and `TripDeriverTest.java`; port the same steps:
  dedupe days (two regions on one day → alphabetically smaller wins), free days = not covered, runs split on region change / gap `>= max(1, interval)` /
  a stored trip lying strictly between two free days, `FI = first-1`, `FO = last+1`, `planFlexibility: 'Flexible'`, then the neighbour check
  (start may equal but not precede the previous item's end; follow the Java for what to do when it must move). `intervalOf` returns the region's `stayOffInterval ?? 0`.
  Derived `_id` = `auto:<worker>:<region>:<start>`. Output sorted by worker then start.
- [ ] `effectiveTrips`: stored = `schedule.tripList ?? []`; derived = `deriveTrips(workedDays(...), stored, region interval)`; return both,
  ordered by worker (order of first appearance in `envConfig.workerList`, unknown workers last) then `startDate`.
- [ ] Tests (port the Java cases that apply): the design example 6.1 (stored Fixed r1 09/01–09/10, tasks r1 09/03,09/04,09/09 and r2 09/12–09/14 →
  derived r2 09/11–09/15 only); gap equal to interval splits; gap below interval merges; region change gives touching end/start; barrier after a stored trip
  (r1 work on 09/11–09/13 after stored r1 09/01–09/10 → derived 09/10–09/14); weekend FI/FO; assignment with empty `workDateList`; assignment without region ignored;
  **user example:** stored trip 09/01–09/15 with tasks 09/02–09/10 → no derived trip; stored trip never modified; task moved outside every stored trip → new derived trip.
- [ ] Run Jest + `tsc -b`. Commit: `feat(client): derive automatic trips for uncovered worked days`.

---

### Task 3: Trip actions, sync, undo/redo, save includes automatic trips

**Files:** modify `src/types/appState.ts`, `src/context/reducer.ts`, `src/context/AppContext.tsx`, `src/context/undoEntries.ts`, `src/services/yamlService.ts`,
`src/services/fileService.ts`, `src/services/handoffService.ts`; tests in `src/__tests__/context/reducer.test.ts`, `undoEntries.test.ts`, `AppContext.test.tsx`, `yamlService.test.ts`.

**Produces:** actions `{ type: 'UPSERT_TRIP'; payload: Trip }`, `{ type: 'DELETE_TRIP'; payload: string }`; `UndoEntry` kind `trip`;
`stringifyScheduleYaml(data, envConfig?)`.

- [ ] Reducer: `UPSERT_TRIP` replaces the trip with the same `_id` in `schedule.tripList` (keeps position) or appends it (creates the list if missing);
  ignore (return `state`) when `!state.schedule`, or `startDate >= endDate`, or `worker`/`region` empty. `DELETE_TRIP` removes by id (no-op if absent). Both return a new schedule object.
- [ ] `AppContext.tsx`: add both types to `SYNCABLE_ACTION_TYPES`.
- [ ] `UndoEntry` gets `| { kind: 'trip'; id: string; before: Trip | null; after: Trip | null }`. In `undoEntries.ts`:
  capture `UPSERT_TRIP` → before = current trip by id or null, after = payload; capture `DELETE_TRIP` → before = current trip, after = null (return null if the trip does not exist);
  `hasConflict`: current trip by id (or null) deep-equals `undo ? after : before`, else conflict; `buildRevertAction`: target value = `undo ? before : after`;
  if null → `DELETE_TRIP` (id), else `UPSERT_TRIP` (value). Tests: capture/conflict/revert both directions, and an AppContext scenario (solo undo of an upsert; session undo blocked when another participant changed the same trip).
- [ ] `stringifyScheduleYaml(data, envConfig?)`: trips written = `envConfig ? effectiveTrips(envConfig, data) : (data.tripList ?? [])`
  (derived trips included, as `Flexible`). Update the four callers (`fileService.ts` ×3, `handoffService.ts` ×1) to pass the `envConfig`
  they already have. Tests: a schedule with tasks but no stored trips writes a derived `trip_list`; a stored trip is written unchanged; no trips and no tasks writes no `trip_list`.
- [ ] Jest + `tsc -b`. Commit: `feat(client): trip upsert/delete actions with sync and undo, save includes automatic trips`.

---

### Task 4: View model and trip bars

**Files:** modify `src/components/GanttChart/workerViewModel.ts`, `src/components/GanttChart/WorkerTimelineGrid.tsx`, `src/components/GanttChart/WorkerViewGantt.tsx` (props only if needed),
`src/config/uiText.ts`; update tests that mention `flightStints` (`workerViewModel.test.ts`, others found by grep).

- [ ] Replace `FlightStint`/`computeFlightStintsMultiRegion`/`flightStints` with `trips: TripView[]` on `WorkerTimelineRow`, where
  `TripView = Trip & { derived: boolean }`, computed from `effectiveTrips(envConfig, schedule)` grouped by worker (derived = `_id` starts with `auto:`). Remove the old helper.
- [ ] `FlightStintBar` → `TripBar`. Same geometry as today (bar at the bottom of the row from the FI cell to the FO cell, `FI`/`FO` labels, tooltip `<region>  <FI> → <FO>  <固定|可変|自動>`).
  Styles: **Fixed** solid colour, opacity 1, 2px darker border, `🔒` before the region label in the tooltip and a lock glyph at the FI label;
  **Flexible (stored)** region colour opacity 0.55 with a 2px dashed border of the region colour; **derived** opacity 0.35 dashed border. Keep `data-testid="trip-bar"` and `data-trip-id`.
  Same-day FI/FO label behaviour as today. Still shown only when `showFlightStints` is on.
- [ ] Tests: model returns stored + derived with correct `derived` flags; bar renders the three styles (check inline style/`data-flex`
  attribute — add `data-flex="fixed|flexible|auto"`); trips outside the visible date range clip like before.
- [ ] Jest + `tsc -b`. Commit: `feat(client): draw stored and automatic trips, fixed vs flexible look`.

---

### Task 5: Right-click popup and drag editing

**Files:** create `src/components/GanttChart/TripPopup.tsx`; modify `WorkerTimelineGrid.tsx`, `WorkerViewGantt.tsx`, `src/config/uiText.ts`;
tests `src/__tests__/components/tripPopup.test.tsx`, extend grid tests.

**Consumes:** `UPSERT_TRIP`/`DELETE_TRIP` (Task 3), `tripIdFor` (Task 1), `isSessionReadOnly`.

- [ ] `TripPopup` (small floating card anchored at the click point, closes on outside click/Esc): title = worker name + region name; two `<input type="date">`
  (FI日 / FO日); a two-button toggle `固定` / `可変`; buttons `OK`, `キャンセル`, and `削除` (shown only for stored trips). Validation: FI must be before FO
  (disable OK and show a short message `FIはFOより前の日付にしてください`). Read-only: inputs disabled, only `閉じる`.
- [ ] Right-click (`onContextMenu`, `preventDefault`) on a `TripBar` opens the popup. OK dispatches `UPSERT_TRIP` with the edited values:
  for a stored trip keep its `_id`; for a derived trip create the id with `tripIdFor(storedIds, worker, region, startDate)` (the **original** derived start) and `planFlexibility` from the toggle
  (default Flexible). `削除` dispatches `DELETE_TRIP`.
- [ ] Drag handles: 6px-wide hit zones at the FI end and FO end of a `TripBar` (cursor `ew-resize`, not shown/enabled when read-only). Mouse down → window `mousemove`
  converts the x delta to whole days with `DATE_CELL_WIDTH` and shows a dashed preview (same pattern as the existing drag preview in this file); mouse up dispatches
  `UPSERT_TRIP` (materialising a derived trip as above) if the result keeps FI before FO; otherwise cancel. A plain click without movement does nothing. Follow the existing
  window-listener pattern and the readOnly guard used by `startDrag` in this file (mind the note in `06_viewer_parity.cy.ts` about listener timing).
- [ ] Tests (RTL): popup opens on context menu; OK upserts with edited dates and flexibility; derived trip materialised with a deterministic id; delete only for stored;
  invalid range disables OK; read-only shows disabled inputs and dispatches nothing; drag end dispatches the expected dates (simulate mousedown/mousemove/mouseup).
- [ ] Jest + `tsc -b`. Commit: `feat(client): edit trips with a right-click popup and by dragging FI/FO`.

---

### Task 6: Stay limits from trips (server) and client request

**Files:** modify `server/src/types.ts`, `server/src/routes/constraints.ts`, `server/src/services/backendConstraints.ts`, `src/hooks/useBackendConstraintCheck.ts`;
tests under `server/src/**` (vitest, find the existing `backendConstraints` tests) and a client test for the request body.

- [ ] Server `ScheduleData` gets `tripList?: Array<{ worker, startDate, endDate, region, planFlexibility? }>`; add `TripSchema` (permissive like the others) to `ScheduleSchema` as optional.
- [ ] `checkStayDuration`: if `schedule.tripList` is defined, use it as the stints (FI = `startDate`, FO = `endDate`, region = `region`); `indices` = indices of that worker's
  assignments in that region whose span intersects `[FI, FO]` (so the violation still highlights bars). Duration = `daysBetween(FI,FO) + 1` (both days included); `maxStayOn` compares with it.
  Annual totals: split `[FI, FO + 1 day)` across calendar years (adapt `splitStayByYear` or call it with `FO + 1`). Messages keep their wording, the stay shown is the inclusive count.
  If `tripList` is undefined keep the old derivation unchanged. `TRAVEL_DAYS` check untouched.
- [ ] `useBackendConstraintCheck`: send `schedule: { ...state.schedule, tripList: effectiveTrips(state.envConfig, state.schedule) }`.
- [ ] Tests: trip FI/FO 09/01–09/11 with `maxStayOn 10` → violation (11 days) and with `maxStayOn 11` none; trip spanning Dec 30–Jan 3 splits 3 / 3 days per year
  (inclusive); a trip that does not match any task (user-moved FO) still counts; old derivation still works when `tripList` is absent; client request body contains the effective trips.
- [ ] Server tests + client Jest + `tsc -b` (client and `server`). Commit: `feat: stay limits count FI to FO from trip_list`.

---

## Self-review notes

- Spec coverage: data/file (T1), automatic generation (T2), save and sync/undo (T3), look (T4), editing (T5), limits (T6). Design §7 items are intentionally out.
- Type consistency: `Trip`, `tripIdFor`, `effectiveTrips`, `UPSERT_TRIP`/`DELETE_TRIP`, `UndoEntry.kind 'trip'` are named the same in every task.
- Known limitation carried from the design: `MERGE_DATA` does not import trips; undoing a deletion of a derived trip is not applicable (derived trips cannot be deleted).
