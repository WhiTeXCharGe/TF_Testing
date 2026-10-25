# GanttChartEditor — `trip_list` (FI/FO) as the main FIFO data

**Date:** 2026-10-23. **Status:** design approved by the user in chat; plan in `GanttChartEditor_TripFifo_Plan20261023.md`.

**Source of truth for the file format and the derivation rules (backend, already implemented):**
`documents/Timefold/Timefold_TripList_Design20261021.md`, `web/Timefold/.../trip/TripDeriver.java` and its test.
Example file: `web/Timefold/work/output/result_Schedule_5module.yaml`.

## 1. What changes for the user

| Before | After |
|---|---|
| 出入国バー are *predicted* from assignments on every render. Not stored. | The `trip_list` of `Schedule.yaml` is the main FIFO. Tasks no trip covers still get an automatic trip. |
| FI = first task day − 1, FO = last task day + 1, always. | FI/FO can be anywhere (not tied to tasks, may break `stay_off_interval`). |
| Stay limits predicted from the same stints. | 滞在日数 / 年間滞在日数 count from each trip's FI→FO. 移動日 is unchanged (last task of one region → first task of next). |
| `trip_list` is dropped when the file is saved. | Saved/sent files contain `trip_list` = stored trips + automatic trips. |

## 2. Data model

`ScheduleData.tripList?: Trip[]` (optional so existing fixtures keep compiling).

```ts
interface Trip { _id: string; worker: string; startDate: string; endDate: string; // FI, FO, YYYY-MM-DD
                 region: string; planFlexibility: PlanFlexibility; description?: string }
```

* YAML (read and write) is the backend format: per worker `trips:`; keys `start_date`, `end_date`, `region`,
  `plan_flexibility` (reader also accepts `flexibility`), `description`; dates `YYYY/MM/DD`; written after
  `assignment_list`. Flat form (`worker` on each trip) is accepted on read. Unknown flexibility reads as `Flexible`.
  Trips with `start >= end` or a missing field are skipped when reading.
* `_id` is session-local (never written). It must be **identical on every collaborator**, so it is deterministic:
  `trip:<worker>:<region>:<startDate>`, with `~2`, `~3` … appended on a clash (counted in list order). It is assigned
  when a file is loaded or a session baseline is applied, and when a trip is created by an edit (the id is computed
  by the editing client and carried in the action).

## 3. Automatic trips (derived, not stored)

`effectiveTrips(envConfig, schedule)` = stored trips + derived trips.

* **Worked day** of a worker = a date with `hour > 0` in an assignment's `workDateList` (if the list is empty: every date
  from `startDate` to `endDate`); the region is the task's region (fab → region, or the misc task's own `region`),
  exactly as the current view model resolves it. Assignments with no region contribute nothing.
* **Covered** = the day lies inside any stored trip of that worker (both ends included, any region, Fixed or Flexible).
* Derived trips come from the **uncovered** days with the backend `TripDeriver` algorithm: split on region change, or when
  the off-days between two worked days are `>= max(1, stay_off_interval)`, or when a stored trip lies between them;
  `FI = first − 1`, `FO = last + 1`; Flexible; then the neighbour check (a trip may start on, but never before, the
  previous trip's end). Derived ids are `auto:<worker>:<region>:<start>` (display keys only).
* Automatic generation **never edits or deletes a stored trip**. So moving FO from 11 to 15 keeps 15; the tasks inside
  stay covered and nothing new is generated. If a task moves out of every trip, a new automatic trip appears for it and
  the old stored trip stays (the user can delete it).
* Because derived trips are a pure function of the synced state, all collaborators see the same trips and undo of an
  assignment edit needs no special handling.

## 4. Editing

* Right-click a trip bar → small popup (no side panel): FI date, FO date, 固定/可変 toggle, 削除. FI must be before FO.
* Drag the FI end or the FO end of a bar to change that date.
* Editing an automatic trip (popup OK or drag) **materialises** it: it becomes a stored trip with the edited values.
  Deleting a derived trip is not offered (it is derived); deleting a stored trip removes it (the days it covered may
  then get an automatic trip again).
* Editing a date never changes 固定/可変 by itself.
* Read-only participants (view role, locked session) see the bars and the popup values but cannot change anything.

Look (UI otherwise unchanged; bars still toggled by the 出入国バー button): 固定 = solid, darker border, 🔒 mark;
可変 = same region colour, lighter, dashed border; automatic = like 可変 but fainter.

## 5. Limits

* The constraint server counts each trip from FI to FO **with both days included** (`FO − FI + 1`; the backend solver
  counts the same). Annual totals split a trip across calendar years the same way. Previously it counted `FO − FI`.
* The client sends `schedule.tripList = effectiveTrips(...)`. If a request has no `tripList` (old client) the server
  falls back to its current derivation.
* 移動日 (`TRAVEL_DAYS`) is unchanged.
* The generation split rule is the backend's (`gap >= interval` splits); the old Gantt merged when `gap <= interval`.

## 6. Collaboration and undo/redo

* New syncable actions `UPSERT_TRIP` (payload: a full `Trip`) and `DELETE_TRIP` (payload: the trip id).
* New undo entry kind `trip`: `{ id, before: Trip | null, after: Trip | null }`. Undo is blocked, like every other kind,
  when the current trip differs from what the edit produced. Revert = `UPSERT_TRIP` with the other value, or
  `DELETE_TRIP` when that value is `null`.
* `MERGE_DATA` (新規製番追加 import) does not import trips; automatic generation covers the merged assignments.

## 7. Not in this pass

* Warning when a task sits on a Fixed trip's FI/FO day or in another region inside it (backend `fixed_trip_conflict`).
* Creating a trip from empty space; trip-aware Excel export; dropping stale empty Flexible trips automatically.

## 8. Test data

Use an excerpt of `result_Schedule_5module.yaml` (inline string in the unit test) and the existing `Test_data/` files.
