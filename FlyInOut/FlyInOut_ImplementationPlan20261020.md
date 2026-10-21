# FI/FO Trips (T0, WP0–WP6) Implementation Plan

> **CORRECTION (2026-10-21, revised):** an earlier note here said `unavailable_dates` would not stop work on
> FI/FO days. More precisely, in the current Timefold the rule `employeeAvailableAllDays` is disabled
> (`EmployeeSchedule.java:1262`), so an unavailable day is **not a hard rule**. Instead the block earns **zero hours**
> that day (`producedHoursForBlock`, `:1348`), the underfill rule makes the solver compensate, and the **output
> skips** such days (`ExportSchedule.java`, `isWorkerUnavailableByWid`). So the written assignments contain no work on
> those days, **but** the day-based rules (one task per day, overtime, transit, stay) still count the worker as
> covering them. A Fixed row blocks the day consistently (`oneTaskPerEmpPerDay`, hard).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the user's real FI/FO from `SU_Others` through decoder → Gantt → SchedulerWeb → Timefold as validated **Trips**, block travel days in the solver input, and show/verify them in the Gantt — **without changing Timefold**.

**Architecture:** The decoder (new `decoder8.py` + pure module `trips.py`) turns dirty FI/FO cells into a `trip_list` in `Schedule.yaml` using work days as ground truth. The Gantt parses/keeps/draws trips and verifies solver output against them. SchedulerWeb's single upload choke point (`submitRun`) injects *tagged* travel-day `unavailable_dates` into the uploaded EnvConfig copy only; the Gantt strips tagged entries on load.

**Tech Stack:** Python 3.12 + pytest 9 + openpyxl (decoder) · TypeScript/React + Jest (GanttChartEditor) · TypeScript + js-yaml + **vitest (new dev dependency)** (SchedulerWeb) · Docker (WP0 only).

**Spec:** `documents/FlyInOut/FlyInOut_Design20261020.md` (v3; D1–D8 accepted)

## Global Constraints

* Timefold (`TFDocker/**`) is **not modified**. `decoder7.py` is **not modified** (copy to `decoder8.py`).
* `trip_list` is optional everywhere. Absent/empty ⇒ output identical to today (SchedulerWeb uploads **byte-identical** files; Gantt writes no `trip_list` key).
* Dates: internal/Python-API ISO `YYYY-MM-DD`; YAML `YYYY/MM/DD` (Gantt: `normalizeDate`/`toYD`; decoder: `_to_ymd`).
* Stay rule (D4, Timefold's): same region, **new stay when off-days ≥ max(1, `stay_off_interval`)** where off-days = `(d − prev).days − 1`; a region change always starts a new stay; two regions on one day → alphabetically smaller wins.
* Fixed-state rule (D2/D5): `state: fixed` iff `quality == "clean"` AND no red-**font** endpoint event AND `region` is neither `null` nor `r_other` AND enforce-mode allows; everything else `review`. `suspect` is never fixed.
* Tagged unavailable entry (T0) shape, exactly: `- date: YYYY/MM/DD`, `source: trip`, `trip: <trip id>` (one entry per travel day, per worker).
* Violation severities: `error` for travel-day and other-region, `warning` for the rest.
* Misc tasks (`phaseTaskList` empty) are **ignored** by the other-region check; they still count as "something inside" for EMPTY.
* Commits: explicit `git add <paths>` only (GanttChartEditor has unrelated modified `Test_data/*.yaml` — never stage them). Work on branch `feature/flyinout-trips` in `GanttChartEditor`, `SchedulerWeb`, `documents`. `decorder/` is not under version control: no commit steps there. End every commit message with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
* Test commands — decoder: `cd decorder && python -m pytest tests -q` · Gantt: `cd GanttChartEditor && npx jest --config jest.config.cjs <file>` · SchedulerWeb: `cd SchedulerWeb && npx vitest run <file>`.

## Review Focus

1. **A worker whose sheet has FI/FO but no usable work** (events only, or no events at all) must produce a `review` empty trip / nothing — never an exception (Task 4 test `test_events_without_work`; Task 5/6 tests with empty `tripList`).
2. **The same worker appearing in two sheet rows** (same name ⇒ same `wid`, overwritten cells) can yield two identical events on one date → deduped, not "duplicate FO" (Task 3 `test_identical_events_deduped`).
3. **EnvConfig shapes SchedulerWeb has not seen:** worker without `unavailable_dates`, flow-style `{}`/`[]`, Japanese/quoted names, numeric-looking ids → injection must not corrupt them (Task 9 round-trip tests).
4. **`trip_list` entries with `null` region/fi/fo, unknown `state`, or malformed dates** must load without crashing and unknown `state` must degrade to `review` (Task 6 `parseTrip` tests).
5. **Open-ended trips (`fi: null` / `fo: null`) and trips partly outside the visible date range** must draw clipped bars without `FI`/`FO` labels on the open side and without verifier false positives (Tasks 7, 10 tests).

---

## File Structure

| File | Responsibility |
|---|---|
| `documents/FlyInOut/wp0/check_passthrough.py` | WP0 proof checker (reads input/result YAML) |
| `decorder/trips.py` (new) | Pure logic: `split_into_stays`, `canonicalize_trips`, `is_red_font`, trip log text. No openpyxl/pandas imports |
| `decorder/decoder8.py` (new, copy of `decoder7.py`) | + FI/FO event capture, trip wiring, `trip_list` writer, CLI flags, log append |
| `decorder/tests/stay_vectors.json`, `test_stays.py`, `test_trips.py`, `test_decoder8_golden.py` | Python tests |
| `GanttChartEditor/src/utils/stays.ts` (new) | `splitIntoStays` (TS twin of Python) |
| `GanttChartEditor/src/__tests__/fixtures/stay_vectors.json` | byte-identical copy of the Python vectors |
| `GanttChartEditor/src/types/schedule.ts`, `types/appState.ts` | `Trip`, `tripList`, `lockReason`; new `Violation` types |
| `GanttChartEditor/src/services/yamlService.ts` | parse/stringify `trip_list`, `lock_reason`; strip tagged unavailable entries |
| `GanttChartEditor/src/components/GanttChart/tripViewModel.ts` (new) | `buildWorkerStints` (stored trips + derived stays) |
| `GanttChartEditor/src/components/GanttChart/workerViewModel.ts`, `WorkerTimelineGrid.tsx` | use `buildWorkerStints`; style by kind |
| `GanttChartEditor/src/services/tripConstraintService.ts` (new), `constraintService.ts`, `config/uiText.ts`, `hooks/useConstraintCheck.ts` | verifier |
| `SchedulerWeb/src/services/tripSolverEnv.ts` (new), `runService.ts`, `vitest.config.ts`, `package.json` | T0 solver-copy builder + hook |

---

### Task 1: WP0 — Prove `trip_list` and tagged unavailable days survive the real solver

**Files:**
- Create: `documents/FlyInOut/wp0/check_passthrough.py`
- Create (scratch, not committed): `TFDocker/work/wp0-a/{input,output,status}`, `TFDocker/work/wp0-b/{input,output,status}`

**Interfaces:**
- Produces: `check_passthrough.py <schedule_in> <result_a> <result_b> <worker> <date>` — exit 0 and print `PASS` iff the checks in Step 4 hold. Verdict recorded in the design doc §10 step 0.

- [ ] **Step 1: Build the image and run baseline A** with the sample inputs

```bash
cd TFDocker && docker compose build solver
mkdir -p work/wp0-a/{input,output,status} work/wp0-b/{input,output,status}
cp src/main/resource/EnvConfig.yaml src/main/resource/Schedule.yaml work/wp0-a/input/
docker run --rm -e RUN_ID=wp0-a -v "$(pwd)/work/wp0-a/input:/work/input:ro" -v "$(pwd)/work/wp0-a/output:/work/output" -v "$(pwd)/work/wp0-a/status:/work/status" --cpus 4 --memory 8g timefold-scheduler:local
```
Expected: `work/wp0-a/output/result_Schedule.yaml` exists; status JSON `"status": "Completed"`.

- [ ] **Step 2: Pick the probe** — a `(worker, date)` with a **Flexible** assignment in result A (any one; `check_passthrough.py --pick` prints one). Create B's inputs: copy A's inputs, then (a) append to `Schedule.yaml` under `schedule:` a `trip_list:` with one entry `{id: t_probe, worker: <worker>, region: <any region id>, fi: <date>, fo: <date+1>, state: fixed, origin: sheet, quality: clean, issues: []}` and (b) append to that worker's `unavailable_dates` in `EnvConfig.yaml` an entry `date: <date>`, `source: trip`, `trip: t_probe` (exact shape in Global Constraints).

- [ ] **Step 3: Run B** with the same `docker run` command, `wp0-b` paths.

- [ ] **Step 4: Implement and run the checker.** `check_passthrough.py` asserts, using PyYAML: (1) result B contains `schedule.trip_list` equal to what was written; (2) no assignment in result B for `<worker>` has `<date>` in its `work_date_list`; (3) every `plan_flexibility: Fixed` assignment in B's input appears unchanged in result B; (4) result A has no `trip_list`.

Run: `python documents/FlyInOut/wp0/check_passthrough.py <in> <resA> <resB> <worker> <date>`
Expected: `PASS`. Flexible assignments elsewhere may differ (solver is time-limited) — that is not a failure.

- [ ] **Step 5: Record and commit.** If PASS, edit design doc §10 step 0 to "verified <date>". If FAIL on (1) only, stop and report: the fallback in §10 (re-attach trips in the editor) changes Task 9's scope (report first, do not start it); if FAIL on (2), T0 cannot work as designed — stop and report.

```bash
cd documents && git add FlyInOut/wp0 FlyInOut/FlyInOut_Design20261020.md && git commit -m "docs(flyinout): WP0 pass-through proof script and verdict"
```

---

### Task 2: WP1 — Shared stay function (Python + TypeScript) with shared vectors

**Files:**
- Create: `decorder/trips.py` (first function only), `decorder/tests/stay_vectors.json`, `decorder/tests/test_stays.py`, `decorder/tests/conftest.py` (adds `decorder/` to `sys.path`)
- Create: `GanttChartEditor/src/utils/stays.ts`, `GanttChartEditor/src/__tests__/utils/stays.test.ts`, `GanttChartEditor/src/__tests__/fixtures/stay_vectors.json`

**Interfaces:**
- Produces (Python): `@dataclass(frozen=True) class Stay: region: str; first: str; last: str; days: tuple[str, ...]` and `def split_into_stays(days: Iterable[tuple[str, str]], interval_of: Callable[[str], int]) -> list[Stay]` — input `(iso_date, region)` pairs, any order, duplicates allowed; output sorted by `first`.
- Produces (TS): `export interface DayRegion { date: string; regionId: string }`, `export interface Stay { regionId: string; firstDate: string; lastDate: string; days: string[] }`, `export function splitIntoStays(days: DayRegion[], stayOffInterval: (regionId: string) => number): Stay[]`.

- [ ] **Step 1: Write `stay_vectors.json`** — array of `{name, intervals: {region: n}, default_interval: n, days: [[date, region]...], expected: [[region, first, last]...]}`. Cases (all dates 2026-09, `r1` unless stated, interval `r1: 3`, `default_interval: 1`):

| name | days (Sep) | expected |
|---|---|---|
| contiguous | 1,2,3 | `[r1,01,03]` |
| gap_below_interval_merges | 1,2,5 (off 2) | `[r1,01,05]` |
| gap_equals_interval_splits | 1,2,6 (off 3) | `[r1,01,02]`,`[r1,06,06]` |
| interval_zero_means_one | intervals `r1: 0`; 1,3 (off 1) | two stays; and 1,2 → one |
| weekend_interval_2_splits | intervals `r1: 2`; 4(Fri),7(Mon) (off 2) | two stays |
| weekend_interval_3_merges | `r1: 3`; 4,7 | one stay |
| region_change_splits | 1:r1, 2:r2 | `[r1,01,01]`,`[r2,02,02]` |
| same_day_two_regions_min_wins | 1:r2 and 1:r1 | `[r1,01,01]` |
| duplicates_collapse | 1,1,2 | one stay, `days == (01,02)` |
| unknown_region_uses_default | region `rX` days 1,3 (off 1), no interval | two stays |

- [ ] **Step 2: Write both tests** — `test_stays.py::test_vectors` (parametrized over the JSON) and `stays.test.ts` "matches shared vectors" (same loop; import the fixture with `resolveJsonModule`, or read it with `fs` if Jest cannot import JSON). Identity of the two vector files is checked by the shell command in Step 6.

- [ ] **Step 3: Run both → FAIL**: `python -m pytest tests/test_stays.py -q` (ImportError) and `npx jest --config jest.config.cjs src/__tests__/utils/stays.test.ts` (module not found).

- [ ] **Step 4: Implement `split_into_stays` in `decorder/trips.py` and `splitIntoStays` in `stays.ts`** per Global Constraints (dedupe by `(date, region)`, pick min region per date, sort by date, walk: new stay on region change or `off ≥ max(1, interval_of(region))`).

- [ ] **Step 5: Run both → PASS.**

- [ ] **Step 6: Verify the two vector files are identical, then commit Gantt part**

```bash
cmp decorder/tests/stay_vectors.json GanttChartEditor/src/__tests__/fixtures/stay_vectors.json && echo SAME
cd GanttChartEditor && git add src/utils/stays.ts src/__tests__/utils/stays.test.ts src/__tests__/fixtures/stay_vectors.json && git commit -m "feat(trips): splitIntoStays with Timefold's stay rule"
```
Expected: `SAME`.

---

### Task 3: WP2a — `decoder8.py`: behaviour-neutral FI/FO event capture

**Files:**
- Create: `decorder/decoder8.py` (copy of `decoder7.py`)
- Modify: `decorder/trips.py` (add `FifoEvent`, `is_red_font`)
- Test: `decorder/tests/test_trips.py`

**Interfaces:**
- Produces (`trips.py`): `@dataclass(frozen=True) class FifoEvent: wid: str; date: str; kind: str; sheet: str; row: int; col: str; fill: str; red_font: bool` (`kind` ∈ `FI|FO`; `date` ISO; `fill` ∈ `plain|red|grey`; `col` Excel letters) and `def is_red_font(cell) -> bool` (duck-typed: `cell.font.color` with `.type == "rgb"` and `.rgb[-6:] == "FF0000"`, or `.type == "indexed"` and `.indexed == 10`; any missing attribute ⇒ `False`).
- Produces (`decoder8.parse_su_others` return dict): new key `"fifo_events": list[FifoEvent]`, in sheet order, **including** events in red/grey-filled cells.

- [ ] **Step 1: Write failing tests** in `test_trips.py`: `test_is_red_font_rgb` (`FFFF0000` and `FF0000` ⇒ True; `FF000000` ⇒ False), `test_is_red_font_indexed` (`indexed=10` ⇒ True), `test_is_red_font_missing_attrs` (`SimpleNamespace()` ⇒ False; no raise).

- [ ] **Step 2: Run → FAIL**; implement `FifoEvent`, `is_red_font` in `trips.py`; run → PASS.

- [ ] **Step 3: Copy** `decoder7.py` → `decoder8.py`; update the header comment to say "Decoder8 = decoder7 + FI/FO trips (see documents/FlyInOut)".

- [ ] **Step 4: Instrument the cell loop in `parse_su_others`** (`decoder8.py`, loop beginning near the `_is_red_cell(cell)` check, original line ~1188): *before* that check, compute `fifo = NFKC(str(value)).strip().upper()` for string cells; if it is `FI` or `FO`, append a `FifoEvent` (`fill` from `_is_red_cell`/`_is_grey_cell`, `red_font` from `is_red_font`, `col` via `openpyxl.utils.get_column_letter(c)`, `row` = `r`, `date` = ISO of `dt`). Do **not** change any existing branch/`continue`. Add `fifo_events` to the returned dict.

- [ ] **Step 5: Test against the real file** (`test_decoder8_golden.py::test_event_counts`, skipped if the xlsm is missing): call `parse_su_others("20260915 SU_Others_skill level有配布禁止_.xlsm")` and assert `len(FI)==2097`, `len(FO)==2334`, `sum(e.fill=="red" and e.kind=="FI")==241`, `sum(e.fill=="red" and e.kind=="FO")==388`.

Run: `python -m pytest tests/test_decoder8_golden.py::test_event_counts -q` · Expected: PASS (these are the numbers measured in design doc §2.3, plus the red-fill finding).

---

### Task 4: WP2b — Canonicalizer `canonicalize_trips`

**Files:**
- Modify: `decorder/trips.py`
- Test: `decorder/tests/test_trips.py`

**Interfaces:**
- Consumes: `FifoEvent`, `split_into_stays`, `Stay` (Tasks 2–3).
- Produces:
  - `@dataclass class Trip: id: str; worker: str; region: str | None; fi: str | None; fo: str | None; state: str; origin: str; quality: str; issues: list[str]` (ISO dates).
  - `@dataclass class Dropped: event: FifoEvent; reason: str` (`reason` ∈ `duplicate|orphan`).
  - `def canonicalize_trips(events: list[FifoEvent], work_days: dict[str, list[tuple[str, str]]], interval_of: Callable[[str], int], *, sheet_start: str, sheet_end: str, plan_start: str, plan_end: str, tol_days: int = 14, lookback_days: int = 60, enforce: str = "clean", other_region: str = "r_other") -> tuple[list[Trip], list[Dropped]]` — `work_days[wid]` = `(iso, region)` pairs.

**Algorithm (decisions the tests do not determine):** per worker, stays = `split_into_stays`. Walk stays left to right. FI candidate window for a stay = events `kind=FI` with `first − tol ≤ date < first` and `date >` the previous stay's `last` and not already used; the nearest to `first` wins, other candidates in the window ⇒ `Dropped(duplicate)` + issue `dup_fi_dropped`. FO symmetric (`last < date ≤ last + tol`, `date <` next stay's `first`). No FI candidate ⇒ if `first − 1 < sheet_start` then `fi=None` (legitimate open start, no issue) else `fi = first − 1` + issue `fi_derived`; FO symmetric with `sheet_end`/`fo_derived`. `quality`: `clean` (no issues) / `repaired` (any `*_dropped`, `*_derived`) / `suspect`. Region `r_other` ⇒ issue `region_other`; any chosen endpoint event with `red_font` ⇒ issue `red_font`. Leftover unused events: consecutive unused `FI` then `FO` (FI<FO) with **no stay work between** ⇒ `Trip(region=None, quality="suspect", issues=["empty_trip"])`; every other unused event ⇒ `Dropped(orphan)`. `state` per Global Constraints and `enforce` (`clean`: as stated; `clean+repaired`: `repaired` also eligible; `none`: all `review`). Final filter: keep trips with `fo is None or fo ≥ plan_start − lookback_days` and `fi is None or fi ≤ plan_end`. Ids `t0001…` assigned after sorting by `(worker, fi or first work date)`. Events identical in `(wid, date, kind)` are deduped first (no `Dropped`).

- [ ] **Step 1: Write failing tests** (helper `ev(w, date, kind, **kw)`; work as `[(d, "r1") ...]`; `sheet_start="2026-01-01"`, `sheet_end="2027-03-31"`, `plan_start="2026-09-15"`, interval 3, `tol_days=14`, trips below asserted on `(region, fi, fo, state, quality, issues)`). Use plan dates in Oct 2026 unless noted:

| test | input | expected |
|---|---|---|
| `test_clean_trip` | work r1 10-02…10-10; FI 10-01; FO 10-11 | one trip `(r1, 10-01, 10-11, fixed, clean, [])` |
| `test_fo_three_days_late_is_bound` | same but FO 10-13 | `fo == 10-13`, clean |
| `test_duplicate_fo_nearest_wins` | FO 10-11 and FO 10-12 | `fo == 10-11`, dropped `[(FO 10-12, duplicate)]`, `repaired`, `review`, issue `dup_fo_dropped` |
| `test_missing_fi_derived` | stay A 10-02…10-10 with FI 10-01, FO 10-11; stay B 10-20…10-25 with FO 10-26 only | B: `fi == 10-19`, issues `["fi_derived"]`, `repaired` |
| `test_fi_fi_second_dropped` | FI 10-01 and FI 10-03, work 10-04…10-10, FO 10-11 | nearest FI (10-03) wins; FI 10-01 ⇒ `Dropped(duplicate)`, `dup_fi_dropped`, `repaired` |
| `test_open_start_is_clean` | work starts 2026-01-01 (sheet start), FO only | `fi is None`, `clean`, `fixed` |
| `test_open_end_is_clean` | work ends 2027-03-31, FI only | `fo is None`, clean |
| `test_region_other_is_review` | work region `r_other`, clean events | `review`, issue `region_other` |
| `test_red_font_is_review` | clean but FI `red_font=True` | `review`, issue `red_font`, quality `clean` |
| `test_two_regions_split` | r1 10-02…10-05, r2 10-08…10-12, FI 10-01, FO 10-13 | two trips: `(r1, 10-01, 10-06 derived)` and `(r2, 10-07 derived, 10-13)`, both `repaired`/`review` |
| `test_empty_pair_is_suspect` | FI 10-01, FO 10-05, no work | trip `region None`, `suspect`, `empty_trip`, `review` |
| `test_stray_far_event_dropped` | FO 60 days after any work | `Dropped(orphan)` |
| `test_events_without_work` | events only, `work_days == {}` | no exception; pairs ⇒ empty trips, singles ⇒ orphans |
| `test_identical_events_deduped` | FO 10-11 twice (same worker/date) | one trip, `dropped == []`, `clean` |
| `test_enforce_modes` | a `repaired` trip | `clean`⇒review, `clean+repaired`⇒fixed, `none`⇒review; a `clean` trip under `none`⇒review |
| `test_lookback_and_range_filter` | trip with `fo` 90 days before plan_start; trip with `fi` after plan_end | both removed |

- [ ] **Step 2: Run → FAIL** (`python -m pytest tests/test_trips.py -q`).
- [ ] **Step 3: Implement `canonicalize_trips`** in `trips.py` per the algorithm block.
- [ ] **Step 4: Run → all PASS.**

---

### Task 5: WP2c — Wire decoder8: build `trip_list`, write YAML, CLI, log, golden test

**Files:**
- Modify: `decorder/decoder8.py` (`build_env_and_schedule_decoder7` → keep the name; before `schedule = {…}` near line 2656; `_write_schedule_yaml`; `main`)
- Modify: `decorder/trips.py` (add `trips_log_text`)
- Test: `decorder/tests/test_decoder8_golden.py`

**Interfaces:**
- Consumes: `canonicalize_trips`, `Trip`, `Dropped`, `su_data["fifo_events"]` (Tasks 3–4).
- Produces: `def trips_log_text(trips: list[Trip], dropped: list[Dropped], events: list[FifoEvent], worker_names: dict[str, str]) -> str` — section titled `---------------------- FI/FO TRIPS ----------------------` with counts `clean / repaired / suspect / dropped (duplicate|orphan)`, `fixed vs review`, per-worker non-clean lines, every dropped event as `<worker> <date> <FI|FO> sheet=<sheet> cell=<col><row> reason=<…>`, and a list of all `red_font` events by cell address (design D5).
- Produces CLI: `--trips-enforce {clean,clean+repaired,none}` (default `clean`), `--trips-lookback-days` (60), `--trips-tol-days` (14).
- Produces `Schedule.yaml`: after `assignment_list`, `  trip_list:` entries with keys in order `id, worker, region, fi, fo, state, origin, quality, issues` (`null` literal for absent region/fi/fo, dates `YYYY/MM/DD`, `issues` as flow list). **Emitted only when the list is non-empty.**

- [ ] **Step 1: Write failing golden test** `test_trip_list_written_and_rest_unchanged` (skip if inputs missing): run `decoder7` and `decoder8` (via `subprocess`, same args as the header comment of `decoder7.py`, outputs into `tmp_path`) and assert: `EnvConfig.yaml` byte-identical; `Schedule.yaml` of decoder8 starts with the full decoder7 text, the remainder starting `  trip_list:`; decoder8 log starts with the decoder7 log; remainder contains `FI/FO TRIPS`; parsed `trip_list` has ≥ 1 `state: fixed`; `--trips-enforce none` run has **zero** `fixed`.

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement.** In `build_env_and_schedule_decoder7`: build `op_region` (tool operation id → region via `tool_tasks_for_yaml[*].fab` → `environment["fab_list"]` region; misc id → its `region`), `work_days` from `assignments` (dates with `hour > 0`, ISO), `interval_of` from `environment["region_list"][*].stay_off_interval` (missing ⇒ 1), sheet range from `su_data["plan_range"]`, then `schedule["trip_list"] = [asdict-like dicts]`; append `trips_log_text(...)` to `log_out` (open in `"a"` mode after `write_transformation_log`). Add the CLI flags and pass through. Extend `_write_schedule_yaml` as specified.
- [ ] **Step 4: Run the golden test + full suite → PASS** (`python -m pytest tests -q`); runtime of the decoder run is slow — run with `-k golden` in the background if > 2 min.
- [ ] **Step 5: Manually inspect** the log section: trips are roughly one per work stay (design §2.2 counted 1,958 stays with tool-code work, more with misc work); the target is ≥ 70 % `clean`. Report the actual `clean/repaired/suspect` and `fixed/review` split to the user — this is the number that shows whether D2 is comfortable.

---

### Task 6: WP3 — Gantt types and YAML round-trip for `trip_list`, `lock_reason`; strip tagged unavailable entries

**Files:**
- Modify: `GanttChartEditor/src/types/schedule.ts`, `src/services/yamlService.ts`
- Test: `GanttChartEditor/src/__tests__/services/yamlService.test.ts`

**Interfaces:**
- Produces (`schedule.ts`): `export type TripState = 'fixed' | 'review'`; `export interface Trip { id: string; worker: string; region: string | null; fi: string | null; fo: string | null; state: TripState; origin: 'sheet' | 'user'; quality: 'clean' | 'repaired' | 'suspect'; issues: string[] }`; `ScheduleData.tripList?: Trip[]`; `Assignment.lockReason?: string`. Dates internal `YYYY-MM-DD`.
- Behaviour: `parseScheduleYaml` fills `tripList` only when `trip_list` is present; unknown `state` ⇒ `'review'`; unknown `quality` ⇒ `'suspect'`; missing `issues` ⇒ `[]`. `stringifyScheduleYaml` writes `trip_list` (same key order/format as Task 5; `null` literals) only when `tripList` is a non-empty array, and `lock_reason` only when defined. `parseUnavailableDates` (EnvConfig) drops entries whose `source === 'trip'` and entries that parse to `{}`.

- [ ] **Step 1: Write failing tests** (add a `describe('trips')` block):
  - `parses trip_list with null region/fi/fo` → `{region: null, fi: null, fo: '2026-09-11'}`
  - `unknown state degrades to review`
  - `round-trips tripList and lockReason` (parse → stringify → parse deep-equal)
  - `omits trip_list when tripList is undefined or empty` (string does not contain `trip_list`)
  - `strips source:trip unavailable entries on EnvConfig load` — worker YAML with entries `- single: {days: [2026/09/20]}` and `- date: 2026/09/01 / source: trip / trip: t1` ⇒ `unavailableDates` equals `[{single:{days:['2026-09-20']}}]`
  - `drops entries that parse to empty` — `- date: 2026/09/01` (untagged) ⇒ `[]`
  - `a YAML without trip_list stringifies byte-identically to before` (use an existing fixture string in the test file; compare against the pre-change output captured in the test as an inline expected substring check on the `assignment_list` tail: file ends with the last `description:` line).
- [ ] **Step 2: Run → FAIL** (`npx jest --config jest.config.cjs src/__tests__/services/yamlService.test.ts`).
- [ ] **Step 3: Implement** types, `parseTrip`, `parseAssignment` (`lock_reason`), stringify blocks, and the filter in `parseUnavailableDates`.
- [ ] **Step 4: Run → PASS**; also run the whole Jest suite (`npx jest --config jest.config.cjs`) → no regressions; `npx tsc -b` → no type errors.
- [ ] **Step 5: Commit**

```bash
cd GanttChartEditor && git add src/types/schedule.ts src/services/yamlService.ts src/__tests__/services/yamlService.test.ts && git commit -m "feat(trips): parse/serialize trip_list and lock_reason, strip tagged unavailable days"
```

---

### Task 7: WP4a — Trip-aware stints in the view model (no rendering change yet)

**Files:**
- Create: `GanttChartEditor/src/components/GanttChart/tripViewModel.ts`, `src/__tests__/gantt/tripViewModel.test.ts`
- Modify: `src/components/GanttChart/workerViewModel.ts` (remove `computeFlightStintsMultiRegion`; call `buildWorkerStints`; re-export `FlightStint`), `src/__tests__/gantt/workerViewModel.test.ts`

**Interfaces:**
- Consumes: `splitIntoStays` (Task 2), `Trip` (Task 6).
- Produces (`tripViewModel.ts`):
  ```ts
  export interface FlightStint {
    regionId: string; firstWorkDate: string; lastWorkDate: string;
    flightInDate: string; flightOutDate: string;      // existing fields, same meaning
    kind: 'fixed' | 'review' | 'derived';
    tripId?: string; openStart?: boolean; openEnd?: boolean;
  }
  export interface StintAssignment { startDate: string; endDate: string; regionId: string; workDates?: string[] }
  export function buildWorkerStints(input: {
    workerId: string; assignments: StintAssignment[]; trips: Trip[];
    stayOffInterval: (regionId: string) => number; viewStart: string; viewEnd: string;
  }): FlightStint[]
  ```
  `workerViewModel.ts` keeps `export type { FlightStint } from './tripViewModel'` so `WorkerTimelineGrid` imports are unchanged.

**Rules (not determined by tests alone):** (1) every trip of `workerId` becomes a stint: `flightInDate = fi ?? viewStart` with `openStart = fi === null`, `flightOutDate = fo ?? viewEnd` with `openEnd`, `regionId = region ?? ''`, `kind = state`, `first/lastWorkDate` = first/last assigned day inside the window or `''` if none. (2) assigned days **not inside any trip window (inclusive)** are grouped with `splitIntoStays` into `derived` stints with `flightIn = first − 1`, `flightOut = last + 1`. (3) days of an assignment = `workDates` when non-empty else the `startDate..endDate` range (use `generateDateRange`). (4) misc tasks stay in the input like today (region from the misc task). (5) result sorted by `flightInDate`.

- [ ] **Step 1: Write failing tests** (`tripViewModel.test.ts`, interval 3, view `2026-09-01..2026-09-30`):
  - `derived only matches old behaviour for a simple stay` — one assignment r1 09-02..09-10, no trips ⇒ one `derived` stint `(09-01 → 09-11)`.
  - `fixed trip drawn as stored dates even when work is shorter` — trip `fi 09-01 fo 09-15`, work 09-02..09-10 ⇒ one `fixed` stint `(09-01 → 09-15)`, `firstWorkDate 09-02`, `lastWorkDate 09-10`, and **no** extra derived stint.
  - `work outside the window becomes its own derived stint` — same trip plus work 09-20..09-22 ⇒ 2 stints, second `derived (09-19 → 09-23)`.
  - `empty fixed trip still drawn` — no work ⇒ stint with `firstWorkDate ''`.
  - `open start/end` — `fi null` ⇒ `flightInDate 09-01`, `openStart true`; `fo null` ⇒ `flightOutDate 09-30`, `openEnd true`.
  - `review kind passes through`; `other worker's trips ignored`.
  - `workDates beats range` — assignment 09-04..09-07 (Fri–Mon) with `workDates [09-04, 09-07]`, interval 2 ⇒ two derived stints (weekend gap splits, D4).
  - In `workerViewModel.test.ts`: `row.flightStints[0].kind === 'derived'` for the existing SCHEDULE fixture, and with `tripList` set the row gets the `fixed` stint.
- [ ] **Step 2: Run → FAIL**; **Step 3: Implement** `buildWorkerStints`; in `buildWorkerTimelineModel` pass `schedule.tripList ?? []`, `regionStayOff` as `r => regionStayOff.get(r) ?? 0`, `dates[0]`, `dates[dates.length-1]`, and add `workDates` (hour > 0 days) when collecting `workerRegionAssignments`.
- [ ] **Step 4: Run** `npx jest --config jest.config.cjs src/__tests__/gantt` and `npx tsc -b` → PASS.
- [ ] **Step 5: Commit** (`git add` the four files above; message `feat(trips): trip-aware FI/FO stints using Timefold's stay rule`). Behaviour change to record in Task 11 Step 4: stays whose off-gap equals the region's `stay_off_interval` now split (previously merged), and Fri→Mon weekends split when the interval is 2.

---

### Task 8: WP4b — Render fixed / review / derived distinctly

**Files:**
- Modify: `GanttChartEditor/src/components/GanttChart/WorkerTimelineGrid.tsx` (`FlightStintBar`, caller near line 937), `src/config/uiText.ts`
- Test: `src/__tests__/components/workerGridUnavailableOpacity.test.tsx` pattern → new `src/__tests__/components/flightStintBar.test.tsx`

**Interfaces:**
- Consumes: `FlightStint.kind/openStart/openEnd/tripId` (Task 7).
- Produces: `UI.tripKindLabel: Record<'fixed' | 'review' | 'derived', string>` = `{fixed: '固定', review: '確認待ち', derived: '予測'}`.

Visual contract: `fixed` = solid bar, opacity 0.95, 2 px solid darker outline; `review` = amber `#f9a825` dashed outline, opacity 0.7; `derived` = today's look but dashed outline, opacity 0.45. `FI` label hidden when `openStart`, `FO` label hidden when `openEnd`. Tooltip: `${regionName}  ${flightInDate} → ${flightOutDate}  [${kindLabel}]` (empty `regionName` ⇒ `地域不明`). Add `data-kind={stint.kind}` on the bar element for tests.

- [ ] **Step 1: Write failing tests** rendering `FlightStintBar` (export it for tests if not exported) with each kind: assert `data-kind`, computed opacity values, absence of the `FI` text when `openStart`, absence of `FO` when `openEnd`, tooltip text on hover for a `fixed` stint.
- [ ] **Step 2: Run → FAIL**; **Step 3: Implement** styles/labels; **Step 4: Run → PASS**; manual check with `npm run dev:mock` is deferred to Task 11.
- [ ] **Step 5: Commit** (`git add` the three files; `feat(trips): draw fixed/review/derived FI-FO bars distinctly`).

---

### Task 9: WP5 — SchedulerWeb: inject tagged travel days into the uploaded EnvConfig copy

**Files:**
- Create: `SchedulerWeb/src/services/tripSolverEnv.ts`, `SchedulerWeb/src/services/tripSolverEnv.test.ts`, `SchedulerWeb/vitest.config.ts`
- Modify: `SchedulerWeb/package.json` (devDependency `vitest ^2.1.9`, script `"test": "vitest run"`), `SchedulerWeb/src/services/runService.ts` (`submitRun`)

**Interfaces:**
- Produces:
  ```ts
  export interface TravelDayResult { envYaml: string; injectedDays: number; workers: number }
  export function injectTripTravelDays(envYaml: string, scheduleYaml: string): TravelDayResult
  export async function buildSolverFiles(envFile: File, schedFile: File): Promise<{ envFile: File; schedFile: File; injectedDays: number; workers: number }>
  ```
- `submitRun` behaviour: uploads `buildSolverFiles(...)`'s files; return type becomes `SubmitResult & { travelDaysBlocked?: number }` (set when `> 0`; UI display is out of scope).
- `vitest.config.ts`: `environment: 'node'`, alias `@` → `src` (do **not** load `vite.config.ts`, which imports the local API).

**Rules:** parse `scheduleYaml` with `js-yaml` (`schedule.trip_list ?? top-level trip_list`); consider only `state === 'fixed'` trips; for each non-null `fi`/`fo` that lies within `schedule.plan_range` and whose worker exists in `environment.worker_list`, add `{date, source: 'trip', trip: id}` to that worker's `unavailable_dates` (create the array when missing; skip when an identical tagged entry exists). **No eligible trips ⇒ return the input strings untouched (`injectedDays: 0`, same string objects).** Otherwise `yaml.dump(obj, { lineWidth: -1, noRefs: true })`. Dates stay `YYYY/MM/DD` strings (js-yaml leaves them as strings).

- [ ] **Step 1: Install and configure vitest**

```bash
cd SchedulerWeb && npm install -D vitest@^2.1.9
```
Create `vitest.config.ts`; add the script. Run `npx vitest run` → "No test files found" (expected).

- [ ] **Step 2: Write failing tests** in `tripSolverEnv.test.ts` (inline YAML fixtures, plan range `2026/09/15..2026/12/31`):
  - `no trip_list returns the identical string object`
  - `only review trips ⇒ identical string object`
  - `fixed trip injects tagged FI and FO` — worker `w1` with `unavailable_dates: []`, trip fi `2026/10/01` fo `2026/10/11` ⇒ parsed worker has exactly `[{date:'2026/10/01',source:'trip',trip:'t1'},{date:'2026/10/11',source:'trip',trip:'t1'}]`, `injectedDays 2`, `workers 1`
  - `appends after existing entries and creates the key when missing`
  - `skips endpoints outside plan_range and null endpoints` (open start `fi: null`)
  - `skips workers not in the env`
  - `dedupes identical tagged entries`
  - `round-trip changes nothing else` — `parse(original)` deep-equals `parse(result)` after removing tagged entries, using a fixture containing a Japanese name, a numeric-looking id string `"007"`, flow-style `skill_map: {p2o1: 2}`, `description:` null, and a `single: {days: [...]}` entry
  - `buildSolverFiles keeps canonical file names` (`EnvConfig.yaml`, `Schedule.yaml`) and returns the original `File` objects when nothing is injected
- [ ] **Step 3: Run → FAIL**; **Step 4: Implement** `tripSolverEnv.ts`; wire `submitRun`; **Step 5: Run `npx vitest run` → PASS** and `npx tsc` (build type-check) → no errors.
- [ ] **Step 6: Commit**

```bash
cd SchedulerWeb && git add package.json package-lock.json vitest.config.ts src/services/tripSolverEnv.ts src/services/tripSolverEnv.test.ts src/services/runService.ts && git commit -m "feat(trips): inject tagged travel-day unavailable dates into the solver EnvConfig copy"
```

---

### Task 10: WP6 — Verifier

**Files:**
- Create: `GanttChartEditor/src/services/tripConstraintService.ts`, `src/__tests__/services/tripConstraintService.test.ts`
- Modify: `src/types/appState.ts` (`Violation.type` += `'TRIP_WORK_ON_TRAVEL_DAY' | 'TRIP_OTHER_REGION_INSIDE' | 'TRIP_EMPTY' | 'TRIP_REVIEW_PENDING'`), `src/services/constraintService.ts` (append `...checkTrips(envConfig, schedule)`), `src/config/uiText.ts`, `src/hooks/useConstraintCheck.ts`

**Interfaces:**
- Produces: `export function checkTrips(envConfig: EnvConfig, schedule: ScheduleData): Violation[]` and `export function summarizeTripViolations(v: Violation[]): { travelDay: number; otherRegion: number; empty: number; reviewPending: number }`.
- UI strings (`UI.*`): `tripWorkOnTravelDayViolation(worker, date, kind: 'FI'|'FO')` = `` `${worker}: ${date} は${kind === 'FI' ? '入国' : '出国'}日のため作業を割り当てられません` ``; `tripOtherRegionViolation(worker, date, region)` = `` `${worker}: ${date} は滞在中(${region})のため他地域の作業は割り当てられません` ``; `tripEmptyViolation(worker, fi, fo)` = `` `${worker}: ${fi}〜${fo} の固定出張に作業がありません` ``; `tripReviewPendingViolation(n)` = `` `確認待ちの出入国データが ${n} 件あります` ``.

**Rules:** only `state === 'fixed'` trips are checked (review trips only feed `REVIEW_PENDING`, one violation for the whole schedule when ≥ 1, `assignmentIndices: []`). Work dates of an assignment = `workDateList` entries with `hour > 0` (fallback: `startDate..endDate`). `TRAVEL_DAY`: a work date equals `fi` or `fo` (one violation per assignment+trip). `OTHER_REGION`: work date `d` with `fi < d < fo` (null ends open) on a **tool** task whose region (fab → region) differs from `trip.region` (misc ignored; trips with `region null` skipped). `EMPTY`: no assignment of the worker has a work date in `[fi, fo]` (open ends unbounded) ⇒ `severity 'warning'`. Others `error`, except REVIEW_PENDING `warning`.

- [ ] **Step 1: Write failing tests** (fixture: env with fab `fab1`→`r1`, `fab2`→`r2`; tool tasks `ot1`@fab1, `ot2`@fab2, misc `misc1` region `r_other`; trip `t1` worker `w1` r1 fi 09-01 fo 09-11 fixed):
  - `work on FO day → TRIP_WORK_ON_TRAVEL_DAY (error)` (assignment 09-10..09-11)
  - `work on FI day → …`
  - `r2 task on 09-05 → TRIP_OTHER_REGION_INSIDE (error)`
  - `misc task inside is ignored for region but prevents EMPTY`
  - `no assignments in window → TRIP_EMPTY (warning)`
  - `open end: fo null, work far after fi in same region → no violation`
  - `review trips are not checked; one TRIP_REVIEW_PENDING with count`
  - `region null trip skipped`
  - `no tripList → []`
  - `summarizeTripViolations counts per type`
- [ ] **Step 2: Run → FAIL**; **Step 3: Implement** + wire into `checkConstraints`, add UI strings, and in `useConstraintCheck` `console.info('[trip-metrics]', summarizeTripViolations(violations))` only when any count > 0.
- [ ] **Step 4: Run** `npx jest --config jest.config.cjs` (whole suite) and `npx tsc -b` → PASS (watch for exhaustive `switch` on `Violation.type` elsewhere; fix with the minimal added case).
- [ ] **Step 5: Commit** (`git add` the listed files; `feat(trips): verify solver output against fixed trips`).

---

### Task 11: End-to-end check and documentation sync (M1/M2 acceptance)

**Files:**
- Modify: `documents/FlyInOut/FlyInOut_Design20261020.md` (record plan deviations, verified items), nothing else

- [ ] **Step 1: Generate trips**: `cd decorder && python decoder8.py --su-others "20260915 SU_Others_skill level有配布禁止_.xlsm" --seiban-info "初期データ追加情報 _20260930.rev1.xlsx" --plan-start 2026/09/15 --plan-end 2027/07/31 --seiban-start-row 65 --su-sheets "予定表_2026" --skill-sheet "予定表_2026 (2)" --envconfig-out decoder8_out/EnvConfig.yaml --schedule-out decoder8_out/Schedule.yaml --log-out decoder8_out/TransformationLog.txt` → files written, log has `FI/FO TRIPS`.
- [ ] **Step 2: M1 visual check** — `preview_start` `gantt-dev-mock` (`.claude/launch.json`), load `decoder8_out/*.yaml` via the app's file open, enable "✈ 出入国バー": solid (fixed), amber dashed (review) and faint dashed (derived) bars appear; spot-check 5 workers from the log against the sheet cells. Report the `clean/repaired/suspect` and `fixed/review` totals to the user.
- [ ] **Step 3: M2 solver check** — run decoder8's output through SchedulerWeb's upload (or `docker run` with a copy produced by `injectTripTravelDays` via a one-off `npx vitest` script) and confirm `result_Schedule.yaml` has no assignment on any tagged day and keeps `trip_list`; open the result in the Gantt: Verifier shows `TRIP_*` counts (these numbers feed the T1 gate, design §14.3).
- [ ] **Step 4: Update the design doc**: add a "Plan deviations" subsection to §14 listing: (a) editor does not re-run the canonicalizer — trips are stored, editor draws stored trips + derived stays for uncovered work; (b) `TRIP_TASK_OUTSIDE`, `STAY_DERIVED_ONLY`, `TRIP_ORPHAN` dropped from Phase 1 (orphan ≡ empty); (c) no separate review list — review trips are amber on the timeline and counted in the verifier; (d) run-log note in SchedulerWeb deferred; (e) §2.3 gains the red-**fill** finding: 629 of 4,431 FI/FO cells (241 FI, 388 FO) sit in red-filled cells that decoder7 already turns into `unavailable_dates`; (f) misc tasks are ignored for the other-region check.
- [ ] **Step 5: Commit** `cd documents && git add FlyInOut && git commit -m "docs(flyinout): plan deviations and WP0-6 verification notes"`.

---

## Self-review (spec coverage)

| Spec item | Task |
|---|---|
| §10 step 0 pass-through proof | 1 |
| D4 shared stay function, vectors, Gantt rule change | 2, 7 |
| §6 canonicalizer, quality, enforce switch, log | 3, 4, 5 |
| §4.1 YAML (`trip_list`, `lock_reason`) | 5, 6 |
| `source: trip` strip on load | 6 |
| §4.5/§5.4 display kinds, open ends, empty trips | 7, 8 |
| §7.1 T0 injection at `submitRun`, byte-identical when unused | 9 |
| §8 verifier (4 of 7 codes; rest deferred, see Task 11 Step 4) | 10 |
| §14.1 WP0–WP6 / M1–M2 acceptance | 11 |
| WP7 editing, lock dialogs, reducer/undo/collab | **out of scope** (separate plan after M2 feedback) |
