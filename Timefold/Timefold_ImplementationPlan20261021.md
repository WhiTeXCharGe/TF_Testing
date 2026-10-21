# Timefold `web/Timefold` — Refactor, Upgrade, Parameters, Misc Region and Trips: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `web/Timefold`, a refactored Maven copy of `TFDocker` on Timefold 2.7.1 / Java 21 that reads `trip_list`, honours misc-task regions, takes run parameters from `SolverParams.yaml`, and writes `trip_list` next to `assignment_list`.

**Architecture:** First copy `TFDocker` unchanged and pin its behaviour with a characterisation test. Then split the single 4,126-line file into packages, add parameters, upgrade 1.27 → 1.35 → 2.7.1, and only then add the new behaviour (misc region, trip calendar marks, stay-limit counting, `TripDeriver`). Every behaviour-preserving step must keep the characterisation test green.

**Tech Stack:** Java (17 until Task 11, then 21), Maven 3.9, Timefold Solver Community (1.27.0 → 1.35.0 → 2.7.1), SnakeYAML 2.2, JUnit 5, Docker.

**Spec:** `documents/Timefold/Timefold_TripList_Design20261021.md` (revision 2; decisions D1–D15 accepted). Teammate explainer: `Timefold_HowItWorks_Explained20261021.md`.

## Global Constraints

* `web/TFDocker` is **never modified** (baseline only). New project: `web/Timefold`, root package `com.yourorg.scheduler`.
* **No local Java/Maven on this machine** (checked). Run Maven through Docker: `docker run --rm -v "$(pwd):/w" -v tf-m2:/root/.m2 -w /w maven:3.9-eclipse-temurin-17 mvn -B test` (use `-21` after Task 11; prefix `MSYS_NO_PATHCONV=1` in Git Bash). Docker Desktop must be running.
* Behaviour-preserving tasks (3–7, 9, 11) change **no** constraint, weight, default or output; the characterisation test (Task 2) is the proof.
* Defaults (verbatim from today's code): stage 1 **300 min** spent, **600 s** unimproved, stop when score reaches `0hard/*medium/*soft`; stage 2 **60 min**, **60 s** unimproved; `DAILY_CAP` 12; `BASE_HOURS_PER_DAY` 8.
* New defaults: `allow_cut_fixed_trip` **true**; `misc_neutral_regions` **[r_other]**; `trip_validation` **lenient**; `trip_history_days` **-1**; `write_score_files` **true**; `stage2_enabled` **true**; `stage1_stop_when_feasible` **true**.
* Stay rule (output **and** solver): within a region a stay continues while `gap < interval`, splits when `gap >= interval`; `gap = next − previous − 1` (calendar days); `interval = max(1, region.stay_off_interval)`; a region change always splits.
* FI = calendar day before the first task; FO = calendar day after the last task. Derived stay length for limits = `(last − first + 1) + 2`. Travel gap (`regionTransitGap`) never looks at FI/FO.
* `trip_list` YAML: per worker `{worker, trips:[{start_date, end_date, region, plan_flexibility, description}]}`, dates `YYYY/MM/DD`, written **right after** `assignment_list`, trips sorted by `start_date`. Accept `flexibility` as alias of `plan_flexibility`, accept `-` dates, accept the flat form on input.
* Within one trip `start_date < end_date`; equal dates occur only between two different trips of a worker.
* Fixed trip marks: `start_date` and `end_date` days → closed for **any** task of that worker; days strictly between → open **only** for the trip's region. Marks are a calendar, **not** a score constraint (unless Task 14's measurement says otherwise).
* Personal `unavailable_dates` behave exactly as today; where a trip mark also applies, the trip mark wins.
* Parameter precedence: CLI `--key=value` → env `TF_KEY` → `SolverParams.yaml` (`$TF_PARAMS_FILE`, default `$INPUT_DIR/SolverParams.yaml`) → default. Bad values fall back to the default with a warning.
* Commits: `git init` inside `web/Timefold` in Task 1, then commit per task with explicit `git add <paths>`; end every message with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **`trip_list` in unexpected shapes** — missing, `null`, `[]`, flat form, wrong types, unknown flexibility, worker/region not in EnvConfig, overlapping Fixed trips, dates outside the plan: never an exception, never a changed run (Task 13).
2. **Region change with zero or missing transit days, and region change exactly one day apart** — FI/FO dates may collide; must produce a warning, not a crash or an overlapping pair (Task 16).
3. **Misc region default** — with `misc_neutral_regions=[r_other]` the totals must equal today's; with `[]` a misc `r_other` row must start to constrain neighbours (Task 12).
4. **Worker with a Fixed trip but no tasks / the first test with an empty `assignment_list`** — no exception; the trip is kept or cut exactly as `allow_cut_fixed_trip` says (Tasks 16, 18).
5. **Personal unavailable day on an FI/FO day, inside a trip, or next to the first/last task** — same output as without it, apart from what the unavailable day itself removes (Tasks 14, 16).

---

## File Structure

| File / package (under `src/main/java/com/yourorg/scheduler/`) | Responsibility |
|---|---|
| `Main.java` | arguments → params → run → export |
| `config/SolverParams.java`, `config/Warnings.java` | parameters + precedence; collected warnings |
| `io/YamlFiles.java`, `io/EnvConfigParser.java`, `io/ScheduleParser.java` | YAML loading and parsing (today's `loadYaml`, `parseEnv`, `parseSchedule`, `preprocessScheduleYaml`) |
| `model/*` | `EmployeeFact`, `DaySlot`, `TaskWindow`, `OpTaskMeta`, `BlockDecision`, `CrewSeat`, `SinglePassPlan`, `FixedAssign`, `FlexibleAssign`, `InitialAssign`, `ParsedEnv`, `ParsedSchedule` |
| `calendar/Calendars.java`, `calendar/SolveContext.java`, `calendar/RegionIndex.java`, `calendar/TripCalendar.java`, `calendar/WorkerCalendar.java` | closed-day sets, the shared solve state, region ↔ int, trip marks, the single `isOpen` test |
| `build/PlanBuilder.java` | blocks, seats, candidates, warm start (today's `buildEntitiesSinglePass`, `fillSeatCandidatesSinglePass`, `applyInitialPlan`) |
| `constraints/{Block,Seat,Region,Overtime,Soft}Constraints.java`, `constraints/SinglePassConstraints.java` | rule groups; the last class only assembles them |
| `solver/SolverFactories.java`, `solver/TwoStageRunner.java`, `solver/StatusReporter.java`, `solver/ScoreLogger.java` | solver config, the two stages, `status.json`, score text files |
| `trip/Trip.java`, `trip/TripListParser.java`, `trip/TripDeriver.java`, `trip/TripWriter.java` | trip model, input, output derivation, YAML writing |
| `export/ScheduleExporter.java` | assignment rows + `trip_list` (today's `ExportSchedule`) |
| `src/test/java/...`, `src/test/resources/fixtures/*`, `src/test/resources/golden/*` | tests, fixtures, characterisation goldens |

---

### Task 1: Project skeleton, baseline copy, git

**Files:**
- Create: `web/Timefold/` with `pom.xml`, `Dockerfile`, `docker/entrypoint.sh`, `docker-compose.yml`, `.gitignore`, `README.md`, `src/main/java/com/yourorg/scheduler/{EmployeeSchedule,ExportSchedule}.java` (copied **unchanged**), `src/main/resource/*.yaml` (samples)

**Interfaces:**
- Produces: a building project whose behaviour equals `TFDocker` (same main class `com.yourorg.scheduler.EmployeeSchedule`).

- [ ] **Step 1:** Copy `TFDocker` except the `score_progress_*.txt`, `Result20260525.md`, `update.txt` files. Replace the parent-pom with a single standalone `pom.xml` (artifact `timefold-scheduler`, Java 17, `timefold.version` 1.27.0, `snakeyaml.version` 2.2, JUnit Jupiter 5.10.x test dependency, surefire 3.2.x, the existing dependency-copy and exec plugins). Drop `docker/pom-standalone.xml`.
- [ ] **Step 2:** `git init`; add `.gitignore` (`target/`, `work/`, `*.log`, score text files).
- [ ] **Step 3: Build** with the Docker Maven command above, goal `package -DskipTests`. Expected: `BUILD SUCCESS`, a jar in `target/` and dependencies in `target/dependency/`.
- [ ] **Step 4:** Commit `chore: copy TFDocker baseline into web/Timefold`.

---

### Task 2: Fixtures and characterisation test

**Files:**
- Modify: `EmployeeSchedule.java` (test seam only)
- Create: `src/test/resources/fixtures/small/{EnvConfig,Schedule}.yaml`, `src/test/resources/golden/small-1.27.json`, `src/test/java/com/yourorg/scheduler/Characterization.java`, `.../CharacterizationTest.java`

**Interfaces:**
- Produces (seam in `EmployeeSchedule`): `static SinglePassPlan buildPlanForTests(String envPath, String schedPath)` — the same steps as `solveFromYaml` up to, but not including, solving; and `static SolverFactory<SinglePassPlan> factoryForTests(Integer stepLimit, Long seed)`.
- Produces (`Characterization`): `static Map<String,ConstraintTotal> initialTotals(SinglePassPlan p)` where `ConstraintTotal(String score, int matches)` for every constraint name; `static String seededRun(SinglePassPlan p, int stepLimit, long seed)` returning the final score string.

- [ ] **Step 1: Write the fixture** (hand-made, ≈3 workers, 2 modules in 2 regions with 2 fabs, one manager, one Fixed row, one misc row with `region: r_other`, `transite_day_map` r1↔r2 = 1, `stay_off_interval` 3, a personal unavailable day, a region calendar closing one date, plan range 14 days).
- [ ] **Step 2: Write the failing test** `CharacterizationTest.initial_totals_match_golden`: builds the plan from the fixture, calls `initialTotals`, compares every constraint name and score with `golden/small-1.27.json` (first run: file absent → test fails with "golden missing").
- [ ] **Step 3: Add the seam** to `EmployeeSchedule` (no logic change): extract the build part of `solveFromYaml` into `buildPlanForTests`; add `factoryForTests` that sets `TerminationConfig.setStepCountLimit` and `withRandomSeed`.
- [ ] **Step 4: Generate the golden** by running the test with `-Dgolden.write=true` (the harness writes the JSON when the property is set). Review the file: all 23 constraints registered in `defineConstraints` must appear (12 hard rules + `block-recommend-headcount-range` + 10 soft rules; use the exact ids from the source; the disabled `seat-worker-available-all-days` is not listed).
- [ ] **Step 5: Add** `CharacterizationTest.seeded_run_matches_golden` (stepLimit 200, seed 42; expected string stored in the same JSON) and an **optional** test `real_sample_initial_totals_match_golden` enabled by `-Dreal.sample=<dir>` using `decorder/decoder7_out` files (golden `real-1.27.json`, generated the same way). 
- [ ] **Step 6:** Run `mvn test` → PASS. Commit `test: characterisation of baseline behaviour`.

---

### Task 3: Extract model classes

**Files:** Create `model/*` (list in File Structure); Modify `EmployeeSchedule.java`.

**Interfaces:** Consumes the nested classes at `EmployeeSchedule.java:107–556` and the parsed-data classes at `:2244–2290`. Produces top-level public classes with identical fields, annotations and comparators; `BlockSpanVariableListener` moves with `BlockDecision` (still the 1.x listener at this point).

- [ ] **Step 1:** Move one class at a time to `model/`, fixing imports, compiling after each move (`mvn -q -DskipTests compile`).
- [ ] **Step 2:** Run `CharacterizationTest` → PASS unchanged goldens.
- [ ] **Step 3:** Commit `refactor: move model classes to model package`.

---

### Task 4: `SolveContext` replaces static globals; calendars in their own package

**Files:** Create `calendar/Calendars.java`, `calendar/SolveContext.java`; Modify `EmployeeSchedule.java` and everything that referenced `CAL`, `FIXED_*`, `DAY_TO_YEAR`, `DAY_TO_YM`, `SOLVE_MIN_DAY`, `SOLVE_MAX_DAY`, `TARGET_HOURS_PER_EMP`, `OP_CAPACITY`, `OP_AVG_SKILL`, `AFFINITY_TAG_WEIGHTS`.

**Interfaces:**
- Produces: `SolveContext` with the same data under typed fields, `static SolveContext current()`, `static void set(SolveContext)`, `static void clear()`; `Calendars` keeps `isWorkingDay(int day, String fabId)`, `transitDays(from,to)`, `stayOffInterval(region)` (min 1), `maxStayOn`, `annualMaxStay`, `annualOtLimit`, `monthlyOtLimit`, `regionOfFab`, `customerOfFab`, and the personal-off set accessor `isWorkerUnavailable(String wid, int day)`.

- [ ] **Step 1:** Create the two classes; move the data and helpers; leave one-line delegating statics in `EmployeeSchedule` only if a call site cannot move yet.
- [ ] **Step 2:** `CharacterizationTest` → PASS. Add `SolveContextTest.clear_resets_state` (two fixtures built one after the other do not leak values).
- [ ] **Step 3:** Commit `refactor: SolveContext and calendar package`.

---

### Task 5: IO, plan builder and exporter split

**Files:** Create `io/YamlFiles.java`, `io/EnvConfigParser.java`, `io/ScheduleParser.java`, `build/PlanBuilder.java`, `export/ScheduleExporter.java`; Modify `EmployeeSchedule.java`, delete `ExportSchedule.java` after its content moves.

**Interfaces:**
- `YamlFiles.load(String path): Map<String,Object>`; `EnvConfigParser.parse(String envPath): ParsedEnv`; `ScheduleParser.parse(String schedPath, ParsedEnv env): ParsedSchedule` (covers `preprocessScheduleYaml` and `parseSchedule`);
- `PlanBuilder.build(ParsedSchedule sch, ParsedEnv env): SinglePassPlan` (blocks, seats, pinned seats, candidates, warm start);
- `ScheduleExporter.overwriteScheduleWithAssignments(SinglePassPlan plan, LocalDate planStart, String schedPath, String envPath)` — same signature as today.

- [ ] **Step 1:** Move each function body as-is; keep method names inside the new classes.
- [ ] **Step 2:** `CharacterizationTest` → PASS. Add `ScheduleExporterTest.fixture_roundtrip_keeps_fixed_rows_and_replaces_flexible` (solve the fixture for 20 steps via the seam, export to a temp file, assert Fixed rows byte-equal to the input, Flexible rows replaced, `misc_task_list` untouched).
- [ ] **Step 3:** Commit `refactor: split parsing, plan building and export`.

---

### Task 6: Split the constraints

**Files:** Create `constraints/*` (list in File Structure); Modify `SinglePassConstraints`.

**Interfaces:** Each group class exposes `static Constraint[] constraints(ConstraintFactory f)`-style methods as instance methods taking `ConstraintFactory`; `SinglePassConstraints.defineConstraints` returns the same array **in the same order** with the same names and weights. Grouping: Block = `endWithinWindow, hoursValueAllowed, phaseOrder, noUnderfillByBlock, overfillAtMostOneDayByBlock`; Seat = `oneTaskPerEmpPerDay, dailyCap12h`; Region = `regionTransitGap, regionStayMaxOn, regionAnnualStayMax`; Overtime = `annualOvertimeLimit, monthlyOvertimeLimit`; Soft = the rest.

- [ ] **Step 1:** Move group by group, compile, run `CharacterizationTest` after each group.
- [ ] **Step 2:** Add `ConstraintNamesTest.names_and_order_unchanged` asserting the constraint ids/order equal the list in `golden/small-1.27.json`.
- [ ] **Step 3:** Commit `refactor: group constraints`.

---

### Task 7: Solver runner, status, score logging, `Main`

**Files:** Create `solver/SolverFactories.java`, `solver/TwoStageRunner.java`, `solver/StatusReporter.java`, `solver/ScoreLogger.java`, `Main.java`; Modify `docker/entrypoint.sh` (main class), `pom.xml` (exec main class), remove the remaining content of `EmployeeSchedule.java`.

**Interfaces:**
- `SolverFactories.build(Class<?>[] entities, Integer spentMinutes, Integer unimprovedSeconds, String bestScoreLimit, Long seed): SolverFactory<SinglePassPlan>`;
- `TwoStageRunner.run(SinglePassPlan p, TimerConfig t): SinglePassPlan` where `record TimerConfig(int stage1Minutes, int stage1UnimprovedSeconds, boolean stage1StopWhenFeasible, boolean stage2Enabled, int stage2Minutes, int stage2UnimprovedSeconds, Long seed)`; its **defaults** are the Global Constraints values;
- `StatusReporter(TimerConfig t)` computes the progress bar budget from `t` instead of the constants at `EmployeeSchedule.java:3483–3485`.

- [ ] **Step 1:** Move code; `Main` calls the same sequence as today's `main`.
- [ ] **Step 2:** `CharacterizationTest` → PASS. Build the Docker image and run the fixture once (`docker compose run` with the fixture mounted and `TF_...` unset): expected `status.json` ends `"status": "Completed"` and `result_Schedule.yaml` exists.
- [ ] **Step 3:** Commit `refactor: runner, status, logging and Main`.

---

### Task 8: `SolverParams`

**Files:** Create `config/SolverParams.java`, `config/Warnings.java`, `src/test/java/.../SolverParamsTest.java`; Modify `Main.java`, `TwoStageRunner`, `ScoreLogger` (honour `write_score_files`).

**Interfaces:**
```
record SolverParams(int stage1Minutes, int stage1UnimprovedSeconds, boolean stage1StopWhenFeasible,
  boolean stage2Enabled, int stage2Minutes, int stage2UnimprovedSeconds, Long randomSeed,
  String tripValidation, Set<String> miscNeutralRegions, int tripHistoryDays,
  boolean writeScoreFiles, boolean allowCutFixedTrip, List<String> warnings)
static SolverParams load(String[] cliArgs, Map<String,String> env, Path paramsFile)
TimerConfig timers()
```
Keys equal the names above in snake_case (`stage1_minutes` …); env = `TF_` + upper case; `allow_cut_fixed_trip` also accepts `ALLOW_CUT_FIXED_TRIP` / `ALLOWCUTFIXEDFIFO`; `move_thread_count` is recognised and ignored with a warning.

- [ ] **Step 1: Write failing tests** `SolverParamsTest`:
  - `defaults_when_nothing_given` → 300/600/true/true/60/60, `allow_cut_fixed_trip` true, `misc_neutral_regions == {"r_other"}`.
  - `precedence_cli_over_env_over_file_over_default` (same key set in all four places with different values).
  - `file_missing_is_not_an_error`; `file_malformed_yaml_falls_back_with_warning`.
  - `bad_values_fall_back_with_warning` (`stage1_minutes: -5`, `"abc"`, `0`).
  - `alias_spellings_for_allow_cut_fixed_trip`.
  - `unknown_key_is_ignored_with_warning`; `move_thread_count_warns_enterprise_only`.
- [ ] **Step 2:** Run → FAIL; **Step 3:** implement `load` (SnakeYAML for the file); wire `Main` (`TF_PARAMS_FILE`, else `$INPUT_DIR/SolverParams.yaml`) and the runner; **Step 4:** tests + `CharacterizationTest` → PASS.
- [ ] **Step 5:** Commit `feat: solver parameters (timers and flags) with precedence`.

---

### Task 9: Upgrade to Timefold 1.35.0

**Files:** Modify `pom.xml`.

- [ ] **Step 1:** Set `timefold.version` 1.35.0; build. Fix deprecation errors only (no behaviour change). Record every warning in `README.md` under "Upgrade notes".
- [ ] **Step 2:** `CharacterizationTest.initial_totals_match_golden` must pass **unchanged**; if `seeded_run_matches_golden` differs, regenerate only that expected string (search path may change between versions) and note it in the commit message.
- [ ] **Step 3:** Commit `chore: Timefold 1.35.0`.

---

### Task 10: Spike — what Community 2.7.1 provides (no production code)

**Files:** Create `src/test/java/.../spike/Spike271Test.java` (deleted at the end of Task 11).

- [ ] **Step 1:** On a scratch branch set `timefold.version` 2.7.1 and Java 21 in `pom.xml`, use `maven:3.9-eclipse-temurin-21`.
- [ ] **Step 2:** Write the spike to answer, with a pass/fail line each, and put the answers into the design doc §9:
  1. Does `SolutionManager` give a per-constraint breakdown on Community? (`explain`/`analyze`). If not → fallback "evaluate each constraint alone with a one-constraint `ConstraintProvider`".
  2. Are ids like `emp-region-transit-gap` accepted?
  3. Does a custom shadow variable (`@ShadowVariable(supplierName=…)` + `@ShadowSources({"startDay","days"})`) work for a `List<Integer>` field with `@PlanningPin` on the entity?
  4. Does the `@PlanningVariable(comparatorClass=…)` rename compile for `Integer` and `EmployeeFact` variables?
  5. Is `ValueRangeFactory.createIntValueRange` still available; what replaces `ListValueRange`?
- [ ] **Step 3:** Commit the findings to `documents/Timefold/Timefold_TripList_Design20261021.md` §9 (separate repo `documents`: `git add Timefold/…`). No commit of spike code.

---

### Task 11: Migrate to 2.7.1 / Java 21

**Files:** Modify `pom.xml`, `Dockerfile` (`maven:3.9-eclipse-temurin-21`, `eclipse-temurin:21-jre`), `model/BlockDecision.java`, `constraints/*`, `solver/*`, `calendar/*` as the spike dictates; delete the spike.

**Interfaces:** `BlockDecision.coveredWorkingDays` becomes `@ShadowVariable(supplierName = "coveredWorkingDaysSupplier")` fed by a pure method annotated `@ShadowSources({"startDay","days"})` that returns the same list `computeCoveredWorkingDays(block)` returns today (clamped to the horizon, working days of the block's fab). `BlockSpanVariableListener` is deleted.

- [ ] **Step 1:** Run the OpenRewrite recipe named on the official page `docs.timefold.ai/timefold-solver/latest/upgrading-timefold-solver/upgrade-from-v1` (do not invent the recipe id; copy it from that page).
- [ ] **Step 2:** Fix the rest by hand using the table in design §9 (collectors `count()` → `Long`, `sum(ToIntFunction)` → `sum(ToLongFunction)`, comparator renames, `ValueRange`, score getters, score-analysis replacement from Task 10).
- [ ] **Step 3:** `CharacterizationTest` must give **identical per-constraint totals** to `golden/small-1.27.json` (golden files are not edited; only the seeded-run string may be regenerated).
- [ ] **Step 4:** Build the Docker image on Java 21 and run the fixture end-to-end as in Task 7 Step 2.
- [ ] **Step 5:** Commit `chore: Timefold 2.7.1 on Java 21`.

---

### Task 12: Misc task region

**Files:** Create `calendar/RegionIndex.java`; Modify `model/FixedAssign.java`, `model/CrewSeat.java`, `build/PlanBuilder.java`, `io/ScheduleParser.java`, `constraints/RegionConstraints.java`, `constraints/SoftConstraints.java`, plus tests `RegionIndexTest`, `MiscRegionTest`.

**Interfaces:**
- `RegionIndex.of(Collection<String> regionIds)`, `int indexOf(String id)` (−1 if unknown/blank), `String idOf(int)`, `int size()`;
- `FixedAssign.region: String`; `CrewSeat.regionIdx: int` set once at construction (module tasks: fab → region; pinned misc rows: the misc task's own `region`, unless it is in `misc_neutral_regions`, then −1);
- every `CAL.regionOfFab(s.factory)` call in the region and soft constraints and the last-fixed-region seeding (`EmployeeSchedule.java:3151–3173`) uses `regionIdx` / `fa.region`.

- [ ] **Step 1: Write failing tests** `MiscRegionTest`:
  - `default_neutral_list_keeps_baseline_totals` — fixture with a misc `r_other` row beside an `r1` task: totals equal `golden/small-1.27.json`.
  - `misc_with_real_region_counts_for_transit` — a Fixed misc row region `r2` on day 5 and a flexible `r1` task starting day 6 with transit 2 → `emp-region-transit-gap` penalised; with transit 1 and a 1-day gap → not.
  - `misc_without_region_is_regionless` — no `region` key → no transit effect.
  - `empty_neutral_list_makes_r_other_count` — `misc_neutral_regions=[]`: the baseline fixture now differs from golden (documents the behaviour change).
- [ ] **Step 2:** Run → FAIL; implement; run → PASS plus `CharacterizationTest` (default config).
- [ ] **Step 3:** Commit `feat: misc tasks use their own region`.

---

### Task 13: Trip model, parser, validator

**Files:** Create `trip/Trip.java`, `trip/TripListParser.java`, `src/test/java/.../TripListParserTest.java`; Modify `io/ScheduleParser.java` (read `trip_list`, keep it in `ParsedSchedule.trips`).

**Interfaces:**
```
enum Flexibility { FIXED, FLEXIBLE }
record Trip(String worker, LocalDate start, LocalDate end, String region, Flexibility flexibility, String description, Map<String,Object> extra)
record TripEntry(Trip trip, Map<String,Object> raw, boolean enforced)       // raw = original map, used to copy Fixed trips as-is
record TripContext(Set<String> workerIds, Set<String> regionIds, LocalDate planStart, LocalDate planEnd)
record TripParseResult(List<TripEntry> entries, List<String> warnings)
static TripParseResult TripListParser.parse(Object rawTripList, TripContext ctx, boolean strict)   // strict → throws InvalidInputDataException(message)
```
`enforced` is true only for valid **Fixed** trips that survive overlap checks.

- [ ] **Step 1: Write failing tests** (one per row of design §3 validation table, names `missing_list_is_empty`, `null_list_is_empty`, `empty_list_is_empty`, `flat_form_accepted`, `alias_flexibility_accepted`, `dash_dates_accepted`, `unknown_worker_skipped_with_warning`, `missing_field_skipped_but_fixed_copied`, `unknown_region_not_enforced`, `start_after_end_skipped`, `start_equals_end_in_one_trip_skipped`, `touching_trips_allowed` (end == next start), `overlapping_fixed_second_not_enforced`, `unknown_flexibility_treated_flexible`, `outside_plan_range_still_listed`, `duplicates_deduped`, `strict_mode_throws`, `extra_keys_kept_in_raw`). Example assertion: input `[{worker: w1, trips: [{start_date: 2026/09/01, end_date: 2026/09/10, region: r1, plan_flexibility: Fixed}, {start_date: 2026/09/10, end_date: 2026/09/15, region: r2, plan_flexibility: Flexible}]}]` → two entries; entry 0 enforced; entry 1 not enforced; no warnings.
- [ ] **Step 2:** Run → FAIL; implement; run → PASS.
- [ ] **Step 3:** Warnings go to `Warnings` and are written to `/work/output/trip_warnings.txt` by Task 17; here only collected.
- [ ] **Step 4:** Commit `feat: trip_list input parsing and validation`.

---

### Task 14: Trip calendar marks and the single `isOpen` test

**Files:** Create `calendar/TripCalendar.java`, `calendar/WorkerCalendar.java`, tests `TripCalendarTest`, `TripMarksSolverTest`; Modify the five consumers: `seatCoveredDays`, `expandToEmpDays`, `expandToEmpDayHours`, `producedHoursForBlock`, and the exporter's per-day loop.

**Interfaces:**
```
TripCalendar TripCalendar.build(List<TripEntry> entries, Function<String,Integer> workerIndexOf,
                                RegionIndex regions, int planStartOffset, int horizonDays)
boolean isBlocked(int empIdx, int day)          // a Fixed trip's start or end day
int     lockRegion(int empIdx, int day)         // region index for days strictly inside, else -1
int     presenceRegion(int empIdx, int day)     // region for start..end inclusive, else -1 (used by Task 15)
boolean hasMarks(int empIdx)                    // fast path: false for workers with no trips
static boolean WorkerCalendar.isOpen(int empIdx, int day, int regionIdx)   // false if blocked, or lockRegion != -1 && != regionIdx
```
Only `enforced` entries produce marks; marks are clamped to the horizon; touching trips block the shared day for both.

- [ ] **Step 1: Write failing tests** `TripCalendarTest`: `blocked_on_start_and_end`, `interior_locked_to_region`, `touching_trips_block_shared_day`, `outside_horizon_clamped`, `worker_without_trips_uses_fast_path` (`hasMarks == false`), `flexible_and_unenforced_entries_leave_no_marks`.
- [ ] **Step 2: Write failing solver-level tests** `TripMarksSolverTest` on the fixture with a Fixed trip r1 09/01–09/10 and a flexible r1 and r2 task wanting that window: after a 200-step seeded run + export, assert (a) no row of that worker covers 09/01 or 09/10, (b) no r2 row inside, (c) r1 rows inside are allowed, (d) a personal unavailable date on 09/01 (and 09/10, and 09/05) gives the same rows as without it.
- [ ] **Step 3:** Run → FAIL; implement the calendar and route the five consumers through `WorkerCalendar.isOpen`; keep personal-off exactly as today for workers/days without trip marks.
- [ ] **Step 4:** Run tests + `CharacterizationTest` (no trips ⇒ identical totals) → PASS.
- [ ] **Step 5 (measurement gate for D3):** on the fixture and then on the real sample, run 60 s with and without trips; record whether the search wastes steps on blocked assignments. Only if it clearly does, add `constraints/TripConstraints.tripCalendarLock` (array lookup, hard, registered only when trips exist) and a test; otherwise note "not needed" in the README.
- [ ] **Step 6:** Commit `feat: fixed trip marks on a per-worker calendar`.

---

### Task 15: Stay limits count FI → FO

**Files:** Modify `constraints/RegionConstraints.java` (`maxSegmentSpanWithBreak`, `totalSegmentSpanWithBreak` callers), `calendar/TripCalendar.java`; tests `StayLimitTest`.

**Interfaces:** the two span helpers take the day list plus the worker's `presenceRegion` days; a segment's length is `(last − first + 1) + 2` for a derived stay, and for a stay overlapping a Fixed trip `max(trip.end, last + 1) − min(trip.start, first − 1) + 1`; overlapping end/start days of neighbouring stays count once. `regionTransitGap` is **not** touched.

- [ ] **Step 1: Write failing tests** `StayLimitTest` (region `max_stay_on` / `max_annual_stay` set per case, interval 3):
  - `derived_stay_counts_plus_two`: limit 5, tasks day 2–4 (3 days) → length 5 → no penalty; tasks day 2–5 → 6 → penalty 1.
  - `annual_sums_stays`: annual limit 10, two stays of length 6 → penalty 2.
  - `fixed_trip_length_counts`: Fixed trip day 1–10 with one task day 5, limit 8 → length 10 → penalty 2; limit 10 → none.
  - `touching_stays_count_shared_day_once`.
  - `travel_gap_ignores_trips`: Fixed trip r1 day 1–10 and flexible r2 task day 11 with transit 1 → no `emp-region-transit-gap` penalty; same layout with transit 2 → penalty (based on worked days only).
  - `limits_off_when_ten_thousand`: totals equal baseline golden.
- [ ] **Step 2:** Run → FAIL; implement; run → PASS plus `CharacterizationTest` (limits are 10000 in the fixture).
- [ ] **Step 3:** Commit `feat: stay and annual-stay limits count FI to FO`.

---

### Task 16: `TripDeriver` (pure class)

**Files:** Create `trip/TripDeriver.java`, `trip/TripDeriverTest.java`.

**Interfaces:**
```
record WorkedDay(String worker, LocalDate date, String region)           // region non-null, non-neutral only
record DeriveResult(List<Trip> derived, List<Trip> cutFixed, List<String> warnings)
static DeriveResult TripDeriver.derive(List<WorkedDay> days, List<Trip> fixedTrips,
                                       Function<String,Integer> intervalOf, boolean allowCutFixedTrip)
```
Algorithm exactly as design §6 and 6.2: apply the cut first; days inside a remaining Fixed trip window (inclusive) are removed; runs split by region / `gap >= interval` / a Fixed trip in between; `start = first − 1`, `end = last + 1`; neighbour clamp with warning `tight_transit` when `start < previous.end`; all derived trips `FLEXIBLE`, `description` empty; "days inside" for the cut counts only worked days with `region == trip.region` and `start < day < end`.

- [ ] **Step 1: Write failing tests** (dates in 2026, interval 3 unless stated, `allowCutFixedTrip=true` unless stated):
  - `example_from_design`: Fixed w1 r1 09/01–09/10; days r1 09/03, 09/04, 09/09; r2 09/12, 09/13, 09/14 → derived `[w1 r2 09/11–09/15]`, `cutFixed` empty.
  - `gap_below_interval_continues`: r1 09/01, 09/03 → one trip 08/31–09/04.
  - `gap_equal_interval_splits`: r1 09/01, 09/05 → 08/31–09/02 and 09/04–09/06.
  - `region_change_shares_boundary_day`: r1 09/01–09/03, r2 09/05–09/06 → r1 08/31–09/04, r2 09/04–09/07.
  - `same_region_after_fixed_trip_is_separate`: Fixed r1 09/01–09/10; r1 09/11–09/13 → derived r1 09/10–09/14.
  - `weekend_fi_fo`: first task Monday 2026-09-07 → start 2026-09-06 (Sunday).
  - `fixed_trip_without_tasks_is_cut_when_allowed` / `…_kept_when_not_allowed`.
  - `other_region_day_inside_does_not_count_as_inside`.
  - `tight_transit_warns`: r1 09/01, r2 09/02 → warning, r2 start == r1 end.
  - `no_days_no_trips`; `workers_are_independent`.
  - `invariants_property`: for 200 random day sets, trips of one worker never overlap except end == next start, every worked day lies in exactly one trip's `[start+1, end−1]`, and Fixed trips are returned byte-identical.
- [ ] **Step 2:** Run → FAIL; implement; run → PASS.
- [ ] **Step 3:** Commit `feat: TripDeriver`.

---

### Task 17: Export integration, `trip_list` writing, warnings file

**Files:** Create `trip/TripWriter.java`; Modify `export/ScheduleExporter.java`, `Main.java`; tests `TripExportTest`.

**Interfaces:**
- `TripWriter.toYaml(List<TripEntry> keptFixed, List<Trip> derived, int historyDays, LocalDate planStart): List<Map<String,Object>>` — per-worker nested form, sorted, Fixed entries emitted from `raw` unchanged;
- exporter collects `WorkedDay`s from the **final** assignment list (Fixed rows, preserved rows, solved rows; module task region from its fab, misc region from `misc_task_list` unless neutral), calls `TripDeriver`, applies `trip_history_days` (drop derived trips with `end < planStart − N` when `N ≥ 0`), places `trip_list` **immediately after** `assignment_list` in the dumped map, writes `trip_warnings.txt` next to the result (parser warnings, `cut_fixed_trip`, `tight_transit`, conflicts between Fixed rows and Fixed trips), and writes `solver_version` (artifact version + Timefold version) at the top level of `schedule`.

- [ ] **Step 1: Write failing tests** `TripExportTest`: `trip_list_follows_assignment_list` (key order in the dumped file), `fixed_trip_copied_byte_equal`, `cut_fixed_trip_removed_and_logged`, `history_days_trims_old_trips`, `no_trip_list_in_input_still_writes_derived_trips`, `warnings_file_written_only_when_warnings_exist`, `trip_list_absent_and_no_region_work_writes_empty_list`.
- [ ] **Step 2:** Run → FAIL; implement; run → PASS; `CharacterizationTest` PASS.
- [ ] **Step 3:** Commit `feat: write trip_list next to assignment_list`.

---

### Task 18: End-to-end scenarios, Docker interface, documentation

**Files:** Create `src/test/resources/fixtures/firstrun/{EnvConfig,Schedule}.yaml`, `src/test/java/.../EndToEndTest.java`; Modify `docker/entrypoint.sh`, `Dockerfile`, `docker-compose.yml`, `README.md`.

- [ ] **Step 1: First-run fixture** (your planned test): `assignment_list: []`, three modules in `workflow_task_list`, EnvConfig with 2 regions, one Fixed and one Flexible `trip_list` entry. `EndToEndTest.empty_assignment_list_first_run`: run through `Main` with a 10-second budget (`stage1_minutes` via env) and assert: assignments produced; none on the Fixed trip's start/end day; none of another region inside it; `trip_list` right after `assignment_list`; Fixed trip unchanged (or cut per option); derived trips satisfy the TripDeriver invariants.
- [ ] **Step 2:** Tests `no_trip_list_regression` (output assignments for the baseline fixture satisfy the baseline invariants, no `trip_list` content other than derived), `params_file_changes_budget` (a `SolverParams.yaml` with `stage1_minutes: 1` shortens the run), `bad_params_do_not_stop_run`.
- [ ] **Step 3:** `entrypoint.sh` passes `SolverParams.yaml` when present (`TF_PARAMS_FILE`), keeps `status.json` format and `/work/*` folders, passes through `TF_*` env vars; `docker-compose.yml` names updated; `README.md` documents parameters, files, how to run tests via Docker.
- [ ] **Step 4:** Build the image and run `docker run` with the first-run fixture mounted; expected `status.json` `Completed`, `result_Schedule.yaml` as above.
- [ ] **Step 5:** Commit `test: end-to-end scenarios and Docker interface`.

---

### Task 19: Measured performance pass

**Files:** Modify `constraints/RegionConstraints.java`, `model/CrewSeat.java`, `model/BlockDecision.java`, `calendar/SolveContext.java`; add `docs/perf.md` in `web/Timefold`.

- [ ] **Step 1:** Baseline on the real sample (`decoder7_out`): record, for a fixed 10-minute stage 1 run with seed 42, the number of score calculations/steps and the best score over time (`score_progress.txt`), plus a JDK Flight Recorder profile (`-XX:StartFlightRecording`) — save the top 10 hot methods in `docs/perf.md`.
- [ ] **Step 2:** Apply, **one at a time**, each followed by `CharacterizationTest` (identical totals) and the same 10-minute measurement; keep a change only if steps/second improves and the best score at 10 minutes is not worse: (a) register `regionStayMaxOn`/`regionAnnualStayMax` only when a region has a finite limit; (b) compute the sorted day→region map once per employee in `regionTransitGap` and once in the stay helpers instead of in both `filter` and `penalize`; (c) build `eligibleEmployeesForSeat` and the start/days value ranges once per seat/block.
- [ ] **Step 3:** Record in `docs/perf.md` what was measured for items (d)–(f) of design §11 (cached `int` regions on fixed rows, fixed-presence calendar instead of pinned seats, packed `long` keys) and whether the profile justifies doing them in a follow-up plan.
- [ ] **Step 4:** Commit per kept change, `perf: <what>`.

---

## Self-review (spec coverage)

| Spec item | Task |
|---|---|
| R1 travel gap unchanged; R13/D14 stay limits FI→FO | 15 |
| R2/R3/R6/R7 output derivation, barrier after Fixed trip, Flexible ignored | 16, 17 |
| R4/R5 calendar marks, Fixed trips independent from tasks | 14 |
| R8 misc region, D2 neutral list | 12 |
| R9/R15/D11 parameters + `SolverParams.yaml` | 7, 8, 18 |
| R10 upgrade to 2.7.1 / Java 21 (D6, D7) | 9, 10, 11 |
| R11 refactor into files; D12 `TFDocker` baseline | 1–7 |
| R12 first test with empty `assignment_list` | 18 |
| R14/D13 `allow_cut_fixed_trip` | 8, 16, 17 |
| D1 `gap >= interval` | Global Constraints, 16 |
| D3 calendar only, constraint only if measured | 14 (Step 5) |
| D4/D15 personal unavailable untouched, trip wins | 14 |
| D5 Flexible input trips ignored | 13, 16 |
| D8 `/work/*`, `status.json` kept, optional params file | 7, 18 |
| D9/D10 lenient validation, key aliases | 8, 13 |
| §11 speed suggestions | 19 |
| Decoder, Gantt, `fix_edited_modules`, SchedulerWeb, API controller changes | **out of scope** (later, after Timefold output is correct) |
