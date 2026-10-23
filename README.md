# Timefold employee scheduler

Refactored copy of `TFDocker`, the Timefold Solver employee scheduler. It now runs on Timefold Solver Community 2.7.1 and
Java 21. The baseline was 1.27.0 on Java 17, and an intermediate step was 1.35.0.
Main class: `com.yourorg.scheduler.Main`.

## Build

    mvn -B -DskipTests package

Needs JDK 21. Produces `target/timefold-scheduler-0.1.0.jar` and the runtime dependency jars in `target/dependency/`.

Where Maven is not installed locally, run it in Docker instead:

    docker run --rm -v "$(pwd):/w" -v tf-m2:/root/.m2 -w /w maven:3.9-eclipse-temurin-21 mvn -B -DskipTests package
    docker run --rm -v "$(pwd):/w" -v tf-m2:/root/.m2 -w /w maven:3.9-eclipse-temurin-21 mvn -B test

On Git Bash for Windows, prefix the command with `MSYS_NO_PATHCONV=1` and use `$(pwd -W)` instead of `$(pwd)`.

## Docker

    docker build -t timefold-scheduler:local .

The builder image is `maven:3.9-eclipse-temurin-21` and the runtime image is `eclipse-temurin:21-jre`.

Mount `EnvConfig.yaml` and `Schedule.yaml` (and optionally `SolverParams.yaml`) at `/work/input`. The result is written to
`/work/output/result_Schedule.yaml`, and the run status to `/work/status/<RUN_ID>.json`. Sample inputs are in `src/main/resource/`.

Run the solver (Git Bash on Windows: prefix with `MSYS_NO_PATHCONV=1` and use `$(pwd -W)`; `RUN_ID` names the status file):

    docker run --rm -e RUN_ID=run1 \
      -v "$(pwd)/work/input:/work/input:ro" -v "$(pwd)/work/output:/work/output" -v "$(pwd)/work/status:/work/status" \
      timefold-scheduler:local

A short smoke run of the first-run fixture (empty `assignment_list`, one Fixed and one Flexible trip), about 10 seconds:

    mkdir -p work/input work/output work/status
    cp src/test/resources/fixtures/firstrun/*.yaml work/input/
    printf 'stage1_unimproved_seconds: 5\nstage2_enabled: false\n' > work/input/SolverParams.yaml
    docker run --rm -e RUN_ID=firstrun -v "$(pwd)/work/input:/work/input:ro" -v "$(pwd)/work/output:/work/output" \
      -v "$(pwd)/work/status:/work/status" timefold-scheduler:local

`docker-compose.yml` does the same with `docker compose up --build solver` (folders `./work/{input,output,status}`).

The tests, including the end-to-end scenarios (`EndToEndTest`, about 30 s), run in Docker too (see "Build" above):

    docker run --rm -v "$(pwd):/w" -v tf-m2:/root/.m2 -w /w maven:3.9-eclipse-temurin-21 mvn -B test

### Input files

| File | Required | Content |
|---|---|---|
| `EnvConfig.yaml` | yes | the world: workflows, fabs, regions (calendars, `max_stay_on`, `stay_off_interval`, transit days), workers |
| `Schedule.yaml` | yes | `plan_range`, `workflow_task_list`, `misc_task_list`, `assignment_list` (may be `[]`: a first run), optional `trip_list` |
| `SolverParams.yaml` | no | a top-level map of the parameters below, for example `stage1_minutes: 30` |

### Output files

| File | Content |
|---|---|
| `/work/output/result_Schedule.yaml` | `Schedule.yaml` with the new `assignment_list`, the `trip_list` right after it, and `solver_version` as the last key |
| `/work/output/trip_warnings.txt` | only when there is something to report (see "trip_list output"); a file of an earlier run is deleted when there is none |
| `/work/status/<RUN_ID>.json` | `runId`, `status` (`Running`, `Completed`, `Failed`, `Cancelled`), `stage`, `progress`, `startedAt`, `updatedAt`, `finishedAt`, `error`, `output` |

Exit codes of the container: 0 done, 2 input file missing, 3 output file cannot be created, 4 result missing, 143 cancelled
(SIGTERM), otherwise the exit code of the solver (status `Failed`).

### Parameters

Every parameter is resolved on its own, first source that mentions the key wins:

1. command line `--key=value` (or `--key value`), after the two file arguments (`Main <EnvConfig> <Schedule> [flags]`);
2. environment variable `TF_<KEY>` (upper case, so `TF_STAGE1_MINUTES`; a blank variable counts as not set);
3. the params file `SolverParams.yaml`: `--params-file=<path>`, else `TF_PARAMS_FILE`, else `${INPUT_DIR}/SolverParams.yaml`
   (`/work/input/SolverParams.yaml` in the container; `entrypoint.sh` sets `TF_PARAMS_FILE` when the file exists);
4. the default.

Key spellings are case-insensitive and ignore `_` and `-` (`--Stage1-Minutes` works). The params file is lenient: a missing file
is fine; a file that is not valid YAML or not a map is ignored with a warning; an unknown key warns; a value of the wrong type
falls back to that key's DEFAULT (not to a lower source) and warns, the other keys are still used. Warnings are printed as
`[params] WARN ...` and never stop the run. The effective values are printed in the `[params]` line at the start.

| Key | Default | Meaning |
|---|---|---|
| `stage1_minutes` | 300 | stage 1 time limit in minutes (>= 1) |
| `stage1_unimproved_seconds` | 600 | stage 1 stops after this many seconds without a new best score (>= 1) |
| `stage1_stop_when_feasible` | true | stage 1 also stops as soon as the hard score is 0 |
| `stage2_enabled` | true | false skips the polish stage (the stage 1 result is the result) |
| `stage2_minutes` | 60 | stage 2 time limit in minutes (>= 1) |
| `stage2_unimproved_seconds` | 60 | stage 2 stops after this many seconds without a new best score (>= 1) |
| `random_seed` | none | random seed for both stages |
| `trip_validation` | `lenient` | `lenient`: trip_list problems are warnings; `strict`: any trip_list warning fails the run (`InvalidInputData`) |
| `misc_neutral_regions` | `[r_other]` | misc task regions that count as "no region" (list or comma separated; empty = none) |
| `trip_history_days` | -1 | derived trips ending more than N days before the plan start are dropped from the output; -1 keeps all |
| `write_score_files` | true | false skips `initial_score.txt`, `stage1_score.txt`, `final_score.txt`, `score_progress.txt` (`solver_log.txt` is always written) |
| `allow_cut_fixed_trip` | true | an enforced Fixed trip with no same-region work inside is removed from the output and logged (`cut_fixed_trip`); also read from `TF_ALLOWCUTFIXEDFIFO`, `ALLOW_CUT_FIXED_TRIP`, `ALLOWCUTFIXEDFIFO` |
| `move_thread_count` | - | recognised but ignored (Timefold Enterprise Edition only) |

Example `SolverParams.yaml` for a quick check run:

    stage1_minutes: 2
    stage1_unimproved_seconds: 30
    stage2_enabled: false
    random_seed: 42

Any `TF_*` variable of the container is inherited by the solver: `docker run -e TF_STAGE1_MINUTES=10 ...` beats the file.
The seconds-level keys are the ones to use for very short runs (`stage1_unimproved_seconds: 5` with `stage2_enabled: false`).

The trip_list (input, marks, output) is described in the sections "trip_list input", "Trip marks (calendar)", "Stay limits
count FI to FO", "TripDeriver" and "trip_list output" below.

## Block weights

Three hard penalties of a block (named constants in `constraints/BlockConstraints`):

| Constraint | Weight |
|---|---|
| underfill (`UNDERFILL_MULT`) | 10 per missing hour |
| `phase-order` (`TIME_VIOLATION_MULT`) | 100 per day |
| `block-end-within-window` (`TIME_VIOLATION_MULT`) | 100 per day |

Why: a late block can only be fixed by shortening it and adding a worker. With 1 per day of violation, shortening alone
costs more (missing hours x 10) than it gains, so both steps were rejected and the search stayed stuck (5-module sample:
-98 hard for good). With 100 per day it reaches 0 hard in 13 to 30 s on 4 of 5 seeds.

Underfill stays larger than the worker rules (1 per clash day or hour over the cap), so an infeasible plan still staffs
every block and breaks worker rules instead of leaving blocks empty.

## Misc task region

A misc task (an id in `misc_task_list`, for example training or a visa trip) has its own `region:`. A Fixed row of such a task now
counts as presence in that region for the region rules: the transit gap, the stay limits (`max_stay_on`, `max_annual_stay`),
the soft "fewer region changes" rule, and the last-fixed-region of the history before the horizon. A module task keeps using
the region of its fab. A misc task without `region` (or with a blank one) stays region-less: no stay, no transit.

The region is resolved once: `calendar/RegionIndex` maps the region ids to an `int`, `FixedAssign.region` holds the misc
task's region, and `CrewSeat.regionIdx` (-1 = none) is set when the seat is built. The constraints only read the int.

`misc_neutral_regions` (default `[r_other]`) lists misc regions that are treated as "no region". The decoder gives every misc
task `region: r_other` and the default `transite_day_map` makes `r_other` a 1-day transit to every region, so with the default
nothing changes: all results are the same as before this feature. Any other misc region (`r1`, `r2`, ...) counts.

Change it like any parameter: `--misc_neutral_regions=r_other,r_x`, `TF_MISC_NEUTRAL_REGIONS=r_other,r_x`, or
`misc_neutral_regions: [r_other]` in `SolverParams.yaml`. `[]` makes `r_other` count too, which changes the results
(`MiscRegionTest.empty_neutral_list_makes_r_other_count` shows it on the small fixture).

Note: pinned seats hang on the dummy fixed block 0. Before solving that block is uninitialised, so only the pinned misc seats
with a region of their own (added separately) take part in the region rules. The construction heuristic initialises block 0
(it has planning variables and is not pinned; small fixture: start 0, 7 days), so after construction every pinned seat (module
rows too) joins the region rules on its own work days (`TripMarksSolverTest.dummy_fixed_block_is_initialised_after_construction`).
A pinned misc seat is then counted twice, which changes none of the three region rules (duplicate days do not change a span).
History before the horizon is covered by the last-fixed-region.

## trip_list input

`Schedule.yaml` may carry a `trip_list` (next to `assignment_list`; the root may be `schedule:`-wrapped or bare). It is read and
validated here; the enforced Fixed trips then mark the worker's calendar (see "Trip marks (calendar)" below). Flexible trips have
no effect on solving.

    trip_list:
    - worker: w1
      trips:
      - {start_date: 2026/09/01, end_date: 2026/09/10, region: r1, plan_flexibility: Fixed, description: ""}
      - {start_date: 2026/09/10, end_date: 2026/09/15, region: r2, plan_flexibility: Flexible}

The flat form (`worker` on each trip, no `trips:` list) is read too; both forms can be mixed in one list. Dates are `YYYY/MM/DD` or
`YYYY-MM-DD` with zero padding (`2026/09/01`, not `2026/9/1`). `flexibility` is an alias of `plan_flexibility` (Fixed or Flexible, any
case). Any other key of a trip is kept (`Trip.extra`, and `TripEntry.raw` has the whole map). A date value that YAML turned into a
date object is converted to a `yyyy/MM/dd` string in `raw`; strings stay as written.

`trip/TripListParser.parse(raw, TripContext, strict)` returns one `TripEntry(trip, raw, enforced, worker)` per trip in input order
and a list of warnings. `enforced` is true only for a valid Fixed trip that survives the overlap check; a valid Flexible trip is
listed with `enforced=false`. `worker` is the owner (the nested parent's worker, else the trip's own `worker`, trimmed; null = no
owner); exporters must use `entry.worker()`, never `raw.get("worker")` (a nested `raw` has no worker key). A broken Fixed trip
keeps an entry (not enforced; `trip` is null when it cannot be built, `worker` is still set) so that it can be copied to the output
unchanged; a broken Flexible trip is dropped. Broken means: worker not in EnvConfig, missing or unparseable `start_date` /
`end_date` / `region`, unknown region (not a known region: `region_list`, the fabs' regions or the transit map), `start > end`, or
`start == end` inside one trip. A MISSING or unknown `plan_flexibility` warns and the trip is treated as Flexible. If a nested trip
has its own `worker` that differs from its parent's, that warns and the parent's worker is used. Two valid Fixed trips of one
worker that overlap (touching at an end date is fine) keep the earlier one enforced (sorted by start, then end, then input order).
Exact duplicates are dropped silently. Trips outside the plan range are kept. A `trips:` that is not a list, or an element that is
not a map, is dropped with a warning (it never reaches the output).

Warnings are printed as `[trips] WARN <text>` and appended to `load_log.txt`; they are also in `ParsedSchedule.tripWarnings`
(entries in `tripEntries`, both empty without a `trip_list`). `trip_validation: strict` (see the parameters) turns any warning
into an `InvalidInputDataException` ("InvalidInputData: ..."), default `lenient`. The parser is called from
`PlanBuilder.buildProblem` (via `ScheduleParser.parseTrips`) after the calendars are built, because the region ids come from them.
`ScheduleParser.preprocessScheduleYaml` rewrites the schedule file in place for the cut-off rule: the `trip_list` is kept in it, but
unquoted `YYYY-MM-DD` dates are rewritten as timestamps (`2026-09-01T00:00:00Z`); the parser reads those too.

## Trip marks (calendar)

Every enforced Fixed trip (a valid Fixed trip that survived the overlap check) marks its worker's calendar. No score constraint is
added; a closed day simply does not exist for that worker's flexible work:

| Day of a Fixed trip (FI = `start_date`, FO = `end_date`) | Flexible work of that worker |
|---|---|
| FI day and FO day | closed for every task, even of the trip's region |
| strictly between FI and FO | open only for tasks of the trip's region (another region, or a region-less task, is closed) |
| outside FI..FO | nothing changes |

Touching trips (`first.end_date == second.start_date`) close the shared day for both. Marks are clamped to the plan range
(dayId 0 .. plan end). Flexible and not-enforced trips leave no marks.

A closed day acts like a personal day off, plus consistency: the seat does not cover that day anywhere. It produces 0 hours
(`block-no-underfill` / overfill), it is not counted by the one-task-per-day and 12 h daily cap rules, by overtime, the region
rules or the total-hours balance, and it is not written to `work_date_list`. The search is steered only indirectly (a closed day
earns no hours).

Implementation: `calendar/TripCalendar` (built once in `PlanBuilder.buildProblem`, stored in `SolveContext.tripCalendar`, an empty
calendar without trips) holds per-worker arrays indexed by `EmployeeFact.id` and dayId: `isBlocked` (FI/FO), `lockRegion` (region
index strictly inside), `presenceRegion` (FI..FO, clamped; kept for compatibility), `hasMarks`, and `tripsOf` (the unclamped
trips, read by the stay limits, see below). `calendar/WorkerCalendar.isOpen(emp,
day, regionIdx)` is the one test; it is false on a blocked day or inside a trip of another region. All day-based consumers of a
dynamic seat go through it: `ConstraintSupport.seatCoveredDays` (the region, overtime and balance rules read this), `SeatConstraints`
`expandToEmpDays` / `expandToEmpDayHours`, `BlockConstraints.producedHoursForBlock`, both per-day loops of `ScheduleExporter` and the
`[VALIDATE]` log. Workers without trips take the fast path (`hasMarks == false`: one array read, the shared covered-days list is
returned unchanged), so without a `trip_list` all results are identical to before (goldens unchanged).

- Pinned (Fixed) rows are facts and are never filtered or moved. A Fixed row of the worker on an FI/FO day, or inside a trip in
  another region / without a region, is reported once per day: `[trips] WARN fixed task conflicts with fixed trip (worker, date,
  task, task region, trip): ...; the Fixed row is kept` (stdout, `load_log.txt`, `ParsedSchedule.tripWarnings`). It never fails the
  run, also not with `trip_validation: strict`.
- Personal `unavailable_dates` keep their own handling (0 hours that day, not written; the other rules still see the worker).
  When an unavailable day falls on a closed day the trip mark wins; the result is the same as with the trip mark alone.
- The solver still may place a seat of a trip worker on a block that spans closed days; those days then produce nothing. On the
  datagen fixture with 8 crafted Fixed trips (seeded, 2000 steps) 24 of 360 seat-days of the trip workers' seats were closed
  days. No extra constraint was added (design decision D3); re-measure on a real run before adding one.

## Stay limits count FI to FO

The two stay limits measure a stay from the FI day to the FO day, both included (design section 4.1, decision D14). The travel
gap does not change: `emp-region-transit-gap` (`stay_off_interval` / `transite_day_map`) still reads worked days only and never
looks at FI/FO or trips.

| Rule | What it counts |
|---|---|
| `emp-region-stay-max-on` (`max_stay_on`) | the longest stay of a worker in a region, FI..FO |
| `emp-region-annual-stay-max` (`max_annual_stay`) | the sum of the FI..FO lengths of all the worker's stays in that region |

- Worked days of one region closer than the interval form one stay: `gap = next - previous - 1`, a new stay starts when
  `gap >= max(1, stay_off_interval)` (the same rule as before, and as the output).
- A stay without a Fixed trip runs from the day before the first task to the day after the last task:
  `length = (last - first + 1) + 2`.
- A stay that meets an enforced Fixed trip of the same worker and region uses the trip's own dates: present days are the worked
  days plus every day of the trip (FI..FO, the empty days inside included), the interval rule above runs over them, and the stay
  runs from `min(trip.start, first - 1)` to `max(trip.end, last + 1)`. A trip without any task inside is a stay of its own length.
  The trip dates are not clamped to the plan range (`TripCalendar.tripsOf`: a trip that started before the plan start counts in
  full).
- A day that is the FO of one stay and the FI of the next stay of the same region counts once in the annual sum (possible with
  `stay_off_interval: 1`; two touching trips of one region are one stay). A day between two regions (touching trips of different
  regions) is present in both.
- Penalties are unchanged: `longest - max_stay_on` and `sum - max_annual_stay` (at least 1) per (worker, region).

With the decoder's `10000` limits nothing changes; with a real limit the same plan shows two more stay days per stay than before.
The `small` fixture has finite limits, so its golden stay entries changed (+2 days per stay); the datagen and real goldens did not.

Implementation (`constraints/RegionConstraints`: `longestStay`, `totalStay`): workers without a trip in the region take the old
loop plus the `+2` arithmetic (`TripCalendar.tripsOf` returns one shared empty list). Only a (worker, region) with trips merges
the trip intervals in. A problem-fact stream adds one marker row per (worker, trip region), so a trip without tasks still forms a
(worker, region) group; it never changes while solving.

## TripDeriver (output FI/FO derivation)

`trip/TripDeriver.derive(days, fixedTrips, intervalOf, allowCutFixedTrip)` is a pure class (no solver, no file access; the
exporter calls it, see "trip_list output" below) that turns the final worked days into the trips of the output `trip_list` (design section 6). It returns
`DeriveResult(derived, cutFixed, warnings)`; the caller merges `derived` with the Fixed trips it keeps. Input days are
`WorkedDay(worker, date, region)` (region-less days are never passed in). Per worker, independently; results sorted by worker, start:

1. Dedupe days. Two regions on one day: the alphabetically smaller wins, warning `region_conflict_day`.
2. Cut (only if `allowCutFixedTrip`): a Fixed trip with no worked day of its own region strictly inside (`start < day < end`)
   goes to `cutFixed`, warning `cut_fixed_trip`. Days of other regions inside do not count. A cut trip is gone for the rest of
   the derivation (it is no longer a barrier). With `allowCutFixedTrip=true` a Fixed trip without a strict interior
   (`end - start <= 1`) is always cut, since no day can lie strictly inside it.
3. Days inside a remaining Fixed window (`start <= day <= end`) belong to that trip and are not derived.
4. Runs: a new run starts when the region changes, when the off-days between two worked days are `>= max(1, interval)`
   (`gap = next - previous - 1`; interval null/0 count as 1), or when a remaining Fixed trip lies between the two days (barrier).
5. Each run is a FLEXIBLE trip `first - 1 .. last + 1` (calendar days, weekends count, `description` null, no extras).
6. Neighbour check in start order over derived and remaining Fixed trips: a trip may start on the previous trip's end date
   (region to region, or right after a Fixed trip) but never before it; otherwise the start moves to that end date and
   `tight_transit` is warned (zero/missing transit days). Fixed trips are never modified.

Examples (interval 3): Fixed r1 09/01-09/10 with r1 days 09/03, 09/04, 09/09 and r2 days 09/12-09/14 gives one derived trip
r2 09/11-09/15. r1 09/01-09/03 followed by r2 09/05-09/06 gives r1 08/31-09/04 and r2 09/04-09/07 (shared boundary day).
r1 days 09/11-09/13 after the Fixed r1 trip give a separate r1 09/10-09/14 (Fixed trip kept). r1 09/01 and 09/05 (gap 3) split into
08/31-09/02 and 09/04-09/06. r1 09/01 then r2 09/02 gives r2 09/02-09/03 and a `tight_transit` warning.

## trip_list output

After solving, `ScheduleExporter` writes a `trip_list` right after `assignment_list` (also when the input had none), in the
per-worker nested form (`- worker: w1`, `trips: [...]`), workers sorted, trips sorted by `start_date`:

- Worked days come from the final assignment list (Fixed rows, Fixed history, newly solved rows): module task -> its fab's region,
  misc task -> its own region from `misc_task_list` (neutral regions per `misc_neutral_regions` and region-less rows are ignored).
  `TripDeriver` turns them into Flexible trips (`description: ''`); `intervalOf` is the region's `stay_off_interval`.
- Fixed input trips are copied from the parsed trip unchanged (extra keys and key order kept; the flat form is written nested).
  Fixed trips the parser could not use (malformed, not enforced) are copied too and never cut. With `allow_cut_fixed_trip=true`
  (default) an enforced Fixed trip with no same-region day strictly inside is removed and logged as `cut_fixed_trip`.
  A Fixed trip without a worker is dropped with a warning. Flexible input trips are ignored (regenerated).
- `trip_history_days` (N >= 0): derived trips with `end < plan start - N days` are dropped; negative (default) keeps all.
  Fixed entries are never trimmed.
- `trip_warnings.txt` is written next to the result file only when there is at least one warning (parser warnings, Fixed rows vs.
  Fixed trips, `cut_fixed_trip`, `tight_transit`, `region_conflict_day`, workerless Fixed trip). Without warnings the file is
  not written and a file of an earlier run is deleted.
- `solver_version` (last top-level key of `schedule`) is `SolverConstants.SOLVER_VERSION`, for example
  `timefold-scheduler 0.1.0 (timefold-solver 2.7.1)`; a test keeps the constants equal to `pom.xml`.
- The dump uses non-pretty flow style, so an empty list is written `trip_list: []` (also `workflow_task_list: []` etc.).

## Upgrade notes

### Timefold Solver 1.35.0 -> 2.7.1, Java 17 -> 21 (2026-10-21)

The goldens keep their `golden/*-1.27.json` file names; the names record the 1.27.0 baseline (the `small` and `datagen`
goldens were regenerated later for the block weights change, see "Goldens" below).

#### What changed for 2.x

- `pom.xml`: `timefold.version` is 2.7.1 and `maven.compiler.release` is 21. Both Docker images moved to Java 21.
- The OpenRewrite recipe `ai.timefold.solver.migration.ToLatest` (rewrite-maven-plugin 6.28.1, artifact
  `timefold-solver-migration:2.7.1`) did the mechanical part:
  - the score package moved (`ai.timefold.solver.core.api.score.HardMediumSoftScore`);
  - `@PlanningId` moved to `domain.common`;
  - `CountableValueRange` became `ValueRange`;
  - `strength`/`difficultyComparatorClass` became `comparatorClass`, with unchanged semantics: ascending means weakest or easiest first;
  - `ConstraintAnalysis.constraintName()` became `constraintId()`.

  The recipe compiles the project first, so it was run against the 1.35 sources. Its wildcard-import expansion was reverted.
- `ConstraintCollectors.count()`/`sum()` now return `Long`. Six `int` locals became `long`: `OvertimeConstraints` x4,
  `SeatConstraints`, `SoftConstraints`.
- Shadow variable: `VariableListener` and `@ShadowVariable(variableListenerClass)` were removed in 2.x.
  - `BlockDecision.coveredWorkingDays` is now a custom shadow variable: `@ShadowVariable(supplierName = "coveredWorkingDaysSupplier")`.
  - The supplier is annotated `@ShadowSources({"startDay", "days"})` and calls `WorkingDays.computeCoveredWorkingDays`.
  - The supplier is pure and reads the static `SolveContext` calendars, which are fixed during a solve.
  - `BlockSpanVariableListener` is deleted.
- Value ranges are plain `java.util.List`s; `vrDaysWithinWindow` keeps `ValueRangeFactory.createIntValueRange`.
  - `IntValueRange` still cannot be sorted by a `comparatorClass`, so `startDay` uses a List of the same ascending integers.
  - No `impl` value-range class is used any more.
  - Custom `ValueRange` classes cannot be used at all in 2.x: the solver casts every entity range to its sealed
    `AbstractValueRange`, and a custom class fails with a ClassCastException at solve start. So
    `model/ValidationTolerantRange` (the 1.35 fix, see below) is deleted.
- Score analysis: `ScoreExplanation`/`SolutionManager.explain` were removed. `SolutionManager.analyze` (`ScoreAnalysis`)
  is Enterprise-only, and on Community it throws `A commercial feature "Score analysis" was requested but it could not be loaded`.
  See "Internal-API dependency" below.
- `ConstraintNamesTest` reads the registered ids through the public `ConstraintVerifier` API, with no internal casts.
  The 2.x id rule is `^[a-zA-Z0-9]+[a-zA-Z0-9 _.'()-]*$`, and all 23 hyphenated ids pass.
- Per-constraint lines in `solver_log.txt` and `score_progress.txt` show the 2.x ids without the 1.x `package/` prefix and
  are sorted by id. The `*_score.txt` files keep the layout of the 1.x summary: per constraint, its score, id and match
  count, followed by the worst matches.

#### Warm start vs. value ranges (current design)

Timefold 2.x checks once, when a working solution is set, that every initial planning value lies in its entity's value range,
and fails otherwise. This applies to solving and score calculation in every environment mode, and pinned entities are not
exempt. `null` values are skipped and are planned by the construction heuristic. The 1.x model broke this rule.
`PlanBuilder.normalizeInitialValues` (run after the warm start and the seat candidates, and again before stage 2) and the
ranges now work as follows:

- `CrewSeat.UNASSIGNED` is the single shared UNASSIGNED instance. An UNASSIGNED copy on a non-manager seat becomes this
  shared instance; the meaning is the same.
- UNASSIGNED on a manager seat becomes `null`. Manager ranges never contain UNASSIGNED, so the construction heuristic fills the
  seat with a manager, and the search can never empty a manager seat (`ManagerSeatKeepsManagerTest`).
- A real warm-start employee who fails the seat's candidate gates is kept: for example, no skill for the operation, or a
  non-manager warm-started on seat 0. He is stored in `CrewSeat.warmOutOfGateEmployee`, and exactly that employee is
  appended to that one seat's range. This carries the warm start over and passes the check.
- A pinned seat's range is exactly its own employee.
- `PlanBuilder.applyInitialPlan` clamps a phase-1 warm start day after `windowStart` to `windowStart`, keeping the warm end day.
  Phase-1 blocks start at the window start by design, because their start range is `[windowStart]`.
- Every `CrewSeat` stream in the constraints uses `forEachIncludingUnassigned`. Every reader of `s.employee` checks
  `WorkerRules.isUnassigned`, which treats `null` and id 0 alike, so a `null` seat scores exactly like an UNASSIGNED one. Any
  new constraint that reads `s.employee` must keep that guard.

#### Behaviour changes

1. Manager seats that start UNASSIGNED are now filled by the construction heuristic, with managers only. On 1.x they
   started UNASSIGNED and were filled by the local search.
2. A phase-1 block that is warm-started on a day after its `windowStart` now starts at `windowStart`, keeping the warm end
   day. 1.27 and 1.35 kept the later day. A phase-1 block whose warm days all lie outside its window is still not
   warm-started, as before. On the real sample this changes one block (start day 1 -> 0) and none of the initial scores.
3. A seeded, fixed-step run gives different results, because 2.7.1 searches differently and starts from the values above.
   The `seeded_run` golden sections were regenerated once (200 steps or 2000 steps, seed 42):
   - small: `-376hard/-150medium/-7476soft` -> `-36hard/0medium/-7138soft`
   - datagen: `-280306hard/0medium/-229632soft` -> `-26031hard/0medium/-256489soft`
4. Residual: a kept out-of-gate warm-start employee stays in his seat's range for the whole solve, so the search may
   re-select him for that same seat after moving him away. 1.27 could not do that; 1.35 allowed it only through swap moves,
   see the history below. On a manager seat this means a non-manager from decoder input can stay on seat 0, or return to
   it later, although the manager rule wants a manager there. UNASSIGNED is still never offered on seat 0. In the
   weekly loop, warm starts come from solver output and pass the gates, so this only concerns decoder-made input. The
   real sample `decoder7_out` has 80 such seats, 31 of them manager seats.

#### Internal-API dependency: `solver/ConstraintBreakdown`

Per-constraint scores and match counts on Community come from Timefold's own score director with constraint matching enabled.
This is the mechanism `ConstraintVerifier` uses. It relies on internal classes: `DefaultSolverFactory.getScoreDirectorFactory`,
`createScoreDirectorBuilder().withConstraintMatchPolicy(ENABLED)`, `InnerScoreDirector.getConstraintMatchTotalMap`,
`ConstraintMatchTotal` and `ConstraintMatch`. These may change in any 2.x minor release.

`ConstraintBreakdown` is the single place to re-check on every Timefold upgrade. `ScoreLogger` (score files, score progress,
solver log) and the characterisation harness use it. `ConstraintBreakdownTest` and `CharacterizationTest` fail at once if it
breaks. Public fallback: a one-constraint provider (via `withConstraintProviderCustomProperties`) with
`SolutionManager.update` gives a constraint's score, and a `ConstraintVerifier` assertion gives its match count. That
route is slow, so it only suits tests.

#### Goldens: proof that the 23 constraints evaluate identically

- `small` and `datagen`: the `initial` sections were byte-identical to 1.35/1.27. The goldens were regenerated afterwards
  in the commit "fix: weigh phase-order and window violations 100 per day" (see "Block weights"): in the `initial`
  sections only `phase-order` and `block-end-within-window` (and so the total) may differ; in fact only `small`'s
  `phase-order` changed (-2 to -200 hard, total -1208 to -1406 hard) and `datagen`'s `initial` is unchanged. The
  `initial` sections of all other constraints are still the 1.27 values. The `seeded_run` sections follow the new search
  trajectory.
- `real-1.27.json` was not regenerated for that change (the real sample is not run in the normal workflow). Its
  `phase-order` and `block-end-within-window` entries have 0 matches, so they should still hold, but they are unverified
  until someone runs it with `-Dreal.sample=<dir>`. The statements about the real sample below are for 1.35/1.27.
- Real sample (`-Dreal.sample=<dir>`): the total score and all 23 per-constraint scores are identical. The `initial` section
  was regenerated once, and only two match counts changed: `emp-one-task-per-day` and `seat-daily-cap-12h`, from 284 to 286.
  1.35 under-counted them:
  - 1.35 `DefaultConstraintJustification.compareTo` compares facts that are neither comparable nor equal by `hashCode`, and its
    `equals` is `compareTo == 0`.
  - So two distinct matches whose keys collide on hashCode, with the same count, became one entry, and their scores were summed.
  - Example: `EmpDayKey` hash is `31*empId+dayId`. Employee 35 on days 162/163 collides with employee 37 on days 100/101
    (31*35+162 = 31*37+100 = 1247). The daily-cap key `Arrays.asList(emp, day)` collides the same way.
  - An exact dump of all matches on both versions showed 1.35 with one match of -2 for 37|100 and none for 35|162, and 2.7.1
    with -1 for each. Every other match is identical.
  - The collision appears because the out-of-gate warm employees are kept: employee 35 on seats 586 and 745, and
    employee 37 on seat 601.

  2.x reports the true number of matches.

### History: Timefold Solver 1.27.0 -> 1.35.0

Only `timefold.version` changed. 1.35 (not 1.27) introduced the one-time validation of initial values against the value
range, and the original model broke it in four ways:
- manager seats started UNASSIGNED but their range excluded it;
- pinned seats held their real employee but their range was `[UNASSIGNED]`, because `pinnedWid` is never set;
- warm-start employees could fail the gates;
- a phase-1 warm start could lie after `windowStart`.

Two attempts were wrong and were reverted:
- adding UNASSIGNED to the manager range (776c1b8) let the search empty manager seats without penalty;
- offering the entity's current value in its range (ff6a023) kept that value selectable for the whole solve, so UNASSIGNED
  became selectable on manager seats.

The 1.35 fix (27986d8) was `model/ValidationTolerantRange`. It wrapped the original range and also accepted the entity's
initial value in `contains()`, compared by identity, while iteration and sampling never offered it. Because 1.35 also
uses `contains()` in swap and pillar move filters, the tolerated value stayed acceptable to swaps for that same seat; its
safety relied on PlanBuilder giving each seat its own UNASSIGNED instance. The class extended the internal
`AbstractCountableValueRange` and cannot exist on 2.x. It was replaced by the design above.
