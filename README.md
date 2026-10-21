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

Mount `EnvConfig.yaml` and `Schedule.yaml` at `/work/input`. The result is written to `/work/output/result_Schedule.yaml`,
and the run status to `/work/status/<RUN_ID>.json`. Sample inputs are in `src/main/resource/`.

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

Note: only pinned misc seats that have a region of their own take part in the region rules. In-horizon pinned module seats
(Fixed rows of a module) still do not, as before; their history before the horizon is covered by the last-fixed-region.

## trip_list input (validation only, not enforced yet)

`Schedule.yaml` may carry a `trip_list` (next to `assignment_list`; the root may be `schedule:`-wrapped or bare). This step only
reads and validates it. The solver ignores it: a Schedule.yaml with a `trip_list` solves exactly like one without.

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

## Upgrade notes

### Timefold Solver 1.35.0 -> 2.7.1, Java 17 -> 21 (2026-10-21)

The goldens keep their `golden/*-1.27.json` file names; the names record the 1.27.0 baseline.

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

- `small` and `datagen`: the `initial` sections are byte-identical to 1.35/1.27.
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
