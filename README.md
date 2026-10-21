# Timefold employee scheduler

Refactored copy of `TFDocker` (Timefold Solver employee scheduler, now on 1.35.0; baseline was 1.27.0).
Main class: `com.yourorg.scheduler.Main`.

## Build

    mvn -B -DskipTests package

Produces `target/timefold-scheduler-0.1.0.jar` and the runtime dependency jars in `target/dependency/`.

Where Maven is not installed locally, run it in Docker instead:

    docker run --rm -v "$(pwd):/w" -v tf-m2:/root/.m2 -w /w maven:3.9-eclipse-temurin-17 mvn -B -DskipTests package
    docker run --rm -v "$(pwd):/w" -v tf-m2:/root/.m2 -w /w maven:3.9-eclipse-temurin-17 mvn -B test

On Git Bash for Windows prefix the command with `MSYS_NO_PATHCONV=1` and use `$(pwd -W)` instead of `$(pwd)`.

## Docker

    docker build -t timefold-scheduler:local .

Mount `EnvConfig.yaml` and `Schedule.yaml` at `/work/input`; the result is written to `/work/output/result_Schedule.yaml`
and run status to `/work/status/<RUN_ID>.json`. Sample inputs are in `src/main/resource/`.

## Upgrade notes

### Timefold Solver 1.27.0 -> 1.35.0 (2026-10-21)

Only `timefold.version` changed in `pom.xml`. The `golden/*-1.27.json` file names are kept: they record the 1.27.0 baseline.

Compiler and test warnings that mention deprecation or the new version (`javac -Xlint:all`, with the
`@SuppressWarnings` in the sources temporarily removed so the suppressed ones show up; the normal build prints none):

- `BlockSpanVariableListener.java:[19,20]`: `ai.timefold.solver.core.api.domain.variable.VariableListener in ...variable has been deprecated and marked for removal` (reported twice, once per compile pass). Task 11 (2.x) must replace it.
- `BlockDecision.coveredWorkingDays` uses the deprecated `@ShadowVariable(variableListenerClass = BlockSpanVariableListener.class, ...)` form. `javac` prints nothing for the annotation attribute itself (only for `VariableListener`), but it is the same deprecated mechanism and is replaced together with the listener in Task 11. Both classes carry `@SuppressWarnings({"deprecation", "removal"})`.
- Test run: no deprecation output, only `SLF4J(W): No SLF4J providers were found` (no logging backend on the classpath, unchanged from 1.27).
- Internal API: `CrewSeat` and `BlockDecision` instantiate `ai.timefold.solver.core.impl.domain.valuerange.buildin.collection.ListValueRange` (an `impl` class). It still exists in 1.35.0, but check it in 2.x.

Behaviour differences found while upgrading (all found by the characterisation tests):

1. 1.35 (but not 1.27) checks once, when the working solution is set, that every assigned planning value is inside the
   entity's value range (`AbstractScoreDirector.assertValueRangeForBasicVariables`), using `ListValueRange.contains`
   (equals, i.e. identity for `EmployeeFact`). The model violated it: every flexible seat started with a fresh
   `__UNASSIGNED__` copy that was not the range's sentinel, manager seats start UNASSIGNED but their range excluded it, and
   pinned seats had `pinnedWid == null` so their range was `[UNASSIGNED]` while holding a real employee. Fix: one shared
   `CrewSeat.UNASSIGNED` sentinel used as the initial value and in every range (appended last to the manager range), and
   `pinnedWid` is now set for pinned seats. The initial solution is unchanged (all 23 constraint totals identical); the search space of
   manager seats now contains UNASSIGNED, which changes the seeded search path.
2. In 1.35 `IntValueRange` cannot be sorted by a variable's `strengthComparatorClass`
   (`UnsupportedOperationException` from `AbstractCountableValueRange.sort`), which `startDay` (`StartDayStrength`)
   hits. `vrStartWithinWindow` now returns a `ListValueRange` with the same ascending integers. This alone did not change the seeded results on 1.27.
3. Warm-start employees taken from the input schedule need not pass the candidate gates (skill, preference, manager). The fixtures do not
   contain such a case, the real sample (`decoder7_out`) does and failed with the same range error. `CrewSeat.warmStartEmployee`
   (set in `PlanBuilder.applyInitialEmployees`) is appended to that seat's range when it is missing. Initial totals of the real sample
   (`real-1.27.json`) are identical again on 1.35.0 (run once with `-Dreal.sample`).
4. Value ranges are cached per entity for the whole solve (`ValueRangeState`), so a range must not depend on the entity's current variable value.

Golden files: `initial` sections are untouched and pass. The `seeded_run` sections of `small-1.27.json` and
`datagen-1.27.json` were regenerated (the same model on 1.27.0 and 1.35.0 gives the same seeded result, so the change comes from the
model fixes above, not from the solver version).
