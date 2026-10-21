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

What 1.35 needed (found by the characterisation tests; no constraint code was touched):

1. 1.35 (but not 1.27) validates once, when a working solution is set, that every assigned planning value is inside the
   entity's value range (`AbstractScoreDirector.assertValueRangeForBasicVariables`). It is NOT an assertion:
   `EnvironmentMode.NO_ASSERT` does not turn it off (tried). The original model breaks it in three ways: manager seats start
   UNASSIGNED but their range excludes it, pinned seats hold their real employee but `pinnedWid` is never set (range `[UNASSIGNED]`),
   and warm-start employees may fail the candidate gates (the real sample `decoder7_out` does). Phase-1 `startDay` is the same
   case: the range is only `[windowStart]` but a warm start may place the block later.
   Fix (`CrewSeat.eligibleEmployeesForSeat`, `BlockDecision.vrStartWithinWindow`): the range is the ORIGINAL range plus the entity's
   CURRENT value when that is missing. Nothing else is added: UNASSIGNED is not offered to manager seats in any other situation, so
   a manager seat that holds a manager cannot be emptied (`ManagerSeatKeepsManagerTest`), and a manager seat filled by the search is
   never emptied again. A first attempt that put UNASSIGNED into every manager range let the search empty manager seats
   without penalty; it was replaced. `Phase1WarmStartTest` covers the warm-started phase-1 block.
2. 1.35 `IntValueRange` cannot be sorted by a variable's `strengthComparatorClass` (`UnsupportedOperationException` from
   `AbstractCountableValueRange.sort`), which `startDay` (`StartDayStrength`) hits. `vrStartWithinWindow` returns a `ListValueRange`
   with the same ascending integers (alone this leaves the seeded results unchanged on 1.27).
3. Value ranges are cached per entity inside the solver (`ValueRangeState`).

Golden files: the `initial` sections of `small`, `datagen` and the optional real sample are untouched and pass on 1.35.0, so
all 23 constraints evaluate identically. The `seeded_run` sections of `small-1.27.json` and `datagen-1.27.json` were regenerated:
the extra "current value" element in some ranges changes the random draws. The same model on 1.27.0 gives exactly the same new seeded
results, so the difference does not come from the solver version.

- small: `-376hard/-150medium/-7476soft` -> `-278hard/-150medium/-8021soft`
- datagen: `-280306hard/0medium/-229632soft` -> `-261757hard/0medium/-228419soft`
