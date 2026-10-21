# Timefold employee scheduler

Refactored copy of `TFDocker` (Timefold Solver employee scheduler, now on 1.35.0; baseline was 1.27.0).
Main class: `com.yourorg.scheduler.Main`.

## Build

    mvn -B -DskipTests package

Produces `target/timefold-scheduler-0.1.0.jar` and the runtime dependency jars in `target/dependency/`.

Where Maven is not installed locally, run it in Docker instead:

    docker run --rm -v "$(pwd):/w" -v tf-m2:/root/.m2 -w /w maven:3.9-eclipse-temurin-21 mvn -B -DskipTests package
    docker run --rm -v "$(pwd):/w" -v tf-m2:/root/.m2 -w /w maven:3.9-eclipse-temurin-21 mvn -B test

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

1. 1.35 (not 1.27) validates once, when a working solution is set, that every initial planning value is accepted by
   `ValueRange.contains` of its entity's range (`AbstractScoreDirector.assertValueRangeForBasicVariables`; the range is then cached
   for the solve). It is NOT an assertion: `EnvironmentMode.NO_ASSERT` does not turn it off (tried). The original model breaks it:
   manager seats start UNASSIGNED but their range excludes it, pinned seats hold their real employee but `pinnedWid` is never set
   (range `[UNASSIGNED]`), warm-start employees may fail the candidate gates (the real sample `decoder7_out` does), and a warm-started
   phase-1 block can have `startDay` after `windowStart` while its range is `[windowStart]`.
2. Fix: the model keeps its ORIGINAL ranges (same list, same order, same UNASSIGNED placement) wrapped in
   `model/ValidationTolerantRange`. It delegates everything (size, `get`, both iterators, sorted copies) to the original range
   and only `contains(v)` additionally accepts the entity's initial value, compared by identity. Move selection iterates and samples the
   range, so the tolerated value is never offered to the search: manager seats are never offered UNASSIGNED, so a manager seat that
   holds a manager can never be emptied (`ManagerSeatKeepsManagerTest`, 6000 steps on small and datagen, every best solution inspected;
   a 15000-step datagen run gave 0 re-emptied manager seats). Two earlier attempts were wrong and are gone: adding UNASSIGNED to the manager range
   (776c1b8) let the search empty manager seats without penalty, and offering the entity's current value in the range (ff6a023) kept it
   selectable for the whole solve because Timefold caches the range (3 re-emptied at 6000 steps, 19 at 15000 on datagen).
3. 1.35 `IntValueRange` cannot be sorted by a variable's `strengthComparatorClass` (`UnsupportedOperationException` from
   `AbstractCountableValueRange.sort`), which `startDay` (`StartDayStrength`) hits. `vrStartWithinWindow` returns a `ListValueRange` with
   the same ascending integers (alone this leaves the seeded results unchanged on 1.27).
4. `ValidationTolerantRange` extends the internal `AbstractCountableValueRange` (Timefold casts entity ranges to it); check it in 2.x.

Result: the `initial` sections of `small`, `datagen` and the optional real sample are unchanged, and the `seeded_run` sections are
unchanged too (fixed-step runs, seed 42, reproduce the 1.27.0 results exactly on 1.35.0), so the goldens kept their 1.27 content.
