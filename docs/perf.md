# Performance measurements (Task 19)

Measured on `feature/timefold-refactor` at `c505ced` (Timefold Solver Community 2.7.1, Java 21, Docker, 4 CPUs and -Xmx5g per
container). Nothing in `src/main` was changed by this task: no candidate reached the keep rule (see "Result"). The scripts are in
`scripts/perf/`.

## How it was measured

| Item | Value |
|---|---|
| Inputs | `datagen` = `logic/data_generator/{EnvConfig,Schedule}.yaml` (241 blocks, 720 seats, never feasible in the budget); `sample` = the 5-module `src/main/resource/{EnvConfig,Schedule}.yaml` (41 blocks, 120 seats; reaches 0 hard, so `stage1_stop_when_feasible: false` makes it use the whole budget). Both: `assignment_list: []`, limits `max_stay_on: 80`, `max_annual_stay: 240` |
| Budget | `stage1_minutes: 2`, `stage1_unimproved_seconds: 600`, `stage2_enabled: false`, `random_seed: 42`, `stage1_stop_when_feasible: false`, `write_score_files: false` |
| Speed metric | Timefold's own phase summary of the local search phase: `step total / time spent` (**steps/s**) and `move evaluation speed` (**evals/s**, = score calculations per second). Read from the INFO line `Local Search phase (2) ended`. `slf4j-simple` is on the measurement classpath only (not in the pom) so that the line is printed |
| A/B | The base build and the changed build run **at the same time** (2 containers), for 2 repetitions per input. The figure to read is the **B/A ratio of a pair**: absolute speeds differ from round to round (JIT, host load, a different search trajectory) by up to 19 percent on `sample` and 10 percent on `datagen`; the two runs of a pair differ by about 2 percent |
| Profile | JFR (`settings=profile`), one 2-minute run per input on the base build; `jfr print --events jdk.ExecutionSample --stack-depth 64`, aggregated by `scripts/perf/hot.py` |
| Commands | `scripts/perf/run.sh <label> <build dir> <input dir> [jfr] [score_files]`, `scripts/perf/ab.sh <tag> <build A> <build B> <reps> <input dir>...`, `python scripts/perf/compare.py <runs dir> <tag>...`, `python scripts/perf/hot.py samples.txt 10 [substring...]` |

Environment mode: the solver runs in `PHASE_ASSERT` (the Timefold default; the log says "default environment mode"). It was not
changed and not measured separately (see "Follow-ups").

## Step 1: baseline

Base build, no JFR (the A side of the A/B pairs; 2 minutes; rounds are separate pairs):

| Input | steps/s (4 runs) | evals/s (4 runs) | final score (4 runs, hard / soft) |
|---|---|---|---|
| datagen | 182.2, 175.8, 172.7, 191.0 | 5067, 4897, 4860, 5349 | -14503/-225183, -14532/-223517, -14562/-225114, -14492/-224223 |
| sample | 422.9, 479.1, 500.1, 339.2 | 8162, 9465, 9861, 6347 | 0/-27524, 0/-27468, 0/-27441, 0/-27671 |

With JFR on (single run per input): datagen 222.3 steps/s, 6293 evals/s, -14414hard/-224052soft; sample 516.9 steps/s, 10264 evals/s,
0hard/-27436soft. The same seed does not give the same final score twice because the budget is time, not steps (the same build
ends within about 0.5 percent hard and 1.5 percent soft of itself on `datagen`).

### Profile: top 10 frames by self time (share of all samples)

| # | datagen (7383 samples) | | sample (5948 samples) | |
|---|---|---|---|---|
| 1 | `java.util.HashMap.getNode` | 14.0% | `java.util.HashMap.getNode` | 20.0% |
| 2 | `TreeMap.merge` | 4.3% | `ArrayDeque.addLast` | 5.2% |
| 3 | `RegionConstraints.transitRelevantDays` | 4.3% | `StaticPropagationQueue.processAndClear` | 3.7% |
| 4 | `String.equals` | 4.1% | `AbstractJoinNode.innerUpdateRight` | 2.5% |
| 5 | `ElementAwareArrayList.compact` | 3.1% | `LinkedHashMap.forEach` | 2.3% |
| 6 | `TreeMap.fixAfterInsertion` | 3.0% | `ElementAwareArrayList.uncheckedEntryAt` | 2.3% |
| 7 | `AbstractJoinNode.innerUpdateRight` | 2.6% | `TreeMap.merge` | 2.1% |
| 8 | `ArrayDeque.addLast` | 2.5% | `StaticPropagationQueue.propagateRetracts` | 2.0% |
| 9 | `UniversalTuple.setStore` | 2.3% | `UniversalTuple.setStore` | 1.7% |
| 10 | `StaticPropagationQueue.processAndClear` | 2.3% | `AbstractGroupNode.lambda$new$0` | 1.6% |

The engine's incremental score calculation is 92 to 94 percent of all samples (`BavetConstraint` frames); move selection is 2.3
percent. So the time is in the constraint lambdas and the node maintenance.

### Profile: our code, inclusive (share of all samples that have the frame on the stack)

| Frame | datagen | sample |
|---|---|---|
| `RegionConstraints.regionTransitGap` lambdas (filter 13.2%, penalty 5.5% on datagen) | 19.0% | 8.5% |
| `Calendars.isWorkingDay(int, String)` (3 `HashMap<String,..>` lookups per call; the caller of 5.0% / 11.8% of the `HashMap.getNode` samples) | 8.6% | 12.2% |
| `BlockConstraints.producedHoursForBlock` (calls `isWorkingDay` per seat and day; also called twice, in filter and penalty) | 6.4% | 10.3% |
| `RegionConstraints` stay rules (`regionStayMaxOn` 2.6% + `regionAnnualStayMax` 2.0%) | 4.6% | 4.5% |
| `TreeMap` (the day-to-region maps of the transit rule) | 9.4% | 4.3% |
| `BlockDecision.coveredWorkingDaysSupplier` (shadow variable) | 1.7% | 1.3% |
| `SeatConstraints.expandToEmpDays` | 1.7% | 2.5% |
| `EmpDayRegion` + `EmpDayKey` | 2.1% | 1.4% |
| `CrewSeat.eligibleEmployeesForSeat`, `strictEligibleEmployees`, `BlockDecision.vr*` value-range providers | **0.0%** | **0.0%** |
| `SolveContext.fixed*ByEmpDay` maps, `pinnedWorkDays`, `pinnedMiscRows` | 0.0% | 0.0% |
| GC (73 / 96 collections, total pause) | 0.43 s (0.4%) | 0.23 s (0.2%) |

## Step 2: candidates (a), (b), (c)

Rule: keep a change only if the speed improves by 5 percent or more, consistently, on **both** inputs, and the final score at the
same budget is not worse; every candidate runs `CharacterizationTest` first (goldens unchanged).

| | Change | Tests | B/A steps/s datagen (2 pairs) | B/A steps/s sample (2 pairs) | Final scores | Decision |
|---|---|---|---|---|---|---|
| (a) | register `regionStayMaxOn` / `regionAnnualStayMax` only when a region has a finite limit (limit below 10000) | `CharacterizationTest`, `ConstraintNamesTest`, `StayLimitTest` green, goldens unchanged | 0.998, 0.991 | 0.976, 1.007 | equal | **reverted** (no effect: both inputs have finite limits, so both rules are registered) |
| (b) | compute the transit penalty and the two stay excesses once per group update (a `map` step) instead of in both `filter` and `penalize` | `CharacterizationTest`, `ConstraintNamesTest`, `StayLimitTest`, `MiscRegionTest`, `TripMarksSolverTest`, `ConstraintBreakdownTest` green, goldens unchanged | 1.028, 1.025 | 1.014, 0.998 | equal on sample, within noise on datagen | **reverted** (+2.5% and +0.8%, below 5%) |
| (c) | build `eligibleEmployeesForSeat` and the start/days value ranges once | not implemented | n/a | n/a | n/a | **not applied**: 0 of 13331 samples are in these methods, so the possible gain is 0% |

evals/s ratios agree with the steps/s ratios to within 1 percent (a: 0.989-0.997 datagen, 0.973-1.009 sample; b: 1.026-1.030
datagen, 0.997-1.016 sample).

### Supplementary: (a) with unlimited stay limits (not part of the keep rule)

Both measured inputs have finite limits. The real EnvConfig files use `max_stay_on: 10000` / `max_annual_stay: 10000` ("unlimited"),
where (a) removes two rules that can never fire. To show the effect, the same two inputs with every limit set to 10000 (one pair
each, same conditions):

| Input (limits 10000) | base steps/s | (a) steps/s | B/A | evals/s B/A | final score base | final score (a) |
|---|---|---|---|---|---|---|
| datagen | 177.6 | 233.2 | **1.313** | 1.342 | -14526hard/-225514soft | -14349hard/-220385soft |
| sample | 394.4 | 542.6 | **1.376** | 1.447 | 0hard/-27558soft | 0hard/-27430soft |

An earlier, unverified attempt (3-minute budget, older base) saw +34% and +24% on the datagen input with the limits at 10000, so
the direction is consistent. The verified change is on the branch `perf-stay-unlimited` (one commit on top of `c505ced`, with its
test updates); it is **not** merged because of the keep rule above. Whether to take it is a decision for the owner: it is
behaviour-neutral for the score (a rule that cannot fire), it changes the registered constraint list only when no region has a
finite limit, and it is a large win on data like the real samples.

## Step 3: design section 11 items (d) to (f) and what to do next

| Item | What the profile says | Verdict |
|---|---|---|
| (d) cached `int` regions on fixed rows, arrays for the `FIXED_*` day maps | The `fixed*ByEmpDay` maps are 0.0% (both inputs have no Fixed rows, so this path is idle here; it is **not measured** on a history-heavy input). The string-keyed lookups that do cost are elsewhere: `HashMap.getNode` is 14 to 20% self, and `Calendars.isWorkingDay` (fab, region and customer off-day sets keyed by `String`) is its main project caller, 8.6 to 12.2% inclusive, plus `String.equals` 4.1% and the `TreeMap<Integer,String>` of the transit rule (9.4% / 4.3%) | justified in a **narrower form**: int fab/region codes and a per-fab working-day table (array lookup) |
| (e) fixed-presence calendar instead of pinned seats | `pinnedWorkDays` and `pinnedMiscRows` are 0.0%: both inputs have `assignment_list: []`, so there are no pinned seats to cost anything | **cannot be judged** from these inputs; measure on a history-heavy input first (not done: the heavy sample is excluded) |
| (f) packed `long` keys instead of `EmpDayRegion` / `EmpDayKey` | those two classes are 2.1% / 1.4% inclusive; GC pauses are 0.4% / 0.2% of the run | **not justified** |

Follow-up plan: yes, a short one, in this order (all behaviour-neutral, each checked against the goldens and measured with the A/B
scripts):

1. Take `perf-stay-unlimited` (verified above: +31% and +38% when the limits are 10000).
2. `Calendars.isWorkingDay`: a per-fab, per-day `boolean` table built once (or at least no three `String` lookups per call). The call sits
   inside `producedHoursForBlock`, which loops seats times days and is itself evaluated twice (filter and penalty, the same
   duplication that (b) removes for the region rules). `BlockConstraints.noUnderfillByBlock` / `overfillAtMostOneDayByBlock`
   could also read the `coveredWorkingDays` shadow instead of recounting. Together `isWorkingDay` and `producedHoursForBlock` are
   6 to 12% of the time each, so 10% or more is plausible.
3. `regionTransitGap`: rewrite the per-worker check on a sorted `int[]` of days with `int` region codes instead of building a
   `TreeMap<Integer,String>` per group update. (b) alone gained only 2.5% / 0.8% because the penalty half runs only for
   violating workers; the cost is in building the map (transit rule 19.0% / 8.5% inclusive, `TreeMap` 9.4% / 4.3%).
4. Try an explicit environment mode (`REPRODUCIBLE` or `NON_REPRODUCIBLE`) instead of the default `PHASE_ASSERT`. Not measured here.

Not worth a plan on this evidence: (c) value-range caching (0.0%), (f) packed keys, and (e) until a history-heavy profile exists.

## Caveats

- Two repetitions per input and change; the pair ratios of one input agree within about 3 points, but rounds differ much more in
  absolute speed. A 5 percent threshold on 2 pairs is the limit of what this setup can resolve.
- Both profiled inputs have no Fixed history (`assignment_list: []`); the profile says nothing about the heavy real sample.
- Final scores at 2 minutes are compared at the end of the budget only; `score_progress.txt` was not used (writing it slowed the
  search by about a third in an earlier attempt), and a faster run reaches a different trajectory.
