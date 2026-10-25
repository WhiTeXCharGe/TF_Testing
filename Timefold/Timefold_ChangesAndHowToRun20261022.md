# Timefold (web/Timefold): what changed and how to run this version

Date: 2026-10-22. Branch of the code: `Timefold` in https://github.com/WhiTeXCharGe/TF_Testing.git (code in the repository root; these documents are in the branch `documents`).
Design: `Timefold_TripList_Design20261021.md`. Plan: `Timefold_ImplementationPlan20261021.md`. Teammate explainer: `Timefold_HowItWorks_Explained20261021.md`.

## 1. In one paragraph

`web/Timefold` is a refactored copy of `TFDocker`. The solver now **knows about trips** (fly-in / fly-out, FI/FO) through a new
`trip_list` in `Schedule.yaml`, and it **writes a correct `trip_list` into the result**. FI/FO are not part of the score: Fixed
trips close days (like calendars do), stay limits count FI to FO, and after solving the output trips are derived from the final tasks.
It runs on Timefold Solver 2.7.1 and Java 21, the one 4,000-line Java file is split into packages, and all solver settings can be
given from outside (command line, environment, or an optional `SolverParams.yaml`).

## 2. What changed

### 2.1 For the user of the files

| Area | Before (`TFDocker`) | Now |
|---|---|---|
| Trips | Timefold knew nothing about FI/FO | new optional key `trip_list` in `Schedule.yaml`, right after `assignment_list`; result file has it too |
| `assignment_list` | had to contain work | may be `[]` (first run); the solver plans everything and prints the trips |
| Fixed trip | - | FI day and FO day are closed to every task; interior days are only for that trip's region. It is a calendar mark, not a task rule; tasks inside a Fixed trip are not fixed automatically |
| Flexible trip | - | ignored by the solver, regenerated in the output |
| FI / FO in output | - | FI = first task day - 1, FO = last task day + 1 (weekends too). Region change: FO of A and FI of B may be the same day. A Fixed trip is copied unchanged |
| Empty Fixed trip | - | option `allow_cut_fixed_trip` (default true): a Fixed trip with no same-region work inside is removed from the output and logged |
| Stay limit / annual stay | counted work days | count FI to FO (a stay without Fixed trip = `(last - first + 1) + 2` days). Travel gap (`stay_off_interval`, transit days) does **not** look at FI/FO |
| Misc tasks | no region | a misc task uses its own `region`; `misc_neutral_regions` (default `r_other`) means "no region" |
| Personal `unavailable_dates` | - | unchanged; when it falls on an FI/FO day the trip mark wins, no extra penalty, no warning |
| Settings | constants in code | parameters: stage times, unimproved limits, stage 2 on/off, seed, trip validation, history days, ... (table in section 4) |
| Extra output | - | `trip_warnings.txt` (only when there is something to report), `solver_version` at the end of `schedule` |
| Input check | - | a missing, `null` or malformed `trip_list` never crashes: lenient = warnings, `trip_validation: strict` = fail with `InvalidInputData` |

`trip_list` format (per worker; the flat list with a `worker` key per row is accepted too):

    trip_list:
      - worker: w1
        trips:
          - { start_date: 2026/09/01, end_date: 2026/09/10, region: r1, plan_flexibility: Fixed, description: "visa trip" }

Rules: `start_date == end_date` inside one trip is invalid (equal dates occur only between two different trips: the end of the
first equals the start of the next). Details and examples are in the design document, sections 3 to 6.

### 2.2 For the developer

* Timefold Solver 1.27.0 to 2.7.1 (through 1.35.0), Java 17 to 21. The 2.x range check for initial values needed a small wrapper
  (`model/ValidationTolerantRange`); all scoring goldens are identical to the 1.27 baseline.
* Packages `com.yourorg.scheduler.{model,calendar,io,build,constraints,solver,export,config,trip}` and `Main` (was one file).
* Constraint ids and scores are unchanged by the refactor (characterisation tests with goldens: tiny, small, datagen; real sample opt-in).
  The only intended scoring change: the two stay constraints count FI to FO (small golden regenerated for that, datagen/real unchanged
  because their limits are 10000).
* Tests: 159 passing, 1 skipped (the opt-in real sample). New: parser, calendar marks, stay limits, TripDeriver (property test),
  export, end-to-end (empty `assignment_list`, params file, bad params).
* `analyze` is Enterprise only, so score breakdowns use `solver/ConstraintBreakdown` (internal API, isolated in that class).
* Docker: the container keeps `/work/input`, `/work/output`, `/work/status/<RUN_ID>.json`; `SolverParams.yaml` in the input folder
  is passed automatically; a solver crash now really writes status `Failed` (a bug in the entrypoint was fixed).

## 3. How to run

Everything below runs in Docker; no local Java or Maven is needed. In Git Bash on Windows prefix commands with `MSYS_NO_PATHCONV=1`
and use `$(pwd -W)` instead of `$(pwd)`. Run from the repository root.

**Tests (about 3 to 5 minutes):**

    docker run --rm -v "$(pwd):/w" -v tf-m2:/root/.m2 -w /w maven:3.9-eclipse-temurin-21 mvn -B test

**Build the image:**

    docker build -t timefold-scheduler:local .

**Quick check run (about 10 seconds, first run with empty `assignment_list`, one Fixed and one Flexible trip):**

    mkdir -p work/input work/output work/status
    cp src/test/resources/fixtures/firstrun/*.yaml work/input/
    printf 'stage1_unimproved_seconds: 5\nstage2_enabled: false\n' > work/input/SolverParams.yaml
    docker run --rm -e RUN_ID=firstrun -v "$(pwd)/work/input:/work/input:ro" -v "$(pwd)/work/output:/work/output" -v "$(pwd)/work/status:/work/status" timefold-scheduler:local

Check `work/status/firstrun.json` (`status: Completed`) and `work/output/result_Schedule.yaml` (`trip_list` right after `assignment_list`).

**Your own data:** put `EnvConfig.yaml`, `Schedule.yaml` (and optionally `SolverParams.yaml`) in `work/input` and run the same `docker run`
without the `printf` line. For a test run use the small files in `logic/data_generator` first (80 workers; `stage1_unimproved_seconds: 10` and
`stage2_enabled: false` finished in about 1 minute); do not start with the big `decoder7_out` sample.

## 4. Parameters (all optional)

Source order, first wins: command line `--key=value`, environment `TF_KEY`, `SolverParams.yaml`, default.

| Key | Default | Meaning |
|---|---|---|
| `stage1_minutes` | 300 | stage 1 time limit |
| `stage1_unimproved_seconds` | 600 | stop stage 1 after this long without improvement |
| `stage1_stop_when_feasible` | true | stop stage 1 when the hard score reaches 0 |
| `stage2_enabled` | true | false skips the polish stage |
| `stage2_minutes` / `stage2_unimproved_seconds` | 60 / 60 | stage 2 limits |
| `random_seed` | none | fixed seed for repeatable runs |
| `trip_validation` | lenient | `strict` fails the run on a bad trip |
| `allow_cut_fixed_trip` | true | remove empty Fixed trips from the output |
| `trip_history_days` | -1 | drop derived trips ending more than N days before the plan start (-1 keeps all) |
| `misc_neutral_regions` | `[r_other]` | misc regions meaning "no region" |
| `write_score_files` | true | write `*_score.txt` files |

Example for a short check: `stage1_minutes: 2`, `stage1_unimproved_seconds: 30`, `stage2_enabled: false`, `random_seed: 42`.
SchedulerWeb will fill this file later from a settings screen (not built yet).

## 5. Known limits and what is not done yet

* **Blocks past the plan end (older behaviour, not changed):** the exporter writes a block's whole length even if it runs beyond the
  plan end, and calendars are not applied past the plan end. A derived trip can therefore end after the plan end (seen once: 2027/02/02
  for a plan ending 01/24). Suggested follow-up: clamp the export to `[plan start, plan end]` or extend the calendars. Not done because it changes the assignment output.
* **Closed days in Fixed trips:** a measurement on the datagen sample found about 7% of trip-worker seats still covering a closed day
  with the calendar-mark approach (no extra constraint added on purpose; re-measure on a real time-limited run).
* **Short runs give infeasible plans**, so `trip_warnings.txt` may show `region_conflict_day` and `tight_transit`: expected for 5 to 10 second budgets.
* **Not part of this work (later):** the decoder (SU_Others to `Schedule.yaml`), `fix_edited_modules.py`, GanttChartEditor, SchedulerWeb
  and the API controller (settings screen and passing `SolverParams.yaml`) still need their own changes.
* **Speed (Task 19) was not finished:** no performance change is in this version and nothing was measured. A first draft
  (register the stay rules only when a region has a finite limit) was started but not verified; it is kept only in the local
  branch `perf-wip` and is not pushed. The other ideas (compute day-to-region maps once, build value ranges once, cached `int`
  regions on fixed rows, a fixed-presence calendar instead of pinned seats, packed keys) are not implemented. Each needs a
  measured before/after run with a fixed seed.
* The real sample (`decoder7_out`, about 194k lines) was deliberately not run, and the long (300 minute) default budget was not run end to end.

## 6. Where things are (code repository)

`README.md` (build, run, parameters, trip rules), `docs/perf.md`, `src/main/java/com/yourorg/scheduler/*`, tests in `src/test/java`,
fixtures in `src/test/resources/fixtures/{tiny,small,datagen,firstrun}`, goldens in `src/test/resources/golden`.
