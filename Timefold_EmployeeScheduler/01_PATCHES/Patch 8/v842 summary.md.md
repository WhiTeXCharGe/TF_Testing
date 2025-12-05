# v842 summary.md

This document explains **everything** in the v842 incremental scheduler module:

- `pom.xml` (Maven module and how to run it)
    
- `EmployeeSchedule.java` (single-pass Timefold scheduler)
    
- `ExportSchedule.java` (how the solver output is written back to YAML)
    
- `IncrementalConfig.java` (all high-level knobs for incremental generation)
    
- `IncrementalSchedulerRunner.java` (incremental “daily_run + update_schedule2” driver in Java)
    

The goal is that you can explain to your boss/teammates:

- What the code **does**, step by step
    
- How it connects to `EnvConfig.yaml` / `Schedule.yaml`
    
- How Timefold is used (entities, solution, constraints, solver config)
    
- How the **incremental loop** grows modules and flips assignments Fixed/Flexible
    
- How to **build and run** everything with Maven
    

---

## 1. How to build and run

### 1.1 Build

From `apps/v842`:

`mvn -DskipTests clean package`

This:

- Compiles Java 17 sources
    
- Resolves Timefold + SnakeYAML
    
- Copies runtime dependencies into `target/dependency/` for optional fat-jar usage
    
    pom
    

### 1.2 Run the incremental driver

The `exec-maven-plugin` is configured to start **IncrementalSchedulerRunner** by default:

pom

`mvn -DskipTests exec:java`

This:

1. Finds the project root by walking up until it sees `pom.xml`.
    
    IncrementalSchedulerRunner
    
2. Resolves:
    
    - `ENV_PATH = "src/main/resource/EnvConfig.yaml"`
        
    - `SCHEDULE_IN_PATH / OUT_PATH = "src/main/resource/Schedule.yaml"`
        
        IncrementalConfig
        
3. Enters a **working-day loop**:
    
    - Grows workers to the configured size if needed
        
    - Updates `plan_flexibility` of assignments based on a cutoff date
        
    - Optionally creates new modules and extends `workflow_task_list`
        
    - Updates `schedule.plan_range`
        
    - Calls `EmployeeSchedule.solveFromYaml(...)` to re-solve the schedule
        
        IncrementalSchedulerRunner
        
    - Writes snapshots: `src/main/resource/schedule_outputs/Schedule_YYYYMMDD.yaml`
        
        IncrementalSchedulerRunner
        
4. Stops once:
    
    - `current` passes `plan_end`, or
        
    - the number of EQ modules reaches `EQ_NUM`
        
        IncrementalSchedulerRunner
        

---

## 2. Maven module (pom.xml)

The `pom.xml` for v842 is a child module under `eight-parent`:

pom

- **Group/artifact**: `com.yourorg:employee-scheduler-v842`
    
- **Java version**: 17 (`<maven.compiler.source/target>`)
    
- **Dependencies**:
    
    - `ai.timefold.solver:timefold-solver-core:1.27.0`
        
    - `org.yaml:snakeyaml:2.2`
        
        pom
        

Plugins:

- `maven-compiler-plugin` – compiles with Java 17
    
- `maven-dependency-plugin` – copies runtime jars into `target/dependency` at `package` phase
    
    pom
    
- `exec-maven-plugin` – runs `com.yourorg.scheduler.IncrementalSchedulerRunner` as the main entry point
    
    pom
    

---

## 3. Core scheduler: EmployeeSchedule.java

`EmployeeSchedule.java` is a **single-pass** Timefold model that:

- Parses `EnvConfig.yaml` and `Schedule.yaml`
    
- Builds a planning problem from **modules / phases / operations / workers**
    
- Runs Timefold to assign workers to seats (`CrewSeat`) within blocks (`BlockDecision`)
    
- Returns a plan and writes it back to `Schedule.yaml` via `ExportSchedule`.
    
    EmployeeSchedule
    

### 3.1 High-level imports and structure

- Timefold annotations, value ranges, constraints, solver API (`@PlanningEntity`, `@PlanningSolution`, `ConstraintProvider`, `SolverConfig`, etc.)
    
    EmployeeSchedule
    
- SnakeYAML for reading/writing YAML (`Yaml`, `DumperOptions`)
    
    EmployeeSchedule
    

The class includes:

1. **Domain classes**:
    
    - `EmployeeFact`, `DaySlot`, `BlockDecision`, `CrewSeat`, `SinglePassPlan` (planning solution)
        
2. **Calendars and helpers**:
    
    - `Calendars` structure + `isWorkingDay`, `workingDaysCount`, skill helpers, etc.
        
        EmployeeSchedule
        
        EmployeeSchedule
        
3. **YAML I/O**:
    
    - `loadYaml`, `saveYaml`, parsing EnvConfig/Schedule into maps
        
4. **Problem construction**:
    
    - Build operations, windows, required hours, fixed assignments
        
5. **Constraints**:
    
    - Hard/medium/soft rules implemented with Constraint Streams
        
6. **Solver pipeline**:
    
    - Build `SolverConfig`, create solver, run, then export via `ExportSchedule`.
        

### 3.2 Calendars and blackout logic

The `Calendars` class collects all “off” information: weekends, fab off dates, region off, customer off, worker company off, and personal off per worker; plus region stay rules and transit days.

EmployeeSchedule

EmployeeSchedule

- `isWorkingDay(dayId, fabId)` returns false if:
    
    - Day is in `CAL.weekends`, or
        
    - Fab off, or
        
    - Region off, or
        
    - Customer off.
        
        EmployeeSchedule
        
- `workingDaysCount(startDay, dayCount, fabId)` counts how many working days in that span, ignoring weekends and off days.
    
    EmployeeSchedule
    

These functions are reused for:

- Calculating **production** hours for blocks
    
- Enforcing **factory/calendar** constraints
    
- Auto-selecting hours per seat (`autoHours`).
    

### 3.3 Planning entities and solution

`SinglePassPlan` is annotated with `@PlanningSolution` and holds:

EmployeeSchedule

- Problem facts:
    
    - `List<DaySlot> days`
        
    - `List<EmployeeFact> employees`
        
- Planning entities:
    
    - `List<BlockDecision> blocks`
        
    - `List<CrewSeat> seats`
        
- `HardMediumSoftScore score`
    

`CrewSeat` defines a value range provider `eligibleEmployeesForSeat()`:

- If `pinned`, the range is only the pinned worker (or UNASSIGNED as fallback).
    
- If `needManager` is true, filter to managers and do **not** add `UNASSIGNED`.
    
- Otherwise allow the whole candidate list and, if empty, add `UNASSIGNED`.
    
    EmployeeSchedule
    

The actual planning variable is:

``@PlanningVariable(valueRangeProviderRefs = "eligibleEmployeesForSeat") public EmployeeFact employee; ``` :contentReference[oaicite:22]{index=22}    ### 3.4 Globals & helper methods  Global fields include: :contentReference[oaicite:23]{index=23}    - `DAILY_CAP = 12` – used by daily hour cap constraint   - `TARGET_HOURS_PER_EMP` – computed based on total workload / number of employees   - `OP_CAPACITY` – maximum heads per operation (based on skill>0)   - `OP_AVG_SKILL` – average skill per operation    Utility functions:  - `isUnassigned`, `skill(e, opId)`, `isManager`, `company(e)`, `avgSkill(opId)`    `autoHours(BlockDecision)` chooses a reasonable hours-per-day value from allowed hours:  - Computes working days `D` for that block and factory   - For each allowed `h`, evaluates how it covers required hours `R`   - Minimizes shortfall or excessive overage, then proximity to 8h. :contentReference[oaicite:24]{index=24}    ### 3.5 Reading EnvConfig + Schedule  From `EnvConfig.yaml`:  - Workflows, with phases and operations: used to construct `TaskWindow` objects (module+phase+op with date window, min/max heads, allowed hours). :contentReference[oaicite:25]{index=25}   - Worker companies, workers, skills, managers, fab suitability maps, unavailable dates, fab/customer/region calendars, transit day maps, region stay rules. :contentReference[oaicite:26]{index=26}    From `Schedule.yaml`:  - `plan_range.start_date` / `end_date` – the window the solver considers   - `workflow_task_list` – modules with `phase_task_list`, each having operation tasks and `workload_days` used to compute required hours and windows. :contentReference[oaicite:27]{index=27}   - `assignment_list` – fixed/flexible assignments that become `FixedAssign` rows for respecting pinned/fixed work. :contentReference[oaicite:28]{index=28}    The parser:  - Converts dates to `LocalDate` and `dayId` offsets from `planStart`. :contentReference[oaicite:29]{index=29}   - Computes required hours per `(module | op)` and builds windows (`TaskWindow`).   - Tracks latest fixed end dates per module/phase to respect when adding new flexible seats. :contentReference[oaicite:30]{index=30}    ### 3.6 Constraints (overview)  Constraints are implemented in a nested **SinglePassConstraints** style class (see v832 summary for full shape; the same categories apply). Broadly, there are:  - **Block-level hard rules**   - Enough production to cover required hours   - Overfill at most one extra day’s worth   - Daily head capacity per operation   - **Seat-level hard rules**   - Worker available all days (respect personal off & blackout calendars)   - Pinned seats keep the pinned worker   - One factory per employee per day   - Daily cap: total hours per employee per day ≤ 12h   - **Route / travel rules**   - Transit gap between fabs/regions (using `transitDays`)   - Region stay max_on + off-interval constraints   - **Soft rules**   - Balance workload per worker   - Prefer higher skill where possible   - Penalize cross-company mixing if needed    The final score is `HardMediumSoftScore`, and the solver is configured with a termination condition (unimproved time or spent time) via `SolverConfig` + `TerminationConfig`. :contentReference[oaicite:31]{index=31}    ### 3.7 Public API  `EmployeeSchedule` exposes a public static entry:  ```java public static void solveFromYaml(String envPath, String schedPath)``

which:

1. Loads `EnvConfig.yaml` and `Schedule.yaml`
    
2. Builds the `SinglePassPlan`
    
3. Runs Timefold
    
4. Writes the merged assignments back via `ExportSchedule`.
    
    IncrementalSchedulerRunner
    

This is what `IncrementalSchedulerRunner` calls.

---

## 4. ExportSchedule.java

`ExportSchedule` is responsible for turning a solved `SinglePassPlan` into:

- An updated `Schedule.yaml`'s `assignment_list` where:
    
    - Existing fixed rows are preserved
        
    - New flexible rows are appended for seats decided by the solver.
        
        ExportSchedule
        

Key points:

1. **Task ID mapping**
    
    Builds a map `opTaskId` from `(module|op)` to the operation task ID in `workflow_task_list`. Seats refer to `(module, opId)`, so this mapping is used to know which `operation_task` string to write.
    
    ExportSchedule
    
2. **Fixed mask**
    
    Reads existing `assignment_list` and builds `fixedMask` per `(worker, task)` with already-occupied day indices. New flexible work never overwrites those dates.
    
    ExportSchedule
    
3. **Collecting flexible work**
    
    For each non-pinned seat:
    
    - Look up its block to get `startDay`, `days`, `hours`.
        
    - Use `isWorkingDay` to skip weekends/blackouts.
        
        ExportSchedule
        
    - Sum hours per `(dayId)` into `byDay`.
        
    - Remove days that are already fixed for this `(worker, task)` using `fixedMask`.
        
        v832 summary.md
        
4. **Writing YAML rows**
    
    For each `(worker, task)`:
    
    - Sort `byDay` and create:
        
        - `start_date` = first day
            
        - `end_date` = last day
            
        - `work_date_list` with `{date, hour}`
            
        - `plan_flexibility = "Flexible"`
            
            v832 summary.md
            
    
    Append all such rows to `newFlex`, then:
    
    - `assignment_list = fixedRows + newFlex`
        
    - Dump YAML with block style using `DumperOptions`.
        
        v832 summary.md
        

---

## 5. IncrementalConfig.java

`IncrementalConfig` centralizes parameters for the incremental simulation.

IncrementalConfig

### 5.1 Worker and module parameters

- `WORKER_NUM = 400` – target total number of workers (auto-generated if missing).
    
- `EQ_PER_DAYS = 2.5` – average new modules per day.
    
- `EQ_PER_DAYS_SIGMA = 2.5` – daily variance.
    
- `EQ_NUM = 100` – total target number of EQ modules.
    
    IncrementalConfig
    

### 5.2 Skill and manager distributions

- `SKILL_LEVEL_LIST = [1, 2, 3, 4, 5]`
    
- `SKILL_LEVEL_WEIGHTS = [0.05, 0.1, 0.5, 0.25, 0.1]` (most workers around level 3)
    
- `MANAGER_RATE = 0.5` – 50% of workers become managers on average.
    
    IncrementalConfig
    

### 5.3 Calendars & seeds

- `IS_SKIP_WEEKEND = false` – weekends are not automatically holidays; `isHoliday` just consults this flag plus `isWeekend`.
    
    IncrementalConfig
    
- `PLAN_RANGE_EXTRA_DAYS = 3` – when extending `plan_range.end_date`, the end is the max of existing module ends plus 3 days.
    
    IncrementalConfig
    
- `ENV_SEED = 100` – RNG seed for worker generation.
    
- `MODULE_SEED = 200` – base seed for module generation; each evaluation adds an offset.
    
    IncrementalConfig
    

### 5.4 Evaluation cadence and workloads

- `EQ_EVAL_DAYS = 1` – evaluate every working day.
    
    IncrementalConfig
    

Work length patterns: **normal** vs **VIP**. Both are lists of phases; each phase entry is:

``[ phase_total_days, [workload_days_per_operation...] ] ``` :contentReference[oaicite:47]{index=47}    - `NORMAL_WORKLENGTH` – e.g. phase 1: 15 days, operations 30/20/20 days of workload (converted to hours later).   - `VIP_WORKLENGTH` – slightly more compressed total days (e.g. 12,10,6,8). :contentReference[oaicite:48]{index=48}    ### 5.5 File paths  Paths are relative to the **project root**:  - `ENV_PATH        = "src/main/resource/EnvConfig.yaml"`   - `SCHEDULE_IN_PATH`, `SCHEDULE_OUT_PATH` both point to `"src/main/resource/Schedule.yaml"` (in-place overwrite). :contentReference[oaicite:49]{index=49}    ---  ## 6. IncrementalSchedulerRunner.java  `IncrementalSchedulerRunner` is a Java re-implementation of the “daily_run + update_schedule2” script, driving the solver incrementally over simulated days. :contentReference[oaicite:50]{index=50}    High-level responsibilities:  1. Loop over working days from the start of plan.   2. For each evaluation day:    - Grow workers if necessary      - Flip assignments to Fixed/Flexible according to a cutoff      - Possibly add new modules (EQs)      - Extend `plan_range`      - Write `EnvConfig.yaml` + `Schedule.yaml` back      - Call `EmployeeSchedule.solveFromYaml(...)`      - Save a snapshot of the schedule. :contentReference[oaicite:51]{index=51}    ### 6.1 Basic date helpers and YAML I/O  - `parseDate(Object)` – parses `"yyyy/MM/dd"` strings, also accepts `"-"` and fixes `"-"` to `"/"`. :contentReference[oaicite:52]{index=52}   - `isWeekend`, `isHoliday`, `nextWorkingDay`, `advanceWorkingDays` – helpers to step across weekends based on `IS_SKIP_WEEKEND`. :contentReference[oaicite:53]{index=53}    YAML:  - `loadYaml(Path)` – returns `Map<String,Object>` using SnakeYAML `SafeConstructor + LoaderOptions`. If file absent, returns empty map. :contentReference[oaicite:54]{index=54}   - `saveYaml(Path, Map)` – writes YAML in block form using `DumperOptions`. :contentReference[oaicite:55]{index=55}   - `backupFile(Path)` – copies `X` to `X.bak` before overwriting. :contentReference[oaicite:56]{index=56}    ### 6.2 Worker generation (extendWorkersIfNeeded)  `extendWorkersIfNeeded(envRoot)`:  - Reads `environment.worker_list`.   - If current size `< WORKER_NUM`, generates additional workers: :contentReference[oaicite:57]{index=57}     - Collects all operation IDs from the first workflow’s phases and `operation_list`. :contentReference[oaicite:58]{index=58}     - Uses `worker_company_list` as potential companies. :contentReference[oaicite:59]{index=59}     - For each new worker:     - ID: `"w{index+1}"`     - Name: double letters (AA, AB, AC, ...) via `idxToWorkerName`.     - Random number of skills between 3 and 6.     - Random subset of operation IDs as skills, sorted.     - Skill levels drawn from `SKILL_LEVEL_LIST` with `SKILL_LEVEL_WEIGHTS`. :contentReference[oaicite:60]{index=60}       - Random company from `worker_company_list`.     - Manager flag via `weightedChoiceBool(MANAGER_RATE)`.     - Empty `fab_suitability_map` and `unavailable_dates`. :contentReference[oaicite:61]{index=61}    Returns `[beforeCount, addedCount]`, and updates `environment.worker_list` in place.  ### 6.3 Module collection and IDs  `collectExistingModules(schedRoot)`:  - Looks at `schedule.workflow_task_list` and filters modules whose IDs look like `"e{number}"`.   - For each module:   - Computes earliest phase start and latest phase end from `phase_task_list`.   - Returns:   - Sorted module list   - Last module start date   - Last module end date. :contentReference[oaicite:62]{index=62}    `buildOneModule(...)`:  - Creates a module `"e{index+1}"` with a name like `"SU 1001A"`.   - For each phase:   - Uses worklength tuple `[phaseDays, opWorkloadDays]` from `NORMAL/VIP_WORKLENGTH`. :contentReference[oaicite:63]{index=63}     - Calls `buildPhaseTask` to build phase task + per-operation tasks, with `start_date`, `end_date`, `workload_days`.   - Tracks `__END_DATE` for the whole module. :contentReference[oaicite:64]{index=64}    `createNewModules(...)`:  - Chooses for each new module whether it is **normal** or **VIP** based on weights `[0.8, 0.2]`. :contentReference[oaicite:65]{index=65}   - Chooses a fab from available `fab_list` IDs.   - Uses `MODULE_SEED + moduleSeedOffset` as RNG seed so each evaluation day is reproducible. :contentReference[oaicite:66]{index=66}   - Returns the new module maps and updates `lastEndHolder[0]`. :contentReference[oaicite:67]{index=67}    ### 6.4 Assignment update (Fixed vs Flexible)  `updateAssignments(schedRoot, cutoff)`:  - Reads `schedule.assignment_list`. :contentReference[oaicite:68]{index=68}   - For each assignment:   - Computes start date from `start_date` or first entry of `work_date_list` (or misspelled `work_date_lsit`). :contentReference[oaicite:69]{index=69}     - If `start_date < cutoff` → `plan_flexibility = "Fixed"`.     - Else → `plan_flexibility = "Flexible"`.   - Returns `[totalAssignments, changedToFixed]`. :contentReference[oaicite:70]{index=70}    This mimics the “freeze history, keep future flexible” behavior.  ### 6.5 Main loop  The `main(String[] args)` performs:  1. **Locate files**     ```java    Path projectRoot = findProjectRoot(Paths.get("").toAbsolutePath());    Path envPath    = projectRoot.resolve(ENV_PATH);    Path schedPath  = projectRoot.resolve(SCHEDULE_IN_PATH);    Path outDir     = projectRoot.resolve("src/main/resource/schedule_outputs");    Files.createDirectories(outDir);    ``` :contentReference[oaicite:71]{index=71}    2. **Initial plan_range**     - Load `Schedule.yaml`, get `schedule.plan_range.start_date` / `end_date`.      - Compute `planStart`, `planEnd`.      - Collect initial modules to find last start (`lastStart0`).      - `cutoff0 = nextWorkingDay(lastStart0)` if modules exist, else `planStart`.      - `current = advanceWorkingDays(cutoff0, EQ_EVAL_DAYS - 1)`.    3. **Loop over days**     For each `current`:     - Skip if after `planEnd`.      - If weekend, skip and `current++`.      - Reload fresh `EnvConfig` and `Schedule` so every iterate sees the latest YAML.      - Collect existing modules (`modulesBefore`, `lastStartBefore`, `lastEndBefore`).      - Compute `cutoff = nextWorkingDay(lastStartBefore)` or `planStart` if no modules. :contentReference[oaicite:72]{index=72}       **Evaluation index 0 (first run)**     - No new modules are added; just set `plan_range.start_date` if missing.      - Save backups and YAML.      - Call `runSolver(...)` to get the initial baseline schedule.      - Copy `Schedule.yaml` to snapshot `Schedule_YYYYMMDD.yaml`.      - Move `current` by `EQ_EVAL_DAYS` and increment `evalIndex`. :contentReference[oaicite:73]{index=73}       **Subsequent evaluations**     - Compute whether today is allowed to extend modules:      - If there are no assignments and `current == cutoff`, do not extend today; otherwise allow.      - Calculate how many modules to add (`modulesToday`):      - Build a daily demand using Gaussian with `EQ_PER_DAYS` and `EQ_PER_DAYS_SIGMA`, scaled by `EQ_EVAL_DAYS`.        - Convert fractional demand to an integer with random rounding.        - Respect `remaining = EQ_NUM - currentN`. :contentReference[oaicite:74]{index=74}       - If `modulesToday > 0`, call `createNewModules(...)`, append them to `schedule.workflow_task_list`, and update `newLastEnd`. :contentReference[oaicite:75]{index=75}       - Update `plan_range`:      - End candidates = previous `plan_range.end_date`, `lastEndBefore`, `newLastEnd`.        - `endBase` = max of these or `cutoff` if empty.        - `endFinal = endBase + PLAN_RANGE_EXTRA_DAYS`.        - Write back `plan_range.end_date`. :contentReference[oaicite:76]{index=76}       - Save `EnvConfig.yaml` and `Schedule.yaml` with backups. :contentReference[oaicite:77]{index=77}       - Print stats:      - Total assignments, changed fixed        - Workers before/after        - Modules existing/added, target        - New `plan_range`. :contentReference[oaicite:78]{index=78}    4. **Run solver or skip**     - There is a small compatibility hack: checks `assignm_list` (typo) length; if zero, treat as no assignments. :contentReference[oaicite:79]{index=79}      - `runSolver = current.equals(planStart) || modulesAdded > 0 || assignmLen == 0`.      - If `runSolver`:      - Call `EmployeeSchedule.solveFromYaml(envPath, schedPath)` with absolute paths. :contentReference[oaicite:80]{index=80}        - Snapshot `Schedule_YYYYMMDD.yaml`. :contentReference[oaicite:81]{index=81}      - Else:      - Log that nothing changed and skip solver.  5. **Termination**     - If `afterCount >= EQ_NUM`, stop with message. :contentReference[oaicite:82]{index=82}      - Else, move `current = advanceWorkingDays(current, EQ_EVAL_DAYS)` and loop. :contentReference[oaicite:83]{index=83}       When finished, prints `"[DONE] Daily run finished (Java)."` :contentReference[oaicite:84]{index=84}    ---``