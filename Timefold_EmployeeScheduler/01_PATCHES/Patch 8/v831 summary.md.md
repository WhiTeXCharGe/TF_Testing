# v831 summarize.md

This document explains **everything** in the v8.3.1 **two-pass scheduler**:

- `EmployeeSchedule.java`
    
    - all domain classes (Pass 1 & Pass 2)
        
    - calendars, parsing, constraints, solver pipeline
        
- `ExportSchedule.java`
    
    - how we write **Pass2** assignments back into `Schedule.yaml`
        
- `pom.xml`
    
    - Java version, dependencies, exec plugin
        

Compared to v8.3.2:

- **v8.3.1 = full two-pass design**
    
    - **Pass 1**: decide **block shapes** (when / how many days / how many heads / auto hours)
        
    - **Pass 2**: decide **who sits in each seat on each day**
        
- **v8.3.2 = single-pass design**
    
    - Only one planning solution (blocks+seats together).
        

This doc is for walking your boss/teammates through:

- What the code **does**, step by step
    
- How it connects to **EnvConfig.yaml / Schedule.yaml**
    
- How the **two passes** work and why
    
- How calendars/unavailable/fixed rows are handled
    
- How the build & run pipeline works
    

---

## 1. How to build and run this module

`pom.xml` configures `exec-maven-plugin` with:

- `mainClass = com.yourorg.scheduler.EmployeeSchedule`
    

So the typical workflow (same as 8.3.2) is:

1. **Build without tests**
    

`mvn -DskipTests clean package`

- `clean` removes old compiled classes & jars
    
- `package` compiles and builds the jar under `target/`
    
- `-DskipTests` makes it faster
    

2. **Run the solver**
    

`mvn -q exec:java -D"exec.args=src\main\resource\EnvConfig.yaml src\main\resource\Schedule.yaml"`

- `exec.args[0]` → `EnvConfig.yaml`
    
- `exec.args[1]` → `Schedule.yaml`
    
- Main method:
    
    1. calls `solveFromYaml(envPath, schedPath)`
        
    2. gets `RunResult` with final `Pass2Plan` and `planStart`
        
    3. calls `ExportSchedule.overwriteScheduleWithAssignments(...)` to rewrite `Schedule.yaml`
        

**Whole pipeline:**

1. Read `EnvConfig.yaml`, `Schedule.yaml`
    
2. Parse workflow, workers, skills, unavailable dates, plan_range
    
3. Build task windows and **fixed assignments**
    
4. Build **fixed head count per day/op** (for Pass 1)
    
5. **Pass 1**: block-level solver (block size, window, heads, auto hours)
    
6. Expand blocks → **flexible seats + seat-days**
    
7. Expand fixed assignments → **pinned seats + seat-days**
    
8. Merge pinned+flex seats and build **candidate employee lists** per seat
    
9. **Pass 2**: seat-level solver (who sits where each day)
    
10. Export **flexible** rows back into `Schedule.yaml`, keep fixed rows
    

---

## 2. Maven module (`pom.xml`)

### 2.1 Coordinates

- `groupId`: `com.yourorg`
    
- `artifactId`: `employee-scheduler` (this is the v8.3.1 module)
    

In your multi-module project, this module is one scheduler variant.

### 2.2 Java and library versions

`<maven.compiler.source>17</maven.compiler.source> <maven.compiler.target>17</maven.compiler.target> <timefold.version>1.27.0</timefold.version> <snakeyaml.version>2.2</snakeyaml.version>`

- Compiles with **Java 17**
    
- Uses **Timefold 1.27.0**
    
- Uses **SnakeYAML 2.2** for YAML I/O
    

### 2.3 Dependencies

`<dependency>   <groupId>ai.timefold.solver</groupId>   <artifactId>timefold-solver-core</artifactId> </dependency> <dependency>   <groupId>org.yaml</groupId>   <artifactId>snakeyaml</artifactId> </dependency>`

Just Timefold + SnakeYAML; all other things are standard JDK.

### 2.4 Build & exec plugins

- **maven-compiler-plugin** → ensures Java 17
    
- **maven-dependency-plugin** → copies runtime deps into `target/dependency` (optional if you run outside Maven)
    
- **exec-maven-plugin** → `mainClass = com.yourorg.scheduler.EmployeeSchedule`
    

---

## 3. High-level code architecture (v8.3.1)

Everything lives in `EmployeeSchedule.java`:

- **YAML tools** (`loadYaml`, `saveYaml`, `safeStr`, `parseInt`, `dayIdFromDate`)
    
- **Domain model** for both passes:
    
    - common facts: `DaySlot`, `EmployeeFact`, `TaskWindow`, `FixedAssign`, `FixedHeadDay`
        
    - **Pass 1**: `BlockDecision` entity, `Pass1Plan` solution
        
    - **Pass 2**: `CrewSeat` entity, `SeatDay` fact, `Pass2Plan` solution
        
- **Calendars**:
    
    - weekend, fab off, region off, customer off, worker off
        
    - region transit days, region stay limits
        
- **Global stats**:
    
    - `OP_CAPACITY` (how many workers can do each op), `OP_AVG_SKILL`
        
- **Parsing**:
    
    - `parseEnv(...)` → `ParsedEnv`
        
    - `parseSchedule(...)` → `ParsedSchedule` + `FixedAssign`s
        
    - `buildFixedHeadDays(...)` → `FixedHeadDay` list for Pass 1
        
- **Pass 1 utilities**:
    
    - `produced(b)` using `autoHours(b)`
        
    - `Pass1Constraints` (block-level constraints)
        
    - `solvePass1HoursRamp(...)`
        
- **Expansion to seats**:
    
    - `Expanded` helper
        
    - `expandToSeats(...)` (from blocks)
        
    - `expandPinnedSeats(...)` (from fixed assignments)
        
- **Pass 2 utilities**:
    
    - `buildPinnedEmpDayFactory(...)`
        
    - `fillSeatCandidateEmployees(...)`
        
    - `Pass2Constraints` (seat-level constraints)
        
    - `solvePass2Once(...)`
        
- **Public API**:
    
    - `RunResult`
        
    - `solveFromYaml(...)` → orchestrates both passes
        
    - `main(...)`
        

---

## 4. Common domain classes and helpers

### 4.1 Dates and utilities

- `DF = DateTimeFormatter.ofPattern("yyyy/MM/dd")` for all YAML dates
    
- `safeStr(Object o)` → `""` if null, otherwise `String.valueOf(o)`
    
- `parseInt(Object o, int def)` → robust int parse with default
    
- `phaseNumFromId("p3")` → `3`, used to enforce phase order
    

`dayIdFromDate(planStart, "yyyy/MM/dd")`:

- converts a real date into **day index** `0,1,2,...` relative to `planStart`
    

There is also a `CLOCK` formatter and `nowClock()` + `fmt(Duration)` helper to log solver timings.

### 4.2 DaySlot

`public static class DaySlot {     @PlanningId public int id;     public LocalDate date; }`

- Represents one day in the plan horizon
    
- `id` is 0-based index; `date` is actual calendar date
    

### 4.3 EmployeeFact

`public static class EmployeeFact {     @PlanningId public int id;   // 0 = UNASSIGNED     public String wid;          // worker ID from EnvConfig     public String name;     public Map<String,Integer> skills = new HashMap<>(); // opId -> level     public boolean isManager;     public String workerCompany; }`

Special:

- `id == 0` is reserved for the **UNASSIGNED** ghost worker
    
- `skills.get(opId)` ≥ 1 means the worker can work that operation
    

Helpers:

- `static boolean isUnassigned(EmployeeFact e)`
    
- `static int skill(EmployeeFact e, String opId)`
    
- `static boolean isManager(EmployeeFact e)`
    
- `static String company(EmployeeFact e)`
    

### 4.4 TaskWindow

`public static class TaskWindow {     public String module;     public String factory;     public String phaseId;     public int    phaseNum;     public String opId;     public int startDayId;     public int endDayId;     public List<Integer> allowed;   // allowed hours per day (e.g. [8, 10, 12])     public int minHeads;     public int maxHeads;     public int workloadDays;        // from Schedule.yaml }`

Represents one `(module, phase, operation)` window from Schedule:

- `[startDayId .. endDayId]` is where this task can be scheduled
    
- `workloadDays * baselineHours` = total baseline required hours before fixed subtractions
    

### 4.5 FixedAssign and FixedHeadDay

`static class FixedAssign {     String module, opId, factory, wid;     int startDayId, endDayId;     Map<Integer,Integer> hoursByDay = new HashMap<>(); // dayIdx -> hours     String phaseId; int phaseNum; }  class FixedHeadDay {     public int dayId;     public String opId;     public int heads; }`

- `FixedAssign` captures **each fixed row** from `assignment_list` in `Schedule.yaml`
    
    - which module/op/factory/worker
        
    - which days and how many hours on each day
        
- `buildFixedHeadDays(List<FixedAssign>)`:
    
    - For each fixed row and each `(dayIdx, hours>0)`
        
    - Count it as **1 head** on `(dayId, opId)`
        
    - Aggregate into `FixedHeadDay` list
        

These `FixedHeadDay` objects are used in **Pass 1** to:

- ensure total heads (fixed + flexible blocks) do not exceed `OP_CAPACITY`
    
- make Pass 1 aware of already-fixed head counts
    

---

## 5. Calendars and unavailable dates

`static class Calendars` holds everything related to calendar logic:

- `Set<Integer> weekends`
    
- `Map<String,Set<Integer>> fabOff`
    
- `Map<String,Set<Integer>> regionOff`
    
- `Map<String,Set<Integer>> customerOff`
    
- `Map<String,Set<Integer>> workerCompanyOff`
    
- `Map<String,String> fabToRegion`
    
- `Map<String,String> fabToCustomer`
    
- `Map<String,Set<Integer>> workerOffByWid` (personal off)
    
- `Map<String,Map<String,Integer>> transitDays`
    
    - `fromRegion -> (toRegion -> requiredGapDays)`
        
- `Map<String,Integer> regionStayMaxOn`
    
- `Map<String,Integer> regionStayOffInterval`
    

Helpers:

- `int transitDays(String from, String to)`
    
- `String regionOfFab(String fabId)`
    
- `int maxStayOn(String regionId)`
    
- `int stayOffInterval(String regionId)`
    

`buildCalendars(envPath, planStart, planEnd)`:

1. Marks **weekends** (SAT/SUN) as non-working in `weekends`
    
2. Reads `environment.fab_list`:
    
    - maps fab → region & customer
        
    - builds per-fab unavailable day sets
        
3. Reads `region_list`, `customer_company_list`, `worker_company_list`:
    
    - builds **region**, **customer**, **worker-company** off days
        
4. Reads `worker_list`:
    
    - per-worker unavailable days → `workerOffByWid`
        
5. Reads `transite_day_map`:
    
    - builds region-to-region transit gap map
        
6. Reads `max_stay_on` + `stay_off_interval` from `region_list`:
    
    - used to limit continuous stay in a region
        

`isWorkingDay(dayId, fabId)`:

- returns `true` only if:
    
    - not weekend
        
    - day not in `fabOff[fabId]`
        
    - day not in `regionOff[regionOfFab(fabId)]`
        
    - day not in `customerOff[customerOfFab(fabId)]`
        

`workingDaysCount(startDay, dayCount, fabId)`:

- counts working days inside `[startDay .. startDay+dayCount-1]` obeying all off rules
    

Pass 2 also uses helper `maxSegmentSpanWithBreak(dayList, offInterval)`:

- given a list of days when an employee works in a region
    
- and required off interval (like “1 day off every N days”)
    
- computes **longest continuous “on” segment**
    
- used by `regionStayMaxOn` constraint
    

---

## 6. Parsing EnvConfig and Schedule

### 6.1 ParsedEnv and OpDef

`static class OpDef {     String phaseId;     int phaseNum;     List<Integer> allowed;     int min;     int max; }  static class ParsedEnv {     Map<String,OpDef> opdef;     List<EmployeeFact> employees;     Map<String,EmployeeFact> byWid; }`

`parseEnv(envPath)`:

1. Load root YAML and locate `environment`
    
2. From `workflow_list.phase_list.operation_list`:
    
    - For each operation:
        
        - `operation` → opId (`p1o1` etc.)
            
        - `work_hours` → allowed hours list (e.g. `[8,10,12]` or `[4]`)
            
        - `min_worker_num` / `max_worker_num`
            
        - Phase id → `phaseId`, `phaseNum`
            
3. Build `employees` list:
    
    - Add UNASSIGNED worker with `id=0`
        
    - For each worker in `worker_list`:
        
        - id, name, is_manager, worker_company
            
        - `skill_map` → `skills` map
            
    - Put each `EmployeeFact` in `byWid` map
        
4. Build globals:
    
    - `OP_CAPACITY[opId]` = count of workers with `skill>0` on that op
        
    - `OP_AVG_SKILL[opId]` = average skill (used in soft skill balance)
        

### 6.2 ParsedSchedule and FixedAssign

`static class ParsedSchedule {     LocalDate planStart;     LocalDate planEnd;     List<DaySlot> daySlots;     List<TaskWindow> windows;     Map<String,Integer> requiredByKey;   // module|opId -> baseline required hours     List<FixedAssign> fixedRows;     Map<String,Integer> fixedHoursByKey; // module|opId -> sum of fixed hours }`

`parseSchedule(schedPath, opdef)`:

1. Read `schedule.plan_range.start_date` and `end_date` → `planStart`, `planEnd`
    
2. Create `DaySlot` list for the horizon
    
3. For each `workflow_task` (module) and `phase_task`:
    
    - build `TaskWindow`:
        
        - `module`, `factory`, `phaseId`, `phaseNum`, `opId`
            
        - `startDayId` & `endDayId` from phase’s start/end
            
        - `allowed` from `OpDef`
            
        - `minHeads`, `maxHeads` from `OpDef`
            
        - `workloadDays` from `operation_task`
            
    - Compute **baseline required hours** per `(module, opId)` and fill `requiredByKey`
        
4. Read `assignment_list`:
    
    - for each row:
        
        - `operation_task` ID (like `e16p4o1`), `worker`, `plan_flexibility`
            
        - `work_date_list` (or typo `work_date_lsit`) → each has `date` + `hour`
            
        - convert each date to dayIdx; store into `FixedAssign.hoursByDay`
            
    - Only rows with `plan_flexibility == "fixed"` become `FixedAssign`
        
    - For those:
        
        - accumulate hours into `fixedHoursByKey[module|opId]`
            
        - store into `fixedRows`
            
5. **Windows vs fixed**:
    
    v8.3.1 does **not** aggressively push phase windows based on fixed rows like the single-pass version; Pass 1 + `FixedHeadDay` will handle conflicts through constraints instead.
    
6. `buildFixedHeadDays(fixedRows)`:
    
    - For each `FixedAssign fa` and each `(dayIdx, hours>0)`:
        
        - increment `heads` for key `(dayId, opId)`
            
    - Returns list of `FixedHeadDay(dayId, opId, heads)`
        

---

## 7. Globals used by both passes

`static final int DAILY_CAP = 12; static double TARGET_HOURS_PER_EMP = 0.0;  static final Map<String,Integer> OP_CAPACITY = new HashMap<>(); static final Map<String,Double>  OP_AVG_SKILL = new HashMap<>();`

- `DAILY_CAP` = 12 hours per employee per day (hard limit)
    
- `TARGET_HOURS_PER_EMP` computed in `solveFromYaml`:
    
    - `totalRequiredHours / realEmployees` (excluding UNASSIGNED)
        
- `OP_CAPACITY` and `OP_AVG_SKILL` precomputed in `parseEnv`
    

---

## 8. Pass 1 – block-level planning (hours ramp)

### 8.1 BlockDecision entity

`class BlockDecision {     @PlanningId public int id;      public String module;     public String factory;     public String phaseId;     public int    phaseNum;     public String opId;     public int windowStart;     public int windowEnd;     public int requiredHours;     public List<Integer> allowed;     public int minHeads;     public int maxHeads;      @PlanningVariable(valueRangeProviderRefs = "vrDayIds")     public Integer startDay;      @PlanningVariable(valueRangeProviderRefs = "vrHeadOptions")     public Integer heads;      @PlanningVariable(valueRangeProviderRefs = "vrDayCountOptions")     public Integer days;      public int seedHours = 8; }`

In Pass 1, each block decision chooses:

- **startDay** (when the operation starts)
    
- **heads** (how many workers assigned simultaneously in that block)
    
- **days** (how many days the block runs, continuous)
    

Hours per day are **not** a direct planning variable; they are computed by `autoHours(b)`.

### 8.2 Pass1Plan solution

`@PlanningSolution public static class Pass1Plan {     @ValueRangeProvider(id = "vrDayIds")     @ProblemFactCollectionProperty     public List<Integer> dayIds;      @ValueRangeProvider(id = "vrHeadOptions")     @ProblemFactCollectionProperty     public List<Integer> headOptions;      @ValueRangeProvider(id = "vrDayCountOptions")     @ProblemFactCollectionProperty     public List<Integer> dayCountOptions;      @ProblemFactCollectionProperty     public List<DaySlot> daySlots;      @ProblemFactCollectionProperty     public List<FixedHeadDay> fixedHeadDays;      @PlanningEntityCollectionProperty     public List<BlockDecision> blocks;      @PlanningScore     private HardMediumSoftScore score; }`

Value ranges are **global** lists:

- `dayIds` → possible `startDay` values
    
- `headOptions` → `minHeads..maxHeads` found across all windows
    
- `dayCountOptions` → `[1 .. maxWindowLength]`
    

`fixedHeadDays` feed into capacity constraint with fixed rows included.

### 8.3 produced() and autoHours()

`produce(b)` is essentially:

`int produced(BlockDecision b) {     int H = (b.heads == null ? 0 : b.heads);     int D = workingDaysCount(b.startDay, b.days, b.factory);     int h = autoHours(b);     return H * h * D; }`

`autoHours(b)`:

- Sorts allowed hours list (e.g. `[8,10,12]`)
    
- Uses `heads`, `factory`, `days` to compute `D` working days
    
- Tries to find **allowed h** such that:
    
    - `prod = H * h * D` ≥ `requiredHours`
        
    - `over = prod - requiredHours` ≤ one day’s worth (`H * h`)
        
- Among feasible `h`, prefers:
    
    - smallest `|h-8|`, then smallest `h`
        
- If no strictly “good” `h`, it compares all allowed hours using a small multi-criteria tuple:
    
    - whether underfill/overfill
        
    - amount of underfill/overfill
        
    - closeness to 8 hours
        

So Pass 1 **implicitly chooses hours per day** by this function without adding another planning variable.

### 8.4 Pass1Constraints – hard part

`Pass1Constraints` defines block-level rules. Main hard constraints:

1. `withinWindow`
    
    - startDay must be within `[windowStart .. windowEnd]`
        
2. `daysWithinWindowLen`
    
    - `startDay + days - 1` must not exceed `windowEnd`
        
3. `hoursValueAllowed`
    
    - `autoHours(b)` must be within `allowed` list
        
4. `headsInMinMax`
    
    - `heads ∈ [minHeads .. maxHeads]`
        
5. `noUnderfill`
    
    - `produced(b) >= requiredHours`
        
    - penalizes by the missing hours
        
6. `overfillAtMostOneDay`
    
    - overproduction `produced(b) - requiredHours` may not exceed one block-day worth (`H*h`)
        
7. `phaseOrder`
    
    - For each module, phase N must fully finish before phase N+1 starts
        
    - Uses phaseNum from `phaseNumFromId("p3")`
        
8. `dailyHeadCapacityByOp` (Pass 1 version)
    
    - For each `(day, opId)` summarises **flex block heads** and joins with `FixedHeadDay` to know `fixed heads` as well
        
    - Ensures: `(flexHeads + fixedHeads) <= OP_CAPACITY[opId]`
        
9. `penalizeStackByOp`
    
    - For each `(day, opId)` counts how many blocks (even with same op) stack
        
    - Too many stacking blocks is penalized (to encourage spread)
        

### 8.5 Pass1Constraints – soft part

Soft weights:

- `PREF_HOURS_WEIGHT`, `SMALLER_HOURS_W`, `SMALLER_HEADS_W`, `FEWER_DAYS_W`, `EARLIER_START_W`, `STACK_PAIR_WEIGHT`
    

Main soft constraints:

- **preferHoursNear8**
    
    - penalizes `|autoHours(b) - 8|`
        
- **preferSmallerHours**
    
    - penalizes `autoHours(b)` itself (smaller is better)
        
- **preferSmallerHeads**
    
    - penalizes `heads` size (fewer parallel heads is slightly nicer)
        
- **preferFewerDays**
    
    - penalizes longer duration (prefers compact blocks)
        
- **preferEarlierStart**
    
    - penalizes later `startDay` (within the window)
        
- **penalizeStackByOp** (soft component)
    
    - additional penalty for heavily stacked ops on same day
        

The idea: Pass 1 finds block configurations that:

- are hard-feasible with capacities & windows
    
- try to use smaller hours, fewer heads, fewer days, earlier start
    
- but still achieve required hours
    

### 8.6 `solvePass1HoursRamp(...)` – the ramp logic

Pass 1 uses a **“ramp” over allowed hours**:

1. Compute:
    
    - global `headOptions`, `dayIds`, `dayCountOptions`
        
    - `maxChoices` = largest `allowed.size()` among windows
        
2. For tier = 1..`maxChoices`:
    
    - For each `TaskWindow w`:
        
        - `allowedSorted` = sorted allowed hours
            
        - `tiered = allowedSorted.subList(0, tier)`
            
            - tier 1 → only smallest hour (e.g. 8)
                
            - tier 2 → [8,10], etc.
                
        - compute baseline = `4` if only `[4]` else `8`
            
        - `totalReq = workloadDays * baseline`
            
        - subtract `fixedHoursByKey` to get `requiredHours` for flexible part
            
        - create `BlockDecision`:
            
            - `windowStart`, `windowEnd`, `requiredHours`, `allowed=tiered`
                
            - initial guess: `startDay = windowStart`, `heads = minHeads`, `seedHours = tiered[0]`
                
            - compute seed `days` roughly = `ceil(requiredHours / (seedHours*minHeads))`, clamped to window length
                
3. Build `Pass1Plan` for this tier and solve with:
    
    - `bestScoreLimit = "0hard/*medium/*soft"`
        
    - `spentMinutes = 30`
        
    - `unimprovedSeconds = 60`
        
4. If the result has **hardScore = 0**:
    
    - run a **polish** pass:
        
        - no bestScoreLimit
            
        - 20 minutes, unimproved 60 seconds
            
    - return this as `Pass1Result`
        
5. If no tier achieves hardScore=0:
    
    - choose the **best** tier result among those tried (prefers hardZero if any)
        
    - return it anyway (with `hardIsZero=false`)
        

So Pass 1 is pretty robust: it tries smaller hour choices first, then opens up more options if necessary.

---

## 9. Expanding blocks and fixed rows to seats

### 9.1 Expanded helper

`class Expanded {     List<CrewSeat> seats;     List<SeatDay> seatDays; }`

- A `CrewSeat` = “one seat in a block or fixed assignment”
    
- A `SeatDay` = “that seat on one particular day with some hours”
    

### 9.2 CrewSeat and SeatDay (Pass 2 domain)

`public static class CrewSeat {     @PlanningId public int id;      public String module;     public String factory;     public String phaseId;     public int    phaseNum;     public String opId;      public int startDayId;     public int days;     public int hours;     public int seatIndex;     public String seatKey;     public int blockId;      public boolean pinned = false;     public String  pinnedWid = null;      private List<EmployeeFact> candidateEmployees = List.of();     @PlanningVariable(valueRangeProviderRefs = "eligibleEmployeesForSeat")     public EmployeeFact employee; }`

- `seatKey` is a unique string (module|op|seatIndex|dStart or PIN) used to link `SeatDay`s
    
- For pinned seats, several fields are set from `FixedAssign`
    

`public static class SeatDay {     @PlanningId public String id;   // seatKey + "/d" + dayId     public String seatKey;     public DaySlot day;     public int hours;     public String factory; }`

### 9.3 expandToSeats (from Pass 1 blocks)

`expandToSeats(blocks, daySlots)`:

- For each `BlockDecision b`:
    
    - compute `hours = autoHours(b)`
        
    - `start = (b.startDay != null ? b.startDay : b.windowStart)`
        
    - `dcount = (b.days != null ? b.days : 1)`
        
    - `headCount = (b.heads != null ? b.heads : 1)`
        
- For each seat index `sidx = 0 .. headCount-1`:
    
    - `seatKey = module + "|" + opId + "|s" + 4-digit index + "|d" + start`
        
    - create `CrewSeat`:
        
        - copy module/factory/phase/op
            
        - `startDayId = start`, `days = dcount`, `hours = autoHours(b)`
            
        - `blockId = b.id`, `pinned=false`, no employee assigned yet
            
- For each day offset `off = 0 .. dcount-1`:
    
    - `did = start + off`
        
    - if `isWorkingDay(did, factory)`:
        
        - create `SeatDay(seatKey, DaySlot(did), hours, factory)`
            

Result:

- `Expanded.seats` = all **flexible** seats from Pass 1
    
- `Expanded.seatDays` = all those seats split per day
    

### 9.4 expandPinnedSeats (from FixedAssign)

`expandPinnedSeats(sch, env, windows, days)`:

- First build maps:
    
    - `moduleToFactory` from `TaskWindow`
        
    - `moduleOpToPhase`, `moduleOpToPhaseNum` from `TaskWindow`
        
- For each `FixedAssign fa` in `sch.fixedRows`:
    
    - `factory` from `moduleToFactory`
        
    - `seatKey = module + "|" + opId + "|PIN|" + wid + "|d" + startDayId`
        
    - Build `CrewSeat cs`:
        
        - `id = sid++`
            
        - `module`, `factory`, `phaseId`, `phaseNum`, `opId`
            
        - `startDayId = fa.startDayId`
            
        - compute `minDid` / `maxDid` from `hoursByDay` keys
            
        - `days = maxDid - minDid + 1`
            
        - `hours = max(hoursByDay.values())` (largest hours on any day in that assignment)
            
        - `seatIndex = 0`
            
        - `seatKey` = above
            
        - `blockId = -1`
            
        - `pinned = true`, `pinnedWid = fa.wid`
            
        - `employee` = `env.byWid.get(fa.wid)` (or UNASSIGNED fallback)
            
    - For each `(dayIdx, hours)` in `fa.hoursByDay`:
        
        - find `DaySlot dd` by id
            
        - add `new SeatDay(seatKey, dd, hrs, factory)`
            

Result:

- `Expanded.exPinned.seats` = all seats representing original **fixed rows**
    
- `Expanded.exPinned.seatDays` = their day-by-day breakdown
    

### 9.5 Combine for Pass 2

In `solveFromYaml`:

`Expanded exFlex   = expandToSeats(p1.blocks, sch.daySlots); Expanded exPinned = expandPinnedSeats(sch, env, sch.windows, sch.daySlots);  List<CrewSeat> allSeats = new ArrayList<>(); allSeats.addAll(exPinned.seats); allSeats.addAll(exFlex.seats);  List<SeatDay> allSeatDays = new ArrayList<>(); allSeatDays.addAll(exPinned.seatDays); allSeatDays.addAll(exFlex.seatDays);`

Then Pass 2 uses:

- **allSeats** (both flexible and pinned)
    
- **allSeatDays** (all daily assignments)
    

---

## 10. Pass 2 – seat-level planning (who works where)

### 10.1 Pass2Plan

`@PlanningSolution public static class Pass2Plan {     @ProblemFactCollectionProperty     public List<DaySlot> days;      @ProblemFactCollectionProperty     public List<EmployeeFact> employees;      @PlanningEntityCollectionProperty     public List<CrewSeat> seats;      @ProblemFactCollectionProperty     public List<SeatDay> seatDays;      @PlanningScore     private HardMediumSoftScore score; }`

In Pass 2:

- `CrewSeat` is the only **planning entity**
    
- Individual **employees** are chosen from a **per-seat value range** built from candidate lists
    

### 10.2 Pinned mapping and candidate lists

`buildPinnedEmpDayFactory(pinnedSeats, seatDays)`:

- For each `SeatDay` and its seat:
    
    - if seat is pinned and has a real employee:
        
        - store `out[empId][dayId] = factory`
            

Used later to **exclude** candidates that would clash with pinned seats in another factory on the same day.

`fillSeatCandidateEmployees(allSeats, allSeatDays, employees)`:

1. Split into `pinned` and `flex` seat lists
    
2. Build:
    
    - `seatDaysByKey[seatKey]` → list of dayIds a seat covers
        
    - `seatFactoryByKey[seatKey]` → its factory
        
3. Build `pinnedEmpDayFactory` map as above
    

For each **flex** seat:

- gather `seatDayIds` (days it covers)
    
- gather `seatFactory`
    

Then for each employee `e` (excluding UNASSIGNED):

1. Skill filter: `skill(e, cs.opId) >= 1`
    
2. Personal off filter:
    
    - if any `seatDayId` is in `workerOffByWid[e.wid]` → exclude
        
3. Pinned cross-factory filter:
    
    - if `pinnedEmpDayFactory[e.id]` has the same day but **other factory** → exclude
        

If all pass, add to `candidates` list.

Finally:

`cs.setCandidateEmployees(candidates);`

**Important**: There is no automatic UNASSIGNED fallback here; candidate lists can be empty. Hard constraints will push solver to assign valid employees.

### 10.3 CrewSeat value range

Inside `CrewSeat`:

`@ValueRangeProvider(id = "eligibleEmployeesForSeat") public CountableValueRange<EmployeeFact> eligibleEmployeesForSeat() {     List<EmployeeFact> base = (candidateEmployees == null) ? List.of() : candidateEmployees;     if (pinned && pinnedWid != null) {         for (EmployeeFact e : base) if (e != null && pinnedWid.equals(e.wid)) return new ListValueRange<>(List.of(e));         return new ListValueRange<>(List.of());     }     return new ListValueRange<>(base); }`

- For **pinned seats**, only the pinned worker is allowed in the range
    
- For **flex seats**, the range is `candidateEmployees` computed earlier
    

### 10.4 Pass2Constraints – hard rules

`Pass2Constraints` uses `SeatDay`, `CrewSeat`, `CAL`, `OP_CAPACITY`, etc.

Key hard constraints:

1. `assignedAndSkill`
    
    - penalizes if seat is unassigned or skill `< 1`
        
    - ensures each seat gets a skilled worker
        
2. `oneFactoryPerEmpDay`
    
    - groups by `(employee, day)` and counts distinct factories (using SeatDay + CrewSeat)
        
    - penalizes if worker works in more than one factory on same day
        
3. `dailyCap12h`
    
    - groups by `(employee, day)` and sums `SeatDay.hours`
        
    - penalizes if total > `DAILY_CAP` (12)
        
4. `employeeAvailableOnSeatDays`
    
    - uses `workerOffByWid` and `SeatDay.day.id`
        
    - penalizes if worker is assigned on a personal off day for that seat
        
5. `respectPinnedAssignments`
    
    - if `cs.pinned == true`, then `cs.employee` must match `pinnedWid`
        
6. `regionTransitGap`
    
    - for each pair of seat-days (sd1, sd2) for same employee where `sd2.day.id > sd1.day.id`:
        
        - compare `regionOfFab(factory1)` and `regionOfFab(factory2)`
            
        - required gap = `CAL.transitDays(r1, r2)`
            
        - if `deltaDays <= requiredGap` → violation
            
7. `regionStayMaxOn`
    
    - group by `(employeeId, region)`
        
    - compute `maxSpan = maxSegmentSpanWithBreak(dayList, stayOffInterval(region))`
        
    - if `maxSpan > maxStayOn(region)` → violation
        
8. (via OP_CAPACITY) **daily head capacity** is handled in Pass 1+FixedHeadDay; Pass 2 relies on that result and does not recheck capacity by op.
    

### 10.5 Pass2Constraints – soft rules

Soft weights:

- `COMPANY_PAIR_W`, `SKILL_DIVERSITY_W`, `SKILL_AVG_W`, etc.
    

Important soft constraints:

1. `softSameCompanyPairs`
    
    - for each `(SeatDay, CrewSeat)` pair on same block & day:
        
        - counts how many worker pairs are from same `workerCompany`
            
    - rewarded/penalized to encourage grouping by company
        
2. `softEncourageSkillVariety`
    
    - for each block, encourages some variety in skill levels among crew
        
    - if all skills are identical, penalty applied
        
3. `softBalanceBlockAvgSkill`
    
    - for each block and opId:
        
        - compute average skill of assigned workers
            
        - penalize difference from `OP_AVG_SKILL[opId]`
            
    - tries to keep block skill near global average
        
4. `softBalanceTotalHours`
    
    - group by employee across **all SeatDays**
        
    - sum total assigned hours
        
    - penalize `|totalHours - TARGET_HOURS_PER_EMP|`
        
    - encourages fair workload distribution across workers
        

---

## 11. Solver builders and execution

### 11.1 Generic solver builder

`static <S> Solver<S> buildSolver(Class<S> solutionClass,                                  Class<?>[] entityClasses,                                  Class<? extends ConstraintProvider> providerClass,                                  String bestScoreLimit,                                  Integer spentMinutes,                                  Integer unimprovedSeconds) {     SolverConfig cfg = new SolverConfig();     cfg.withSolutionClass(solutionClass);     cfg.withEntityClasses(entityClasses);     cfg.withScoreDirectorFactory(new ScoreDirectorFactoryConfig()         .withConstraintProviderClass(providerClass));      TerminationConfig term = new TerminationConfig();     if (bestScoreLimit != null) term.setBestScoreLimit(bestScoreLimit);     if (spentMinutes != null && spentMinutes > 0)         term.setSpentLimit(Duration.ofMinutes(spentMinutes));     if (unimprovedSeconds != null && unimprovedSeconds > 0)         term.setUnimprovedSpentLimit(Duration.ofSeconds(unimprovedSeconds));     cfg.withTerminationConfig(term);      return SolverFactory.<S>create(cfg).buildSolver(); }`

- No XML config, everything is Java-based
    
- `bestScoreLimit` often `"0hard/*medium/*soft"`
    
- Time limits keep solver from running forever
    

Helper: `hardZero(Score s)` just checks if string starts with `"0hard"`.

### 11.2 Pass 1 solving

`Pass1Result solvePass1HoursRamp(...)` as described above:

- tries multiple hour tiers
    
- for each tier, run solver with 30 min and unimproved 60 seconds
    
- if any tier is hard-feasible, polish that tier’s result with 20 min extra
    
- returns `Pass1Result` with blocks, score, and `tierUsed`
    

### 11.3 Pass 2 solving

`static Pass2Plan solvePass2Once(List<DaySlot> days,                                 List<EmployeeFact> employees,                                 List<CrewSeat> seats,                                 List<SeatDay> seatDays) {      fillSeatCandidateEmployees(seats, seatDays, employees);      Pass2Plan p2 = new Pass2Plan();     p2.days = days; p2.employees = employees;     p2.seats = seats; p2.seatDays = seatDays;      Solver<Pass2Plan> solver = buildSolver(         Pass2Plan.class,         new Class<?>[]{ CrewSeat.class },         Pass2Constraints.class,         "0hard/*medium/*soft",         30, 60     );     Pass2Plan result = solver.solve(p2);      if (hardZero(result.getScore())) {         Solver<Pass2Plan> polish = buildSolver(             Pass2Plan.class,             new Class<?>[]{ CrewSeat.class },             Pass2Constraints.class,             null, 20, 60         );         result = polish.solve(result);     }     return result; }`

- First run tries to reach `0hard` within 30 min, 60s unimproved limit
    
- If success (hardZero), run second pass purely for soft score improvement
    

---

## 12. Public API – putting both passes together

`public static class RunResult {     public Pass2Plan finalPlan;     public LocalDate planStart; }`

`solveFromYaml(envPath, schedPath)`:

1. Parse:
    
    - `ParsedEnv env = parseEnv(envPath)`
        
    - `ParsedSchedule sch = parseSchedule(schedPath, env.opdef)`
        
2. Build calendars:
    
    - `buildCalendars(envPath, sch.planStart, sch.planEnd)`
        
3. Compute `TARGET_HOURS_PER_EMP`:
    
    - `realEmp = env.employees.size() - 1` (exclude UNASSIGNED)
        
    - `totalReq = sum(requiredByKey)` (baseline)
        
    - `TARGET_HOURS_PER_EMP = totalReq / realEmp`
        
4. Build `fixedHeadDays`:
    
    - `List<FixedHeadDay> fixedHeadDays = buildFixedHeadDays(sch.fixedRows);`
        
5. **Pass 1**:
    
    - `Pass1Result p1 = solvePass1HoursRamp(sch.daySlots, sch.windows, sch.fixedHoursByKey, fixedHeadDays);`
        
6. Expand seats (flex + pinned):
    
    - `Expanded exFlex = expandToSeats(p1.blocks, sch.daySlots);`
        
    - `Expanded exPinned = expandPinnedSeats(sch, env, sch.windows, sch.daySlots);`
        
    - Combine to `allSeats` and `allSeatDays`
        
7. **Pass 2**:
    
    - `Pass2Plan finalP2 = solvePass2Once(sch.daySlots, env.employees, allSeats, allSeatDays);`
        
8. Return:
    
    `RunResult rr = new RunResult(); rr.finalPlan = finalP2; rr.planStart = sch.planStart; return rr;`
    

`main(args)`:

- reads env/sched paths
    
- calls `solveFromYaml`
    
- calls `ExportSchedule.overwriteScheduleWithAssignments(rr.finalPlan, rr.planStart, schedPath, envPath);`
    
- prints “Done.”
    

---

## 13. ExportSchedule.java – writing back to Schedule.yaml

`ExportSchedule.overwriteScheduleWithAssignments(Pass2Plan finalPass2, LocalDate planStart, String schedPath, String envPath)`

### 13.1 Load and root

- Load YAML from `schedPath`
    
- `sched = root.get("schedule")` (or `root` fallback)
    

### 13.2 Ensure calendars

- If `EmployeeSchedule.CAL` is not initialized yet:
    
    - read `plan_range.end_date` from `sched`
        
    - call `EmployeeSchedule.buildCalendars(envPath, planStart, planEnd)`
        

### 13.3 Map `(module, op)` → `operation_task` ID

- Walk `workflow_task_list.phase_task_list.operation_task_list`:
    
    - `opTaskId[module + "|" + operation] = operation_task.id`
        

Used to write `operation_task` field in `assignment_list`.

### 13.4 Seat metadata and hours aggregation

Build:

- `seatMeta[seatKey] = { module, opId, factory }`
    
- `empBySeat[seatKey] = EmployeeFact`
    
- `seatPinned[seatKey] = boolean`
    

Then process all `SeatDay sd` in `finalPass2.seatDays`:

- if `seatPinned[sd.seatKey]` is true → **skip** (fixed rows stay as they are)
    
- get `EmployeeFact e = empBySeat[sd.seatKey]`
    
- if `e == null` or `id == 0` → skip (unassigned)
    
- from `seatMeta` get `module`, `opId`, `factory`
    

Aggregate into `perKey`:

- key = `[wid, dayIdx, module, opId]`
    
- value = total `hours` for that combination (sum up if multiple seatDays contribute)
    

After the loop:

- For each `(wid, dayIdx, module, opId)`:
    
    - `byDay[dayIdx] = totalHours`
        
- Then for each `(wid, module, opId)` group:
    
    - sort `dayIdx` list
        
    - `firstIdx` = min day, `lastIdx` = max day
        
    - `start_date` = `planStart + firstIdx`
        
    - `end_date` = `planStart + lastIdx`
        
    - build `work_date_list` entries for **only the days that actually have hours** (gaps allowed)
        

### 13.5 Preserve fixed rows and merge

- Read `assignment_list` from original `sched`
    
- Build `preservedFixed` as all rows with `plan_flexibility == "fixed"`
    
- New flexible rows = all aggregated rows from Pass2:
    
    `Map<String,Object> a = new LinkedHashMap<>(); a.put("worker", wid); a.put("operation_task", taskId);    // from opTaskId[module|opId] a.put("start_date", startD.format(DF)); a.put("end_date",   endD.format(DF)); a.put("work_date_list", work); a.put("plan_flexibility", "Flexible");`
    
- Finally:
    
    `List<Map<String,Object>> merged = new ArrayList<>(); merged.addAll(preservedFixed); merged.addAll(newFlex); sched.put("assignment_list", merged);`
    

### 13.6 Save YAML

- Uses `DumperOptions` with block style and pretty flow
    
- Writes back to `schedPath`
    
- Prints summary: new flexible rows count, preserved fixed rows count
    

---

## 14. How to explain v8.3.1 to boss/teammates

You can summarise like this:

1. **Inputs / Outputs**
    
    - Input: `EnvConfig.yaml` + `Schedule.yaml`
        
    - Output: same `Schedule.yaml` but with **new flexible assignments** appended, and original **fixed rows preserved**
        
2. **Two passes**
    
    - **Pass 1 (Block planner)**:
        
        - Does not care about _who_ works, only _how many heads_ / _how many days_ / _when_
            
        - Respects windows, phase order, capacities (including fixed heads), min/max heads, underfill/overfill rules
            
        - Chooses hours per day via `autoHours` and an hours-ramp strategy
            
    - **Pass 2 (Seat planner)**:
        
        - Takes block output from Pass 1 and exact fixed rows
            
        - Expands into per-seat per-day structure
            
        - Assigns real employees to seats, respecting:
            
            - skills, personal off, region transit gap, region stay limits
                
            - one factory per day, 12h/day cap
                
            - pinned workers from original fixed rows
                
        - Balances hours and skill distributions and encourages crew from same company
            
3. **Calendars**
    
    - Weekends and fab/region/customer/company/worker off days come from `EnvConfig.yaml`
        
    - Region transit days and max stay per region avoid unrealistic travel patterns
        
4. **Fixed vs flexible**
    
    - Fixed rows become **pinned seats** plus **FixedHeadDay** capacities
        
    - Their hours are subtracted before computing flexible `requiredHours`
        
    - They are **never overwritten** in Export; we add only flexible rows
        
5. **Timefold configuration**
    
    - All configuration is in Java code (no XML)
        
    - Two separate planning solutions: `Pass1Plan` and `Pass2Plan`
        
    - Each solved with an initial pass (`0hard/*medium/*soft`) and an optional polish stage for soft score




# v8.3.1 まとめ (v831 summarize.md 日本語版)

このドキュメントは、v8.3.1 の **2パス版スケジューラ** について **すべて** 説明します。

- `EmployeeSchedule.java`
    
    - （パス1・パス2 両方の）ドメインクラス一式
        
    - カレンダー、パース処理、制約、ソルバーのパイプライン
        
- `ExportSchedule.java`
    
    - **パス2** のアサイン結果を `Schedule.yaml` に書き戻す仕組み
        
- `pom.xml`
    
    - Javaバージョン、依存ライブラリ、exec プラグイン設定
        

v8.3.2 との違いは以下の通りです：

- **v8.3.1 = 完全な 2 パス設計**
    
    - **パス1**：各ブロックの形を決める（いつ / 何日間 / 何人 / 何時間にするかを auto hours で決定）
        
    - **パス2**：各シート・各日ごとに「誰を座らせるか」を決定
        
- **v8.3.2 = 1 パス設計**
    
    - ブロックと座席を **1つのプランニングソリューション** の中で一緒に解く
        

このドキュメントは、上司・チームメンバーに次の内容を説明できるようにするためのものです。

- コードが **何をしているか**（ステップごと）
    
- どのように **EnvConfig.yaml / Schedule.yaml** とつながっているか
    
- **2パス構成** がどう動くか、なぜそうしているか
    
- カレンダー／非稼働日／固定行をどう扱っているか
    
- ビルド & 実行のパイプラインがどうなっているか
    

---

## 1. モジュールのビルドと実行方法

`pom.xml` では `exec-maven-plugin` が以下のように設定されています。

- `mainClass = com.yourorg.scheduler.EmployeeSchedule`
    

そのため、基本的なワークフロー（8.3.2 と同じ）は：

1. **テストなしでビルド**
    

`mvn -DskipTests clean package`

- `clean`：古い class や jar を削除
    
- `package`：コードをコンパイルし、`target/` 配下に jar を作成
    
- `-DskipTests`：ユニットテストをスキップして高速化
    

2. **ソルバーを実行**
    

`mvn -q exec:java -D"exec.args=src\main\resource\EnvConfig.yaml src\main\resource\Schedule.yaml"`

- `exec.args[0]` → `EnvConfig.yaml`
    
- `exec.args[1]` → `Schedule.yaml`
    
- `main` メソッドの中では：
    
    1. `solveFromYaml(envPath, schedPath)` を呼び出す
        
    2. 最終的な `Pass2Plan` と `planStart` を含む `RunResult` を受け取る
        
    3. `ExportSchedule.overwriteScheduleWithAssignments(...)` を呼び出し、`Schedule.yaml` を上書きする
        

**全体のパイプライン：**

1. `EnvConfig.yaml`, `Schedule.yaml` を読み込む
    
2. ワークフロー、作業者、スキル、非稼働日、plan_range をパース
    
3. タスクウィンドウと **固定アサインメント** を構築
    
4. パス1用に、各日・各オペレーションの **固定ヘッド数 (fixed head count)** を作る
    
5. **パス1**：ブロックレベルソルバー  
    （ブロック開始日・日数・人数・auto hours による時間を決定）
    
6. ブロックを展開して → **柔軟な座席(CrewSeat) + SeatDay** を生成
    
7. 固定アサインメントを展開して → **ピン留め座席 + SeatDay** を生成
    
8. ピン留め・柔軟な座席をマージし、各座席ごとの **候補従業員リスト** を作る
    
9. **パス2**：座席レベルソルバー（誰をどの日にどの席に座らせるか）
    
10. `Schedule.yaml` に **Flexible 行** を追記し、Fixed 行はそのまま残す
    

---

## 2. Maven モジュール（`pom.xml`）

### 2.1 座標

- `groupId`: `com.yourorg`
    
- `artifactId`: `employee-scheduler` （これが v8.3.1 のモジュール）
    

マルチモジュールプロジェクトの中で、このモジュールが 1 つのスケジューラ実装になっています。

### 2.2 Java とライブラリのバージョン

`<maven.compiler.source>17</maven.compiler.source> <maven.compiler.target>17</maven.compiler.target> <timefold.version>1.27.0</timefold.version> <snakeyaml.version>2.2</snakeyaml.version>`

- コンパイルは **Java 17**
    
- **Timefold 1.27.0** を使用
    
- YAML I/O 用に **SnakeYAML 2.2** を使用
    

### 2.3 依存関係

`<dependency> <groupId>ai.timefold.solver</groupId> <artifactId>timefold-solver-core</artifactId> </dependency> <dependency> <groupId>org.yaml</groupId> <artifactId>snakeyaml</artifactId> </dependency>`

Timefold と SnakeYAML だけで、他は JDK 標準ライブラリです。

### 2.4 Build & exec プラグイン

- **maven-compiler-plugin** → Java 17 コンパイルを保証
    
- **maven-dependency-plugin** → 実行時依存を `target/dependency` にコピー（Maven 外で実行したい場合に便利）
    
- **exec-maven-plugin** → `mainClass = com.yourorg.scheduler.EmployeeSchedule`
    

---

## 3. 高レベルアーキテクチャ（v8.3.1）

コードはすべて `EmployeeSchedule.java` にまとまっています。

- **YAML ツール**  
    (`loadYaml`, `saveYaml`, `safeStr`, `parseInt`, `dayIdFromDate` など)
    
- **2パス共通のドメインモデル**：
    
    - 共通 Fact：`DaySlot`, `EmployeeFact`, `TaskWindow`, `FixedAssign`, `FixedHeadDay`
        
    - **パス1**：`BlockDecision`（エンティティ）、`Pass1Plan`（ソリューション）
        
    - **パス2**：`CrewSeat`（エンティティ）、`SeatDay`（Fact）、`Pass2Plan`（ソリューション）
        
- **カレンダー処理**：
    
    - 週末、fab 休み、region 休み、customer 休み、worker/company 休み
        
    - 地域間移動日数、地域滞在制限
        
- **グローバル統計**：
    
    - `OP_CAPACITY`（各オペレーションを実行できる人数）
        
    - `OP_AVG_SKILL`（オペレーションごとの平均スキル）
        
- **パース処理**：
    
    - `parseEnv(...)` → `ParsedEnv`
        
    - `parseSchedule(...)` → `ParsedSchedule` + `FixedAssign` 一覧
        
    - `buildFixedHeadDays(...)` → パス1用の `FixedHeadDay` 一覧
        
- **パス1ユーティリティ**：
    
    - `produced(b)` と `autoHours(b)`（ブロックで生産される時間計算）
        
    - `Pass1Constraints`（ブロックレベル制約）
        
    - `solvePass1HoursRamp(...)`（段階的に許可時間を増やす「ランプ」ロジック）
        
- **座席への展開**：
    
    - `Expanded` ヘルパー
        
    - `expandToSeats(...)`（ブロック → Flex 座席＋SeatDay）
        
    - `expandPinnedSeats(...)`（固定行 → Pinned 座席＋SeatDay）
        
- **パス2ユーティリティ**：
    
    - `buildPinnedEmpDayFactory(...)`（ピン留め座席から社員×日×工場のマップ作成）
        
    - `fillSeatCandidateEmployees(...)`（各座席の候補従業員リスト構築）
        
    - `Pass2Constraints`（シートレベル制約）
        
    - `solvePass2Once(...)`（パス2ソルバーの実行）
        
- **Public API**：
    
    - `RunResult`
        
    - `solveFromYaml(...)`（両パスをまとめて実行）
        
    - `main(...)`
        

---

## 4. 共通ドメインクラスとユーティリティ

### 4.1 日付とユーティリティ

- `DF = DateTimeFormatter.ofPattern("yyyy/MM/dd")`  
    → YAML の日付はすべてこのフォーマットで扱う
    
- `safeStr(Object o)` → `null` なら空文字、それ以外は `String.valueOf(o)`
    
- `parseInt(Object o, int def)` → パース失敗時はデフォルト値を返す
    
- `phaseNumFromId("p3")` → `3` など、フェーズIDから数値フェーズを取得
    

`dayIdFromDate(planStart, "yyyy/MM/dd")`：

- 実際の日付を、`planStart` からの **日インデックス (0,1,2,...)** に変換
    

`CLOCK` や `nowClock()`, `fmt(Duration)` など、ソルバーの処理時間ログ用のヘルパーもあります。

### 4.2 DaySlot

`public static class DaySlot { @PlanningId public int id; public LocalDate date; }`

- 計画期間内の 1 日を表します
    
- `id` は 0 ベースの日インデックス、`date` は実際のカレンダー日付
    

### 4.3 EmployeeFact

`public static class EmployeeFact { @PlanningId public int id; // 0 = UNASSIGNED public String wid; // EnvConfig 上の worker ID public String name; public Map<String,Integer> skills = new HashMap<>(); // opId -> level public boolean isManager; public String workerCompany; }`

特別な点：

- `id == 0` は **UNASSIGNED（未アサインの幽霊社員）** 用
    
- `skills.get(opId) >= 1` で、そのオペレーションを担当できることを表現
    

ヘルパー関数：

- `isUnassigned(EmployeeFact e)`
    
- `skill(EmployeeFact e, String opId)`
    
- `isManager(EmployeeFact e)`
    
- `company(EmployeeFact e)`
    

### 4.4 TaskWindow

`public static class TaskWindow { public String module; public String factory; public String phaseId; public int phaseNum; public String opId; public int startDayId; public int endDayId; public List<Integer> allowed; // 1 日あたりの許可時間 (例: [8,10,12]) public int minHeads; public int maxHeads; public int workloadDays; // Schedule.yaml の workload_days }`

`Schedule.yaml` の `(module, phase, operation)` ごとのウィンドウを表します。

- `[startDayId .. endDayId]`：このタスクを置ける日インデックス
    
- `workloadDays * baselineHours` が固定行考慮前のベース必要時間
    

### 4.5 FixedAssign と FixedHeadDay

`static class FixedAssign { String module, opId, factory, wid; int startDayId, endDayId; Map<Integer,Integer> hoursByDay = new HashMap<>(); // dayIdx -> hours String phaseId; int phaseNum; } class FixedHeadDay { public int dayId; public String opId; public int heads; }`

- `FixedAssign`：`Schedule.yaml` の `assignment_list` から来る **固定行** を表します
    
    - どの module/op/factory/worker か
        
    - どの日に何時間やるか
        
- `buildFixedHeadDays(List<FixedAssign>)`：
    
    - 各固定行の `(dayIdx, hours>0)` を 1 ヘッドとみなし
        
    - `(dayId, opId)` ごとにヘッド数を合計して `FixedHeadDay` にする
        

`FixedHeadDay` は **パス1** で：

- 固定ヘッド + 柔軟ヘッド を合わせたときに `OP_CAPACITY` を超えないようにする制約で使用
    

---

## 5. カレンダーと非稼働日の扱い

`static class Calendars` はカレンダー関連の情報をすべて持つクラスです。

- `weekends`：週末（日インデックス）
    
- `fabOff`：fab ごとの休み日
    
- `regionOff`：地域ごとの休み日
    
- `customerOff`：顧客会社ごとの休み日
    
- `workerCompanyOff`：worker_company ごとの休み日
    
- `fabToRegion`：fab → region のマッピング
    
- `fabToCustomer`：fab → customer_company のマッピング
    
- `workerOffByWid`：従業員ごとの個人休み日
    
- `transitDays`：
    
    - `fromRegion -> (toRegion -> 必要な移動日数)` のマップ
        
- `regionStayMaxOn`：地域ごとの最大連続稼働日数
    
- `regionStayOffInterval`：地域ごとの休み間隔
    

ヘルパー：

- `transitDays(String from, String to)`
    
- `regionOfFab(String fabId)`
    
- `maxStayOn(String regionId)`
    
- `stayOffInterval(String regionId)`
    

`buildCalendars(envPath, planStart, planEnd)` の中では：

1. まず **週末 (SAT/SUN)** を `weekends` に登録
    
2. `environment.fab_list` を読み：
    
    - fab → region / customer の対応を作成
        
    - fab ごとの `unavailable_dates` を日インデックスに変換して保存
        
3. `region_list`, `customer_company_list`, `worker_company_list` を読み：
    
    - それぞれの休み日をセットに登録
        
4. `worker_list` から個人の `unavailable_dates` を取得し `workerOffByWid` に保存
    
5. `transite_day_map` を読み：
    
    - 地域間の移動日数マップを作成
        
6. `region_list` から `max_stay_on` / `stay_off_interval` を読み：
    
    - 地域滞在制限のパラメータとして利用
        

`isWorkingDay(dayId, fabId)` は、以下をすべて満たすときだけ `true` を返します。

- 週末でない
    
- fab の休みでない
    
- fab の region の休みでない
    
- fab の customer_company の休みでない
    

`workingDaysCount(startDay, dayCount, fabId)` は：

- `startDay .. startDay+dayCount-1` の範囲で、上記の条件を満たす日数を数えます
    

パス2では、`maxSegmentSpanWithBreak(dayList, offInterval)` という関数も使い：

- ある従業員がある region で働く日リストと休み間隔から
    
- 実質的な連続稼働日数の最大値を計算し
    
- `regionStayMaxOn` 制約で利用します
    

---

## 6. EnvConfig と Schedule のパース

### 6.1 ParsedEnv と OpDef

`static class OpDef { String phaseId; int phaseNum; List<Integer> allowed; int min; int max; } static class ParsedEnv { Map<String,OpDef> opdef; List<EmployeeFact> employees; Map<String,EmployeeFact> byWid; }`

`parseEnv(envPath)` の流れ：

1. ルート YAML を読み、`environment` セクションを取得
    
2. `workflow_list.phase_list.operation_list` を走査：
    
    - 各 operation ごとに：
        
        - `operation` → `opId`（例：`p1o1`）
            
        - `work_hours` → 許可時間リスト（例：`[8,10,12]` や `[4]`）
            
        - `min_worker_num`, `max_worker_num`
            
        - 所属 phase の `phaseId`, `phaseNum`
            
3. 従業員リスト `employees` を作成：
    
    - 先に UNASSIGNED worker (`id=0`) を追加
        
    - `worker_list` の各 worker について：
        
        - id, name, is_manager, worker_company
            
        - `skill_map` → `skills` マップ
            
    - `byWid` に `wid` → `EmployeeFact` で登録
        
4. グローバル変数を構築：
    
    - `OP_CAPACITY[opId]`：その op をスキル > 0 で実行できる従業員数
        
    - `OP_AVG_SKILL[opId]`：その op の平均スキル（ソフト制約で使用）
        

### 6.2 ParsedSchedule と FixedAssign

`static class ParsedSchedule { LocalDate planStart; LocalDate planEnd; List<DaySlot> daySlots; List<TaskWindow> windows; Map<String,Integer> requiredByKey; // module|opId -> ベース必要時間 List<FixedAssign> fixedRows; Map<String,Integer> fixedHoursByKey; // module|opId -> 固定時間の合計 }`

`parseSchedule(schedPath, opdef)` の流れ：

1. `schedule.plan_range.start_date` / `end_date` を読み、`planStart`, `planEnd` を決定
    
2. 期間内の `DaySlot` をすべて作成
    
3. 各 `workflow_task` (module) / `phase_task` を走査：
    
    - 各 `operation_task` ごとに `TaskWindow` を構築：
        
        - `module`, `factory`, `phaseId`, `phaseNum`, `opId`
            
        - phase の `start_date` / `end_date` を日インデックスに変換
            
        - `OpDef` から `allowed`, `minHeads`, `maxHeads` を取得
            
        - `operation_task.workload_days` を `workloadDays` として保存
            
    - `(module, opId)` ごとの **ベース必要時間** を計算して `requiredByKey` に登録
        
4. `assignment_list` を読み：
    
    - 各行について：
        
        - `operation_task` ID（例：`e16p4o1`）、`worker`、`plan_flexibility`
            
        - `work_date_list`（またはタイプミスの `work_date_lsit`）から `date` / `hour` を取得
            
        - `date` を日インデックスに変換し `FixedAssign.hoursByDay` に入れる
            
    - `plan_flexibility == "fixed"` の行だけを `FixedAssign` として保持
        
    - それらの行の時間は `(module|opId)` ごとに合計して `fixedHoursByKey` に登録
        
5. v8.3.1 では、v8.3.2 のような「固定行に合わせてウィンドウを後ろに押す処理」は行わず、  
    代わりにパス1で `FixedHeadDay` と制約を使って整合性を取る。
    
6. `buildFixedHeadDays(fixedRows)`：
    
    - 各 `FixedAssign` の `(dayIdx, hours>0)` について
        
        - `(dayId, opId)` のヘッド数を 1 ずつ増やし
            
    - `FixedHeadDay(dayId, opId, heads)` のリストとして返す
        

---

## 7. 両パス共通で使うグローバル

`static final int DAILY_CAP = 12; static double TARGET_HOURS_PER_EMP = 0.0; static final Map<String,Integer> OP_CAPACITY = new HashMap<>(); static final Map<String,Double> OP_AVG_SKILL = new HashMap<>();`

- `DAILY_CAP`：従業員 1 人あたり 1 日の最大稼働時間（12時間）
    
- `TARGET_HOURS_PER_EMP`：`solveFromYaml` 内で計算される、従業員あたり目標総稼働時間  
    （`totalRequiredHours / 実従業員数`）
    
- `OP_CAPACITY` / `OP_AVG_SKILL` は `parseEnv` で事前計算される
    

---

## 8. パス1 – ブロックレベルの計画（hours ramp）

### 8.1 BlockDecision エンティティ

`class BlockDecision { @PlanningId public int id; public String module; public String factory; public String phaseId; public int phaseNum; public String opId; public int windowStart; public int windowEnd; public int requiredHours; public List<Integer> allowed; public int minHeads; public int maxHeads; @PlanningVariable(valueRangeProviderRefs = "vrDayIds") public Integer startDay; @PlanningVariable(valueRangeProviderRefs = "vrHeadOptions") public Integer heads; @PlanningVariable(valueRangeProviderRefs = "vrDayCountOptions") public Integer days; public int seedHours = 8; }`

パス1では、各ブロックに対して以下を決定します。

- **startDay**：いつブロックを開始するか
    
- **heads**：同時に何人の作業者を付けるか
    
- **days**：何日間連続でこのブロックを走らせるか
    

1日あたりの時間はプランニング変数ではなく、`autoHours(b)` によって **自動計算** されます。

### 8.2 Pass1Plan ソリューションクラス

`@PlanningSolution public static class Pass1Plan { @ValueRangeProvider(id = "vrDayIds") @ProblemFactCollectionProperty public List<Integer> dayIds; @ValueRangeProvider(id = "vrHeadOptions") @ProblemFactCollectionProperty public List<Integer> headOptions; @ValueRangeProvider(id = "vrDayCountOptions") @ProblemFactCollectionProperty public List<Integer> dayCountOptions; @ProblemFactCollectionProperty public List<DaySlot> daySlots; @ProblemFactCollectionProperty public List<FixedHeadDay> fixedHeadDays; @PlanningEntityCollectionProperty public List<BlockDecision> blocks; @PlanningScore private HardMediumSoftScore score; }`

ValueRange はグローバルなリストです。

- `dayIds`：startDay 用の候補日
    
- `headOptions`：全ウィンドウの minHeads..maxHeads の範囲をまとめたリスト
    
- `dayCountOptions`：ウィンドウ長に合わせた `[1 .. 最大ウィンドウ長]`
    

`fixedHeadDays` は固定行から来たヘッド数で、  
パス1の capacity 制約で使用されます。

### 8.3 produced() と autoHours()

`produced(b)` はだいたい以下のような計算です：

- `H = heads`
    
- `D = workingDaysCount(startDay, days, factory)`（稼働日のみカウント）
    
- `h = autoHours(b)`（1日あたりの時間）
    
- `produced = H * h * D`
    

`autoHours(b)` は：

- `allowed`（例：`[8,10,12]`）をソート
    
- `heads`, `factory`, `days` から `D` を求め
    
- **以下を満たす h を探す**：
    
    - `prod = H * h * D` が `requiredHours` 以上
        
    - 過剰分 `over = prod - requiredHours` が 1 ブロック日分 (`H*h`) を超えない
        
- 候補が複数ある場合は：
    
    - `|h-8|` が小さいものを優先
        
    - それでも同じなら `h` が小さい方を優先
        
- 1つも「きれいな解」が見つからない場合でも、  
    underfill/overfill と 8時間からの距離を組み合わせて最もマシな h を選ぶ
    

つまり、パス1では「時間を変数にしなくても、`autoHours` でかなりスマートに時間を選ぶ」設計になっています。

### 8.4 Pass1Constraints – ハード制約

`Pass1Constraints` には、ブロックレベルのルールが定義されています。主なハード制約：

1. `withinWindow`  
    → `startDay` が `[windowStart .. windowEnd]` に入っていること
    
2. `daysWithinWindowLen`  
    → `startDay + days - 1 <= windowEnd`
    
3. `hoursValueAllowed`  
    → `autoHours(b)` が `allowed` に含まれていること
    
4. `headsInMinMax`  
    → `heads ∈ [minHeads .. maxHeads]`
    
5. `noUnderfill`  
    → `produced(b) >= requiredHours`  
    足りない分 `requiredHours - produced` をハードペナルティに
    
6. `overfillAtMostOneDay`  
    → `produced - requiredHours <= (H*h)` でなければハード違反
    
7. `phaseOrder`  
    → 同じ module 内で phase N は phase N+1 より先に終わること
    
8. `dailyHeadCapacityByOp`（パス1版）  
    → 各 `(day, opId)` ごとに、固定ヘッド + flex ブロックヘッド を合計し、  
    `OP_CAPACITY[opId]` を超えないようにする
    
9. `penalizeStackByOp`  
    → 同じ日・同じ op でブロックが過度に重なるとペナルティ  
    （ハード + ソフトの両方の側面がある）
    

### 8.5 Pass1Constraints – ソフト制約

ソフトウェイト例：

- `PREF_HOURS_WEIGHT`, `SMALLER_HOURS_W`, `SMALLER_HEADS_W`,  
    `FEWER_DAYS_W`, `EARLIER_START_W`, `STACK_PAIR_WEIGHT`
    

主なソフト制約：

- **preferHoursNear8**  
    → `|autoHours(b) - 8|` をペナルティに
    
- **preferSmallerHours**  
    → `autoHours(b)` 自体にソフトペナルティ（小さい時間を好む）
    
- **preferSmallerHeads**  
    → heads が多いほどソフトペナルティ
    
- **preferFewerDays**  
    → ブロック日数が多いほどソフトペナルティ
    
- **preferEarlierStart**  
    → startDay が遅いほどソフトペナルティ
    
- **penalizeStackByOp (soft)**  
    → 同じ日・同じ op のブロックが密集しすぎると追加ペナルティ
    

狙いとしては：

- パス1では「窓・容量・フェーズ順序・固定ヘッド」を守りつつ
    
- 少人数・短期間・早め開始・小さめ時間（例：8時間）でうまい配置を探す
    

### 8.6 `solvePass1HoursRamp(...)` – hours ramp ロジック

パス1は **「許可時間を段階的に増やしながら解く」** 方式です。

1. まず：
    
    - 全 TaskWindow の `allowed` を見て、最大要素数を `maxChoices` とする
        
    - グローバルな `headOptions`, `dayIds`, `dayCountOptions` を作成
        
2. `tier = 1 .. maxChoices` をループ：
    
    - 各 `TaskWindow w` について：
        
        - `allowedSorted` = ソート済み `allowed`
            
        - `tiered = allowedSorted.subList(0, tier)`  
            → tier=1 なら最小値のみ（例：`[8]`）、tier=2 なら `[8,10]` など
            
        - baseline = `[4]` だけなら 4、それ以外は 8
            
        - `totalReq = workloadDays * baseline`
            
        - `fixedHoursByKey` から固定分を引いて `requiredHours` を決定
            
        - `BlockDecision` を作成：
            
            - `windowStart`, `windowEnd`, `requiredHours`, `allowed=tiered`
                
            - 初期値：`startDay = windowStart`, `heads = minHeads`, `seedHours = tiered[0]`
                
            - `days` は `ceil(requiredHours / (seedHours * minHeads))` をベースにウィンドウ内に収まるよう調整
                
3. この tier 用に `Pass1Plan` を作り、以下の条件でソルバー実行：
    
    - `bestScoreLimit = "0hard/*medium/*soft"`
        
    - `spentMinutes = 30`
        
    - `unimprovedSeconds = 60`
        
4. 結果が **hardScore=0** の場合：
    
    - さらに「磨き用」のポリッシュパスを実行：
        
        - bestScoreLimit なし
            
        - `spentMinutes = 20`, `unimprovedSeconds = 60`
            
    - その結果を `Pass1Result` として返す
        
5. どの tier でも hardScore=0 に到達しなかった場合：
    
    - 試したなかで一番良い tier の結果を選び（あれば hardZero 優先）
        
    - それをそのまま `Pass1Result` として返す
        

このように、パス1は「まず小さい時間だけで何とかできないか」「ダメなら許可時間を増やす」という段階的戦略になっています。

---

## 9. ブロックと固定行から座席へ展開

### 9.1 Expanded ヘルパー

`class Expanded { List<CrewSeat> seats; List<SeatDay> seatDays; }`

- `CrewSeat`：1つのブロック or 固定行に対応する「座席」
    
- `SeatDay`：その座席の「ある1日」での稼働（何時間かを含む）
    

### 9.2 CrewSeat と SeatDay（パス2のドメイン）

`public static class CrewSeat { @PlanningId public int id; public String module; public String factory; public String phaseId; public int phaseNum; public String opId; public int startDayId; public int days; public int hours; public int seatIndex; public String seatKey; public int blockId; public boolean pinned = false; public String pinnedWid = null; private List<EmployeeFact> candidateEmployees = List.of(); @PlanningVariable(valueRangeProviderRefs = "eligibleEmployeesForSeat") public EmployeeFact employee; }`

- `seatKey` は seat を一意に表す文字列（`module|op|seatIndex|dStart` や PIN など）
    
- pinned seat の場合は、FixedAssign から対応する情報をセット
    

`public static class SeatDay { @PlanningId public String id; // seatKey + "/d" + dayId public String seatKey; public DaySlot day; public int hours; public String factory; }`

### 9.3 expandToSeats（パス1のブロック → Flex 座席）

`expandToSeats(blocks, daySlots)` の流れ：

- 各 `BlockDecision b` について：
    
    - `hours = autoHours(b)`
        
    - `start = (b.startDay != null ? b.startDay : b.windowStart)`
        
    - `dcount = (b.days != null ? b.days : 1)`
        
    - `headCount = (b.heads != null ? b.heads : 1)`
        
- `sidx = 0 .. headCount-1` をループして：
    
    - `seatKey = module + "|" + opId + "|s" + 4桁 seatIndex + "|d" + start`
        
    - `CrewSeat` を作成し：
        
        - module/factory/phase/op をコピー
            
        - `startDayId = start`, `days = dcount`, `hours = autoHours(b)`
            
        - `blockId = b.id`, `pinned=false`
            
- 各座席について、`off = 0 .. dcount-1` の日を見て：
    
    - `did = start + off`
        
    - `isWorkingDay(did, factory)` が true の場合だけ `SeatDay` を作成し、  
        `seatKey`, `day`, `hours`, `factory` を設定
        

結果：

- `Expanded.seats`：パス1からの **Flex 座席** 一覧
    
- `Expanded.seatDays`：その座席を日に展開した `SeatDay` 一覧
    

### 9.4 expandPinnedSeats（固定行 → Pinned 座席）

`expandPinnedSeats(sch, env, windows, days)` の流れ：

- まず `TaskWindow` から：
    
    - `moduleToFactory`
        
    - `moduleOpToPhase`
        
    - `moduleOpToPhaseNum`
        
    
    を作る
    
- 各 `FixedAssign fa` について：
    
    - fab は `moduleToFactory` から取得
        
    - `seatKey = module + "|" + opId + "|PIN|" + wid + "|d" + startDayId`
        
    - `CrewSeat cs` を作成：
        
        - `module`, `factory`, `phaseId`, `phaseNum`, `opId`
            
        - `startDayId = fa.startDayId`
            
        - `minDid`, `maxDid` を `hoursByDay` のキーから計算
            
        - `days = maxDid - minDid + 1`
            
        - `hours = hoursByDay.values()` の最大値
            
        - `seatIndex = 0`, `seatKey` は上記
            
        - `blockId = -1`
            
        - `pinned = true`, `pinnedWid = fa.wid`
            
        - `employee = env.byWid.get(fa.wid)`（見つからなければ UNASSIGNED）
            
    - `fa.hoursByDay` の `(dayIdx, hours)` ごとに：
        
        - 対応する `DaySlot dd` を見つけ
            
        - `SeatDay(seatKey, dd, hrs, factory)` を作成
            

結果：

- `exPinned.seats`：元の固定行を表す **Pinned 座席**
    
- `exPinned.seatDays`：その日ごとの展開
    

### 9.5 パス2用に結合

`solveFromYaml` の中で：

`Expanded exFlex   = expandToSeats(p1.blocks, sch.daySlots); Expanded exPinned = expandPinnedSeats(sch, env, sch.windows, sch.daySlots);  List<CrewSeat> allSeats = new ArrayList<>(); allSeats.addAll(exPinned.seats); allSeats.addAll(exFlex.seats);  List<SeatDay> allSeatDays = new ArrayList<>(); allSeatDays.addAll(exPinned.seatDays); allSeatDays.addAll(exFlex.seatDays);`

パス2では：

- `allSeats`（Pinned + Flex 両方）
    
- `allSeatDays`（全 SeatDay）
    

を使います。

---

## 10. パス2 – 座席レベルの計画（誰がどこで働くか）

### 10.1 Pass2Plan ソリューション

`@PlanningSolution public static class Pass2Plan { @ProblemFactCollectionProperty public List<DaySlot> days; @ProblemFactCollectionProperty public List<EmployeeFact> employees; @PlanningEntityCollectionProperty public List<CrewSeat> seats; @ProblemFactCollectionProperty public List<SeatDay> seatDays; @PlanningScore private HardMediumSoftScore score; }`

パス2では：

- `CrewSeat` が唯一の PlanningEntity
    
- `employee` は座席ごとの ValueRange から選ばれる
    
- `SeatDay` は座席と日をリンクする Fact として使用されます
    

### 10.2 ピン留めマップと候補リスト

`buildPinnedEmpDayFactory(pinnedSeats, seatDays)`：

- 各 `SeatDay` とその座席について：
    
    - 座席が pinned かつ実在社員なら
        
        - `out[empId][dayId] = factory` というマップを構築
            

これは後で：

- 同じ日・他工場に既に pinned されている社員を  
    Flex 座席の候補から除外するために使います。
    

`fillSeatCandidateEmployees(allSeats, allSeatDays, employees)`：

1. `allSeats` を pinned と flex に分ける
    
2. `seatDaysByKey[seatKey]` → その seat がカバーする dayId のリスト
    
3. `seatFactoryByKey[seatKey]` → seat の factory
    
4. 上記を使って `pinnedEmpDayFactory` を構築
    

各 **flex** 座席 `cs` について：

- `seatDayIds`：その座席が稼働する日インデックス
    
- `seatFactory`：その座席の fab
    

各従業員 `e`（UNASSIGNED 以外）についてチェック：

1. スキル条件：`skill(e, cs.opId) >= 1`
    
2. 個人休条件：
    
    - `seatDayId` のどれかが `workerOffByWid[e.wid]` に含まれていたら除外
        
3. ピン留め他工場条件：
    
    - `pinnedEmpDayFactory[e.id]` に同じ dayId で **違う factory** があれば除外
        

すべて通った場合、その従業員を `candidates` に追加し、最後に：

`cs.setCandidateEmployees(candidates);`

**注意**：ここでは UNASSIGNED を自動で追加したりしないので、  
候補が 0 の座席もありえます。その場合はハード制約によって解が押し戻されます。

### 10.3 CrewSeat の ValueRange

`CrewSeat` 内の：

`@ValueRangeProvider(id = "eligibleEmployeesForSeat") public CountableValueRange<EmployeeFact> eligibleEmployeesForSeat() {     List<EmployeeFact> base = (candidateEmployees == null) ? List.of() : candidateEmployees;     if (pinned && pinnedWid != null) {         for (EmployeeFact e : base) if (e != null && pinnedWid.equals(e.wid)) return new ListValueRange<>(List.of(e));         return new ListValueRange<>(List.of());     }     return new ListValueRange<>(base); }`

- **pinned 座席**：`pinnedWid` の社員だけが ValueRange に含まれる
    
- **flex 座席**：`fillSeatCandidateEmployees` で作った `candidateEmployees` をそのまま使う
    

### 10.4 Pass2Constraints – ハード制約

`Pass2Constraints` は `SeatDay`, `CrewSeat`, `CAL`, `OP_CAPACITY` などを使って Seat レベルの制約を定義します。

主なハード制約：

1. `assignedAndSkill`
    
    - 座席が未アサイン、または `skill < 1` の社員がアサインされている場合にペナルティ
        
    - 各座席に必ずスキルを持つ社員が付くようにする
        
2. `oneFactoryPerEmpDay`
    
    - `(employee, day)` ごとに `SeatDay` 経由で工場を集計
        
    - 異なる工場数が 2 以上のときペナルティ
        
3. `dailyCap12h`
    
    - `(employee, day)` ごとにすべての `SeatDay.hours` を合計
        
    - 合計が `DAILY_CAP`（12時間）を超えた分をペナルティ
        
4. `employeeAvailableOnSeatDays`
    
    - `workerOffByWid` に入っている日に SeatDay がある場合にペナルティ
        
5. `respectPinnedAssignments`
    
    - `cs.pinned == true` の座席では、`cs.employee.wid` が `pinnedWid` と一致しなければペナルティ
        
6. `regionTransitGap`
    
    - 同じ従業員の SeatDay ペア `(sd1, sd2)` について：
        
        - `sd2.day.id > sd1.day.id` の場合
            
        - `regionOfFab(factory1)` と `regionOfFab(factory2)` を取得
            
        - `requiredGap = CAL.transitDays(r1, r2)` を計算
            
        - `sd2.day.id - sd1.day.id <= requiredGap` ならペナルティ
            
7. `regionStayMaxOn`
    
    - `(employeeId, region)` ごとに、SeatDay の dayId を集め
        
    - `maxSegmentSpanWithBreak` により「連続稼働（休みを考慮）の最大長」を求め
        
    - それが `maxStayOn(region)` を超える場合にペナルティ
        
8. OP容量については、パス1で FixedHeadDay と合わせてチェックしているため、  
    パス2では再度同じ制約はかけていません。
    

### 10.5 Pass2Constraints – ソフト制約

ソフトウェイトの例：

- `COMPANY_PAIR_W`, `SKILL_DIVERSITY_W`, `SKILL_AVG_W` など
    

代表的なソフト制約：

1. `softSameCompanyPairs`
    
    - 同じブロック・同じ日で一緒に働いている SeatDay ペアについて
        
        - `workerCompany` が同じペアをカウントし、  
            同じ会社のメンバーが組みやすいようにバランスを取る
            
2. `softEncourageSkillVariety`
    
    - ブロック内のスキル分布を見る
        
    - 全員同じスキルだとペナルティを与え、スキルのばらつきを推奨
        
3. `softBalanceBlockAvgSkill`
    
    - 各ブロック・各 opId について、アサインされた社員の平均スキルを計算
        
    - `OP_AVG_SKILL[opId]` からの差をペナルティにして、  
        ブロックの平均スキルを全体の平均に近づける
        
4. `softBalanceTotalHours`
    
    - 従業員ごとに全 SeatDay の時間を合計し
        
    - `|totalHours - TARGET_HOURS_PER_EMP|` をペナルティにすることで
        
    - 1人あたりの総稼働時間が公平になるように調整
        

---

## 11. ソルバー構築と実行

### 11.1 共通ソルバー構築ヘルパー

`static <S> Solver<S> buildSolver(         Class<S> solutionClass,         Class<?>[] entityClasses,         Class<? extends ConstraintProvider> providerClass,         String bestScoreLimit,         Integer spentMinutes,         Integer unimprovedSeconds) {     SolverConfig cfg = new SolverConfig();     cfg.withSolutionClass(solutionClass);     cfg.withEntityClasses(entityClasses);     cfg.withScoreDirectorFactory(new ScoreDirectorFactoryConfig()         .withConstraintProviderClass(providerClass));      TerminationConfig term = new TerminationConfig();     if (bestScoreLimit != null) term.setBestScoreLimit(bestScoreLimit);     if (spentMinutes != null && spentMinutes > 0)         term.setSpentLimit(Duration.ofMinutes(spentMinutes));     if (unimprovedSeconds != null && unimprovedSeconds > 0)         term.setUnimprovedSpentLimit(Duration.ofSeconds(unimprovedSeconds));     cfg.withTerminationConfig(term);      return SolverFactory.<S>create(cfg).buildSolver(); }`

- XML 設定は使わず、すべて Java で構成
    
- `bestScoreLimit` には `"0hard/*medium/*soft"` などを設定
    
- `spentMinutes` / `unimprovedSeconds` で時間制限をかける
    

`hardZero(Score s)` は `score.toString()` が `"0hard"` で始まるかどうかでチェックします。

### 11.2 パス1のソルビング

`Pass1Result solvePass1HoursRamp(...)` は前述の通り：

- 複数 tier（allowed hours の段階）でソルバーを実行
    
- 各 tier で 30分 + unimproved 60秒
    
- hardZero を見つけたら、その tier を 20分ポリッシュして採用
    
- 見つからない場合は一番良い tier を採用
    

### 11.3 パス2のソルビング

`static Pass2Plan solvePass2Once(List<DaySlot> days,                                 List<EmployeeFact> employees,                                 List<CrewSeat> seats,                                 List<SeatDay> seatDays) {      fillSeatCandidateEmployees(seats, seatDays, employees);      Pass2Plan p2 = new Pass2Plan();     p2.days = days; p2.employees = employees;     p2.seats = seats; p2.seatDays = seatDays;      Solver<Pass2Plan> solver = buildSolver(         Pass2Plan.class,         new Class<?>[]{ CrewSeat.class },         Pass2Constraints.class,         "0hard/*medium/*soft",         30, 60     );     Pass2Plan result = solver.solve(p2);      if (hardZero(result.getScore())) {         Solver<Pass2Plan> polish = buildSolver(             Pass2Plan.class,             new Class<?>[]{ CrewSeat.class },             Pass2Constraints.class,             null, 20, 60         );         result = polish.solve(result);     }     return result; }`

- 最初のパスで `0hard` を目指して 30分（改善なし60秒で終了）
    
- `0hard` を達成したら、さらに 20分のポリッシュでソフトスコアを改善
    

---

## 12. Public API – 2パス全体をまとめる部分

`public static class RunResult { public Pass2Plan finalPlan; public LocalDate planStart; }`

`solveFromYaml(envPath, schedPath)` の流れ：

1. パース：
    
    - `ParsedEnv env = parseEnv(envPath)`
        
    - `ParsedSchedule sch = parseSchedule(schedPath, env.opdef)`
        
2. カレンダー構築：
    
    - `buildCalendars(envPath, sch.planStart, sch.planEnd)`
        
3. `TARGET_HOURS_PER_EMP` 計算：
    
    - `realEmp = env.employees.size() - 1`（UNASSIGNED を除く）
        
    - `totalReq = sum(requiredByKey)`（ベースの総必要時間）
        
    - `TARGET_HOURS_PER_EMP = totalReq / realEmp`
        
4. `FixedHeadDay` 作成：
    
    - `List<FixedHeadDay> fixedHeadDays = buildFixedHeadDays(sch.fixedRows);`
        
5. **パス1**：
    
    - `Pass1Result p1 = solvePass1HoursRamp(sch.daySlots, sch.windows, sch.fixedHoursByKey, fixedHeadDays);`
        
6. 座席に展開（Flex + Pinned）：
    
    - `Expanded exFlex = expandToSeats(p1.blocks, sch.daySlots);`
        
    - `Expanded exPinned = expandPinnedSeats(sch, env, sch.windows, sch.daySlots);`
        
    - 2つを結合して `allSeats`, `allSeatDays` を作成
        
7. **パス2**：
    
    - `Pass2Plan finalP2 = solvePass2Once(sch.daySlots, env.employees, allSeats, allSeatDays);`
        
8. 戻り値：
    
    `RunResult rr = new RunResult(); rr.finalPlan = finalP2; rr.planStart = sch.planStart; return rr;`
    

`main(args)` では：

- env/sched パスを受け取り
    
- `solveFromYaml` を呼び出し
    
- `ExportSchedule.overwriteScheduleWithAssignments(rr.finalPlan, rr.planStart, schedPath, envPath);`
    
- `"Done."` を出力
    

---

## 13. ExportSchedule.java – Schedule.yaml への書き戻し

`ExportSchedule.overwriteScheduleWithAssignments(Pass2Plan finalPass2, LocalDate planStart, String schedPath, String envPath)`

### 13.1 読み込みとルート取得

- `schedPath` から YAML を読み込む
    
- `sched = root.get("schedule")` があればそれを使用、なければ `root` をスケジュールのルートとみなす
    

### 13.2 カレンダーの初期化

- `EmployeeSchedule.CAL` がまだ初期化されていない場合：
    
    - `sched.plan_range.end_date` を読み
        
    - `EmployeeSchedule.buildCalendars(envPath, planStart, planEnd)` を呼び出す
        

### 13.3 `(module, op)` → `operation_task` ID のマッピング

- `workflow_task_list.phase_task_list.operation_task_list` を走査：
    
    - `operation_task.id` を `opTaskId[module + "|" + operation]` に保存
        

`assignment_list.operation_task` フィールドにこの ID を書きます。

### 13.4 座席メタデータと時間集計

構築するマップ：

- `seatMeta[seatKey] = { module, opId, factory }`
    
- `empBySeat[seatKey] = EmployeeFact`
    
- `seatPinned[seatKey] = boolean`
    

`finalPass2.seatDays` について：

- `seatPinned[sd.seatKey]` が true → **固定行なのでスキップ**
    
- `empBySeat[sd.seatKey]` から `EmployeeFact e` を取得
    
- `e == null` または `id == 0` → UNASSIGNED なのでスキップ
    
- `seatMeta` から `module`, `opId`, `factory` を取得
    

`perKey` に集約：

- key = `[wid, dayIdx, module, opId]`
    
- value = その組み合わせの時間の合計（複数 SeatDay が同じ組み合わせに乗ることもあるため）
    

集約後：

- `(wid, dayIdx, module, opId)` ごとに `byDay[dayIdx] = hours` を作り
    
- `(wid, module, opId)` ごとに dayIdx をソートして：
    
    - `firstIdx` = 最小の日
        
    - `lastIdx` = 最大の日
        
    - `start_date` = `planStart + firstIdx`
        
    - `end_date` = `planStart + lastIdx`
        
    - `work_date_list` は、実際に hours がある日だけを列挙（間の空白日は出さない）
        

### 13.5 Fixed 行の保存とマージ

- 元の `sched.assignment_list` を読み
    
- `plan_flexibility == "fixed"` の行だけ `preservedFixed` として保存
    
- パス2から作った柔軟行は新しい Flexible 行として作成：
    
    `Map<String,Object> a = new LinkedHashMap<>(); a.put("worker", wid); a.put("operation_task", taskId);  // opTaskId[module|opId] より a.put("start_date", startD.format(DF)); a.put("end_date",   endD.format(DF)); a.put("work_date_list", work); a.put("plan_flexibility", "Flexible");`
    
- 最後に：
    
    `List<Map<String,Object>> merged = new ArrayList<>(); merged.addAll(preservedFixed); merged.addAll(newFlex); sched.put("assignment_list", merged);`
    

### 13.6 YAML の保存

- `DumperOptions` を設定（BLOCK スタイル + pretty flow）
    
- `schedPath` に上書き保存
    
- 新規 Flexible 行数・保持した Fixed 行数をログ出力
    

---

## 14. v8.3.1 を上司・チームに説明するときのまとめ

説明するときは、次のように話すと分かりやすいです。

1. **入力 / 出力**
    
    - 入力：`EnvConfig.yaml` と `Schedule.yaml`
        
    - 出力：同じ `Schedule.yaml` に **新しい Flexible 行** を追加したもの  
        （元からある **Fixed 行はそのまま保持**）
        
2. **2パス構成**
    
    - **パス1（ブロックプランナー）**：
        
        - 「誰が働くか」には関心を持たず、「いつ / 何日 / 何人 / 1日何時間」のブロック構造だけを決める
            
        - ウィンドウ、フェーズ順序、オペレーションごとのキャパ（FixedHead を含む）、min/max 人数、underfill/overfill のルールをすべて守る
            
        - `autoHours` と hours-ramp 戦略で 1 日あたりの時間を自動的に選択
            
    - **パス2（シートプランナー）**：
        
        - パス1のブロック結果と、元の固定行を受け取り
            
        - 座席＋SeatDay に展開して、具体的に「誰がどの日にどの fab で働くか」を決める
            
        - スキル、個人休み、地域間移動、地域滞在制限、1日12時間上限、1日1工場、  
            固定行のピン留めなどをすべて守りつつ
            
        - 合計時間が偏りすぎないようにバランスをとり、  
            同じ会社メンバーで組ませるなどのソフトな好みも反映する
            
3. **カレンダー**
    
    - 週末、fab/region/customer/company/worker の off 情報はすべて `EnvConfig.yaml` から構築
        
    - 地域間の `transitDays` と `max_stay_on` / `stay_off_interval` によって、  
        現実的な出張パターン（移動日が足りない / 同じ地域に居座りすぎない）になるよう制約
        
4. **Fixed vs Flexible**
    
    - 固定行は `FixedAssign` → `FixedHeadDay` → Pinned 座席として表現
        
    - その分の時間はパス1の `requiredHours` から差し引いて、  
        solver がその時間を二重に割り当てないようにする
        
    - Export 時には固定行を上書きせず、Flexible 行だけ追加する設計
        
5. **Timefold 設定**
    
    - すべて Java コード内で設定（XML は不使用）
        
    - `Pass1Plan` と `Pass2Plan` の 2 つの PlanningSolution があり、  
        どちらも最初は `0hard/*medium/*soft` を狙って解き、  
        成功したら追加のポリッシュステージでソフトスコアを改善する構成