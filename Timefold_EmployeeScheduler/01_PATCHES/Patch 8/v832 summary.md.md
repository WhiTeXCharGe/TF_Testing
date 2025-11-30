# v832 summarize.md

This document explains **everything** in the v8.3.2 single-pass scheduler:

- `EmployeeSchedule.java` (core domain, parsing, constraints, solver pipeline)
    
- `ExportSchedule.java` (how we write assignments back into `Schedule.yaml`)
    
- `pom.xml` (Maven module, dependencies, compiler settings, exec setup)
    

The goal is that you can walk your boss and teammates through:

- What the code **does**, step by step
    
- How it connects to **EnvConfig.yaml / Schedule.yaml**
    
- How Timefold is used (entities, value ranges, constraints, solver config)
    
- How calendars and unavailable dates are built
    
- How fixed assignments are treated
    
- How the build & run pipeline works (`mvn` + `exec`)
    

---

## 1. How to build and run this module

At the very top of `EmployeeSchedule.java` you have comments:

`// mvn -DskipTests clean package   // mvn -q exec:java -D"exec.args=src\main\resource\EnvConfig.yaml src\main\resource\Schedule.yaml"`

This is the standard workflow:

1. **Build** (without tests):
    
    `mvn -DskipTests clean package`
    
    - `clean` removes old build artifacts.
        
    - `package` compiles the code and builds the module JAR under `target/`.
        
    - `-DskipTests` skips unit tests (faster).
        
2. **Run the solver**:
    
    `mvn -q exec:java -D"exec.args=src\main\resource\EnvConfig.yaml src\main\resource\Schedule.yaml"`
    
    - Uses `exec-maven-plugin` (configured in `pom.xml`) with `mainClass = com.yourorg.scheduler.EmployeeSchedule`.
        
    - `exec.args` passes two arguments into `main(String[] args)`:
        
        - `args[0]` = `EnvConfig.yaml`
            
        - `args[1]` = `Schedule.yaml`
            
    - After solving, the code calls `ExportSchedule.overwriteScheduleWithAssignments`, which **rewrites** `Schedule.yaml` with new flexible rows.
        

So the **whole pipeline** is:

- Read `EnvConfig.yaml` and `Schedule.yaml`
    
- Build calendars, domain objects, candidate employees
    
- Run Timefold solver (stage1 + stage2)
    
- Explain constraint scores
    
- Export seats back into YAML
    

---

## 2. Maven module (`pom.xml`)

### 2.1 Basic coordinates and parent

`<parent>   <groupId>com.yourorg</groupId>   <artifactId>eight-parent</artifactId>   <version>0.1.0</version>   <relativePath>../../pom.xml</relativePath> </parent>  <groupId>com.yourorg</groupId> <artifactId>employee-scheduler-v832</artifactId>  <!-- UNIQUE -->`

- This module is part of a **multi-module Maven project**.
    
- Parent POM is `eight-parent` (`../../pom.xml`).
    
    - Typically defines shared repositories, plugin versions, maybe dependency management.
        
- This child module has a **unique artifactId**: `employee-scheduler-v832`.
    
    - Makes it easy to recognize which scheduler version is used.
        

### 2.2 Java and library versions

`<properties>   <maven.compiler.source>17</maven.compiler.source>   <maven.compiler.target>17</maven.compiler.target>   <timefold.version>1.27.0</timefold.version>   <snakeyaml.version>2.2</snakeyaml.version> </properties>`

- Compile to Java **17** (consistent with Timefold’s recommendation).
    
- `timefold.version` and `snakeyaml.version` are defined as properties so you can upgrade versions in one place.
    

### 2.3 Dependencies

`<dependencies>   <dependency>     <groupId>ai.timefold.solver</groupId>     <artifactId>timefold-solver-core</artifactId>     <version>${timefold.version}</version>   </dependency>   <dependency>     <groupId>org.yaml</groupId>     <artifactId>snakeyaml</artifactId>     <version>${snakeyaml.version}</version>   </dependency> </dependencies>`

- **Timefold Solver Core**:
    
    - Provides `@PlanningSolution`, `@PlanningEntity`, `ConstraintProvider`, `SolverFactory`, etc.
        
- **SnakeYAML**:
    
    - Used for reading/writing YAML (`EnvConfig.yaml`, `Schedule.yaml`).
        

No other external dependencies — everything else is standard Java.

### 2.4 Build plugins

#### Compiler plugin

`<plugin>   <groupId>org.apache.maven.plugins</groupId>   <artifactId>maven-compiler-plugin</artifactId>   <version>3.13.0</version>   <configuration>     <source>${maven.compiler.source}</source>     <target>${maven.compiler.target}</target>   </configuration> </plugin>`

- Ensures Java 17 is used for both source and target bytecode.
    

#### Dependency copy plugin

`<plugin>   <groupId>org.apache.maven.plugins</groupId>   <artifactId>maven-dependency-plugin</artifactId>   <version>3.6.1</version>   <executions>     <execution>       <id>copy-dependencies</id>       <phase>package</phase>       <goals><goal>copy-dependencies</goal></goals>       <configuration>         <outputDirectory>${project.build.directory}/dependency</outputDirectory>         <includeScope>runtime</includeScope>       </configuration>     </execution>   </executions> </plugin>`

- On `mvn package`, copies all runtime JARs into `target/dependency/`.
    
- Useful if you want to run `java -cp target/classes:target/dependency/* ...` outside Maven.
    

#### Exec plugin

`<plugin>   <groupId>org.codehaus.mojo</groupId>   <artifactId>exec-maven-plugin</artifactId>   <version>3.2.0</version>   <configuration>     <mainClass>com.yourorg.scheduler.EmployeeSchedule</mainClass>   </configuration> </plugin>`

- Configures the main entry point; used by `mvn exec:java`.
    
- Relies on the `EmployeeSchedule.main()` method we’ll discuss later.
    

---

## 3. EmployeeSchedule.java – high-level overview

This single file contains:

1. **YAML I/O helpers**
    
2. **Domain model**:
    
    - `DaySlot`, `EmployeeFact`, `TaskWindow`
        
    - `BlockDecision` (planning entity for work blocks)
        
    - `CrewSeat` (planning entity for worker seats)
        
    - `SinglePassPlan` (planning solution)
        
3. **Calendar handling**:
    
    - Weekends, fab off, region off, customer off, worker off, transit days, max stay.
        
4. **Constraint model**: `SinglePassConstraints` implements Timefold’s `ConstraintProvider`.
    
5. **YAML parsing** of EnvConfig + Schedule:
    
    - Build operation definitions, employees, capacities, required hours, fixed assignments.
        
6. **Entity building**:
    
    - Create blocks and seats, subtract fixed hours, create pinned seats for fixed rows.
        
7. **Candidate employees**:
    
    - Filter by skill, availability, manager requirement.
        
8. **Solver pipeline**:
    
    - Build solver configs, stage 1 and stage 2.
        
    - Score explanation (per constraint).
        
    - Print earlier-start per block.
        
9. **Public API and main**:
    
    - `solveFromYaml(...)` and `main(...)`.
        

---

## 4. Imports and what they’re used for

### 4.1 Java standard imports

`import java.io.*; import java.nio.file.*; import java.time.LocalDate; import java.time.format.DateTimeFormatter; import java.util.*; import java.util.stream.Collectors;`

- `java.io.*`:
    
    - `InputStream`, `Writer`, `IOException` – used by `loadYaml`, `saveYaml`, reading/writing files.
        
- `java.nio.file.*`:
    
    - `Files`, `Paths` – used to open YAML files as streams and writers.
        
- `java.time.*`:
    
    - `LocalDate` – represents dates for the plan horizon, unavailable dates, etc.
        
    - `DateTimeFormatter` – parsing and formatting `"yyyy/MM/dd"`.
        
- `java.util.*`:
    
    - `List`, `Map`, `HashMap`, `Set`, `HashSet`, `Collections`, `Comparator`, `Arrays`, `TreeMap`, etc.
        
- `java.util.stream.Collectors`:
    
    - Used to collect streams into `List` or `Map` when parsing YAML.
        

### 4.2 Timefold imports

`import ai.timefold.solver.core.api.domain.entity.PlanningEntity; import ai.timefold.solver.core.api.domain.lookup.PlanningId; import ai.timefold.solver.core.api.domain.solution.PlanningEntityCollectionProperty; import ai.timefold.solver.core.api.domain.solution.PlanningSolution; import ai.timefold.solver.core.api.domain.solution.ProblemFactCollectionProperty; import ai.timefold.solver.core.api.domain.valuerange.CountableValueRange; import ai.timefold.solver.core.api.domain.valuerange.ValueRange; import ai.timefold.solver.core.api.domain.valuerange.ValueRangeFactory; import ai.timefold.solver.core.api.domain.valuerange.ValueRangeProvider; import ai.timefold.solver.core.api.domain.variable.PlanningVariable; import ai.timefold.solver.core.api.domain.solution.PlanningScore; import ai.timefold.solver.core.api.score.Score; import ai.timefold.solver.core.api.score.buildin.hardmediumsoft.HardMediumSoftScore; import ai.timefold.solver.core.api.score.stream.Constraint; import ai.timefold.solver.core.api.score.stream.ConstraintCollectors; import ai.timefold.solver.core.api.score.stream.ConstraintFactory; import ai.timefold.solver.core.api.score.stream.ConstraintProvider; import ai.timefold.solver.core.api.score.stream.Joiners; import ai.timefold.solver.core.api.solver.Solver; import ai.timefold.solver.core.api.solver.SolverFactory; import ai.timefold.solver.core.config.score.director.ScoreDirectorFactoryConfig; import ai.timefold.solver.core.config.solver.SolverConfig; import ai.timefold.solver.core.config.solver.termination.TerminationConfig; import ai.timefold.solver.core.impl.domain.valuerange.buildin.collection.ListValueRange; import ai.timefold.solver.core.api.solver.SolutionManager; import ai.timefold.solver.core.api.score.ScoreExplanation;`

Used for:

- **Annotations**:
    
    - `@PlanningSolution`, `@PlanningEntity`, `@PlanningVariable`, `@ProblemFactCollectionProperty`
        
- **ID & score**:
    
    - `@PlanningId`, `@PlanningScore`, `HardMediumSoftScore`
        
- **Value ranges**:
    
    - `ValueRangeProvider`, `CountableValueRange`, `ValueRangeFactory`, `ListValueRange`
        
- **Constraint definition**:
    
    - `ConstraintProvider`, `ConstraintFactory`, `ConstraintCollectors`, `Joiners`, `Constraint`
        
- **Solver configuration and execution**:
    
    - `SolverConfig`, `SolverFactory`, `Solver`, `TerminationConfig`
        
- **Score explanation**:
    
    - `SolutionManager`, `ScoreExplanation`
        

### 4.3 YAML imports

`import org.yaml.snakeyaml.DumperOptions; import org.yaml.snakeyaml.Yaml;`

- `Yaml`:
    
    - `new Yaml().load(in)` to parse YAML to `Map<String,Object>`.
        
    - `yaml.dump(root, writer)` to write back a YAML map to file.
        
- `DumperOptions`:
    
    - Configures block-style YAML and pretty flow when saving.
        

---

## 5. YAML helpers and basic utilities

### 5.1 YAML load/save

`static Map<String, Object> loadYaml(String path) throws IOException {     try (InputStream in = Files.newInputStream(Paths.get(path))) {         return new Yaml().load(in);     } } static void saveYaml(String path, Map<String, Object> root) throws IOException {     DumperOptions opt = new DumperOptions();     opt.setDefaultFlowStyle(DumperOptions.FlowStyle.BLOCK);     opt.setPrettyFlow(true);     Yaml yaml = new Yaml(opt);     try (Writer w = Files.newBufferedWriter(Paths.get(path))) {         yaml.dump(root, w);     } }`

- Load:
    
    - Uses NIO `Files.newInputStream`.
        
    - SnakeYAML returns a generic `Map<String,Object>` structure.
        
- Save:
    
    - Use block style (`key: value` with indentation).
        
    - `prettyFlow` for nicer formatting.
        

### 5.2 Utility functions

`static String safeStr(Object o) { return o == null ? "" : String.valueOf(o); } static int parseInt(Object o, int def) {     if (o == null) return def;     try { return Integer.parseInt(String.valueOf(o)); } catch (Exception e) { return def; } } static int phaseNumFromId(String pid) {     if (pid == null) return 0;     try { return Integer.parseInt(pid.trim().toLowerCase().replace("p","")); }     catch (Exception e) { return 0; } }`

- `safeStr`:
    
    - Converts any YAML object to String, safely handling `null`.
        
- `parseInt`:
    
    - Converts a YAML object (Integer, String, etc.) to int with default.
        
- `phaseNumFromId`:
    
    - Converts `p1`, `p2`, `p3` etc. into numeric 1, 2, 3.
        
    - Used for phase ordering constraints.
        

### 5.3 Date formatter

`static final DateTimeFormatter DF = DateTimeFormatter.ofPattern("yyyy/MM/dd");`

- Consistent date parsing/formatting for YAML fields.
    

---

## 6. Domain model (facts and entities)

### 6.1 DaySlot

`public static class DaySlot {     @PlanningId public int id;     public LocalDate date;     public DaySlot() {}     public DaySlot(int id, LocalDate d) { this.id = id; this.date = d; } }`

- Represents a **single day** in the planning horizon.
    
- `id` is a day index (0, 1, 2, ...).
    
- Used when grouping constraints by `(day, op)` or `(day, emp)`.
    

### 6.2 EmployeeFact

`public static class EmployeeFact {     @PlanningId public int id;     // 0 = UNASSIGNED     public String wid;     public String name;     public Map<String,Integer> skills = new HashMap<>();     public boolean isManager;     public String workerCompany;     ... }`

- Represents a **worker**:
    
    - `wid`: worker ID from EnvConfig.
        
    - `name`: worker’s name.
        
    - `skills`: map from `opId` (e.g., `p3o1`) to skill level (1–5).
        
    - `isManager`: controls manager seat requirement.
        
    - `workerCompany`: used for “same company pairs” soft constraint.
        
- `id=0` is reserved for the **UNASSIGNED** ghost worker.
    

### 6.3 TaskWindow (internal)

`public static class TaskWindow {     public String module;     public String factory;     public String phaseId;     public int phaseNum;     public String opId;     public int startDayId;     public int endDayId;     public List<Integer> allowed;     public int minHeads;     public int maxHeads;     public int workloadDays; }`

- Represents one `(module, phase, operation)` from `Schedule.yaml`:
    
    - Plan window `[startDayId, endDayId]`.
        
    - Allowed hours per day (8, 10, 12, or 4).
        
    - Min/max workers per day.
        
    - Total workload in “equivalent days” (`workload_days`).
        

### 6.4 BlockDecision (planning entity for a block of work)

`@PlanningEntity public static class BlockDecision {     @PlanningId public int id;      // facts...     public String module;     public String factory;     public String phaseId;     public int phaseNum;     public String opId;      public int windowStart;     public int windowEnd;     public int requiredHours;     public List<Integer> allowed;     public int minHeads;     public int maxHeads;      @PlanningVariable(valueRangeProviderRefs = "vrStartWithinWindow",               strengthComparatorClass = StartDayStrength.class)     public Integer startDay;      @PlanningVariable(valueRangeProviderRefs = "vrDaysWithinWindow")     public Integer days;      ...     @PlanningVariable(valueRangeProviderRefs = "vrAllowedHours",                       strengthComparatorClass = HoursStrength.class)     public Integer hours; }`

This entity holds the **decision variables for each operation block**:

- Fixed attributes:
    
    - `module`, `factory`, `phaseId`, `phaseNum`, `opId`
        
    - `windowStart`, `windowEnd` from the schedule
        
    - `requiredHours`: **remaining hours** after subtracting fixed assignments.
        
    - `allowed`: allowed hours per day from EnvConfig (e.g., [8, 10, 12]).
        
    - `minHeads`, `maxHeads` from EnvConfig.
        
- Planning variables:
    
    1. `startDay` ∈ `[windowStart .. windowEnd]`
        
        - The day index when the block begins.
            
        - Uses a strength comparator that prefers earlier start dates (good for heuristic).
            
    2. `days` ∈ `[1 .. window length]`
        
        - How many days the block spans (continuous).
            
    3. `hours` ∈ discrete list (e.g., [8, 10, 12])
        
        - The per-seat hours for all days in the block.
            

#### 6.4.1 Strength comparators

`public static final class StartDayStrength implements Comparator<Integer> {     @Override public int compare(Integer a, Integer b) {         // earlier dates first; nulls last so solver can set startDay early         if (a == null) return (b == null) ? 0 : 1;         if (b == null) return -1;         return Integer.compare(a, b);     } }`

- Helps Timefold prefer earlier `startDay` during construction heuristics.
    
- `null` is treated as “worst”.
    

`public static final class HoursStrength implements Comparator<Integer> {     @Override public int compare(Integer a, Integer b) {         // prefer smaller hours (8 < 10 < 12)         if (a == null) return (b == null) ? 0 : 1;         if (b == null) return -1;         return Integer.compare(a, b);     } }`

- Encourages smaller hours (towards 8) early in search, consistent with soft preferences.
    

#### 6.4.2 Value ranges

`@ValueRangeProvider(id = "vrStartWithinWindow") public CountableValueRange<Integer> vrStartWithinWindow() {     return ValueRangeFactory.createIntValueRange(windowStart, windowEnd + 1); }`

- Int range `[windowStart, windowEnd]` inclusive.
    

`@ValueRangeProvider(id = "vrDaysWithinWindow") public CountableValueRange<Integer> vrDaysWithinWindow() {     int maxLen = Math.max(1, windowEnd - windowStart + 1);     return ValueRangeFactory.createIntValueRange(1, maxLen + 1); }`

- Int range for number of days `[1 .. window length]`.  
    If window is 10 days, `days` ∈ [1..10].
    

`@ValueRangeProvider(id = "vrAllowedHours") public CountableValueRange<Integer> vrAllowedHours() {     List<Integer> a = (allowed == null || allowed.isEmpty())             ? List.of(8)             : allowed.stream().distinct().sorted().toList();     return new ListValueRange<>(a); }`

- Hours domain:
    
    - If `allowed` is null/empty → default `[8]`.
        
    - Else unique and sorted list from EnvConfig.
        
    - Use `ListValueRange` for discrete values.
        

#### 6.4.3 Helper: chosenHours

`public int chosenHours() {     if (hours != null) return hours;     // default to smallest allowed to bias toward 8 even before it’s set     return (allowed == null || allowed.isEmpty()) ? 8             : allowed.stream().mapToInt(Integer::intValue).min().orElse(8); }`

- For constraints, we need a stable hour value even if `hours` is not yet set.
    
- By default, it uses the **smallest allowed** hours.
    

### 6.5 CrewSeat (planning entity for worker seat)

`@PlanningEntity public static class CrewSeat {     @PlanningId public int id;      public int blockId;     public String module;     public String factory;     public String phaseId;     public int phaseNum;     public String opId;      public int seatIndex;     public boolean needManager;      public boolean pinned = false;     public String  pinnedWid = null;     public Integer pinnedStart = null;     public Integer pinnedDays  = null;     public Integer pinnedHours = null;      // Computed candidates     private List<EmployeeFact> candidateEmployees = List.of();     ...     @PlanningVariable(valueRangeProviderRefs = "eligibleEmployeesForSeat")     public EmployeeFact employee; }`

This entity represents **one seat** inside a block:

- `blockId` links to a `BlockDecision`.
    
- `seatIndex` (0,1,2,...) is the seat number. Typically:
    
    - Seat 0: `needManager = true` (manager required).
        
    - Other seats: non-manager seats.
        

Pinned-related fields:

- For **fixed assignments** (coming from Schedule.yaml):
    
    - `pinned = true`
        
    - `pinnedWid` = worker ID that must occupy this seat.
        
    - `pinnedStart`, `pinnedDays`, `pinnedHours` specify fixed schedule.
        

#### 6.5.1 UNASSIGNED employee

`private static final EmployeeFact UNASSIGNED =     new EmployeeFact(0, "__UNASSIGNED__", "__UNASSIGNED__", Map.of(), false, "");`

- Used as fallback to represent empty seat.
    

#### 6.5.2 Candidate employees and value range

`public void setCandidateEmployees(List<EmployeeFact> list) {     this.candidateEmployees = (list == null) ? List.of() : list; } public List<EmployeeFact> getCandidateEmployees() { return candidateEmployees; }  @ValueRangeProvider(id = "eligibleEmployeesForSeat") public CountableValueRange<EmployeeFact> eligibleEmployeesForSeat() {     // Pinned → pin hard     if (pinned && pinnedWid != null) {         for (EmployeeFact e : candidateEmployees) {             if (e != null && pinnedWid.equals(e.wid)) {                 return new ListValueRange<>(List.of(e));             }         }         return new ListValueRange<>(List.of(UNASSIGNED));     }      List<EmployeeFact> base = (candidateEmployees == null) ? List.of() : candidateEmployees;      if (needManager) {         base = base.stream().filter(emp -> emp != null && emp.isManager).toList();         return new ListValueRange<>(base);     } else {         if (base.isEmpty()) base = List.of(UNASSIGNED);         return new ListValueRange<>(base);     } }`

- Pinned seats:
    
    - Value range is singleton list containing only the pinned employee.
        
- Manager seats:
    
    - Only manager employees allowed.
        
- Normal seats:
    
    - All `candidateEmployees`.
        
    - If none, allow `UNASSIGNED`.
        

---

### 6.6 SinglePassPlan (planning solution)

`@PlanningSolution public static class SinglePassPlan {     @ProblemFactCollectionProperty     public List<DaySlot> days;      @ProblemFactCollectionProperty     public List<EmployeeFact> employees;      @PlanningEntityCollectionProperty     public List<BlockDecision> blocks;      @PlanningEntityCollectionProperty     public List<CrewSeat> seats;      @PlanningScore     private HardMediumSoftScore score;     ... }`

- `days` and `employees` are **problem facts**.
    
- `blocks` and `seats` are **planning entities**.
    
- `score` is the result of constraint evaluation.
    

---

## 7. Global helpers and calendars

### 7.1 Global constants and helper methods

`static final int DAILY_CAP = 12; static double TARGET_HOURS_PER_EMP = 0.0;  static final Map<String,Integer> OP_CAPACITY = new HashMap<>(); static final Map<String,Double>  OP_AVG_SKILL = new HashMap<>();  static boolean isUnassigned(EmployeeFact e) { return e == null || e.id == 0; } static int skill(EmployeeFact e, String opId) { return (e == null) ? 0 : e.skills.getOrDefault(opId, 0); } static boolean isManager(EmployeeFact e) { return e != null && e.isManager; } static String company(EmployeeFact e) { return e == null ? "" : (e.workerCompany == null ? "" : e.workerCompany); } static double avgSkill(String opId) { return OP_AVG_SKILL.getOrDefault(opId, 3.0); }`

- `DAILY_CAP = 12` is used in the seat daily cap constraint.
    
- `TARGET_HOURS_PER_EMP` is computed based on total required hours and number of employees.
    
- `OP_CAPACITY` = how many employees can work each operation in total (based on skill > 0).
    
- `OP_AVG_SKILL` = average skill level per operation across employees.
    

### 7.2 autoHours helper

`static int autoHours(BlockDecision b) {     List<Integer> allowed = (b.allowed == null || b.allowed.isEmpty())             ? List.of(8) : b.allowed.stream().sorted().collect(Collectors.toList());     int D = (b.startDay == null || b.days == null) ? 0 : workingDaysCount(b.startDay, b.days, b.factory);     if (D == 0) return allowed.get(0);     int H = 1; // per-seat     int R = Math.max(1, b.requiredHours);     ... }`

- This function explores allowed values to choose the best hours; **in current code, constraints use `chosenHours()`** and export uses `b.hours != null ? b.hours : b.chosenHours()`.
    
- `autoHours` is more of a utility and not central in the constraint scoring now.
    

### 7.3 Calendar class `Calendars`

`static class Calendars {     Set<Integer> weekends = new HashSet<>();     Map<String, Set<Integer>> fabOff = new HashMap<>();     Map<String, Set<Integer>> regionOff = new HashMap<>();     Map<String, Set<Integer>> customerOff = new HashMap<>();     Map<String, Set<Integer>> workerCompanyOff = new HashMap<>();     Map<String, String> fabToRegion = new HashMap<>();     Map<String, String> fabToCustomer = new HashMap<>();     Map<String, Set<Integer>> workerOffByWid = new HashMap<>();     Map<String, Map<String, Integer>> transitDays = new HashMap<>();     Map<String, Integer> regionStayMaxOn = new HashMap<>();     Map<String, Integer> regionStayOffInterval = new HashMap<>();     ... }`

- Contains all **unavailability** and **transit** data for the whole horizon:
    
    - `weekends`: indices of Saturday/Sunday.
        
    - `fabOff` / `regionOff` / `customerOff` / `workerCompanyOff`: set of day indices where the fab/region/customer/worker-company is closed.
        
    - `fabToRegion`, `fabToCustomer`: mapping from fab to region/customer.
        
    - `workerOffByWid`: personal off days per worker.
        
    - `transitDays[fromRegion][toRegion]`: required gap when moving regions.
        
    - `regionStayMaxOn`, `regionStayOffInterval`: parameters to enforce “maximum continuous stay in same region”.
        

Helpers:

`int transitDays(String from, String to) { ... } String regionOfFab(String fabId) { ... } int maxStayOn(String regionId) { ... } int stayOffInterval(String regionId) { ... }`

### 7.4 Building calendars from EnvConfig (unavailable dates)

`@SuppressWarnings("unchecked") public static void buildCalendars(String envPath, LocalDate planStart, LocalDate planEnd) throws IOException {     CAL = new Calendars();      int horizon = (int) (planEnd.toEpochDay() - planStart.toEpochDay()) + 1;     for (int i = 0; i < horizon; i++) {         LocalDate d = planStart.plusDays(i);         switch (d.getDayOfWeek()) {             case SATURDAY:             case SUNDAY: CAL.weekends.add(i); break;             default: ;         }     }     ... }`

- Step 1: Mark Saturdays and Sundays as weekend days in `CAL.weekends`.
    

Then:

1. Load `environment` section from `EnvConfig.yaml`:
    
    `Map<String,Object> env = (Map<String,Object>) root.getOrDefault("environment", root);`
    
2. **Fab list**:
    
    `List<Map<String,Object>> fabs = (List<Map<String,Object>>) env.getOrDefault("fab_list", List.of()); for (Map<String,Object> f : fabs) {     String fid = String.valueOf(f.get("id"));     String rid = String.valueOf(f.get("region"));     String cid = String.valueOf(f.get("customer_company"));     CAL.fabToRegion.put(fid, rid);     CAL.fabToCustomer.put(fid, cid);     Set<Integer> off = new HashSet<>();     List<Object> dates = (List<Object>) f.getOrDefault("unavailable_dates", List.of());     for (Object o : dates) {         Integer did = dayIdFromDate(planStart, String.valueOf(o));         if (did != null) off.add(did);     }     CAL.fabOff.put(fid, off); }`
    
    - For each fab:
        
        - Map fab → region & customer.
            
        - Convert each unavailable date from YAML to day index relative to `planStart`.
            
3. **Regions** (`region_list`) and **customers** (`customer_company_list`):
    
    - Same pattern: read `unavailable_dates`, convert to day indices, store in maps.
        
4. **Worker companies** (`worker_company_list`):
    
    - Unavailable dates per worker company.
        
5. **Workers** (`worker_list`):
    
    - `unavailable_dates` per worker, stored in `CAL.workerOffByWid`.
        
6. **Transit day map** (`transite_day_map`):
    
    `List<Map<String,Object>> tmap = ... for (Map<String,Object> t : tmap) {     String from = String.valueOf(t.get("from"));     String to   = String.valueOf(t.get("to"));     int days    = parseInt(t.get("days"), 0);     ...     CAL.transitDays.computeIfAbsent(from, k -> new HashMap<>()).put(to, days); }`
    
7. **Region stay limits** (second pass through `region_list`):
    
    `int maxStayOn   = parseInt(r.get("max_stay_on"), Integer.MAX_VALUE); int offInterval = parseInt(r.get("stay_off_interval"), 1); CAL.regionStayMaxOn.put(rid, maxStayOn); CAL.regionStayOffInterval.put(rid, Math.max(1, offInterval));`
    

### 7.5 Working day check and working day count

`static boolean isWorkingDay(int dayId, String fabId) {     if (CAL.weekends.contains(dayId)) return false;     if (fabId == null) return true;     if (CAL.fabOff.getOrDefault(fabId, Set.of()).contains(dayId)) return false;     String rid = CAL.fabToRegion.get(fabId);     if (rid != null && CAL.regionOff.getOrDefault(rid, Set.of()).contains(dayId)) return false;     String cid = CAL.fabToCustomer.get(fabId);     if (cid != null && CAL.customerOff.getOrDefault(cid, Set.of()).contains(dayId)) return false;     return true; }`

- A day is working only if:
    
    - Not weekend.
        
    - Not in `fabOff`.
        
    - Not in `regionOff` for that fab’s region.
        
    - Not in `customerOff` for that fab’s customer.
        

`static int workingDaysCount(Integer startDay, Integer dayCount, String fabId) {     if (startDay == null || dayCount == null || startDay < 0 || dayCount == 0) return 0;     int end = startDay + dayCount - 1;     int n = 0;     for (int d = startDay; d <= end; d++) if (isWorkingDay(d, fabId)) n++;     return n; }`

- Counts working days (excluding weekends and other off days) in a block’s span.
    

---

## 8. Constraint model (SinglePassConstraints)

`SinglePassConstraints` implements `ConstraintProvider` and defines all rules.

### 8.1 Soft weights

`static final int PREF_HOURS_WEIGHT = 3000; static final int SMALLER_HOURS_W   = 40; static final int EARLIER_START_W   = 1; static final int COMPANY_PAIR_W    = 5; static final int SKILL_DIVERSITY_W = 3; static final int SKILL_AVG_W       = 50;`

- `PREF_HOURS_WEIGHT` was used for “near 8 hours” soft (currently commented-out).
    
- `SMALLER_HOURS_W`: multiplier for total hours in `preferSmallerHours`.
    
- `EARLIER_START_W`: weight per startDay index for `preferEarlierStart`.
    
- `COMPANY_PAIR_W`, `SKILL_DIVERSITY_W`, `SKILL_AVG_W`: for optional softs (some commented out in defineConstraints).
    

### 8.2 Enabled constraints in `defineConstraints`

`@Override public Constraint[] defineConstraints(ConstraintFactory f) {     return new Constraint[] {         // block feasibility         endWithinWindow(f),         hoursValueAllowed(f),         phaseOrder(f),          // production & capacity         noUnderfillByBlock(f),         overfillAtMostOneDayByBlock(f),         dailyHeadCapacityByOp(f),          // seat-level hard rules         employeeAvailableAllDays(f),         pinnedRespected(f),         oneFactoryPerEmpPerDay(f),         dailyCap12h(f),          // softs         preferSmallerHours(f),         preferEarlierStart(f),         softBalanceTotalHours(f)     }; }`

**Commented out** (not active right now):

- `withinWindow`, `daysWithinWindowLen`
    
- `assignedAndSkill`
    
- `atLeastOneManagerPerBlock`
    
- `regionTransitGap`, `regionStayMaxOn`
    
- `preferHoursNear8`
    
- `softSameCompanyPairs`, `softEncourageSkillVariety`, `softBalanceBlockAvgSkill`
    

You can re-enable by uncommenting.

---

### 8.3 Block-level constraints

#### 8.3.1 endWithinWindow (hard)

`Constraint endWithinWindow(ConstraintFactory f) {     return f.forEach(BlockDecision.class)         .filter(b -> b.startDay != null && b.days != null                 && (b.startDay + b.days - 1) > b.windowEnd)         .penalize(HardMediumSoftScore.ONE_HARD,             b -> (b.startDay + b.days - 1) - b.windowEnd)         .asConstraint("block-end-within-window"); }`

- Ensures the block does not extend beyond `windowEnd`.
    
- Penalizes by how many days the end exceeds the window.
    

#### 8.3.2 hoursValueAllowed (hard)

`Constraint hoursValueAllowed(ConstraintFactory f) {     return f.forEach(BlockDecision.class)         .filter(b -> b.allowed == null || b.allowed.isEmpty() || !b.allowed.contains(b.chosenHours()))         .penalize(HardMediumSoftScore.ONE_HARD)         .asConstraint("block-hours-in-allowed"); }`

- Block hours must be one of the allowed values.
    

#### 8.3.3 phaseOrder (hard)

`Constraint phaseOrder(ConstraintFactory f) {     return f.forEach(BlockDecision.class)         .join(f.forEach(BlockDecision.class),             Joiners.equal((BlockDecision a) -> a.module, (BlockDecision b) -> b.module),             Joiners.equal((BlockDecision a) -> a.phaseNum + 1, (BlockDecision b) -> b.phaseNum))         .filter((a,b) -> a.startDay != null && a.days != null && b.startDay != null                 && (a.startDay + a.days - 1) >= b.startDay)         .penalize(HardMediumSoftScore.ONE_HARD, (a,b) -> (a.startDay + a.days - 1) - b.startDay + 1)         .asConstraint("phase-order"); }`

- For each module, ensures phase N **finishes before** phase N+1 **starts**.
    
- If they overlap or are reversed, penalize by the overlap length.
    

---

### 8.4 Production & capacity constraints

Helpers:

`private static int staffedCountForBlock(List<CrewSeat> seats) {     int c = 0;     for (CrewSeat s : seats) if (!isUnassigned(s.employee)) c++;     return c; }`

- Count how many seats in a block are actually staffed (excluding unassigned).
    

`private static boolean seatCoversDayAndWorking(DaySlot d, CrewSeat s, BlockDecision b) {     final boolean pinned = s.pinned;     final Integer start = pinned ? s.pinnedStart : b.startDay;     final Integer days  = pinned ? s.pinnedDays  : b.days;     if (start == null || days == null || days <= 0) return false;     return start <= d.id && d.id <= (start + days - 1) && isWorkingDay(d.id, s.factory); }`

- Checks whether seat `s` is active on day `d` and that day is a working day at that fab.
    

#### 8.4.1 noUnderfillByBlock (hard)

`Constraint noUnderfillByBlock(ConstraintFactory f) {     var perBlock = f.forEach(BlockDecision.class)         .join(f.forEach(CrewSeat.class),             Joiners.equal((BlockDecision b) -> b.id, (CrewSeat s) -> s.blockId))         .groupBy((b, s) -> b,                 ConstraintCollectors.toList((b, s) -> s));      return perBlock         .filter((b, seats) -> {             int D = workingDaysCount(b.startDay, b.days, b.factory);             int hours = b.chosenHours();             int staffed = staffedCountForBlock(seats);             int prod = staffed * hours * Math.max(0, D);             return prod < b.requiredHours;         })         .penalize(HardMediumSoftScore.ONE_HARD,             (b, seats) -> {                 int D = workingDaysCount(b.startDay, b.days, b.factory);                 int hours = b.chosenHours();                 int staffed = staffedCountForBlock(seats);                 int prod = staffed * hours * Math.max(0, D);                 return b.requiredHours - prod;             })         .asConstraint("block-no-underfill"); }`

- For each block:
    
    - Compute production = `staffed * chosenHours * workingDaysCount`.
        
    - If production < requiredHours → **hard** violation, penalty = shortfall in hours.
        

#### 8.4.2 overfillAtMostOneDayByBlock (hard)

`Constraint overfillAtMostOneDayByBlock(ConstraintFactory f) {     var perBlock = f.forEach(BlockDecision.class)         .join(f.forEach(CrewSeat.class),             Joiners.equal((BlockDecision b) -> b.id, (CrewSeat s) -> s.blockId))         .groupBy((b, s) -> b,                 ConstraintCollectors.toList((b, s) -> s));      return perBlock         .filter((b, seats) -> {             int D = workingDaysCount(b.startDay, b.days, b.factory);             int hours = b.chosenHours();             int staffed = staffedCountForBlock(seats);             int prod = staffed * hours * Math.max(0, D);             int over = prod - b.requiredHours;             return over > staffed * hours; // more than one extra day worth         })         .penalize(HardMediumSoftScore.ONE_HARD,             (b, seats) -> {                 int D = workingDaysCount(b.startDay, b.days, b.factory);                 int hours = b.chosenHours();                 int staffed = staffedCountForBlock(seats);                 int prod = staffed * hours * Math.max(0, D);                 int over = prod - b.requiredHours;                 return Math.max(0, over - staffed * hours);             })         .asConstraint("block-overfill-at-most-one-day"); }`

- Over-production is permitted **up to one extra day’s worth**.
    
- Additional overage is penalized as hard.
    

#### 8.4.3 dailyHeadCapacityByOp (hard)

`Constraint dailyHeadCapacityByOp(ConstraintFactory f) {     return f.forEach(DaySlot.class)         .join(f.forEach(CrewSeat.class),             Joiners.filtering((DaySlot d, CrewSeat s) -> !isUnassigned(s.employee)))         .join(f.forEach(BlockDecision.class),             Joiners.equal((DaySlot d, CrewSeat s) -> s.blockId, (BlockDecision b) -> b.id))         .filter(SinglePassConstraints::seatCoversDayAndWorking)         .groupBy((d, s, b) -> d.id,                  (d, s, b) -> s.opId,                  ConstraintCollectors.sum((d, s, b) -> 1))         .filter((dayId, opId, heads) -> heads > OP_CAPACITY.getOrDefault(opId, Integer.MAX_VALUE))         .penalize(HardMediumSoftScore.ONE_HARD,             (dayId, opId, heads) -> heads - OP_CAPACITY.getOrDefault(opId, Integer.MAX_VALUE))         .asConstraint("daily-head-capacity-by-op"); }`

- For each `(day, operation)` count assigned heads.
    
- If assigned > capacity (how many people have skill > 0) → hard violation.
    

---

### 8.5 Seat-level hard rules

#### 8.5.1 employeeAvailableAllDays (hard)

`Constraint employeeAvailableAllDays(ConstraintFactory f) {     return f.forEach(CrewSeat.class)         .join(f.forEach(BlockDecision.class),             Joiners.equal((CrewSeat s) -> s.blockId, (BlockDecision b) -> b.id))         .filter((s, b) -> !isUnassigned(s.employee))         .filter((s, b) -> {             final boolean pinned = s.pinned;             final Integer start = pinned ? s.pinnedStart : b.startDay;             final Integer days  = pinned ? s.pinnedDays  : b.days;             if (start == null || days == null || days <= 0) return false;              Set<Integer> off = CAL.workerOffByWid.getOrDefault(s.employee.wid, Set.of());             for (int di = 0; di < days; di++) {                 int did = start + di;                 if (!isWorkingDay(did, s.factory)) continue;                 if (off.contains(did)) return true; // violation             }             return false;         })         .penalize(HardMediumSoftScore.ONE_HARD)         .asConstraint("seat-worker-available-all-days"); }`

- If any **working day** within that seat’s span is in the worker’s personal off list → hard violation.
    

#### 8.5.2 pinnedRespected (hard)

`Constraint pinnedRespected(ConstraintFactory f) {     return f.forEach(CrewSeat.class)         .filter(s -> s.pinned)         .filter(s -> s.employee == null || s.employee.wid == null || !s.employee.wid.equals(s.pinnedWid))         .penalize(HardMediumSoftScore.ONE_HARD)         .asConstraint("seat-pinned-respected"); }`

- Pinned seat must keep the pinned worker.
    

#### 8.5.3 oneFactoryPerEmpPerDay (hard)

`Constraint oneFactoryPerEmpPerDay(ConstraintFactory f) {     return f.forEach(DaySlot.class)         .join(f.forEach(CrewSeat.class),             Joiners.filtering((DaySlot d, CrewSeat s) -> !isUnassigned(s.employee)))         .join(f.forEach(BlockDecision.class),             Joiners.equal((DaySlot d, CrewSeat s) -> s.blockId, (BlockDecision b) -> b.id))         .filter(SinglePassConstraints::seatCoversDayAndWorking)         .groupBy((d, s, b) -> Arrays.asList(s.employee.id, d.id),                  ConstraintCollectors.toSet((d, s, b) -> s.factory))         .filter((key, facs) -> facs.size() > 1)         .penalize(HardMediumSoftScore.ONE_HARD, (key, facs) -> facs.size() - 1)         .asConstraint("seat-one-factory-per-emp-day"); }`

- For each `(employee, day)`, count distinct factories they work at.
    
- If more than 1 → violation.
    

#### 8.5.4 dailyCap12h (hard)

`Constraint dailyCap12h(ConstraintFactory f) {     return f.forEach(DaySlot.class)         .join(f.forEach(CrewSeat.class),             Joiners.filtering((DaySlot d, CrewSeat s) -> !isUnassigned(s.employee)))         .join(f.forEach(BlockDecision.class),             Joiners.equal((DaySlot d, CrewSeat s) -> s.blockId, (BlockDecision b) -> b.id))         .filter(SinglePassConstraints::seatCoversDayAndWorking)         .groupBy((d, s, b) -> Arrays.asList(s.employee.id, d.id),                  ConstraintCollectors.sum((d, s, b) ->                      (s.pinned && s.pinnedHours != null && s.pinnedHours > 0)                          ? s.pinnedHours                          : b.chosenHours()))         .filter((key, tot) -> tot > DAILY_CAP)         .penalize(HardMediumSoftScore.ONE_HARD, (key, tot) -> tot - DAILY_CAP)         .asConstraint("seat-daily-cap-12h"); }`

- Total hours per `(employee, day)` must not exceed 12.
    

---

### 8.6 Soft constraints

#### 8.6.1 preferSmallerHours (soft)

`Constraint preferSmallerHours(ConstraintFactory f) {     return f.forEach(BlockDecision.class)         .penalize(HardMediumSoftScore.ONE_SOFT, b -> SMALLER_HOURS_W * b.chosenHours())         .asConstraint("soft-smaller-hours"); }`

- Bigger hours cost more soft points.
    
- Encourages blocks to choose fewer hours per day (but still constrained by requiredHours).
    

#### 8.6.2 preferEarlierStart (soft)

`Constraint preferEarlierStart(ConstraintFactory f) {     return f.forEach(BlockDecision.class)         .penalize(HardMediumSoftScore.ONE_SOFT, b -> b.startDay == null ? 0 : EARLIER_START_W * b.startDay)         .asConstraint("soft-earlier-start"); }`

- Later `startDay` indices cost more.
    
- Encourages earlier scheduling within the window.
    

#### 8.6.3 softBalanceTotalHours (soft)

`Constraint softBalanceTotalHours(ConstraintFactory f) {     return f.forEach(DaySlot.class)         .join(f.forEach(CrewSeat.class),             Joiners.filtering((DaySlot d, CrewSeat s) -> !isUnassigned(s.employee)))         .join(f.forEach(BlockDecision.class),             Joiners.equal((DaySlot d, CrewSeat s) -> s.blockId, (BlockDecision b) -> b.id))         .filter(SinglePassConstraints::seatCoversDayAndWorking)         .groupBy((d, s, b) -> s.employee.id,                  ConstraintCollectors.sum((d, s, b) ->                      (s.pinned && s.pinnedHours != null && s.pinnedHours > 0)                          ? s.pinnedHours                          : b.chosenHours()))         .penalize(HardMediumSoftScore.ONE_SOFT, (empId, tot) -> (int)Math.abs(tot - TARGET_HOURS_PER_EMP))         .asConstraint("soft-balance-total-hours"); }`

- For each employee, sum total scheduled hours.
    
- Penalize deviation from `TARGET_HOURS_PER_EMP` (computed from total required hours).
    

---

## 9. Parsing EnvConfig and Schedule

### 9.1 OpDef and ParsedEnv

`static class OpDef {     String phaseId; int phaseNum;     List<Integer> allowed; int min; int max; } static class ParsedEnv {     Map<String,OpDef> opdef; List<EmployeeFact> employees;     Map<String, EmployeeFact> byWid; }`

- `OpDef` describes each operation:
    
    - Phase, allowed hours, min and max workers.
        
- `ParsedEnv` holds:
    
    - `opdef` by operation ID.
        
    - `employees` list.
        
    - `byWid` map for quick lookup by worker ID.
        

### 9.2 ParsedSchedule and FixedAssign

`static class ParsedSchedule {     LocalDate planStart; LocalDate planEnd;     List<DaySlot> daySlots; List<TaskWindow> windows;     Map<String,Integer> requiredByKey;     List<FixedAssign> fixedRows;     Map<String,Integer> fixedHoursByKey; } static class FixedAssign {     String module; String opId; String factory; String wid;     int startDayId; int endDayId;     Map<Integer,Integer> hoursByDay = new HashMap<>();     String phaseId; int phaseNum; }`

- `ParsedSchedule` contains:
    
    - Horizon (start, end, list of `DaySlot`).
        
    - Task windows per `(module, op)`.
        
    - `requiredByKey` = total baseline required hours (before subtracting fixed).
        
    - `fixedRows` = fixed assignments from `Schedule.yaml`.
        
    - `fixedHoursByKey` = total fixed hours per `(module, op)`.
        

### 9.3 parseEnv

`@SuppressWarnings("unchecked") static ParsedEnv parseEnv(String envPath) throws IOException {     Map<String,Object> root = loadYaml(envPath);     Map<String,Object> env = (Map<String,Object>) root.getOrDefault("environment", root);     ... }`

**Workflow:**

1. Read `workflow_list.phase_list.operation_list`:
    
    - For each phase:
        
        - `phId`, `phNum` via `phaseNumFromId`.
            
    - For each operation:
        
        - `opId`, `work_hours` → allowed hours.
            
        - `min_worker_num`, `max_worker_num`.
            
    - Build `OpDef` per `opId`.
        
2. Build employees:
    
    `employees.add(new EmployeeFact(0, "__UNASSIGNED__", "__UNASSIGNED__", Map.of(), false, "")); List<Map<String,Object>> workers = (List<Map<String,Object>>) env.getOrDefault("worker_list", List.of()); ...`
    
    - For each worker:
        
        - `id`, `name`, `is_manager`, `worker_company`.
            
        - `skill_map` → `Map<String,Integer>` by operation.
            
    - Add to `employees` and `byWid`.
        
3. Compute `OP_CAPACITY`:
    
    - For each opId:
        
        - Count employees with `skill > 0` for that op.
            
    - Used in `dailyHeadCapacityByOp`.
        
4. Compute `OP_AVG_SKILL`:
    
    - For each opId:
        
        - Sum skills over employees with `lv > 0`, compute average.
            
    - Used in optional soft constraint for average skill (currently commented out).
        

### 9.4 parseSchedule

`@SuppressWarnings("unchecked") static ParsedSchedule parseSchedule(String schedPath, Map<String,OpDef> opdef) throws IOException {     Map<String,Object> root = loadYaml(schedPath);     Map<String,Object> s = (Map<String,Object>) root.getOrDefault("schedule", root);     ... }`

**Steps:**

1. Read `plan_range.start_date` & `end_date`, create `DaySlot` horizon.
    
2. For each `workflow_task` (module) and `phase_task`:
    
    - Determine:
        
        - `module` ID, `fab`.
            
        - Phase `phId`, `phNum`.
            
        - `start_date`, `end_date` for that phase.
            
    - For each `operation_task`:
        
        - `operation` → `opId`.
            
        - `workload_days` from Schedule.yaml.
            
        - Look up `OpDef` from `opdef` (ensures consistency with EnvConfig).
            
        - Baseline hours:
            
            - If allowed = [4] → baseline 4
                
            - Else → baseline 8
                
        - `requiredHours = workload_days * baseline`.
            
        - Create `TaskWindow` with:
            
            - Phase info, allowed hours, min/max heads, workloadDays.
                
3. Read `assignment_list` for **fixed assignments**.
    
    - Each assignment row:
        
        - `plan_flexibility` (string).
            
        - `operation_task` (like `e16p4o1`).
            
            - Extract module and opId by splitting at `'p'`.
                
        - `worker` (wid).
            
        - `start_date` / `end_date` → day indices.
            
        - `work_date_list` or `work_date_lsit` (typo tolerant).
            
            - For each date:
                
                - compute day index and hours, accumulate in `byDay`.
                    
    - If `plan_flexibility == "fixed"`:
        
        - Add total hours to `fixedHoursByKey[module|opId]`.
            
        - Record `FixedAssign` with module, opId, wid, start/end indices, hoursByDay, phase info.
            
        - Track `latestFixedEndInRange` and `latestFixedEndAny` per `(module, phaseNum)` to **push later phases**’ start windows.
            
4. Push windows based on fixed assignments:
    
    - For each `TaskWindow w`:
        
        - For previous phase `prev = w.phaseNum - 1`:
            
            - Find `endPrev` = max of `latestFixedEndInRange` and `latestFixedEndAny` for `(module, prev)`.
                
            - If exists, set `w.startDayId = max(w.startDayId, endPrev + 1)`.
                

This ensures that if previous phase has fixed assignment at the end, the next phase cannot start before that.

---

## 10. Building entities (single pass)

`static BuildOut buildEntitiesSinglePass(ParsedSchedule sch, ParsedEnv env) {     ... }`

### 10.1 Mapping windows and factories

- Create:
    
    - `keyToWin[module|opId] = TaskWindow`.
        
    - `moduleWins[module] = List<TaskWindow>`.
        
- For each `FixedAssign fa`:
    
    - Set `fa.factory` = factory from any window of that module.
        

### 10.2 Create BlockDecision entities

For each `TaskWindow w`:

`int baseline = (w.allowed.size()==1 && w.allowed.get(0)==4) ? 4 : 8; int totalReq = w.workloadDays * baseline; int fixed = sch.fixedHoursByKey.getOrDefault(w.module + "|" + w.opId, 0); int req = Math.max(0, totalReq - fixed);  if (req == 0) continue;`

- Baseline required hours based on allowed hours.
    
- Subtract **fixed hours** from Schedule.yaml:
    
    - We do not re-schedule those; they are pinned.
        
- If nothing remains (req == 0) → no block created.
    

Else:

`BlockDecision b = new BlockDecision(); b.id = bid++; b.module = w.module; b.factory = w.factory; b.phaseId = w.phaseId; b.phaseNum = w.phaseNum; b.opId = w.opId; b.windowStart = w.startDayId; b.windowEnd = w.endDayId; b.requiredHours = req; b.allowed = new ArrayList<>(w.allowed); b.minHeads = w.minHeads; b.maxHeads = w.maxHeads; blocks.add(b);`

### 10.3 Create CrewSeats for blocks

`for (int sidx = 0; sidx < Math.max(1, w.maxHeads); sidx++) {     CrewSeat cs = new CrewSeat();     cs.id = sid++;     cs.blockId = b.id;     cs.module = w.module; cs.factory = w.factory;     cs.phaseId = w.phaseId; cs.phaseNum = w.phaseNum;     cs.opId = w.opId;     cs.seatIndex = sidx;     cs.needManager = (sidx == 0);     cs.employee = new EmployeeFact(0, "__UNASSIGNED__", "__UNASSIGNED__", Map.of(), false, "");     seats.add(cs); }`

- For each block, create up to `maxHeads` seats.
    
- First seat requires a manager.
    

### 10.4 Mappings for fixed seats

`Map<String,String> moduleToFactory = new HashMap<>(); Map<String,String> moduleOpToPhase = new HashMap<>(); Map<String,Integer> moduleOpToPhaseNum = new HashMap<>(); ... for (TaskWindow w : sch.windows) {     moduleToFactory.put(w.module, w.factory);     moduleOpToPhase.put(w.module + "|" + w.opId, w.phaseId);     moduleOpToPhaseNum.put(w.module + "|" + w.opId, w.phaseNum); }`

### 10.5 Create pinned seats for fixed assignments

For each `FixedAssign fa`:

`String factory = moduleToFactory.getOrDefault(fa.module, fa.factory); CrewSeat cs = new CrewSeat(); cs.id = sid++; cs.blockId = -1; // independent pinned seat cs.module = fa.module; cs.factory = factory; cs.phaseId = moduleOpToPhase.getOrDefault(fa.module + "|" + fa.opId, fa.phaseId); cs.phaseNum = moduleOpToPhaseNum.getOrDefault(fa.module + "|" + fa.opId, fa.phaseNum); cs.opId = fa.opId; cs.seatIndex = 0; cs.needManager = false;  cs.pinned = true; cs.pinnedWid = fa.wid; int minDid = fa.hoursByDay.keySet().stream().min(Integer::compareTo).orElse(fa.startDayId); int maxDid = fa.hoursByDay.keySet().stream().max(Integer::compareTo).orElse(fa.startDayId); cs.pinnedStart = minDid; cs.pinnedDays  = Math.max(1, maxDid - minDid + 1); cs.pinnedHours = fa.hoursByDay.values().stream().max(Integer::compareTo).orElse(8);`

- These seats are **not associated with any block** (`blockId=-1`).
    
- They represent **fixed** schedule that the solver must respect.
    
- The worker is looked up from `env.byWid`; if not found, UNASSIGNED is used.
    

---

## 11. Filling seat candidates (per-seat eligible employees)

`private static void fillSeatCandidatesSinglePass(         List<CrewSeat> seats,         List<BlockDecision> blocks,         List<EmployeeFact> employees) {     ... }`

- Build `byBlock` map to access block by `blockId`.
    
- `personalOff` = `CAL.workerOffByWid`.
    

For each `CrewSeat s`:

1. **Pinned seat**:
    
    - Only candidate is the pinned employee (if found).
        
    - If not found → empty list.
        
2. **Non-pinned**:
    
    - Estimate start and days:
        
        `BlockDecision b = byBlock.get(s.blockId); int estStart = (b != null && b.startDay != null) ? b.startDay : b.windowStart; int estDays  = (b != null && b.days     != null) ? b.days     : Math.max(1, b.windowEnd - b.windowStart + 1);`
        
    - Loop all employees (except id=0):
        
        - Skill gate: `e.skills.getOrDefault(s.opId, 0) >= 1`
            
        - Availability gate:
            
            - For each day in `[estStart .. estStart+estDays-1]`:
                
                - If `!isWorkingDay(did, s.factory)` → ignore this day (fab closed).
                    
                - Else if worker has personal off on that day → exclude employee.
                    
    - For manager seats:
        
        - Filter candidates to `isManager == true`.
            
        - If no candidate → throw `IllegalStateException`, because you cannot schedule this module without a manager.
            

This step effectively **pre-filters** the value ranges to avoid impossible combinations and reduce search space.

---

## 12. Solver configuration and execution

### 12.1 buildSolverFactory helper

`static <S> SolverFactory<S> buildSolverFactory(Class<S> solutionClass,                                             Class<?>[] entityClasses,                                             Class<? extends ConstraintProvider> providerClass,                                             String bestScoreLimit,                                             Integer spentMinutes,                                             Integer unimprovedSeconds) {     SolverConfig cfg = new SolverConfig();     cfg.withSolutionClass(solutionClass);     cfg.withEntityClasses(entityClasses);     cfg.withScoreDirectorFactory(             new ScoreDirectorFactoryConfig().withConstraintProviderClass(providerClass)     );      TerminationConfig term = new TerminationConfig();     if (bestScoreLimit != null) term.setBestScoreLimit(bestScoreLimit);     if (spentMinutes != null && spentMinutes > 0) {         term.setSpentLimit(java.time.Duration.ofMinutes(spentMinutes));     }     if (unimprovedSeconds != null && unimprovedSeconds > 0) {         term.setUnimprovedSpentLimit(java.time.Duration.ofSeconds(unimprovedSeconds));     }     cfg.withTerminationConfig(term);      return SolverFactory.create(cfg); }`

- Creates an **in-memory** `SolverFactory` without XML.
    
- Parameters:
    
    - `bestScoreLimit` (string, e.g. `"0hard/*medium/*soft"`)
        
    - `spentMinutes` (overall time limit)
        
    - `unimprovedSeconds` (stop if no improvement for this long).
        

> **Random seed**:  
> This code does **not** set a fixed random seed. Timefold will use its default random generator.  
> If you need deterministic runs, you’d add config (e.g., `.withEnvironmentMode` or randomSeed in config file).

### 12.2 Running stage1 and stage2

Inside `solveFromYaml`:

`// Compute TARGET_HOURS_PER_EMP int realEmp = Math.max(1, env.employees.size() - 1); int totalReq = sch.requiredByKey.values().stream().mapToInt(Integer::intValue).sum(); TARGET_HOURS_PER_EMP = totalReq / (double) realEmp;`

- Uses full baseline requiredHours (before subtracting fixed) for balancing.
    

Build `SinglePassPlan p` with days, employees, blocks, seats.

#### Stage 1

`SolverFactory<SinglePassPlan> factoryStage1 = buildSolverFactory(         SinglePassPlan.class,         new Class<?>[]{ BlockDecision.class, CrewSeat.class },         SinglePassConstraints.class,         "0hard/*medium/*soft", 90, 300);  Solver<SinglePassPlan> stage1 = factoryStage1.buildSolver(); SinglePassPlan best1 = stage1.solve(p);`

- Termination:
    
    - Best score limit: `0hard/*medium/*soft` (i.e. must reach zero hard and zero medium).
        
    - Max 90 minutes.
        
    - Stop if no improvement for 300 seconds.
        

#### Stage 2 (polish)

`SolverFactory<SinglePassPlan> factoryStage2 = buildSolverFactory(         SinglePassPlan.class,         new Class<?>[]{ BlockDecision.class, CrewSeat.class },         SinglePassConstraints.class,         null /* bestScoreLimit */,         60  /* spentMinutes */,         300 /* unimprovedSeconds */);  Solver<SinglePassPlan> stage2 = factoryStage2.buildSolver(); SinglePassPlan best2 = stage2.solve(best1);`

- No `bestScoreLimit` (just trying to improve soft score).
    
- 60-minute time limit.
    

#### Score explanation

`SolutionManager<SinglePassPlan, HardMediumSoftScore> solutionManager =         SolutionManager.create(factoryStage2);  ScoreExplanation<SinglePassPlan, HardMediumSoftScore> explanation =         solutionManager.explain(best2);  explanation.getConstraintMatchTotalMap().forEach((constraintName, cmt) -> {     System.out.println(constraintName + " = " + cmt.getScore()); });`

- Prints each constraint’s contribution (hard/medium/soft) to the final score.
    

#### Earlier-start per block printout

`System.out.println("=== Java earlier-start per block ==="); for (BlockDecision b : best2.blocks) {     int sd = (b.startDay == null ? -1 : b.startDay);     int pen = (b.startDay == null ? 0 : SinglePassConstraints.EARLIER_START_W * b.startDay);     System.out.printf(         "JAVA blockId=%d module=%s op=%s startDay=%d penalty=%d%n",         b.id, b.module, b.opId, sd, pen     ); }`

- Used to compare with Python’s earlier-start penalty.
    

---

## 13. ExportSchedule.java – writing back the YAML

This class takes the final `SinglePassPlan`, merges it into existing `Schedule.yaml`, and preserves fixed rows.

### 13.1 Method signature

`public static void overwriteScheduleWithAssignments(         EmployeeSchedule.SinglePassPlan plan,         LocalDate planStart,         String schedPath,         String envPath) throws IOException`

### 13.2 Load YAML and get `schedule` root

`Map<String,Object> root; try (InputStream in = Files.newInputStream(Paths.get(schedPath))) {     root = new Yaml().load(in); }  Map<String,Object> sched = (Map<String,Object>) root.get("schedule"); if (sched == null || sched.isEmpty()) {     sched = root; // support both styles }`

- Works whether YAML nests everything under `schedule:` or not.
    

### 13.3 Ensure calendars (optional)

`try {     if (EmployeeSchedule.CAL == null || EmployeeSchedule.CAL.weekends.isEmpty()) {         Map<String,Object> pr = (Map<String,Object>) sched.get("plan_range");         LocalDate planEnd = LocalDate.parse(             String.valueOf(pr.get("end_date")).replace("-", "/"),             EmployeeSchedule.DF         );         EmployeeSchedule.buildCalendars(envPath, planStart, planEnd);     } } catch (Exception ignore) {}`

- Rebuilds calendars if not yet initialized.
    

### 13.4 Build `(module, op)` → `operation_task_id`

`Map<String,String> opTaskId = new HashMap<>(); List<Map<String,Object>> wfList =     (List<Map<String,Object>>) sched.getOrDefault("workflow_task_list", List.of()); for (Map<String,Object> wf : wfList) {     String module = String.valueOf(wf.get("id"));     List<Map<String,Object>> phases =         (List<Map<String,Object>>) wf.getOrDefault("phase_task_list", List.of());     for (Map<String,Object> ph : phases) {         List<Map<String,Object>> ops =             (List<Map<String,Object>>) ph.getOrDefault("operation_task_list", List.of());         for (Map<String,Object> ot : ops) {             String op = String.valueOf(ot.get("operation"));             String otId = String.valueOf(ot.get("id"));             opTaskId.put(module + "|" + op, otId);         }     } }`

- YAML uses `operation_task` IDs for assignment rows.
    
- We map them from `(module, operation)`.
    

### 13.5 Preserve fixed rows

`Object assignmentObj = sched.get("assignment_list"); List<Map<String,Object>> original; if (assignmentObj instanceof List) {     original = (List<Map<String,Object>>) assignmentObj; } else {     original = new ArrayList<>(); }  List<Map<String,Object>> preservedFixed = new ArrayList<>(); for (Map<String,Object> a : original) {     String flex = String.valueOf(a.getOrDefault("plan_flexibility", "Flexible"));     if ("fixed".equalsIgnoreCase(flex)) preservedFixed.add(a); }`

- All original **fixed** rows are kept as-is.
    

### 13.6 Fixed mask

`Map<String, Set<Integer>> fixedMask = new HashMap<>();  for (Map<String, Object> a : original) {     String flex = String.valueOf(a.getOrDefault("plan_flexibility", "Flexible"));     if (!"fixed".equalsIgnoreCase(flex)) continue;      String wid   = String.valueOf(a.get("worker"));     String task  = String.valueOf(a.get("operation_task"));     if (wid == null || task == null) continue;      String wdKey = a.containsKey("work_date_lsit") ? "work_date_lsit" : "work_date_list";     List<Map<String, Object>> wdl =         (List<Map<String, Object>>) a.getOrDefault(wdKey, List.of());      Set<Integer> set = fixedMask.computeIfAbsent(wid + "|" + task, k -> new HashSet<>());     for (Map<String, Object> item : wdl) {         String dateStr = String.valueOf(item.get("date"));         Integer did = EmployeeSchedule.dayIdFromDate(planStart, dateStr);         if (did != null) set.add(did);     } }`

- For each fixed row, compute a set of day indices where that `(worker, operation_task)` **already has fixed assignment**.
    
- Used later to avoid duplicating these days in flexible output.
    

### 13.7 Index blocks and prepare new flexible rows

`Map<Integer, EmployeeSchedule.BlockDecision> blockById = new HashMap<>(); if (plan.blocks != null) {     for (EmployeeSchedule.BlockDecision b : plan.blocks) blockById.put(b.id, b); } DateTimeFormatter DF = DateTimeFormatter.ofPattern("yyyy/MM/dd");  List<Map<String,Object>> newFlex = new ArrayList<>(); if (plan.seats != null) {     for (EmployeeSchedule.CrewSeat s : plan.seats) {         if (s == null || EmployeeSchedule.isUnassigned(s.employee)) continue;         if (s.pinned) continue; // skip fixed seats entirely          String module = s.module;         String opId   = s.opId;         String taskId = opTaskId.get(module + "|" + opId);         if (taskId == null) continue;          Map<Integer,Integer> byDay = new TreeMap<>();          EmployeeSchedule.BlockDecision b = blockById.get(s.blockId);         if (b != null && b.startDay != null && b.days != null && b.days > 0) {             int h = (b.hours != null) ? b.hours : b.chosenHours();             for (int i = 0; i < b.days; i++) {                 int did = b.startDay + i;                 if (!EmployeeSchedule.isWorkingDay(did, s.factory)) continue;                 byDay.merge(did, h, Integer::sum);             }         }          Set<Integer> mask = fixedMask.get(s.employee.wid + "|" + taskId);         if (mask != null && !mask.isEmpty()) {             byDay.keySet().removeAll(mask);         }         if (byDay.isEmpty()) continue;          int firstIdx = ((TreeMap<Integer,Integer>)byDay).firstKey();         int lastIdx  = ((TreeMap<Integer,Integer>)byDay).lastKey();          List<Map<String,Object>> work = new ArrayList<>();         for (Map.Entry<Integer,Integer> e : byDay.entrySet()) {             work.add(Map.of(                 "date", planStart.plusDays(e.getKey()).format(DF),                 "hour", e.getValue()             ));         }          Map<String,Object> row = new LinkedHashMap<>();         row.put("worker", s.employee.wid);         row.put("operation_task", taskId);         row.put("start_date", planStart.plusDays(firstIdx).format(DF));         row.put("end_date",   planStart.plusDays(lastIdx).format(DF));         row.put("work_date_list", work);         row.put("plan_flexibility", "Flexible");         newFlex.add(row);     } }`

Important points:

- **Pinned seats are not exported**:
    
    - They are already represented by the original fixed rows.
        
- Hours per day taken from solver:
    
    - If `b.hours` is explicitly set by solver, use that.
        
    - Otherwise fallback to `b.chosenHours()`.
        

### 13.8 Write back merged assignment list

`List<Map<String,Object>> merged = new ArrayList<>(); merged.addAll(preservedFixed); merged.addAll(newFlex); sched.put("assignment_list", merged);`

- Final `assignment_list` is `[fixed rows] + [new flexible rows]`.
    

Then write YAML:

`DumperOptions opt = new DumperOptions(); opt.setDefaultFlowStyle(DumperOptions.FlowStyle.BLOCK); opt.setPrettyFlow(true); Yaml yaml = new Yaml(opt); try (Writer out = Files.newBufferedWriter(Paths.get(schedPath))) {     yaml.dump(root, out); }  System.out.println(     "Overwrote " + schedPath +     " | new flexible rows=" + newFlex.size() +     " | preserved fixed rows=" + preservedFixed.size() );`

---

## 14. Public API and main()

### 14.1 RunResult

`public static class RunResult { public SinglePassPlan plan; public LocalDate planStart; }`

- Convenience holder for solution + planStart (needed for export).
    

### 14.2 solveFromYaml

`public static RunResult solveFromYaml(String envPath, String schedPath) throws IOException {     ParsedEnv env = parseEnv(envPath);     ParsedSchedule sch = parseSchedule(schedPath, env.opdef);      buildCalendars(envPath, sch.planStart, sch.planEnd);     ...     RunResult rr = new RunResult();     rr.plan = best2; rr.planStart = sch.planStart;     return rr; }`

- This is the **main entry** for other code (or tests) to run the solver.
    

### 14.3 main method

`public static void main(String[] args) throws Exception {     String envPath = args.length > 0 ? args[0] : "EnvConfig.yaml";     String schedPath = args.length > 1 ? args[1] : "Schedule.yaml";      RunResult rr = solveFromYaml(envPath, schedPath);      ExportSchedule.overwriteScheduleWithAssignments(             rr.plan, rr.planStart, schedPath, envPath);      System.out.println("Done."); }`

- Command-line entry used by `mvn exec:java`.
    
- After solving, immediately updates `Schedule.yaml`.
    

---

## 15. Summary for your code review explanation

When explaining to your boss and teammates, you can structure it like this:

1. **Input/Output**:
    
    - Inputs: `EnvConfig.yaml` (environment, workers, workflow, capacities, unavailable dates) and `Schedule.yaml` (initial tasks and fixed assignments).
        
    - Output: updated `Schedule.yaml` with new flexible `assignment_list` rows; fixed rows preserved.
        
2. **Domain model**:
    
    - `BlockDecision`: when and how big each operation block is (startDay, days, hours).
        
    - `CrewSeat`: which employee is assigned to each seat in those blocks, plus pinned seats for fixed rows.
        
3. **Calendars & unavailable dates**:
    
    - Built from EnvConfig and plan range.
        
    - Used by `isWorkingDay` and availability constraints.
        
4. **Constraints**:
    
    - Hard:
        
        - Block stays within window, respect phase order, meet (or slightly exceed) required hours, respect daily capacity, personal availability, one-factory-per-day, daily 12h cap, pinned seats respected.
            
    - Soft:
        
        - Prefer smaller hours per block.
            
        - Prefer earlier starts.
            
        - Balance total hours per worker.
            
5. **Fixed vs flexible**:
    
    - Fixed assignments from `Schedule.yaml` are:
        
        - Modeled as pinned seats (for respecting them).
            
        - Their hours are subtracted from requiredHours for variable blocks.
            
        - Preserved in export; solver only adds flexible rows.
            
6. **Solver execution**:
    
    - Stage 1: drive to a hard-feasible solution with 90-minute limit and bestScoreLimit.
        
    - Stage 2: polish soft score for up to 60 minutes.
        
    - Score explanation per constraint is printed.
        
7. **Build & run**:
    
    - `mvn -DskipTests clean package`
        
    - `mvn -q exec:java -D"exec.args=EnvConfig.yaml Schedule.yaml"`


# v832 summarize.md

このドキュメントは、v8.3.2 シングルパス・スケジューラの**すべて**を説明します。

- `EmployeeSchedule.java`（コアドメイン、パース、制約、ソルバーパイプライン）
    
- `ExportSchedule.java`（`Schedule.yaml` へアサイン結果を書き戻す処理）
    
- `pom.xml`（Maven モジュール、依存関係、コンパイラ設定、exec 実行設定）
    

このドキュメントのゴールは、上司やチームメンバーに対して次の点を説明できるようにすることです。

- コードが **何をしているか** をステップごとに説明できること
    
- `EnvConfig.yaml / Schedule.yaml` とどうつながっているか
    
- Timefold をどう使っているか（エンティティ、値レンジ、制約、ソルバー設定）
    
- カレンダーや「利用不可日」をどのように構築しているか
    
- 固定アサインをどのように扱っているか
    
- `mvn` と `exec` を使ったビルド & 実行パイプラインがどう動いているか
    

---

## 1. このモジュールのビルドと実行方法

`EmployeeSchedule.java` の一番上には、次のようなコメントがあります。

`// mvn -DskipTests clean package // mvn -q exec:java -D"exec.args=src\main\resource\EnvConfig.yaml src\main\resource\Schedule.yaml"`

これは標準的なワークフローです。

1. **ビルド**（テストなし）:
    
    `mvn -DskipTests clean package`
    
    - `clean` は古いビルド成果物を削除します。
        
    - `package` はコードをコンパイルし、モジュールの JAR を `target/` 配下に作成します。
        
    - `-DskipTests` はユニットテストをスキップします（高速化のため）。
        
2. **ソルバーの実行**:
    
    `mvn -q exec:java -D"exec.args=src\main\resource\EnvConfig.yaml src\main\resource\Schedule.yaml"`
    
    - `pom.xml` 内で設定している `exec-maven-plugin` を利用します（`mainClass = com.yourorg.scheduler.EmployeeSchedule`）。
        
    - `exec.args` は `main(String[] args)` に 2 つの引数を渡します。
        
        - `args[0]` = `EnvConfig.yaml`
            
        - `args[1]` = `Schedule.yaml`
            
    - ソルバーが終了すると、コードは `ExportSchedule.overwriteScheduleWithAssignments` を呼び出し、新しいフレキシブル行で `Schedule.yaml` を**上書き**します。
        

つまり、**全体のパイプライン**は次のようになります。

- `EnvConfig.yaml` と `Schedule.yaml` を読み込む
    
- カレンダー、ドメインオブジェクト、候補従業員を構築する
    
- Timefold ソルバーを実行する（ステージ1 + ステージ2）
    
- 制約スコアを説明する
    
- 席アサイン結果を YAML に書き戻す
    

---

## 2. Maven モジュール（`pom.xml`）

### 2.1 基本設定（親 POM と座標）

`<parent> <groupId>com.yourorg</groupId> <artifactId>eight-parent</artifactId> <version>0.1.0</version> <relativePath>../../pom.xml</relativePath> </parent> <groupId>com.yourorg</groupId> <artifactId>employee-scheduler-v832</artifactId> <!-- UNIQUE -->`

- このモジュールは、**マルチモジュール Maven プロジェクト**の一部です。
    
- 親 POM は `eight-parent`（`../../pom.xml`）です。
    
    - 共通のリポジトリ設定、プラグインバージョン、依存関係管理などを定義している想定です。
        
- この子モジュールは一意な `artifactId` を持ちます：`employee-scheduler-v832`
    
    - どのスケジューラバージョンを使っているかがすぐ分かります。
        

### 2.2 Java とライブラリのバージョン

`<properties> <maven.compiler.source>17</maven.compiler.source> <maven.compiler.target>17</maven.compiler.target> <timefold.version>1.27.0</timefold.version> <snakeyaml.version>2.2</snakeyaml.version> </properties>`

- Java **17** でコンパイルします（Timefold の推奨バージョンと整合）。
    
- `timefold.version` と `snakeyaml.version` はプロパティとして定義しておき、一箇所でバージョンを変更できるようにしています。
    

### 2.3 依存関係

`<dependencies> <dependency> <groupId>ai.timefold.solver</groupId> <artifactId>timefold-solver-core</artifactId> <version>${timefold.version}</version> </dependency> <dependency> <groupId>org.yaml</groupId> <artifactId>snakeyaml</artifactId> <version>${snakeyaml.version}</version> </dependency> </dependencies>`

- **Timefold Solver Core**
    
    - `@PlanningSolution`, `@PlanningEntity`, `ConstraintProvider`, `SolverFactory` などを提供します。
        
- **SnakeYAML**
    
    - `EnvConfig.yaml` と `Schedule.yaml` の読み書きに使用します。
        

その他の外部依存はなく、それ以外は標準 Java だけで動作します。

### 2.4 ビルドプラグイン

#### コンパイラプラグイン

`<plugin> <groupId>org.apache.maven.plugins</groupId> <artifactId>maven-compiler-plugin</artifactId> <version>3.13.0</version> <configuration> <source>${maven.compiler.source}</source> <target>${maven.compiler.target}</target> </configuration> </plugin>`

- Java 17 をソース・バイトコード両方に使用することを保証します。
    

#### 依存関係コピー用プラグイン

`<plugin> <groupId>org.apache.maven.plugins</groupId> <artifactId>maven-dependency-plugin</artifactId> <version>3.6.1</version> <executions> <execution> <id>copy-dependencies</id> <phase>package</phase> <goals><goal>copy-dependencies</goal></goals> <configuration> <outputDirectory>${project.build.directory}/dependency</outputDirectory> <includeScope>runtime</includeScope> </configuration> </execution> </executions> </plugin>`

- `mvn package` 実行時に、ランタイム用 JAR をすべて `target/dependency/` にコピーします。
    
- Maven を使わずに `java -cp target/classes:target/dependency/* ...` のように手動実行したいときに便利です。
    

#### Exec プラグイン

`<plugin> <groupId>org.codehaus.mojo</groupId> <artifactId>exec-maven-plugin</artifactId> <version>3.2.0</version> <configuration> <mainClass>com.yourorg.scheduler.EmployeeSchedule</mainClass> </configuration> </plugin>`

- `mvn exec:java` で使用されるエントリポイント（`EmployeeSchedule.main()`）を設定します。
    

---

## 3. EmployeeSchedule.java – 全体像

この 1 ファイルの中に、次のものがすべて入っています。

1. **YAML I/O ヘルパー**
    
2. **ドメインモデル**:
    
    - `DaySlot`, `EmployeeFact`, `TaskWindow`
        
    - `BlockDecision`（作業ブロック用のプランニングエンティティ）
        
    - `CrewSeat`（作業者席用のプランニングエンティティ）
        
    - `SinglePassPlan`（プランニングソリューション）
        
3. **カレンダー処理**:
    
    - 週末、工場休、地域休、顧客休、ワーカー休、移動日、最大滞在日数など
        
4. **制約モデル**: `SinglePassConstraints` が Timefold の `ConstraintProvider` を実装
    
5. **EnvConfig / Schedule の YAML パース**:
    
    - 工程定義、従業員、キャパシティ、必要工数、固定アサインの構築
        
6. **エンティティ生成**:
    
    - ブロック・席の作成、固定工数の差し引き、固定行用ピン留め席の作成
        
7. **候補従業員の絞り込み**:
    
    - スキル・稼働可能日・管理者要件でフィルタリング
        
8. **ソルバーパイプライン**:
    
    - ソルバー設定の構築、ステージ1・ステージ2の実行
        
    - 制約ごとのスコア説明
        
    - ブロックごとの早期開始ペナルティの出力
        
9. **公開 API と main**:
    
    - `solveFromYaml(...)` と `main(...)`
        

---

## 4. import 文と用途

### 4.1 Java 標準ライブラリ

`import java.io.*; import java.nio.file.*; import java.time.LocalDate; import java.time.format.DateTimeFormatter; import java.util.*; import java.util.stream.Collectors;`

- `java.io.*`
    
    - `InputStream`, `Writer`, `IOException` など。`loadYaml`, `saveYaml` でファイル読み書きに使用。
        
- `java.nio.file.*`
    
    - `Files`, `Paths` など。YAML ファイルをストリームや writer として開くのに使用。
        
- `java.time.*`
    
    - `LocalDate`：計画期間の各日付、利用不可日などを表現。
        
    - `DateTimeFormatter`：`"yyyy/MM/dd"` フォーマットで日付をパース/フォーマット。
        
- `java.util.*`
    
    - `List`, `Map`, `HashMap`, `Set`, `HashSet`, `Collections`, `Comparator`, `Arrays`, `TreeMap` など。
        
- `java.util.stream.Collectors`
    
    - YAML パース時などに `List` や `Map` に集約するために使用。
        

### 4.2 Timefold 関連

（import 一覧は原文と同じ）

これらは以下の用途で使用されます。

- **アノテーション**
    
    - `@PlanningSolution`, `@PlanningEntity`, `@PlanningVariable`, `@ProblemFactCollectionProperty` など
        
- **ID & スコア**
    
    - `@PlanningId`, `@PlanningScore`, `HardMediumSoftScore`
        
- **値レンジ**
    
    - `ValueRangeProvider`, `CountableValueRange`, `ValueRangeFactory`, `ListValueRange`
        
- **制約定義**
    
    - `ConstraintProvider`, `ConstraintFactory`, `ConstraintCollectors`, `Joiners`, `Constraint`
        
- **ソルバー設定と実行**
    
    - `SolverConfig`, `SolverFactory`, `Solver`, `TerminationConfig`
        
- **スコア説明**
    
    - `SolutionManager`, `ScoreExplanation`
        

### 4.3 YAML 関連

`import org.yaml.snakeyaml.DumperOptions; import org.yaml.snakeyaml.Yaml;`

- `Yaml`
    
    - `new Yaml().load(in)` で YAML を `Map<String,Object>` として読み込み。
        
    - `yaml.dump(root, writer)` で YAML をファイルに書き出し。
        
- `DumperOptions`
    
    - YAML のブロックスタイルや整形設定を行う。
        

---

## 5. YAML ヘルパーとユーティリティ

### 5.1 YAML の読み書き

（コードは原文のまま）

- 読み込み:
    
    - NIO の `Files.newInputStream` を使って入力ストリームを開きます。
        
    - SnakeYAML が汎用的な `Map<String,Object>` 構造で返します。
        
- 書き込み:
    
    - ブロックスタイル（インデント付きの `key: value`）で出力します。
        
    - `prettyFlow` によって見やすい YAML を生成します。
        

### 5.2 ユーティリティ関数

（コードは原文のまま）

- `safeStr`
    
    - YAML から読み込んだオブジェクトを `String` に変換し、`null` の場合は空文字を返します。
        
- `parseInt`
    
    - YAML オブジェクト（`Integer`, `String` など）を `int` に変換し、失敗したらデフォルト値を返します。
        
- `phaseNumFromId`
    
    - `p1`, `p2`, `p3` のような ID を数値 1, 2, 3 に変換します。
        
    - フェーズ順序制約で使用します。
        

### 5.3 日付フォーマッタ

`static final DateTimeFormatter DF = DateTimeFormatter.ofPattern("yyyy/MM/dd");`

- YAML の日付フィールド用に、共通のパターンを定義しています。
    

---

## 6. ドメインモデル（ファクトとエンティティ）

### 6.1 DaySlot

（コードは原文のまま）

- 計画期間内の**1 日**を表します。
    
- `id` は日インデックス（0, 1, 2, ...）です。
    
- `(day, op)` や `(day, emp)` でグルーピングする制約で使用します。
    

### 6.2 EmployeeFact

（コードは原文のまま）

- **従業員**を表します。
    
    - `wid`: EnvConfig からの worker ID
        
    - `name`: 従業員名
        
    - `skills`: `opId`（例: `p3o1`）からスキルレベル（1–5）へのマップ
        
    - `isManager`: 管理者席が必要な場合に使用
        
    - `workerCompany`: 「同一会社ペア」のソフト制約（オプション）で使用
        
- `id=0` は **UNASSIGNED（未割り当て）** の疑似従業員として予約されています。
    

### 6.3 TaskWindow（内部クラス）

（コードは原文のまま）

- `Schedule.yaml` の各 `(module, phase, operation)` に対応する情報を表します。
    
    - 計画ウィンドウ `[startDayId, endDayId]`
        
    - 1 日あたりの許容時間（8, 10, 12, 4 など）
        
    - 1 日あたりの最小/最大ヘッド数
        
    - 「等価日数」のワークロード（`workload_days`）
        

### 6.4 BlockDecision（作業ブロック用プランニングエンティティ）

（クラス定義は原文のまま）

このエンティティは、各オペレーションブロックの**意思決定変数**を保持します。

- 固定属性:
    
    - `module`, `factory`, `phaseId`, `phaseNum`, `opId`
        
    - `windowStart`, `windowEnd`（スケジュール側のウィンドウ）
        
    - `requiredHours`: 固定アサインを差し引いた**残り**必要工数
        
    - `allowed`: EnvConfig に定義された許容時間（例: [8, 10, 12]）
        
    - `minHeads`, `maxHeads`: EnvConfig に定義された最小/最大ヘッド数
        
- プランニング変数:
    
    1. `startDay` ∈ `[windowStart .. windowEnd]`
        
        - ブロックの開始日（インデックス）
            
        - ストレングスコンパレータで「早い開始日」を優先
            
    2. `days` ∈ `[1 .. ウィンドウ長]`
        
        - ブロックが連続して何日続くか
            
    3. `hours` ∈ 離散リスト（例: [8, 10, 12]）
        
        - ブロックの全日で共通の 1 席あたり時間
            

#### 6.4.1 ストレングスコンパレータ

（`StartDayStrength`, `HoursStrength` のコードは原文のまま）

- `StartDayStrength`
    
    - 早い `startDay` を優先することで、ヒューリスティックを改善します。
        
    - `null` は「最も弱い」値として扱います。
        
- `HoursStrength`
    
    - `8 < 10 < 12` のように、より小さい時間を優先します。
        
    - ソフト制約の好みに合わせて、小さい工数から探索させます。
        

#### 6.4.2 値レンジ

`vrStartWithinWindow`, `vrDaysWithinWindow`, `vrAllowedHours` は、それぞれ `startDay`, `days`, `hours` の取り得る値の範囲を定義しています（コードは原文のまま）。

- `vrStartWithinWindow`
    
    - `[windowStart, windowEnd]` の整数レンジを返します。
        
- `vrDaysWithinWindow`
    
    - ウィンドウ長に応じた `[1 .. window length]` を返します。
        
- `vrAllowedHours`
    
    - EnvConfig の `allowed` から重複排除 & ソートしたリストを `ListValueRange` として返します。
        
    - `allowed` が空なら `[8]` をデフォルトとします。
        

#### 6.4.3 `chosenHours` ヘルパー

（コードは原文のまま）

- 制約計算時に、`hours` がまだ `null` の場合でも安定した値を使うためのヘルパーです。
    
- `allowed` の中で最小の時間をデフォルトとして使います。
    

### 6.5 CrewSeat（席用プランニングエンティティ）

（クラス定義は原文のまま）

このエンティティは、ブロック内の**1 席**を表します。

- `blockId` は対応する `BlockDecision` を指します。
    
- `seatIndex`（0,1,2,...）は席番号です。一般的には:
    
    - `seatIndex = 0`: `needManager = true`（管理者席）
        
    - その他: 一般席
        
- ピン留め関連フィールド:
    
    - 固定アサイン由来の席に対して使用します。
        
    - `pinned=true`、`pinnedWid` に従業員 ID、`pinnedStart`, `pinnedDays`, `pinnedHours` に固定スケジュールを設定します。
        

#### 6.5.1 UNASSIGNED 従業員

（`UNASSIGNED` 定義は原文のまま）

- 空席を表す特殊な従業員です。
    

#### 6.5.2 候補従業員と値レンジ

`setCandidateEmployees`, `eligibleEmployeesForSeat` など（コードは原文のまま）

- ピン留め席:
    
    - 値レンジは「そのピン留め従業員のみ」を含むリストになります。
        
- 管理者席:
    
    - `isManager = true` の従業員のみを候補とします。
        
- 通常席:
    
    - 事前に絞り込んだ `candidateEmployees` 全員を候補とします。
        
    - 候補がいない場合は `UNASSIGNED` を 1件だけ候補として許可します。
        

---

### 6.6 SinglePassPlan（プランニングソリューション）

（クラス定義は原文のまま）

- `days` と `employees` は **Problem Fact**（不変の入力）です。
    
- `blocks` と `seats` は **Planning Entity**（ソルバーが変更するオブジェクト）です。
    
- `score` は制約評価の結果です。
    

---

## 7. グローバルヘルパーとカレンダー

### 7.1 グローバル定数とヘルパー

（コードは原文のまま）

- `DAILY_CAP = 12`
    
    - 1 日あたりの上限工数（12 時間制約）です。
        
- `TARGET_HOURS_PER_EMP`
    
    - 全体の必要工数と従業員数から計算され、ソフト制約で使用されます。
        
- `OP_CAPACITY`
    
    - 各オペレーションごとに「スキル > 0 の従業員数」を記録します。
        
- `OP_AVG_SKILL`
    
    - 各オペレーションごとにスキルの平均値を記録します（オプションのソフト制約用）。
        

ヘルパー関数 `isUnassigned`, `skill`, `isManager`, `company`, `avgSkill` もここで定義されています。

### 7.2 `autoHours` ヘルパー

（コードは原文のまま）

- 許容時間リストから「適切な時間」を探索するユーティリティ関数です。
    
- 現行コードでは、制約は主に `chosenHours()` を使用し、エクスポート時も `b.hours != null ? b.hours : b.chosenHours()` を使うため、`autoHours` は補助的な役割になっています。
    

### 7.3 カレンダークラス `Calendars`

（クラス定義は原文のまま）

- 計画期間全体に対する **利用不可情報**と**移動情報**を保持します。
    
    - `weekends`: 土日
        
    - `fabOff` / `regionOff` / `customerOff` / `workerCompanyOff`:
        
        - 工場・地域・顧客・会社単位の休業日
            
    - `fabToRegion`, `fabToCustomer`:
        
        - 工場から地域・顧客へのマッピング
            
    - `workerOffByWid`:
        
        - 従業員ごとの個人休
            
    - `transitDays`:
        
        - 地域間移動に必要な日数
            
    - `regionStayMaxOn`, `regionStayOffInterval`:
        
        - 地域ごとの最大連続稼働日数と休み間隔
            

### 7.4 EnvConfig からのカレンダー構築

`buildCalendars`（コードは原文のまま）

- ステップ 1: 計画期間内の土日を `CAL.weekends` に登録。
    
- 次に `EnvConfig.yaml` の `environment` セクションから以下を読み込みます。
    
    1. **fab_list**
        
        - 各工場の `id`, `region`, `customer_company`, `unavailable_dates` を読み込み、日インデックスに変換。
            
    2. **region_list / customer_company_list**
        
        - 地域・顧客単位の `unavailable_dates` を登録。
            
    3. **worker_company_list**
        
        - 会社単位の `unavailable_dates` を登録。
            
    4. **worker_list**
        
        - 従業員ごとの `unavailable_dates` を `workerOffByWid` に登録。
            
    5. **transite_day_map**
        
        - `from`, `to`, `days` を読み込み、地域間の移動日数マップを設定。
            
    6. 再度 `region_list` を見て:
        
        - `max_stay_on`, `stay_off_interval` を読み込み、地域ごとの連続稼働上限と休み間隔を設定。
            

### 7.5 稼働日チェックと稼働日数カウント

`isWorkingDay`, `workingDaysCount`（コードは原文のまま）

- `isWorkingDay`
    
    - 土日でないこと
        
    - fab がオフでないこと
        
    - fab の地域がオフでないこと
        
    - fab の顧客がオフでないこと  
        のすべてを満たす場合に「稼働日」と判定します。
        
- `workingDaysCount`
    
    - ブロックの期間内で、上記条件を満たす日数を数えます。
        

---

## 8. 制約モデル（SinglePassConstraints）

`SinglePassConstraints` は `ConstraintProvider` を実装し、全ての制約ルールを定義します。

### 8.1 ソフト制約の重み

（定数定義は原文のまま）

- `PREF_HOURS_WEIGHT`
    
    - 「8時間に近づけたい」ソフト制約用（現在はコメントアウト）。
        
- `SMALLER_HOURS_W`
    
    - `preferSmallerHours` 制約で、時間が大きいほどペナルティを増やすための係数。
        
- `EARLIER_START_W`
    
    - `preferEarlierStart` 制約で、開始日が遅いほどペナルティを増やすための係数。
        
- `COMPANY_PAIR_W`, `SKILL_DIVERSITY_W`, `SKILL_AVG_W`
    
    - オプションのソフト制約用（`defineConstraints` ではコメントアウトされています）。
        

### 8.2 `defineConstraints` で有効になっている制約

（メソッド定義は原文のまま）

有効になっている主な制約は:

- ブロックの整合性:
    
    - `endWithinWindow`
        
    - `hoursValueAllowed`
        
    - `phaseOrder`
        
- 生産量・キャパシティ:
    
    - `noUnderfillByBlock`
        
    - `overfillAtMostOneDayByBlock`
        
    - `dailyHeadCapacityByOp`
        
- 席レベルのハード制約:
    
    - `employeeAvailableAllDays`
        
    - `pinnedRespected`
        
    - `oneFactoryPerEmpPerDay`
        
    - `dailyCap12h`
        
- ソフト制約:
    
    - `preferSmallerHours`
        
    - `preferEarlierStart`
        
    - `softBalanceTotalHours`
        

コメントアウトされている制約（`withinWindow`, `daysWithinWindowLen`, `regionTransitGap` など）は、必要に応じて再度有効化できます。

---

### 8.3 ブロックレベルの制約

#### 8.3.1 `endWithinWindow`（ハード）

（コードは原文のまま）

- ブロックの終了日が `windowEnd` を超えないようにします。
    
- 超えた日数分だけハードペナルティを与えます。
    

#### 8.3.2 `hoursValueAllowed`（ハード）

（コードは原文のまま）

- ブロックの時間数が `allowed` に含まれる値であることを保証します。
    

#### 8.3.3 `phaseOrder`（ハード）

（コードは原文のまま）

- 同じモジュール内で、フェーズ N がフェーズ N+1 より前に終了していることを保証します。
    
- フェーズが重なっていたり逆転している場合、その重なり長に応じてペナルティを与えます。
    

---

### 8.4 生産量 & キャパシティ制約

`staffedCountForBlock`, `seatCoversDayAndWorking` のヘルパーを使って計算します（コードは原文のまま）。

#### 8.4.1 `noUnderfillByBlock`（ハード）

（コードは原文のまま）

- 各ブロックに対して、
    
    - 生産量 = `staffed * chosenHours * workingDaysCount`
        
    - これが `requiredHours` を下回らないようにします。
        
- 不足分の時間だけハードペナルティを与えます。
    

#### 8.4.2 `overfillAtMostOneDayByBlock`（ハード）

（コードは原文のまま）

- 生産量が `requiredHours` を超えることは許容しますが、「1 日分の生産量（`staffed * hours`）を超えてはならない」という制約です。
    
- 1 日分をさらに超えたオーバー分に対してハードペナルティを与えます。
    

#### 8.4.3 `dailyHeadCapacityByOp`（ハード）

（コードは原文のまま）

- 各 `(day, operation)` ごとに、割り当てられた人数を数えます。
    
- その数が `OP_CAPACITY[opId]`（= スキル > 0 の従業員数）を超えた場合、超過人数分のハードペナルティを与えます。
    

---

### 8.5 席レベルのハード制約

#### 8.5.1 `employeeAvailableAllDays`（ハード）

（コードは原文のまま）

- 各席について、その席の期間内で「稼働日」かつ「個人休」の日が 1 日でもあるとハード違反とします。
    
- 個人の `unavailable_dates` を考慮しています。
    

#### 8.5.2 `pinnedRespected`（ハード）

（コードは原文のまま）

- `pinned = true` の席は、必ず `pinnedWid` の従業員が割り当てられていなければなりません。
    

#### 8.5.3 `oneFactoryPerEmpPerDay`（ハード）

（コードは原文のまま）

- 各 `(従業員, 日)` について、担当する fab（工場）の種類が 1 つだけであることを保証します。
    
- 2 つ以上の fab に出ている場合、その数 - 1 をハードペナルティとして与えます。
    

#### 8.5.4 `dailyCap12h`（ハード）

（コードは原文のまま）

- 各 `(従業員, 日)` の合計工数（時間数）が `DAILY_CAP`（=12）を超えないことを保証します。
    
- 超過時間数だけハードペナルティを与えます。
    

---

### 8.6 ソフト制約

#### 8.6.1 `preferSmallerHours`（ソフト）

（コードは原文のまま）

- 各ブロックについて、`SMALLER_HOURS_W * chosenHours` だけソフトペナルティを加算します。
    
- 時間数が大きいほどペナルティが大きくなるため、より小さい時間のブロックを好む傾向が生まれます。
    

#### 8.6.2 `preferEarlierStart`（ソフト）

（コードは原文のまま）

- `startDay` が遅いほど `EARLIER_START_W * startDay` のソフトペナルティを与えます。
    
- 結果として、ウィンドウ内でなるべく早く開始するスケジュールが好まれます。
    

#### 8.6.3 `softBalanceTotalHours`（ソフト）

（コードは原文のまま）

- 従業員ごとの総工数を求め、`TARGET_HOURS_PER_EMP` との絶対差に応じてソフトペナルティを与えます。
    
- 全員の工数を均等に近づけるように働きます。
    

---

## 9. EnvConfig と Schedule のパース

### 9.1 OpDef と ParsedEnv

（クラス定義は原文のまま）

- `OpDef`
    
    - 各オペレーションのフェーズ、許容時間リスト、最小・最大ヘッド数などを表します。
        
- `ParsedEnv`
    
    - `opdef`: `opId` → `OpDef`
        
    - `employees`: `EmployeeFact` のリスト
        
    - `byWid`: worker ID → `EmployeeFact`
        

### 9.2 ParsedSchedule と FixedAssign

（クラス定義は原文のまま）

- `ParsedSchedule`
    
    - 計画開始・終了日と `DaySlot` リスト
        
    - 各 `(module, op)` の `TaskWindow` リスト
        
    - `requiredByKey`: `(module|opId)` ごとの基準必要工数（固定アサイン前）
        
    - `fixedRows`: `Schedule.yaml` に記載された固定アサイン行
        
    - `fixedHoursByKey`: `(module|opId)` ごとの固定工数合計
        
- `FixedAssign`
    
    - 固定アサイン 1 行を表し、モジュール、オペレーション、fab、worker、日インデックス、時間数マップなどを持ちます。
        

### 9.3 `parseEnv`

（メソッド本体は原文のまま）

主な処理フロー:

1. `workflow_list.phase_list.operation_list` を読む
    
    - 各フェーズについて:
        
        - `phId`, `phNum` を取得し、`phaseNumFromId` で数値化。
            
    - 各オペレーションについて:
        
        - `opId`, `work_hours`（許容時間）、`min_worker_num`, `max_worker_num` を取得。
            
        - それぞれ `OpDef` として登録。
            
2. 従業員の構築
    
    - 先頭に UNASSIGNED 従業員（id=0）を追加。
        
    - `worker_list` から各従業員を読み込み:
        
        - `id`, `name`, `is_manager`, `worker_company`
            
        - `skill_map` を `Map<String,Integer>` に変換。
            
3. `OP_CAPACITY` の計算
    
    - 各 `opId` ごとに `skill > 0` の従業員数を数えます。
        
4. `OP_AVG_SKILL` の計算
    
    - 各 `opId` ごとにスキルの平均値を算出します（オプションのソフト制約用）。
        

### 9.4 `parseSchedule`

（メソッド本体は原文のまま）

主な処理フロー:

1. `plan_range.start_date` と `end_date` を読み、`DaySlot` のリストを作成。
    
2. 各 `workflow_task`（モジュール）と `phase_task` をループ:
    
    - モジュール ID、fab ID
        
    - フェーズ ID / 番号
        
    - フェーズの `start_date` / `end_date`
        
    - 各 `operation_task` について:
        
        - `operation` → `opId`
            
        - `workload_days`
            
        - EnvConfig 側の `opdef` と突き合わせて整合性を確認。
            
        - 許容時間が `[4]` の場合は基準時間 4、それ以外は 8 として、  
            `requiredHours = workload_days * baseline` を計算。
            
        - これらを元に `TaskWindow` を作成。
            
3. `assignment_list` から**固定アサイン**を読み込み:
    
    - 各行について:
        
        - `plan_flexibility`
            
        - `operation_task`（例: `e16p4o1`）からモジュールと opId を抽出。
            
        - `worker`（wid）、`start_date` / `end_date`、`work_date_list` または `work_date_lsit` を読み込み。
            
        - 日ごとの時間数を `hoursByDay` に集計。
            
    - `plan_flexibility == "fixed"` の場合:
        
        - `(module|opId)` の `fixedHoursByKey` に工数を加算。
            
        - `FixedAssign` を作成し、リストに追加。
            
        - `(module, phaseNum)` ごとに、固定アサインの最終日インデックスを記録し、後続フェーズの開始ウィンドウを押し出す際に使用。
            
4. 固定アサインに基づくウィンドウ調整:
    
    - 各 `TaskWindow w` について:
        
        - 前フェーズ `phaseNum - 1` の固定アサイン最終日を見て、  
            `w.startDayId = max(w.startDayId, endPrev + 1)` として開始日を後ろにずらします。
            

---

## 10. エンティティの構築（シングルパス）

`buildEntitiesSinglePass`（メソッド本体は原文のまま）

### 10.1 ウィンドウと fab のマッピング

- `keyToWin[module|opId] = TaskWindow`
    
- `moduleWins[module] = List<TaskWindow>`
    
- 各 `FixedAssign fa` に対して:
    
    - モジュールに対応する `factory` を設定します。
        

### 10.2 BlockDecision エンティティの作成

各 `TaskWindow w` について:

- 基準時間 `baseline` を決定:
    
    - `allowed` が `[4]` の場合は 4、それ以外は 8。
        
- `totalReq = workloadDays * baseline`
    
- `fixed = fixedHoursByKey[module|opId]` を取得。
    
- `req = max(0, totalReq - fixed)` を計算。
    
    - 固定アサイン分を差し引きます。
        
- `req == 0` の場合はブロックを作成しません。
    
- それ以外の場合は `BlockDecision` を作成し、  
    モジュール・fab・フェーズ・opId・ウィンドウ・必要工数・許容時間・最小/最大ヘッド数を設定して `blocks` に追加します。
    

### 10.3 CrewSeat の作成

各 `TaskWindow w` について、その `maxHeads` に応じて席を作成します（コードは原文のまま）。

- 各ブロックに対して `maxHeads` 個の `CrewSeat` を作成。
    
- `seatIndex = 0` の席は `needManager = true` に設定（管理者席）。
    
- 初期状態ではすべて `UNASSIGNED` 従業員を割り当てておきます。
    

### 10.4 固定席用マッピング

`moduleToFactory`, `moduleOpToPhase`, `moduleOpToPhaseNum` などのマップを作成し、`FixedAssign` の補完に使用します。

### 10.5 固定アサイン用のピン留め席の作成

各 `FixedAssign fa` について:

- モジュールに対応する fab を決定。
    
- `CrewSeat` を 1 席作成し、`blockId = -1`（通常ブロックとは独立）として扱う。
    
- フェーズ情報などをモジュール側の `TaskWindow` から補完。
    
- `pinned = true`, `pinnedWid = fa.wid` を設定。
    
- `hoursByDay` から最小/最大日インデックスと最大時間を取得し、  
    `pinnedStart`, `pinnedDays`, `pinnedHours` を設定。
    
- Solver 用には、`env.byWid` から該当従業員を探し、見つからない場合は `UNASSIGNED` を使います。
    

---

## 11. 席ごとの候補従業員（Seat Candidates）の設定

`fillSeatCandidatesSinglePass`（メソッド本体は原文のまま）

- `byBlock` マップを作り、`blockId` → `BlockDecision` を引けるようにします。
    
- 個人休は `CAL.workerOffByWid` から取得します。
    

各 `CrewSeat s` ごとに:

1. **ピン留め席**
    
    - 候補はそのピン留め従業員のみ（見つからなければ空リスト）。
        
2. **ピン留めでない席**
    
    - 推定開始日と期間を決定:
        
        - `BlockDecision b` があれば `b.startDay`, `b.days` を優先。
            
        - `null` の場合はウィンドウの `windowStart`, `windowEnd` から最大長を使う。
            
    - 全従業員（id ≠ 0）をループして:
        
        - スキルチェック:
            
            - `skill >= 1` の従業員のみ許可。
                
        - 稼働可能日チェック:
            
            - `[estStart .. estStart+estDays-1]` をループ。
                
            - `isWorkingDay` で fab が稼働している日だけを対象にし、  
                その日が個人休に入っていれば候補から除外。
                
    - 管理者席の場合:
        
        - `isManager == true` の従業員のみ残す。
            
        - 1 人もいなければ `IllegalStateException` を投げます（管理者なしでそのモジュールを進められないため）。
            

このステップで、現実的に不可能な組み合わせを事前に除外し、探索空間を縮小しています。

---

## 12. ソルバー設定と実行

### 12.1 `buildSolverFactory` ヘルパー

（メソッド本体は原文のまま）

- XML を使わずに、コード上で `SolverFactory` を構築します。
    
- パラメータ:
    
    - `solutionClass`: ソリューションクラス（ここでは `SinglePassPlan`）
        
    - `entityClasses`: エンティティクラス配列
        
    - `providerClass`: 制約プロバイダ
        
    - `bestScoreLimit`: 目標スコア（例: `"0hard/*medium/*soft"`）
        
    - `spentMinutes`: 最大実行時間（分）
        
    - `unimprovedSeconds`: 無改善時間（秒）で打ち切り
        

> **乱数シードについて**  
> コードでは乱数シードを指定していません。Timefold のデフォルト乱数生成器が使われます。  
> 実行結果を完全に再現したい場合は、`SolverConfig` に乱数シード関連の設定を追加する必要があります。

### 12.2 ステージ1とステージ2

`solveFromYaml` の中で次のように実行します。

- `TARGET_HOURS_PER_EMP` の計算:
    
    - 実従業員数 = `env.employees.size() - 1`（UNASSIGNED を除外）
        
    - `totalReq` = `sch.requiredByKey` の合計
        
    - `TARGET_HOURS_PER_EMP = totalReq / realEmp`
        

`SinglePassPlan p` に `days`, `employees`, `blocks`, `seats` を詰めます。

#### ステージ1

（ステージ1設定と実行のコードは原文のまま）

- 終了条件:
    
    - `bestScoreLimit = "0hard/*medium/*soft"`（ハードとミディアムが 0 になるまで）
        
    - 最大 90 分
        
    - 300 秒無改善で打ち切り
        

#### ステージ2（仕上げ）

（ステージ2設定と実行のコードは原文のまま）

- `bestScoreLimit` なし（ソフトスコア改善に専念）。
    
- 最大 60 分。
    

#### スコア説明

（`SolutionManager` を使った `ScoreExplanation` のコードは原文のまま）

- 各制約名ごとのスコア寄与を `constraintName = score` 形式で出力します。
    

#### ブロックごとの早期開始ペナルティ表示

（`=== Java earlier-start per block ===` の出力部分は原文のまま）

- 各ブロックについて `startDay` と `penalty` を出力し、Python 側の実装との比較に使っています。
    

---

## 13. ExportSchedule.java – YAML への書き戻し

このクラスは、最終的な `SinglePassPlan` を既存の `Schedule.yaml` にマージし、固定行を保持したままフレキシブル行を上書き保存します。

### 13.1 メソッドシグネチャ

`public static void overwriteScheduleWithAssignments( EmployeeSchedule.SinglePassPlan plan, LocalDate planStart, String schedPath, String envPath) throws IOException`

### 13.2 YAML の読み込みと `schedule` ルート取得

（コードは原文のまま）

- `schedule:` キーの有無に関わらず動作するように、  
    ネストスタイルとフラットスタイルの両方をサポートしています。
    

### 13.3 カレンダーの確保（必要に応じて）

（コードは原文のまま）

- `CAL` がまだ初期化されていない場合、`EnvConfig.yaml` と `plan_range` からカレンダーを再構築します。
    

### 13.4 `(module, op)` → `operation_task_id` のマッピング

（コードは原文のまま）

- YAML の `assignment_list` では `operation_task` ID を使っているので、  
    `(module, operation)` から `operation_task.id` を引けるようにマッピングを作ります。
    

### 13.5 固定行の保持

（コードは原文のまま）

- 元の `assignment_list` から `plan_flexibility = fixed` の行だけを `preservedFixed` にコピーし、そのまま残します。
    

### 13.6 固定アサインのマスク

（コードは原文のまま）

- `(worker, operation_task)` ごとに、すでに固定アサインされている日インデックスの集合を `fixedMask` として作成します。
    
- これにより、ソルバー結果から生成するフレキシブル行で、固定分と重複する日を削除できます。
    

### 13.7 ブロックのインデックスと新しいフレキシブル行の生成

（コードは原文のまま）

重要なポイント:

- ピン留め席（固定アサイン由来）は**書き出さない**:
    
    - もともとの固定行として `Schedule.yaml` に残っているため、ここで再出力する必要はありません。
        
- `byDay` に従業員ごとの日別時間を計算する際:
    
    - ソルバーで決定された `b.hours` があればそれを使用。
        
    - なければ `b.chosenHours()` を使います。
        
- `fixedMask` に含まれる日付はフレキシブル行から削除し、  
    結果として `byDay` が空になった場合はその席については何も出力しません。
    
- 最終的に `work_date_list` と `start_date`, `end_date`, `plan_flexibility="Flexible"` を持つ YAML 行を作成します。
    

### 13.8 assignment_list のマージと書き戻し

（コードは原文のまま）

- 最終的な `assignment_list` は:
    
    - 先に `preservedFixed`（固定行）
        
    - その後に `newFlex`（新しいフレキシブル行）
        
- これを `sched.put("assignment_list", merged)` として設定し、YAML 全体を再度ファイルに書き出します。
    

最後に、`Overwrote ... new flexible rows=... preserved fixed rows=...` とログ出力します。

---

## 14. 公開 API と main()

### 14.1 RunResult

（クラス定義は原文のまま）

- ソルバー結果 (`SinglePassPlan`) と `planStart` をまとめて返すための小さなコンテナクラスです。
    

### 14.2 `solveFromYaml`

（メソッド本体は原文のまま）

- `EnvConfig.yaml` と `Schedule.yaml` を読み込み、
    
    - `ParsedEnv` と `ParsedSchedule` を構築
        
    - `buildCalendars` でカレンダーを構築
        
    - `buildEntitiesSinglePass` でブロック & 席を生成
        
    - ステージ1・ステージ2で解を求める
        
- `RunResult` に最終解と `planStart` を詰めて返却します。
    

### 14.3 `main` メソッド

（メソッド本体は原文のまま）

- コマンドラインから実行されるエントリポイントで、
    
    - `args[0]` に `EnvConfig.yaml`（省略時は `"EnvConfig.yaml"`）
        
    - `args[1]` に `Schedule.yaml`（省略時は `"Schedule.yaml"`）  
        を使います。
        
- `solveFromYaml` を呼び出し、得られたプランを `ExportSchedule.overwriteScheduleWithAssignments` で `Schedule.yaml` に書き戻し、最後に `"Done."` を出力します。
    

---

## 15. コードレビュー用の説明まとめ

上司やチームメンバーに説明するときは、次のような流れで話すと分かりやすいです。

1. **入力と出力**
    
    - 入力:
        
        - `EnvConfig.yaml`: 環境設定、従業員リスト、ワークフロー、キャパシティ、利用不可日など
            
        - `Schedule.yaml`: 初期タスクと固定アサイン
            
    - 出力:
        
        - 更新された `Schedule.yaml` の `assignment_list`：  
            固定行はそのまま残し、新しいフレキシブル行を追加したもの。
            
2. **ドメインモデル**
    
    - `BlockDecision`:
        
        - 各オペレーションブロックの「いつ・何日間・何時間/日」で作業するかを表す。
            
    - `CrewSeat`:
        
        - 各ブロック内の「どの席に誰が入るか」を表す。
            
        - 固定アサインはピン留め席として表現。
            
    - `SinglePassPlan`:
        
        - これらのブロックと席をまとめた Timefold のソリューション。
            
3. **カレンダーと利用不可日**
    
    - EnvConfig から fab/地域/顧客/会社/個人の休業日や週末を読み込み、  
        `isWorkingDay` で「実際に作業可能な日」だけを数える仕組み。
        
    - 個人休・fab 休などはハード制約で守られる。
        
4. **制約**
    
    - ハード制約:
        
        - ブロックの終了日はウィンドウ内 (`endWithinWindow`)
            
        - フェーズ順序（前フェーズが終わってから次フェーズを開始）
            
        - 必要工数を満たす（または 1 日分以内の超過まで許容）
            
        - 日ごとのヘッドキャパシティを超えない
            
        - 個人の利用不可日を侵害しない
            
        - 1 日に複数 fab に行かない
            
        - 1 日の総工数は 12 時間以内
            
        - ピン留め席は必ず指定従業員を使う
            
    - ソフト制約:
        
        - なるべく時間数の小さいブロックを選ぶ
            
        - なるべく早くウィンドウ内で開始する
            
        - 従業員間で総工数をバランスさせる
            
5. **固定 vs フレキシブル**
    
    - `Schedule.yaml` の固定アサインは:
        
        - Solver 内ではピン留め席としてモデリングされる。
            
        - その分の工数は `requiredHours` から差し引かれる。
            
        - Export 時には元の固定行をそのまま残し、Solver が決めたフレキシブル行だけを追加出力する。
            
6. **ソルバーの流れ**
    
    - ステージ1:
        
        - ハード制約を満たす解を目標に 90 分以内で探索。
            
    - ステージ2:
        
        - ソフトスコアを改善するために最大 60 分追加で探索。
            
    - 最後に制約ごとのスコア内訳を出力することで、  
        「どの制約をどれくらい満たしているか」を確認できる。
        
7. **ビルド & 実行**
    
    - ビルド:
        
        - `mvn -DskipTests clean package`
            
    - 実行:
        
        - `mvn -q exec:java -D"exec.args=EnvConfig.yaml Schedule.yaml"`
            
    - 実行後は `Schedule.yaml` の `assignment_list` が Solver の結果で更新される。