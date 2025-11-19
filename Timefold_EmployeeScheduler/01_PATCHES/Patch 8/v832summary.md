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
