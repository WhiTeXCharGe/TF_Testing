package com.yourorg.scheduler;

import java.io.*;
import java.nio.file.*;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.stream.Collectors;

import ai.timefold.solver.core.api.domain.entity.PlanningEntity;
import ai.timefold.solver.core.api.domain.lookup.PlanningId;
import ai.timefold.solver.core.api.domain.solution.PlanningEntityCollectionProperty;
import ai.timefold.solver.core.api.domain.solution.PlanningSolution;
import ai.timefold.solver.core.api.domain.solution.ProblemFactCollectionProperty;
import ai.timefold.solver.core.api.domain.valuerange.ValueRangeProvider;
import ai.timefold.solver.core.api.domain.variable.PlanningVariable;
import ai.timefold.solver.core.api.domain.solution.PlanningScore;
import ai.timefold.solver.core.api.score.Score;
import ai.timefold.solver.core.api.score.buildin.hardmediumsoft.HardMediumSoftScore;
import ai.timefold.solver.core.api.score.stream.Constraint;
import ai.timefold.solver.core.api.score.stream.ConstraintCollectors;
import ai.timefold.solver.core.api.score.stream.ConstraintFactory;
import ai.timefold.solver.core.api.score.stream.ConstraintProvider;
import ai.timefold.solver.core.api.score.stream.Joiners;
import ai.timefold.solver.core.api.solver.Solver;
import ai.timefold.solver.core.api.solver.SolverFactory;
import ai.timefold.solver.core.config.score.director.ScoreDirectorFactoryConfig;
import ai.timefold.solver.core.config.solver.SolverConfig;
import ai.timefold.solver.core.config.solver.termination.TerminationConfig;

import org.yaml.snakeyaml.DumperOptions;
import org.yaml.snakeyaml.Yaml;

/**
 * v822
 *
 * Pass 1: SELECTIVE RAMP LOOP (from v812) with per-iteration snapshots.
 *         - Production uses workingDaysCount(...) so weekends + fab/region/customer OFF are skipped.
 *         - Head capacity & stacking constraints also ignore non-working days.
 *         - Demand subtracts FIXED hours parsed from Schedule.yaml.
 *
 * Fixed assignments:
 *         - Parsed from Schedule.assignment_list (plan_flexibility: fixed).
 *         - Built as PINNED CrewSeat(s) (exact day/hours, assigned to given worker).
 *         - PINNED seat-days also skip non-working days.
 *
 * Pass 2: assignment with hard constraints:
 *         - assigned & skill, one-factory/day, 12h/day cap, at least one manager per block,
 *         - worker personal OFF, and respect pinned seats.
 *         Soft: company pairing, skill variety, avg skill, total hours balancing.
 *
 * Snapshots: include both flexible and pinned seats; pinned seats keep their worker.
 */
public class EmployeeSchedule {

    // ---------------- YAML I/O ----------------

    @SuppressWarnings("unchecked")
    static Map<String, Object> loadYaml(String path) throws IOException {
        try (InputStream in = Files.newInputStream(Paths.get(path))) {
            return new Yaml().load(in);
        }
    }

    static void saveYaml(String path, Map<String, Object> root) throws IOException {
        DumperOptions opt = new DumperOptions();
        opt.setDefaultFlowStyle(DumperOptions.FlowStyle.BLOCK);
        opt.setPrettyFlow(true);
        Yaml yaml = new Yaml(opt);
        try (Writer w = Files.newBufferedWriter(Paths.get(path))) {
            yaml.dump(root, w);
        }
    }

    static String safeStr(Object o) { return o == null ? "" : String.valueOf(o); }
    static int parseInt(Object o, int def) {
        if (o == null) return def;
        try { return Integer.parseInt(String.valueOf(o)); } catch (Exception e) { return def; }
    }
    static int phaseNumFromId(String pid) {
        if (pid == null) return 0;
        try { return Integer.parseInt(pid.trim().toLowerCase().replace("p","")); }
        catch (Exception e) { return 0; }
    }

    static final DateTimeFormatter DF = DateTimeFormatter.ofPattern("yyyy/MM/dd");

    // ---------------- Domain ----------------

    public static class DaySlot {
        @PlanningId public int id;
        public LocalDate date;
        public DaySlot() {}
        public DaySlot(int id, LocalDate d) { this.id = id; this.date = d; }
    }

    public static class EmployeeFact {
        @PlanningId public int id;     // 0 = UNASSIGNED
        public String wid;
        public String name;
        public Map<String,Integer> skills = new HashMap<>();
        public boolean isManager;
        public String workerCompany;

        public EmployeeFact() {}
        public EmployeeFact(int id, String wid, String name, Map<String,Integer> skills, boolean isManager, String company) {
            this.id = id; this.wid = wid; this.name = name;
            if (skills != null) this.skills = new HashMap<>(skills);
            this.isManager = isManager; this.workerCompany = company;
        }
    }

    public static class TaskWindow {
        public String module;
        public String factory;
        public String phaseId;
        public int phaseNum;
        public String opId;
        public int startDayId;
        public int endDayId;
        public List<Integer> allowed;
        public int minHeads;
        public int maxHeads;
        public int workloadDays;

        public List<Integer> fullAllowedSorted() {
            return (allowed == null || allowed.isEmpty())
                    ? List.of(8)
                    : allowed.stream().sorted().collect(Collectors.toList());
        }
    }

    @PlanningEntity
    public static class BlockDecision {
        @PlanningId public int id;

        public String module;
        public String factory;
        public String phaseId;
        public int phaseNum;
        public String opId;
        public int windowStart;
        public int windowEnd;
        public int requiredHours; // already subtracts fixed
        public List<Integer> allowed;
        public int minHeads;
        public int maxHeads;

        @PlanningVariable(valueRangeProviderRefs = "vrDayIds")
        public Integer startDay;
        @PlanningVariable(valueRangeProviderRefs = "vrHeadOptions")
        public Integer heads;
        @PlanningVariable(valueRangeProviderRefs = "vrDayCountOptions")
        public Integer days;

        public int seedHours = 8;
        public BlockDecision() {}
    }

    @PlanningSolution
    public static class Pass1Plan {
        @ValueRangeProvider(id = "vrDayIds")
        @ProblemFactCollectionProperty
        public List<Integer> dayIds;

        @ValueRangeProvider(id = "vrHeadOptions")
        @ProblemFactCollectionProperty
        public List<Integer> headOptions;

        @ValueRangeProvider(id = "vrDayCountOptions")
        @ProblemFactCollectionProperty
        public List<Integer> dayCountOptions;

        @ProblemFactCollectionProperty
        public List<DaySlot> daySlots;

        @PlanningEntityCollectionProperty
        public List<BlockDecision> blocks;

        @PlanningScore
        private HardMediumSoftScore score;
        public HardMediumSoftScore getScore() { return score; }
        public void setScore(HardMediumSoftScore s) { this.score = s; }
        public Pass1Plan() {}
    }

    public static class SeatDay {
        public String seatKey;
        public DaySlot day;
        public int hours;
        public String factory;
        public SeatDay() {}
        public SeatDay(String seatKey, DaySlot day, int hours, String factory) {
            this.seatKey = seatKey; this.day = day; this.hours = hours; this.factory = factory;
        }
    }

    @PlanningEntity
    public static class CrewSeat {
        @PlanningId public int id;

        public String module;
        public String factory;
        public String phaseId;
        public int phaseNum;
        public String opId;
        public int startDayId;
        public int days;
        public int hours;
        public int seatIndex;
        public String seatKey;
        public int blockId;

        @PlanningVariable
        public EmployeeFact employee;

        // pinned (for fixed rows)
        public boolean pinned = false;
        public String pinnedWid = null;

        public CrewSeat() {}
    }

    @PlanningSolution
    public static class Pass2Plan {
        @ValueRangeProvider @ProblemFactCollectionProperty
        public List<DaySlot> days;

        @ValueRangeProvider @ProblemFactCollectionProperty
        public List<EmployeeFact> employees;

        @ProblemFactCollectionProperty
        public List<SeatDay> seatDays;

        @PlanningEntityCollectionProperty
        public List<CrewSeat> seats;

        @PlanningScore
        private HardMediumSoftScore score;
        public HardMediumSoftScore getScore() { return score; }
        public void setScore(HardMediumSoftScore s) { this.score = s; }
        public Pass2Plan() {}
    }

    // ---------------- Globals ----------------

    static final int DAILY_CAP = 12;
    static double TARGET_HOURS_PER_EMP = 0.0;

    static final Map<String,Integer> OP_CAPACITY = new HashMap<>();
    static final Map<String,Double>  OP_AVG_SKILL = new HashMap<>();

    // snapshot context
    static class SnapshotCfg {
        LocalDate planStart;
        String schedulePath; // original schedule path; snapshots are siblings
        List<EmployeeFact> employees;
        // include env/schedule data so snapshots include pinned seats correctly
        ParsedEnv env;
        ParsedSchedule sch;
        List<TaskWindow> windows;
        List<DaySlot> daySlots;
        String envPath; 
    }

    static boolean isUnassigned(EmployeeFact e) { return e == null || e.id == 0; }
    static int skill(EmployeeFact e, String opId) { return (e == null) ? 0 : e.skills.getOrDefault(opId, 0); }
    static boolean isManager(EmployeeFact e) { return e != null && e.isManager; }
    static String company(EmployeeFact e) { return e == null ? "" : (e.workerCompany == null ? "" : e.workerCompany); }
    static double avgSkill(String opId) { return OP_AVG_SKILL.getOrDefault(opId, 3.0); }

    static int produced(BlockDecision b) {
        int h = autoHours(b);
        int H = Math.max(1, b.heads == null ? 1 : b.heads);
        int D = (b.startDay == null || b.days == null) ? 0 : workingDaysCount(b.startDay, b.days, b.factory);
        return H * h * Math.max(0, D);
    }

    static int autoHours(BlockDecision b) {
        List<Integer> allowed = (b.allowed == null || b.allowed.isEmpty())
                ? List.of(8) : b.allowed.stream().sorted().collect(Collectors.toList());
        int H = Math.max(1, b.heads == null ? 1 : b.heads);
        int D = (b.startDay == null || b.days == null) ? 0 : workingDaysCount(b.startDay, b.days, b.factory);
        int R = b.requiredHours;

        List<int[]> feasible = new ArrayList<>();
        for (int h : allowed) {
            int prod = H * h * D;
            int over = prod - R;
            if (prod >= R && over <= H * h) feasible.add(new int[]{h, Math.abs(h - 8), h});
        }
        if (!feasible.isEmpty()) {
            feasible.sort(Comparator.<int[]>comparingInt(a -> a[1]).thenComparingInt(a -> a[2]));
            return feasible.get(0)[0];
        }
        int bestH = allowed.get(0);
        int[] bestKey = null; // [mode, penalty, |h-8|, h]
        for (int h : allowed) {
            int prod = H * h * D;
            int[] key;
            if (prod < R) {
                key = new int[]{0, R - prod, Math.abs(h - 8), h};
            } else {
                int extra = Math.max(0, (prod - R) - H * h);
                key = new int[]{1, extra, Math.abs(h - 8), h};
            }
            if (bestKey == null || compareTuple(key, bestKey) < 0) { bestKey = key; bestH = h; }
        }
        return bestH;
    }

    static int compareTuple(int[] a, int[] b) {
        for (int i = 0; i < Math.min(a.length, b.length); i++) {
            int c = Integer.compare(a[i], b[i]); if (c != 0) return c;
        }
        return Integer.compare(a.length, b.length);
    }

    // ---- Calendar / blackout support ----
    static class Calendars {
        Set<Integer> weekends = new HashSet<>();
        Map<String, Set<Integer>> fabOff = new HashMap<>();
        Map<String, Set<Integer>> regionOff = new HashMap<>();
        Map<String, Set<Integer>> customerOff = new HashMap<>();
        Map<String, Set<Integer>> workerCompanyOff = new HashMap<>();
        Map<String, String> fabToRegion = new HashMap<>();
        Map<String, String> fabToCustomer = new HashMap<>();
        Map<String, Set<Integer>> workerOffByWid = new HashMap<>();
    }
    static Calendars CAL = new Calendars();

    private static final java.time.format.DateTimeFormatter CLOCK =
            java.time.format.DateTimeFormatter.ofPattern("HH:mm:ss");

    private static String nowClock() { return java.time.LocalTime.now().format(CLOCK); }
    private static String fmt(java.time.Duration d) {
        long h = d.toHours();
        long m = d.toMinutesPart();
        long s = d.toSecondsPart();
        long ms = d.toMillisPart();
        return String.format("%02d:%02d:%02d.%03d", h, m, s, ms);
    }

    // Turn yyyy/MM/dd strings into day indexes
    static Integer dayIdFromDate(LocalDate planStart, String ymd) {
        LocalDate d = LocalDate.parse(ymd.replace("-", "/"), DF);
        return (int) (d.toEpochDay() - planStart.toEpochDay());
    }

    // Build calendars from EnvConfig, aligned to the plan date range
    @SuppressWarnings("unchecked")
    public static void buildCalendars(String envPath, LocalDate planStart, LocalDate planEnd) throws IOException {
        CAL = new Calendars();

        // Weekends for the whole horizon
        int horizon = (int) (planEnd.toEpochDay() - planStart.toEpochDay()) + 1;
        for (int i = 0; i < horizon; i++) {
            LocalDate d = planStart.plusDays(i);
            switch (d.getDayOfWeek()) {
                case SATURDAY:
                case SUNDAY:
                    CAL.weekends.add(i);
                    break;
                default: /* working day */
            }
        }

        Map<String,Object> root;
        try (InputStream in = Files.newInputStream(Paths.get(envPath))) {
            root = new Yaml().load(in);
        }
        Map<String,Object> env = (Map<String,Object>) root.getOrDefault("environment", root);

        // fab_list -> fab off & fab->region / fab->customer
        List<Map<String,Object>> fabs = (List<Map<String,Object>>) env.getOrDefault("fab_list", List.of());
        for (Map<String,Object> f : fabs) {
            String fid = String.valueOf(f.get("id"));
            String rid = String.valueOf(f.get("region"));
            String cid = String.valueOf(f.get("customer_company"));
            CAL.fabToRegion.put(fid, rid);
            CAL.fabToCustomer.put(fid, cid);
            Set<Integer> off = new HashSet<>();
            List<Object> dates = (List<Object>) f.getOrDefault("unavailable_dates", List.of());
            for (Object o : dates) {
                Integer did = dayIdFromDate(planStart, String.valueOf(o));
                if (did != null) off.add(did);
            }
            CAL.fabOff.put(fid, off);
        }

        // region_list -> region off
        List<Map<String,Object>> regions = (List<Map<String,Object>>) env.getOrDefault("region_list", List.of());
        for (Map<String,Object> r : regions) {
            String rid = String.valueOf(r.get("id"));
            Set<Integer> off = new HashSet<>();
            List<Object> dates = (List<Object>) r.getOrDefault("unavailable_dates", List.of());
            for (Object o : dates) {
                Integer did = dayIdFromDate(planStart, String.valueOf(o));
                if (did != null) off.add(did);
            }
            CAL.regionOff.put(rid, off);
        }

        // customer_company_list -> customer off
        List<Map<String,Object>> custs = (List<Map<String,Object>>) env.getOrDefault("customer_company_list", List.of());
        for (Map<String,Object> c : custs) {
            String cid = String.valueOf(c.get("id"));
            Set<Integer> off = new HashSet<>();
            List<Object> dates = (List<Object>) c.getOrDefault("unavailable_dates", List.of());
            for (Object o : dates) {
                Integer did = dayIdFromDate(planStart, String.valueOf(o));
                if (did != null) off.add(did);
            }
            CAL.customerOff.put(cid, off);
        }

        // worker_company_list -> worker company off (currently not enforced in p2 hard rule)
        List<Map<String,Object>> wcomps = (List<Map<String,Object>>) env.getOrDefault("worker_company_list", List.of());
        for (Map<String,Object> wc : wcomps) {
            String cid = String.valueOf(wc.get("id"));
            Set<Integer> off = new HashSet<>();
            List<Object> dates = (List<Object>) wc.getOrDefault("unavailable_dates", List.of());
            for (Object o : dates) {
                Integer did = dayIdFromDate(planStart, String.valueOf(o));
                if (did != null) off.add(did);
            }
            CAL.workerCompanyOff.put(cid, off);
        }

        // worker_list -> individual worker off (keyed by worker "id"/wid)
        List<Map<String,Object>> workers = (List<Map<String,Object>>) env.getOrDefault("worker_list", List.of());
        for (Map<String,Object> w : workers) {
            String wid = String.valueOf(w.get("id"));
            Set<Integer> off = new HashSet<>();
            List<Object> dates = (List<Object>) w.getOrDefault("unavailable_dates", List.of());
            for (Object o : dates) {
                Integer did = dayIdFromDate(planStart, String.valueOf(o));
                if (did != null) off.add(did);
            }
            CAL.workerOffByWid.put(wid, off);
        }
    }

    // Is this a *working* day for the given fab?
    static boolean isWorkingDay(int dayId, String fabId) {
        if (CAL.weekends.contains(dayId)) return false;
        if (fabId == null) return true;
        if (CAL.fabOff.getOrDefault(fabId, Set.of()).contains(dayId)) return false;
        String rid = CAL.fabToRegion.get(fabId);
        if (rid != null && CAL.regionOff.getOrDefault(rid, Set.of()).contains(dayId)) return false;
        String cid = CAL.fabToCustomer.get(fabId);
        if (cid != null && CAL.customerOff.getOrDefault(cid, Set.of()).contains(dayId)) return false;
        return true;
    }

    // Count working days inside a box
    static int workingDaysCount(int startDay, int dayCount, String fabId) {
        if (startDay < 0 || dayCount == 0) return 0;
        int end = startDay + dayCount - 1;
        int n = 0;
        for (int d = startDay; d <= end; d++) if (isWorkingDay(d, fabId)) n++;
        return n;
    }

    // ---------------- Constraints ----------------

    // Pass 1
    public static class Pass1Constraints implements ConstraintProvider {
        // weights
        static final int PREF_HOURS_WEIGHT = 1000;
        static final int SMALLER_HOURS_W   = 100;
        static final int SMALLER_HEADS_W   = 10;
        static final int FEWER_DAYS_W      = 1;
        static final int EARLIER_START_W   = 1;
        static final int STACK_PAIR_WEIGHT = 2;

        @Override public Constraint[] defineConstraints(ConstraintFactory f) {
            return new Constraint[] {
                withinWindow(f),
                daysWithinWindowLen(f),
                hoursValueAllowed(f),
                headsInMinMax(f),
                noUnderfill(f),
                overfillAtMostOneDay(f),
                phaseOrder(f),
                dailyHeadCapacityByOp(f),
                penalizeStackByOp(f),

                preferHoursNear8(f),
                preferSmallerHours(f),
                minimizeHeads(f),
                minimizeDays(f),
                preferEarlierStart(f)
            };
        }

        Constraint withinWindow(ConstraintFactory f) {
            return f.forEach(BlockDecision.class)
                .filter(b -> {
                    if (b.days == null || b.startDay == null) return true;
                    int end = b.startDay + b.days - 1;
                    return !(b.startDay >= b.windowStart && end <= b.windowEnd && b.days >= 1);
                })
                .penalize(HardMediumSoftScore.ONE_HARD)
                .asConstraint("p1-within-window");
        }

        Constraint daysWithinWindowLen(ConstraintFactory f) {
            return f.forEach(BlockDecision.class)
                .filter(b -> {
                    if (b.days == null) return false;
                    int maxLen = b.windowEnd - b.windowStart + 1;
                    return b.days > maxLen;
                })
                .penalize(HardMediumSoftScore.ONE_HARD,
                        b -> b.days - (b.windowEnd - b.windowStart + 1))
                .asConstraint("p1-days-within-window-length");
        }

        Constraint hoursValueAllowed(ConstraintFactory f) {
            return f.forEach(BlockDecision.class)
                .filter(b -> {
                    int h = autoHours(b);
                    return b.allowed == null || b.allowed.isEmpty() || !b.allowed.contains(h);
                })
                .penalize(HardMediumSoftScore.ONE_HARD)
                .asConstraint("p1-hours-in-allowed");
        }

        Constraint headsInMinMax(ConstraintFactory f) {
            return f.forEach(BlockDecision.class)
                .filter(b -> b.heads == null || b.heads < b.minHeads || b.heads > b.maxHeads)
                .penalize(HardMediumSoftScore.ONE_HARD)
                .asConstraint("p1-heads-in-minmax");
        }

        Constraint noUnderfill(ConstraintFactory f) {
            return f.forEach(BlockDecision.class)
                .filter(b -> produced(b) < b.requiredHours)
                .penalize(HardMediumSoftScore.ONE_HARD, b -> b.requiredHours - produced(b))
                .asConstraint("p1-no-underfill");
        }

        Constraint overfillAtMostOneDay(ConstraintFactory f) {
            return f.forEach(BlockDecision.class)
                .filter(b -> {
                    int prod = produced(b);
                    int over = prod - b.requiredHours;
                    int h = autoHours(b);
                    int H = Math.max(1, b.heads == null ? 1 : b.heads);
                    return over > H * h;
                })
                .penalize(HardMediumSoftScore.ONE_HARD, b -> {
                    int prod = produced(b);
                    int over = prod - b.requiredHours;
                    int h = autoHours(b);
                    int H = Math.max(1, b.heads == null ? 1 : b.heads);
                    return over - H * h;
                })
                .asConstraint("p1-overfill-at-most-one-day");
        }

        Constraint phaseOrder(ConstraintFactory f) {
            return f.forEach(BlockDecision.class)
                .join(f.forEach(BlockDecision.class),
                        Joiners.equal((BlockDecision a) -> a.module, (BlockDecision b) -> b.module),
                        Joiners.equal((BlockDecision a) -> a.phaseNum + 1, (BlockDecision b) -> b.phaseNum))
                .filter((a,b) -> a.startDay != null && a.days != null && b.startDay != null
                        && (a.startDay + a.days - 1) >= b.startDay)
                .penalize(HardMediumSoftScore.ONE_HARD, (a,b) -> (a.startDay + a.days - 1) - b.startDay + 1)
                .asConstraint("p1-phase-order");
        }

        Constraint dailyHeadCapacityByOp(ConstraintFactory f) {
            return f.forEach(DaySlot.class)
                .join(f.forEach(BlockDecision.class),
                    Joiners.filtering((DaySlot d, BlockDecision b) ->
                        b.startDay != null && b.days != null &&
                        b.startDay <= d.id && d.id <= (b.startDay + b.days - 1) &&
                        isWorkingDay(d.id, b.factory)))
                .groupBy((DaySlot d, BlockDecision b) -> d.id,
                         (DaySlot d, BlockDecision b) -> b.opId,
                         ConstraintCollectors.sum((DaySlot d, BlockDecision b) -> b.heads == null ? 0 : b.heads))
                .filter((dayId, opId, totalHeads) -> totalHeads >
                        OP_CAPACITY.getOrDefault(opId, Integer.MAX_VALUE))
                .penalize(HardMediumSoftScore.ONE_HARD, (dayId, opId, totalHeads) ->
                        totalHeads - OP_CAPACITY.getOrDefault(opId, Integer.MAX_VALUE))
                .asConstraint("p1-daily-head-capacity-by-op");
        }

        Constraint penalizeStackByOp(ConstraintFactory f) {
            return f.forEach(DaySlot.class)
                .join(f.forEach(BlockDecision.class),
                    Joiners.filtering((DaySlot d, BlockDecision b) ->
                        b.startDay != null && b.days != null &&
                        b.startDay <= d.id && d.id <= (b.startDay + b.days - 1) &&
                        isWorkingDay(d.id, b.factory)))
                .groupBy(
                    (DaySlot d, BlockDecision b) -> d.id,
                    (DaySlot d, BlockDecision b) -> b.opId,
                    ConstraintCollectors.sum((DaySlot d, BlockDecision b) -> 1)
                )
                .filter((dayId, opId, n) -> n > 1)
                .penalize(HardMediumSoftScore.ONE_MEDIUM,
                    (dayId, opId, n) -> STACK_PAIR_WEIGHT * (n * (n - 1) / 2))
                .asConstraint("p1-med-penalize-stack-by-op");
        }

        Constraint preferHoursNear8(ConstraintFactory f) {
            return f.forEach(BlockDecision.class)
                .penalize(HardMediumSoftScore.ONE_SOFT, b -> PREF_HOURS_WEIGHT * Math.abs(autoHours(b) - 8))
                .asConstraint("p1-soft-prefer-hours-near-8");
        }
        Constraint preferSmallerHours(ConstraintFactory f) {
            return f.forEach(BlockDecision.class)
                .penalize(HardMediumSoftScore.ONE_SOFT, b -> SMALLER_HOURS_W * autoHours(b))
                .asConstraint("p1-soft-prefer-smaller-hours");
        }
        Constraint minimizeHeads(ConstraintFactory f) {
            return f.forEach(BlockDecision.class)
                .penalize(HardMediumSoftScore.ONE_SOFT, b -> b.heads == null ? 0 : SMALLER_HEADS_W * b.heads)
                .asConstraint("p1-soft-minimize-heads");
        }
        Constraint minimizeDays(ConstraintFactory f) {
            return f.forEach(BlockDecision.class)
                .penalize(HardMediumSoftScore.ONE_SOFT, b -> b.days == null ? 0 : FEWER_DAYS_W * b.days)
                .asConstraint("p1-soft-minimize-days");
        }
        Constraint preferEarlierStart(ConstraintFactory f) {
            return f.forEach(BlockDecision.class)
                .penalize(HardMediumSoftScore.ONE_SOFT, b -> b.startDay == null ? 0 : EARLIER_START_W * b.startDay)
                .asConstraint("p1-soft-prefer-earlier-start");
        }
    }

    // Pass 2
    public static class Pass2Constraints implements ConstraintProvider {
        static final int COMPANY_PAIR_W = 5;
        static final int SKILL_DIVERSITY_W = 3;
        static final int SKILL_AVG_W = 50;

        @Override public Constraint[] defineConstraints(ConstraintFactory f) {
            return new Constraint[] {
                assignedAndSkill(f),
                oneFactoryPerEmpDay(f),
                dailyCap12h(f),
                atLeastOneManagerPerBlock(f),
                employeeAvailableOnSeatDays(f),
                respectPinnedAssignments(f),

                softSameCompanyPairs(f),
                softEncourageSkillVariety(f),
                softBalanceBlockAvgSkill(f),
                softBalanceTotalHours(f)
            };
        }

        Constraint assignedAndSkill(ConstraintFactory f) {
            return f.forEach(CrewSeat.class)
                .filter(s -> isUnassigned(s.employee) || skill(s.employee, s.opId) < 1)
                .penalize(HardMediumSoftScore.ONE_HARD)
                .asConstraint("p2-assigned+eligible-skill");
        }

        Constraint oneFactoryPerEmpDay(ConstraintFactory f) {
            return f.forEach(SeatDay.class)
                .join(f.forEach(CrewSeat.class),
                        Joiners.equal((SeatDay sd) -> sd.seatKey, (CrewSeat cs) -> cs.seatKey))
                .filter((sd, cs) -> !isUnassigned(cs.employee))
                .groupBy((sd, cs) -> Arrays.asList(cs.employee.id, sd.day.id),
                        ConstraintCollectors.countDistinct((sd, cs) -> sd.factory))
                .filter((key, facCnt) -> facCnt > 1)
                .penalize(HardMediumSoftScore.ONE_HARD, (key, facCnt) -> facCnt - 1)
                .asConstraint("p2-one-factory-per-emp-day");
        }

        Constraint dailyCap12h(ConstraintFactory f) {
            return f.forEach(SeatDay.class)
                .join(f.forEach(CrewSeat.class),
                        Joiners.equal((SeatDay sd) -> sd.seatKey, (CrewSeat cs) -> cs.seatKey))
                .filter((sd, cs) -> !isUnassigned(cs.employee))
                .groupBy((sd, cs) -> Arrays.asList(cs.employee.id, sd.day.id),
                        ConstraintCollectors.sum((sd, cs) -> sd.hours))
                .filter((key, tot) -> tot > DAILY_CAP)
                .penalize(HardMediumSoftScore.ONE_HARD, (key, tot) -> tot - DAILY_CAP)
                .asConstraint("p2-daily-cap-12h");
        }

        Constraint atLeastOneManagerPerBlock(ConstraintFactory f) {
            return f.forEach(CrewSeat.class)
                .filter(s -> !isUnassigned(s.employee))
                .groupBy(s -> s.blockId, ConstraintCollectors.sum(s -> isManager(s.employee) ? 1 : 0))
                .filter((blockId, mgrCount) -> mgrCount < 1)
                .penalize(HardMediumSoftScore.ONE_HARD)
                .asConstraint("p2-at-least-one-manager-per-block");
        }

        Constraint employeeAvailableOnSeatDays(ConstraintFactory f) {
            return f.forEach(EmployeeSchedule.SeatDay.class)
                .join(f.forEach(EmployeeSchedule.CrewSeat.class),
                    Joiners.equal((EmployeeSchedule.SeatDay sd) -> sd.seatKey,
                                  (EmployeeSchedule.CrewSeat cs) -> cs.seatKey))
                .filter((sd, cs) -> {
                    if (cs == null || cs.employee == null) return false;
                    String wid = cs.employee.wid;
                    Set<Integer> personalOff = CAL.workerOffByWid.getOrDefault(wid, Set.of());
                    // Company-wide worker-company OFF is available but not enforced as hard here:
                    // String compId = company(cs.employee);
                    // Set<Integer> companyOff = CAL.workerCompanyOff.getOrDefault(compId, Set.of());
                    Set<Integer> companyOff = Set.of();
                    return personalOff.contains(sd.day.id) || companyOff.contains(sd.day.id);
                })
                .penalize(HardMediumSoftScore.ONE_HARD)
                .asConstraint("p2-worker-unavailable-day");
        }

        Constraint respectPinnedAssignments(ConstraintFactory f) {
            return f.forEach(CrewSeat.class)
                .filter(s -> s.pinned)
                .filter(s -> s.employee == null || s.employee.wid == null || !s.employee.wid.equals(s.pinnedWid))
                .penalize(HardMediumSoftScore.ONE_HARD)
                .asConstraint("p2-hard-respect-pinned-assignments");
        }

        Constraint softSameCompanyPairs(ConstraintFactory f) {
            return f.forEach(CrewSeat.class)
                .filter(a -> !isUnassigned(a.employee))
                .join(f.forEach(CrewSeat.class),
                        Joiners.equal((CrewSeat a) -> a.blockId, (CrewSeat b) -> b.blockId))
                .filter((a, b) -> !isUnassigned(b.employee) && a.id < b.id)
                .filter((a, b) -> !company(a.employee).isEmpty()
                        && company(a.employee).equals(company(b.employee)))
                .reward(HardMediumSoftScore.ONE_SOFT, (a, b) -> COMPANY_PAIR_W)
                .asConstraint("p2-soft-same-company-pairs");
        }

        Constraint softEncourageSkillVariety(ConstraintFactory f) {
            return f.forEach(CrewSeat.class)
                .filter(a -> !isUnassigned(a.employee))
                .join(f.forEach(CrewSeat.class),
                        Joiners.equal((CrewSeat a) -> a.blockId, (CrewSeat b) -> b.blockId),
                        Joiners.equal((CrewSeat a) -> a.opId,    (CrewSeat b) -> b.opId))
                .filter((a, b) -> !isUnassigned(b.employee) && a.id < b.id)
                .filter((a, b) -> skill(a.employee, a.opId) == skill(b.employee, b.opId))
                .penalize(HardMediumSoftScore.ONE_SOFT, (a, b) -> SKILL_DIVERSITY_W)
                .asConstraint("p2-soft-encourage-skill-variety");
        }

        Constraint softBalanceBlockAvgSkill(ConstraintFactory f) {
            return f.forEach(CrewSeat.class)
                .filter(s -> !isUnassigned(s.employee))
                .groupBy(s -> Arrays.asList(s.blockId, s.opId),
                        ConstraintCollectors.sum(s -> skill(s.employee, s.opId)),
                        ConstraintCollectors.count())
                .filter((key, sumLv, cnt) -> cnt > 0)
                .penalize(HardMediumSoftScore.ONE_SOFT,
                        (key, sumLv, cnt) -> (int) (SKILL_AVG_W *
                                Math.abs((sumLv / Math.max(1.0, cnt)) - avgSkill((String) key.get(1))) * 100))
                .asConstraint("p2-soft-balance-block-avg-skill");
        }

        Constraint softBalanceTotalHours(ConstraintFactory f) {
            return f.forEach(SeatDay.class)
                .join(f.forEach(CrewSeat.class),
                        Joiners.equal((SeatDay sd) -> sd.seatKey, (CrewSeat cs) -> cs.seatKey))
                .filter((sd, cs) -> !isUnassigned(cs.employee))
                .groupBy((sd, cs) -> cs.employee.id,
                        ConstraintCollectors.sum((sd, cs) -> sd.hours))
                .penalize(HardMediumSoftScore.ONE_SOFT,
                        (empId, tot) -> (int) Math.abs(tot - TARGET_HOURS_PER_EMP))
                .asConstraint("p2-soft-balance-total-hours");
        }
    }

    // ---------------- Parsing ----------------

    static class OpDef {
        String phaseId; int phaseNum;
        List<Integer> allowed; int min; int max;
    }
    static class ParsedEnv {
        Map<String,OpDef> opdef; List<EmployeeFact> employees;

        // helper: find employee by wid
        Map<String, EmployeeFact> byWid;
    }
    static class ParsedSchedule {
        LocalDate planStart; LocalDate planEnd;
        List<DaySlot> daySlots; List<TaskWindow> windows;
        Map<String,Integer> requiredByKey;

        // Fixed rows parsed
        List<FixedAssign> fixedRows;
        Map<String,Integer> fixedHoursByKey;
    }

    static class FixedAssign {
        String module; String opId; String wid;
        int startDayId; int endDayId;
        Map<Integer,Integer> hoursByDay = new HashMap<>(); // dayId -> hours
        String phaseId; int phaseNum;
    }

    @SuppressWarnings("unchecked")
    static ParsedEnv parseEnv(String envPath) throws IOException {
        Map<String,Object> root = loadYaml(envPath);
        Map<String,Object> env = (Map<String,Object>) root.getOrDefault("environment", root);

        Map<String,OpDef> opdef = new HashMap<>();
        List<Map<String,Object>> wfl = (List<Map<String,Object>>) env.getOrDefault("workflow_list", List.of());
        if (!wfl.isEmpty()) {
            Map<String,Object> wf0 = wfl.get(0);
            List<Map<String,Object>> phases = (List<Map<String,Object>>) wf0.getOrDefault("phase_list", List.of());
            for (Map<String,Object> ph : phases) {
                String phId = safeStr(ph.get("id"));
                int phNum = phaseNumFromId(phId);
                List<Map<String,Object>> ops = (List<Map<String,Object>>) ph.getOrDefault("operation_list", List.of());
                for (Map<String,Object> op : ops) {
                    String opId = safeStr(op.get("id"));
                    List<Object> hrs = (List<Object>) op.getOrDefault("work_hours", List.of(8));
                    List<Integer> allowed = hrs.isEmpty() ? List.of(8)
                            : hrs.stream().map(x -> parseInt(x,8)).sorted().collect(Collectors.toList());
                    OpDef od = new OpDef();
                    od.phaseId = phId; od.phaseNum = phNum; od.allowed = allowed;
                    od.min = parseInt(op.get("min_worker_num"), 1);
                    od.max = parseInt(op.get("max_worker_num"), 999_999);
                    opdef.put(opId, od);
                }
            }
        }

        List<EmployeeFact> employees = new ArrayList<>();
        employees.add(new EmployeeFact(0, "__UNASSIGNED__", "__UNASSIGNED__", Map.of(), false, ""));
        List<Map<String,Object>> workers = (List<Map<String,Object>>) env.getOrDefault("worker_list", List.of());
        int eid = 1;
        Map<String, EmployeeFact> byWid = new HashMap<>();
        for (Map<String,Object> w : workers) {
            String wid = safeStr(w.get("id"));
            String name = safeStr(w.get("name"));
            boolean isMgr = Boolean.TRUE.equals(w.get("is_manager"));
            String company = safeStr(w.get("worker_company"));
            Map<String,Integer> skills = new HashMap<>();
            Map<String,Object> smap = (Map<String,Object>) w.getOrDefault("skill_map", Map.of());
            for (Map.Entry<String,Object> e : smap.entrySet()) {
                skills.put(e.getKey(), parseInt(e.getValue(), 0));
            }
            EmployeeFact ef = new EmployeeFact(eid++, wid, name, skills, isMgr, company);
            employees.add(ef);
            byWid.put(wid, ef);
        }

        // capacities
        OP_CAPACITY.clear();
        for (String opId : opdef.keySet()) {
            int c = 0;
            for (EmployeeFact e : employees) {
                if (e.id == 0) continue;
                if (e.skills.getOrDefault(opId, 0) > 0) c++;
            }
            OP_CAPACITY.put(opId, c);
        }

        // average skills
        OP_AVG_SKILL.clear();
        for (String opId : opdef.keySet()) {
            int sum = 0, cnt = 0;
            for (EmployeeFact e : employees) {
                if (e.id == 0) continue;
                int lv = e.skills.getOrDefault(opId, 0);
                if (lv > 0) { sum += lv; cnt++; }
            }
            OP_AVG_SKILL.put(opId, cnt > 0 ? (sum / (double) cnt) : 3.0);
        }

        ParsedEnv out = new ParsedEnv();
        out.opdef = opdef; out.employees = employees; out.byWid = byWid;
        return out;
    }

    @SuppressWarnings("unchecked")
    static ParsedSchedule parseSchedule(String schedPath, Map<String,OpDef> opdef) throws IOException {
        Map<String,Object> root = loadYaml(schedPath);
        Map<String,Object> s = (Map<String,Object>) root.getOrDefault("schedule", root);

        LocalDate start = LocalDate.parse(
                safeStr(((Map<String,Object>)s.get("plan_range")).get("start_date")).replace("-", "/"), DF);
        LocalDate end   = LocalDate.parse(
                safeStr(((Map<String,Object>)s.get("plan_range")).get("end_date")).replace("-", "/"), DF);

        int horizon = (int) (end.toEpochDay() - start.toEpochDay()) + 1;
        List<DaySlot> days = new ArrayList<>();
        for (int i=0;i<horizon;i++) days.add(new DaySlot(i, start.plusDays(i)));

        List<TaskWindow> windows = new ArrayList<>();
        Map<String,Integer> required = new HashMap<>();

        Object wfObj = s.get("workflow_task_list");
        @SuppressWarnings("unchecked")
        List<Map<String,Object>> wfTasks = (wfObj instanceof List) ? (List<Map<String,Object>>) wfObj : List.of();
        for (Map<String,Object> wf : wfTasks) {
            String module = safeStr(wf.get("id"));
            String fab = safeStr(wf.get("fab"));
            Object phasesObj = wf.get("phase_task_list");
            @SuppressWarnings("unchecked")
            List<Map<String,Object>> phases = (phasesObj instanceof List) ? (List<Map<String,Object>>) phasesObj : List.of();
            for (Map<String,Object> ph : phases) {
                String phId = safeStr(ph.get("phase"));
                int phNum = phaseNumFromId(phId);
                LocalDate pStart = LocalDate.parse(safeStr(ph.get("start_date")).replace("-", "/"), DF);
                LocalDate pEnd   = LocalDate.parse(safeStr(ph.get("end_date")).replace("-", "/"), DF);
                int startId = (int) (pStart.toEpochDay() - start.toEpochDay());
                int endId   = (int) (pEnd.toEpochDay()   - start.toEpochDay());

                Object opsObj = ph.get("operation_task_list");
                @SuppressWarnings("unchecked")
                List<Map<String,Object>> opTasks = (opsObj instanceof List) ? (List<Map<String,Object>>) opsObj : List.of();
                for (Map<String,Object> ot : opTasks) {
                    String opId = safeStr(ot.get("operation"));
                    int workloadDays = parseInt(ot.get("workload_days"), 0);

                    OpDef od = opdef.get(opId);
                    if (od == null) throw new IllegalArgumentException("operation "+opId+" missing in EnvConfig");

                    int baseline = (od.allowed.size()==1 && od.allowed.get(0)==4) ? 4 : 8;
                    int req = workloadDays * baseline;
                    required.merge(module + "|" + opId, req, Integer::sum);

                    TaskWindow tw = new TaskWindow();
                    tw.module = module; tw.factory = fab;
                    tw.phaseId = phId; tw.phaseNum = phNum;
                    tw.opId = opId; tw.startDayId = startId; tw.endDayId = endId;
                    tw.allowed = od.allowed; tw.minHeads = od.min; tw.maxHeads = od.max;
                    tw.workloadDays = workloadDays;
                    windows.add(tw);
                }
            }
        }

        // Parse Fixed assignments (plan_flexibility: fixed)
        List<FixedAssign> fixedRows = new ArrayList<>();
        Map<String,Integer> fixedHoursByKey = new HashMap<>();

        Object asgObj = s.get("assignment_list");
        @SuppressWarnings("unchecked")
        List<Map<String,Object>> asgs = (asgObj instanceof List) ? (List<Map<String,Object>>) asgObj : List.of();

        for (Map<String,Object> a : asgs) {
            String wdKey = a.containsKey("work_date_lsit") ? "work_date_lsit" : "work_date_list";
            Object wdObj = a.get(wdKey);
            @SuppressWarnings("unchecked")
            List<Map<String,Object>> wdl = (wdObj instanceof List) ? (List<Map<String,Object>>) wdObj : List.of();

            String flex = safeStr(a.get("plan_flexibility"));
            if (!"fixed".equalsIgnoreCase(flex)) continue;

            String opTask = safeStr(a.get("operation_task")); // e.g., e16p4o1
            int idx = opTask.indexOf("p");
            String module = (idx > 0) ? opTask.substring(0, idx) : opTask;
            String opId   = (idx > 0) ? opTask.substring(idx) : "";

            String wid = safeStr(a.get("worker"));
            LocalDate sd = a.get("start_date") == null ? null : LocalDate.parse(safeStr(a.get("start_date")).replace("-", "/"), DF);
            LocalDate ed = a.get("end_date")   == null ? null : LocalDate.parse(safeStr(a.get("end_date")).replace("-", "/"), DF);
            int sId = sd == null ? -1 : (int)(sd.toEpochDay() - start.toEpochDay());
            int eId = ed == null ? -1 : (int)(ed.toEpochDay() - start.toEpochDay());

            Map<Integer,Integer> byDay = new HashMap<>();
            for (Map<String,Object> item : wdl) {
                LocalDate d = LocalDate.parse(safeStr(item.get("date")).replace("-", "/"), DF);
                int did = (int)(d.toEpochDay() - start.toEpochDay());
                int h = parseInt(item.get("hour"), 0);
                byDay.merge(did, h, Integer::sum);
                fixedHoursByKey.merge(module + "|" + opId, h, Integer::sum);
            }

            // phase info (best-effort)
            String phId = "";
            int phNum = 0;
            try {
                String pPart = opId.split("o", 2)[0]; // p4
                phId = pPart;
                phNum = phaseNumFromId(pPart);
            } catch (Exception ignore) {}

            FixedAssign fa = new FixedAssign();
            fa.module = module; fa.opId = opId; fa.wid = wid;
            fa.startDayId = sId; fa.endDayId = eId;
            fa.hoursByDay = byDay;
            fa.phaseId = phId; fa.phaseNum = phNum;
            fixedRows.add(fa);
        }

        ParsedSchedule out = new ParsedSchedule();
        out.planStart = start; out.planEnd = end;
        out.daySlots = days; out.windows = windows; out.requiredByKey = required;
        out.fixedRows = fixedRows; out.fixedHoursByKey = fixedHoursByKey;
        return out;
    }

    // ---------------- Build seats from blocks ----------------

    static class Expanded { List<CrewSeat> seats; List<SeatDay> seatDays; }

    static Expanded expandToSeats(List<BlockDecision> blocks, List<DaySlot> days) {
        Map<Integer, DaySlot> byId = days.stream().collect(Collectors.toMap(d -> d.id, d -> d));
        List<CrewSeat> seats = new ArrayList<>();
        List<SeatDay> seatDays = new ArrayList<>();
        int sid = 1;

        for (BlockDecision b : blocks) {
            int hours = autoHours(b);
            int start = b.startDay == null ? b.windowStart : b.startDay;
            int dcount = b.days == null ? 1 : b.days;
            int headCount = b.heads == null ? 1 : b.heads;

            for (int sidx = 0; sidx < headCount; sidx++) {
                String seatKey = b.module + "|" + b.opId + "|s" + String.format("%04d", sidx) + "|d" + start;
                CrewSeat cs = new CrewSeat();
                cs.id = sid++;
                cs.module = b.module; cs.factory = b.factory;
                cs.phaseId = b.phaseId; cs.phaseNum = b.phaseNum;
                cs.opId = b.opId; cs.startDayId = start;
                cs.days = dcount; cs.hours = hours; cs.seatIndex = sidx;
                cs.seatKey = seatKey; cs.blockId = b.id;
                cs.employee = new EmployeeFact(0, "__UNASSIGNED__", "__UNASSIGNED__", Map.of(), false, "");
                seats.add(cs);

                for (int off=0; off<dcount; off++) {
                    int did = start + off;
                    if (!isWorkingDay(did, b.factory)) continue; // skip weekends/unavailable
                    DaySlot dd = byId.get(did);
                    if (dd != null) seatDays.add(new SeatDay(seatKey, dd, hours, b.factory));
                }
            }
        }
        Expanded ex = new Expanded();
        ex.seats = seats; ex.seatDays = seatDays;
        return ex;
    }

    // Build pinned seats directly from Fixed rows (exact days & hours), assigned to the given worker.
    static Expanded expandPinnedSeats(ParsedSchedule sch, ParsedEnv env, List<TaskWindow> windows, List<DaySlot> days) {
        Map<Integer, DaySlot> byId = days.stream().collect(Collectors.toMap(d -> d.id, d -> d));
        // quick lookup for factory by module (from windows)
        Map<String,String> moduleToFactory = new HashMap<>();
        Map<String,String> moduleOpToPhase = new HashMap<>();
        Map<String,Integer> moduleOpToPhaseNum = new HashMap<>();
        for (TaskWindow w : windows) {
            moduleToFactory.put(w.module, w.factory);
            moduleOpToPhase.put(w.module + "|" + w.opId, w.phaseId);
            moduleOpToPhaseNum.put(w.module + "|" + w.opId, w.phaseNum);
        }

        List<CrewSeat> seats = new ArrayList<>();
        List<SeatDay> seatDays = new ArrayList<>();
        int sid = 1_000_000; // keep pinned IDs separate

        for (FixedAssign fa : sch.fixedRows) {
            String factory = moduleToFactory.getOrDefault(fa.module, null);
            String seatKey = fa.module + "|" + fa.opId + "|PIN|" + fa.wid + "|d" + fa.startDayId;
            CrewSeat cs = new CrewSeat();
            cs.id = sid++;
            cs.module = fa.module; cs.factory = factory;
            cs.phaseId = moduleOpToPhase.getOrDefault(fa.module + "|" + fa.opId, fa.phaseId);
            cs.phaseNum = moduleOpToPhaseNum.getOrDefault(fa.module + "|" + fa.opId, fa.phaseNum);
            cs.opId = fa.opId; cs.startDayId = fa.startDayId < 0 ? 0 : fa.startDayId;
            // derive days count from explicit map
            int minDid = fa.hoursByDay.keySet().stream().min(Integer::compareTo).orElse(cs.startDayId);
            int maxDid = fa.hoursByDay.keySet().stream().max(Integer::compareTo).orElse(cs.startDayId);
            cs.days = maxDid - minDid + 1;
            cs.hours = fa.hoursByDay.values().stream().max(Integer::compareTo).orElse(8);
            cs.seatIndex = 0;
            cs.seatKey = seatKey;
            cs.blockId = -1; // not from Pass 1
            cs.pinned = true;
            cs.pinnedWid = fa.wid;

            // assign employee object
            EmployeeFact ef = env.byWid.get(fa.wid);
            if (ef == null) ef = new EmployeeFact(0, "__UNASSIGNED__", "__UNASSIGNED__", Map.of(), false, "");
            cs.employee = ef;

            seats.add(cs);

            // exact seat-days with per-day hours
            for (Map.Entry<Integer,Integer> e : fa.hoursByDay.entrySet()) {
                int did = e.getKey();
                int hrs = e.getValue();

                // respect blackout days like weekends
                if (!isWorkingDay(did, factory)) continue;

                DaySlot dd = byId.get(did);
                if (dd != null) seatDays.add(new SeatDay(seatKey, dd, hrs, factory));
            }
        }

        Expanded ex = new Expanded();
        ex.seats = seats; ex.seatDays = seatDays;
        return ex;
    }

    // ---------------- Solver builders ----------------

    static <S> Solver<S> buildSolver(Class<S> solutionClass,
                                     Class<?>[] entityClasses,
                                     Class<? extends ConstraintProvider> providerClass,
                                     String bestScoreLimit,
                                     Integer spentMinutes,
                                     Integer unimprovedSeconds) {
        SolverConfig cfg = new SolverConfig();
        cfg.withSolutionClass(solutionClass);
        cfg.withEntityClasses(entityClasses);
        cfg.withScoreDirectorFactory(new ScoreDirectorFactoryConfig().withConstraintProviderClass(providerClass));

        TerminationConfig term = new TerminationConfig();
        if (bestScoreLimit != null) term.setBestScoreLimit(bestScoreLimit);
        if (spentMinutes != null && spentMinutes > 0)
            term.setSpentLimit(java.time.Duration.ofMinutes(spentMinutes));
        if (unimprovedSeconds != null && unimprovedSeconds > 0)
            term.setUnimprovedSpentLimit(java.time.Duration.ofSeconds(unimprovedSeconds));
        cfg.withTerminationConfig(term);

        return SolverFactory.<S>create(cfg).buildSolver();
    }

    static boolean hardZero(Score<?> s) { return s != null && s.toString().startsWith("0hard"); }

    // ---------------- Helpers for selective loop + snapshots ----------------

    static class Pass1Result { List<BlockDecision> blocks; HardMediumSoftScore score; boolean hardIsZero; }

    static Set<Integer> detectHardViolators(List<BlockDecision> blocks) {
        Set<Integer> bad = new HashSet<>();

        for (BlockDecision b : blocks) {
            Integer sd = b.startDay, d = b.days, hds = b.heads;
            if (d != null && sd != null) {
                int end = sd + d - 1;
                if (!(sd >= b.windowStart && end <= b.windowEnd && d >= 1)) bad.add(b.id);
                int maxLen = b.windowEnd - b.windowStart + 1;
                if (d > maxLen) bad.add(b.id);
            }
            if (hds == null || hds < b.minHeads || hds > b.maxHeads) bad.add(b.id);

            int auto = autoHours(b);
            if (b.allowed == null || b.allowed.isEmpty() || !b.allowed.contains(auto)) bad.add(b.id);

            if (produced(b) < b.requiredHours) bad.add(b.id);
            {
                int prod = produced(b);
                int over = prod - b.requiredHours;
                int H = Math.max(1, (hds == null ? 1 : hds));
                if (over > H * auto) bad.add(b.id);
            }
        }

        // phase order check
        for (BlockDecision a : blocks) {
            for (BlockDecision b : blocks) {
                if (!Objects.equals(a.module, b.module)) continue;
                if (a.phaseNum + 1 != b.phaseNum) continue;
                if (a.startDay != null && a.days != null && b.startDay != null) {
                    if ((a.startDay + a.days - 1) >= b.startDay) {
                        bad.add(a.id); bad.add(b.id);
                    }
                }
            }
        }

        // per-day head capacity by op (working days only)
        Map<Integer, List<BlockDecision>> byDay = new HashMap<>();
        for (BlockDecision b : blocks) {
            if (b.startDay == null || b.days == null || b.heads == null) continue;
            for (int day = b.startDay; day <= b.startDay + b.days - 1; day++) {
                if (!isWorkingDay(day, b.factory)) continue;
                byDay.computeIfAbsent(day, k -> new ArrayList<>()).add(b);
            }
        }
        for (Map.Entry<Integer, List<BlockDecision>> e : byDay.entrySet()) {
            Map<String, Integer> sumByOp = new HashMap<>();
            for (BlockDecision b : e.getValue()) {
                sumByOp.merge(b.opId, (b.heads == null ? 0 : b.heads), Integer::sum);
            }
            for (Map.Entry<String,Integer> s : sumByOp.entrySet()) {
                String op = s.getKey(); int total = s.getValue();
                int cap = OP_CAPACITY.getOrDefault(op, Integer.MAX_VALUE);
                if (total > cap) {
                    for (BlockDecision b : e.getValue()) {
                        if (Objects.equals(b.opId, op)) bad.add(b.id);
                    }
                }
            }
        }
        return bad;
    }

    static List<BlockDecision> seedBlocksForTier(List<TaskWindow> windows,
                                                 Map<Integer,Integer> perBlockTier,
                                                 Map<String,Integer> fixedHoursByKey) {
        List<BlockDecision> blocks = new ArrayList<>();
        int bid = 1;
        for (TaskWindow w : windows) {
            List<Integer> full = w.fullAllowedSorted();
            int tier = perBlockTier.getOrDefault(bid, 1);
            List<Integer> tiered = full.subList(0, Math.min(tier, full.size()));

            int baseline = (tiered.size()==1 && tiered.get(0)==4) ? 4 : 8;
            int totalReq = w.workloadDays * baseline;
            int fixed = fixedHoursByKey.getOrDefault(w.module + "|" + w.opId, 0);
            int req = Math.max(0, totalReq - fixed);

            if (req == 0) { bid++; continue; } // nothing to plan

            int seedHours = tiered.get(0);
            int minH = Math.max(1, w.minHeads);
            int maxDays = w.endDayId - w.startDayId + 1;
            int safeDen = Math.max(1, seedHours * minH);
            int seedDays = Math.max(1, Math.min((req + safeDen - 1) / safeDen, maxDays));

            BlockDecision b = new BlockDecision();
            b.id = bid;
            b.module = w.module; b.factory = w.factory;
            b.phaseId = w.phaseId; b.phaseNum = w.phaseNum;
            b.opId = w.opId;
            b.windowStart = w.startDayId; b.windowEnd = w.endDayId;
            b.requiredHours = req;
            b.allowed = new ArrayList<>(tiered);
            b.minHeads = w.minHeads; b.maxHeads = w.maxHeads;
            b.startDay = w.startDayId; b.heads = minH; b.days = seedDays;
            b.seedHours = seedHours;
            blocks.add(b);
            bid++;
        }
        return blocks;
    }

    static List<BlockDecision> cloneBlocks(List<BlockDecision> src) {
        List<BlockDecision> out = new ArrayList<>();
        for (BlockDecision s : src) {
            BlockDecision b = new BlockDecision();
            b.id = s.id; b.module = s.module; b.factory = s.factory;
            b.phaseId = s.phaseId; b.phaseNum = s.phaseNum; b.opId = s.opId;
            b.windowStart = s.windowStart; b.windowEnd = s.windowEnd;
            b.requiredHours = s.requiredHours;
            b.allowed = (s.allowed == null ? null : new ArrayList<>(s.allowed));
            b.minHeads = s.minHeads; b.maxHeads = s.maxHeads;
            b.startDay = s.startDay; b.heads = s.heads; b.days = s.days;
            b.seedHours = s.seedHours;
            out.add(b);
        }
        return out;
    }

    // --- write a per-iteration snapshot: include pinned + flexible seats
    static void writeScheduleSnapshot(
            int iterIndex,
            List<BlockDecision> blocks,
            SnapshotCfg cfg
    ) {
        try {
            if (cfg == null) return;

            // expand flexible + pinned
            Expanded exFlex = expandToSeats(blocks, cfg.daySlots);
            Expanded exPinned = expandPinnedSeats(cfg.sch, cfg.env, cfg.windows, cfg.daySlots);

            // pick employee id=1; if absent, pick first non-zero
            EmployeeFact emp1 = null;
            for (EmployeeFact e : cfg.employees) { if (e != null && e.id == 1) { emp1 = e; break; } }
            if (emp1 == null) for (EmployeeFact e : cfg.employees) { if (e != null && e.id != 0) { emp1 = e; break; } }
            if (emp1 == null) { System.out.println("[snapshot] No employee available. Skipping."); return; }

            // merge seats and days
            List<CrewSeat> allSeats = new ArrayList<>();
            allSeats.addAll(exPinned.seats);
            allSeats.addAll(exFlex.seats);
            List<SeatDay> allSeatDays = new ArrayList<>();
            allSeatDays.addAll(exPinned.seatDays);
            allSeatDays.addAll(exFlex.seatDays);

            // assign employee to flexible seats only
            Set<String> pinnedKeys = exPinned.seats.stream().map(s -> s.seatKey).collect(Collectors.toSet());
            for (CrewSeat s : allSeats) {
                if (!pinnedKeys.contains(s.seatKey)) s.employee = emp1;
            }

            Pass2Plan snap = new Pass2Plan();
            snap.days = cfg.daySlots;
            snap.employees = cfg.employees;
            snap.seatDays = allSeatDays;
            snap.seats = allSeats;

            // paths
            Path orig = Paths.get(cfg.schedulePath);
            String baseName = "Schedule" + iterIndex + ".yaml";
            Path out = (orig.getParent() == null) ? Paths.get(baseName) : orig.getParent().resolve(baseName);

            // ensure directory + ensure the snapshot file exists by copying from the template
            Path parent = out.getParent();
            if (parent != null) Files.createDirectories(parent);
            if (Files.notExists(out)) {
                Files.copy(orig, out, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
            }
            System.out.println("[snapshot] Target: " + out.toAbsolutePath());

            // overwrite assignments in the snapshot file
            ExportSchedule.overwriteScheduleWithAssignments(snap, cfg.planStart, out.toString(), cfg.envPath);
            System.out.println("[snapshot] Wrote " + out);
        } catch (Exception ex) {
            System.out.println("[snapshot] Failed to write snapshot for loop " + iterIndex + ": " + ex.getMessage());
            ex.printStackTrace(System.out);
        }
    }

    // ---------------- Pass 1: SELECTIVE RAMP LOOP with snapshots ----------------

    static Pass1Result solvePass1SelectiveRamp(
            List<DaySlot> daySlots,
            List<TaskWindow> windows,
            Map<String,Integer> fixedHoursByKey,
            SnapshotCfg snapshotCfg
    ) {
        // Global value ranges
        int maxHeads = windows.stream().mapToInt(w -> w.maxHeads).max().orElse(1);
        int minHeads = windows.stream().mapToInt(w -> w.minHeads).min().orElse(1);
        int maxWin   = windows.stream().mapToInt(w -> w.endDayId - w.startDayId + 1).max().orElse(1);

        List<Integer> headOptions = new ArrayList<>();
        for (int h=minHeads; h<=maxHeads; h++) headOptions.add(h);
        List<Integer> dayIds = daySlots.stream().map(d -> d.id).collect(Collectors.toList());
        List<Integer> dayCountOptions = new ArrayList<>();
        for (int d=1; d<=maxWin; d++) dayCountOptions.add(d);

        // per-block current tier (start at 1)
        Map<Integer,Integer> perBlockTier = new HashMap<>();
        int nBlocks = windows.size();
        for (int i=1; i<=nBlocks; i++) perBlockTier.put(i, 1);

        // compute per-block max tier
        Map<Integer,Integer> perBlockMaxTier = new HashMap<>();
        {
            int id = 1;
            for (TaskWindow w : windows) {
                perBlockMaxTier.put(id++, Math.max(1, w.fullAllowedSorted().size()));
            }
        }

        Pass1Result best = new Pass1Result();
        best.score = null; best.blocks = null; best.hardIsZero = false;

        int globalMaxLoops = perBlockMaxTier.values().stream().mapToInt(Integer::intValue).max().orElse(1);
        for (int iter = 1; iter <= globalMaxLoops; iter++) {
            System.out.println("Ramp tier" + iter + " at " + nowClock());
            List<BlockDecision> blocks = seedBlocksForTier(windows, perBlockTier, fixedHoursByKey);

            Pass1Plan p1 = new Pass1Plan();
            p1.dayIds = dayIds; p1.headOptions = headOptions; p1.dayCountOptions = dayCountOptions;
            p1.daySlots = daySlots; p1.blocks = blocks;

            Solver<Pass1Plan> solver = buildSolver(Pass1Plan.class, new Class<?>[]{ BlockDecision.class },
                    Pass1Constraints.class, "0hard/*medium/*soft", 30, 60);
            Pass1Plan solved = solver.solve(p1);

            // snapshot for this iteration
            writeScheduleSnapshot(iter, solved.blocks, snapshotCfg);
            System.out.printf("Done tier %d at %s | score=%s%n",
                    iter, nowClock(), String.valueOf(solved.getScore()));

            if (best.score == null ||
                (hardZero(solved.getScore()) && !hardZero(best.score)) ||
                (!hardZero(best.score) && solved.getScore().toString().compareTo(best.score.toString()) < 0)) {
                best.blocks = cloneBlocks(solved.blocks);
                best.score = solved.getScore();
            }

            if (hardZero(solved.getScore())) {
                // polish
                System.out.println("Polish Pass 1 at " + nowClock());
                Solver<Pass1Plan> polish = buildSolver(Pass1Plan.class, new Class<?>[]{ BlockDecision.class },
                        Pass1Constraints.class, null, 20, 60);
                Pass1Plan polished = polish.solve(solved);

                // write polished snapshot over the same iter index
                writeScheduleSnapshot(iter, polished.blocks, snapshotCfg);

                Pass1Result r = new Pass1Result();
                r.blocks = polished.blocks; r.score = polished.getScore(); r.hardIsZero = true;
                return r;
            }

            // Detect hard violators (with working-day head capacity etc.)
            Set<Integer> violators = detectHardViolators(solved.blocks);

            // Build next-iteration seed: tier-up violators, retain hours choice for non-violators
            boolean anyTierChange = false;
            List<BlockDecision> next = new ArrayList<>();
            Map<Integer, BlockDecision> solvedById = solved.blocks.stream()
                    .collect(Collectors.toMap(b -> b.id, b -> b));
            int bid = 1;
            for (TaskWindow w : windows) {
                List<Integer> full = w.fullAllowedSorted();
                int curTier = perBlockTier.getOrDefault(bid, 1);

                // compute req again (subtract fixed)
                int baseline = (full.size()==1 && full.get(0)==4) ? 4 : 8;
                int totalReq = w.workloadDays * baseline;
                int fixed = fixedHoursByKey.getOrDefault(w.module + "|" + w.opId, 0);
                int req = Math.max(0, totalReq - fixed);

                // If this window had 0 req after subtracting fixed, skip from next seed
                if (req == 0) { bid++; continue; }

                BlockDecision solvedB = solvedById.get(bid);
                BlockDecision nb = new BlockDecision();
                nb.id = bid;
                nb.module = w.module; nb.factory = w.factory;
                nb.phaseId = w.phaseId; nb.phaseNum = w.phaseNum;
                nb.opId = w.opId;
                nb.windowStart = w.startDayId; nb.windowEnd = w.endDayId;
                nb.requiredHours = req;
                nb.minHeads = w.minHeads; nb.maxHeads = w.maxHeads;

                if (!violators.contains(bid) && solvedB != null) {
                    int h = autoHours(solvedB);
                    nb.allowed = List.of(h);          // freeze only the hours choice
                    nb.startDay = solvedB.startDay;
                    nb.heads    = solvedB.heads;
                    nb.days     = solvedB.days;
                } else {
                    int maxTier = perBlockMaxTier.get(bid);
                    int newTier = Math.min(maxTier, curTier + 1);
                    if (newTier != curTier) {
                        anyTierChange = true;
                        perBlockTier.put(bid, newTier);
                    }
                    List<Integer> tiered = full.subList(0, Math.min(perBlockTier.get(bid), full.size()));
                    nb.allowed = new ArrayList<>(tiered);

                    int seedHours = tiered.get(0);
                    int minH = Math.max(1, w.minHeads);
                    int maxDays = w.endDayId - w.startDayId + 1;
                    int safeDen = Math.max(1, seedHours * minH);
                    int seedDays = Math.max(1, Math.min((req + safeDen - 1) / safeDen, maxDays));
                    nb.startDay = w.startDayId; nb.heads = minH; nb.days = seedDays;
                }

                next.add(nb);
                bid++;
            }

            if (!anyTierChange) {
                Pass1Result r = new Pass1Result();
                r.blocks = (best.blocks != null ? best.blocks : next);
                r.score = (best.score != null ? best.score : solved.getScore());
                r.hardIsZero = hardZero(r.score);
                return r;
            }
        }

        Pass1Result r = new Pass1Result();
        r.blocks = best.blocks;
        r.score = (best.score == null) ? HardMediumSoftScore.ofHard(1) : best.score;
        r.hardIsZero = hardZero(r.score);
        return r;
    }

    // ---------------- Pass 2 once (+ polish if hard==0) ----------------

    static Pass2Plan solvePass2Once(List<DaySlot> days, List<EmployeeFact> employees,
                                    List<CrewSeat> seats, List<SeatDay> seatDays) {
        Pass2Plan p2 = new Pass2Plan();
        p2.days = days; p2.employees = employees; p2.seats = seats; p2.seatDays = seatDays;

        Solver<Pass2Plan> solver = buildSolver(Pass2Plan.class, new Class<?>[]{ CrewSeat.class },
                Pass2Constraints.class, "0hard/*medium/*soft", 30, 60);
        Pass2Plan result = solver.solve(p2);

        if (hardZero(result.getScore())) {
            System.out.println("Polish Pass 2 at " + nowClock());
            Solver<Pass2Plan> polish = buildSolver(Pass2Plan.class, new Class<?>[]{ CrewSeat.class },
                    Pass2Constraints.class, null, 20, 60);
            result = polish.solve(result);
        }
        return result;
    }

    // ---------------- Public API ----------------

    public static class RunResult { public Pass2Plan finalPlan; public LocalDate planStart; }

    public static RunResult solveFromYaml(String envPath, String schedPath) throws IOException {
        ParsedEnv env = parseEnv(envPath);
        ParsedSchedule sch = parseSchedule(schedPath, env.opdef);

        // Build blackout calendars based on EnvConfig + plan range
        buildCalendars(envPath, sch.planStart, sch.planEnd);

        int realEmp = Math.max(1, env.employees.size() - 1);
        int totalReq = sch.requiredByKey.values().stream().mapToInt(Integer::intValue).sum();
        TARGET_HOURS_PER_EMP = totalReq / (double) realEmp;

        // Snapshot config
        SnapshotCfg snapCfg = new SnapshotCfg();
        snapCfg.planStart = sch.planStart;
        snapCfg.schedulePath = schedPath;
        snapCfg.employees = env.employees;
        snapCfg.env = env;
        snapCfg.sch = sch;
        snapCfg.windows = sch.windows;
        snapCfg.daySlots = sch.daySlots;
        snapCfg.envPath = envPath;

        // ---- PASS 1 ----
        System.out.println("Start PASS 1 (selective ramp + snapshots) at " + nowClock());
        long t1 = System.nanoTime();
        Pass1Result p1 = solvePass1SelectiveRamp(sch.daySlots, sch.windows, sch.fixedHoursByKey, snapCfg);
        long t1End = System.nanoTime();
        System.out.printf(
            "Done PASS 1 at %s | duration=%s | score=%s | blocks=%d%n",
            nowClock(),
            fmt(java.time.Duration.ofNanos(t1End - t1)),
            String.valueOf(p1.score),
            p1.blocks == null ? 0 : p1.blocks.size()
        );

        // Build seats pre-Pass2: flexible + pinned merged
        Expanded exFlex = expandToSeats(p1.blocks, sch.daySlots);
        Expanded exPinned = expandPinnedSeats(sch, env, sch.windows, sch.daySlots);

        List<CrewSeat> allSeats = new ArrayList<>();
        allSeats.addAll(exPinned.seats);
        allSeats.addAll(exFlex.seats);

        List<SeatDay> allSeatDays = new ArrayList<>();
        allSeatDays.addAll(exPinned.seatDays);
        allSeatDays.addAll(exFlex.seatDays);

        System.out.printf("Expanded to seats: %d seats (pinned=%d, flexible=%d), %d seat-days%n",
            allSeats.size(),
            exPinned.seats.size(),
            exFlex.seats.size(),
            allSeatDays.size()
        );

        // ---- PASS 2 ----
        Pass2Plan finalP2;
        if (p1.hardIsZero) {
            System.out.println("Start PASS 2 at " + nowClock());
            long t2 = System.nanoTime();
            finalP2 = solvePass2Once(sch.daySlots, env.employees, allSeats, allSeatDays);
            long t2End = System.nanoTime();
            System.out.printf(
                "Done PASS 2 at %s | duration=%s | score=%s | seats=%d%n",
                nowClock(),
                fmt(java.time.Duration.ofNanos(t2End - t2)),
                String.valueOf(finalP2.getScore()),
                finalP2.seats == null ? 0 : finalP2.seats.size()
            );
        } else {
            System.out.println("PASS 1 hard score != 0. Skipping PASS 2 (diagnostic output only).");
            finalP2 = new Pass2Plan();
            finalP2.days = sch.daySlots; finalP2.employees = env.employees;
            finalP2.seats = allSeats; finalP2.seatDays = allSeatDays;
        }

        RunResult rr = new RunResult();
        rr.finalPlan = finalP2; rr.planStart = sch.planStart;
        return rr;
    }

    public static void main(String[] args) throws Exception {
        String envPath = args.length > 0 ? args[0] : "EnvConfig.yaml";
        String schedPath = args.length > 1 ? args[1] : "Schedule.yaml";

        RunResult rr = solveFromYaml(envPath, schedPath);

        // Final: overwrite original Schedule.yaml with real (Pass 2) assignments.
        // Keep the 4-arg signature like v821 (if your ExportSchedule only has 2-arg, adjust here).
        ExportSchedule.overwriteScheduleWithAssignments(rr.finalPlan, rr.planStart, schedPath, envPath);

        System.out.println("Done.");
    }
}
