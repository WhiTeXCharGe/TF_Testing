// mvn -q exec:java -D"exec.args=src\main\resource\EnvConfig.yaml src\main\resource\Schedule.yaml"

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
 * Pass 1: selective ramp with pinning + per-iteration snapshots (Schedule1.yaml, Schedule2.yaml, ...)
 * Snapshots assign ALL seats to employee ID=1 (no Pass 2).
 * Final polished Pass 1 then goes to Pass 2 and overwrites original Schedule.yaml.
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
        public int requiredHours;
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

        // pinning
        public boolean pinned = false;
        public Integer pinStart = null;
        public Integer pinHeads = null;
        public Integer pinDays  = null;
        public Integer pinHours = null;

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
    }

    static boolean isUnassigned(EmployeeFact e) { return e == null || e.id == 0; }
    static int skill(EmployeeFact e, String opId) { return (e == null) ? 0 : e.skills.getOrDefault(opId, 0); }
    static boolean isManager(EmployeeFact e) { return e != null && e.isManager; }
    static String company(EmployeeFact e) { return e == null ? "" : (e.workerCompany == null ? "" : e.workerCompany); }
    static double avgSkill(String opId) { return OP_AVG_SKILL.getOrDefault(opId, 3.0); }

    static int produced(BlockDecision b) {
        int h = autoHours(b);
        int H = Math.max(1, b.heads == null ? 1 : b.heads);
        int D = Math.max(1, b.days  == null ? 1 : b.days);
        return H * h * D;
    }
    static int autoHours(BlockDecision b) {
        List<Integer> allowed = (b.allowed == null || b.allowed.isEmpty())
                ? List.of(8) : b.allowed.stream().sorted().collect(Collectors.toList());
        int H = Math.max(1, b.heads == null ? 1 : b.heads);
        int D = Math.max(1, b.days  == null ? 1 : b.days);
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

    // --- simple timing/format helpers ---
    private static final java.time.format.DateTimeFormatter CLOCK =
            java.time.format.DateTimeFormatter.ofPattern("HH:mm:ss");

    private static String nowClock() {
        return java.time.LocalTime.now().format(CLOCK);
    }
    private static String fmt(java.time.Duration d) {
        long h = d.toHours();
        long m = d.toMinutesPart();
        long s = d.toSecondsPart();
        long ms = d.toMillisPart();
        return String.format("%02d:%02d:%02d.%03d", h, m, s, ms);
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

                respectPins(f),

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
                                b.startDay <= d.id && d.id <= (b.startDay + b.days - 1)))
                .groupBy((DaySlot d, BlockDecision b) -> d.id,
                         (DaySlot d, BlockDecision b) -> b.opId,
                         ConstraintCollectors.sum((DaySlot d, BlockDecision b) -> b.heads == null ? 0 : b.heads))
                .filter((dayId, opId, totalHeads) -> totalHeads >
                        OP_CAPACITY.getOrDefault(opId, Integer.MAX_VALUE))
                .penalize(HardMediumSoftScore.ONE_HARD, (dayId, opId, totalHeads) ->
                        totalHeads - OP_CAPACITY.getOrDefault(opId, Integer.MAX_VALUE))
                .asConstraint("p1-daily-head-capacity-by-op");
        }

        Constraint respectPins(ConstraintFactory f) {
            return f.forEach(BlockDecision.class)
                .filter(b -> b.pinned)
                .filter(b ->
                        (b.startDay == null || !b.startDay.equals(b.pinStart)) ||
                        (b.heads    == null || !b.heads.equals(b.pinHeads))   ||
                        (b.days     == null || !b.days.equals(b.pinDays))     ||
                        (b.pinHours != null && autoHours(b) != b.pinHours))
                .penalize(HardMediumSoftScore.ONE_HARD)
                .asConstraint("p1-pinned-fixed");
        }

        Constraint penalizeStackByOp(ConstraintFactory f) {
            return f.forEach(DaySlot.class)
                .join(f.forEach(BlockDecision.class),
                    Joiners.filtering((DaySlot d, BlockDecision b) ->
                        b.startDay != null && b.days != null &&
                        b.startDay <= d.id && d.id <= (b.startDay + b.days - 1)))
                .groupBy(
                    (DaySlot d, BlockDecision b) -> d.id,
                    (DaySlot d, BlockDecision b) -> b.opId,
                    ConstraintCollectors.sum((DaySlot d, BlockDecision b) -> 1)
                )
                .filter((dayId, opId, n) -> n > 1)
                .penalize(HardMediumSoftScore.ONE_MEDIUM,
                    (dayId, opId, n) -> 2 * (n * (n - 1) / 2))
                .asConstraint("p1-med-penalize-stack-by-op");
        }

        Constraint preferHoursNear8(ConstraintFactory f) {
            return f.forEach(BlockDecision.class)
                .penalize(HardMediumSoftScore.ONE_SOFT, b -> 1000 * Math.abs(autoHours(b) - 8))
                .asConstraint("p1-soft-prefer-hours-near-8");
        }
        Constraint preferSmallerHours(ConstraintFactory f) {
            return f.forEach(BlockDecision.class)
                .penalize(HardMediumSoftScore.ONE_SOFT, b -> 100 * autoHours(b))
                .asConstraint("p1-soft-prefer-smaller-hours");
        }
        Constraint minimizeHeads(ConstraintFactory f) {
            return f.forEach(BlockDecision.class)
                .penalize(HardMediumSoftScore.ONE_SOFT, b -> b.heads == null ? 0 : 10 * b.heads)
                .asConstraint("p1-soft-minimize-heads");
        }
        Constraint minimizeDays(ConstraintFactory f) {
            return f.forEach(BlockDecision.class)
                .penalize(HardMediumSoftScore.ONE_SOFT, b -> b.days == null ? 0 : 1 * b.days)
                .asConstraint("p1-soft-minimize-days");
        }
        Constraint preferEarlierStart(ConstraintFactory f) {
            return f.forEach(BlockDecision.class)
                .penalize(HardMediumSoftScore.ONE_SOFT, b -> b.startDay == null ? 0 : 1 * b.startDay)
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
    }
    static class ParsedSchedule {
        LocalDate planStart; LocalDate planEnd;
        List<DaySlot> daySlots; List<TaskWindow> windows;
        Map<String,Integer> requiredByKey;
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
            employees.add(new EmployeeFact(eid++, wid, name, skills, isMgr, company));
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
        out.opdef = opdef; out.employees = employees;
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

        List<Map<String,Object>> wfTasks = (List<Map<String,Object>>) s.getOrDefault("workflow_task_list", List.of());
        for (Map<String,Object> wf : wfTasks) {
            String module = safeStr(wf.get("id"));
            String fab = safeStr(wf.get("fab"));
            List<Map<String,Object>> phases = (List<Map<String,Object>>) wf.getOrDefault("phase_task_list", List.of());
            for (Map<String,Object> ph : phases) {
                String phId = safeStr(ph.get("phase"));
                int phNum = phaseNumFromId(phId);
                LocalDate pStart = LocalDate.parse(safeStr(ph.get("start_date")).replace("-", "/"), DF);
                LocalDate pEnd   = LocalDate.parse(safeStr(ph.get("end_date")).replace("-", "/"), DF);
                int startId = (int) (pStart.toEpochDay() - start.toEpochDay());
                int endId   = (int) (pEnd.toEpochDay()   - start.toEpochDay());

                List<Map<String,Object>> opTasks = (List<Map<String,Object>>) ph.getOrDefault("operation_task_list", List.of());
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

        ParsedSchedule out = new ParsedSchedule();
        out.planStart = start; out.planEnd = end;
        out.daySlots = days; out.windows = windows; out.requiredByKey = required;
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
                    DaySlot dd = byId.get(start + off);
                    if (dd != null) seatDays.add(new SeatDay(seatKey, dd, hours, b.factory));
                }
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

    static class Pass1Result { List<BlockDecision> blocks; int tierUsed; HardMediumSoftScore score; boolean hardIsZero; }

    static Set<Integer> detectHardViolators(List<BlockDecision> blocks, List<DaySlot> daySlots) {
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

        Map<Integer, List<BlockDecision>> byDay = new HashMap<>();
        for (BlockDecision b : blocks) {
            if (b.startDay == null || b.days == null || b.heads == null) continue;
            for (int day = b.startDay; day <= b.startDay + b.days - 1; day++) {
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

    static List<BlockDecision> seedBlocksForTier(List<TaskWindow> windows, Map<Integer,Integer> perBlockTier) {
        List<BlockDecision> blocks = new ArrayList<>();
        int bid = 1;
        for (TaskWindow w : windows) {
            List<Integer> full = w.fullAllowedSorted();
            int tier = perBlockTier.getOrDefault(bid, 1);
            List<Integer> tiered = full.subList(0, Math.min(tier, full.size()));

            int baseline = (tiered.size()==1 && tiered.get(0)==4) ? 4 : 8;
            int req = w.workloadDays * baseline;

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
            b.pinned = s.pinned; b.pinStart = s.pinStart; b.pinHeads = s.pinHeads; b.pinDays = s.pinDays; b.pinHours = s.pinHours;
            out.add(b);
        }
        return out;
    }

    // --- write a per-iteration snapshot: assign every seat to employee id=1 ---
    static void writeScheduleSnapshot(
            int iterIndex,
            List<BlockDecision> blocks,
            List<DaySlot> days,
            SnapshotCfg cfg
    ) {
        try {
            if (cfg == null) return;

            // pick employee id=1; if absent, pick first non-zero
            EmployeeFact emp1 = null;
            for (EmployeeFact e : cfg.employees) {
                if (e != null && e.id == 1) { emp1 = e; break; }
            }
            if (emp1 == null) {
                for (EmployeeFact e : cfg.employees) {
                    if (e != null && e.id != 0) { emp1 = e; break; }
                }
            }
            if (emp1 == null) {
                System.out.println("[snapshot] No employee available to write snapshot. Skipping.");
                return;
            }

            Expanded ex = expandToSeats(blocks, days);
            // build a fake Pass2Plan with everyone = emp1
            Pass2Plan snap = new Pass2Plan();
            snap.days = days;
            snap.employees = cfg.employees;
            snap.seatDays = ex.seatDays;
            snap.seats = ex.seats;
            if (snap.seats != null) {
                for (CrewSeat s : snap.seats) s.employee = emp1;
            }

            // target file path: sibling "Schedule{iter}.yaml"
            Path orig = Paths.get(cfg.schedulePath);
            String baseName = "Schedule" + iterIndex + ".yaml";
            Path out = orig.getParent() == null ? Paths.get(baseName) : orig.getParent().resolve(baseName);

            ExportSchedule.overwriteScheduleWithAssignments(snap, cfg.planStart, out.toString());
            System.out.println("[snapshot] Wrote " + out);
        } catch (Exception ex) {
            System.out.println("[snapshot] Failed to write snapshot for loop " + iterIndex + ": " + ex.getMessage());
        }
    }

    // ---------------- Pass 1: SELECTIVE RAMP LOOP with snapshots ----------------

    static Pass1Result solvePass1SelectiveRamp(
            List<DaySlot> daySlots,
            List<TaskWindow> windows,
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
        best.score = null; best.blocks = null; best.tierUsed = 1; best.hardIsZero = false;

        for (int iter = 1; iter <= 50; iter++) {
            List<BlockDecision> blocks = seedBlocksForTier(windows, perBlockTier);

            Pass1Plan p1 = new Pass1Plan();
            p1.dayIds = dayIds; p1.headOptions = headOptions; p1.dayCountOptions = dayCountOptions;
            p1.daySlots = daySlots; p1.blocks = blocks;

            Solver<Pass1Plan> solver = buildSolver(Pass1Plan.class, new Class<?>[]{ BlockDecision.class },
                    Pass1Constraints.class, "0hard/*medium/*soft", 30, 0);
            Pass1Plan solved = solver.solve(p1);

            // snapshot for this iteration (using the just-solved blocks)
            writeScheduleSnapshot(iter, solved.blocks, daySlots, snapshotCfg);

            if (best.score == null ||
                (hardZero(solved.getScore()) && !hardZero(best.score)) ||
                (!hardZero(best.score) && solved.getScore().toString().compareTo(best.score.toString()) < 0)) {
                best.blocks = cloneBlocks(solved.blocks);
                best.score = solved.getScore();
            }

            if (hardZero(solved.getScore())) {
                // polish
                Solver<Pass1Plan> polish = buildSolver(Pass1Plan.class, new Class<?>[]{ BlockDecision.class },
                        Pass1Constraints.class, null, 20, 60);
                Pass1Plan polished = polish.solve(solved);

                // write polished snapshot over the same iter index
                writeScheduleSnapshot(iter, polished.blocks, daySlots, snapshotCfg);

                Pass1Result r = new Pass1Result();
                r.blocks = polished.blocks; r.tierUsed = -1; r.score = polished.getScore(); r.hardIsZero = true;
                return r;
            }

            // Detect hard violators
            Set<Integer> violators = detectHardViolators(solved.blocks, daySlots);

            // Build next-iteration seed: pin non-violators, tier-up violators
            boolean anyTierChange = false;
            List<BlockDecision> next = new ArrayList<>();
            Map<Integer, BlockDecision> solvedById = solved.blocks.stream()
                    .collect(Collectors.toMap(b -> b.id, b -> b));
            int bid = 1;
            for (TaskWindow w : windows) {
                BlockDecision solvedB = solvedById.get(bid);
                List<Integer> full = w.fullAllowedSorted();
                int curTier = perBlockTier.getOrDefault(bid, 1);

                BlockDecision nb = new BlockDecision();
                nb.id = bid;
                nb.module = w.module; nb.factory = w.factory;
                nb.phaseId = w.phaseId; nb.phaseNum = w.phaseNum;
                nb.opId = w.opId;
                nb.windowStart = w.startDayId; nb.windowEnd = w.endDayId;
                nb.requiredHours = (w.workloadDays * ((full.size()==1 && full.get(0)==4)?4:8));
                nb.minHeads = w.minHeads; nb.maxHeads = w.maxHeads;

                if (!violators.contains(bid)) {
                    nb.pinned = true;
                    nb.pinStart = solvedB.startDay;
                    nb.pinHeads = solvedB.heads;
                    nb.pinDays  = solvedB.days;
                    int h = autoHours(solvedB);
                    nb.pinHours = h;
                    nb.allowed = List.of(h);
                    nb.startDay = nb.pinStart;
                    nb.heads    = nb.pinHeads;
                    nb.days     = nb.pinDays;
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
                    int seedDays = Math.max(1, Math.min((nb.requiredHours + safeDen - 1) / safeDen, maxDays));
                    nb.startDay = w.startDayId; nb.heads = minH; nb.days = seedDays;
                }

                next.add(nb);
                bid++;
            }

            if (!anyTierChange) {
                Pass1Result r = new Pass1Result();
                r.blocks = (best.blocks != null ? best.blocks : next);
                r.tierUsed = -1; r.score = (best.score != null ? best.score : solved.getScore());
                r.hardIsZero = hardZero(r.score);
                return r;
            }
        }

        Pass1Result r = new Pass1Result();
        r.blocks = best.blocks; r.tierUsed = -1;
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

        int realEmp = Math.max(1, env.employees.size() - 1);
        int totalReq = sch.requiredByKey.values().stream().mapToInt(Integer::intValue).sum();
        TARGET_HOURS_PER_EMP = totalReq / (double) realEmp;

        SnapshotCfg snapCfg = new SnapshotCfg();
        snapCfg.planStart = sch.planStart;
        snapCfg.schedulePath = schedPath;
        snapCfg.employees = env.employees;

        // ---- PASS 1 ----
        System.out.println("Start PASS 1 (selective ramp + snapshots) at " + nowClock());
        long t1 = System.nanoTime();
        Pass1Result p1 = solvePass1SelectiveRamp(sch.daySlots, sch.windows, snapCfg);
        long t1End = System.nanoTime();
        System.out.printf(
            "Done PASS 1 at %s | duration=%s | score=%s | blocks=%d%n",
            nowClock(),
            fmt(java.time.Duration.ofNanos(t1End - t1)),
            String.valueOf(p1.score),
            p1.blocks == null ? 0 : p1.blocks.size()
        );

        // Build seats pre-Pass2
        Expanded ex = expandToSeats(p1.blocks, sch.daySlots);
        System.out.printf("Expanded to seats: %d seats, %d seat-days%n",
            ex.seats == null ? 0 : ex.seats.size(),
            ex.seatDays == null ? 0 : ex.seatDays.size()
        );

        // ---- PASS 2 ----
        Pass2Plan finalP2;
        if (p1.hardIsZero) {
            System.out.println("Start PASS 2 at " + nowClock());
            long t2 = System.nanoTime();
            finalP2 = solvePass2Once(sch.daySlots, env.employees, ex.seats, ex.seatDays);
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
            finalP2.seats = ex.seats; finalP2.seatDays = ex.seatDays;
        }

        RunResult rr = new RunResult();
        rr.finalPlan = finalP2; rr.planStart = sch.planStart;
        return rr;
    }

    public static void main(String[] args) throws Exception {
        String envPath = args.length > 0 ? args[0] : "EnvConfig.yaml";
        String schedPath = args.length > 1 ? args[1] : "Schedule.yaml";

        RunResult rr = solveFromYaml(envPath, schedPath);

        // Final: overwrite original Schedule.yaml with real (Pass 2) assignments
        ExportSchedule.overwriteScheduleWithAssignments(rr.finalPlan, rr.planStart, schedPath);
        System.out.println("Done.");
    }
}
