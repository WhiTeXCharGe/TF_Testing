// mvn -DskipTests clean package   
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
import ai.timefold.solver.core.api.domain.valuerange.CountableValueRange;
import ai.timefold.solver.core.api.domain.valuerange.ValueRange;
import ai.timefold.solver.core.api.domain.valuerange.ValueRangeFactory;
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
import ai.timefold.solver.core.impl.domain.valuerange.buildin.collection.ListValueRange;
import ai.timefold.solver.core.api.solver.SolutionManager;
import ai.timefold.solver.core.api.score.ScoreExplanation;

import org.yaml.snakeyaml.DumperOptions;
import org.yaml.snakeyaml.Yaml;
import org.yaml.snakeyaml.LoaderOptions;
import org.yaml.snakeyaml.constructor.SafeConstructor;

/**
 * SINGLE PASS scheduler (Timefold 1.27-compatible).
 */
public class EmployeeSchedule {

    // ---------------- YAML I/O ----------------

    @SuppressWarnings("unchecked")
    static Map<String, Object> loadYaml(String path) throws IOException {
        try (InputStream in = Files.newInputStream(Paths.get(path))) {
            LoaderOptions opts = new LoaderOptions();
            opts.setCodePointLimit(5 * 1024 * 1024); // ≈ 5 MB

            Yaml yaml = new Yaml(new SafeConstructor(opts));
            return yaml.load(in);
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

        public Map<String,Integer> regionPreference   = new HashMap<>();
        public Map<String,Integer> customerPreference = new HashMap<>();

        public EmployeeFact() {}
        public EmployeeFact(int id, String wid, String name, Map<String,Integer> skills,
                            boolean isManager, String company) {
            this.id = id; this.wid = wid; this.name = name;
            if (skills != null) this.skills = new HashMap<>(skills);
            this.isManager = isManager; this.workerCompany = company;
            // maps already initialized above
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
    }

    // ---------- Planning Entities (single pass) ----------

    @PlanningEntity
    public static class BlockDecision {
        @PlanningId public int id;

        // facts...
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

        @PlanningVariable(valueRangeProviderRefs = "vrStartWithinWindow",
                  strengthComparatorClass = StartDayStrength.class)
        public Integer startDay;

        @PlanningVariable(valueRangeProviderRefs = "vrDaysWithinWindow")
        public Integer days;

        public static final class StartDayStrength implements Comparator<Integer> {
            @Override public int compare(Integer a, Integer b) {
                // earlier dates first; nulls last so solver can set startDay early
                if (a == null) return (b == null) ? 0 : 1;
                if (b == null) return -1;
                return Integer.compare(a, b);
            }
        }

        public static final class HoursStrength implements Comparator<Integer> {
            @Override public int compare(Integer a, Integer b) {
                // prefer smaller hours (8 < 10 < 12)
                if (a == null) return (b == null) ? 0 : 1;
                if (b == null) return -1;
                return Integer.compare(a, b);
            }
        }

        // --- PATCH: ValueRangeFactory instead of hand-built lists ---
        @PlanningVariable(valueRangeProviderRefs = "vrAllowedHours",
                        strengthComparatorClass = HoursStrength.class)
        public Integer hours; // null at start → falls back to first allowed

        @ValueRangeProvider(id = "vrStartWithinWindow")
        public CountableValueRange<Integer> vrStartWithinWindow() {
            return ValueRangeFactory.createIntValueRange(windowStart, windowEnd + 1);
        }

        @ValueRangeProvider(id = "vrDaysWithinWindow")
        public CountableValueRange<Integer> vrDaysWithinWindow() {
            int maxLen = Math.max(1, windowEnd - windowStart + 1);
            return ValueRangeFactory.createIntValueRange(1, maxLen + 1);
        }

        // NEW: discrete list value range for hours (8,10,12, …)
        @ValueRangeProvider(id = "vrAllowedHours")
        public CountableValueRange<Integer> vrAllowedHours() {
            List<Integer> a = (allowed == null || allowed.isEmpty())
                    ? List.of(8)
                    : allowed.stream().distinct().sorted().toList();
            return new ListValueRange<>(a);
        }


        // helper used in constraints where hours is needed
        public int chosenHours() {
            if (hours != null) return hours;
            // default to smallest allowed to bias toward 8 even before it’s set
            return (allowed == null || allowed.isEmpty()) ? 8
                    : allowed.stream().mapToInt(Integer::intValue).min().orElse(8);
        }
    }

    @PlanningEntity
    public static class CrewSeat {
        @PlanningId public int id;

        public int blockId;
        public String module;
        public String factory;
        public String phaseId;
        public int phaseNum;
        public String opId;

        public int seatIndex;
        public boolean needManager;

        public boolean pinned = false;
        public String  pinnedWid = null;
        public Integer pinnedStart = null;
        public Integer pinnedDays  = null;
        public Integer pinnedHours = null;

        // Computed candidates (filled once per solve stage)
        private List<EmployeeFact> candidateEmployees = List.of();

        private static final EmployeeFact UNASSIGNED =
            new EmployeeFact(0, "__UNASSIGNED__", "__UNASSIGNED__", Map.of(), false, "");

        public CrewSeat() {}

        public void setCandidateEmployees(List<EmployeeFact> list) {
            this.candidateEmployees = (list == null) ? List.of() : list;
        }
        public List<EmployeeFact> getCandidateEmployees() { return candidateEmployees; }

        @ValueRangeProvider(id = "eligibleEmployeesForSeat")
        public CountableValueRange<EmployeeFact> eligibleEmployeesForSeat() {
            // Pinned → pin hard (or UNASSIGNED if you prefer; this doesn’t affect manager logic)
            if (pinned && pinnedWid != null) {
                for (EmployeeFact e : candidateEmployees) {
                    if (e != null && pinnedWid.equals(e.wid)) {
                        return new ListValueRange<>(List.of(e));
                    }
                }
                // For pinned but missing, you can still allow UNASSIGNED here if you want.
                return new ListValueRange<>(List.of(UNASSIGNED));
            }

            List<EmployeeFact> base = (candidateEmployees == null) ? List.of() : candidateEmployees;

            if (needManager) {
                // STRICT: manager-only, no UNASSIGNED fallback
                base = base.stream().filter(emp -> emp != null && emp.isManager).toList();
                // IMPORTANT: do NOT add UNASSIGNED here
                return new ListValueRange<>(base);
            } else {
                // Non-manager seat may keep UNASSIGNED as a legal value to ease feasibility
                if (base.isEmpty()) base = List.of(UNASSIGNED);
                return new ListValueRange<>(base);
            }
        }

        @PlanningVariable(valueRangeProviderRefs = "eligibleEmployeesForSeat")
        public EmployeeFact employee;
    }


    @PlanningSolution
    public static class SinglePassPlan {
        @ProblemFactCollectionProperty
        public List<DaySlot> days;

        @ProblemFactCollectionProperty
        public List<EmployeeFact> employees;

        @PlanningEntityCollectionProperty
        public List<BlockDecision> blocks;

        @PlanningEntityCollectionProperty
        public List<CrewSeat> seats;

        @PlanningScore
        private HardMediumSoftScore score;
        public HardMediumSoftScore getScore() { return score; }
        public void setScore(HardMediumSoftScore s) { this.score = s; }
        public SinglePassPlan() {}
    }

    // ---------------- Globals & helpers ----------------

    static final int DAILY_CAP = 12;
    static double TARGET_HOURS_PER_EMP = 0.0;

    static final Map<String,Integer> OP_CAPACITY = new HashMap<>();
    static final Map<String,Double>  OP_AVG_SKILL = new HashMap<>();


    // fixed schedule background (built once per solveFromYaml)
    static Map<Integer, Map<Integer, Integer>> FIXED_HOURS_BY_EMP_DAY = new HashMap<>();
    static Map<Integer, Map<Integer, Set<String>>> FIXED_FACTORIES_BY_EMP_DAY = new HashMap<>();

    // total fixed hours per employee
    static Map<Integer, Integer> FIXED_TOTAL_HOURS_BY_EMP = new HashMap<>();

    // fixed overtime (max(0, hours-8)) per employee per year
    static Map<Integer, Map<Integer, Integer>> FIXED_ANNUAL_OT_BY_EMP_YEAR = new HashMap<>();

    // fixed overtime per employee per year-month (ym = year*100 + month)
    static Map<Integer, Map<Integer, Integer>> FIXED_MONTHLY_OT_BY_EMP_YM = new HashMap<>();

    public static final boolean TRIM_FINISHED_MODULES = true;
    // if module ended more than this many days before cutOffDate,
    // it is considered finished and can be trimmed.
    public static final int MODULE_TRIM_GRACE_DAYS = 6; // x days, change as you like

    static long EXTERNAL_RANDOM_SEED = 0L;

    static boolean isUnassigned(EmployeeFact e) { return e == null || e.id == 0; }
    static int skill(EmployeeFact e, String opId) { return (e == null) ? 0 : e.skills.getOrDefault(opId, 0); }
    static boolean isManager(EmployeeFact e) { return e != null && e.isManager; }
    static String company(EmployeeFact e) { return e == null ? "" : (e.workerCompany == null ? "" : e.workerCompany); }
    static double avgSkill(String opId) { return OP_AVG_SKILL.getOrDefault(opId, 3.0); }

    static int regionPref(EmployeeFact e, String regionId) {
        if (e == null || regionId == null || regionId.isBlank()) return 1;
        return e.regionPreference.getOrDefault(regionId, 1);
    }
    static int customerPref(EmployeeFact e, String customerId) {
        if (e == null || customerId == null || customerId.isBlank()) return 1;
        return e.customerPreference.getOrDefault(customerId, 1);
    }

    static int autoHours(BlockDecision b) {
        List<Integer> allowed = (b.allowed == null || b.allowed.isEmpty())
                ? List.of(8) : b.allowed.stream().sorted().collect(Collectors.toList());
        int D = (b.startDay == null || b.days == null) ? 0 : workingDaysCount(b.startDay, b.days, b.factory);
        if (D == 0) return allowed.get(0);
        int H = 1; // per-seat
        int R = Math.max(1, b.requiredHours);
        int best = allowed.get(0);
        int[] bestKey = null; // [mode, penalty, |h-8|, h]
        for (int h : allowed) {
            int prod = H * h * D;
            int[] key;
            if (prod < R) key = new int[]{0, R - prod, Math.abs(h - 8), h};
            else {
                int extra = Math.max(0, (prod - R) - H * h);
                key = new int[]{1, extra, Math.abs(h - 8), h};
            }
            if (bestKey == null || compareTuple(key, bestKey) < 0) { bestKey = key; best = h; }
        }
        return best;
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
        Map<String, Map<String, Integer>> transitDays = new HashMap<>(); // fromRegion -> (toRegion -> days)
        Map<String, Integer> regionStayMaxOn = new HashMap<>();
        Map<String, Integer> regionStayOffInterval = new HashMap<>();

        // NEW: annual-stay per region
        Map<String, Integer> regionAnnualMaxStay = new HashMap<>();

        // NEW: overtime limits per worker company (used per-worker)
        Map<String, Integer> workerCompanyAnnualOtLimit = new HashMap<>();
        Map<String, Integer> workerCompanyMonthlyOtLimit = new HashMap<>();

        int transitDays(String from, String to) {
            if (from == null || to == null || from.equals(to)) return 0;
            return transitDays.getOrDefault(from, Map.of()).getOrDefault(to, 0);
        }
        String regionOfFab(String fabId) { return fabId == null ? null : fabToRegion.get(fabId); }
        String customerOfFab(String fabId) { return fabId == null ? null : fabToCustomer.get(fabId); }

        int maxStayOn(String regionId) { return regionStayMaxOn.getOrDefault(regionId, Integer.MAX_VALUE); }
        int stayOffInterval(String regionId) { return Math.max(1, regionStayOffInterval.getOrDefault(regionId, 1)); }

        // NEW: helper getters
        int annualMaxStay(String regionId) {
            return regionAnnualMaxStay.getOrDefault(regionId, Integer.MAX_VALUE);
        }
        int annualOtLimit(String companyId) {
            return workerCompanyAnnualOtLimit.getOrDefault(companyId, Integer.MAX_VALUE);
        }
        int monthlyOtLimit(String companyId) {
            return workerCompanyMonthlyOtLimit.getOrDefault(companyId, Integer.MAX_VALUE);
        }
    }

    static Calendars CAL = new Calendars();

    private static final java.time.format.DateTimeFormatter CLOCK =
            java.time.format.DateTimeFormatter.ofPattern("HH:mm:ss");
    private static String nowClock() { return java.time.LocalTime.now().format(CLOCK); }
    private static String fmt(java.time.Duration d) {
        long h = d.toHours(); long m = d.toMinutesPart(); long s = d.toSecondsPart(); long ms = d.toMillisPart();
        return String.format("%02d:%02d:%02d.%03d", h, m, s, ms);
    }

    static Integer dayIdFromDate(LocalDate planStart, String ymd) {
        LocalDate d = LocalDate.parse(ymd.replace("-", "/"), DF);
        return (int) (d.toEpochDay() - planStart.toEpochDay());
    }

    @SuppressWarnings("unchecked")
    private static Set<Integer> parseUnavailableDates(Object raw,
                                                    LocalDate planStart,
                                                    LocalDate planEnd) {
        Set<Integer> off = new HashSet<>();
        if (raw == null) return off;

        int horizon = (int) (planEnd.toEpochDay() - planStart.toEpochDay()) + 1;

        // For weekly patterns we need the actual dates in horizon
        List<LocalDate> allDates = new ArrayList<>(horizon);
        for (int i = 0; i < horizon; i++) {
            allDates.add(planStart.plusDays(i));
        }

        // Helper: convert "sat", "sun", "mon" etc to DayOfWeek
        java.util.function.Function<String, java.time.DayOfWeek> parseWeekday = (s) -> {
            if (s == null) return null;
            String t = s.trim().toLowerCase(java.util.Locale.ROOT);
            switch (t) {
                case "mon": case "monday":    return java.time.DayOfWeek.MONDAY;
                case "tue": case "tuesday":   return java.time.DayOfWeek.TUESDAY;
                case "wed": case "wednesday": return java.time.DayOfWeek.WEDNESDAY;
                case "thu": case "thursday":  return java.time.DayOfWeek.THURSDAY;
                case "fri": case "friday":    return java.time.DayOfWeek.FRIDAY;
                case "sat": case "saturday":  return java.time.DayOfWeek.SATURDAY;
                case "sun": case "sunday":    return java.time.DayOfWeek.SUNDAY;
                default: return null;
            }
        };

        // ---- Normalize raw into a List<?> ----
        // Supported shapes:
        //   unavailable_dates: 2025/09/10
        //   unavailable_dates: [2025/09/10, 2025/09/11]
        //   unavailable_dates:
        //     - { single: { days: [...] } }
        //     - { weekly: { weekdays: [...] } }
        //   unavailable_dates:
        //     weekly:
        //       weekdays: [sat, sun]
        List<?> list;
        if (raw instanceof List<?> l) {
            list = l;
        } else if (raw instanceof Map<?,?> m) {
            // Top-level map like {weekly: {weekdays:[sat,sun]}} or {single:{...}}
            list = List.of(m);
        } else {
            // Single scalar date string
            try {
                Integer did = dayIdFromDate(planStart, String.valueOf(raw));
                if (did != null && did >= 0 && did < horizon) {
                    off.add(did);
                }
            } catch (Exception ignore) {}
            return off;
        }

        // We support:
        // - "2025/09/10" style entries directly in the list
        // - { single: { days: [...] } }
        // - { weekly: { weekdays: [sat, sun] } }
        Set<java.time.DayOfWeek> weeklyOff = new HashSet<>();

        for (Object item : list) {
            if (item == null) continue;

            if (item instanceof Map<?,?> map) {
                // --- single: { days: [...] } ---
                Object singleObj = map.get("single");
                if (singleObj instanceof Map<?,?> singleMap) {
                    Object daysObj = singleMap.get("days");
                    if (daysObj instanceof List<?> daysList) {
                        for (Object dObj : daysList) {
                            try {
                                Integer did = dayIdFromDate(planStart, String.valueOf(dObj));
                                if (did != null && did >= 0 && did < horizon) {
                                    off.add(did);
                                }
                            } catch (Exception ignore) {}
                        }
                    }
                }

                // --- weekly: { weekdays: [sat, sun] } ---
                Object weeklyObj = map.get("weekly");
                if (weeklyObj instanceof Map<?,?> weeklyMap) {
                    Object wdaysObj = weeklyMap.get("weekdays");
                    if (wdaysObj instanceof List<?> wdaysList) {
                        for (Object wd : wdaysList) {
                            java.time.DayOfWeek dow = parseWeekday.apply(String.valueOf(wd));
                            if (dow != null) weeklyOff.add(dow);
                        }
                    }
                }
            } else {
                // plain string in the list, treat as single date
                try {
                    Integer did = dayIdFromDate(planStart, String.valueOf(item));
                    if (did != null && did >= 0 && did < horizon) {
                        off.add(did);
                    }
                } catch (Exception ignore) {}
            }
        }

        // Expand weekly patterns into actual dayIds within the horizon
        if (!weeklyOff.isEmpty()) {
            for (int i = 0; i < horizon; i++) {
                LocalDate d = allDates.get(i);
                if (weeklyOff.contains(d.getDayOfWeek())) {
                    off.add(i);
                }
            }
        }

        return off;
    }


    @SuppressWarnings("unchecked")
    public static void buildCalendars(String envPath, LocalDate planStart, LocalDate planEnd) throws IOException {
        CAL = new Calendars();

        int horizon = (int) (planEnd.toEpochDay() - planStart.toEpochDay()) + 1;

        Map<String,Object> root;
        try (InputStream in = Files.newInputStream(Paths.get(envPath))) {
            LoaderOptions opts = new LoaderOptions();
            opts.setCodePointLimit(5 * 1024 * 1024);
            Yaml yaml = new Yaml(new SafeConstructor(opts));
            root = yaml.load(in);
        }
        Map<String,Object> env = (Map<String,Object>) root.getOrDefault("environment", root);

        List<Map<String,Object>> fabs =
                (List<Map<String,Object>>) env.getOrDefault("fab_list", List.of());
        for (Map<String,Object> f : fabs) {
            String fid = String.valueOf(f.get("id"));
            String rid = String.valueOf(f.get("region"));
            String cid = String.valueOf(f.get("customer_company"));
            CAL.fabToRegion.put(fid, rid);
            CAL.fabToCustomer.put(fid, cid);

            Set<Integer> off = parseUnavailableDates(
                    f.get("unavailable_dates"), planStart, planEnd);
            CAL.fabOff.put(fid, off);
        }

        List<Map<String,Object>> custs =
                (List<Map<String,Object>>) env.getOrDefault("customer_company_list", List.of());
        for (Map<String,Object> c : custs) {
            String cid = String.valueOf(c.get("id"));
            Set<Integer> off = parseUnavailableDates(
                    c.get("unavailable_dates"), planStart, planEnd);
            CAL.customerOff.put(cid, off);
        }

        List<Map<String,Object>> wcomps =
                (List<Map<String,Object>>) env.getOrDefault("worker_company_list", List.of());
        for (Map<String,Object> wc : wcomps) {
            String cid = String.valueOf(wc.get("id"));

            Set<Integer> off = parseUnavailableDates(
                    wc.get("unavailable_dates"), planStart, planEnd);
            CAL.workerCompanyOff.put(cid, off);

            int annualOt  = parseInt(wc.get("annual_overtime_limit"), Integer.MAX_VALUE);
            int monthlyOt = parseInt(wc.get("monthly_overtime_limit"), Integer.MAX_VALUE);
            CAL.workerCompanyAnnualOtLimit.put(cid, annualOt);
            CAL.workerCompanyMonthlyOtLimit.put(cid, monthlyOt);
        }

        List<Map<String,Object>> workers =
                (List<Map<String,Object>>) env.getOrDefault("worker_list", List.of());
        for (Map<String,Object> w : workers) {
            String wid = String.valueOf(w.get("id"));
            Set<Integer> off = parseUnavailableDates(
                    w.get("unavailable_dates"), planStart, planEnd);
            CAL.workerOffByWid.put(wid, off);
        }

        List<Map<String,Object>> tmap = (List<Map<String,Object>>) env.getOrDefault("transite_day_map", List.of());
        for (Map<String,Object> t : tmap) {
            String from = String.valueOf(t.get("from"));
            String to   = String.valueOf(t.get("to"));
            int days    = parseInt(t.get("days"), 0);
            if (from != null && to != null && !from.isBlank() && !to.isBlank() && days > 0) {
                CAL.transitDays.computeIfAbsent(from, k -> new HashMap<>()).put(to, days);
            }
        }

        List<Map<String,Object>> regionList =
                (List<Map<String,Object>>) env.getOrDefault("region_list", List.of());
        for (Map<String,Object> r : regionList) {
            String rid = String.valueOf(r.get("id"));

            // region-level unavailable dates (weekly + single supported)
            Set<Integer> off = parseUnavailableDates(
                    r.get("unavailable_dates"), planStart, planEnd);
            CAL.regionOff.put(rid, off);

            int maxStayOn   = parseInt(r.get("max_stay_on"), Integer.MAX_VALUE);
            int maxAnnual   = parseInt(r.get("max_annual_stay"), Integer.MAX_VALUE);
            int offInterval = parseInt(r.get("stay_off_interval"), 1);

            CAL.regionStayMaxOn.put(rid, maxStayOn);
            CAL.regionAnnualMaxStay.put(rid, maxAnnual);
            CAL.regionStayOffInterval.put(rid, Math.max(1, offInterval));
        }

    }

    static boolean isWorkingDay(int dayId, String fabId) {
        if (dayId < 0) return false;
        if (fabId == null) return true;

        if (CAL.fabOff.getOrDefault(fabId, Set.of()).contains(dayId)) return false;
        String rid = CAL.fabToRegion.get(fabId);
        if (rid != null && CAL.regionOff.getOrDefault(rid, Set.of()).contains(dayId)) return false;
        String cid = CAL.fabToCustomer.get(fabId);
        if (cid != null && CAL.customerOff.getOrDefault(cid, Set.of()).contains(dayId)) return false;
        return true;
    }


    // IMPORTANT: Integer (nullable) parameters here
    static int workingDaysCount(Integer startDay, Integer dayCount, String fabId) {
        if (startDay == null || dayCount == null || startDay < 0 || dayCount == 0) return 0;
        int end = startDay + dayCount - 1;
        int n = 0;
        for (int d = startDay; d <= end; d++) if (isWorkingDay(d, fabId)) n++;
        return n;
    }

    // ---------------- Constraints ----------------

    public static class SinglePassConstraints implements ConstraintProvider {
        // soft weights
        static final int PREF_HOURS_WEIGHT = 3000;
        static final int SMALLER_HOURS_W   = 40;
        static final int EARLIER_START_W   = 1;
        static final int COMPANY_PAIR_W    = 5;
        static final int SKILL_DIVERSITY_W = 3;
        static final int SKILL_AVG_W       = 50;
        static final int BASE_HOURS_PER_DAY = 8;
        static final int CONTINUOUS_REGION_STAY_W = 5;
        static final int FAB_PREFERENCE_W  = 5;

        @Override public Constraint[] defineConstraints(ConstraintFactory f) {
            return new Constraint[] {
                // block feasibility
                // withinWindow(f),
                // daysWithinWindowLen(f),
                endWithinWindow(f),
                hoursValueAllowed(f),
                phaseOrder(f),

                // production & capacity (count only staffed seats)
                noUnderfillByBlock(f),
                overfillAtMostOneDayByBlock(f),
                dailyHeadCapacityByOp(f),

                // seat-level hard rules
                // assignedAndSkill(f),
                employeeAvailableAllDays(f),
                pinnedRespected(f), //2
                oneFactoryPerEmpPerDay(f),
                dailyCap12h(f),
                // atLeastOneManagerPerBlock(f),
                regionTransitGap(f), //1
                regionStayMaxOn(f), //1
                regionAnnualStayMax(f),
                annualOvertimeLimit(f),  //1 NEW
                monthlyOvertimeLimit(f), //1 NEW

                // softs
                // preferHoursNear8(f),
                preferSmallerHours(f), //3
                preferEarlierStart(f), //3
                softSameCompanyPairs(f), //3
                softEncourageSkillVariety(f), //3
                softBalanceBlockAvgSkill(f), //3
                softBalanceTotalHours(f), //3
                softContinuousRegionStay(f), //3
                softFabPreference(f),
            };
        }

        // ---------- Block-level guards ----------
        Constraint withinWindow(ConstraintFactory f) {
            return f.forEach(BlockDecision.class)
                .filter(b -> b.startDay == null || b.days == null
                        || b.startDay < b.windowStart
                        || (b.startDay + b.days - 1) > b.windowEnd
                        || b.days < 1)
                .penalize(HardMediumSoftScore.ONE_HARD)
                .asConstraint("block-within-window");
        }

        Constraint daysWithinWindowLen(ConstraintFactory f) {
            return f.forEach(BlockDecision.class)
                .filter(b -> b.days != null && b.days > (b.windowEnd - b.windowStart + 1))
                .penalize(HardMediumSoftScore.ONE_HARD, b -> b.days - (b.windowEnd - b.windowStart + 1))
                .asConstraint("block-days-window-length");
        }
        
        Constraint endWithinWindow(ConstraintFactory f) {
            return f.forEach(BlockDecision.class)
                .filter(b -> b.startDay != null && b.days != null
                        && (b.startDay + b.days - 1) > b.windowEnd)
                .penalize(HardMediumSoftScore.ONE_HARD,
                    b -> (b.startDay + b.days - 1) - b.windowEnd) // how far you overran
                .asConstraint("block-end-within-window");
        }

        Constraint hoursValueAllowed(ConstraintFactory f) {
            return f.forEach(BlockDecision.class)
                .filter(b -> b.allowed == null || b.allowed.isEmpty() || !b.allowed.contains(b.chosenHours()))
                .penalize(HardMediumSoftScore.ONE_HARD)
                .asConstraint("block-hours-in-allowed");
        }

        Constraint phaseOrder(ConstraintFactory f) {
            return f.forEach(BlockDecision.class)
                .join(f.forEach(BlockDecision.class),
                    Joiners.equal((BlockDecision a) -> a.module, (BlockDecision b) -> b.module),
                    Joiners.equal((BlockDecision a) -> a.phaseNum + 1, (BlockDecision b) -> b.phaseNum))
                .filter((a,b) -> a.startDay != null && a.days != null && b.startDay != null
                        && (a.startDay + a.days - 1) >= b.startDay)
                .penalize(HardMediumSoftScore.ONE_HARD, (a,b) -> (a.startDay + a.days - 1) - b.startDay + 1)
                .asConstraint("phase-order");
        }

        // ---------- helpers ----------
        private static int staffedCountForBlock(List<CrewSeat> seats) {
            int c = 0;
            for (CrewSeat s : seats) if (!isUnassigned(s.employee)) c++;
            return c;
        }

        private static boolean seatCoversDayAndWorking(DaySlot d, CrewSeat s, BlockDecision b) {
            final boolean pinned = s.pinned;
            final Integer start = pinned ? s.pinnedStart : b.startDay;
            final Integer days  = pinned ? s.pinnedDays  : b.days;
            if (start == null || days == null || days <= 0) return false;
            return start <= d.id && d.id <= (start + days - 1) && isWorkingDay(d.id, s.factory);
        }

        // ---------- Production from staffed seats ----------
        Constraint noUnderfillByBlock(ConstraintFactory f) {
            var perBlock = f.forEach(BlockDecision.class)
                .join(f.forEach(CrewSeat.class),
                    Joiners.equal((BlockDecision b) -> b.id, (CrewSeat s) -> s.blockId))
                .groupBy((b, s) -> b,
                        ConstraintCollectors.toList((b, s) -> s));

            return perBlock
                .filter((b, seats) -> {
                    int D = workingDaysCount(b.startDay, b.days, b.factory);
                    int hours = b.chosenHours();
                    int staffed = staffedCountForBlock(seats);
                    int prod = staffed * hours * Math.max(0, D);
                    return prod < b.requiredHours;
                })
                .penalize(HardMediumSoftScore.ONE_HARD,
                    (b, seats) -> {
                        int D = workingDaysCount(b.startDay, b.days, b.factory);
                        int hours = b.chosenHours();
                        int staffed = staffedCountForBlock(seats);
                        int prod = staffed * hours * Math.max(0, D);
                        return b.requiredHours - prod;
                    })
                .asConstraint("block-no-underfill");
        }

        Constraint overfillAtMostOneDayByBlock(ConstraintFactory f) {
            var perBlock = f.forEach(BlockDecision.class)
                .join(f.forEach(CrewSeat.class),
                    Joiners.equal((BlockDecision b) -> b.id, (CrewSeat s) -> s.blockId))
                .groupBy((b, s) -> b,
                        ConstraintCollectors.toList((b, s) -> s));

            return perBlock
                .filter((b, seats) -> {
                    int D = workingDaysCount(b.startDay, b.days, b.factory);
                    int hours = b.chosenHours();
                    int staffed = staffedCountForBlock(seats);
                    int prod = staffed * hours * Math.max(0, D);
                    int over = prod - b.requiredHours;
                    return over > staffed * hours; // more than one extra day worth
                })
                .penalize(HardMediumSoftScore.ONE_HARD,
                    (b, seats) -> {
                        int D = workingDaysCount(b.startDay, b.days, b.factory);
                        int hours = b.chosenHours();
                        int staffed = staffedCountForBlock(seats);
                        int prod = staffed * hours * Math.max(0, D);
                        int over = prod - b.requiredHours;
                        return Math.max(0, over - staffed * hours);
                    })
                .asConstraint("block-overfill-at-most-one-day");
        }

        // Sum heads by (day, op) for staffed seats whose span covers the day
        Constraint dailyHeadCapacityByOp(ConstraintFactory f) {
            return f.forEach(DaySlot.class)
                .join(f.forEach(CrewSeat.class),
                    Joiners.filtering((DaySlot d, CrewSeat s) -> !isUnassigned(s.employee)))
                .join(f.forEach(BlockDecision.class),
                    Joiners.equal((DaySlot d, CrewSeat s) -> s.blockId, (BlockDecision b) -> b.id))
                .filter(SinglePassConstraints::seatCoversDayAndWorking)
                .groupBy((d, s, b) -> d.id,
                         (d, s, b) -> s.opId,
                         ConstraintCollectors.sum((d, s, b) -> 1))
                .filter((dayId, opId, heads) -> heads > OP_CAPACITY.getOrDefault(opId, Integer.MAX_VALUE))
                .penalize(HardMediumSoftScore.ONE_HARD,
                    (dayId, opId, heads) -> heads - OP_CAPACITY.getOrDefault(opId, Integer.MAX_VALUE))
                .asConstraint("daily-head-capacity-by-op");
        }

        // ---------- Seat hard rules ----------
        Constraint assignedAndSkill(ConstraintFactory f) {
            return f.forEach(CrewSeat.class)
                .filter(s -> isUnassigned(s.employee) || skill(s.employee, s.opId) < 1)
                .penalize(HardMediumSoftScore.ONE_HARD)
                .asConstraint("seat-assigned+eligible-skill");
        }

        Constraint employeeAvailableAllDays(ConstraintFactory f) {
            return f.forEach(CrewSeat.class)
                .join(f.forEach(BlockDecision.class),
                    Joiners.equal((CrewSeat s) -> s.blockId, (BlockDecision b) -> b.id))
                .filter((s, b) -> !isUnassigned(s.employee))
                .filter((s, b) -> {
                    final boolean pinned = s.pinned;
                    final Integer start = pinned ? s.pinnedStart : b.startDay;
                    final Integer days  = pinned ? s.pinnedDays  : b.days;
                    if (start == null || days == null || days <= 0) return false;

                    Set<Integer> off = CAL.workerOffByWid.getOrDefault(s.employee.wid, Set.of());
                    for (int di = 0; di < days; di++) {
                        int did = start + di;
                        if (!isWorkingDay(did, s.factory)) continue;
                        if (off.contains(did)) return true; // violation
                    }
                    return false;
                })
                .penalize(HardMediumSoftScore.ONE_HARD)
                .asConstraint("seat-worker-available-all-days");
        }

        Constraint pinnedRespected(ConstraintFactory f) {
            return f.forEach(CrewSeat.class)
                .filter(s -> s.pinned)
                .filter(s -> s.employee == null || s.employee.wid == null || !s.employee.wid.equals(s.pinnedWid))
                .penalize(HardMediumSoftScore.ONE_HARD)
                .asConstraint("seat-pinned-respected");
        }

        Constraint oneFactoryPerEmpPerDay(ConstraintFactory f) {
            return f.forEach(DaySlot.class)
                .join(f.forEach(CrewSeat.class),
                    // NEW: ignore unassigned AND pinned; fixed fabs are taken from FIXED_FACTORIES_BY_EMP_DAY
                    Joiners.filtering((DaySlot d, CrewSeat s) ->
                        !isUnassigned(s.employee) && !s.pinned))
                .join(f.forEach(BlockDecision.class),
                    Joiners.equal((DaySlot d, CrewSeat s) -> s.blockId, (BlockDecision b) -> b.id))
                .filter(SinglePassConstraints::seatCoversDayAndWorking)
                .groupBy((d, s, b) -> Arrays.asList(s.employee.id, d.id),
                        ConstraintCollectors.toSet((d, s, b) -> s.factory))
                .filter((key, dynamicFabs) -> {
                    int empId = (int) key.get(0);
                    int dayId = (int) key.get(1);

                    Map<Integer, Set<String>> byDay =
                            FIXED_FACTORIES_BY_EMP_DAY.get(empId);
                    Set<String> fixedFabs = (byDay == null)
                            ? Collections.emptySet()
                            : byDay.getOrDefault(dayId, Collections.emptySet());

                    if (fixedFabs.isEmpty()) {
                        return dynamicFabs.size() > 1;
                    }

                    Set<String> all = new HashSet<>(dynamicFabs);
                    all.addAll(fixedFabs);
                    return all.size() > 1;
                })
                .penalize(HardMediumSoftScore.ONE_HARD, (key, dynamicFabs) -> {
                    int empId = (int) key.get(0);
                    int dayId = (int) key.get(1);

                    Map<Integer, Set<String>> byDay =
                            FIXED_FACTORIES_BY_EMP_DAY.get(empId);
                    Set<String> fixedFabs = (byDay == null)
                            ? Collections.emptySet()
                            : byDay.getOrDefault(dayId, Collections.emptySet());

                    Set<String> all = new HashSet<>(dynamicFabs);
                    all.addAll(fixedFabs);
                    return all.size() - 1;
                })
                .asConstraint("seat-one-factory-per-emp-day");
        }


        Constraint dailyCap12h(ConstraintFactory f) {
            return f.forEach(DaySlot.class)
                .join(f.forEach(CrewSeat.class),
                    // NEW: ignore pinned, hours come from FIXED_HOURS_BY_EMP_DAY instead
                    Joiners.filtering((DaySlot d, CrewSeat s) ->
                        !isUnassigned(s.employee) && !s.pinned))
                .join(f.forEach(BlockDecision.class),
                    Joiners.equal((DaySlot d, CrewSeat s) -> s.blockId, (BlockDecision b) -> b.id))
                .filter(SinglePassConstraints::seatCoversDayAndWorking)
                .groupBy((d, s, b) -> Arrays.asList(s.employee.id, d.id),
                        ConstraintCollectors.sum((d, s, b) -> b.chosenHours()))
                .filter((key, dynamicHours) -> {
                    int empId = (int) key.get(0);
                    int dayId = (int) key.get(1);

                    int fixedHours = FIXED_HOURS_BY_EMP_DAY
                            .getOrDefault(empId, Collections.emptyMap())
                            .getOrDefault(dayId, 0);

                    int total = dynamicHours + fixedHours;
                    return total > DAILY_CAP;
                })
                .penalize(HardMediumSoftScore.ONE_HARD, (key, dynamicHours) -> {
                    int empId = (int) key.get(0);
                    int dayId = (int) key.get(1);

                    int fixedHours = FIXED_HOURS_BY_EMP_DAY
                            .getOrDefault(empId, Collections.emptyMap())
                            .getOrDefault(dayId, 0);

                    int total = dynamicHours + fixedHours;
                    return total - DAILY_CAP;
                })
                .asConstraint("seat-daily-cap-12h");
        }



        Constraint atLeastOneManagerPerBlock(ConstraintFactory f) {
            return f.forEach(CrewSeat.class)
                .filter(s -> !isUnassigned(s.employee))
                .groupBy(s -> s.blockId, ConstraintCollectors.sum(s -> isManager(s.employee) ? 1 : 0))
                .filter((blockId, mgrCount) -> mgrCount < 1)
                .penalize(HardMediumSoftScore.ONE_HARD)
                .asConstraint("block-at-least-one-manager");
        }

        // Helper record for Uni↔Uni join
        static final class EmpDay {
            final int empId;
            final int dayId;
            final String factory;
            EmpDay(int e, int d, String f) { empId = e; dayId = d; factory = f; }
            int empId() { return empId; }
            int dayId() { return dayId; }
            String factory() { return factory; }
        }

        // ---- Region transit (map Tri -> Uni then join Uni↔Uni)
        Constraint regionTransitGap(ConstraintFactory f) {
            var empDayUni = f.forEach(DaySlot.class)
                .join(f.forEach(CrewSeat.class),
                    Joiners.filtering((DaySlot d, CrewSeat s) -> !isUnassigned(s.employee)))
                .join(f.forEach(BlockDecision.class),
                    Joiners.equal((DaySlot d, CrewSeat s) -> s.blockId, (BlockDecision b) -> b.id))
                .filter(SinglePassConstraints::seatCoversDayAndWorking)
                .groupBy((d, s, b) -> s.employee.id,
                         (d, s, b) -> d.id,
                         (d, s, b) -> s.factory)
                .map((empId, dayId, factory) -> new EmpDay(empId, dayId, factory));

            return empDayUni
                .join(empDayUni,
                    Joiners.equal(EmpDay::empId, EmpDay::empId),
                    Joiners.lessThan(EmpDay::dayId, EmpDay::dayId))
                .filter((a, b) -> {
                    String r1 = CAL.regionOfFab(a.factory());
                    String r2 = CAL.regionOfFab(b.factory());
                    int need = CAL.transitDays(r1, r2);
                    if (need <= 0) return false;
                    int delta = b.dayId() - a.dayId();
                    return delta <= need;
                })
                .penalize(HardMediumSoftScore.ONE_HARD,
                    (a, b) -> {
                        String r1 = CAL.regionOfFab(a.factory());
                        String r2 = CAL.regionOfFab(b.factory());
                        int need  = CAL.transitDays(r1, r2);
                        int delta = b.dayId() - a.dayId();
                        return Math.max(1, need - delta + 1);
                    })
                .asConstraint("emp-region-transit-gap");
        }

        private static int maxSegmentSpanWithBreak(List<Integer> dayList, int offInterval) {
            if (dayList == null || dayList.isEmpty()) return 0;
            int brk = Math.max(1, offInterval);
            List<Integer> ds = new ArrayList<>(dayList);
            Collections.sort(ds);
            int best = 1, segStart = ds.get(0), prev = ds.get(0);
            for (int i=1;i<ds.size();i++){
                int d=ds.get(i); int gap=d-prev-1;
                if (gap >= brk) { best = Math.max(best, prev - segStart + 1); segStart = d; }
                prev = d;
            }
            best = Math.max(best, prev - segStart + 1);
            return best;
        }

        Constraint regionStayMaxOn(ConstraintFactory f) {
            var items = f.forEach(DaySlot.class)
                .join(f.forEach(CrewSeat.class),
                    Joiners.filtering((DaySlot d, CrewSeat s) -> !isUnassigned(s.employee)))
                .join(f.forEach(BlockDecision.class),
                    Joiners.equal((DaySlot d, CrewSeat s) -> s.blockId, (BlockDecision b) -> b.id))
                .filter(SinglePassConstraints::seatCoversDayAndWorking)
                .groupBy((d, s, b) -> Arrays.asList(s.employee.id, CAL.regionOfFab(s.factory)),
                         ConstraintCollectors.toList((d, s, b) -> d.id))
                .filter((key, dayList) -> key.get(1) != null);

            return items
                .filter((key, dayList) -> {
                    String regionId = (String) key.get(1);
                    int maxOn  = CAL.maxStayOn(regionId);
                    int offInt = CAL.stayOffInterval(regionId);
                    if (maxOn == Integer.MAX_VALUE) return false;
                    int maxSpan = maxSegmentSpanWithBreak(dayList, offInt);
                    return maxSpan > maxOn;
                })
                .penalize(HardMediumSoftScore.ONE_HARD,
                    (key, dayList) -> {
                        String regionId = (String) key.get(1);
                        int maxOn  = CAL.maxStayOn(regionId);
                        int offInt = CAL.stayOffInterval(regionId);
                        int maxSpan = maxSegmentSpanWithBreak(dayList, offInt);
                        return Math.max(1, maxSpan - maxOn);
                    })
                .asConstraint("emp-region-stay-max-on");
        }

        private static int totalSegmentSpanWithBreak(List<Integer> dayList, int offInterval) {
            if (dayList == null || dayList.isEmpty()) return 0;
            int brk = Math.max(1, offInterval);

            // deduplicate days just in case one person has multiple seats on same day/region
            List<Integer> ds = new ArrayList<>(new HashSet<>(dayList));
            Collections.sort(ds);

            int total = 0;
            int segStart = ds.get(0);
            int prev     = ds.get(0);

            for (int i = 1; i < ds.size(); i++) {
                int d   = ds.get(i);
                int gap = d - prev - 1;
                if (gap >= brk) {
                    // close previous segment
                    total += (prev - segStart + 1);
                    segStart = d;
                }
                prev = d;
            }
            // last segment
            total += (prev - segStart + 1);
            return total;
        }

        Constraint regionAnnualStayMax(ConstraintFactory f) {
            var items = f.forEach(DaySlot.class)
                .join(f.forEach(CrewSeat.class),
                    Joiners.filtering((DaySlot d, CrewSeat s) -> !isUnassigned(s.employee)))
                .join(f.forEach(BlockDecision.class),
                    Joiners.equal((DaySlot d, CrewSeat s) -> s.blockId, (BlockDecision b) -> b.id))
                .filter(SinglePassConstraints::seatCoversDayAndWorking)
                .groupBy((d, s, b) -> Arrays.asList(s.employee.id, CAL.regionOfFab(s.factory)),
                        ConstraintCollectors.toList((d, s, b) -> d.id))
                .filter((key, dayList) -> key.get(1) != null);

            return items
                .filter((key, dayList) -> {
                    String regionId = (String) key.get(1);
                    int maxAnnual = CAL.annualMaxStay(regionId);
                    int offInt    = CAL.stayOffInterval(regionId);
                    if (maxAnnual == Integer.MAX_VALUE) return false;
                    int totalSpan = totalSegmentSpanWithBreak(dayList, offInt);
                    return totalSpan > maxAnnual;
                })
                .penalize(
                    HardMediumSoftScore.ONE_HARD,
                    (key, dayList) -> {
                        String regionId = (String) key.get(1);
                        int maxAnnual = CAL.annualMaxStay(regionId);
                        int offInt    = CAL.stayOffInterval(regionId);
                        int totalSpan = totalSegmentSpanWithBreak(dayList, offInt);
                        return Math.max(1, totalSpan - maxAnnual);
                    }
                )
                .asConstraint("emp-region-annual-stay-max");
        }


        Constraint annualOvertimeLimit(ConstraintFactory f) {
            var dynEmpYearOt = f.forEach(DaySlot.class)
                .join(f.forEach(CrewSeat.class),
                    Joiners.filtering((DaySlot d, CrewSeat s) ->
                        !isUnassigned(s.employee)))
                .join(f.forEach(BlockDecision.class),
                    Joiners.equal((DaySlot d, CrewSeat s) -> s.blockId, (BlockDecision b) -> b.id))
                .filter(SinglePassConstraints::seatCoversDayAndWorking)
                .groupBy(
                    (d, s, b) -> Arrays.asList(
                        s.employee.id,
                        company(s.employee),
                        d.date.getYear()
                    ),
                    ConstraintCollectors.sum((d, s, b) -> {
                        if (s.pinned) {
                            // hours for fixed part are loaded from FIXED_ANNUAL_OT_BY_EMP_YEAR
                            return 0;
                        }
                        int hrs = b.chosenHours();
                        return Math.max(0, hrs - BASE_HOURS_PER_DAY);
                    })
                );

            return dynEmpYearOt
                .filter((key, dynOtHours) -> {
                    int empId      = (int) key.get(0);
                    String companyId = (String) key.get(1);
                    int year       = (int) key.get(2);

                    if (companyId == null || companyId.isBlank()) return false;
                    int limit = CAL.annualOtLimit(companyId);
                    if (limit == Integer.MAX_VALUE) return false;

                    int fixedOt = FIXED_ANNUAL_OT_BY_EMP_YEAR
                            .getOrDefault(empId, Collections.emptyMap())
                            .getOrDefault(year, 0);

                    int totalOt = dynOtHours + fixedOt;
                    return totalOt > limit;
                })
                .penalize(
                    HardMediumSoftScore.ONE_HARD,
                    (key, dynOtHours) -> {
                        int empId      = (int) key.get(0);
                        String companyId = (String) key.get(1);
                        int year       = (int) key.get(2);

                        int limit = CAL.annualOtLimit(companyId);
                        int fixedOt = FIXED_ANNUAL_OT_BY_EMP_YEAR
                                .getOrDefault(empId, Collections.emptyMap())
                                .getOrDefault(year, 0);

                        int totalOt = dynOtHours + fixedOt;
                        return Math.max(1, totalOt - limit);
                    }
                )
                .asConstraint("emp-annual-overtime-limit");
        }



        Constraint monthlyOvertimeLimit(ConstraintFactory f) {
            var dynEmpMonthOt = f.forEach(DaySlot.class)
                .join(f.forEach(CrewSeat.class),
                    Joiners.filtering((DaySlot d, CrewSeat s) ->
                        !isUnassigned(s.employee)))
                .join(f.forEach(BlockDecision.class),
                    Joiners.equal((DaySlot d, CrewSeat s) -> s.blockId, (BlockDecision b) -> b.id))
                .filter(SinglePassConstraints::seatCoversDayAndWorking)
                .groupBy(
                    (d, s, b) -> Arrays.asList(
                        s.employee.id,
                        company(s.employee),
                        d.date.getYear(),
                        d.date.getMonthValue()
                    ),
                    ConstraintCollectors.sum((d, s, b) -> {
                        if (s.pinned) {
                            // hours for fixed part are loaded from FIXED_MONTHLY_OT_BY_EMP_YM
                            return 0;
                        }
                        int hrs = b.chosenHours();
                        return Math.max(0, hrs - BASE_HOURS_PER_DAY);
                    })
                );

            return dynEmpMonthOt
                .filter((key, dynOtHours) -> {
                    int empId      = (int) key.get(0);
                    String companyId = (String) key.get(1);
                    int year       = (int) key.get(2);
                    int month      = (int) key.get(3);

                    if (companyId == null || companyId.isBlank()) return false;
                    int limit = CAL.monthlyOtLimit(companyId);
                    if (limit == Integer.MAX_VALUE) return false;

                    int ym = year * 100 + month;
                    int fixedOt = FIXED_MONTHLY_OT_BY_EMP_YM
                            .getOrDefault(empId, Collections.emptyMap())
                            .getOrDefault(ym, 0);

                    int totalOt = dynOtHours + fixedOt;
                    return totalOt > limit;
                })
                .penalize(
                    HardMediumSoftScore.ONE_HARD,
                    (key, dynOtHours) -> {
                        int empId      = (int) key.get(0);
                        String companyId = (String) key.get(1);
                        int year       = (int) key.get(2);
                        int month      = (int) key.get(3);

                        int limit = CAL.monthlyOtLimit(companyId);
                        int ym = year * 100 + month;
                        int fixedOt = FIXED_MONTHLY_OT_BY_EMP_YM
                                .getOrDefault(empId, Collections.emptyMap())
                                .getOrDefault(ym, 0);

                        int totalOt = dynOtHours + fixedOt;
                        return Math.max(1, totalOt - limit);
                    }
                )
                .asConstraint("emp-monthly-overtime-limit");
        }



        // ---------- Softs ----------
        Constraint preferHoursNear8(ConstraintFactory f) {
            return f.forEach(BlockDecision.class)
                .penalize(HardMediumSoftScore.ONE_SOFT,
                    b -> PREF_HOURS_WEIGHT * Math.abs(b.chosenHours() - 8))
                .asConstraint("soft-hours-near-8");
        }

        Constraint preferSmallerHours(ConstraintFactory f) {
            return f.forEach(BlockDecision.class)
                .penalize(HardMediumSoftScore.ONE_SOFT, b -> SMALLER_HOURS_W * b.chosenHours())
                .asConstraint("soft-smaller-hours");
        }

        // Constraint preferEarlierStart(ConstraintFactory f) {
        //     return f.forEach(BlockDecision.class)
        //         .penalize(HardMediumSoftScore.ONE_SOFT, b -> b.startDay == null ? 0 : EARLIER_START_W * b.startDay)
        //         .asConstraint("soft-earlier-start");
        // }

        Constraint preferEarlierStart(ConstraintFactory f) {
            return f.forEach(BlockDecision.class)
                    // only consider blocks that actually have a startDay
                    .filter(b -> b.startDay != null)
                    .penalize(HardMediumSoftScore.ONE_SOFT, b -> {
                        // delay relative to earliest allowed day in the window
                        int delay = b.startDay - b.windowStart;
                        if (delay <= 0) {
                            return 0; // already as early as window allows
                        }
                        return EARLIER_START_W * delay;
                    })
                    .asConstraint("soft-earlier-start");
        }



        Constraint softSameCompanyPairs(ConstraintFactory f) {
            return f.forEach(CrewSeat.class)
                .filter(s -> !isUnassigned(s.employee))
                .filter(s -> !company(s.employee).isEmpty())
                .groupBy(
                    // key: [blockId, company]
                    s -> Arrays.asList(s.blockId, company(s.employee)),
                    ConstraintCollectors.count()
                )
                .filter((key, count) -> count > 1)
                .reward(HardMediumSoftScore.ONE_SOFT,
                    (key, count) -> COMPANY_PAIR_W * (count * (count - 1) / 2)
                )
                .asConstraint("soft-same-company-pairs");
        }

        Constraint softEncourageSkillVariety(ConstraintFactory f) {
            return f.forEach(CrewSeat.class)
                .filter(s -> !isUnassigned(s.employee))
                .groupBy(
                    // key: [blockId, opId, skillLevel]
                    s -> Arrays.asList(s.blockId, s.opId, skill(s.employee, s.opId)),
                    ConstraintCollectors.count()
                )
                .filter((key, count) -> count > 1)
                .penalize(HardMediumSoftScore.ONE_SOFT,
                    (key, count) -> SKILL_DIVERSITY_W * (count * (count - 1) / 2)
                )
                .asConstraint("soft-encourage-skill-variety");
        }

        Constraint softBalanceBlockAvgSkill(ConstraintFactory f) {
            return f.forEach(CrewSeat.class)
                .filter(s -> !isUnassigned(s.employee))
                .groupBy(s -> Arrays.asList(s.blockId, s.opId),
                         ConstraintCollectors.sum(s -> skill(s.employee, s.opId)),
                         ConstraintCollectors.count())
                .filter((key, sumLv, cnt) -> cnt > 0)
                .penalize(HardMediumSoftScore.ONE_SOFT,
                    (key, sumLv, cnt) -> (int)(SKILL_AVG_W *
                        Math.abs((sumLv / Math.max(1.0, cnt)) - avgSkill((String) key.get(1))) * 100))
                .asConstraint("soft-balance-block-avg-skill");
        }

        Constraint softBalanceTotalHours(ConstraintFactory f) {
            return f.forEach(DaySlot.class)
                .join(f.forEach(CrewSeat.class),
                    Joiners.filtering((DaySlot d, CrewSeat s) ->
                        !isUnassigned(s.employee)))
                .join(f.forEach(BlockDecision.class),
                    Joiners.equal((DaySlot d, CrewSeat s) -> s.blockId, (BlockDecision b) -> b.id))
                .filter(SinglePassConstraints::seatCoversDayAndWorking)
                .groupBy((d, s, b) -> s.employee.id,
                         ConstraintCollectors.sum((d, s, b) ->
                             // fixed hours are loaded from FIXED_TOTAL_HOURS_BY_EMP
                             s.pinned ? 0 : b.chosenHours()))
                .penalize(HardMediumSoftScore.ONE_SOFT, (empId, dynTot) -> {
                    int fixedTot = FIXED_TOTAL_HOURS_BY_EMP.getOrDefault(empId, 0);
                    int total = dynTot + fixedTot;
                    return (int) Math.abs(total - TARGET_HOURS_PER_EMP);
                })
                .asConstraint("soft-balance-total-hours");
        }


        Constraint softContinuousRegionStay(ConstraintFactory f) {
            var items = f.forEach(DaySlot.class)
                .join(f.forEach(CrewSeat.class),
                    Joiners.filtering((DaySlot d, CrewSeat s) -> !isUnassigned(s.employee)))
                .join(f.forEach(BlockDecision.class),
                    Joiners.equal((DaySlot d, CrewSeat s) -> s.blockId, (BlockDecision b) -> b.id))
                .filter(SinglePassConstraints::seatCoversDayAndWorking)
                .groupBy(
                    (d, s, b) -> Arrays.asList(s.employee.id, CAL.regionOfFab(s.factory)),
                    ConstraintCollectors.toList((d, s, b) -> d.id)
                )
                .filter((key, dayList) -> key.get(1) != null);

            return items
                .reward(
                    HardMediumSoftScore.ONE_SOFT,
                    (key, dayList) -> {
                        String regionId = (String) key.get(1);
                        int offInt = CAL.stayOffInterval(regionId);

                        // total days staying in this region (split by stay_off_interval)
                        int totalSpan = totalSegmentSpanWithBreak(dayList, offInt);

                        // Optionally cap by legal limits (maxOn / maxAnnual) if you want:
                        int maxOn     = CAL.maxStayOn(regionId);
                        int maxAnnual = CAL.annualMaxStay(regionId);
                        int allowedCap = totalSpan;
                        if (maxOn != Integer.MAX_VALUE) {
                            allowedCap = Math.min(allowedCap, maxOn);
                        }
                        if (maxAnnual != Integer.MAX_VALUE) {
                            allowedCap = Math.min(allowedCap, maxAnnual);
                        }

                        // reward each legal day of stay
                        return CONTINUOUS_REGION_STAY_W * Math.max(0, allowedCap);
                    }
                )
                .asConstraint("soft-continuous-region-stay");
        }

        Constraint softFabPreference(ConstraintFactory f) {
            return f.forEach(CrewSeat.class)
                .join(f.forEach(BlockDecision.class),
                    Joiners.equal((CrewSeat s) -> s.blockId, (BlockDecision b) -> b.id))
                .filter((s, b) -> !isUnassigned(s.employee))
                .reward(
                    HardMediumSoftScore.ONE_SOFT,
                    (s, b) -> {
                        String fabId      = s.factory;
                        String regionId   = CAL.regionOfFab(fabId);
                        String customerId = CAL.customerOfFab(fabId);

                        int rLevel = regionPref(s.employee, regionId);
                        int cLevel = customerPref(s.employee, customerId);

                        // level 0/1 → 0 reward; >=2 → n(n-1)
                        int rScore = (rLevel <= 1) ? 0 : (rLevel * (rLevel - 1));
                        int cScore = (cLevel <= 1) ? 0 : (cLevel * (cLevel - 1));

                        return FAB_PREFERENCE_W * (rScore + cScore);
                    }
                )
                .asConstraint("soft-fab-preference");
        }


    }

    // ---------------- Parsing ----------------

    static class OpDef {
        String phaseId; int phaseNum;
        List<Integer> allowed; int min; int max;
    }
    static class ParsedEnv {
        Map<String,OpDef> opdef; List<EmployeeFact> employees;
        Map<String, EmployeeFact> byWid;
    }

    static class InitialAssign {
    String module;
    String opId;
    Map<Integer,Integer> hoursByDay = new HashMap<>();
    }

    static class ParsedSchedule {
        LocalDate planStart;
        LocalDate planEnd;

        List<DaySlot> daySlots;
        List<TaskWindow> windows;
        Map<String,Integer> requiredByKey;

        // Fixed assignments parsed from Schedule.yaml
        List<FixedAssign> fixedRows;
        Map<String,Integer> fixedHoursByKey;

        // modules that are still relevant after trim (for dynamic blocks / warm start)
        Set<String> activeModules;

        // flexible warm-start rows
        List<InitialAssign> initialRows;

        // for logging / debugging
        int cutOffDayId;
        LocalDate cutOffDate;
    }
    static class FixedAssign {
        String module; String opId; String factory; String wid;
        int startDayId; int endDayId;
        Map<Integer,Integer> hoursByDay = new HashMap<>();
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

            // NEW: parse fab_suitability_map (region & customer_company)
            List<Map<String,Object>> fsmList =
                    (List<Map<String,Object>>) w.getOrDefault("fab_suitability_map", List.of());
            for (Map<String,Object> fs : fsmList) {
                if (fs == null) continue;
                String kind = safeStr(fs.get("kind"));
                Map<String,Object> suitRaw =
                        (Map<String,Object>) fs.getOrDefault("suitability", Map.of());
                for (Map.Entry<String,Object> se : suitRaw.entrySet()) {
                    String key = safeStr(se.getKey());
                    int lvl = parseInt(se.getValue(), 1); // default = 1 (neutral) if missing
                    if ("region".equals(kind)) {
                        ef.regionPreference.put(key, lvl);
                    } else if ("customer_company".equals(kind)) {
                        ef.customerPreference.put(key, lvl);
                    }
                }
            }
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

        List<TaskWindow> windows = new ArrayList<>();
        Map<String,Integer> required = new HashMap<>();

        // Track last end dayId per module (for "finished module" trim)
        Map<String,Integer> moduleLastEnd = new HashMap<>();

        Object wfObj = s.get("workflow_task_list");
        List<Map<String,Object>> wfTasks = (wfObj instanceof List) ? (List<Map<String,Object>>) wfObj : List.of();
        for (Map<String,Object> wf : wfTasks) {
            String module = safeStr(wf.get("id"));
            String fab = safeStr(wf.get("fab"));
            Object phasesObj = wf.get("phase_task_list");
            List<Map<String,Object>> phases = (phasesObj instanceof List) ? (List<Map<String,Object>>) phasesObj : List.of();
            for (Map<String,Object> ph : phases) {
                String phId = safeStr(ph.get("phase"));
                int phNum = phaseNumFromId(phId);
                LocalDate pStart = LocalDate.parse(safeStr(ph.get("start_date")).replace("-", "/"), DF);
                LocalDate pEnd   = LocalDate.parse(safeStr(ph.get("end_date")).replace("-", "/"), DF);
                int startId = (int) (pStart.toEpochDay() - start.toEpochDay());
                int endId   = (int) (pEnd.toEpochDay()   - start.toEpochDay());

                // remember the latest end per module (even if before horizon)
                moduleLastEnd.merge(module, endId, Math::max);

                Object opsObj = ph.get("operation_task_list");
                List<Map<String,Object>> opTasks = (opsObj instanceof List) ? (List<Map<String,Object>>) opsObj : List.of();
                for (Map<String,Object> ot : opTasks) {
                    String opId = safeStr(ot.get("operation"));
                    int workloadDays = parseInt(ot.get("workload_days"), 0);

                    OpDef od = opdef.get(opId);
                    if (od == null) {
                        throw new IllegalArgumentException("operation " + opId + " missing in EnvConfig");
                    }

                    int baseline = (od.allowed.size() == 1 && od.allowed.get(0) == 4) ? 4 : 8;
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

        // ---- Determine cut-off date ----
        // If "cut_off_date" exists in schedule, use it; otherwise use plan_range.start_date
        LocalDate cutOffDate;
        Object cutOffObj = s.get("cut_off_date");
        if (cutOffObj != null) {
            cutOffDate = LocalDate.parse(safeStr(cutOffObj).replace("-", "/"), DF);
        } else {
            cutOffDate = start; // default: anything that ends before plan_start is "finished"
        }
        int cutOffDayId = (int) (cutOffDate.toEpochDay() - start.toEpochDay());

        // ---- Decide activeModules using grace-gap x days ----
        Set<String> activeModules = new HashSet<>();
        if (TRIM_FINISHED_MODULES) {
            int grace = MODULE_TRIM_GRACE_DAYS; // x days of tolerance

            for (Map.Entry<String,Integer> e : moduleLastEnd.entrySet()) {
                String module = e.getKey();
                int lastEndDayId = e.getValue();

                // gap = how many days between module end and cutOff
                int gap = cutOffDayId - lastEndDayId;

                // if module ended long before cutOff, treat as finished (trim)
                // if it ended close to or after cutOff, keep as active
                if (gap <= grace) {
                    activeModules.add(module);
                }
            }

            // Edge case: if none pass the rule, keep everything
            if (activeModules.isEmpty()) {
                activeModules.addAll(moduleLastEnd.keySet());
            }

            // remove windows for modules that are already fully finished
            windows.removeIf(w -> !activeModules.contains(w.module));

            // remove requiredByKey entries for finished modules
            required.entrySet().removeIf(e -> {
                String key = e.getKey();
                String mod = key.contains("|") ? key.substring(0, key.indexOf('|')) : key;
                return !activeModules.contains(mod);
            });
        } else {
            // No trimming → treat all modules as active
            activeModules.addAll(moduleLastEnd.keySet());
        }

        // ---- Read fixed + flexible assignments ----
        List<FixedAssign> fixedRows = new ArrayList<>();
        Map<String,Integer> fixedHoursByKey = new HashMap<>();
        List<InitialAssign> initialRows = new ArrayList<>();

        Object asgObj = s.get("assignment_list");
        List<Map<String,Object>> asgs = (asgObj instanceof List) ? (List<Map<String,Object>>) asgObj : List.of();

        Map<String,Integer> latestFixedEndInRange = new HashMap<>();
        Map<String,Integer> latestFixedEndAny     = new HashMap<>();

        for (Map<String,Object> a : asgs) {
            String flex = safeStr(a.get("plan_flexibility"));  // e.g. "Fixed" or "Flexible"
            String opTask = safeStr(a.get("operation_task"));  // e.g. e16p4o1
            int idx = opTask.indexOf("p");
            String module = (idx > 0) ? opTask.substring(0, idx) : opTask;
            String opId   = (idx > 0) ? opTask.substring(idx) : "";

            boolean isFixed    = "fixed".equalsIgnoreCase(flex);
            boolean isFlexible = "flexible".equalsIgnoreCase(flex);

            String phId = "";
            int phNum = 0;
            try {
                String pPart = opId.split("o", 2)[0];
                phId = pPart; phNum = phaseNumFromId(pPart);
            } catch (Exception ignore) {}

            String wid = safeStr(a.get("worker"));

            LocalDate sd = a.get("start_date") == null ? null :
                    LocalDate.parse(safeStr(a.get("start_date")).replace("-", "/"), DF);
            LocalDate ed = a.get("end_date")   == null ? null :
                    LocalDate.parse(safeStr(a.get("end_date")).replace("-", "/"), DF);
            int sId = (sd == null) ? -1 : (int)(sd.toEpochDay() - start.toEpochDay());
            int eId = (ed == null) ? -1 : (int)(ed.toEpochDay() - start.toEpochDay());

            if (isFixed && eId >= Integer.MIN_VALUE) {
                // last fixed end for (module, phase) anywhere
                latestFixedEndAny.merge(module + "|" + phNum, eId, Math::max);
            }

            String wdKey = a.containsKey("work_date_lsit") ? "work_date_lsit" : "work_date_list";
            List<Map<String,Object>> wdl = (a.get(wdKey) instanceof List) ? (List<Map<String,Object>>) a.get(wdKey) : List.of();

            Map<Integer,Integer> byDay = new HashMap<>();
            int totalFixedHours = 0;
            for (Map<String,Object> item : wdl) {
                LocalDate d = LocalDate.parse(safeStr(item.get("date")).replace("-", "/"), DF);
                int did = (int)(d.toEpochDay() - start.toEpochDay());
                int h = parseInt(item.get("hour"), 0);

                // always store by-day hours (for both fixed and flexible)
                byDay.merge(did, h, Integer::sum);

                if (isFixed) {
                    totalFixedHours += h;

                    // last fixed end inside current plan_range for this phase
                    latestFixedEndInRange.merge(module + "|" + phNum, did, Math::max);
                }
            }

            // fixed → hard background (used for OT, daily cap, one-factory-per-day, etc.)
            if (isFixed && totalFixedHours > 0) {
                fixedHoursByKey.merge(module + "|" + opId, totalFixedHours, Integer::sum);
            }
            if (isFixed && !byDay.isEmpty()) {
                FixedAssign fa = new FixedAssign();
                fa.module = module; fa.opId = opId; fa.wid = wid;
                fa.startDayId = sId; fa.endDayId = eId;
                fa.hoursByDay = byDay; fa.phaseId = phId; fa.phaseNum = phNum;
                fa.factory = null; // will be inferred later
                fixedRows.add(fa);
            }

            // flexible → only used as initial (warm start), not fixed
            if (isFlexible && !byDay.isEmpty()) {
                InitialAssign ia = new InitialAssign();
                ia.module = module;
                ia.opId   = opId;
                ia.hoursByDay.putAll(byDay);
                initialRows.add(ia);
            }
        }


        ParsedSchedule out = new ParsedSchedule();
        out.planStart = start; out.planEnd = end;
        out.windows = windows; out.requiredByKey = required;
        out.fixedRows = fixedRows; out.fixedHoursByKey = fixedHoursByKey;
        out.activeModules = activeModules;
        out.initialRows = initialRows;

        // ---- Push phase windows based on fixed ends (even if outside horizon) ----
        for (TaskWindow w : windows) {
            int prev = w.phaseNum - 1;
            if (prev <= 0) continue;
            Integer inRangeEnd = latestFixedEndInRange.get(w.module + "|" + prev);
            Integer anyEnd     = latestFixedEndAny.get(w.module + "|" + prev);
            Integer endPrev = null;
            if (inRangeEnd != null && anyEnd != null) endPrev = Math.max(inRangeEnd, anyEnd);
            else if (inRangeEnd != null) endPrev = inRangeEnd;
            else if (anyEnd != null) endPrev = anyEnd;
            if (endPrev != null) {
                w.startDayId = Math.max(w.startDayId, endPrev + 1);
            }
        }

        // After pushing, some windows may have start > end (no free days left in horizon).
        // For those, we effectively treat them as "no flexible workload left".
        for (TaskWindow w : windows) {
            if (w.startDayId > w.endDayId) {
                System.out.printf(
                    "[WARN] Collapsed window for module=%s phase=%s op=%s (startDayId=%d > endDayId=%d). " +
                    "Marking workloadDays=0 so this block is not scheduled.%n",
                    w.module, w.phaseId, w.opId, w.startDayId, w.endDayId
                );
                w.workloadDays = 0;
            }
        }

        // ---- Decide the earliest dayId we actually keep as DaySlot ----
        int firstDayId = 0;
        if (TRIM_FINISHED_MODULES) {
            // Never earlier than the cut-off
            firstDayId = Math.max(0, cutOffDayId);

            // Also respect the first active flexible window (so we don't create days
            // that are never used by any BlockDecision)
            int minDynamicStart = Integer.MAX_VALUE;
            for (TaskWindow w : windows) {
                if (w.workloadDays <= 0) continue;
                if (!activeModules.contains(w.module)) continue;
                if (w.startDayId < minDynamicStart) {
                    minDynamicStart = w.startDayId;
                }
            }
            if (minDynamicStart != Integer.MAX_VALUE) {
                firstDayId = Math.max(firstDayId, minDynamicStart);
            }
        }

        // ---- Build DaySlots only from firstDayId .. horizon-1 ----
        // IMPORTANT: DaySlot.id stays as the *global* dayId (0-based from original plan_start)
        // so exporter + fixed-hours maps still work without any change.
        List<DaySlot> days = new ArrayList<>();
        for (int dayId = Math.max(0, firstDayId); dayId < horizon; dayId++) {
            days.add(new DaySlot(dayId, start.plusDays(dayId)));
        }
        out.daySlots = days;
        out.activeModules = activeModules;
        out.cutOffDayId = cutOffDayId;
        out.cutOffDate = cutOffDate;

        return out;
    }



    // ---------------- Build entities for single pass ----------------

    static class BuildOut {
        List<BlockDecision> blocks;
        List<CrewSeat> seats;
    }

    static BuildOut buildEntitiesSinglePass(ParsedSchedule sch, ParsedEnv env) {
        Map<String, TaskWindow> keyToWin = new HashMap<>();
        Map<String, List<TaskWindow>> moduleWins = new HashMap<>();
        for (TaskWindow w : sch.windows) {
            keyToWin.putIfAbsent(w.module + "|" + w.opId, w);
            moduleWins.computeIfAbsent(w.module, k -> new ArrayList<>()).add(w);
        }

        for (FixedAssign fa : sch.fixedRows) {
            List<TaskWindow> ws = moduleWins.getOrDefault(fa.module, List.of());
            if (!ws.isEmpty()) fa.factory = ws.get(0).factory;
        }

        List<BlockDecision> blocks = new ArrayList<>();
        List<CrewSeat> seats = new ArrayList<>();

        // ---- NEW: dummy block for all pinned seats ----
        BlockDecision fixedBlock = new BlockDecision();
        fixedBlock.id = 0;
        fixedBlock.module = "__FIXED__";
        fixedBlock.factory = null;
        fixedBlock.phaseId = "";
        fixedBlock.phaseNum = 0;
        fixedBlock.opId = "__FIXED__";
        fixedBlock.windowStart = 0;
        fixedBlock.windowEnd = Math.max(0, sch.daySlots.size() - 1);
        fixedBlock.requiredHours = 0;
        fixedBlock.allowed = List.of(8);
        fixedBlock.minHeads = 0;
        fixedBlock.maxHeads = 0;
        blocks.add(fixedBlock);

        int bid = 1;
        int sid = 1;

        // ---- existing dynamic block creation below stays the same ----
        for (TaskWindow w : sch.windows) {
            int baseline = (w.allowed.size()==1 && w.allowed.get(0)==4) ? 4 : 8;
            int totalReq = w.workloadDays * baseline;
            int fixed = sch.fixedHoursByKey.getOrDefault(w.module + "|" + w.opId, 0);
            int req = Math.max(0, totalReq - fixed);

            if (req == 0) continue;

            BlockDecision b = new BlockDecision();
            b.id = bid++;
            b.module = w.module; b.factory = w.factory;
            b.phaseId = w.phaseId; b.phaseNum = w.phaseNum;
            b.opId = w.opId;
            b.windowStart = w.startDayId; b.windowEnd = w.endDayId;
            b.requiredHours = req;
            b.allowed = new ArrayList<>(w.allowed);
            b.minHeads = w.minHeads; b.maxHeads = w.maxHeads;
            blocks.add(b);

            for (int sidx = 0; sidx < Math.max(1, w.maxHeads); sidx++) {
                CrewSeat cs = new CrewSeat();
                cs.id = sid++;
                cs.blockId = b.id;
                cs.module = w.module; cs.factory = w.factory;
                cs.phaseId = w.phaseId; cs.phaseNum = w.phaseNum;
                cs.opId = w.opId;
                cs.seatIndex = sidx;
                cs.needManager = (sidx == 0);
                cs.employee = new EmployeeFact(0, "__UNASSIGNED__", "__UNASSIGNED__", Map.of(), false, "");
                seats.add(cs);
            }
        }

        Map<String,String> moduleToFactory = new HashMap<>();
        Map<String,String> moduleOpToPhase = new HashMap<>();
        Map<String,Integer> moduleOpToPhaseNum = new HashMap<>();
        for (TaskWindow w : sch.windows) {
            moduleToFactory.put(w.module, w.factory);
            moduleOpToPhase.put(w.module + "|" + w.opId, w.phaseId);
            moduleOpToPhaseNum.put(w.module + "|" + w.opId, w.phaseNum);
        }

        // ---- pinned seats (fixed assignments) ----
        for (FixedAssign fa : sch.fixedRows) {
            String factory = moduleToFactory.getOrDefault(fa.module, fa.factory);
            CrewSeat cs = new CrewSeat();
            cs.id = sid++;

            // IMPORTANT: attach to dummy fixed block so constraints can see them
            cs.blockId = fixedBlock.id; // 0

            cs.module = fa.module; cs.factory = factory;
            cs.phaseId = moduleOpToPhase.getOrDefault(fa.module + "|" + fa.opId, fa.phaseId);
            cs.phaseNum = moduleOpToPhaseNum.getOrDefault(fa.module + "|" + fa.opId, fa.phaseNum);
            cs.opId = fa.opId;
            cs.seatIndex = 0;
            cs.needManager = false;

            cs.pinned = true;
            cs.pinnedWid = fa.wid;
            int minDid = fa.hoursByDay.keySet().stream().min(Integer::compareTo).orElse(fa.startDayId);
            int maxDid = fa.hoursByDay.keySet().stream().max(Integer::compareTo).orElse(fa.startDayId);
            cs.pinnedStart = minDid;
            cs.pinnedDays  = Math.max(1, maxDid - minDid + 1);
            cs.pinnedHours = fa.hoursByDay.values().stream().max(Integer::compareTo).orElse(8);

            EmployeeFact ef = env.byWid.get(fa.wid);
            if (ef == null) ef = new EmployeeFact(0, "__UNASSIGNED__", "__UNASSIGNED__", Map.of(), false, "");
            cs.employee = ef;

            seats.add(cs);
        }

        BuildOut out = new BuildOut();
        out.blocks = blocks; out.seats = seats;
        return out;
    }

    // ---------------- Entity-dependent employee ranges ----------------

    private static void fillSeatCandidatesSinglePass(
            List<CrewSeat> seats,
            List<BlockDecision> blocks,
            List<EmployeeFact> employees) {

        Map<Integer, BlockDecision> byBlock = new HashMap<>();
        for (BlockDecision b : blocks) byBlock.put(b.id, b);

        Map<String, Set<Integer>> personalOff = CAL.workerOffByWid;

        for (CrewSeat s : seats) {
            // Pinned seats → restrict to the pinned worker in value range
            if (s.pinned) {
                EmployeeFact pinnedEmp = null;
                for (EmployeeFact e : employees) {
                    if (e != null && s.pinnedWid != null && s.pinnedWid.equals(e.wid)) { pinnedEmp = e; break; }
                }
                s.setCandidateEmployees(pinnedEmp == null ? List.of() : List.of(pinnedEmp));
                continue;
            }

            BlockDecision b = byBlock.get(s.blockId);
            // Use current entity vars if set, else conservative estimate = entire window
            int estStart = (b != null && b.startDay != null) ? b.startDay : b.windowStart;
            int estDays  = (b != null && b.days     != null) ? b.days     : Math.max(1, b.windowEnd - b.windowStart + 1);

            List<EmployeeFact> cand = new ArrayList<>();
            for (EmployeeFact e : employees) {
                if (e == null || e.id == 0) continue;

                // Skill gate
                if (e.skills.getOrDefault(s.opId, 0) < 1) continue;

                // NEW: preference gate (level 0 for region OR customer_company → disallow)
                String fabId     = s.factory;
                String regionId  = CAL.regionOfFab(fabId);
                String customerId = CAL.customerOfFab(fabId);
                int rp = regionPref(e, regionId);
                int cp = customerPref(e, customerId);
                if (rp == 0 || cp == 0) continue; // hate → never candidate

                // Availability gate (skip person off days, fab closure handled separately)
                Set<Integer> off = personalOff.getOrDefault(e.wid, Set.of());
                boolean clash = false;
                for (int i = 0; i < estDays; i++) {
                    int did = estStart + i;
                    if (!isWorkingDay(did, s.factory)) continue; // fab closed; not a person off violation
                    if (off.contains(did)) { clash = true; break; }
                }
                if (clash) continue;

                cand.add(e);
            }

            // Manager gate (strict, no UNASSIGNED fallback)
            if (s.needManager) {
                cand = cand.stream().filter(emp -> emp != null && emp.isManager).toList();
                if (cand.isEmpty()) {
                    throw new IllegalStateException(
                        "No manager candidates for block " + s.blockId +
                        " seatIndex=" + s.seatIndex + " (" + s.module + " " + s.opId + ")."
                    );
                }
            }
            s.setCandidateEmployees(cand);

        }
    }


    // ---------------- Solver builder ----------------

    static <S> SolverFactory<S> buildSolverFactory(Class<S> solutionClass,
                                                   Class<?>[] entityClasses,
                                                   Class<? extends ConstraintProvider> providerClass,
                                                   String bestScoreLimit,
                                                   Integer spentMinutes,
                                                   Integer unimprovedSeconds) {
        SolverConfig cfg = new SolverConfig();
        cfg.withSolutionClass(solutionClass);
        cfg.withEntityClasses(entityClasses);
        cfg.withScoreDirectorFactory(
                new ScoreDirectorFactoryConfig().withConstraintProviderClass(providerClass)
        );

        // Optional: if EXTERNAL_RANDOM_SEED != 0, fix the random seed for reproducibility
        if (EXTERNAL_RANDOM_SEED != 0L) {
            cfg.withRandomSeed(EXTERNAL_RANDOM_SEED);
        }

        TerminationConfig term = new TerminationConfig();
        if (bestScoreLimit != null) term.setBestScoreLimit(bestScoreLimit);
        if (spentMinutes != null && spentMinutes > 0) {
            term.setSpentLimit(java.time.Duration.ofMinutes(spentMinutes));
        }
        if (unimprovedSeconds != null && unimprovedSeconds > 0) {
            term.setUnimprovedSpentLimit(java.time.Duration.ofSeconds(unimprovedSeconds));
        }
        cfg.withTerminationConfig(term);

        return SolverFactory.create(cfg);
    }

    static boolean hardZero(Score<?> s) { return s != null && s.toString().startsWith("0hard"); }

    // ---------------- Logging helper ----------------

        static void appendSolveLog(
            SolverFactory<SinglePassPlan> factoryStage2,
            SinglePassPlan best2,
            ParsedSchedule sch,
            java.time.Duration stage1Duration,
            java.time.Duration stage2Duration) {

        Path logFile = Paths.get("solver_log.txt"); // will be created in working directory

        try (BufferedWriter bw = Files.newBufferedWriter(
                logFile,
                StandardOpenOption.CREATE,
                StandardOpenOption.APPEND)) {

            // SolutionManager + explanation
            SolutionManager<SinglePassPlan, HardMediumSoftScore> solutionManager =
                    SolutionManager.create(factoryStage2);
            ScoreExplanation<SinglePassPlan, HardMediumSoftScore> explanation =
                    solutionManager.explain(best2);

            // Separator
            bw.write("============================================================");
            bw.newLine();
            bw.write("Run at: " + java.time.LocalDateTime.now());
            bw.newLine();

            // Basic timing and score
            bw.write("Stage1 duration: " + fmt(stage1Duration));
            bw.newLine();
            bw.write("Stage2 duration: " + fmt(stage2Duration));
            bw.newLine();
            bw.write("Final score: " + String.valueOf(best2.getScore()));
            bw.newLine();

            // Plan and cut-off
            bw.write("Plan range (plan_range): " + sch.planStart + " .. " + sch.planEnd);
            bw.newLine();
            if (sch.cutOffDate != null) {
                bw.write("Cut-off date (cut_off_date): " + sch.cutOffDate +
                        "  (cutOffDayId=" + sch.cutOffDayId + ")");
            } else {
                bw.write("Cut-off date (cut_off_date): [none]");
            }
            bw.newLine();

            // DaySlot range actually used in solver
            int minDayId = Integer.MAX_VALUE;
            int maxDayId = Integer.MIN_VALUE;
            LocalDate minDate = null;
            LocalDate maxDate = null;
            int dayCount = 0;

            if (best2.days != null) {
                for (DaySlot d : best2.days) {
                    if (d == null) continue;
                    dayCount++;
                    if (d.id < minDayId) {
                        minDayId = d.id;
                        minDate = d.date;
                    }
                    if (d.id > maxDayId) {
                        maxDayId = d.id;
                        maxDate = d.date;
                    }
                }
            }

            if (dayCount > 0) {
                bw.write("DaySlots in solver: id=" + minDayId + ".." + maxDayId +
                        " (count=" + dayCount + "), dates=" + minDate + " .. " + maxDate);
            } else {
                bw.write("DaySlots in solver: [none]");
            }
            bw.newLine();

            // Module sets
            java.util.Set<String> activeMods =
                    (sch.activeModules == null) ? new java.util.HashSet<>() :
                            new java.util.TreeSet<>(sch.activeModules);

            java.util.Set<String> dynamicBlockMods = new java.util.TreeSet<>();
            if (best2.blocks != null) {
                for (BlockDecision b : best2.blocks) {
                    if (b == null) continue;
                    // id 0 is the dummy fixed block
                    if (b.id == 0) continue;
                    if (b.module != null && !b.module.isBlank()) {
                        dynamicBlockMods.add(b.module);
                    }
                }
            }

            java.util.Set<String> pinnedMods = new java.util.TreeSet<>();
            if (best2.seats != null) {
                for (CrewSeat s : best2.seats) {
                    if (s == null) continue;
                    if (!s.pinned) continue;
                    if (s.module != null && !s.module.isBlank()) {
                        pinnedMods.add(s.module);
                    }
                }
            }

            bw.write("Active modules by trim logic (gap <= " + MODULE_TRIM_GRACE_DAYS + " days): "
                    + (activeMods.isEmpty() ? "[none]" : String.join(", ", activeMods)));
            bw.newLine();

            bw.write("Modules with dynamic blocks (BlockDecision.id > 0): "
                    + (dynamicBlockMods.isEmpty() ? "[none]" : String.join(", ", dynamicBlockMods)));
            bw.newLine();

            bw.write("Modules with pinned seats (fixed assignments visible to constraints): "
                    + (pinnedMods.isEmpty() ? "[none]" : String.join(", ", pinnedMods)));
            bw.newLine();

            bw.write("Per-constraint scores:");
            bw.newLine();

            // Per-constraint scores
            explanation.getConstraintMatchTotalMap().forEach((constraintName, cmt) -> {
                try {
                    bw.write("  " + constraintName + " = " + cmt.getScore());
                    bw.newLine();
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            });

            bw.flush();
        } catch (IOException e) {
            System.err.println("Failed to write solver_log.txt: " + e.getMessage());
        }
    }


    static void applyInitialPlan(ParsedSchedule sch, List<BlockDecision> blocks) {
        if (sch.initialRows == null || sch.initialRows.isEmpty()) return;

        // Earliest day index that actually exists as a DaySlot (after trimming)
        int minActiveDay = sch.daySlots.stream()
                .mapToInt(d -> d.id)
                .min()
                .orElse(0);

        // Group flexible rows by (module|opId)
        Map<String, List<InitialAssign>> byKey = new HashMap<>();
        for (InitialAssign ia : sch.initialRows) {
            if (ia == null) continue;
            if (sch.activeModules != null && !sch.activeModules.isEmpty()
                    && !sch.activeModules.contains(ia.module)) {
                continue;
            }
            byKey.computeIfAbsent(ia.module + "|" + ia.opId, k -> new ArrayList<>()).add(ia);
        }
        if (byKey.isEmpty()) return;

        for (BlockDecision b : blocks) {
            if (b == null || b.id == 0) continue; // skip dummy fixed block

            List<InitialAssign> list = byKey.get(b.module + "|" + b.opId);
            if (list == null || list.isEmpty()) continue;

            List<Integer> days = new ArrayList<>();
            int totalHours = 0;
            for (InitialAssign ia : list) {
                for (Map.Entry<Integer,Integer> e : ia.hoursByDay.entrySet()) {
                    int dayId = e.getKey();
                    if (dayId < minActiveDay) continue; // ignore very old days
                    days.add(dayId);
                    totalHours += e.getValue();
                }
            }
            if (days.isEmpty()) continue;

            Collections.sort(days);
            int startDay = days.get(0);
            int endDay   = days.get(days.size() - 1);

            // Clamp the initial plan inside the legal window
            startDay = Math.max(startDay, b.windowStart);
            endDay   = Math.min(endDay,   b.windowEnd);
            if (startDay > endDay) continue;

            b.startDay = startDay;
            b.days     = Math.max(1, endDay - startDay + 1);

            // Choose hours close to average per-day usage, but inside allowed[]
            if (b.allowed != null && !b.allowed.isEmpty()) {
                int avgPerDay = totalHours / Math.max(1, days.size());
                int best = b.allowed.get(0);
                int bestDelta = Math.abs(best - avgPerDay);
                for (int h : b.allowed) {
                    int d = Math.abs(h - avgPerDay);
                    if (d < bestDelta) {
                        bestDelta = d;
                        best = h;
                    }
                }
                b.hours = best;
            }
        }
    }

    /**
     * v843_sp helper:
     * - set random seed,
     * - run solveFromYaml,
     * - overwrite Schedule.yaml via ExportSchedule.
     */
    public static void solveWithSeed(String envPath, String schedPath, long seed) throws Exception {
        EXTERNAL_RANDOM_SEED = seed;

        RunResult rr = solveFromYaml(envPath, schedPath);

        ExportSchedule.overwriteScheduleWithAssignments(
                rr.plan, rr.planStart, schedPath, envPath);

        System.out.println("Done (seed=" + seed + ").");

        // Optional: reset seed if you want
        EXTERNAL_RANDOM_SEED = 0L;
    }
    // ---------------- Public API ----------------

    public static class RunResult { public SinglePassPlan plan; public LocalDate planStart; }

    public static RunResult solveFromYaml(String envPath, String schedPath) throws IOException {
        ParsedEnv env = parseEnv(envPath);
        ParsedSchedule sch = parseSchedule(schedPath, env.opdef);
        System.out.printf("A");
        buildCalendars(envPath, sch.planStart, sch.planEnd);
        System.out.printf("B");
        int realEmp = Math.max(1, env.employees.size() - 1);
        int totalReq = sch.requiredByKey.values().stream().mapToInt(Integer::intValue).sum();
        TARGET_HOURS_PER_EMP = totalReq / (double) realEmp;

        BuildOut built = buildEntitiesSinglePass(sch, env);
        applyInitialPlan(sch, built.blocks);
        // --------------------------------------------------
        // Build background fixed-hours/factories per (emp, day)
        // --------------------------------------------------
        FIXED_HOURS_BY_EMP_DAY.clear();
        FIXED_FACTORIES_BY_EMP_DAY.clear();

        for (FixedAssign fa : sch.fixedRows) {
            EmployeeFact ef = env.byWid.get(fa.wid);
            if (ef == null || ef.id == 0) {
                continue; // unknown worker, skip
            }
            int empId = ef.id;

            // hours per day
            Map<Integer, Integer> hoursMap =
                FIXED_HOURS_BY_EMP_DAY.computeIfAbsent(empId, k -> new HashMap<>());
            for (Map.Entry<Integer, Integer> e : fa.hoursByDay.entrySet()) {
                int dayId = e.getKey();
                int h = e.getValue();
                hoursMap.merge(dayId, h, Integer::sum);
            }

            // factories per day
            if (fa.factory != null && !fa.factory.isBlank()) {
                Map<Integer, Set<String>> facMap =
                    FIXED_FACTORIES_BY_EMP_DAY.computeIfAbsent(empId, k -> new HashMap<>());
                for (Integer dayId : fa.hoursByDay.keySet()) {
                    Set<String> set =
                        facMap.computeIfAbsent(dayId, k -> new HashSet<>());
                    set.add(fa.factory);
                }
            }
        }

        // ---- NEW: aggregate fixed totals and overtime ----
        FIXED_TOTAL_HOURS_BY_EMP.clear();
        FIXED_ANNUAL_OT_BY_EMP_YEAR.clear();
        FIXED_MONTHLY_OT_BY_EMP_YM.clear();

        for (FixedAssign fa : sch.fixedRows) {
            EmployeeFact ef = env.byWid.get(fa.wid);
            if (ef == null || ef.id == 0) {
                continue;
            }
            int empId = ef.id;

            for (Map.Entry<Integer, Integer> e : fa.hoursByDay.entrySet()) {
                int dayId = e.getKey();
                int h     = e.getValue();

                // total hours per employee
                FIXED_TOTAL_HOURS_BY_EMP.merge(empId, h, Integer::sum);

                // date → year, month
                LocalDate d  = sch.planStart.plusDays(dayId);
                int year     = d.getYear();
                int month    = d.getMonthValue();
                int ym       = year * 100 + month;

                int ot = Math.max(0, h - SinglePassConstraints.BASE_HOURS_PER_DAY);

                // annual OT
                FIXED_ANNUAL_OT_BY_EMP_YEAR
                    .computeIfAbsent(empId, k -> new HashMap<>())
                    .merge(year, ot, Integer::sum);

                // monthly OT
                FIXED_MONTHLY_OT_BY_EMP_YM
                    .computeIfAbsent(empId, k -> new HashMap<>())
                    .merge(ym, ot, Integer::sum);
            }
        }

        fillSeatCandidatesSinglePass(built.seats, built.blocks, env.employees);

        SinglePassPlan p = new SinglePassPlan();
        p.days = sch.daySlots;
        p.employees = env.employees;
        p.blocks = built.blocks;
        p.seats  = built.seats;

        System.out.println("Start SINGLE PASS (stage1) at " + nowClock());
        long t0 = System.nanoTime();

        // ---- Stage 1: your current settings (e.g., 90/90) ----
        SolverFactory<SinglePassPlan> factoryStage1 = buildSolverFactory(
                SinglePassPlan.class,
                new Class<?>[]{ BlockDecision.class, CrewSeat.class },
                SinglePassConstraints.class,
                "0hard/*medium/*soft", 240, 300);
        Solver<SinglePassPlan> stage1 = factoryStage1.buildSolver();
        SinglePassPlan best1 = stage1.solve(p);

        long t1 = System.nanoTime();
        System.out.printf("Stage1 done %s | duration=%s | score=%s | blocks=%d | seats=%d%n",
                nowClock(), fmt(java.time.Duration.ofNanos(t1 - t0)),
                String.valueOf(best1.getScore()),
                best1.blocks == null ? 0 : best1.blocks.size(),
                best1.seats == null ? 0 : best1.seats.size());

        // ---- Stage 2 (polish): 60 minutes, start from stage1 result ----
        System.out.println("Start POLISH (stage2, 60m) at " + nowClock());

        SolverFactory<SinglePassPlan> factoryStage2 = buildSolverFactory(
                SinglePassPlan.class,
                new Class<?>[]{ BlockDecision.class, CrewSeat.class },
                SinglePassConstraints.class,
                null /* bestScoreLimit */,
                60  /* spentMinutes */,
                300 /* unimprovedSeconds */);

        Solver<SinglePassPlan> stage2 = factoryStage2.buildSolver();
        SinglePassPlan best2 = stage2.solve(best1);

        long t2 = System.nanoTime();
        System.out.printf("Stage2 done %s | duration=%s | score=%s%n",
                nowClock(), fmt(java.time.Duration.ofNanos(t2 - t1)),
                String.valueOf(best2.getScore()));

        java.time.Duration stage1Dur = java.time.Duration.ofNanos(t1 - t0);
        java.time.Duration stage2Dur = java.time.Duration.ofNanos(t2 - t1);

        System.out.printf("Stage1 done %s | duration=%s | score=%s | blocks=%d | seats=%d%n",
                nowClock(), fmt(stage1Dur),
                String.valueOf(best1.getScore()),
                best1.blocks == null ? 0 : best1.blocks.size(),
                best1.seats == null ? 0 : best1.seats.size());

        System.out.printf("Stage2 done %s | duration=%s | score=%s%n",
                nowClock(), fmt(stage2Dur),
                String.valueOf(best2.getScore()));

        // Append log to file (score + durations + per-constraint score)
        appendSolveLog(factoryStage2, best2, sch, stage1Dur, stage2Dur);

        RunResult rr = new RunResult();
        rr.plan = best2; rr.planStart = sch.planStart;
        return rr;

    }

    // ---------------- Export hook ----------------
    public static void main(String[] args) throws Exception {
        String envPath = args.length > 0 ? args[0] : "EnvConfig.yaml";
        String schedPath = args.length > 1 ? args[1] : "Schedule.yaml";

        RunResult rr = solveFromYaml(envPath, schedPath);

        ExportSchedule.overwriteScheduleWithAssignments(
                rr.plan, rr.planStart, schedPath, envPath);

        System.out.println("Done.");
    }
}