package com.yourorg.scheduler.calendar;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Per-solve state that used to live in static fields of the old single-file scheduler. One current context per JVM
 * ({@link #current()}); it is refilled by the build step, never replaced by it.
 */
public class SolveContext {

    private static SolveContext current = new SolveContext();

    /** Never null. */
    public static SolveContext current() {
        return current;
    }

    public static void set(SolveContext ctx) {
        current = Objects.requireNonNull(ctx);
    }

    /** Installs a fresh empty context (calendars empty, maps empty, numbers 0). Intended for tests. */
    public static void clear() {
        current = new SolveContext();
    }

    /** Fab/region/customer/worker calendars and stay/overtime limits; empty until EnvConfigParser.buildCalendars fills it. */
    public Calendars calendars = new Calendars();

    /**
     * Region id <-> int index (EnvConfig region_list, fab regions and transit regions, plus the regions of misc rows);
     * filled by EnvConfigParser.buildCalendars / PlanBuilder.buildProblem. Resolve a region once with it, then carry
     * the int ({@code CrewSeat.regionIdx}).
     */
    public RegionIndex regions = RegionIndex.of(java.util.List.of());

    /**
     * Misc regions treated as "no region" (design section 7, parameter {@code misc_neutral_regions}); default
     * {@code {r_other}}. Set by Main from SolverParams before the build step, never touched by the build step.
     */
    public Set<String> miscNeutralRegions = Set.of("r_other");

    /**
     * trip_list validation mode (parameter {@code trip_validation}): true = any trip_list warning fails the build with
     * InvalidInputDataException, false = warnings only. Set by Main from SolverParams before the build step, never
     * touched by the build step.
     */
    public boolean tripValidationStrict = false;

    /**
     * Day marks of the enforced Fixed trips (design section 5), indexed by employee id and dayId; read through
     * {@link WorkerCalendar#isOpen}. Built by PlanBuilder.buildProblem; empty (no marks, every worker on the fast
     * path) without a trip_list.
     */
    public TripCalendar tripCalendar = TripCalendar.empty();

    /** Average hours per real employee that soft-balance-total-hours aims for; set by PlanBuilder.buildProblem. */
    public double targetHoursPerEmp = 0.0;
    /** Affinity tag id -> weight; filled by EnvConfigParser.parseEnv. */
    public final Map<String, Integer> affinityTagWeights = new HashMap<>();
    /** Operation id -> number of employees able to do it; filled by EnvConfigParser.parseEnv. */
    public final Map<String, Integer> opCapacity = new HashMap<>();
    /** Operation id -> average skill level of the employees able to do it; filled by EnvConfigParser.parseEnv. */
    public final Map<String, Double> opAvgSkill = new HashMap<>();
    /** First dayId of the solver horizon; set by PlanBuilder.buildProblem. */
    public int solveMinDay = 0;
    /** Last dayId of the solver horizon; set by PlanBuilder.buildProblem. */
    public int solveMaxDay = 0;

    // dayId -> calendar bucket lookups. The overtime constraints previously
    // joined DaySlot only to read the year/month of a dayId; these caches give
    // the same information without the join. Populated once per solve.
    /** dayId -> year; filled by PlanBuilder.buildProblem. */
    public final Map<Integer, Integer> dayToYear = new HashMap<>();
    /** dayId -> year*100 + month; filled by PlanBuilder.buildProblem. */
    public final Map<Integer, Integer> dayToYm = new HashMap<>();

    // fixed schedule background (built once per solveFromYaml)
    /** empId -> dayId -> hours of Fixed work inside the horizon; filled by PlanBuilder.buildProblem. */
    public final Map<Integer, Map<Integer, Integer>> fixedHoursByEmpDay = new HashMap<>();
    /** empId -> dayId -> fabs of Fixed work inside the horizon; filled by PlanBuilder.buildProblem. */
    public final Map<Integer, Map<Integer, Set<String>>> fixedFactoriesByEmpDay = new HashMap<>();

    /** empId -> total Fixed hours inside the horizon; filled by PlanBuilder.buildProblem. */
    public final Map<Integer, Integer> fixedTotalHoursByEmp = new HashMap<>();

    /** empId -> year -> Fixed overtime (max(0, hours-8)); filled by PlanBuilder.buildProblem. */
    public final Map<Integer, Map<Integer, Integer>> fixedAnnualOtByEmpYear = new HashMap<>();

    /** empId -> year*100+month -> Fixed overtime; filled by PlanBuilder.buildProblem. */
    public final Map<Integer, Map<Integer, Integer>> fixedMonthlyOtByEmpYm = new HashMap<>();

    /** empId -> dayId -> number of Fixed tasks (counts tasks, not hours); filled by PlanBuilder.buildProblem. */
    public final Map<Integer, Map<Integer, Integer>> fixedTaskcountByEmpDay = new HashMap<>();
}
