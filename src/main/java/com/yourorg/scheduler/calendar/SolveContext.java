package com.yourorg.scheduler.calendar;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Per-solve state that used to live in static fields of EmployeeSchedule. One current context per JVM
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

    /** Fab/region/customer/worker calendars and stay/overtime limits; empty until EmployeeSchedule.buildCalendars fills it. */
    public Calendars calendars = new Calendars();

    /** Average hours per real employee that soft-balance-total-hours aims for; set by EmployeeSchedule.buildProblem. */
    public double targetHoursPerEmp = 0.0;
    /** Affinity tag id -> weight; filled by EmployeeSchedule.parseEnv. */
    public final Map<String, Integer> affinityTagWeights = new HashMap<>();
    /** Operation id -> number of employees able to do it; filled by EmployeeSchedule.parseEnv. */
    public final Map<String, Integer> opCapacity = new HashMap<>();
    /** Operation id -> average skill level of the employees able to do it; filled by EmployeeSchedule.parseEnv. */
    public final Map<String, Double> opAvgSkill = new HashMap<>();
    /** First dayId of the solver horizon; set by EmployeeSchedule.buildProblem. */
    public int solveMinDay = 0;
    /** Last dayId of the solver horizon; set by EmployeeSchedule.buildProblem. */
    public int solveMaxDay = 0;

    // dayId -> calendar bucket lookups. The overtime constraints previously
    // joined DaySlot only to read the year/month of a dayId; these caches give
    // the same information without the join. Populated once per solve.
    /** dayId -> year; filled by EmployeeSchedule.buildProblem. */
    public final Map<Integer, Integer> dayToYear = new HashMap<>();
    /** dayId -> year*100 + month; filled by EmployeeSchedule.buildProblem. */
    public final Map<Integer, Integer> dayToYm = new HashMap<>();

    // fixed schedule background (built once per solveFromYaml)
    /** empId -> dayId -> hours of Fixed work inside the horizon; filled by EmployeeSchedule.buildProblem. */
    public final Map<Integer, Map<Integer, Integer>> fixedHoursByEmpDay = new HashMap<>();
    /** empId -> dayId -> fabs of Fixed work inside the horizon; filled by EmployeeSchedule.buildProblem. */
    public final Map<Integer, Map<Integer, Set<String>>> fixedFactoriesByEmpDay = new HashMap<>();

    /** empId -> total Fixed hours inside the horizon; filled by EmployeeSchedule.buildProblem. */
    public final Map<Integer, Integer> fixedTotalHoursByEmp = new HashMap<>();

    /** empId -> year -> Fixed overtime (max(0, hours-8)); filled by EmployeeSchedule.buildProblem. */
    public final Map<Integer, Map<Integer, Integer>> fixedAnnualOtByEmpYear = new HashMap<>();

    /** empId -> year*100+month -> Fixed overtime; filled by EmployeeSchedule.buildProblem. */
    public final Map<Integer, Map<Integer, Integer>> fixedMonthlyOtByEmpYm = new HashMap<>();

    /** empId -> dayId -> number of Fixed tasks (counts tasks, not hours); filled by EmployeeSchedule.buildProblem. */
    public final Map<Integer, Map<Integer, Integer>> fixedTaskcountByEmpDay = new HashMap<>();
}
