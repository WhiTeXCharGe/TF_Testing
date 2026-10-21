package com.yourorg.scheduler.calendar;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Per-solve state that used to live in static fields of EmployeeSchedule. One current context per JVM
 * ({@link #current()}); it is refilled by the build step, never replaced by it.
 */
public class SolveContext {

    private static volatile SolveContext current;

    /** Never null: lazily creates a fresh empty context. */
    public static SolveContext current() {
        SolveContext c = current;
        if (c == null) {
            synchronized (SolveContext.class) {
                c = current;
                if (c == null) {
                    c = new SolveContext();
                    current = c;
                }
            }
        }
        return c;
    }

    public static void set(SolveContext ctx) {
        current = ctx;
    }

    /** Installs a fresh empty context (calendars null, maps empty, numbers 0). Intended for tests. */
    public static void clear() {
        current = new SolveContext();
    }

    // calendars (null until buildCalendars has run)
    public Calendars calendars = null;

    public double targetHoursPerEmp = 0.0;
    public final Map<String, Integer> affinityTagWeights = new HashMap<>();
    public final Map<String, Integer> opCapacity = new HashMap<>();
    public final Map<String, Double> opAvgSkill = new HashMap<>();
    public int solveMinDay = 0;
    public int solveMaxDay = 0;

    // dayId -> calendar bucket lookups. The overtime constraints previously
    // joined DaySlot only to read the year/month of a dayId; these caches give
    // the same information without the join. Populated once per solve.
    public final Map<Integer, Integer> dayToYear = new HashMap<>();
    public final Map<Integer, Integer> dayToYm = new HashMap<>(); // year*100 + month

    // fixed schedule background (built once per solveFromYaml)
    public final Map<Integer, Map<Integer, Integer>> fixedHoursByEmpDay = new HashMap<>();
    public final Map<Integer, Map<Integer, Set<String>>> fixedFactoriesByEmpDay = new HashMap<>();

    // total fixed hours per employee
    public final Map<Integer, Integer> fixedTotalHoursByEmp = new HashMap<>();

    // fixed overtime (max(0, hours-8)) per employee per year
    public final Map<Integer, Map<Integer, Integer>> fixedAnnualOtByEmpYear = new HashMap<>();

    // fixed overtime per employee per year-month (ym = year*100 + month)
    public final Map<Integer, Map<Integer, Integer>> fixedMonthlyOtByEmpYm = new HashMap<>();

    // fixed task-count per employee per day (counts tasks, not hours)
    public final Map<Integer, Map<Integer, Integer>> fixedTaskcountByEmpDay = new HashMap<>();
}
