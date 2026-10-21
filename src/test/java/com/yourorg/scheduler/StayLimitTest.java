package com.yourorg.scheduler;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import com.yourorg.scheduler.Characterization.ConstraintTotal;
import com.yourorg.scheduler.calendar.SolveContext;
import com.yourorg.scheduler.calendar.TripCalendar;
import com.yourorg.scheduler.model.SinglePassPlan;

/**
 * Design section 4.1 / decision D14: the stay limits ({@code emp-region-stay-max-on}, {@code emp-region-annual-stay-max})
 * measure a stay from FI to FO, both included; the travel gap ({@code emp-region-transit-gap}) keeps counting worked days
 * only.
 *
 * <p>Every scenario is a crafted one- or two-worker world (workers W and V, both managers): each Flexible task is its
 * own module whose phase window starts on the task's first day, with a warm-start row of 8 h per day, so the initial
 * (warm-start) solution covers exactly the listed days. Fab f1 lies in r1, f2 in r2. Fixed trips go to the
 * {@code trip_list}. Only the initial totals are measured (no solving), so the expected totals are exact.
 */
class StayLimitTest {

    static final String STAY = "emp-region-stay-max-on";
    static final String ANNUAL = "emp-region-annual-stay-max";
    static final String TRANSIT = "emp-region-transit-gap";
    static final LocalDate DAY0 = LocalDate.of(2027, 1, 11);
    static final int PLAN_END = 24;
    static final int OFF = 10000;
    static final ConstraintTotal ZERO = new ConstraintTotal("0hard/0medium/0soft", 0);

    @TempDir
    Path tmp;

    @BeforeEach
    @AfterEach
    void resetContext() {
        SolveContext.clear();
    }

    // ------------------------------------------------------------------ scenario builder

    /** A Flexible task of {@code worker} in fab {@code fab} on {@code days} days from {@code start}. */
    record Task(String worker, String fab, int start, int days) {}

    /** An enforced Fixed trip of {@code worker} in {@code region} from FI {@code start} to FO {@code end} (dayIds). */
    record TripDef(String worker, String region, int start, int end) {}

    static final class Scenario {
        int stayR1 = OFF, annualR1 = OFF, stayR2 = OFF, annualR2 = OFF;
        int interval = 3;
        int transit = 1;
        final List<Task> tasks = new ArrayList<>();
        final List<TripDef> trips = new ArrayList<>();

        Scenario r1(int stay, int annual) { stayR1 = stay; annualR1 = annual; return this; }
        Scenario r2(int stay, int annual) { stayR2 = stay; annualR2 = annual; return this; }
        Scenario interval(int i) { interval = i; return this; }
        Scenario transit(int t) { transit = t; return this; }
        Scenario task(String fab, int start, int days) { return task("W", fab, start, days); }
        Scenario task(String worker, String fab, int start, int days) {
            tasks.add(new Task(worker, fab, start, days));
            return this;
        }
        Scenario trip(String region, int start, int end) { return trip("W", region, start, end); }
        Scenario trip(String worker, String region, int start, int end) {
            trips.add(new TripDef(worker, region, start, end));
            return this;
        }
    }

    private SinglePassPlan build(String name, Scenario s) throws Exception {
        String env = """
                environment:
                  workflow_list:
                  - id: wf1
                    name: Tool
                    phase_list:
                    - id: p1
                      name: Move In
                      operation_list:
                      - id: p1o1
                        name: Mech
                        work_hours: [8]
                        min_worker_num: 1
                        max_worker_num: 1
                  fab_list:
                  - {id: f1, name: Fab One, region: r1, customer_company: c1, unavailable_dates: []}
                  - {id: f2, name: Fab Two, region: r2, customer_company: c1, unavailable_dates: []}
                  region_list:
                  - {id: r_other, name: Other, max_stay_on: 10000, max_annual_stay: 10000, stay_off_interval: %d, unavailable_dates: []}
                  - {id: r1, name: R1, max_stay_on: %d, max_annual_stay: %d, stay_off_interval: %d, unavailable_dates: []}
                  - {id: r2, name: R2, max_stay_on: %d, max_annual_stay: %d, stay_off_interval: %d, unavailable_dates: []}
                  customer_company_list:
                  - {id: c1, name: C1, unavailable_dates: []}
                  worker_company_list:
                  - {id: wc, name: WC, annual_overtime_limit: 10000, monthly_overtime_limit: 10000, unavailable_dates: []}
                  transite_day_map:
                  - {from: r1, to: r2, days: %d}
                  - {from: r2, to: r1, days: %d}
                  affinity_tag: []
                  worker_list:
                  - {id: W, name: Worker W, worker_company: wc, is_manager: true, skill_map: {p1o1: 3}, worker_type_by_operation: {}, fab_suitability_map: [], unavailable_dates: [], affinity: []}
                  - {id: V, name: Worker V, worker_company: wc, is_manager: true, skill_map: {p1o1: 3}, worker_type_by_operation: {}, fab_suitability_map: [], unavailable_dates: [], affinity: []}
                """.formatted(s.interval, s.stayR1, s.annualR1, s.interval, s.stayR2, s.annualR2, s.interval,
                s.transit, s.transit);

        StringBuilder modules = new StringBuilder();
        StringBuilder rows = new StringBuilder();
        int n = 0;
        for (Task t : s.tasks) {
            n++;
            String mod = "x" + n;
            modules.append("""
                      - id: %1$s
                        name: Module %1$s
                        workflow: wf1
                        fab: %2$s
                        phase_task_list:
                        - id: %1$sp1
                          name: Move In
                          phase: p1
                          start_date: %3$s
                          end_date: %4$s
                          operation_task_list:
                          - id: %1$sp1o1
                            name: Mech
                            operation: p1o1
                            workload_hours: %5$d
                            recommends_worker_min: 1
                            recommends_worker_max: 1
                    """.formatted(mod, t.fab(), date(t.start()), date(PLAN_END), 8 * t.days()));
            rows.append("  - worker: ").append(t.worker()).append("\n    operation_task: ").append(mod)
                .append("p1o1\n    start_date: ").append(date(t.start())).append("\n    end_date: ")
                .append(date(t.start() + t.days() - 1)).append("\n    work_date_list:\n");
            for (int d = t.start(); d < t.start() + t.days(); d++) {
                rows.append("    - {date: ").append(date(d)).append(", hour: 8}\n");
            }
            rows.append("    plan_flexibility: Flexible\n");
        }
        StringBuilder trips = new StringBuilder();
        if (!s.trips.isEmpty()) {
            trips.append("  trip_list:\n");
            for (String w : new String[] {"W", "V"}) {
                List<TripDef> mine = s.trips.stream().filter(t -> t.worker().equals(w)).toList();
                if (mine.isEmpty()) continue;
                trips.append("  - worker: ").append(w).append("\n    trips:\n");
                for (TripDef t : mine) {
                    trips.append("    - {start_date: ").append(date(t.start())).append(", end_date: ")
                        .append(date(t.end())).append(", region: ").append(t.region())
                        .append(", plan_flexibility: Fixed, description: site}\n");
                }
            }
        }

        String sched = "schedule:\n  plan_range:\n    start_date: " + date(0) + "\n    end_date: " + date(PLAN_END)
                + "\n  workflow_task_list:\n" + modules + "  misc_task_list: []\n  assignment_list:\n" + rows + trips;

        Path dir = tmp.resolve(name);
        Files.createDirectories(dir);
        Files.writeString(dir.resolve("EnvConfig.yaml"), env);
        Files.writeString(dir.resolve("Schedule.yaml"), sched);
        return Characterization.buildFrom(dir, dir.resolve("work"));
    }

    private static String date(int dayId) {
        return DAY0.plusDays(dayId).toString().replace('-', '/');
    }

    private static ConstraintTotal hard(int penalty, int matches) {
        return penalty == 0 ? ZERO : new ConstraintTotal("-" + penalty + "hard/0medium/0soft", matches);
    }

    private Map<String, ConstraintTotal> totals(String name, Scenario s) throws Exception {
        return Characterization.initialTotals(build(name, s));
    }

    // ------------------------------------------------------------------ tests

    @Test
    void derived_stay_counts_plus_two() throws Exception {
        // tasks day 2..4: stay FI 1 .. FO 5 = 5 days, limit 5 -> no penalty
        Map<String, ConstraintTotal> three = totals("three", new Scenario().r1(5, OFF).task("f1", 2, 3));
        assertEquals(ZERO, three.get(STAY));
        // tasks day 2..5: 1..6 = 6 days -> penalty 1
        Map<String, ConstraintTotal> four = totals("four", new Scenario().r1(5, OFF).task("f1", 2, 4));
        assertEquals(hard(1, 1), four.get(STAY));
    }

    @Test
    void annual_sums_stays() throws Exception {
        // days 2..5 and 10..13 (gap 4 >= interval 3): stays 1..6 and 9..14, 6 + 6 = 12, annual limit 10 -> penalty 2
        Map<String, ConstraintTotal> t = totals("annual",
                new Scenario().r1(OFF, 10).task("f1", 2, 4).task("f1", 10, 4));
        assertEquals(hard(2, 1), t.get(ANNUAL));
        assertEquals(ZERO, t.get(STAY));
    }

    @Test
    void fixed_trip_length_counts() throws Exception {
        // Fixed trip r1 day 1..10 with one task on day 5: the stay is the trip, 10 days
        Map<String, ConstraintTotal> eight = totals("eight", new Scenario().r1(8, OFF).trip("r1", 1, 10)
                .task("f1", 5, 1));
        assertEquals(hard(2, 1), eight.get(STAY));
        Map<String, ConstraintTotal> ten = totals("ten", new Scenario().r1(10, OFF).trip("r1", 1, 10)
                .task("f1", 5, 1));
        assertEquals(ZERO, ten.get(STAY));
        // annual: the same 10 days
        Map<String, ConstraintTotal> annual = totals("annual", new Scenario().r1(OFF, 7).trip("r1", 1, 10)
                .task("f1", 5, 1));
        assertEquals(hard(3, 1), annual.get(ANNUAL));
    }

    @Test
    void trip_without_tasks_counts_as_a_stay() throws Exception {
        // r1 trip 2..8 (7 days) without any r1 task; W works r2 on day 15 only
        Map<String, ConstraintTotal> t = totals("notask", new Scenario().r1(5, 6).trip("r1", 2, 8)
                .task("f2", 15, 1));
        assertEquals(hard(2, 1), t.get(STAY));
        assertEquals(hard(1, 1), t.get(ANNUAL));
    }

    @Test
    void task_outside_trip_extends_stay() throws Exception {
        // trip 3..7, task day 9 (present-day gap 1 < interval 3): stay = min(3, 8) .. max(7, 10) = 3..10 = 8 days
        Map<String, ConstraintTotal> six = totals("six", new Scenario().r1(6, OFF).trip("r1", 3, 7)
                .task("f1", 9, 1));
        assertEquals(hard(2, 1), six.get(STAY));
        Map<String, ConstraintTotal> eight = totals("eight", new Scenario().r1(8, 8).trip("r1", 3, 7)
                .task("f1", 9, 1));
        assertEquals(ZERO, eight.get(STAY));
        assertEquals(ZERO, eight.get(ANNUAL));
        // a task before the trip: day 1, trip 3..7 -> 0..7 = 8 days
        Map<String, ConstraintTotal> before = totals("before", new Scenario().r1(7, OFF).trip("r1", 3, 7)
                .task("f1", 1, 1));
        assertEquals(hard(1, 1), before.get(STAY));
    }

    @Test
    void touching_stays_same_region_count_shared_day_once() throws Exception {
        // interval 1: tasks 2..3 and 5..6 (gap 1 >= 1) are two stays, 1..4 and 4..7; day 4 counts once: 4 + 4 - 1 = 7
        Map<String, ConstraintTotal> derived = totals("derived", new Scenario().interval(1).r1(3, 5)
                .task("f1", 2, 2).task("f1", 5, 2));
        assertEquals(hard(2, 1), derived.get(ANNUAL));
        assertEquals(hard(1, 1), derived.get(STAY)); // longest stay 4, limit 3
        // two touching r1 trips 1..4 and 4..8 are one stay 1..8 = 8 days (not 9)
        Map<String, ConstraintTotal> trips = totals("trips", new Scenario().r1(7, 7).trip("r1", 1, 4)
                .trip("r1", 4, 8).task("f1", 6, 1));
        assertEquals(hard(1, 1), trips.get(STAY));
        assertEquals(hard(1, 1), trips.get(ANNUAL));
    }

    @Test
    void touching_trips_different_regions_present_in_both() throws Exception {
        // r1 trip 1..5 and r2 trip 5..9: day 5 is present in r1 and in r2, each stay is 5 days
        Map<String, ConstraintTotal> t = totals("touch", new Scenario().r1(4, 4).r2(4, 4)
                .trip("r1", 1, 5).trip("r2", 5, 9).task("f1", 3, 1).task("f2", 7, 1));
        assertEquals(hard(2, 2), t.get(STAY));
        assertEquals(hard(2, 2), t.get(ANNUAL));
    }

    @Test
    void travel_gap_ignores_trips() throws Exception {
        // trip r1 1..10, r1 task day 9 (last day inside), r2 task day 11 (the day after FO)
        // transit 1: earliest legal r2 day = 9 + 1 + 1 = 11 -> fine (counting FO day 10 would need day 12)
        Map<String, ConstraintTotal> one = totals("one", new Scenario().transit(1).trip("r1", 1, 10)
                .task("f1", 9, 1).task("f2", 11, 1));
        assertEquals(ZERO, one.get(TRANSIT));
        // transit 2: earliest legal 12 -> penalty 1, measured from the worked day 9
        Map<String, ConstraintTotal> two = totals("two", new Scenario().transit(2).trip("r1", 1, 10)
                .task("f1", 9, 1).task("f2", 11, 1));
        assertEquals(hard(1, 1), two.get(TRANSIT));
        // the trip alone (no r1 worked day) never feeds the travel gap
        Map<String, ConstraintTotal> tripOnly = totals("triponly", new Scenario().transit(2).trip("r1", 1, 10)
                .task("f2", 11, 1));
        assertEquals(ZERO, tripOnly.get(TRANSIT));
    }

    @Test
    void limits_off_keeps_zero() throws Exception {
        Map<String, ConstraintTotal> t = totals("off", new Scenario().trip("r1", -5, 12).trip("r2", 14, 24)
                .task("f1", 3, 6).task("f2", 16, 5).task("V", "f1", 1, 20));
        assertEquals(ZERO, t.get(STAY));
        assertEquals(ZERO, t.get(ANNUAL));
    }

    @Test
    void unclamped_trip_before_horizon_still_counts() throws Exception {
        // trip r1 from day -3 (before the plan start) to day 2, task day 1: the stay is -3..2 = 6 days
        Map<String, ConstraintTotal> t = totals("before", new Scenario().r1(5, 5).trip("r1", -3, 2)
                .task("f1", 1, 1));
        assertEquals(hard(1, 1), t.get(STAY));
        assertEquals(hard(1, 1), t.get(ANNUAL));
        // and past the plan end: trip 20..30 (plan ends on day 24), task day 22 -> 11 days
        Map<String, ConstraintTotal> after = totals("after", new Scenario().r1(9, OFF).trip("r1", 20, 30)
                .task("f1", 22, 1));
        assertEquals(hard(2, 1), after.get(STAY));
    }

    @Test
    void trip_intervals_are_unclamped_and_shared_empty_without_trips() throws Exception {
        build("intervals", new Scenario().trip("r1", -3, 2).trip("r2", 20, 30).task("f1", 1, 1));
        TripCalendar c = SolveContext.current().tripCalendar;
        int w = -1;
        for (int e = 0; e < 4 && w < 0; e++) if (!c.tripsOf(e).isEmpty()) w = e;
        assertTrue(w > 0, "W has trips");
        List<TripCalendar.TripInterval> trips = c.tripsOf(w);
        assertEquals(2, trips.size());
        assertEquals(-3, trips.get(0).startDay());
        assertEquals(2, trips.get(0).endDay());
        assertEquals(SolveContext.current().regions.indexOf("r1"), trips.get(0).regionIdx());
        assertEquals(20, trips.get(1).startDay());
        assertEquals(30, trips.get(1).endDay());
        // workers without trips (and ids outside the calendar) share one empty list
        assertSame(c.tripsOf(-1), c.tripsOf(999));
        assertSame(TripCalendar.empty().tripsOf(0), c.tripsOf(999));
        assertTrue(c.tripsOf(999).isEmpty());
    }

    @Test
    void workers_without_trips_unaffected_except_plus_two() throws Exception {
        // V (no trips): r1 days 2..5 -> stay 6, limit 5 -> penalty 1; W has an r1 trip 10..14 with a task on day 12
        // (stay 5, no penalty). Only V's +2 shows.
        Map<String, ConstraintTotal> t = totals("mixed", new Scenario().r1(5, 6).task("V", "f1", 2, 4)
                .trip("W", "r1", 10, 14).task("W", "f1", 12, 1));
        assertEquals(hard(1, 1), t.get(STAY));
        assertEquals(ZERO, t.get(ANNUAL));
        // V alone gives the same totals
        SolveContext.clear();
        Map<String, ConstraintTotal> alone = totals("alone", new Scenario().r1(5, 6).task("V", "f1", 2, 4));
        assertEquals(t.get(STAY), alone.get(STAY));
        assertEquals(t.get(ANNUAL), alone.get(ANNUAL));
    }
}
