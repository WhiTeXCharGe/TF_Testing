package com.yourorg.scheduler;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDate;
import java.util.Map;
import java.util.Set;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import com.yourorg.scheduler.Characterization.ConstraintTotal;
import com.yourorg.scheduler.Characterization.Snapshot;
import com.yourorg.scheduler.calendar.SolveContext;
import com.yourorg.scheduler.model.CrewSeat;
import com.yourorg.scheduler.model.SinglePassPlan;

/**
 * Design section 7: a misc task uses its own region (unless it is in misc_neutral_regions, default {r_other}).
 * Scenario tests build a one-worker world: worker W (manager) with a Fixed misc row on day 5 and a Flexible r1 task
 * of W; the misc region, the transit to r1 and the flexible start day are parameters.
 */
class MiscRegionTest {

    static final String TRANSIT = "emp-region-transit-gap";
    static final LocalDate DAY0 = LocalDate.of(2027, 1, 11);
    static final ConstraintTotal ZERO = new ConstraintTotal("0hard/0medium/0soft", 0);

    @TempDir
    Path tmp;

    @BeforeEach
    @AfterEach
    void resetContext() {
        SolveContext.clear();
    }

    // ------------------------------------------------------------------ scenario builder

    /**
     * The phase window starts at the flexible task's start day: a phase-1 block is always moved back to its window
     * start (warm-start clamp), so this keeps the warm start exactly where the rows say.
     *
     * @param miscRegionLine yaml line under the misc task, e.g. "    region: r2", or "" for none */
    private SinglePassPlan scenario(String name, String miscRegionLine, int transitToR1, int flexStartDay)
            throws Exception {
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
                  - id: f1
                    name: Fab One
                    region: r1
                    customer_company: c1
                    unavailable_dates: []
                  region_list:
                  - {id: r_other, name: Other, max_stay_on: 10000, max_annual_stay: 10000, stay_off_interval: 1, unavailable_dates: []}
                  - {id: r1, name: R1, max_stay_on: 10000, max_annual_stay: 10000, stay_off_interval: 1, unavailable_dates: []}
                  - {id: r2, name: R2, max_stay_on: 10000, max_annual_stay: 10000, stay_off_interval: 1, unavailable_dates: []}
                  customer_company_list:
                  - {id: c1, name: C1, unavailable_dates: []}
                  worker_company_list:
                  - {id: wc, name: WC, annual_overtime_limit: 10000, monthly_overtime_limit: 10000, unavailable_dates: []}
                  transite_day_map:
                  - {from: r2, to: r1, days: %d}
                  - {from: r1, to: r2, days: %d}
                  - {from: r_other, to: r1, days: %d}
                  - {from: r1, to: r_other, days: %d}
                  affinity_tag: []
                  worker_list:
                  - id: W
                    name: Worker
                    worker_company: wc
                    is_manager: true
                    skill_map: {p1o1: 3}
                    worker_type_by_operation: {}
                    fab_suitability_map: []
                    unavailable_dates: []
                    affinity: []
                """.formatted(transitToR1, transitToR1, transitToR1, transitToR1);

        StringBuilder rows = new StringBuilder();
        // the fixed misc row on day 5
        rows.append("  - worker: W\n    operation_task: misc_1\n");
        rows.append("    start_date: ").append(date(5)).append("\n    end_date: ").append(date(5)).append("\n");
        rows.append("    work_date_list:\n    - {date: ").append(date(5)).append(", hour: 8}\n");
        rows.append("    plan_flexibility: Fixed\n");
        // the flexible r1 task, 5 days of 8 h from flexStartDay (workload 40 h)
        rows.append("  - worker: W\n    operation_task: x1p1o1\n");
        rows.append("    start_date: ").append(date(flexStartDay)).append("\n    end_date: ")
            .append(date(flexStartDay + 4)).append("\n    work_date_list:\n");
        for (int d = flexStartDay; d < flexStartDay + 5; d++) {
            rows.append("    - {date: ").append(date(d)).append(", hour: 8}\n");
        }
        rows.append("    plan_flexibility: Flexible\n");

        String sched = """
                schedule:
                  plan_range:
                    start_date: %s
                    end_date: %s
                  workflow_task_list:
                  - id: x1
                    name: Module X
                    workflow: wf1
                    fab: f1
                    phase_task_list:
                    - id: x1p1
                      name: Move In
                      phase: p1
                      start_date: %s
                      end_date: %s
                      operation_task_list:
                      - id: x1p1o1
                        name: Mech
                        operation: p1o1
                        workload_hours: 40
                        recommends_worker_min: 1
                        recommends_worker_max: 1
                  misc_task_list:
                  - id: misc_1
                    name: Misc
                %s
                  assignment_list:
                %s""".formatted(date(0), date(13), date(flexStartDay), date(13), miscRegionLine, rows);

        Path dir = tmp.resolve(name);
        Files.createDirectories(dir);
        Files.writeString(dir.resolve("EnvConfig.yaml"), env);
        Files.writeString(dir.resolve("Schedule.yaml"), sched);
        return Characterization.buildFrom(dir, dir.resolve("work"));
    }

    private static String date(int dayId) {
        return DAY0.plusDays(dayId).toString().replace('-', '/');
    }

    private static ConstraintTotal transit(SinglePassPlan p) {
        return Characterization.initialTotals(p).get(TRANSIT);
    }

    private static CrewSeat miscSeat(SinglePassPlan p) {
        return p.seats.stream().filter(s -> s.pinnedFixed && "misc_1".equals(s.opId)).findFirst().orElseThrow();
    }

    // ------------------------------------------------------------------ tests

    @Test
    void default_neutral_list_keeps_baseline_totals() throws Exception {
        assertEquals(Set.of("r_other"), SolveContext.current().miscNeutralRegions);
        SinglePassPlan p = Characterization.buildFrom(CharacterizationTest.SMALL_FIXTURE, tmp.resolve("small"));
        Snapshot initial = Characterization.analyze(p);
        // throws with a per-constraint diff when any total differs from the 1.27 baseline
        Characterization.checkSection(CharacterizationTest.SMALL_GOLDEN, "initial", Map.of(), initial);
    }

    @Test
    void misc_with_real_region_counts_for_transit() throws Exception {
        // misc in r2 on day 5, flexible r1 task from day 6, transit r2->r1 = 2 days: earliest legal day 8 -> violated
        SinglePassPlan tight = scenario("tight", "    region: r2", 2, 6);
        ConstraintTotal t = transit(tight);
        assertEquals(1, t.matches(), t.toString());
        assertEquals("-2hard/0medium/0soft", t.score());

        // transit 1 and one free day (day 6) between: earliest legal day 7 -> fine
        SinglePassPlan loose = scenario("loose", "    region: r2", 1, 7);
        assertEquals(ZERO, transit(loose));

        // the seat of the misc row carries its own region, the module seat its fab's region
        assertTrue(miscSeat(tight).regionIdx >= 0);
        assertEquals("r2", miscSeat(tight).regionId());
        CrewSeat module = tight.seats.stream().filter(s -> !s.pinnedFixed).findFirst().orElseThrow();
        assertEquals("r1", module.regionId());
    }

    @Test
    void misc_without_region_is_regionless() throws Exception {
        SinglePassPlan p = scenario("noregion", "", 2, 6);
        assertEquals(ZERO, transit(p));
        assertEquals(-1, miscSeat(p).regionIdx);
        assertNull(miscSeat(p).regionId());

        // a blank region is the same as none
        SinglePassPlan blank = scenario("blank", "    region: ''", 2, 6);
        assertEquals(ZERO, transit(blank));
        assertEquals(-1, miscSeat(blank).regionIdx);
    }

    @Test
    void default_neutral_r_other_misc_is_regionless_in_a_crafted_scenario() throws Exception {
        SinglePassPlan p = scenario("rother-default", "    region: r_other", 2, 6);
        assertEquals(ZERO, transit(p));
        assertEquals(-1, miscSeat(p).regionIdx);
    }

    @Test
    void empty_neutral_list_makes_r_other_count() throws Exception {
        // crafted: r_other misc on day 5, r1 task from day 6, transit r_other->r1 = 2
        SolveContext.current().miscNeutralRegions = Set.of();
        SinglePassPlan p = scenario("rother-empty", "    region: r_other", 2, 6);
        ConstraintTotal t = transit(p);
        assertEquals(1, t.matches(), t.toString());
        assertEquals("-2hard/0medium/0soft", t.score());
        assertEquals("r_other", miscSeat(p).regionId());

        // the small fixture: its r_other misc row now counts, so the baseline totals change (documented behaviour
        // change; e.g. soft-minimize-region-transitions sees a region pair it did not see before)
        SolveContext.clear();
        SolveContext.current().miscNeutralRegions = Set.of();
        SinglePassPlan small = Characterization.buildFrom(CharacterizationTest.SMALL_FIXTURE, tmp.resolve("small"));
        Snapshot initial = Characterization.analyze(small);
        @SuppressWarnings("unchecked")
        Snapshot golden = Characterization.snapshotFromJson(
                (Map<String, Object>) Characterization.readGolden(CharacterizationTest.SMALL_GOLDEN).get("initial"));
        String diff = Characterization.diff(golden, initial);
        assertFalse(diff.isEmpty(), "an r_other misc row that counts must change the small fixture totals");
        System.out.println("[MiscRegionTest] empty neutral list changes the small fixture initial totals:\n" + diff);
    }
}
