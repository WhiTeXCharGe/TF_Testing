package com.yourorg.scheduler;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import com.yourorg.scheduler.build.PlanBuilder;
import com.yourorg.scheduler.calendar.SolveContext;
import com.yourorg.scheduler.export.ScheduleExporter;
import com.yourorg.scheduler.model.BlockDecision;
import com.yourorg.scheduler.model.SinglePassPlan;

/**
 * Trip marks end to end on the {@code small} fixture (design sections 5 and 5.2): seeded solve
 * ({@link SolverFactories#forTests}, {@value #STEPS} steps, seed {@value #SEED}) and export through
 * {@link ScheduleExporter}.
 *
 * <p>Main scenario: w2 gets two Fixed trips, r2 from 2026/03/23 (FI, day 0) to 03/27 (FO, day 4) and r1 from 03/30
 * (FI, day 7; also w2's personal unavailable date in the fixture) to 04/05 (FO, day 13). Without trips the solver
 * puts w2 on e2p1o1 (r2) 03/23..03/25, on e1p2o2 (r1) 03/25..03/27 and on e1p2o1 (r1) up to 04/05, i.e. on FI/FO days
 * and on r1 work inside the r2 trip. With the trips it puts w2 on e2p1o1 (r2) 03/24 and 03/25, inside the r2 trip.
 * w2's Fixed rows conflict with the first trip: e1p1o1 (r1) on 03/23 (FI) and 03/24 (r1 inside an r2 trip), the
 * region-less misc row on 03/27 (FO); they are kept and reported.
 *
 * <p>Other-region scenario: one r1 trip 03/24..03/27; without it w2 works r2 (e2p1o1) on 03/25 inside that window.
 *
 * <p>These descriptions are of the seeded trajectory (seed {@value #SEED}, 300 to 2000 steps) after the block weights
 * change (README, "Block weights"); the assertions only rely on the preconditions named in their messages.
 */
class TripMarksSolverTest {

    static final Path SMALL_FIXTURE = CharacterizationTest.SMALL_FIXTURE;
    static final int STEPS = 300;
    /**
     * Not 42: the scenario preconditions below depend on the seeded search trajectory. When phase-order and
     * block-end-within-window were re-weighted to 100 per day (task 20) the trajectory of seed 42 changed and no
     * longer put w2 on r2 work inside the r2 trip, at 300, 500, 1000 or 2000 steps (nor did seeds 1, 2, 7 at 1000 steps).
     * Seed 3 satisfies every precondition at 300, 1000, 1500 and 2000 steps; 300 keeps the class at about 6 s.
     */
    static final long SEED = 3L;

    static final String WORKER = "w2";

    /** {region, FI, FO} of the main scenario. */
    static final String[][] TRIPS = {
            {"r2", "2026/03/23", "2026/03/27"},
            {"r1", "2026/03/30", "2026/04/05"},
    };
    static final String[][] OTHER_REGION_TRIP = {{"r1", "2026/03/24", "2026/03/27"}};

    static final String INSIDE_UNAVAILABLE = "2026/03/25";
    static final String MISC_DAY = "2026/03/27";

    @TempDir
    Path tmp;

    @BeforeEach
    @AfterEach
    void resetContext() {
        SolveContext.clear();
    }

    // ------------------------------------------------------------------ scenario

    record Run(Map<String, Object> output, String stdout, SinglePassPlan best, LocalDate planStart, Path dir) {}

    /**
     * Copies the small fixture into {@code dir}, appends a trip_list with {@code trips} of w2 (none when null) and adds
     * {@code extraUnavailable} to w2's unavailable dates.
     */
    static Path prepare(Path dir, String[][] trips, String... extraUnavailable) throws Exception {
        Files.createDirectories(dir);
        String sched = Files.readString(SMALL_FIXTURE.resolve("Schedule.yaml"), StandardCharsets.UTF_8);
        if (trips != null) {
            StringBuilder sb = new StringBuilder(sched.endsWith("\n") ? sched : sched + "\n");
            sb.append("  trip_list:\n  - worker: w2\n    trips:\n");
            for (String[] t : trips) {
                sb.append("    - {start_date: ").append(t[1]).append(", end_date: ").append(t[2])
                        .append(", region: ").append(t[0]).append(", plan_flexibility: Fixed, description: site}\n");
            }
            sched = sb.toString();
        }
        Files.writeString(dir.resolve("Schedule.yaml"), sched, StandardCharsets.UTF_8);
        String env = Files.readString(SMALL_FIXTURE.resolve("EnvConfig.yaml"), StandardCharsets.UTF_8);
        String anchor = "        - 2026/03/30"; // w2's only unavailable date in the fixture
        assertTrue(env.contains(anchor));
        StringBuilder extra = new StringBuilder(anchor);
        for (String d : extraUnavailable) extra.append("\n        - ").append(d);
        env = env.replace(anchor, extra.toString());
        Files.writeString(dir.resolve("EnvConfig.yaml"), env, StandardCharsets.UTF_8);
        return dir;
    }

    /** Build (stdout captured), seeded solve, export into the same directory, read the output back. */
    static Run solveAndExport(Path dir) throws Exception {
        PrintStream old = System.out;
        ByteArrayOutputStream buf = new ByteArrayOutputStream();
        PlanBuilder.BuiltProblem bp;
        System.setOut(new PrintStream(buf, true, StandardCharsets.UTF_8));
        try {
            bp = PlanBuilder.buildProblem(dir.resolve("EnvConfig.yaml").toString(),
                    dir.resolve("Schedule.yaml").toString());
        } finally {
            System.setOut(old);
        }
        String stdout = buf.toString(StandardCharsets.UTF_8);
        SinglePassPlan best = Characterization.solveSeeded(bp.plan, STEPS, SEED);
        ScheduleExporter.overwriteScheduleWithAssignments(best, bp.sch.planStart,
                dir.resolve("Schedule.yaml").toString(), dir.resolve("EnvConfig.yaml").toString());
        return new Run(ScheduleExporterTest.schedule(dir.resolve("Schedule.yaml")), stdout, best,
                bp.sch.planStart, dir);
    }

    static Map<String, String> taskFab(Map<String, Object> sched) {
        Map<String, String> m = new HashMap<>();
        ScheduleExporterTest.collectTasks(sched, new HashSet<>(), m);
        return m;
    }

    /** Region of an operation_task in the small fixture (misc tasks: none). */
    static String regionOf(Map<String, String> taskFab, String task) {
        String fab = taskFab.get(task);
        return "f1".equals(fab) ? "r1" : "f2".equals(fab) ? "r2" : null;
    }

    /** The trip whose window strictly contains {@code date}, or null. */
    static String[] tripAround(String[][] trips, String date) {
        for (String[] t : trips) if (date.compareTo(t[1]) > 0 && date.compareTo(t[2]) < 0) return t;
        return null;
    }

    static boolean isTripEnd(String[][] trips, String date) {
        for (String[] t : trips) if (date.equals(t[1]) || date.equals(t[2])) return true;
        return false;
    }

    /** (task, date) of every work day of {@code worker} in rows with the given flexibility (null = all rows). */
    static List<String[]> workDays(Map<String, Object> sched, String worker, Boolean fixed) {
        List<String[]> out = new ArrayList<>();
        for (Map<String, Object> a : ScheduleExporterTest.assignments(sched)) {
            if (!worker.equals(String.valueOf(a.get("worker")))) continue;
            if (fixed != null && fixed != ScheduleExporterTest.isFixed(a)) continue;
            for (Map<String, Object> item : ScheduleExporterTest.workDates(a)) {
                out.add(new String[] {String.valueOf(a.get("operation_task")), String.valueOf(item.get("date"))});
            }
        }
        return out;
    }

    /** (a) and (b): no solved (Flexible) row of w2 on an FI/FO day, or in another region strictly inside a trip. */
    static void assertMarksHonoured(Run run, String[][] trips) {
        Map<String, String> fabs = taskFab(run.output());
        for (String[] td : workDays(run.output(), WORKER, false)) {
            assertFalse(isTripEnd(trips, td[1]), "w2 works on an FI/FO day: " + td[0] + " " + td[1]);
            String[] t = tripAround(trips, td[1]);
            if (t != null) {
                assertEquals(t[0], regionOf(fabs, td[0]), "w2 inside the " + t[0] + " trip in another region: "
                        + td[0] + " " + td[1]);
            }
        }
    }

    /** Days of w2's solved rows that a trip would close (FI/FO, or another region inside). */
    static List<String> closedDaysUsed(Run run, String[][] trips) {
        Map<String, String> fabs = taskFab(run.output());
        List<String> out = new ArrayList<>();
        for (String[] td : workDays(run.output(), WORKER, false)) {
            String[] t = tripAround(trips, td[1]);
            if (isTripEnd(trips, td[1]) || (t != null && !t[0].equals(regionOf(fabs, td[0])))) {
                out.add(td[0] + " " + td[1]);
            }
        }
        return out;
    }

    // ------------------------------------------------------------------ (a) (b) (c) (e)

    @Test
    void fixed_trip_marks_shape_the_solved_output() throws Exception {
        Run run = solveAndExport(prepare(tmp.resolve("trip"), TRIPS));
        Map<String, String> fabs = taskFab(run.output());

        // (a) + (b)
        assertMarksHonoured(run, TRIPS);

        // (c) work of a trip's own region inside the trip is allowed, and the solver uses it in the r2 trip here
        // (300..2000 steps, seed 3: e2p1o1 on 03/24 and 03/25). Only the r2 trip is asserted: that the r1 trip also
        // gets r1 work inside it (e1p2o1 03/31..04/04) is a property of this trajectory, not of the rules.
        for (String[] t : new String[][] {TRIPS[0]}) {
            boolean sameRegionInside = false;
            for (String[] td : workDays(run.output(), WORKER, false)) {
                if (tripAround(new String[][] {t}, td[1]) != null && t[0].equals(regionOf(fabs, td[0]))) {
                    sameRegionInside = true;
                }
            }
            assertTrue(sameRegionInside, "expected " + t[0] + " work of w2 inside the trip " + t[1] + ".." + t[2]);
        }

        // (e) the conflicting Fixed rows of w2 are kept unchanged and reported
        List<String> fixedDays = new ArrayList<>();
        for (String[] td : workDays(run.output(), WORKER, true)) fixedDays.add(td[0] + " " + td[1]);
        assertTrue(fixedDays.contains("e1p1o1 2026/03/23"), "Fixed row on the FI day kept: " + fixedDays);
        assertTrue(fixedDays.contains("e1p1o1 2026/03/24"), "Fixed r1 row inside the r2 trip kept: " + fixedDays);
        assertTrue(fixedDays.contains("misc_other_1 " + MISC_DAY), "Fixed misc row on the FO day kept: " + fixedDays);
        for (String date : new String[] {"2026/03/23", "2026/03/24", MISC_DAY}) {
            assertTrue(run.stdout().contains("[trips] WARN fixed task conflicts with fixed trip (worker w2, date "
                    + date), "warning for " + date + " missing; stdout:\n" + run.stdout());
        }
        // e1p1o1 03/21 and 03/22 lie before the plan (no marks there): no warning
        assertFalse(run.stdout().contains("fixed task conflicts with fixed trip (worker w2, date 2026/03/22"));
        // the output has a trip_list (derived from the final assignments; see TripExportTest)
        assertNotNull(run.output().get("trip_list"));
    }

    @Test
    void other_region_inside_trip_is_never_written() throws Exception {
        Run without = solveAndExport(prepare(tmp.resolve("without"), null));
        assertFalse(closedDaysUsed(without, OTHER_REGION_TRIP).isEmpty(),
                "scenario: without the trip w2 works r2 inside its window");
        Run with = solveAndExport(prepare(tmp.resolve("with"), OTHER_REGION_TRIP));
        assertMarksHonoured(with, OTHER_REGION_TRIP);
        assertEquals(List.of(), closedDaysUsed(with, OTHER_REGION_TRIP));
    }

    @Test
    void no_trip_list_means_no_warning_and_the_closed_days_are_used() throws Exception {
        Run run = solveAndExport(prepare(tmp.resolve("notrip"), null));
        assertFalse(run.stdout().contains("fixed task conflicts with fixed trip"));
        // sanity check of the main scenario: without the trips w2 works on days they close (else (a) proves nothing)
        assertFalse(closedDaysUsed(run, TRIPS).isEmpty(), "without the trips w2 should use some closed day");
    }

    // ------------------------------------------------------------------ (d) personal unavailable dates

    @Test
    void unavailable_on_fi_or_fo_day_changes_nothing() throws Exception {
        Run base = solveAndExport(prepare(tmp.resolve("base"), TRIPS));
        // 2026/03/30 (FI of the r1 trip) already is w2's unavailable date in the fixture
        for (String day : new String[] {"2026/03/23", "2026/03/27", "2026/04/05"}) {
            Run run = solveAndExport(prepare(tmp.resolve("u" + day.replace('/', '_')), TRIPS, day));
            assertEquals(String.valueOf(base.best().getScore()), String.valueOf(run.best().getScore()),
                    "score with w2 unavailable on " + day);
            assertEquals(ScheduleExporterTest.assignments(base.output()),
                    ScheduleExporterTest.assignments(run.output()), "assignment_list with w2 unavailable on " + day);
        }
    }

    @Test
    void unavailable_inside_trip_only_removes_that_day() throws Exception {
        Run base = solveAndExport(prepare(tmp.resolve("base"), TRIPS));
        assertTrue(workDays(base.output(), WORKER, false).stream().anyMatch(td -> INSIDE_UNAVAILABLE.equals(td[1])),
                "scenario: w2 should work on " + INSIDE_UNAVAILABLE + " without the unavailable date");

        // the same solved plan, exported against the calendar that has the unavailable date: only that day goes away
        Path dir = prepare(tmp.resolve("inside"), TRIPS, INSIDE_UNAVAILABLE);
        PlanBuilder.buildProblem(dir.resolve("EnvConfig.yaml").toString(), dir.resolve("Schedule.yaml").toString());
        ScheduleExporter.overwriteScheduleWithAssignments(base.best(), base.planStart(),
                dir.resolve("Schedule.yaml").toString(), dir.resolve("EnvConfig.yaml").toString());
        List<Map<String, Object>> actual = ScheduleExporterTest.assignments(ScheduleExporterTest.schedule(
                dir.resolve("Schedule.yaml")));
        assertEquals(withoutDay(ScheduleExporterTest.assignments(base.output()), WORKER, INSIDE_UNAVAILABLE), actual);

        // and a full solve with the unavailable date still honours the trip marks and the unavailable date
        Run run = solveAndExport(prepare(tmp.resolve("inside-solve"), TRIPS, INSIDE_UNAVAILABLE));
        assertMarksHonoured(run, TRIPS);
        for (String[] td : workDays(run.output(), WORKER, false)) assertNotEquals(INSIDE_UNAVAILABLE, td[1]);
    }

    /** Expected export when {@code day} becomes a personal off day of {@code worker}: drop it from the Flexible rows. */
    @SuppressWarnings("unchecked")
    static List<Map<String, Object>> withoutDay(List<Map<String, Object>> rows, String worker, String day) {
        List<Map<String, Object>> out = new ArrayList<>();
        for (Map<String, Object> a : rows) {
            if (ScheduleExporterTest.isFixed(a) || !worker.equals(String.valueOf(a.get("worker")))) {
                out.add(a);
                continue;
            }
            List<Map<String, Object>> kept = new ArrayList<>();
            for (Map<String, Object> item : (List<Map<String, Object>>) a.get("work_date_list")) {
                if (!day.equals(String.valueOf(item.get("date")))) kept.add(item);
            }
            if (kept.isEmpty()) continue;
            Map<String, Object> row = new LinkedHashMap<>(a);
            row.put("work_date_list", kept);
            row.put("start_date", kept.get(0).get("date"));
            row.put("end_date", kept.get(kept.size() - 1).get("date"));
            out.add(row);
        }
        return out;
    }

    // ------------------------------------------------------------------ Task 12 review question

    /**
     * The dummy fixed block (id 0) holds every pinned seat. It has planning variables (startDay, days, hours) and is
     * not pinned, so the construction heuristic initialises it; from then on the pinned module seats join the
     * region rules through the CrewSeat-BlockDecision join (before solving they do not, block 0 being uninitialised).
     */
    @Test
    void dummy_fixed_block_is_initialised_after_construction() throws Exception {
        SinglePassPlan p = Characterization.buildFrom(SMALL_FIXTURE, tmp.resolve("dummy"));
        BlockDecision before = block0(p);
        assertNull(before.startDay, "before solving the dummy block is uninitialised");
        assertNull(before.days, "before solving the dummy block is uninitialised");
        SinglePassPlan best = Characterization.solveSeeded(p, CharacterizationTest.STEP_LIMIT, SEED);
        BlockDecision after = block0(best);
        assertNotNull(after.startDay, "dummy block startDay after construction");
        assertNotNull(after.days, "dummy block days after construction");
    }

    static BlockDecision block0(SinglePassPlan p) {
        for (BlockDecision b : p.blocks) if (b.id == 0) return b;
        throw new AssertionError("no block 0");
    }
}
