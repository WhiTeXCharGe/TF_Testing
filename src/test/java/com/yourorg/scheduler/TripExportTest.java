package com.yourorg.scheduler;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import com.yourorg.scheduler.build.PlanBuilder;
import com.yourorg.scheduler.calendar.SolveContext;
import com.yourorg.scheduler.config.SolverConstants;
import com.yourorg.scheduler.export.ScheduleExporter;
import com.yourorg.scheduler.model.SinglePassPlan;
import com.yourorg.scheduler.trip.Flexibility;
import com.yourorg.scheduler.trip.Trip;
import com.yourorg.scheduler.trip.TripEntry;
import com.yourorg.scheduler.trip.TripWriter;

/**
 * Output side of the trip_list (Task 17): build the small fixture (plus extra rows / trips appended to its
 * Schedule.yaml), seeded solve, export, read the result back.
 */
class TripExportTest {

    static final Path SMALL = CharacterizationTest.SMALL_FIXTURE;
    static final int STEPS = 50;
    static final long SEED = 42L;

    /** A Fixed history row of w3 (region r2) long before the plan start 2026/03/23. */
    static final String OLD_ROW = """
              - worker: w3
                operation_task: e2p1o1
                start_date: 2026/03/05
                end_date: 2026/03/06
                work_date_list:
                - {date: 2026/03/05, hour: 10}
                - {date: 2026/03/06, hour: 10}
                plan_flexibility: Fixed
            """;

    @TempDir
    Path tmp;

    @BeforeEach
    @AfterEach
    void resetContext() {
        SolveContext.clear();
    }

    record Run(Map<String, Object> sched, Path dir, String text) {
        Path warningsFile() {
            return dir.resolve(ScheduleExporter.TRIP_WARNINGS_FILE);
        }
    }

    /** The small fixture with {@code assignmentRows} appended to assignment_list and {@code tripListYaml} after it. */
    static Path prepare(Path dir, String assignmentRows, String tripListYaml) throws Exception {
        Files.createDirectories(dir);
        String sched = Files.readString(SMALL.resolve("Schedule.yaml"), StandardCharsets.UTF_8);
        StringBuilder sb = new StringBuilder(sched.endsWith("\n") ? sched : sched + "\n");
        if (assignmentRows != null) sb.append(assignmentRows);
        if (tripListYaml != null) sb.append(tripListYaml);
        Files.writeString(dir.resolve("Schedule.yaml"), sb.toString(), StandardCharsets.UTF_8);
        Files.copy(SMALL.resolve("EnvConfig.yaml"), dir.resolve("EnvConfig.yaml"));
        return dir;
    }

    static Run solveAndExport(Path dir, boolean allowCut, int historyDays) throws Exception {
        PrintStream old = System.out;
        System.setOut(new PrintStream(new ByteArrayOutputStream(), true, StandardCharsets.UTF_8));
        PlanBuilder.BuiltProblem bp;
        try {
            bp = PlanBuilder.buildProblem(dir.resolve("EnvConfig.yaml").toString(),
                    dir.resolve("Schedule.yaml").toString());
        } finally {
            System.setOut(old);
        }
        SolveContext.current().allowCutFixedTrip = allowCut;
        SolveContext.current().tripHistoryDays = historyDays;
        SinglePassPlan best = Characterization.solveSeeded(bp.plan, STEPS, SEED);
        System.setOut(new PrintStream(new ByteArrayOutputStream(), true, StandardCharsets.UTF_8));
        try {
            ScheduleExporter.overwriteScheduleWithAssignments(best, bp.sch.planStart,
                    dir.resolve("Schedule.yaml").toString(), dir.resolve("EnvConfig.yaml").toString());
        } finally {
            System.setOut(old);
        }
        return new Run(ScheduleExporterTest.schedule(dir.resolve("Schedule.yaml")), dir,
                Files.readString(dir.resolve("Schedule.yaml"), StandardCharsets.UTF_8));
    }

    @SuppressWarnings("unchecked")
    static List<Map<String, Object>> tripList(Run run) {
        return (List<Map<String, Object>>) run.sched().get("trip_list");
    }

    /** The trips of one worker in the output trip_list (empty when the worker has none). */
    @SuppressWarnings("unchecked")
    static List<Map<String, Object>> tripsOf(Run run, String worker) {
        for (Map<String, Object> w : tripList(run)) {
            if (worker.equals(String.valueOf(w.get("worker")))) return (List<Map<String, Object>>) w.get("trips");
        }
        return List.of();
    }

    // ------------------------------------------------------------------ key order

    @Test
    void trip_list_follows_assignment_list() throws Exception {
        // no trip_list in the input
        Run plain = solveAndExport(prepare(tmp.resolve("plain"), null, null), true, -1);
        List<String> keys = new ArrayList<>(plain.sched().keySet());
        assertTrue(keys.contains("trip_list"), "the output has a trip_list although the input had none: " + keys);
        assertEquals(keys.indexOf("assignment_list") + 1, keys.indexOf("trip_list"), "keys: " + keys);
        assertEquals(SolverConstants.SOLVER_VERSION, plain.sched().get("solver_version"));
        assertEquals("solver_version", keys.get(keys.size() - 1), "solver_version is the last key: " + keys);

        // an input trip_list that stands BEFORE assignment_list moves right behind it
        Path dir = prepare(tmp.resolve("before"), null, null);
        String text = Files.readString(dir.resolve("Schedule.yaml"), StandardCharsets.UTF_8);
        String block = """
                  trip_list:
                  - worker: w2
                    trips:
                    - {start_date: '2026/04/06', end_date: '2026/04/10', region: r2, plan_flexibility: Fixed}
                """;
        Files.writeString(dir.resolve("Schedule.yaml"),
                text.replace("  assignment_list:\n", block + "  assignment_list:\n"), StandardCharsets.UTF_8);
        Run moved = solveAndExport(dir, false, -1);
        List<String> keys2 = new ArrayList<>(moved.sched().keySet());
        assertEquals(1, keys2.stream().filter("trip_list"::equals).count(), "keys: " + keys2);
        assertEquals(keys2.indexOf("assignment_list") + 1, keys2.indexOf("trip_list"), "keys: " + keys2);
        // the other keys keep their relative order
        assertTrue(keys2.indexOf("plan_range") < keys2.indexOf("workflow_task_list"));
        assertTrue(keys2.indexOf("workflow_task_list") < keys2.indexOf("assignment_list"));
        // and the file text agrees
        assertTrue(moved.text().indexOf("assignment_list:") < moved.text().indexOf("trip_list:"));
    }

    // ------------------------------------------------------------------ Fixed trips copied unchanged

    @Test
    @SuppressWarnings("unchecked")
    void fixed_trip_copied_byte_equal() throws Exception {
        String trips = """
                  trip_list:
                  - {worker: w2, start_date: '2026/04/06', end_date: '2026/04/10', region: r2, plan_flexibility: Fixed, description: site visit, contact: abc, badge: 7}
                  - worker: w1
                    trips:
                    - {note: x, plan_flexibility: Fixed, end_date: '2026/04/02', start_date: '2026/04/01'}
                """;
        Run run = solveAndExport(prepare(tmp.resolve("fixed"), null, trips), false, -1);

        List<Map<String, Object>> w2 = tripsOf(run, "w2");
        Map<String, Object> expected = new LinkedHashMap<>();
        expected.put("start_date", "2026/04/06");
        expected.put("end_date", "2026/04/10");
        expected.put("region", "r2");
        expected.put("plan_flexibility", "Fixed");
        expected.put("description", "site visit");
        expected.put("contact", "abc");
        expected.put("badge", 7);
        Map<String, Object> got = w2.stream().filter(t -> "Fixed".equals(t.get("plan_flexibility"))).findFirst()
                .orElseThrow();
        assertEquals(expected, got);
        assertEquals(new ArrayList<>(expected.keySet()), new ArrayList<>(got.keySet()), "key order of the raw trip");
        assertFalse(got.containsKey("worker"), "the flat form is written nested, without the worker key");

        // a malformed Fixed trip (no region) is copied unchanged under its worker
        Map<String, Object> bad = new LinkedHashMap<>();
        bad.put("note", "x");
        bad.put("plan_flexibility", "Fixed");
        bad.put("end_date", "2026/04/02");
        bad.put("start_date", "2026/04/01");
        List<Map<String, Object>> w1 = tripsOf(run, "w1");
        Map<String, Object> badGot = w1.stream().filter(t -> "x".equals(t.get("note"))).findFirst().orElseThrow();
        assertEquals(bad, badGot);
        assertEquals(new ArrayList<>(bad.keySet()), new ArrayList<>(badGot.keySet()));
    }

    // ------------------------------------------------------------------ allow_cut_fixed_trip

    /** w3 never works in r1 (candidate gate), so a Fixed r1 trip of w3 has nothing inside. */
    static final String W3_R1_TRIP = """
              trip_list:
              - worker: w3
                trips:
                - {start_date: '2026/04/01', end_date: '2026/04/04', region: r1, plan_flexibility: Fixed, description: empty}
            """;

    @Test
    void cut_fixed_trip_removed_and_logged() throws Exception {
        Run cut = solveAndExport(prepare(tmp.resolve("cut"), null, W3_R1_TRIP), true, -1);
        assertTrue(tripsOf(cut, "w3").stream().noneMatch(t -> "r1".equals(t.get("region"))),
                "the empty Fixed r1 trip is removed: " + tripsOf(cut, "w3"));
        assertTrue(Files.exists(cut.warningsFile()), "the cut is logged in trip_warnings.txt");
        String log = Files.readString(cut.warningsFile(), StandardCharsets.UTF_8);
        assertTrue(log.contains("cut_fixed_trip worker=w3 start=2026-04-01 end=2026-04-04 region=r1"), log);

        // allow_cut_fixed_trip=false: the trip stays, nothing is logged
        Run keep = solveAndExport(prepare(tmp.resolve("keep"), null, W3_R1_TRIP), false, -1);
        assertTrue(tripsOf(keep, "w3").stream().anyMatch(t -> "r1".equals(t.get("region"))
                && "2026/04/01".equals(t.get("start_date")) && "Fixed".equals(t.get("plan_flexibility"))
                && "empty".equals(t.get("description"))), "kept: " + tripsOf(keep, "w3"));
        assertTrue(!Files.exists(keep.warningsFile())
                || !Files.readString(keep.warningsFile(), StandardCharsets.UTF_8).contains("cut_fixed_trip"),
                "nothing is cut, so nothing is logged as cut");
    }

    // ------------------------------------------------------------------ trip_history_days

    @Test
    void history_days_trims_old_trips() throws Exception {
        // w3 works r2 on 03/05 and 03/06 (Fixed history) -> derived r2 trip 03/04..03/07; plan start is 03/23
        Run all = solveAndExport(prepare(tmp.resolve("all"), OLD_ROW, null), true, -1);
        assertTrue(tripsOf(all, "w3").stream().anyMatch(t -> "2026/03/04".equals(t.get("start_date"))
                && "2026/03/07".equals(t.get("end_date")) && "r2".equals(t.get("region"))
                && "Flexible".equals(t.get("plan_flexibility"))), "history trip present: " + tripsOf(all, "w3"));

        // 03/07 is 16 days before the plan start: N = 20 keeps it, N = 16 keeps it (end < start - N is false)
        for (int n : new int[] {20, 16}) {
            Run keep = solveAndExport(prepare(tmp.resolve("keep" + n), OLD_ROW, null), true, n);
            assertTrue(tripsOf(keep, "w3").stream().anyMatch(t -> "2026/03/04".equals(t.get("start_date"))),
                    "N=" + n + ": " + tripsOf(keep, "w3"));
        }
        // N = 15 and N = 0 drop it, the recent trips stay
        for (int n : new int[] {15, 0}) {
            Run trimmed = solveAndExport(prepare(tmp.resolve("trim" + n), OLD_ROW, null), true, n);
            assertTrue(tripsOf(trimmed, "w3").stream().noneMatch(t -> "2026/03/04".equals(t.get("start_date"))),
                    "N=" + n + ": " + tripsOf(trimmed, "w3"));
            assertFalse(tripList(trimmed).isEmpty(), "recent trips are kept");
        }
    }

    // ------------------------------------------------------------------ derived trips without an input trip_list

    @Test
    void no_trip_list_in_input_still_writes_derived_trips() throws Exception {
        Run run = solveAndExport(prepare(tmp.resolve("derived"), null, null), true, -1);
        List<Map<String, Object>> list = tripList(run);
        assertFalse(list.isEmpty(), "derived trips are written");

        // workers sorted, trips sorted, every trip Flexible with an empty description
        List<String> workers = new ArrayList<>();
        for (Map<String, Object> w : list) workers.add(String.valueOf(w.get("worker")));
        List<String> sorted = new ArrayList<>(workers);
        sorted.sort(null);
        assertEquals(sorted, workers);
        for (String worker : workers) {
            String prev = "";
            for (Map<String, Object> t : tripsOf(run, worker)) {
                assertEquals("Flexible", t.get("plan_flexibility"));
                assertEquals("", t.get("description"));
                String start = String.valueOf(t.get("start_date"));
                assertTrue(start.compareTo(String.valueOf(t.get("end_date"))) < 0, "start < end: " + t);
                assertTrue(start.compareTo(prev) >= 0, "sorted by start_date: " + tripsOf(run, worker));
                prev = start;
            }
        }

        // w1's Fixed history 03/20..03/22 (r1) gives FI on 03/19
        assertTrue(tripsOf(run, "w1").stream().anyMatch(t -> "2026/03/19".equals(t.get("start_date"))
                && "r1".equals(t.get("region"))), "w1: " + tripsOf(run, "w1"));

        // every worked day of the final assignment_list lies strictly inside a trip of its region. The 50-step plan
        // is not feasible (w1 hops regions without the transit gap), the deriver reports that as region_conflict_day /
        // tight_transit; workers named in such a warning are not checked (their trips are moved on purpose).
        String log = Files.exists(run.warningsFile()) ? Files.readString(run.warningsFile(), StandardCharsets.UTF_8) : "";
        Map<String, String> fabs = TripMarksSolverTest.taskFab(run.sched());
        int checked = 0;
        for (String worker : workers) {
            if (log.contains("worker=" + worker + " ")) continue;
            for (String[] td : TripMarksSolverTest.workDays(run.sched(), worker, null)) {
                String region = TripMarksSolverTest.regionOf(fabs, td[0]);
                if (region == null) continue;
                checked++;
                assertTrue(tripsOf(run, worker).stream().anyMatch(t -> region.equals(t.get("region"))
                                && td[1].compareTo(String.valueOf(t.get("start_date"))) > 0
                                && td[1].compareTo(String.valueOf(t.get("end_date"))) < 0),
                        worker + " " + td[0] + " " + td[1] + " not inside a " + region + " trip: "
                                + tripsOf(run, worker));
            }
        }
        assertTrue(checked > 5, "the check must have looked at real work days, saw " + checked);
    }

    // ------------------------------------------------------------------ trip_warnings.txt

    @Test
    void warnings_file_written_only_when_warnings_exist() throws Exception {
        // no warning: no file, and an old file of an earlier run is removed
        Path clean = tmp.resolve("clean");
        Files.createDirectories(clean);
        Files.writeString(clean.resolve(ScheduleExporter.TRIP_WARNINGS_FILE), "stale\n", StandardCharsets.UTF_8);
        Run quiet = exportOnly(clean, "r_other", true);
        assertFalse(Files.exists(quiet.warningsFile()), "no warning, no trip_warnings.txt");

        // a parser warning (unknown region) is written to the file next to the result
        String bad = """
                  trip_list:
                  - worker: w2
                    trips:
                    - {start_date: '2026/04/01', end_date: '2026/04/03', region: nowhere, plan_flexibility: Fixed}
                """;
        Run loud = solveAndExport(prepare(tmp.resolve("loud"), null, bad), true, -1);
        assertTrue(Files.exists(loud.warningsFile()));
        String log = Files.readString(loud.warningsFile(), StandardCharsets.UTF_8);
        assertTrue(log.contains("nowhere"), log);
        assertTrue(log.contains("trip_list worker w2"), log);
    }

    // ------------------------------------------------------------------ exporter-only scenarios (hand written files)

    static final String MISC_SCHEDULE = """
            schedule:
              plan_range:
                start_date: 2026/03/23
                end_date: 2026/04/05
              workflow_task_list: []
              misc_task_list:
              - id: misc_other_1
                name: Training
                region: %s
              assignment_list:
              - worker: w2
                operation_task: misc_other_1
                start_date: 2026/03/27
                end_date: 2026/03/27
                work_date_list:
                - {date: 2026/03/27, hour: 8}
                plan_flexibility: Fixed
            """;

    static Run exportOnly(Path dir, String miscRegion, boolean neutral) throws Exception {
        Files.createDirectories(dir);
        Files.copy(SMALL.resolve("EnvConfig.yaml"), dir.resolve("EnvConfig.yaml"));
        Files.writeString(dir.resolve("Schedule.yaml"), MISC_SCHEDULE.formatted(miscRegion), StandardCharsets.UTF_8);
        SolveContext.clear();
        if (!neutral) SolveContext.current().miscNeutralRegions = java.util.Set.of();
        PrintStream old = System.out;
        System.setOut(new PrintStream(new ByteArrayOutputStream(), true, StandardCharsets.UTF_8));
        try {
            ScheduleExporter.overwriteScheduleWithAssignments(null, LocalDate.of(2026, 3, 23),
                    dir.resolve("Schedule.yaml").toString(), dir.resolve("EnvConfig.yaml").toString());
        } finally {
            System.setOut(old);
        }
        return new Run(ScheduleExporterTest.schedule(dir.resolve("Schedule.yaml")), dir,
                Files.readString(dir.resolve("Schedule.yaml"), StandardCharsets.UTF_8));
    }

    @Test
    void trip_list_absent_and_no_region_work_writes_empty_list() throws Exception {
        // the only worked day is a misc row in the neutral region r_other: no region, no trip
        Run run = exportOnly(tmp.resolve("neutral"), "r_other", true);
        assertEquals(List.of(), run.sched().get("trip_list"));
        assertTrue(run.text().contains("trip_list: []"), run.text());
        assertFalse(Files.exists(run.warningsFile()));
    }

    @Test
    void misc_row_with_its_own_region_counts_as_a_worked_day() throws Exception {
        Run run = exportOnly(tmp.resolve("own"), "r1", true);
        List<Map<String, Object>> trips = tripsOf(run, "w2");
        assertEquals(1, trips.size(), String.valueOf(trips));
        assertEquals("2026/03/26", trips.get(0).get("start_date"));
        assertEquals("2026/03/28", trips.get(0).get("end_date"));
        assertEquals("r1", trips.get(0).get("region"));
    }

    // ------------------------------------------------------------------ TripWriter / constants

    private static Trip trip(String worker, String start, String end, String region, Flexibility f) {
        return new Trip(worker, LocalDate.parse(start), LocalDate.parse(end), region, f, null, Map.of());
    }

    @Test
    void trip_writer_sorts_workers_and_trips_and_skips_workerless_entries() {
        Map<String, Object> rawFixed = new LinkedHashMap<>();
        rawFixed.put("worker", "w2");
        rawFixed.put("start_date", "2026/05/01");
        rawFixed.put("end_date", "2026/05/09");
        rawFixed.put("region", "r1");
        rawFixed.put("plan_flexibility", "Fixed");
        Map<String, Object> rawNoWorker = new LinkedHashMap<>(rawFixed);
        rawNoWorker.remove("worker");
        List<TripEntry> fixed = List.of(
                new TripEntry(trip("w2", "2026-05-01", "2026-05-09", "r1", Flexibility.FIXED), rawFixed, true, "w2"),
                new TripEntry(null, rawNoWorker, false, null));
        List<Trip> derived = List.of(
                trip("w2", "2026-04-01", "2026-04-05", "r2", Flexibility.FLEXIBLE),
                trip("w1", "2026-04-10", "2026-04-12", "r1", Flexibility.FLEXIBLE));

        List<Map<String, Object>> out = TripWriter.toYaml(fixed, derived, -1, LocalDate.of(2026, 4, 1));
        assertEquals(2, out.size());
        assertEquals("w1", out.get(0).get("worker"));
        assertEquals("w2", out.get(1).get("worker"));
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> w2 = (List<Map<String, Object>>) out.get(1).get("trips");
        assertEquals(List.of("2026/04/01", "2026/05/01"), List.of(w2.get(0).get("start_date"), w2.get(1).get("start_date")));
        assertFalse(w2.get(1).containsKey("worker"));
        assertEquals(List.of("start_date", "end_date", "region", "plan_flexibility"), new ArrayList<>(w2.get(1).keySet()));
        assertNull(TripWriter.toYaml(null, null, -1, null).stream().findAny().orElse(null));
    }

    @Test
    void trip_writer_history_never_trims_fixed_entries() {
        Map<String, Object> raw = new LinkedHashMap<>();
        raw.put("start_date", "2020/01/01");
        raw.put("end_date", "2020/01/05");
        raw.put("region", "r1");
        raw.put("plan_flexibility", "Fixed");
        List<TripEntry> fixed = List.of(
                new TripEntry(trip("w1", "2020-01-01", "2020-01-05", "r1", Flexibility.FIXED), raw, true, "w1"));
        List<Trip> derived = List.of(trip("w1", "2020-02-01", "2020-02-05", "r1", Flexibility.FLEXIBLE));
        List<Map<String, Object>> out = TripWriter.toYaml(fixed, derived, 30, LocalDate.of(2026, 3, 23));
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> w1 = (List<Map<String, Object>>) out.get(0).get("trips");
        assertEquals(1, w1.size());
        assertEquals("Fixed", w1.get(0).get("plan_flexibility"));
        assertNotNull(w1.get(0).get("region"));
    }

    @Test
    void solver_version_constants_match_the_pom() throws Exception {
        String pom = Files.readString(Path.of("pom.xml"), StandardCharsets.UTF_8);
        assertTrue(pom.contains("<timefold.version>" + SolverConstants.TIMEFOLD_VERSION + "</timefold.version>"));
        assertTrue(pom.contains("<artifactId>timefold-scheduler</artifactId>\n  <version>"
                + SolverConstants.APP_VERSION + "</version>"));
        assertTrue(SolverConstants.SOLVER_VERSION.contains(SolverConstants.APP_VERSION));
        assertTrue(SolverConstants.SOLVER_VERSION.contains(SolverConstants.TIMEFOLD_VERSION));
    }
}
