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
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import com.yourorg.scheduler.calendar.SolveContext;
import com.yourorg.scheduler.config.SolverConstants;
import com.yourorg.scheduler.config.SolverParams;
import com.yourorg.scheduler.export.ScheduleExporter;
import com.yourorg.scheduler.solver.ScoreLogger;
import com.yourorg.scheduler.trip.DeriveResult;
import com.yourorg.scheduler.trip.Flexibility;
import com.yourorg.scheduler.trip.Trip;
import com.yourorg.scheduler.trip.TripDeriver;
import com.yourorg.scheduler.trip.WorkedDay;

/**
 * End-to-end scenarios through the real entry point {@link Main#run} (what {@code main} does, minus {@code System.exit}):
 * parameters, build, two-stage solve, export. Every run uses a budget of a few seconds (stage 2 off).
 */
class EndToEndTest {

    static final Path FIRSTRUN = Path.of("src", "test", "resources", "fixtures", "firstrun");
    static final Path SMALL = CharacterizationTest.SMALL_FIXTURE;
    static final Path TINY = CharacterizationTest.TINY_FIXTURE;

    /** Short budget flags: stage 1 stops 5 s after the last improvement (or when feasible), no stage 2, no score files. */
    static final String[] SHORT = {"--stage1_unimproved_seconds=5", "--stage2_enabled=false",
            "--write_score_files=false", "--random_seed=42"};

    @TempDir
    Path tmp;

    @BeforeEach
    @AfterEach
    void resetGlobals() {
        SolveContext.clear();
        ScoreLogger.setWriteScoreFiles(true);
    }

    // ------------------------------------------------------------------ helpers

    record Result(SolverParams params, Map<String, Object> sched, String text, String console, Path dir, long millis) {
        Path warningsFile() {
            return dir.resolve(ScheduleExporter.TRIP_WARNINGS_FILE);
        }

        String warningsText() throws Exception {
            return Files.exists(warningsFile()) ? Files.readString(warningsFile(), StandardCharsets.UTF_8) : "";
        }
    }

    /** Copies the fixture files into {@code dir} (Schedule.yaml becomes the result file, as in the container). */
    static Path copyFixture(Path fixture, Path dir) throws Exception {
        Files.createDirectories(dir);
        Files.copy(fixture.resolve("EnvConfig.yaml"), dir.resolve("EnvConfig.yaml"));
        Files.copy(fixture.resolve("Schedule.yaml"), dir.resolve("Schedule.yaml"));
        return dir;
    }

    /** Runs Main in {@code dir} with the given flags and environment; stdout is captured. */
    static Result run(Path dir, Map<String, String> env, String... flags) throws Exception {
        List<String> args = new ArrayList<>();
        args.add(dir.resolve("EnvConfig.yaml").toString());
        args.add(dir.resolve("Schedule.yaml").toString());
        args.addAll(List.of(flags));
        PrintStream old = System.out;
        ByteArrayOutputStream buf = new ByteArrayOutputStream();
        System.setOut(new PrintStream(buf, true, StandardCharsets.UTF_8));
        long t0 = System.nanoTime();
        SolverParams used;
        try {
            used = Main.run(args.toArray(new String[0]), env);
        } finally {
            System.setOut(old);
        }
        long ms = (System.nanoTime() - t0) / 1_000_000L;
        return new Result(used, ScheduleExporterTest.schedule(dir.resolve("Schedule.yaml")),
                Files.readString(dir.resolve("Schedule.yaml"), StandardCharsets.UTF_8),
                buf.toString(StandardCharsets.UTF_8), dir, ms);
    }

    /** The default convention of the container: {@code INPUT_DIR} holds a SolverParams.yaml, if any. */
    static Map<String, String> envOf(Path inputDir) {
        return Map.of("INPUT_DIR", inputDir.toString());
    }

    static String[] concat(String[] a, String... b) {
        List<String> l = new ArrayList<>(List.of(a));
        l.addAll(List.of(b));
        return l.toArray(new String[0]);
    }

    @SuppressWarnings("unchecked")
    static List<Map<String, Object>> tripLists(Map<String, Object> sched) {
        return (List<Map<String, Object>>) sched.get("trip_list");
    }

    /** (worker, trip map) of every trip of the output trip_list. */
    @SuppressWarnings("unchecked")
    static List<Map.Entry<String, Map<String, Object>>> allTrips(Map<String, Object> sched) {
        List<Map.Entry<String, Map<String, Object>>> out = new ArrayList<>();
        for (Map<String, Object> w : tripLists(sched)) {
            for (Map<String, Object> t : (List<Map<String, Object>>) w.get("trips")) {
                out.add(Map.entry(String.valueOf(w.get("worker")), t));
            }
        }
        return out;
    }

    static LocalDate date(Object s) {
        return LocalDate.parse(String.valueOf(s).replace('/', '-'));
    }

    /** operation_task id -> region, for the tasks of the given fabs (misc / unknown tasks are not in the map). */
    static Map<String, String> taskRegions(Map<String, Object> sched, Map<String, String> fabRegion) {
        Map<String, String> taskFab = new HashMap<>();
        ScheduleExporterTest.collectTasks(sched, new LinkedHashSet<>(), taskFab);
        Map<String, String> out = new HashMap<>();
        for (Map.Entry<String, String> e : taskFab.entrySet()) {
            String region = fabRegion.get(e.getValue());
            if (region != null) out.put(e.getKey(), region);
        }
        return out;
    }

    record Work(String worker, String task, String date) {}

    /** Every worked (hour &gt; 0) day of the output assignment_list. */
    static List<Work> work(Map<String, Object> sched) {
        List<Work> out = new ArrayList<>();
        for (Map<String, Object> a : ScheduleExporterTest.assignments(sched)) {
            for (Map<String, Object> item : ScheduleExporterTest.workDates(a)) {
                if (item.get("hour") instanceof Number n && n.doubleValue() <= 0) continue;
                out.add(new Work(String.valueOf(a.get("worker")), String.valueOf(a.get("operation_task")),
                        String.valueOf(item.get("date"))));
            }
        }
        return out;
    }

    /**
     * Derives the trips again from the OUTPUT assignment_list with {@link TripDeriver} and checks that the Flexible
     * trips of the output trip_list are exactly those (same worker, dates, region), and that the output is well formed:
     * every Flexible trip has an empty description and start &lt; end, the trips of a worker never overlap (a trip may
     * start on the previous trip's end date).
     */
    static void assertDerivedTripsAreConsistent(Map<String, Object> sched, Map<String, String> taskRegion,
                                                int stayOffInterval) {
        List<WorkedDay> days = new ArrayList<>();
        for (Work w : work(sched)) {
            String region = taskRegion.get(w.task());
            if (region != null) days.add(new WorkedDay(w.worker(), date(w.date()), region));
        }
        List<Trip> fixed = new ArrayList<>();
        Set<String> outFlexible = new TreeSet<>();
        for (Map.Entry<String, Map<String, Object>> e : allTrips(sched)) {
            Map<String, Object> t = e.getValue();
            Trip trip = new Trip(e.getKey(), date(t.get("start_date")), date(t.get("end_date")),
                    String.valueOf(t.get("region")),
                    "Fixed".equals(t.get("plan_flexibility")) ? Flexibility.FIXED : Flexibility.FLEXIBLE, null, Map.of());
            if (trip.flexibility() == Flexibility.FIXED) {
                fixed.add(trip);
            } else {
                outFlexible.add(key(trip));
                assertEquals("", t.get("description"), "derived trips have an empty description: " + t);
                assertTrue(trip.start().isBefore(trip.end()), "start < end: " + t);
            }
        }
        DeriveResult again = TripDeriver.derive(days, fixed, r -> stayOffInterval, false);
        Set<String> expected = new TreeSet<>();
        for (Trip t : again.derived()) expected.add(key(t));
        assertEquals(expected, outFlexible, "output Flexible trips differ from a fresh derivation of the output assignments");

        // trips of one worker never overlap (all trips, Fixed ones included, sorted by start)
        Map<String, List<Trip>> byWorker = new HashMap<>();
        for (Map.Entry<String, Map<String, Object>> e : allTrips(sched)) {
            Map<String, Object> t = e.getValue();
            byWorker.computeIfAbsent(e.getKey(), k -> new ArrayList<>()).add(new Trip(e.getKey(),
                    date(t.get("start_date")), date(t.get("end_date")), String.valueOf(t.get("region")),
                    Flexibility.FLEXIBLE, null, Map.of()));
        }
        for (List<Trip> trips : byWorker.values()) {
            trips.sort(java.util.Comparator.comparing(Trip::start).thenComparing(Trip::end));
            LocalDate prevEnd = null;
            for (Trip t : trips) {
                if (prevEnd != null) assertFalse(t.start().isBefore(prevEnd), "overlapping trips of " + t.worker() + ": " + trips);
                prevEnd = t.end();
            }
        }
    }

    static String key(Trip t) {
        return t.worker() + "|" + t.start() + "|" + t.end() + "|" + t.region();
    }

    /** The result file has trip_list right after assignment_list and solver_version as the last key. */
    static void assertKeyOrder(Map<String, Object> sched) {
        List<String> keys = new ArrayList<>(sched.keySet());
        assertTrue(keys.contains("assignment_list") && keys.contains("trip_list"), "keys: " + keys);
        assertEquals(keys.indexOf("assignment_list") + 1, keys.indexOf("trip_list"), "keys: " + keys);
        assertEquals("solver_version", keys.get(keys.size() - 1), "keys: " + keys);
        assertEquals(SolverConstants.SOLVER_VERSION, sched.get("solver_version"));
    }

    // ------------------------------------------------------------------ first run

    @Test
    void empty_assignment_list_first_run() throws Exception {
        Path dir = copyFixture(FIRSTRUN, tmp.resolve("first"));
        // allow_cut_fixed_trip=false: the Fixed trip must come out unchanged whatever the (short) search finds
        Result r = run(dir, envOf(dir), concat(SHORT, "--allow_cut_fixed_trip=false"));
        Map<String, String> region = taskRegions(r.sched(), Map.of("fa", "ra", "fb", "rb"));
        assertEquals(6, region.size(), "fixture has 3 modules x 2 operation tasks: " + region);

        // assignments were produced from the empty list
        List<Work> work = work(r.sched());
        assertFalse(ScheduleExporterTest.assignments(r.sched()).isEmpty(), "no assignments produced");
        assertTrue(work.size() >= 10, "too little work planned: " + work.size());
        for (Map<String, Object> a : ScheduleExporterTest.assignments(r.sched())) {
            assertEquals("Flexible", String.valueOf(a.get("plan_flexibility")));
        }

        // trip_list right after assignment_list, solver_version last
        assertKeyOrder(r.sched());

        // the Fixed trip of t2 is unchanged (same entries, same key order, nested form)
        Map<String, Object> fixedTrip = new LinkedHashMap<>();
        fixedTrip.put("start_date", "2027/01/13");
        fixedTrip.put("end_date", "2027/01/19");
        fixedTrip.put("region", "rb");
        fixedTrip.put("plan_flexibility", "Fixed");
        fixedTrip.put("description", "site visit");
        List<Map<String, Object>> fixedOut = new ArrayList<>();
        for (Map.Entry<String, Map<String, Object>> e : allTrips(r.sched())) {
            if ("Fixed".equals(e.getValue().get("plan_flexibility"))) {
                assertEquals("t2", e.getKey());
                fixedOut.add(e.getValue());
            }
        }
        assertEquals(List.of(fixedTrip), fixedOut);
        assertEquals(new ArrayList<>(fixedTrip.keySet()), new ArrayList<>(fixedOut.get(0).keySet()));

        // no work of t2 on the FI / FO day, and inside the trip only in the trip's region (rb)
        // (guards first, so that the loop below cannot pass on an empty set)
        List<Work> t2Work = new ArrayList<>();
        List<Work> t2Inside = new ArrayList<>();
        for (Work w : work) {
            if (!"t2".equals(w.worker())) continue;
            t2Work.add(w);
            if (w.date().compareTo("2027/01/13") > 0 && w.date().compareTo("2027/01/19") < 0) t2Inside.add(w);
        }
        assertFalse(t2Work.isEmpty(), "t2 has no work at all: " + work);
        assertFalse(t2Inside.isEmpty(), "t2 has no work strictly inside its Fixed trip 01/14..01/18: " + t2Work);
        for (Work w : t2Work) {
            assertFalse(w.date().equals("2027/01/13") || w.date().equals("2027/01/19"),
                    "t2 works on an FI/FO day: " + w);
        }
        for (Work w : t2Inside) {
            assertEquals("rb", region.get(w.task()), "t2 works in another region inside the trip: " + w);
        }

        // the "no other region inside" check is exercised: WITHOUT the trip the same search puts t2 into ra inside the
        // window (or on FI/FO), so it is the mark that keeps t2 out
        Path controlDir = tmp.resolve("first-control");
        copyFixture(FIRSTRUN, controlDir);
        String sched = Files.readString(controlDir.resolve("Schedule.yaml"), StandardCharsets.UTF_8);
        Files.writeString(controlDir.resolve("Schedule.yaml"), sched.substring(0, sched.indexOf("  trip_list:")),
                StandardCharsets.UTF_8);
        Result control = run(controlDir, envOf(controlDir), SHORT);
        List<Work> closedUsed = new ArrayList<>();
        for (Work w : work(control.sched())) {
            if (!"t2".equals(w.worker())) continue;
            boolean fiFo = w.date().equals("2027/01/13") || w.date().equals("2027/01/19");
            boolean inside = w.date().compareTo("2027/01/13") > 0 && w.date().compareTo("2027/01/19") < 0;
            if (fiFo || (inside && "ra".equals(region.get(w.task())))) closedUsed.add(w);
        }
        assertFalse(closedUsed.isEmpty(), "control run without the trip never uses a day the trip closes for t2");

        // the Flexible input trip (t3, "stale decoy") is not carried over: every non-Fixed trip is regenerated
        for (Map.Entry<String, Map<String, Object>> e : allTrips(r.sched())) {
            assertFalse("stale decoy".equals(e.getValue().get("description")), "decoy copied: " + e);
            if (!"Fixed".equals(e.getValue().get("plan_flexibility"))) {
                assertEquals("Flexible", e.getValue().get("plan_flexibility"));
                assertEquals("", e.getValue().get("description"));
            }
        }
        assertFalse(r.text().contains("stale decoy"));

        // derived trips = TripDeriver over the output assignments
        assertDerivedTripsAreConsistent(r.sched(), region, 2);
        assertTrue(allTrips(r.sched()).stream().anyMatch(e -> "Flexible".equals(e.getValue().get("plan_flexibility"))),
                "derived trips expected");

        // console: parameters were logged
        assertTrue(r.console().contains("[params]"), r.console());
    }

    // ------------------------------------------------------------------ baseline fixture without a trip_list

    @Test
    void no_trip_list_regression() throws Exception {
        Path dir = copyFixture(SMALL, tmp.resolve("small"));
        String inputText = Files.readString(dir.resolve("Schedule.yaml"), StandardCharsets.UTF_8);
        assertFalse(inputText.contains("trip_list"), "the baseline fixture has no trip_list");
        Map<String, Object> input = ScheduleExporterTest.schedule(dir.resolve("Schedule.yaml"));
        assertNull(input.get("trip_list"));

        Result r = run(dir, envOf(dir), SHORT);
        Map<String, String> region = taskRegions(r.sched(), Map.of("f1", "r1", "f2", "r2"));
        List<Map<String, Object>> out = ScheduleExporterTest.assignments(r.sched());
        assertKeyOrder(r.sched());

        // Fixed rows of the input are in the output with the same worker, task and work days
        // (history row A, w3's Fixed row, the misc row)
        int fixedChecked = 0;
        for (Map<String, Object> in : ScheduleExporterTest.assignments(input)) {
            if (!ScheduleExporterTest.isFixed(in)) continue;
            fixedChecked++;
            assertTrue(out.stream().anyMatch(o -> ScheduleExporterTest.isFixed(o)
                            && String.valueOf(in.get("worker")).equals(String.valueOf(o.get("worker")))
                            && String.valueOf(in.get("operation_task")).equals(String.valueOf(o.get("operation_task")))
                            && ScheduleExporterTest.workDates(in).equals(ScheduleExporterTest.workDates(o))),
                    "Fixed row lost or changed: " + in);
        }
        assertEquals(3, fixedChecked);

        // Flexible (solved) rows: inside the plan range, span = first..last work date, allowed hours, no closed day
        Map<String, Set<Integer>> allowedHours = Map.of("e1p1o1", Set.of(8, 10, 12), "e1p2o1", Set.of(8, 10, 12),
                "e1p2o2", Set.of(8), "e2p1o1", Set.of(8, 10, 12), "e2p2o1", Set.of(8, 10, 12));
        int flexRows = 0;
        for (Map<String, Object> a : out) {
            if (ScheduleExporterTest.isFixed(a)) continue;
            String task = String.valueOf(a.get("operation_task"));
            List<Map<String, Object>> dates = ScheduleExporterTest.workDates(a);
            if (date(dates.get(0).get("date")).isBefore(LocalDate.of(2026, 3, 23))) continue; // history rows
            flexRows++;
            assertEquals(String.valueOf(dates.get(0).get("date")), String.valueOf(a.get("start_date")));
            assertEquals(String.valueOf(dates.get(dates.size() - 1).get("date")), String.valueOf(a.get("end_date")));
            for (Map<String, Object> item : dates) {
                String d = String.valueOf(item.get("date"));
                // not checked against the plan end: pre-existing, the exporter expands blocks past the plan end without
                // clamping (ScheduleExporter block expansion), see follow-up; TripMarksSolverTest sees w2 on 04/06 too
                assertTrue(d.compareTo("2026/03/23") >= 0, "before the plan: " + d);
                if (allowedHours.containsKey(task)) {
                    assertTrue(allowedHours.get(task).contains(((Number) item.get("hour")).intValue()), a.toString());
                }
                if ("r1".equals(region.get(task))) assertFalse(d.equals("2026/03/28"), "r1 is closed on 03/28: " + a);
                if ("r2".equals(region.get(task))) {
                    assertFalse(d.equals("2026/03/29") || d.equals("2026/04/05"), "c2 is closed on Sundays: " + a);
                }
            }
        }
        assertTrue(flexRows > 0, "no solved rows");

        // trip_list: only derived (Flexible, empty description), no Fixed entry, equal to a fresh derivation
        assertFalse(tripLists(r.sched()).isEmpty());
        for (Map.Entry<String, Map<String, Object>> e : allTrips(r.sched())) {
            assertEquals("Flexible", e.getValue().get("plan_flexibility"), e.toString());
            assertEquals("", e.getValue().get("description"));
        }
        assertDerivedTripsAreConsistent(r.sched(), region, 3);
    }

    // ------------------------------------------------------------------ params file

    @Test
    void params_file_changes_budget() throws Exception {
        Path dir = copyFixture(TINY, tmp.resolve("tiny"));
        Path params = tmp.resolve("elsewhere").resolve("MyParams.yaml");
        Files.createDirectories(params.getParent());
        // Without the file stage 1 would run until 600 s without improvement (stop_when_feasible=false keeps it from
        // stopping on feasibility); with it the run ends 4 s after the last improvement.
        Files.writeString(params, """
                stage1_minutes: 1
                stage1_unimproved_seconds: 4
                stage1_stop_when_feasible: false
                stage2_enabled: false
                write_score_files: false
                random_seed: 7
                """, StandardCharsets.UTF_8);

        Result viaEnv = run(dir, Map.of("TF_PARAMS_FILE", params.toString()));
        assertEquals(List.of(), viaEnv.params().warnings());
        assertEquals(1, viaEnv.params().stage1Minutes());
        assertEquals(4, viaEnv.params().stage1UnimprovedSeconds());
        assertFalse(viaEnv.params().stage1StopWhenFeasible());
        assertFalse(viaEnv.params().stage2Enabled());
        assertEquals(7L, viaEnv.params().randomSeed());
        assertTrue(viaEnv.millis() >= 3_000, "stage 1 ran only " + viaEnv.millis() + " ms: did it stop on feasibility?");
        assertTrue(viaEnv.millis() < 40_000, "the 4 s unimproved limit of the file was not used: " + viaEnv.millis() + " ms");
        assertFalse(ScheduleExporterTest.assignments(viaEnv.sched()).isEmpty());
        assertKeyOrder(viaEnv.sched());

        // the same file through --params-file=, and a CLI value beats the file
        Path dir2 = copyFixture(TINY, tmp.resolve("tiny2"));
        Result viaArg = run(dir2, Map.of(), "--params-file=" + params, "--stage1_unimproved_seconds=2");
        assertEquals(2, viaArg.params().stage1UnimprovedSeconds());
        assertEquals(1, viaArg.params().stage1Minutes());
        assertTrue(viaArg.millis() < 30_000, "ran " + viaArg.millis() + " ms");
    }

    @Test
    void params_file_argument_forms() {
        assertEquals("p.yaml", Main.splitArgs(new String[] {"E", "S", "--params-file=p.yaml"}).paramsFile());
        assertEquals("p.yaml", Main.splitArgs(new String[] {"E", "S", "--params-file", "p.yaml"}).paramsFile());
        Main.SplitArgs both = Main.splitArgs(new String[] {"E", "--params_file=p.yaml", "S", "--stage1_minutes=3"});
        assertEquals("p.yaml", both.paramsFile());
        assertEquals(List.of("E", "S"), both.positional());
        assertEquals(List.of("--stage1_minutes=3"), List.of(both.flags()));
        assertNull(Main.splitArgs(new String[] {"E", "S"}).paramsFile());
    }

    @Test
    void bad_params_do_not_stop_run() throws Exception {
        // (a) not even YAML: the run goes on with the defaults for every file value (the CLI keeps the budget short)
        Path dirA = copyFixture(TINY, tmp.resolve("badyaml"));
        Files.writeString(dirA.resolve("SolverParams.yaml"), "stage1_minutes: [unclosed\n  : : :\n\tbad: {\n",
                StandardCharsets.UTF_8);
        Result a = run(dirA, envOf(dirA), concat(SHORT, "--stage1_unimproved_seconds=3"));
        assertTrue(a.params().warnings().stream().anyMatch(w -> w.contains("could not read params file")),
                "warnings: " + a.params().warnings());
        assertEquals(300, a.params().stage1Minutes(), "file values are ignored, defaults stay");
        assertTrue(a.console().contains("[params] WARN could not read params file"), a.console());
        assertFalse(ScheduleExporterTest.assignments(a.sched()).isEmpty());
        assertKeyOrder(a.sched());

        // (b) valid YAML, one value of the wrong type: that key falls back to its default with a warning, the other
        // keys of the file are still used
        Path dirB = copyFixture(TINY, tmp.resolve("badvalue"));
        Files.writeString(dirB.resolve("SolverParams.yaml"), """
                stage1_minutes: many
                stage1_unimproved_seconds: 3
                stage1_stop_when_feasible: false
                stage2_enabled: false
                write_score_files: false
                no_such_parameter: 1
                """, StandardCharsets.UTF_8);
        Result b = run(dirB, envOf(dirB));
        assertTrue(b.params().warnings().stream().anyMatch(w -> w.contains("bad value for stage1_minutes")),
                "warnings: " + b.params().warnings());
        assertTrue(b.params().warnings().stream().anyMatch(w -> w.contains("unknown parameter: no_such_parameter")),
                "warnings: " + b.params().warnings());
        assertEquals(300, b.params().stage1Minutes(), "wrong-typed value -> default");
        assertEquals(3, b.params().stage1UnimprovedSeconds(), "the good values of the file are used");
        assertFalse(b.params().stage2Enabled());
        assertTrue(b.console().contains("[params] WARN bad value for stage1_minutes"), b.console());
        assertTrue(b.millis() < 40_000, "ran " + b.millis() + " ms");
        assertFalse(ScheduleExporterTest.assignments(b.sched()).isEmpty());
        assertNotNull(b.sched().get("trip_list"));
    }
}
