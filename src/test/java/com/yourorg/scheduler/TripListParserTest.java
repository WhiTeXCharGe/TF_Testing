package com.yourorg.scheduler;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.time.LocalDate;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.yaml.snakeyaml.LoaderOptions;
import org.yaml.snakeyaml.Yaml;
import org.yaml.snakeyaml.constructor.SafeConstructor;

import com.yourorg.scheduler.build.PlanBuilder;
import com.yourorg.scheduler.calendar.SolveContext;
import com.yourorg.scheduler.config.InvalidInputDataException;
import com.yourorg.scheduler.io.YamlFiles;
import com.yourorg.scheduler.trip.Flexibility;
import com.yourorg.scheduler.trip.TripContext;
import com.yourorg.scheduler.trip.TripEntry;
import com.yourorg.scheduler.trip.TripListParser;
import com.yourorg.scheduler.trip.TripParseResult;

/** trip_list input model, parser and validator (design section 3, "Reading input safely"). Validation only. */
class TripListParserTest {

    static final TripContext CTX = new TripContext(
            Set.of("w1", "w2"), Set.of("r1", "r2"), LocalDate.of(2026, 8, 1), LocalDate.of(2026, 12, 31));

    static final Path SMALL = Path.of("src", "test", "resources", "fixtures", "small");

    @TempDir
    Path tmp;

    @BeforeEach
    @AfterEach
    void resetContext() {
        SolveContext.clear();
    }

    /** Loads yaml the way the production loader does (SafeConstructor: unquoted 2026-09-01 becomes a java.util.Date). */
    static Object yaml(String text) {
        return new Yaml(new SafeConstructor(new LoaderOptions())).load(text);
    }

    static TripParseResult parse(String yamlText) {
        return TripListParser.parse(yaml(yamlText), CTX, false);
    }

    static final String W1_FIXED_THEN_FLEXIBLE = """
            - worker: w1
              trips:
              - {start_date: 2026/09/01, end_date: 2026/09/10, region: r1, plan_flexibility: Fixed}
              - {start_date: 2026/09/10, end_date: 2026/09/15, region: r2, plan_flexibility: Flexible}
            """;

    // ------------------------------------------------------------------ empty / shape

    @Test
    void missing_list_is_empty() {
        Object root = yaml("plan_range: {start_date: 2026/09/01}");
        TripParseResult r = TripListParser.parse(((Map<?, ?>) root).get("trip_list"), CTX, false);
        assertTrue(r.entries().isEmpty());
        assertTrue(r.warnings().isEmpty());
    }

    @Test
    void null_list_is_empty() {
        TripParseResult r = TripListParser.parse(null, CTX, false);
        assertTrue(r.entries().isEmpty());
        assertTrue(r.warnings().isEmpty());
        Object root = yaml("trip_list:\n");
        r = TripListParser.parse(((Map<?, ?>) root).get("trip_list"), CTX, false);
        assertTrue(r.entries().isEmpty());
        assertTrue(r.warnings().isEmpty());
    }

    @Test
    void empty_list_is_empty() {
        TripParseResult r = parse("[]");
        assertTrue(r.entries().isEmpty());
        assertTrue(r.warnings().isEmpty());
    }

    @Test
    void non_list_value_warns() {
        TripParseResult r = parse("not a list");
        assertTrue(r.entries().isEmpty());
        assertEquals(1, r.warnings().size());
        assertTrue(r.warnings().get(0).contains("trip_list"), r.warnings().get(0));

        r = parse("{worker: w1}");
        assertTrue(r.entries().isEmpty());
        assertEquals(1, r.warnings().size());
    }

    // ------------------------------------------------------------------ the happy path

    @Test
    void flexible_trip_is_listed_not_enforced() {
        TripParseResult r = parse(W1_FIXED_THEN_FLEXIBLE);
        assertEquals(2, r.entries().size());
        assertTrue(r.warnings().isEmpty(), r.warnings().toString());
        TripEntry fixed = r.entries().get(0);
        TripEntry flexible = r.entries().get(1);
        assertTrue(fixed.enforced());
        assertFalse(flexible.enforced());
        assertEquals(Flexibility.FIXED, fixed.trip().flexibility());
        assertEquals(Flexibility.FLEXIBLE, flexible.trip().flexibility());
        assertEquals("w1", fixed.trip().worker());
        assertEquals(LocalDate.of(2026, 9, 1), fixed.trip().start());
        assertEquals(LocalDate.of(2026, 9, 10), fixed.trip().end());
        assertEquals("r1", fixed.trip().region());
        assertEquals("r2", flexible.trip().region());
        assertNull(fixed.trip().description());
    }

    @Test
    void flat_form_accepted() {
        TripParseResult r = parse("""
                - {worker: w1, start_date: 2026/09/01, end_date: 2026/09/10, region: r1, plan_flexibility: Fixed}
                - {worker: w2, start_date: 2026/09/03, end_date: 2026/09/04, region: r2, plan_flexibility: Fixed,
                   description: visit}
                """);
        assertTrue(r.warnings().isEmpty(), r.warnings().toString());
        assertEquals(2, r.entries().size());
        assertEquals("w1", r.entries().get(0).trip().worker());
        assertEquals("w2", r.entries().get(1).trip().worker());
        assertEquals("visit", r.entries().get(1).trip().description());
        assertTrue(r.entries().get(0).enforced());
        assertTrue(r.entries().get(1).enforced());
    }

    @Test
    void alias_flexibility_accepted() {
        TripParseResult r = parse("""
                - worker: w1
                  trips:
                  - {start_date: 2026/09/01, end_date: 2026/09/10, region: r1, flexibility: fixed}
                  - {start_date: 2026/09/20, end_date: 2026/09/25, region: r1, flexibility: FLEXIBLE}
                """);
        assertTrue(r.warnings().isEmpty(), r.warnings().toString());
        assertEquals(2, r.entries().size());
        assertTrue(r.entries().get(0).enforced());
        assertEquals(Flexibility.FIXED, r.entries().get(0).trip().flexibility());
        assertFalse(r.entries().get(1).enforced());
        assertEquals(Flexibility.FLEXIBLE, r.entries().get(1).trip().flexibility());
    }

    @Test
    void dash_dates_accepted() {
        TripParseResult r = parse("""
                - worker: w1
                  trips:
                  - {start_date: '2026-09-01', end_date: '2026-09-10', region: r1, plan_flexibility: Fixed}
                """);
        assertTrue(r.warnings().isEmpty(), r.warnings().toString());
        assertEquals(LocalDate.of(2026, 9, 1), r.entries().get(0).trip().start());
        assertEquals(LocalDate.of(2026, 9, 10), r.entries().get(0).trip().end());
    }

    @Test
    void yaml_date_objects_accepted() {
        // unquoted 2026-09-01 is a java.util.Date for SnakeYAML
        Object loaded = yaml("""
                - worker: w1
                  trips:
                  - {start_date: 2026-09-01, end_date: 2026-09-10, region: r1, plan_flexibility: Fixed}
                """);
        Object start = ((Map<?, ?>) ((List<?>) ((Map<?, ?>) ((List<?>) loaded).get(0)).get("trips")).get(0))
                .get("start_date");
        assertTrue(start instanceof java.util.Date, "precondition: unquoted date is a Date but was " + start.getClass());
        TripParseResult r = TripListParser.parse(loaded, CTX, false);
        assertTrue(r.warnings().isEmpty(), r.warnings().toString());
        assertEquals(LocalDate.of(2026, 9, 1), r.entries().get(0).trip().start());
        assertEquals(LocalDate.of(2026, 9, 10), r.entries().get(0).trip().end());

        // java.time objects are accepted as well
        Map<String, Object> trip = new LinkedHashMap<>();
        trip.put("worker", "w2");
        trip.put("start_date", LocalDate.of(2026, 10, 1));
        trip.put("end_date", LocalDate.of(2026, 10, 5));
        trip.put("region", "r2");
        trip.put("plan_flexibility", "Fixed");
        r = TripListParser.parse(List.of(trip), CTX, false);
        assertTrue(r.warnings().isEmpty(), r.warnings().toString());
        assertEquals(LocalDate.of(2026, 10, 1), r.entries().get(0).trip().start());
    }

    // ------------------------------------------------------------------ the validation table

    @Test
    void unknown_worker_skipped_with_warning() {
        TripParseResult r = parse("""
                - worker: ghost
                  trips:
                  - {start_date: 2026/09/01, end_date: 2026/09/10, region: r1, plan_flexibility: Fixed}
                  - {start_date: 2026/09/20, end_date: 2026/09/25, region: r1, plan_flexibility: Flexible}
                """);
        assertEquals(2, r.warnings().size(), r.warnings().toString());
        assertTrue(r.warnings().get(0).contains("ghost"), r.warnings().get(0));
        assertTrue(r.warnings().get(0).contains("2026"), r.warnings().get(0));
        // Fixed one is kept (to be copied unchanged) but not enforced; the Flexible one is dropped
        assertEquals(1, r.entries().size());
        assertFalse(r.entries().get(0).enforced());
        assertEquals(Flexibility.FIXED, r.entries().get(0).trip().flexibility());
        assertEquals("ghost", r.entries().get(0).trip().worker());
    }

    @Test
    void missing_field_skipped_but_fixed_copied() {
        TripParseResult r = parse("""
                - worker: w1
                  trips:
                  - {end_date: 2026/09/10, region: r1, plan_flexibility: Fixed, description: no start}
                  - {start_date: 2026/09/01, end_date: not-a-date, region: r1, plan_flexibility: Fixed}
                  - {start_date: 2026/09/01, end_date: 2026/09/10, plan_flexibility: Fixed}
                  - {start_date: 2026/09/01, region: r1, plan_flexibility: Flexible}
                """);
        assertEquals(4, r.warnings().size(), r.warnings().toString());
        // three Fixed entries kept, the broken Flexible one dropped; none enforced
        assertEquals(3, r.entries().size());
        for (TripEntry e : r.entries()) {
            assertFalse(e.enforced());
            assertNotNull(e.raw());
        }
        assertNull(r.entries().get(0).trip());
        assertEquals("no start", r.entries().get(0).raw().get("description"));
        assertEquals("not-a-date", r.entries().get(1).raw().get("end_date"));
        assertTrue(r.warnings().get(0).contains("w1"), r.warnings().get(0));
        assertTrue(r.warnings().get(0).contains("start_date"), r.warnings().get(0));
    }

    @Test
    void unknown_region_not_enforced() {
        TripParseResult r = parse("""
                - worker: w1
                  trips:
                  - {start_date: 2026/09/01, end_date: 2026/09/10, region: nowhere, plan_flexibility: Fixed}
                """);
        assertEquals(1, r.warnings().size());
        assertTrue(r.warnings().get(0).contains("nowhere"), r.warnings().get(0));
        assertEquals(1, r.entries().size());
        assertFalse(r.entries().get(0).enforced());
        assertEquals("nowhere", r.entries().get(0).trip().region());
    }

    @Test
    void start_after_end_skipped() {
        TripParseResult r = parse("""
                - worker: w1
                  trips:
                  - {start_date: 2026/09/10, end_date: 2026/09/01, region: r1, plan_flexibility: Fixed}
                  - {start_date: 2026/09/10, end_date: 2026/09/01, region: r1, plan_flexibility: Flexible}
                """);
        assertEquals(2, r.warnings().size(), r.warnings().toString());
        assertEquals(1, r.entries().size());
        assertFalse(r.entries().get(0).enforced());
        assertEquals(Flexibility.FIXED, r.entries().get(0).trip().flexibility());
    }

    @Test
    void start_equals_end_in_one_trip_skipped() {
        TripParseResult r = parse("""
                - worker: w1
                  trips:
                  - {start_date: 2026/09/05, end_date: 2026/09/05, region: r1, plan_flexibility: Fixed}
                """);
        assertEquals(1, r.warnings().size());
        assertEquals(1, r.entries().size());
        assertFalse(r.entries().get(0).enforced());
    }

    @Test
    void touching_trips_allowed() {
        TripParseResult r = parse("""
                - worker: w1
                  trips:
                  - {start_date: 2026/09/01, end_date: 2026/09/10, region: r1, plan_flexibility: Fixed}
                  - {start_date: 2026/09/10, end_date: 2026/09/15, region: r2, plan_flexibility: Fixed}
                """);
        assertTrue(r.warnings().isEmpty(), r.warnings().toString());
        assertEquals(2, r.entries().size());
        assertTrue(r.entries().get(0).enforced());
        assertTrue(r.entries().get(1).enforced());
    }

    @Test
    void overlapping_fixed_second_not_enforced() {
        // listed out of order on purpose: the EARLIER one (by start) stays enforced
        TripParseResult r = parse("""
                - worker: w1
                  trips:
                  - {start_date: 2026/09/08, end_date: 2026/09/20, region: r2, plan_flexibility: Fixed}
                  - {start_date: 2026/09/01, end_date: 2026/09/10, region: r1, plan_flexibility: Fixed}
                  - {start_date: 2026/09/25, end_date: 2026/09/28, region: r1, plan_flexibility: Fixed}
                - worker: w2
                  trips:
                  - {start_date: 2026/09/01, end_date: 2026/09/10, region: r1, plan_flexibility: Fixed}
                """);
        assertEquals(1, r.warnings().size(), r.warnings().toString());
        assertTrue(r.warnings().get(0).contains("w1"), r.warnings().get(0));
        assertEquals(4, r.entries().size());
        // input order is kept in the result
        assertFalse(r.entries().get(0).enforced(), "later (by start) overlapping trip");
        assertTrue(r.entries().get(1).enforced(), "earlier trip");
        assertTrue(r.entries().get(2).enforced(), "after the overlap, no conflict with the enforced one");
        assertTrue(r.entries().get(3).enforced(), "another worker is independent");
    }

    @Test
    void unknown_flexibility_treated_flexible() {
        TripParseResult r = parse("""
                - worker: w1
                  trips:
                  - {start_date: 2026/09/01, end_date: 2026/09/10, region: r1, plan_flexibility: Maybe}
                """);
        assertEquals(1, r.warnings().size());
        assertTrue(r.warnings().get(0).contains("Maybe"), r.warnings().get(0));
        assertEquals(1, r.entries().size());
        assertFalse(r.entries().get(0).enforced());
        assertEquals(Flexibility.FLEXIBLE, r.entries().get(0).trip().flexibility());
    }

    @Test
    void outside_plan_range_still_listed() {
        TripParseResult r = parse("""
                - worker: w1
                  trips:
                  - {start_date: 2025/01/01, end_date: 2025/01/10, region: r1, plan_flexibility: Fixed}
                  - {start_date: 2026/12/28, end_date: 2027/01/05, region: r1, plan_flexibility: Fixed}
                """);
        assertTrue(r.warnings().isEmpty(), r.warnings().toString());
        assertEquals(2, r.entries().size());
        assertTrue(r.entries().get(0).enforced());
        assertTrue(r.entries().get(1).enforced());
    }

    @Test
    void duplicates_deduped() {
        TripParseResult r = parse("""
                - worker: w1
                  trips:
                  - {start_date: 2026/09/01, end_date: 2026/09/10, region: r1, plan_flexibility: Fixed}
                  - {start_date: 2026-09-01, end_date: '2026-09-10', region: r1, flexibility: fixed, description: again}
                - worker: w1
                  trips:
                  - {start_date: 2026/09/01, end_date: 2026/09/10, region: r1, plan_flexibility: Fixed}
                  - {start_date: 2026/09/12, end_date: 2026/09/14, region: r1, plan_flexibility: Flexible}
                  - {start_date: 2026/09/12, end_date: 2026/09/14, region: r1, plan_flexibility: Flexible}
                """);
        assertTrue(r.warnings().isEmpty(), "duplicates are silent: " + r.warnings());
        assertEquals(2, r.entries().size());
        assertTrue(r.entries().get(0).enforced());
        assertFalse(r.entries().get(1).enforced());
    }

    @Test
    void strict_mode_throws() {
        Object loaded = yaml("""
                - worker: ghost
                  trips:
                  - {start_date: 2026/09/01, end_date: 2026/09/10, region: r1, plan_flexibility: Fixed}
                """);
        InvalidInputDataException ex =
                assertThrows(InvalidInputDataException.class, () -> TripListParser.parse(loaded, CTX, true));
        assertTrue(ex.getMessage().startsWith("InvalidInputData: "), ex.getMessage());
        assertTrue(ex.getMessage().contains("ghost"), ex.getMessage());

        // strict with nothing to warn about does not throw
        TripParseResult ok = TripListParser.parse(yaml(W1_FIXED_THEN_FLEXIBLE), CTX, true);
        assertEquals(2, ok.entries().size());
        // lenient with the same bad input does not throw
        assertEquals(1, TripListParser.parse(loaded, CTX, false).warnings().size());
    }

    @Test
    void extra_keys_kept_in_raw() {
        TripParseResult r = parse("""
                - worker: w1
                  trips:
                  - start_date: 2026/09/01
                    end_date: 2026/09/10
                    region: r1
                    plan_flexibility: Fixed
                    description: install
                    ticket: T-42
                    meta: {a: 1}
                """);
        assertTrue(r.warnings().isEmpty(), r.warnings().toString());
        TripEntry e = r.entries().get(0);
        assertEquals("T-42", e.raw().get("ticket"));
        assertEquals(Map.of("a", 1), e.raw().get("meta"));
        assertEquals("install", e.raw().get("description"));
        assertEquals("2026/09/01", e.raw().get("start_date"));
        assertEquals(List.of("start_date", "end_date", "region", "plan_flexibility", "description", "ticket", "meta"),
                List.copyOf(e.raw().keySet()), "raw keeps the original key order");
        assertEquals(Set.of("ticket", "meta"), e.trip().extra().keySet());
        assertEquals("install", e.trip().description());
    }

    // ------------------------------------------------------------------ wiring into the schedule file

    @Test
    void trip_list_survives_preprocess() throws Exception {
        Path dir = tmp.resolve("work");
        Files.createDirectories(dir);
        Path env = dir.resolve("EnvConfig.yaml");
        Path sched = dir.resolve("Schedule.yaml");
        Files.copy(SMALL.resolve("EnvConfig.yaml"), env, StandardCopyOption.REPLACE_EXISTING);
        Files.copy(SMALL.resolve("Schedule.yaml"), sched, StandardCopyOption.REPLACE_EXISTING);
        // the small fixture ends with assignment_list under "schedule:"; the small fixture has a phase that started before
        // the cut-off, so preprocessScheduleYaml really rewrites the file
        Files.writeString(sched, """
                  trip_list:
                  - worker: w1
                    trips:
                    - start_date: 2026-04-01
                      end_date: '2026/04/03'
                      region: r1
                      plan_flexibility: Fixed
                      description: unquoted dash date, becomes a timestamp when the file is rewritten
                      ticket: T-1
                    - {start_date: 2026/04/03, end_date: 2026/04/05, region: r2, plan_flexibility: Flexible}
                  - worker: ghost
                    trips:
                    - {start_date: 2026/04/01, end_date: 2026/04/02, region: r1, plan_flexibility: Fixed}
                """, StandardOpenOption.APPEND);

        PlanBuilder.BuiltProblem built = PlanBuilder.buildProblem(env.toString(), sched.toString());

        // the key is still in the (rewritten) file
        String text = Files.readString(sched);
        assertTrue(text.contains("trip_list"), "trip_list dropped by preprocessScheduleYaml:\n" + text);
        assertTrue(text.contains("T00:00:00"), "precondition: the file really was rewritten (dates dumped as timestamps)");
        Map<String, Object> root = YamlFiles.loadYaml(sched.toString());
        @SuppressWarnings("unchecked")
        Map<String, Object> s = (Map<String, Object>) root.get("schedule");
        List<?> list = (List<?>) s.get("trip_list");
        assertEquals(2, list.size());
        Map<?, ?> w1 = (Map<?, ?>) list.get(0);
        assertEquals("w1", w1.get("worker"));
        assertEquals(2, ((List<?>) w1.get("trips")).size());
        assertEquals("T-1", ((Map<?, ?>) ((List<?>) w1.get("trips")).get(0)).get("ticket"));

        // and the parsed schedule has the entries (ghost worker: warned, Fixed entry kept, not enforced)
        assertEquals(3, built.sch.tripEntries.size());
        assertTrue(built.sch.tripEntries.get(0).enforced());
        assertEquals(LocalDate.of(2026, 4, 1), built.sch.tripEntries.get(0).trip().start());
        assertEquals(LocalDate.of(2026, 4, 3), built.sch.tripEntries.get(0).trip().end());
        assertFalse(built.sch.tripEntries.get(1).enforced());
        assertFalse(built.sch.tripEntries.get(2).enforced());
        assertEquals(1, built.sch.tripWarnings.size(), built.sch.tripWarnings.toString());
        assertTrue(built.sch.tripWarnings.get(0).contains("ghost"));
    }

    @Test
    void schedule_without_trip_list_has_no_entries() throws Exception {
        Path dir = tmp.resolve("work2");
        Files.createDirectories(dir);
        Path env = dir.resolve("EnvConfig.yaml");
        Path sched = dir.resolve("Schedule.yaml");
        Files.copy(SMALL.resolve("EnvConfig.yaml"), env, StandardCopyOption.REPLACE_EXISTING);
        Files.copy(SMALL.resolve("Schedule.yaml"), sched, StandardCopyOption.REPLACE_EXISTING);
        PlanBuilder.BuiltProblem built = PlanBuilder.buildProblem(env.toString(), sched.toString());
        assertTrue(built.sch.tripEntries.isEmpty());
        assertTrue(built.sch.tripWarnings.isEmpty());
        assertFalse(Files.readString(sched).contains("trip_list"));
    }

    @Test
    void strict_flag_in_solve_context_throws_on_build() throws Exception {
        Path dir = tmp.resolve("work3");
        Files.createDirectories(dir);
        Path env = dir.resolve("EnvConfig.yaml");
        Path sched = dir.resolve("Schedule.yaml");
        Files.copy(SMALL.resolve("EnvConfig.yaml"), env, StandardCopyOption.REPLACE_EXISTING);
        Files.copy(SMALL.resolve("Schedule.yaml"), sched, StandardCopyOption.REPLACE_EXISTING);
        Files.writeString(sched, """
                  trip_list:
                  - worker: ghost
                    trips:
                    - {start_date: 2026/04/01, end_date: 2026/04/02, region: r1, plan_flexibility: Fixed}
                """, StandardOpenOption.APPEND);
        SolveContext.current().tripValidationStrict = true;
        InvalidInputDataException ex = assertThrows(InvalidInputDataException.class,
                () -> PlanBuilder.buildProblem(env.toString(), sched.toString()));
        assertTrue(ex.getMessage().startsWith("InvalidInputData: "));
    }
}
