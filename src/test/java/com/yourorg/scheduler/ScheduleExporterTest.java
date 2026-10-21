package com.yourorg.scheduler;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.DayOfWeek;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.yaml.snakeyaml.LoaderOptions;
import org.yaml.snakeyaml.Yaml;
import org.yaml.snakeyaml.constructor.SafeConstructor;

import com.yourorg.scheduler.export.ScheduleExporter;
import com.yourorg.scheduler.model.SinglePassPlan;

/**
 * Round trip through {@link ScheduleExporter#overwriteScheduleWithAssignments} on the {@code small} fixture:
 * build from a temp copy (the build rewrites Schedule.yaml in place), solve 20 steps with seed 42, export into the
 * same temp copy and read the result back with SnakeYAML.
 */
class ScheduleExporterTest {

    static final Path SMALL_FIXTURE = CharacterizationTest.SMALL_FIXTURE;
    static final DateTimeFormatter DF = DateTimeFormatter.ofPattern("yyyy/MM/dd");

    /** w2's personal unavailable date in the fixture's EnvConfig.yaml (worker_list, w2). */
    static final String W2_UNAVAILABLE = "2026/03/30";
    /** Region r1 (fab f1, module e1) is closed on this date (region_list, r1). */
    static final String R1_CLOSED = "2026/03/28";

    @TempDir
    Path tmp;

    @SuppressWarnings("unchecked")
    static Map<String, Object> schedule(Path file) throws Exception {
        try (InputStream in = Files.newInputStream(file)) {
            LoaderOptions opts = new LoaderOptions();
            opts.setCodePointLimit(10 * 1024 * 1024);
            Map<String, Object> root = new Yaml(new SafeConstructor(opts)).load(in);
            return (Map<String, Object>) root.getOrDefault("schedule", root);
        }
    }

    @SuppressWarnings("unchecked")
    static List<Map<String, Object>> assignments(Map<String, Object> sched) {
        return (List<Map<String, Object>>) sched.get("assignment_list");
    }

    static boolean isFixed(Map<String, Object> a) {
        return "fixed".equalsIgnoreCase(String.valueOf(a.getOrDefault("plan_flexibility", "Flexible")));
    }

    @SuppressWarnings("unchecked")
    static List<Map<String, Object>> workDates(Map<String, Object> a) {
        return (List<Map<String, Object>>) a.getOrDefault("work_date_list", List.of());
    }

    /** operation_task id -> fab, over the workflow_task_list (misc tasks have no fab). */
    @SuppressWarnings("unchecked")
    static void collectTasks(Map<String, Object> sched, Set<String> tasks, Map<String, String> taskFab) {
        for (Map<String, Object> wf : (List<Map<String, Object>>) sched.get("workflow_task_list")) {
            String fab = String.valueOf(wf.get("fab"));
            for (Map<String, Object> ph : (List<Map<String, Object>>) wf.get("phase_task_list")) {
                for (Map<String, Object> ot : (List<Map<String, Object>>) ph.get("operation_task_list")) {
                    String id = String.valueOf(ot.get("id"));
                    tasks.add(id);
                    taskFab.put(id, fab);
                }
            }
        }
    }

    @Test
    void fixture_roundtrip_keeps_fixed_rows_and_replaces_flexible() throws Exception {
        // the fixture as shipped
        Map<String, Object> original = schedule(SMALL_FIXTURE.resolve("Schedule.yaml"));

        // build from a temp copy; this rewrites the copy's Schedule.yaml (phase-started rule), and that rewritten file
        // is what the exporter reads, so it is the exporter's real input
        Path work = tmp.resolve("work");
        SinglePassPlan plan = Characterization.buildFrom(SMALL_FIXTURE, work);
        Path sched = work.resolve("Schedule.yaml");
        Path env = work.resolve("EnvConfig.yaml");
        Map<String, Object> input = schedule(sched);

        List<Map<String, Object>> inputFixed = new ArrayList<>();
        List<Map<String, Object>> inputFlexible = new ArrayList<>();
        for (Map<String, Object> a : assignments(input)) {
            (isFixed(a) ? inputFixed : inputFlexible).add(a);
        }
        assertFalse(inputFixed.isEmpty(), "fixture must have Fixed rows");
        assertFalse(inputFlexible.isEmpty(), "fixture must have Flexible rows to replace");

        SinglePassPlan best = Characterization.solveSeeded(plan, 20, 42L);
        LocalDate planStart = LocalDate.parse(
                String.valueOf(((Map<?, ?>) input.get("plan_range")).get("start_date")), DF);
        ScheduleExporter.overwriteScheduleWithAssignments(best, planStart, sched.toString(), env.toString());

        Map<String, Object> output = schedule(sched);
        List<Map<String, Object>> outAll = assignments(output);
        assertNotNull(outAll);

        // (a) every Fixed row of the input is unchanged, and the Fixed rows come first, in input order
        int nFixed = inputFixed.size();
        assertTrue(outAll.size() > nFixed, "export must add the newly solved Flexible rows");
        assertEquals(inputFixed, outAll.subList(0, nFixed), "Fixed rows first and unchanged");
        // the fixture as shipped has 3 Fixed rows (row A, misc task, e2p1o1 of w3): all of them survive unchanged
        int seen = 0;
        for (Map<String, Object> a : assignments(original)) {
            if (!isFixed(a)) continue;
            assertTrue(outAll.subList(0, nFixed).contains(a), "original Fixed row missing from output: " + a);
            seen++;
        }
        assertEquals(3, seen);

        // (b) everything after the Fixed rows is a freshly built Flexible row
        Set<String> knownTasks = new HashSet<>();
        Map<String, String> taskFab = new HashMap<>();
        collectTasks(input, knownTasks, taskFab);
        List<Map<String, Object>> outFlexible = outAll.subList(nFixed, outAll.size());
        for (Map<String, Object> a : outFlexible) {
            assertEquals("Flexible", a.get("plan_flexibility"), "row after the Fixed block: " + a);
            assertNotNull(a.get("worker"), "worker: " + a);
            assertTrue(knownTasks.contains(String.valueOf(a.get("operation_task"))), "operation_task: " + a);
            assertFalse(workDates(a).isEmpty(), "work_date_list: " + a);
            assertNotNull(a.get("start_date"));
            assertNotNull(a.get("end_date"));
        }
        for (Map<String, Object> a : outAll.subList(0, nFixed)) {
            assertTrue(isFixed(a), "Fixed block holds only Fixed rows: " + a);
        }

        // (c) the other sections are untouched
        assertEquals(input.get("misc_task_list"), output.get("misc_task_list"));
        assertEquals(input.get("workflow_task_list"), output.get("workflow_task_list"));

        // (d) no new Flexible row works on a closed day of its worker / fab
        // (w2: personal unavailable date; fab f1 sits in region r1 which is closed on R1_CLOSED;
        //  fab f2 belongs to customer c2, closed on Sundays)
        for (Map<String, Object> a : outFlexible) {
            String worker = String.valueOf(a.get("worker"));
            String fab = taskFab.get(String.valueOf(a.get("operation_task")));
            for (Map<String, Object> item : workDates(a)) {
                String date = String.valueOf(item.get("date"));
                if ("w2".equals(worker)) assertNotEquals(W2_UNAVAILABLE, date, "w2 unavailable on " + date);
                if ("f1".equals(fab)) assertNotEquals(R1_CLOSED, date, "region r1 closed on " + date);
                if ("f2".equals(fab)) {
                    assertNotEquals(DayOfWeek.SUNDAY, LocalDate.parse(date, DF).getDayOfWeek(),
                            "customer c2 closed on " + date);
                }
            }
        }
    }
}
