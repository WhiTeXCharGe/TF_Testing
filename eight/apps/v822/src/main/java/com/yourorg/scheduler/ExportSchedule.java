package com.yourorg.scheduler;

import java.io.InputStream;
import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.LocalDate;
import java.util.*;
import org.yaml.snakeyaml.DumperOptions;
import org.yaml.snakeyaml.Yaml;

/**
 * Writes Pass2 assignments back into Schedule.yaml.
 * Provides both (plan,start,schedulePath) and (plan,start,schedulePath,envPath) overloads.
 */
public final class ExportSchedule {

    private ExportSchedule() {}

    private static String safeStr(Object o) { return o == null ? "" : String.valueOf(o); }

    /** Back-compat overload. */
    public static void overwriteScheduleWithAssignments(
            EmployeeSchedule.Pass2Plan plan,
            LocalDate planStart,
            String schedulePath
    ) throws IOException {
        overwriteScheduleWithAssignments(plan, planStart, schedulePath, null);
    }

    @SuppressWarnings("unchecked")
    public static void overwriteScheduleWithAssignments(
            EmployeeSchedule.Pass2Plan plan,
            LocalDate planStart,
            String schedulePath,
            String envPath // currently unused; present to satisfy callers that pass EnvConfig path
    ) throws IOException {

        // --- Load existing YAML (or create a fresh root if file is empty)
        Map<String, Object> root;
        try (InputStream in = Files.newInputStream(Paths.get(schedulePath))) {
            Yaml yaml = new Yaml();
            root = yaml.load(in);
        }
        if (root == null) root = new LinkedHashMap<>();

        Map<String, Object> schedule =
                (Map<String, Object>) root.getOrDefault("schedule", root);

        // --- Build assignment_list from Pass2 seats/seatDays
        List<Map<String, Object>> assignmentList = new ArrayList<>();

        if (plan != null && plan.seats != null) {
            // Index seatDays by seatKey
            Map<String, List<EmployeeSchedule.SeatDay>> daysBySeat = new HashMap<>();
            if (plan.seatDays != null) {
                for (EmployeeSchedule.SeatDay sd : plan.seatDays) {
                    daysBySeat.computeIfAbsent(sd.seatKey, k -> new ArrayList<>()).add(sd);
                }
            }

            for (EmployeeSchedule.CrewSeat seat : plan.seats) {
                if (EmployeeSchedule.isUnassigned(seat.employee)) continue;

                Map<String, Object> row = new LinkedHashMap<>();
                // operation_task format like "e16p4o1" (module + opId already contains p*o*)
                String opTask = safeStr(seat.module) + safeStr(seat.opId);
                row.put("operation_task", opTask);
                row.put("worker", safeStr(seat.employee.wid));
                row.put("plan_flexibility", seat.pinned ? "fixed" : "flexible");
                row.put("start_date", planStart.plusDays(Math.max(0, seat.startDayId)).toString());

                // Work date list (respect hours/day produced in Pass2)
                List<Map<String, Object>> wdl = new ArrayList<>();
                List<EmployeeSchedule.SeatDay> sdays =
                        daysBySeat.getOrDefault(seat.seatKey, Collections.emptyList());
                sdays.sort(Comparator.comparing(sd -> sd.day.date));
                for (EmployeeSchedule.SeatDay sd : sdays) {
                    Map<String, Object> di = new LinkedHashMap<>();
                    di.put("date", sd.day.date.toString());
                    di.put("hour", sd.hours);
                    wdl.add(di);
                }
                row.put("work_date_list", wdl);

                assignmentList.add(row);
            }
        }

        schedule.put("assignment_list", assignmentList);

        // If the original YAML had a top-level "schedule", keep it; otherwise write flat.
        if (root.containsKey("schedule")) {
            root.put("schedule", schedule);
        } else {
            root = schedule;
        }

        // --- Save YAML
        DumperOptions opt = new DumperOptions();
        opt.setDefaultFlowStyle(DumperOptions.FlowStyle.BLOCK);
        opt.setPrettyFlow(true);
        Yaml yaml = new Yaml(opt);
        try (Writer w = Files.newBufferedWriter(Paths.get(schedulePath))) {
            yaml.dump(root, w);
        }
    }
}
