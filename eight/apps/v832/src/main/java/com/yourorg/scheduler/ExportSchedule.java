package com.yourorg.scheduler;

import java.io.*;
import java.nio.file.*;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.*;

import org.yaml.snakeyaml.DumperOptions;
import org.yaml.snakeyaml.Yaml;

public class ExportSchedule {

    @SuppressWarnings("unchecked")
    public static void overwriteScheduleWithAssignments(
            EmployeeSchedule.SinglePassPlan plan,
            LocalDate planStart,
            String schedPath,
            String envPath) throws IOException {

        // ---- Load YAML ----
        Map<String,Object> root;
        try (InputStream in = Files.newInputStream(Paths.get(schedPath))) {
            root = new Yaml().load(in);
        }
        Map<String,Object> sched = (Map<String,Object>) root.get("schedule");

        // ---- Ensure calendars (optional) ----
        try {
            if (EmployeeSchedule.CAL == null || EmployeeSchedule.CAL.weekends.isEmpty()) {
                Map<String,Object> pr = (Map<String,Object>) sched.get("plan_range");
                LocalDate planEnd = LocalDate.parse(
                    String.valueOf(pr.get("end_date")).replace("-", "/"),
                    EmployeeSchedule.DF
                );
                EmployeeSchedule.buildCalendars(envPath, planStart, planEnd);
            }
        } catch (Exception ignore) {}

        // ---- Build (module, op) -> operation_task_id map ----
        Map<String,String> opTaskId = new HashMap<>();
        List<Map<String,Object>> wfList = (List<Map<String,Object>>) sched.getOrDefault("workflow_task_list", List.of());
        for (Map<String,Object> wf : wfList) {
            String module = String.valueOf(wf.get("id"));
            List<Map<String,Object>> phases = (List<Map<String,Object>>) wf.getOrDefault("phase_task_list", List.of());
            for (Map<String,Object> ph : phases) {
                List<Map<String,Object>> ops = (List<Map<String,Object>>) ph.getOrDefault("operation_task_list", List.of());
                for (Map<String,Object> ot : ops) {
                    String op = String.valueOf(ot.get("operation"));
                    String otId = String.valueOf(ot.get("id"));
                    opTaskId.put(module + "|" + op, otId);
                }
            }
        }

        // ---- Index facts ----
        Map<Integer, EmployeeSchedule.BlockDecision> blockById = new HashMap<>();
        if (plan.blocks != null) {
            for (EmployeeSchedule.BlockDecision b : plan.blocks) blockById.put(b.id, b);
        }
        DateTimeFormatter DF = DateTimeFormatter.ofPattern("yyyy/MM/dd");

        // ---- Original fixed rows preserved ----
        List<Map<String,Object>> original = (List<Map<String,Object>>) sched.getOrDefault("assignment_list", List.of());
        List<Map<String,Object>> preservedFixed = new ArrayList<>();
        for (Map<String,Object> a : original) {
            String flex = String.valueOf(a.getOrDefault("plan_flexibility", "Flexible"));
            if ("fixed".equalsIgnoreCase(flex)) preservedFixed.add(a);
        }

        // ---- Build flexible rows from solved seats ----
        List<Map<String,Object>> newFlex = new ArrayList<>();
        if (plan.seats != null) {
            for (EmployeeSchedule.CrewSeat s : plan.seats) {
                if (s == null || EmployeeSchedule.isUnassigned(s.employee)) continue;

                String module = s.module;
                String opId   = s.opId;
                String taskId = opTaskId.get(module + "|" + opId);
                if (taskId == null) continue; // unmatched op

                // Determine per-day worked hours and span
                Map<Integer,Integer> byDay = new TreeMap<>();

                if (s.pinned) {
                    if (s.pinnedStart != null && s.pinnedDays != null && s.pinnedDays > 0) {
                        int h = (s.pinnedHours != null && s.pinnedHours > 0) ? s.pinnedHours : 8;
                        for (int i=0;i<s.pinnedDays;i++) {
                            int did = s.pinnedStart + i;
                            if (!EmployeeSchedule.isWorkingDay(did, s.factory)) continue;
                            byDay.merge(did, h, Integer::sum);
                        }
                    }
                } else {
                    EmployeeSchedule.BlockDecision b = blockById.get(s.blockId);
                    if (b != null && b.startDay != null && b.days != null && b.days > 0) {
                        int h = EmployeeSchedule.autoHours(b);
                        for (int i=0;i<b.days;i++) {
                            int did = b.startDay + i;
                            if (!EmployeeSchedule.isWorkingDay(did, s.factory)) continue;
                            byDay.merge(did, h, Integer::sum);
                        }
                    }
                }

                if (byDay.isEmpty()) continue;

                int firstIdx = byDay.keySet().iterator().next();
                int lastIdx  = ((TreeMap<Integer,Integer>)byDay).lastKey();

                List<Map<String,Object>> work = new ArrayList<>();
                for (Map.Entry<Integer,Integer> e : byDay.entrySet()) {
                    work.add(Map.of(
                        "date", planStart.plusDays(e.getKey()).format(DF),
                        "hour", e.getValue()
                    ));
                }

                Map<String,Object> row = new LinkedHashMap<>();
                row.put("worker", s.employee.wid);
                row.put("operation_task", taskId);
                row.put("start_date", planStart.plusDays(firstIdx).format(DF));
                row.put("end_date",   planStart.plusDays(lastIdx).format(DF));
                row.put("work_date_list", work);
                row.put("plan_flexibility", "Flexible");
                newFlex.add(row);
            }
        }

        // ---- Write back ----
        List<Map<String,Object>> merged = new ArrayList<>();
        merged.addAll(preservedFixed);
        merged.addAll(newFlex);
        sched.put("assignment_list", merged);

        DumperOptions opt = new DumperOptions();
        opt.setDefaultFlowStyle(DumperOptions.FlowStyle.BLOCK);
        opt.setPrettyFlow(true);
        Yaml yaml = new Yaml(opt);
        try (Writer out = Files.newBufferedWriter(Paths.get(schedPath))) {
            yaml.dump(root, out);
        }

        System.out.println(
            "Overwrote " + schedPath +
            " | new flexible rows=" + newFlex.size() +
            " | preserved fixed rows=" + preservedFixed.size()
        );
    }
}
