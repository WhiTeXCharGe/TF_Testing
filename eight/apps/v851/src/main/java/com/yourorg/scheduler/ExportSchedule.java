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

        // Support both:
        //   { schedule: { ... } }
        // and:
        //   { plan_range: ..., workflow_task_list: ..., assignment_list: ... }
        Map<String,Object> sched = (Map<String,Object>) root.get("schedule");
        if (sched == null || sched.isEmpty()) {
            sched = root;
        }

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
        List<Map<String,Object>> wfList =
                (List<Map<String,Object>>) sched.getOrDefault("workflow_task_list", List.of());

        for (Map<String,Object> wf : wfList) {
            String module = String.valueOf(wf.get("id"));
            List<Map<String,Object>> phases =
                    (List<Map<String,Object>>) wf.getOrDefault("phase_task_list", List.of());

            for (Map<String,Object> ph : phases) {
                List<Map<String,Object>> ops =
                        (List<Map<String,Object>>) ph.getOrDefault("operation_task_list", List.of());

                for (Map<String,Object> ot : ops) {
                    String op   = String.valueOf(ot.get("operation")); // e.g. p2o2
                    String otId = String.valueOf(ot.get("id"));        // e.g. e1p2o2
                    opTaskId.put(module + "|" + op, otId);
                }
            }
        }

        // ---- Original assignment list ----
        Object assignmentObj = sched.get("assignment_list");
        List<Map<String,Object>> original =
                (assignmentObj instanceof List)
                        ? (List<Map<String,Object>>) assignmentObj
                        : new ArrayList<>();

        // Preserve original fixed + original flexible
        List<Map<String,Object>> preservedFixed = new ArrayList<>();
        List<Map<String,Object>> preservedFlexible = new ArrayList<>();

        for (Map<String,Object> a : original) {
            if (a == null) continue;
            String flex = String.valueOf(a.getOrDefault("plan_flexibility", "Flexible"));
            if ("fixed".equalsIgnoreCase(flex)) {
                preservedFixed.add(a);
            } else {
                preservedFlexible.add(a);
            }
        }

        // ---- Detect "stage1 only" situation (warmPinned still exists) ----
        boolean hasWarmPinned = false;
        if (plan != null && plan.seats != null) {
            for (EmployeeSchedule.CrewSeat s : plan.seats) {
                if (s != null && s.warmPinned && s.pinnedWid != null) {
                    hasWarmPinned = true;
                    break;
                }
            }
        }

        // If stage1 only: DO NOT overwrite flexible rows at all
        if (hasWarmPinned) {
            List<Map<String,Object>> merged = new ArrayList<>();
            merged.addAll(preservedFixed);
            merged.addAll(preservedFlexible);

            sched.put("assignment_list", merged);

            DumperOptions opt = new DumperOptions();
            opt.setDefaultFlowStyle(DumperOptions.FlowStyle.BLOCK);
            opt.setPrettyFlow(true);
            Yaml yaml = new Yaml(opt);
            try (Writer out = Files.newBufferedWriter(Paths.get(schedPath))) {
                yaml.dump(root, out);
            }

            System.out.println(
                    "Stage1-only detected (warmPinned exists). Kept original assignments as-is."
                            + " | fixed=" + preservedFixed.size()
                            + " | flexible=" + preservedFlexible.size()
            );
            return;
        }

        // ---- Build a mask of fixed dates per (worker, operation_task_id) so exporter won't duplicate them ----
        Map<String, Set<Integer>> fixedMask = new HashMap<>();

        for (Map<String, Object> a : preservedFixed) {
            if (a == null) continue;

            String wid  = String.valueOf(a.get("worker"));
            String task = String.valueOf(a.get("operation_task")); // operation_task_id (e.g. e1p2o2)
            if (wid == null || task == null) continue;

            String wdKey = a.containsKey("work_date_lsit") ? "work_date_lsit" : "work_date_list";
            List<Map<String, Object>> wdl =
                    (List<Map<String, Object>>) a.getOrDefault(wdKey, List.of());

            Set<Integer> set = fixedMask.computeIfAbsent(wid + "|" + task, k -> new HashSet<>());
            for (Map<String, Object> item : wdl) {
                if (item == null) continue;
                String dateStr = String.valueOf(item.get("date"));
                Integer did = EmployeeSchedule.dayIdFromDate(planStart, dateStr);
                if (did != null) set.add(did);
            }
        }

        // ---- Index blocks by id ----
        Map<Integer, EmployeeSchedule.BlockDecision> blockById = new HashMap<>();
        if (plan != null && plan.blocks != null) {
            for (EmployeeSchedule.BlockDecision b : plan.blocks) {
                if (b != null) blockById.put(b.id, b);
            }
        }

        DateTimeFormatter DF = DateTimeFormatter.ofPattern("yyyy/MM/dd");

        // ---- Build NEW flexible rows from solved seats (final run) ----
        List<Map<String,Object>> newFlex = new ArrayList<>();

        if (plan != null && plan.seats != null) {
            for (EmployeeSchedule.CrewSeat s : plan.seats) {
                if (s == null || EmployeeSchedule.isUnassigned(s.employee)) continue;

                // Skip fixed pinned seats (they come from original fixed rows)
                if (s.pinnedFixed) continue;

                String module = s.module;
                String opId   = s.opId; // p2o2
                String taskId = opTaskId.get(module + "|" + opId); // e1p2o2
                if (taskId == null || "null".equals(taskId)) continue;

                EmployeeSchedule.BlockDecision b = blockById.get(s.blockId);

                // Build work map once
                Map<Integer,Integer> byDay = new TreeMap<>();

                // Case A: warmPinned schedule on dummy block (blockId=0) => use pinned span
                if (s.warmPinned && s.blockId == 0
                        && s.pinnedStart != null && s.pinnedDays != null && s.pinnedDays > 0) {

                    int h = (s.pinnedHours != null) ? s.pinnedHours : 8;

                    for (int i = 0; i < s.pinnedDays; i++) {
                        int did = s.pinnedStart + i;
                        if (!EmployeeSchedule.isWorkingDay(did, s.factory)) continue;
                        byDay.merge(did, h, Integer::sum);
                    }

                // Case B: normal seat attached to a real block => use block span
                } else if (b != null && b.startDay != null && b.days != null && b.days > 0) {

                    int h = (b.hours != null) ? b.hours : b.chosenHours();

                    for (int i = 0; i < b.days; i++) {
                        int did = b.startDay + i;
                        if (!EmployeeSchedule.isWorkingDay(did, s.factory)) continue;
                        byDay.merge(did, h, Integer::sum);
                    }

                } else {
                    // No usable span -> skip
                    continue;
                }

                // Remove dates that are already fixed for (worker, taskId)
                Set<Integer> mask = fixedMask.get(s.employee.wid + "|" + taskId);
                if (mask != null && !mask.isEmpty()) {
                    byDay.keySet().removeAll(mask);
                }

                if (byDay.isEmpty()) continue;

                int firstIdx = ((TreeMap<Integer,Integer>)byDay).firstKey();
                int lastIdx  = ((TreeMap<Integer,Integer>)byDay).lastKey();

                // build work_date_list
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

        // ---- Write back: fixed preserved + NEW flexible (overwrite old flexible) ----
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
                "Overwrote " + schedPath
                        + " | new flexible rows=" + newFlex.size()
                        + " | preserved fixed rows=" + preservedFixed.size()
                        + " | (old flexible overwritten because warmPinned not present)"
        );
    }

}
