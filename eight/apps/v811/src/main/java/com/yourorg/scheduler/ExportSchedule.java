package com.yourorg.scheduler;

import java.io.*;
import java.nio.file.*;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.*;

import org.yaml.snakeyaml.DumperOptions;
import org.yaml.snakeyaml.Yaml;

/** Writes assignment_list back into Schedule.yaml from Pass2Plan (seats + seatDays). */
public class ExportSchedule {

    @SuppressWarnings("unchecked")
    public static void overwriteScheduleWithAssignments(EmployeeSchedule.Pass2Plan finalPass2,
                                                        LocalDate planStart,
                                                        String schedPath) throws IOException {

        Map<String,Object> root;
        try (InputStream in = Files.newInputStream(Paths.get(schedPath))) {
            root = new Yaml().load(in);
        }
        Map<String,Object> sched = (Map<String,Object>) root.get("schedule");

        // (module, op) -> operation_task_id
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

        // seatKey -> (module, op, factory), and employee by seat
        Map<String, String[]> seatMeta = new HashMap<>();
        Map<String, EmployeeSchedule.EmployeeFact> empBySeat = new HashMap<>();
        for (EmployeeSchedule.CrewSeat s : finalPass2.seats) {
            if (s == null) continue;
            seatMeta.put(s.seatKey, new String[]{ s.module, s.opId, s.factory });
            empBySeat.put(s.seatKey, s.employee);
        }

        // sum hours per (wid, dayIdx, module, op)
        Map<List<Object>, Integer> per = new HashMap<>();
        for (EmployeeSchedule.SeatDay sd : finalPass2.seatDays) {
            EmployeeSchedule.EmployeeFact e = empBySeat.get(sd.seatKey);
            if (e == null || e.id == 0) continue;
            String[] meta = seatMeta.get(sd.seatKey);
            if (meta == null) continue;
            String module = meta[0]; String opId = meta[1];

            List<Object> key = List.of(e.wid, sd.day.id, module, opId);
            per.merge(key, sd.hours, Integer::sum);
        }

        // bucket by (wid, module, op) -> {dayIdx -> hours}
        Map<List<Object>, Map<Integer,Integer>> buckets = new HashMap<>();
        for (Map.Entry<List<Object>, Integer> en : per.entrySet()) {
            List<Object> k = en.getKey();
            String wid = (String) k.get(0);
            int didx    = (Integer) k.get(1);
            String mod  = (String) k.get(2);
            String opId = (String) k.get(3);
            List<Object> head = List.of(wid, mod, opId);
            buckets.computeIfAbsent(head, kk -> new HashMap<>())
                   .merge(didx, en.getValue(), Integer::sum);
        }

        DateTimeFormatter DF = DateTimeFormatter.ofPattern("yyyy/MM/dd");
        List<Map<String,Object>> assignments = new ArrayList<>();

        for (Map.Entry<List<Object>, Map<Integer,Integer>> en : buckets.entrySet()) {
            List<Object> head = en.getKey();
            String wid = (String) head.get(0);
            String mod = (String) head.get(1);
            String opId = (String) head.get(2);

            Map<Integer,Integer> byDay = en.getValue();
            List<Integer> didxs = new ArrayList<>(byDay.keySet());
            Collections.sort(didxs);
            if (didxs.isEmpty()) continue;

            List<List<Integer>> runs = new ArrayList<>();
            List<Integer> run = new ArrayList<>(List.of(didxs.get(0)));
            for (int i=1;i<didxs.size();i++) {
                int d = didxs.get(i);
                if (d == run.get(run.size()-1) + 1) run.add(d);
                else { runs.add(run); run = new ArrayList<>(List.of(d)); }
            }
            runs.add(run);

            for (List<Integer> r : runs) {
                LocalDate start = planStart.plusDays(r.get(0));
                LocalDate end   = planStart.plusDays(r.get(r.size()-1));

                List<Map<String,Object>> work = new ArrayList<>();
                for (int d : r) {
                    work.add(Map.of(
                            "date", planStart.plusDays(d).format(DF),
                            "hour", byDay.get(d)
                    ));
                }

                String taskId = opTaskId.get(mod + "|" + opId);
                if (taskId == null) continue;

                Map<String,Object> a = new LinkedHashMap<>();
                a.put("worker", wid);
                a.put("operation_task", taskId);
                a.put("start_date", start.format(DF));
                a.put("end_date", end.format(DF));
                a.put("work_date_list", work);
                a.put("plan_flexibility", "Flexible");
                assignments.add(a);
            }
        }

        sched.put("assignment_list", assignments);

        DumperOptions opt = new DumperOptions();
        opt.setDefaultFlowStyle(DumperOptions.FlowStyle.BLOCK);
        opt.setPrettyFlow(true);
        Yaml yaml = new Yaml(opt);

        try (Writer out = Files.newBufferedWriter(Paths.get(schedPath))) {
            yaml.dump(root, out);
        }
        System.out.println("Overwrote " + schedPath + " with " + assignments.size() + " assignments.");
    }
}
