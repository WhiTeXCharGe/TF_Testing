package com.yourorg.scheduler.io;

import java.io.*;
import java.nio.file.*;
import java.time.LocalDate;
import java.util.*;

import com.yourorg.scheduler.calendar.SolveContext;
import com.yourorg.scheduler.config.InvalidInputDataException;
import com.yourorg.scheduler.config.SolverConstants;
import com.yourorg.scheduler.model.*;
import com.yourorg.scheduler.trip.TripContext;
import com.yourorg.scheduler.trip.TripListParser;
import com.yourorg.scheduler.trip.TripParseResult;

/** Schedule.yaml preprocessing and parsing (windows, fixed/flexible/initial rows, day slots). */
public class ScheduleParser {

    public static String preprocessScheduleYaml(String schedPath) throws IOException {
        Map<String,Object> root = YamlFiles.loadYaml(schedPath);
        @SuppressWarnings("unchecked")
        Map<String,Object> s = (Map<String,Object>) root.getOrDefault("schedule", root);

        // IMPORTANT: We don't have a separate "cut_off_date".
        // The "cutoff" is the plan_range.start_date (history before that is fixed).
        Map<String, Object> planRange = YamlFiles.asMap(s.get("plan_range"));
        if (planRange == null) {
            return schedPath;
        }
        Object startObj = planRange.get("start_date");
        if (startObj == null) {
            return schedPath;
        }
        LocalDate cutOffDate = LocalDate.parse(YamlFiles.safeStr(startObj).replace("-", "/"), YamlFiles.DF);

        Object asgObj = s.get("assignment_list");
        @SuppressWarnings("unchecked")
        List<Map<String,Object>> asgs =
            (asgObj instanceof List) ? (List<Map<String,Object>>) asgObj : List.of();

        // --- helper: get (module, phaseNum) from operation_task like
        //     e72_p1, e72_p1o2, e72p1, e72p1o2
        record PhaseKey(String module, int phaseNum) {}

        java.util.function.Function<String, PhaseKey> phaseKeyOfOpTask = (opTask) -> {
            ParsedOpTaskRef ref = YamlFiles.parseOpTaskRef(opTask);
            return new PhaseKey(ref.module, ref.phaseNum);
        };

        // PASS 1: detect phases that started before cut-off
        Set<PhaseKey> started = new HashSet<>();

        for (Map<String,Object> a : asgs) {
            if (a == null) continue;

            String opTask = YamlFiles.safeStr(a.get("operation_task"));
            PhaseKey key = phaseKeyOfOpTask.apply(opTask);

            String wdKey = a.containsKey("work_date_lsit") ? "work_date_lsit" : "work_date_list";
            @SuppressWarnings("unchecked")
            List<Map<String,Object>> wdl =
                (a.get(wdKey) instanceof List) ? (List<Map<String,Object>>) a.get(wdKey) : List.of();

            boolean anyBefore = false;
            for (Map<String,Object> item : wdl) {
                if (item == null) continue;
                String ds = YamlFiles.safeStr(item.get("date"));
                if (ds.isBlank()) continue;
                LocalDate d = LocalDate.parse(ds.replace("-", "/"), YamlFiles.DF);
                if (d.isBefore(cutOffDate)) {
                    anyBefore = true;
                    break;
                }
            }
            if (anyBefore) started.add(key);
        }

        if (started.isEmpty()) {
            // nothing started before cutoff => no change
            return schedPath;
        }

        // PASS 2: convert ALL assignments in those phases to Fixed (NO trimming!)
        for (Map<String,Object> a : asgs) {
            if (a == null) continue;
            String opTask = YamlFiles.safeStr(a.get("operation_task"));
            PhaseKey key = phaseKeyOfOpTask.apply(opTask);

            if (started.contains(key)) {
                a.put("plan_flexibility", "Fixed"); // force fixed
            }
        }

        // Write to temp and return path
        YamlFiles.saveYaml(schedPath, root);
        return schedPath;
    }

    /**
     * Reads {@code trip_list} of the schedule file (root may be {@code schedule:}-wrapped or bare, as for
     * assignment_list), validates it against {@code ctx} and stores the result in {@code sch.tripEntries} /
     * {@code sch.tripWarnings} (empty when the key is absent). Every warning is printed as {@code [trips] WARN <text>}
     * and appended to the load log. In strict mode a warning makes this throw InvalidInputDataException (after the
     * warnings were printed). Validation only: nothing else in {@code sch} is touched.
     *
     * <p>Not part of {@link #parseSchedule}: the region ids of the context come from the EnvConfig calendars, which are
     * built from the plan range that parseSchedule reads.
     */
    public static void parseTrips(ParsedSchedule sch, String schedPath, TripContext ctx, boolean strict)
            throws IOException {
        Map<String,Object> root = YamlFiles.loadYaml(schedPath);
        @SuppressWarnings("unchecked")
        Map<String,Object> s = (Map<String,Object>) root.getOrDefault("schedule", root);
        Object rawTripList = (s == null) ? null : s.get("trip_list");
        try {
            TripParseResult r = TripListParser.parse(rawTripList, ctx, strict);
            sch.tripEntries = new ArrayList<>(r.entries());
            sch.tripWarnings = new ArrayList<>(r.warnings());
            reportTripWarnings(r.warnings());
        } catch (InvalidInputDataException e) {
            reportTripWarnings(e.details());
            throw e;
        }
    }

    private static void reportTripWarnings(List<String> warnings) {
        for (String w : warnings) {
            System.out.println("[trips] WARN " + w);
            LoadLog.appendLoadLog("[trips] WARN " + w);
        }
    }

    @SuppressWarnings("unchecked")
    public static ParsedSchedule parseSchedule(String schedPath, Map<String,OpDef> opdef) throws IOException {
        Map<String,Object> root = YamlFiles.loadYaml(schedPath);
        Map<String,Object> s = (Map<String,Object>) root.getOrDefault("schedule", root);

        LocalDate start = LocalDate.parse(
                YamlFiles.safeStr(((Map<String,Object>)s.get("plan_range")).get("start_date")).replace("-", "/"), YamlFiles.DF);
        LocalDate end   = LocalDate.parse(
                YamlFiles.safeStr(((Map<String,Object>)s.get("plan_range")).get("end_date")).replace("-", "/"), YamlFiles.DF);

        int horizon = (int) (end.toEpochDay() - start.toEpochDay()) + 1;

        List<TaskWindow> windows = new ArrayList<>();
        Map<String,Integer> required = new HashMap<>();

        // Track last end dayId per module (for "finished module" trim)
        Map<String,Integer> moduleLastEnd = new HashMap<>();

        Map<String,OpTaskMeta> opTaskMeta = new HashMap<>();

        Object wfObj = s.get("workflow_task_list");
        List<Map<String,Object>> wfTasks = (wfObj instanceof List) ? (List<Map<String,Object>>) wfObj : List.of();
        for (Map<String,Object> wf : wfTasks) {
            String module = YamlFiles.safeStr(wf.get("id"));
            String workflowId = YamlFiles.safeStr(wf.get("workflow"));
            if (workflowId.isBlank()) workflowId = YamlFiles.safeStr(wf.get("workflow_id"));
            String fab = YamlFiles.safeStr(wf.get("fab"));

            Object phasesObj = wf.get("phase_task_list");
            List<Map<String,Object>> phases = (phasesObj instanceof List) ? (List<Map<String,Object>>) phasesObj : List.of();

            // find phase 1 start date once per module
            LocalDate phase1Start = null;
            for (Map<String,Object> ph : phases) {
                if (ph == null) continue;
                String phId = YamlFiles.safeStr(ph.get("phase"));
                int phNum = YamlFiles.phaseNumFromId(phId);
                if (phNum == 1) {
                    phase1Start = LocalDate.parse(YamlFiles.safeStr(ph.get("start_date")).replace("-", "/"), YamlFiles.DF);
                    break;
                }
            }

            for (Map<String,Object> ph : phases) {
                String phId = YamlFiles.safeStr(ph.get("phase"));
                int phNum = YamlFiles.phaseNumFromId(phId);

                // use phase 1 start for every phase if it exists
                LocalDate originalStart = LocalDate.parse(YamlFiles.safeStr(ph.get("start_date")).replace("-", "/"), YamlFiles.DF);
                LocalDate pStart = (phase1Start != null) ? phase1Start : originalStart;

                LocalDate pEnd = LocalDate.parse(YamlFiles.safeStr(ph.get("end_date")).replace("-", "/"), YamlFiles.DF);

                int startId = (int) (pStart.toEpochDay() - start.toEpochDay());
                int endId   = (int) (pEnd.toEpochDay() - start.toEpochDay());

                // remember the latest end per module (even if before horizon)
                moduleLastEnd.merge(module, endId, Math::max);

                Object opsObj = ph.get("operation_task_list");
                List<Map<String,Object>> opTasks = (opsObj instanceof List) ? (List<Map<String,Object>>) opsObj : List.of();
                for (Map<String,Object> ot : opTasks) {

                    String opTaskId = YamlFiles.safeStr(ot.get("id"));
                    String opId = YamlFiles.safeStr(ot.get("operation"));
                    if (opId.isBlank()) opId = YamlFiles.safeStr(ot.get("operation_id"));
                    OpDef od = null;
                    if (workflowId != null && !workflowId.isBlank()) {
                        od = opdef.get(workflowId + "|" + opId);
                    }
                    if (od == null) {
                        od = opdef.get(opId);
                    }
                    // final fallback: safe defaults if EnvConfig doesn't define this op
                    if (od == null) {
                        od = new OpDef();
                        od.phaseId = phId;
                        od.phaseNum = phNum;
                        od.allowed = List.of(8);
                        od.min = 1;
                        od.max = 999_999;
                        // optional debug:
                        // System.out.println("[WARN] OpDef missing for workflowId=" + workflowId + " opId=" + opId + " -> default used");
                    }

                    if (!opTaskId.isBlank()) {
                        OpTaskMeta m = new OpTaskMeta();
                        m.module = module;
                        m.workflowId = workflowId;
                        m.factory = fab;
                        m.opId = opId;
                        m.phaseId = phId;
                        m.phaseNum = phNum;
                        opTaskMeta.put(opTaskId, m);
                    }

                    int workloadDays = YamlFiles.parseInt(ot.get("workload_days"), 0);
                    int workloadHoursOverride = -1;
                    if (ot.containsKey("workload_hours")) {
                        workloadHoursOverride = YamlFiles.parseInt(ot.get("workload_hours"), 0);
                    }

                    int minHeads = od.min;
                    int maxHeads = od.max;

                    if (ot.containsKey("min_worker_num")) {
                        int v = YamlFiles.parseInt(ot.get("min_worker_num"), minHeads);
                        if (v > 0) minHeads = v;
                    }
                    if (ot.containsKey("max_worker_num")) {
                        int v = YamlFiles.parseInt(ot.get("max_worker_num"), maxHeads);
                        if (v > 0) maxHeads = v;
                    }

                    if (minHeads < 1) minHeads = 1;
                    if (maxHeads < minHeads) maxHeads = minHeads;

                    int recMin = 0;
                    int recMax = 0;

                    if (ot.containsKey("recommends_worker_min")) {
                        recMin = YamlFiles.parseInt(ot.get("recommends_worker_min"), 0);
                    }
                    if (ot.containsKey("recommends_worker_max")) {
                        recMax = YamlFiles.parseInt(ot.get("recommends_worker_max"), 0);
                    }
                    // fallback / normalization
                    if (recMin <= 0 && recMax > 0) recMin = recMax;
                    if (recMax <= 0 && recMin > 0) recMax = recMin;
                    if (recMin > 0 && recMax > 0 && recMax < recMin) recMax = recMin;
                    // if (recMax > 0) {
                    //     maxHeads = Math.max(minHeads, recMax);
                    // }
                    TaskWindow tw = new TaskWindow();
                    tw.module = module; tw.workflowId = workflowId; tw.factory = fab;
                    tw.phaseId = phId; tw.phaseNum = phNum;
                    tw.opId = opId; tw.startDayId = startId; tw.endDayId = endId;
                    tw.allowed = od.allowed;
                    tw.minHeads = minHeads;
                    tw.maxHeads = maxHeads;
                    tw.workloadDays = workloadDays;
                    tw.workloadHours = workloadHoursOverride;
                    tw.recommendMinHeads = recMin;
                    tw.recommendMaxHeads = recMax;
                    windows.add(tw);
                }
            }
        }

        // ---- Determine cut-off date ----
        LocalDate cutOffDate;
        Object cutOffObj = s.get("cut_off_date");
        if (cutOffObj != null) {
            cutOffDate = LocalDate.parse(YamlFiles.safeStr(cutOffObj).replace("-", "/"), YamlFiles.DF);
        } else {
            cutOffDate = start;
        }
        int cutOffDayId = (int) (cutOffDate.toEpochDay() - start.toEpochDay());

        // ---- Decide activeModules using grace-gap x days ----
        Set<String> activeModules = new HashSet<>();
        if (SolverConstants.TRIM_FINISHED_MODULES) {
            int grace = SolverConstants.MODULE_TRIM_GRACE_DAYS;

            for (Map.Entry<String,Integer> e : moduleLastEnd.entrySet()) {
                String module = e.getKey();
                int lastEndDayId = e.getValue();
                int gap = cutOffDayId - lastEndDayId;
                if (gap <= grace) {
                    activeModules.add(module);
                }
            }

            if (activeModules.isEmpty()) {
                activeModules.addAll(moduleLastEnd.keySet());
            }

            windows.removeIf(w -> !activeModules.contains(w.module));

            required.entrySet().removeIf(e -> {
                String key = e.getKey();
                String mod = key.contains("|") ? key.substring(0, key.indexOf('|')) : key;
                return !activeModules.contains(mod);
            });
        } else {
            activeModules.addAll(moduleLastEnd.keySet());
        }

        // ---- Read fixed + flexible assignments (TWO PASS) ----

        // temp container
        class RawAssignRow {
            String module, opId, factory, wid;
            String phaseId; int phaseNum;
            String flex;                 // "fixed" / "flexible"
            int sId, eId;                // from start_date/end_date if present
            Map<Integer,Integer> byDay;   // summed hours by dayId
            boolean isRealOp;             // false for misc/dummy tasks (no OpTaskMeta match)
            String region;                // misc rows only: the misc task's own region, null if none/neutral
        }

        // misc_task_list: misc task id -> its own region (design section 7). A region listed in
        // misc_neutral_regions (default r_other) or a blank/missing one means "no region" (null).
        Set<String> neutralMiscRegions = SolveContext.current().miscNeutralRegions;
        Map<String,String> miscRegionById = new HashMap<>();
        Object miscObj = s.get("misc_task_list");
        if (miscObj instanceof List) {
            for (Object o : (List<?>) miscObj) {
                if (!(o instanceof Map)) continue;
                Map<?,?> mt = (Map<?,?>) o;
                String mid = YamlFiles.safeStr(mt.get("id"));
                String mreg = YamlFiles.safeStr(mt.get("region")).trim();
                if (mid.isBlank() || mreg.isBlank() || neutralMiscRegions.contains(mreg)) continue;
                miscRegionById.put(mid, mreg);
            }
        }

        // outputs
        List<FixedAssign> fixedRows = new ArrayList<>();
        Map<String,Integer> fixedHoursByKeyAll = new HashMap<>();

        List<FlexibleAssign> flexibleRows = new ArrayList<>();
        Map<String,Integer> flexibleHoursByKey = new HashMap<>();
        List<InitialAssign> initialRows = new ArrayList<>();

        // used later for pushing phase windows
        Map<String,Integer> latestFixedEndInRange = new HashMap<>();
        Map<String,Integer> latestFixedEndAny     = new HashMap<>();

        //  phase-wide started-before-cutoff marker
        Set<String> phaseStartedBeforeCutoff = new HashSet<>();

        List<RawAssignRow> rawRows = new ArrayList<>();

        Object asgObj = s.get("assignment_list");
        List<Map<String,Object>> asgs = (asgObj instanceof List) ? (List<Map<String,Object>>) asgObj : List.of();

        // ---------- PASS 1: parse + detect phaseStartedBeforeCutoff ----------
        for (Map<String,Object> a : asgs) {

            String flex = YamlFiles.safeStr(a.get("plan_flexibility"));
            String opTask = YamlFiles.safeStr(a.get("operation_task"));
            OpTaskMeta meta = opTaskMeta.get(opTask);

            String module, opId, phId; int phNum;

            if (meta != null) {
                module = meta.module;
                opId   = meta.opId;
                phId   = meta.phaseId;
                phNum  = meta.phaseNum;
            } else {
                ParsedOpTaskRef ref = YamlFiles.parseOpTaskRef(opTask);
                module = ref.module;
                opId   = ref.opToken;
                phId   = ref.phaseToken;
                phNum  = ref.phaseNum;
            }

            String wid = YamlFiles.safeStr(a.get("worker"));

            LocalDate sd = a.get("start_date") == null ? null :
                    LocalDate.parse(YamlFiles.safeStr(a.get("start_date")).replace("-", "/"), YamlFiles.DF);
            LocalDate ed = a.get("end_date")   == null ? null :
                    LocalDate.parse(YamlFiles.safeStr(a.get("end_date")).replace("-", "/"), YamlFiles.DF);
            int sId = (sd == null) ? -1 : (int)(sd.toEpochDay() - start.toEpochDay());
            int eId = (ed == null) ? -1 : (int)(ed.toEpochDay() - start.toEpochDay());

            // work_date_list
            String wdKey = a.containsKey("work_date_lsit") ? "work_date_lsit" : "work_date_list";
            List<Map<String,Object>> wdl =
                    (a.get(wdKey) instanceof List) ? (List<Map<String,Object>>) a.get(wdKey) : List.of();

            Map<Integer,Integer> byDay = new HashMap<>();
            boolean anyBeforeCutoff = false;

            for (Map<String,Object> item : wdl) {
                LocalDate d = LocalDate.parse(YamlFiles.safeStr(item.get("date")).replace("-", "/"), YamlFiles.DF);
                int did = (int)(d.toEpochDay() - start.toEpochDay());
                int h = YamlFiles.parseInt(item.get("hour"), 0);
                if (h <= 0) continue;

                byDay.merge(did, h, Integer::sum);

                if (did < cutOffDayId) anyBeforeCutoff = true;
            }

            // mark phase started if ANY row (fixed OR flexible) has work before cutoff
            if (anyBeforeCutoff) {
                phaseStartedBeforeCutoff.add(module + "|" + phNum);
            }

            // collect raw row
            RawAssignRow rr = new RawAssignRow();
            rr.module = module; rr.opId = opId; rr.phaseId = phId; rr.phaseNum = phNum;
            rr.factory = (meta != null) ? meta.factory : null;
            rr.wid = wid; rr.flex = flex;
            rr.sId = sId; rr.eId = eId;
            rr.byDay = byDay;
            rr.isRealOp = (meta != null);
            rr.region = (meta != null) ? null : miscRegionById.get(opTask);

            rawRows.add(rr);

            // keep "any end" tracking for true fixed rows (used for pushing phase windows)
            if ("fixed".equalsIgnoreCase(flex)) {
                latestFixedEndAny.merge(module + "|" + phNum, eId, Math::max);
            }
        }

        // ---------- PASS 2: categorize rows using phaseStartedBeforeCutoff ----------
        for (RawAssignRow rr : rawRows) {
            if (rr.byDay == null || rr.byDay.isEmpty()) continue;

            // Rows whose operation_task doesn't resolve to a real workflow op (misc/dummy
            // tasks like PTC/VISA) are always treated as blocking the worker's availability,
            // regardless of the row's own plan_flexibility — they can never be a real
            // schedulable block, so there's nothing "flexible" about them, and only reading
            // them as Fixed keeps them feeding FIXED_HOURS_BY_EMP_DAY / FIXED_TASKCOUNT_BY_EMP_DAY
            // for every day they cover (previously this depended on the data always tagging
            // them Fixed; a Flexible-tagged misc row would silently stop blocking availability
            // for any dates after the cutoff).
            boolean isFixed    = "fixed".equalsIgnoreCase(rr.flex) || !rr.isRealOp;
            boolean isFlexible = "flexible".equalsIgnoreCase(rr.flex) && rr.isRealOp;

            boolean phaseStarted = phaseStartedBeforeCutoff.contains(rr.module + "|" + rr.phaseNum);

            if (isFixed) {
                // unchanged: always fixed
                int tot = rr.byDay.values().stream().mapToInt(Integer::intValue).sum();
                if (tot > 0) fixedHoursByKeyAll.merge(rr.module + "|" + rr.opId, tot, Integer::sum);

                FixedAssign fa = new FixedAssign();
                fa.module = rr.module; fa.opId = rr.opId; fa.wid = rr.wid;
                fa.phaseId = rr.phaseId; fa.phaseNum = rr.phaseNum;
                fa.factory = rr.factory;
                fa.region = rr.region;
                fa.startDayId = rr.sId; fa.endDayId = rr.eId;
                fa.hoursByDay.putAll(rr.byDay);
                fixedRows.add(fa);

                // in-range end for pushing next phase
                for (var ent : rr.byDay.entrySet()) {
                    int did = ent.getKey();
                    int h = ent.getValue();
                    if (h <= 0) continue;
                    latestFixedEndInRange.merge(rr.module + "|" + rr.phaseNum, did, Math::max);
                }
                continue;
            }

            if (!isFlexible) continue;

            // NEW RULE:
            // If phase started before cutoff => ALL flexible rows in this phase become FIXED (even if row starts after cutoff)
            if (phaseStarted) {
                int tot = rr.byDay.values().stream().mapToInt(Integer::intValue).sum();
                if (tot > 0) fixedHoursByKeyAll.merge(rr.module + "|" + rr.opId, tot, Integer::sum);

                FixedAssign fa = new FixedAssign();
                fa.module = rr.module; fa.opId = rr.opId; fa.wid = rr.wid;
                fa.phaseId = rr.phaseId; fa.phaseNum = rr.phaseNum;
                fa.factory = rr.factory;
                fa.startDayId = rr.sId; fa.endDayId = rr.eId;
                fa.hoursByDay.putAll(rr.byDay);
                fixedRows.add(fa);

                for (var ent : rr.byDay.entrySet()) {
                    int did = ent.getKey();
                    int h = ent.getValue();
                    if (h <= 0) continue;
                    latestFixedEndInRange.merge(rr.module + "|" + rr.phaseNum, did, Math::max);
                }
                continue;
            }

            // OLD BEHAVIOR (only if phase NOT started before cutoff):
            Map<Integer,Integer> after = new HashMap<>();
            for (var ent : rr.byDay.entrySet()) {
                int did = ent.getKey();
                int h   = ent.getValue();
                if (h <= 0) continue;
                if (did >= cutOffDayId) after.put(did, h);
                else {
                    // before cutoff -> treat as fixed history
                    fixedHoursByKeyAll.merge(rr.module + "|" + rr.opId, h, Integer::sum);

                    FixedAssign fx = new FixedAssign();
                    fx.module = rr.module; fx.opId = rr.opId; fx.wid = rr.wid;
                    fx.phaseId = rr.phaseId; fx.phaseNum = rr.phaseNum;
                    fx.factory = rr.factory;
                    fx.hoursByDay.put(did, h);
                    fixedRows.add(fx);

                    latestFixedEndInRange.merge(rr.module + "|" + rr.phaseNum, did, Math::max);
                }
            }

            if (!after.isEmpty()) {
                FlexibleAssign fa = new FlexibleAssign();
                fa.module = rr.module; fa.opId = rr.opId; fa.wid = rr.wid;
                fa.phaseId = rr.phaseId; fa.phaseNum = rr.phaseNum;
                fa.factory = rr.factory;
                fa.hoursByDay.putAll(after);
                flexibleRows.add(fa);

                int totAfter = after.values().stream().mapToInt(Integer::intValue).sum();
                if (totAfter > 0) flexibleHoursByKey.merge(rr.module + "|" + rr.opId, totAfter, Integer::sum);

                InitialAssign ia = new InitialAssign();
                ia.module = rr.module; ia.opId = rr.opId;
                ia.hoursByDay.putAll(after);
                initialRows.add(ia);
            }
        }

        // IMPORTANT: assign these back to ParsedSchedule at end
        ParsedSchedule out = new ParsedSchedule();
        out.planStart = start;
        out.planEnd   = end;
        out.windows   = windows;
        out.requiredByKey = required;
        out.activeModules = activeModules;
        out.cutOffDayId = cutOffDayId;
        out.cutOffDate = cutOffDate;

// and keep  window push logic below, using latestFixedEndInRange/latestFixedEndAny

        // ---- Push phase windows based on fixed ends (even if outside horizon) ----
        for (TaskWindow w : windows) {
            int prev = w.phaseNum - 1;
            if (prev <= 0) continue;
            Integer inRangeEnd = latestFixedEndInRange.get(w.module + "|" + prev);
            Integer anyEnd     = latestFixedEndAny.get(w.module + "|" + prev);
            Integer endPrev = null;
            if (inRangeEnd != null && anyEnd != null) endPrev = Math.max(inRangeEnd, anyEnd);
            else if (inRangeEnd != null) endPrev = inRangeEnd;
            else if (anyEnd != null) endPrev = anyEnd;
            if (endPrev != null) {
                w.startDayId = Math.max(w.startDayId, endPrev + 1);
            }
        }

        for (TaskWindow w : windows) {
            if (w.startDayId > w.endDayId) {
                System.out.printf(
                    "[WARN] Collapsed window for module=%s phase=%s op=%s (startDayId=%d > endDayId=%d). " +
                    "Marking workloadDays=0 so this block is not scheduled.%n",
                    w.module, w.phaseId, w.opId, w.startDayId, w.endDayId
                );
                w.workloadDays = 0;
                w.workloadHours = 0;
            }
        }

        if (SolverConstants.TRIM_FINISHED_MODULES) {
            for (TaskWindow w : windows) {

                boolean phaseStarted =
                        phaseStartedBeforeCutoff.contains(w.module + "|" + w.phaseNum);

                if (!phaseStarted) {
                    w.startDayId = Math.max(w.startDayId, cutOffDayId);
                    if (w.startDayId > w.endDayId) {
                        w.workloadDays = 0;
                        w.workloadHours = 0;
                    }
                }

                // If phase started before cutoff → DO NOT TRIM WINDOW
            }
        }

        int firstDayId = Math.max(0, cutOffDayId);

        int lastDayId = horizon - 1;

        List<DaySlot> days = new ArrayList<>();
        for (int dayId = Math.max(0, firstDayId); dayId <= lastDayId; dayId++) {
            days.add(new DaySlot(dayId, start.plusDays(dayId)));
        }
        out.daySlots = days;

        out.cutOffDayId = cutOffDayId;
        out.cutOffDate = cutOffDate;
        // IMPORTANT: never leave these null
        out.fixedRows = fixedRows;
        out.flexibleRows = flexibleRows;
        out.fixedHoursByKeyAll = fixedHoursByKeyAll;
        out.flexibleHoursByKey = flexibleHoursByKey;
        out.initialRows = initialRows;

        // also keep these non-null
        if (out.windows == null) out.windows = new ArrayList<>();
        if (out.requiredByKey == null) out.requiredByKey = new HashMap<>();
        if (out.activeModules == null) out.activeModules = new HashSet<>();
        return out;
    }
}
