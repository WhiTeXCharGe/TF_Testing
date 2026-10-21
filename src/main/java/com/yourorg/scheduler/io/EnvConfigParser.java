package com.yourorg.scheduler.io;

import java.io.*;
import java.nio.file.*;
import java.time.LocalDate;
import java.util.*;
import java.util.stream.Collectors;

import com.yourorg.scheduler.calendar.Calendars;
import com.yourorg.scheduler.calendar.SolveContext;
import com.yourorg.scheduler.model.*;

import org.yaml.snakeyaml.LoaderOptions;
import org.yaml.snakeyaml.Yaml;
import org.yaml.snakeyaml.constructor.SafeConstructor;

/** EnvConfig.yaml parsing (workers, op definitions) and fab/region/worker calendar building. */
public class EnvConfigParser {

    @SuppressWarnings("unchecked")
    public static ParsedEnv parseEnv(String envPath) throws IOException {
        Map<String,Object> root = YamlFiles.loadYaml(envPath);
        Map<String,Object> env = (Map<String,Object>) root.getOrDefault("environment", root);

        Map<String,OpDef> opdef = new HashMap<>();
        List<Map<String,Object>> wfl = (List<Map<String,Object>>) env.getOrDefault("workflow_list", List.of());
        for (Map<String,Object> wf : wfl) {
            if (wf == null) continue;
            String wfId = YamlFiles.safeStr(wf.get("id"));

            List<Map<String,Object>> phases =
                    (List<Map<String,Object>>) wf.getOrDefault("phase_list", List.of());
            for (Map<String,Object> ph : phases) {
                if (ph == null) continue;
                String phId = YamlFiles.safeStr(ph.get("id"));
                int phNum = YamlFiles.phaseNumFromId(phId);

                List<Map<String,Object>> ops =
                        (List<Map<String,Object>>) ph.getOrDefault("operation_list", List.of());
                for (Map<String,Object> op : ops) {
                    if (op == null) continue;

                    String opId = YamlFiles.safeStr(op.get("id"));
                    List<Object> hrs = (List<Object>) op.getOrDefault("work_hours", List.of(8));
                    List<Integer> allowed = hrs.isEmpty() ? List.of(8)
                            : hrs.stream()
                                .map(x -> YamlFiles.parseInt(x, 8))
                                .sorted()
                                .collect(Collectors.toList());

                    OpDef od = new OpDef();
                    od.phaseId = phId; od.phaseNum = phNum; od.allowed = allowed;
                    od.min = YamlFiles.parseInt(op.get("min_worker_num"), 1);
                    od.max = YamlFiles.parseInt(op.get("max_worker_num"), 999_999);

                    // Backward compatibility: still allow lookup by plain opId when unique.
                    opdef.putIfAbsent(opId, od);

                    // New format safety: also allow lookup by workflowId|opId.
                    if (!wfId.isBlank()) {
                        opdef.put(wfId + "|" + opId, od);
                    }
                }
            }
        }

        // load affinity tag weights
        SolveContext.current().affinityTagWeights.clear();
        List<Map<String,Object>> affinityTagList =
                (List<Map<String,Object>>) env.getOrDefault("affinity_tag", List.of());
        for (Map<String,Object> at : affinityTagList) {
            if (at == null) continue;
            String tagId = YamlFiles.safeStr(at.get("id"));
            int tagWeight = YamlFiles.parseInt(at.get("weight"), 0);
            if (!tagId.isBlank()) SolveContext.current().affinityTagWeights.put(tagId, tagWeight);
        }

        List<EmployeeFact> employees = new ArrayList<>();
        employees.add(new EmployeeFact(0, "__UNASSIGNED__", "__UNASSIGNED__", Map.of(), false, ""));
        List<Map<String,Object>> workers = (List<Map<String,Object>>) env.getOrDefault("worker_list", List.of());
        int eid = 1;
        Map<String, EmployeeFact> byWid = new HashMap<>();
        for (Map<String,Object> w : workers) {
            String wid = YamlFiles.safeStr(w.get("id"));
            String name = YamlFiles.safeStr(w.get("name"));
            boolean isMgr = Boolean.TRUE.equals(w.get("is_manager"));
            String company = YamlFiles.safeStr(w.get("worker_company"));
            Map<String,Integer> skills = new HashMap<>();
            Map<String,Object> smap = (Map<String,Object>) w.getOrDefault("skill_map", Map.of());
            for (Map.Entry<String,Object> e : smap.entrySet()) {
                skills.put(e.getKey(), YamlFiles.parseInt(e.getValue(), 0));
            }
            EmployeeFact ef = new EmployeeFact(eid++, wid, name, skills, isMgr, company);
            employees.add(ef);
            byWid.put(wid, ef);

            //  parse fab_suitability_map (region & customer_company)
            List<Map<String,Object>> fsmList =
                    (List<Map<String,Object>>) w.getOrDefault("fab_suitability_map", List.of());
            for (Map<String,Object> fs : fsmList) {
                if (fs == null) continue;
                String kind = YamlFiles.safeStr(fs.get("kind"));
                Map<String,Object> suitRaw =
                        (Map<String,Object>) fs.getOrDefault("suitability", Map.of());
                for (Map.Entry<String,Object> se : suitRaw.entrySet()) {
                    String key = YamlFiles.safeStr(se.getKey());
                    int lvl = YamlFiles.parseInt(se.getValue(), 1); // default = 1 (neutral) if missing
                    if ("region".equals(kind)) {
                        ef.regionPreference.put(key, lvl);
                    } else if ("customer_company".equals(kind)) {
                        ef.customerPreference.put(key, lvl);
                    }
                }
            }

            Map<String,Object> typeMap = (Map<String,Object>) w.getOrDefault("worker_type_by_operation", Map.of());
            for (Map.Entry<String,Object> te : typeMap.entrySet()) {
                String opId = YamlFiles.safeStr(te.getKey());
                String workerType = YamlFiles.safeStr(te.getValue()).trim().toLowerCase(Locale.ROOT);
                if (workerType.isBlank()) workerType = "regular";
                if (!"spot".equals(workerType)) workerType = "regular";
                ef.workerTypeByOperation.put(opId, workerType);
            }

            // parse affinity tags
            Object affObj = w.get("affinity");
            if (affObj instanceof List) {
                for (Object t : (List<?>) affObj) {
                    String tag = YamlFiles.safeStr(t).trim();
                    if (!tag.isBlank()) ef.affinityTags.add(tag);
                }
            }
        }

        // capacities
        SolveContext.current().opCapacity.clear();
        Set<String> opIds = opdef.keySet().stream()
                .filter(k -> !k.contains("|"))
                .collect(Collectors.toCollection(LinkedHashSet::new));
        for (String opId : opIds) {
            int c = 0;
            for (EmployeeFact e : employees) {
                if (e.id == 0) continue;
                if (e.skills.getOrDefault(opId, 0) > 0) c++;
            }
            SolveContext.current().opCapacity.put(opId, c);
        }

        // average skills
        SolveContext.current().opAvgSkill.clear();
        for (String opId : opIds) {
            int sum = 0, cnt = 0;
            for (EmployeeFact e : employees) {
                if (e.id == 0) continue;
                int lv = e.skills.getOrDefault(opId, 0);
                if (lv > 0) { sum += lv; cnt++; }
            }
            SolveContext.current().opAvgSkill.put(opId, cnt > 0 ? (sum / (double) cnt) : 3.0);
        }

        ParsedEnv out = new ParsedEnv();
        out.opdef = opdef; out.employees = employees; out.byWid = byWid;
        return out;
    }

    @SuppressWarnings("unchecked")
    private static Set<Integer> parseUnavailableDates(Object raw,
                                                    LocalDate planStart,
                                                    LocalDate planEnd) {
        Set<Integer> off = new HashSet<>();
        if (raw == null) return off;

        int horizon = (int) (planEnd.toEpochDay() - planStart.toEpochDay()) + 1;

        // For weekly patterns we need the actual dates in horizon
        List<LocalDate> allDates = new ArrayList<>(horizon);
        for (int i = 0; i < horizon; i++) {
            allDates.add(planStart.plusDays(i));
        }

        // Helper: convert "sat", "sun", "mon" etc to DayOfWeek
        java.util.function.Function<String, java.time.DayOfWeek> parseWeekday = (s) -> {
            if (s == null) return null;
            String t = s.trim().toLowerCase(java.util.Locale.ROOT);
            switch (t) {
                case "mon": case "monday":    return java.time.DayOfWeek.MONDAY;
                case "tue": case "tuesday":   return java.time.DayOfWeek.TUESDAY;
                case "wed": case "wednesday": return java.time.DayOfWeek.WEDNESDAY;
                case "thu": case "thursday":  return java.time.DayOfWeek.THURSDAY;
                case "fri": case "friday":    return java.time.DayOfWeek.FRIDAY;
                case "sat": case "saturday":  return java.time.DayOfWeek.SATURDAY;
                case "sun": case "sunday":    return java.time.DayOfWeek.SUNDAY;
                default: return null;
            }
        };

        // ---- Normalize raw into a List<?> ----
        // Supported shapes:
        //   unavailable_dates: 2025/09/10
        //   unavailable_dates: [2025/09/10, 2025/09/11]
        //   unavailable_dates:
        //     - { single: { days: [...] } }
        //     - { weekly: { weekdays: [...] } }
        //   unavailable_dates:
        //     weekly:
        //       weekdays: [sat, sun]
        List<?> list;
        if (raw instanceof List<?> l) {
            list = l;
        } else if (raw instanceof Map<?,?> m) {
            // Top-level map like {weekly: {weekdays:[sat,sun]}} or {single:{...}}
            list = List.of(m);
        } else {
            // Single scalar date string
            try {
                Integer did = YamlFiles.dayIdFromDate(planStart, String.valueOf(raw));
                if (did != null && did >= 0 && did < horizon) {
                    off.add(did);
                }
            } catch (Exception ignore) {}
            return off;
        }

        // We support:
        // - "2025/09/10" style entries directly in the list
        // - { single: { days: [...] } }
        // - { weekly: { weekdays: [sat, sun] } }
        Set<java.time.DayOfWeek> weeklyOff = new HashSet<>();

        for (Object item : list) {
            if (item == null) continue;

            if (item instanceof Map<?,?> map) {
                Object dateObj = map.get("date");
                if (dateObj != null) {
                    try {
                        Integer did = YamlFiles.dayIdFromDate(planStart, String.valueOf(dateObj));
                        if (did != null && did >= 0 && did < horizon) {
                            off.add(did);
                        }
                    } catch (Exception ignore) {}
                }

                // --- single: { days: [...] } ---
                Object singleObj = map.get("single");
                if (singleObj instanceof Map<?,?> singleMap) {
                    Object daysObj = singleMap.get("days");
                    if (daysObj instanceof List<?> daysList) {
                        for (Object dObj : daysList) {
                            try {
                                Integer did = YamlFiles.dayIdFromDate(planStart, String.valueOf(dObj));
                                if (did != null && did >= 0 && did < horizon) {
                                    off.add(did);
                                }
                            } catch (Exception ignore) {}
                        }
                    }
                }

                // --- weekly: { weekdays: [sat, sun] } ---
                Object weeklyObj = map.get("weekly");
                if (weeklyObj instanceof Map<?,?> weeklyMap) {
                    Object wdaysObj = weeklyMap.get("weekdays");
                    if (wdaysObj instanceof List<?> wdaysList) {
                        for (Object wd : wdaysList) {
                            java.time.DayOfWeek dow = parseWeekday.apply(String.valueOf(wd));
                            if (dow != null) weeklyOff.add(dow);
                        }
                    }
                }
            } else {
                // plain string in the list, treat as single date
                try {
                    Integer did = YamlFiles.dayIdFromDate(planStart, String.valueOf(item));
                    if (did != null && did >= 0 && did < horizon) {
                        off.add(did);
                    }
                } catch (Exception ignore) {}
            }
        }

        // Expand weekly patterns into actual dayIds within the horizon
        if (!weeklyOff.isEmpty()) {
            for (int i = 0; i < horizon; i++) {
                LocalDate d = allDates.get(i);
                if (weeklyOff.contains(d.getDayOfWeek())) {
                    off.add(i);
                }
            }
        }

        return off;
    }

    @SuppressWarnings("unchecked")
    public static void buildCalendars(String envPath, LocalDate planStart, LocalDate planEnd) throws IOException {
        SolveContext.current().calendars = new Calendars();

        int horizon = (int) (planEnd.toEpochDay() - planStart.toEpochDay()) + 1;

        Map<String,Object> root;
        try (InputStream in = Files.newInputStream(Paths.get(envPath))) {
            LoaderOptions opts = new LoaderOptions();
            opts.setCodePointLimit(10 * 1024 * 1024);
            Yaml yaml = new Yaml(new SafeConstructor(opts));
            root = yaml.load(in);
        }
        Map<String,Object> env = (Map<String,Object>) root.getOrDefault("environment", root);

        List<Map<String,Object>> fabs =
                (List<Map<String,Object>>) env.getOrDefault("fab_list", List.of());
        for (Map<String,Object> f : fabs) {
            String fid = String.valueOf(f.get("id"));
            String rid = String.valueOf(f.get("region"));
            String cid = String.valueOf(f.get("customer_company"));
            SolveContext.current().calendars.fabToRegion.put(fid, rid);
            SolveContext.current().calendars.fabToCustomer.put(fid, cid);

            Set<Integer> off = parseUnavailableDates(
                    f.get("unavailable_dates"), planStart, planEnd);
            SolveContext.current().calendars.fabOff.put(fid, off);
        }

        List<Map<String,Object>> custs =
                (List<Map<String,Object>>) env.getOrDefault("customer_company_list", List.of());
        for (Map<String,Object> c : custs) {
            String cid = String.valueOf(c.get("id"));
            Set<Integer> off = parseUnavailableDates(
                    c.get("unavailable_dates"), planStart, planEnd);
            SolveContext.current().calendars.customerOff.put(cid, off);
        }

        List<Map<String,Object>> wcomps =
                (List<Map<String,Object>>) env.getOrDefault("worker_company_list", List.of());
        for (Map<String,Object> wc : wcomps) {
            String cid = String.valueOf(wc.get("id"));

            Set<Integer> off = parseUnavailableDates(
                    wc.get("unavailable_dates"), planStart, planEnd);
            SolveContext.current().calendars.workerCompanyOff.put(cid, off);

            int annualOt  = YamlFiles.parseInt(wc.get("annual_overtime_limit"), Integer.MAX_VALUE);
            int monthlyOt = YamlFiles.parseInt(wc.get("monthly_overtime_limit"), Integer.MAX_VALUE);
            SolveContext.current().calendars.workerCompanyAnnualOtLimit.put(cid, annualOt);
            SolveContext.current().calendars.workerCompanyMonthlyOtLimit.put(cid, monthlyOt);
        }

        List<Map<String,Object>> workers =
                (List<Map<String,Object>>) env.getOrDefault("worker_list", List.of());
        for (Map<String,Object> w : workers) {
            String wid = String.valueOf(w.get("id"));
            Set<Integer> off = parseUnavailableDates(
                    w.get("unavailable_dates"), planStart, planEnd);
            SolveContext.current().calendars.workerOffByWid.put(wid, off);
        }

        List<Map<String,Object>> tmap = (List<Map<String,Object>>) env.getOrDefault("transite_day_map", List.of());
        for (Map<String,Object> t : tmap) {
            String from = String.valueOf(t.get("from"));
            String to   = String.valueOf(t.get("to"));
            int days    = YamlFiles.parseInt(t.get("days"), 0);
            if (from != null && to != null && !from.isBlank() && !to.isBlank() && days > 0) {
                SolveContext.current().calendars.transitDays.computeIfAbsent(from, k -> new HashMap<>()).put(to, days);
            }
        }

        List<Map<String,Object>> regionList =
                (List<Map<String,Object>>) env.getOrDefault("region_list", List.of());
        for (Map<String,Object> r : regionList) {
            String rid = String.valueOf(r.get("id"));

            // region-level unavailable dates (weekly + single supported)
            Set<Integer> off = parseUnavailableDates(
                    r.get("unavailable_dates"), planStart, planEnd);
            SolveContext.current().calendars.regionOff.put(rid, off);

            int maxStayOn   = YamlFiles.parseInt(r.get("max_stay_on"), Integer.MAX_VALUE);
            int maxAnnual   = YamlFiles.parseInt(r.get("max_annual_stay"), Integer.MAX_VALUE);
            int offInterval = YamlFiles.parseInt(r.get("stay_off_interval"), 1);

            SolveContext.current().calendars.regionStayMaxOn.put(rid, maxStayOn);
            SolveContext.current().calendars.regionAnnualMaxStay.put(rid, maxAnnual);
            SolveContext.current().calendars.regionStayOffInterval.put(rid, Math.max(1, offInterval));
        }

    }
}
