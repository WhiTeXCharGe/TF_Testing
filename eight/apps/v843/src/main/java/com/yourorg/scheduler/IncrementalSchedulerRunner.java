package com.yourorg.scheduler;

import org.yaml.snakeyaml.Yaml;
import org.yaml.snakeyaml.constructor.SafeConstructor;
import org.yaml.snakeyaml.DumperOptions;
import org.yaml.snakeyaml.LoaderOptions;

import java.io.*;
import java.nio.file.*;
import java.time.DayOfWeek;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.*;

import static com.yourorg.scheduler.IncrementalConfig.*;

/**
 * Java version of daily_run.py + update_schedule2.py
 *
 * - Incremental scheduler driver with "evaluate every X working days"
 *   and cutoff = (last module start + 1 working day).
 *
 * - Works purely in Java:
 *    * loads EnvConfig.yaml / Schedule.yaml
 *    * updates workers, assignments, modules, plan_range
 *    * runs EmployeeSchedule.main(...) when needed
 *    * writes snapshots: src/main/resource/schedule_outputs/Schedule_YYYYMMDD.yaml
 *
 * NOTE:
 *  - You MUST fill the CONFIG section to match config_base.py and update_config.py.
 *  - This code treats YAML as Map<String,Object> (no domain POJOs).
 */
public class IncrementalSchedulerRunner {

    private static final DateTimeFormatter YMD = DateTimeFormatter.ofPattern("yyyy/MM/dd");

    private static String ymd(LocalDate d) {
        return d.format(YMD);
    }

    // ------------------------------------------------------------------
    // Region / fab holiday calendar
    // ------------------------------------------------------------------

    private static class HolidayCalendar {
        final Map<String, String> fabToRegionId;
        final Map<String, Set<Integer>> regionWeeklyDows;
        final Map<String, Set<LocalDate>> regionSingleDates;
        final Set<Integer> globalWeeklyDows;
        final Set<LocalDate> globalSingleDates;

        HolidayCalendar(Map<String, String> fabToRegionId,
                        Map<String, Set<Integer>> regionWeeklyDows,
                        Map<String, Set<LocalDate>> regionSingleDates,
                        Set<Integer> globalWeeklyDows,
                        Set<LocalDate> globalSingleDates) {
            this.fabToRegionId = fabToRegionId;
            this.regionWeeklyDows = regionWeeklyDows;
            this.regionSingleDates = regionSingleDates;
            this.globalWeeklyDows = globalWeeklyDows;
            this.globalSingleDates = globalSingleDates;
        }
    }

    @SuppressWarnings("unchecked")
    private static HolidayCalendar buildHolidayCalendar(Map<String, Object> envRoot) {
        Map<String, Object> env = getOrCreateMap(envRoot, "environment");

        Map<String, String> fabToRegionId = new HashMap<>();
        Map<String, Set<Integer>> regionWeekly = new HashMap<>();
        Map<String, Set<LocalDate>> regionSingle = new HashMap<>();
        Set<Integer> globalWeekly = new HashSet<>();
        Set<LocalDate> globalSingle = new HashSet<>();

        // region_list -> unavailable_dates
        List<Map<String, Object>> regionList =
                (List<Map<String, Object>>) env.getOrDefault("region_list", new ArrayList<>());
        for (Map<String, Object> r : regionList) {
            Object idObj = r.get("id");
            if (idObj == null) continue;
            String regionId = idObj.toString();

            Object unavailObj = r.get("unavailable_dates");
            if (!(unavailObj instanceof List<?> unavailListRaw)) continue;

            for (Object o : unavailListRaw) {
                if (!(o instanceof Map<?, ?> m)) continue;

                if (m.containsKey("weekly")) {
                    Map<String, Object> weekly = (Map<String, Object>) m.get("weekly");
                    Object wListObj = weekly.get("weekdays");
                    if (wListObj instanceof List<?> wList) {
                        for (Object w : wList) {
                            if (w == null) continue;
                            int dow = weekdayCodeToInt(w.toString());
                            regionWeekly
                                    .computeIfAbsent(regionId, k -> new HashSet<>())
                                    .add(dow);
                            globalWeekly.add(dow);
                        }
                    }
                }

                if (m.containsKey("single")) {
                    Map<String, Object> single = (Map<String, Object>) m.get("single");
                    Object dListObj = single.get("days");
                    if (dListObj instanceof List<?> dList) {
                        for (Object d : dList) {
                            if (d == null) continue;
                            LocalDate ld = parseDate(d);
                            regionSingle
                                    .computeIfAbsent(regionId, k -> new HashSet<>())
                                    .add(ld);
                            globalSingle.add(ld);
                        }
                    }
                }
            }
        }

        // If no unavailable_dates are defined at all, fallback to weekend skip
        if (globalWeekly.isEmpty() && globalSingle.isEmpty() && IS_SKIP_WEEKEND) {
            globalWeekly.add(DayOfWeek.SATURDAY.getValue());
            globalWeekly.add(DayOfWeek.SUNDAY.getValue());
        }

        // fab_list -> region
        List<Map<String, Object>> fabList =
                (List<Map<String, Object>>) env.getOrDefault("fab_list", new ArrayList<>());
        for (Map<String, Object> f : fabList) {
            Object fid = f.get("id");
            Object reg = f.get("region");
            if (fid != null && reg != null) {
                fabToRegionId.put(fid.toString(), reg.toString());
            }
        }

        return new HolidayCalendar(fabToRegionId, regionWeekly, regionSingle, globalWeekly, globalSingle);
    }

    private static int weekdayCodeToInt(String code) {
        String c = code.toLowerCase(Locale.ROOT);
        return switch (c) {
            case "mon", "月" -> DayOfWeek.MONDAY.getValue();
            case "tue", "tues", "火" -> DayOfWeek.TUESDAY.getValue();
            case "wed", "水" -> DayOfWeek.WEDNESDAY.getValue();
            case "thu", "thur", "thu.", "木" -> DayOfWeek.THURSDAY.getValue();
            case "fri", "金" -> DayOfWeek.FRIDAY.getValue();
            case "sat", "土" -> DayOfWeek.SATURDAY.getValue();
            case "sun", "日" -> DayOfWeek.SUNDAY.getValue();
            default -> throw new IllegalArgumentException("Unknown weekday code: " + code);
        };
    }

    private static boolean isGlobalHoliday(LocalDate d, HolidayCalendar hc) {
        int dow = d.getDayOfWeek().getValue();
        if (hc.globalWeeklyDows.contains(dow)) return true;
        if (hc.globalSingleDates.contains(d)) return true;
        return false;
    }

    private static boolean isFabHoliday(LocalDate d, String fabId, HolidayCalendar hc) {
        if (fabId == null || hc == null) {
            return isGlobalHoliday(d, hc);
        }
        String regionId = hc.fabToRegionId.get(fabId);
        if (regionId == null) {
            return isGlobalHoliday(d, hc);
        }
        Set<Integer> wset = hc.regionWeeklyDows.get(regionId);
        Set<LocalDate> sset = hc.regionSingleDates.get(regionId);
        int dow = d.getDayOfWeek().getValue();
        if (wset != null && wset.contains(dow)) return true;
        if (sset != null && sset.contains(d)) return true;
        // fallback to global holidays
        return isGlobalHoliday(d, hc);
    }

    private static LocalDate nextGlobalWorkingDay(LocalDate d, HolidayCalendar hc) {
        LocalDate nd = d.plusDays(1);
        while (isGlobalHoliday(nd, hc)) {
            nd = nd.plusDays(1);
        }
        return nd;
    }

    private static LocalDate addRegionWorkingDays(LocalDate start, int days, String fabId, HolidayCalendar hc) {
        LocalDate current = start;
        int remaining = days;
        while (remaining > 0) {
            current = current.plusDays(1);
            if (!isFabHoliday(current, fabId, hc)) {
                remaining--;
            }
        }
        return current;
    }

    private static LocalDate parseDate(Object s) {
        if (s == null) return null;
        String str = s.toString().trim().replace("-", "/");
        return LocalDate.parse(str, YMD);
    }

    // YAML load/save
    private static Map<String, Object> loadYaml(Path path) throws IOException {
        if (!Files.exists(path)) {
            return new LinkedHashMap<>();
        }
        try (InputStream in = Files.newInputStream(path)) {
            LoaderOptions opts = new LoaderOptions();
            // default is 3_000_000; raise to about 5 MB
            opts.setCodePointLimit(5 * 1024 * 1024);  // ≈ 5 MB

            Yaml yaml = new Yaml(new SafeConstructor(opts));
            Object obj = yaml.load(in);
            if (obj instanceof Map) {
                return (Map<String, Object>) obj;
            }
            return new LinkedHashMap<>();
        }
    }

    private static void saveYaml(Path path, Map<String, Object> root) throws IOException {
        DumperOptions opts = new DumperOptions();
        opts.setDefaultFlowStyle(DumperOptions.FlowStyle.BLOCK);
        opts.setPrettyFlow(true);
        opts.setDefaultScalarStyle(DumperOptions.ScalarStyle.PLAIN);
        Yaml yaml = new Yaml(opts);
        try (Writer w = Files.newBufferedWriter(path)) {
            yaml.dump(root, w);
        }
    }

    private static void backupFile(Path path) {
        if (!Files.exists(path)) return;
        Path bak = Paths.get(path.toString() + ".bak");
        try {
            Files.copy(path, bak, StandardCopyOption.REPLACE_EXISTING);
            System.out.println("Backup written: " + bak);
        } catch (IOException e) {
            System.out.println("Warning: failed to write backup: " + e.getMessage());
        }
    }

    @SuppressWarnings("unchecked")
    private static boolean hasExistingAssignments(Map<String, Object> root) {
        Object schedObj = root.get("schedule");
        Map<String, Object> sched;
        if (schedObj instanceof Map) {
            sched = (Map<String, Object>) schedObj;
        } else {
            sched = root;
        }

        Object assignmentObj = sched.get("assignment_list");
        if (!(assignmentObj instanceof List)) {
            return false;
        }
        return !((List<?>) assignmentObj).isEmpty();
    }

    @SuppressWarnings("unchecked")
    static boolean isInitialRun(Map<String, Object> schedRoot) {
        Map<String, Object> sched = (Map<String, Object>) schedRoot.get("schedule");
        Object alObj = sched.get("assignment_list");
        if (!(alObj instanceof List<?> list)) return true;
        return list.isEmpty();  // initial if empty
    }

    // ------------------------------------------------------------------
    // update_schedule2.py equivalents
    // ------------------------------------------------------------------

    private static String idxToWorkerName(int idx) {
        return "" + (char) ('A' + idx / 26) + (char) ('A' + idx % 26);
    }

    @SuppressWarnings("unchecked")
    private static int[] extendWorkersIfNeeded(Map<String, Object> envRoot) {
        Map<String, Object> env = getOrCreateMap(envRoot, "environment");

        List<Map<String, Object>> workerList =
                (List<Map<String, Object>>) env.getOrDefault("worker_list", new ArrayList<>());
        int beforeN = workerList.size();
        int targetN = WORKER_NUM;
        if (beforeN >= targetN) {
            env.put("worker_list", workerList);
            return new int[]{beforeN, 0};
        }

        List<Map<String, Object>> wfList =
                (List<Map<String, Object>>) env.getOrDefault("workflow_list", new ArrayList<>());
        if (wfList.isEmpty()) {
            throw new RuntimeException("EnvConfig has no workflow_list");
        }
        Map<String, Object> workflow = wfList.get(0);

        List<String> ops = new ArrayList<>();
        List<Map<String, Object>> phaseList =
                (List<Map<String, Object>>) workflow.getOrDefault("phase_list", new ArrayList<>());
        for (Map<String, Object> ph : phaseList) {
            List<Map<String, Object>> opList =
                    (List<Map<String, Object>>) ph.getOrDefault("operation_list", new ArrayList<>());
            for (Map<String, Object> op : opList) {
                Object opId = op.get("id");
                if (opId != null) ops.add(opId.toString());
            }
        }
        if (ops.isEmpty()) throw new RuntimeException("EnvConfig workflow has no operations");

        List<Map<String, Object>> companyList =
                (List<Map<String, Object>>) env.getOrDefault("worker_company_list", new ArrayList<>());
        if (companyList.isEmpty()) throw new RuntimeException("EnvConfig has no worker_company_list");

        // NEW: region IDs and customer_company IDs for suitability
        List<Map<String, Object>> regionList =
                (List<Map<String, Object>>) env.getOrDefault("region_list", new ArrayList<>());
        if (regionList.isEmpty()) throw new RuntimeException("EnvConfig has no region_list");
        List<String> regionIds = new ArrayList<>();
        for (Map<String, Object> r : regionList) {
            Object id = r.get("id");
            if (id != null) regionIds.add(id.toString());
        }

        List<Map<String, Object>> customerList =
                (List<Map<String, Object>>) env.getOrDefault("customer_company_list", new ArrayList<>());
        if (customerList.isEmpty()) throw new RuntimeException("EnvConfig has no customer_company_list");
        List<String> customerIds = new ArrayList<>();
        for (Map<String, Object> c : customerList) {
            Object id = c.get("id");
            if (id != null) customerIds.add(id.toString());
        }

        int added = 0;
        int skillMin = 3, skillMax = 6;

        Random rng = new Random(ENV_SEED);

        for (int i = beforeN; i < targetN; i++) {
            String wid = "w" + (i + 1);
            String wname = idxToWorkerName(i);

            int skillNum = skillMin + rng.nextInt(skillMax - skillMin + 1);
            List<Integer> indices = new ArrayList<>();
            for (int k = 0; k < ops.size(); k++) indices.add(k);
            Collections.shuffle(indices, rng);
            indices = indices.subList(0, skillNum);
            Collections.sort(indices);

            List<String> skillIds = new ArrayList<>();
            for (int idx : indices) skillIds.add(ops.get(idx));

            List<Integer> skillLevels = new ArrayList<>();
            for (int k = 0; k < skillIds.size(); k++) {
                skillLevels.add(weightedChoiceInt(rng, SKILL_LEVEL_LIST, SKILL_LEVEL_WEIGHTS));
            }

            Map<String, Object> company = companyList.get(rng.nextInt(companyList.size()));
            boolean isManager = weightedChoiceBool(rng, MANAGER_RATE);

            Map<String, Object> skillMap = new LinkedHashMap<>();
            for (int k = 0; k < skillIds.size(); k++) {
                skillMap.put(skillIds.get(k), skillLevels.get(k));
            }

            // NEW: region suitability
            Map<String, Object> regionSuitability = new LinkedHashMap<>();
            for (String rid : regionIds) {
                int lvl = weightedChoiceInt(rng, REGION_SUITABILITY_LIST, REGION_SUITABILITY_WEIGHTS);
                regionSuitability.put(rid, lvl);
            }

            // NEW: customer_company suitability
            Map<String, Object> customerSuitability = new LinkedHashMap<>();
            for (String cid : customerIds) {
                int lvl = weightedChoiceInt(rng, CUSTOMER_SUITABILITY_LIST, CUSTOMER_SUITABILITY_WEIGHTS);
                customerSuitability.put(cid, lvl);
            }

            List<Map<String, Object>> fabSuitabilityMap = new ArrayList<>();
            Map<String, Object> regionEntry = new LinkedHashMap<>();
            regionEntry.put("kind", "region");
            regionEntry.put("suitability", regionSuitability);
            fabSuitabilityMap.add(regionEntry);

            Map<String, Object> customerEntry = new LinkedHashMap<>();
            customerEntry.put("kind", "customer_company");
            customerEntry.put("suitability", customerSuitability);
            fabSuitabilityMap.add(customerEntry);

            Map<String, Object> worker = new LinkedHashMap<>();
            worker.put("id", wid);
            worker.put("name", wname);
            worker.put("worker_company", company.get("id"));
            worker.put("is_manager", isManager);
            worker.put("skill_map", skillMap);
            worker.put("fab_suitability_map", fabSuitabilityMap);
            worker.put("unavailable_dates", new ArrayList<>());

            workerList.add(worker);
            added++;
        }

        env.put("worker_list", workerList);
        return new int[]{beforeN, added};
    }

    private static int weightedChoiceInt(Random rng, List<Integer> values, List<Double> weights) {
        double sum = 0.0;
        for (double w : weights) sum += w;
        double r = rng.nextDouble() * sum;
        double acc = 0.0;
        for (int i = 0; i < values.size(); i++) {
            acc += weights.get(i);
            if (r <= acc) return values.get(i);
        }
        return values.get(values.size() - 1);
    }

    private static boolean weightedChoiceBool(Random rng, double trueProb) {
        return rng.nextDouble() < trueProb;
    }

    @SuppressWarnings("unchecked")
    private static List<String> collectFabIds(Map<String, Object> envRoot) {
        Map<String, Object> env = getOrCreateMap(envRoot, "environment");
        List<Map<String, Object>> fabList =
                (List<Map<String, Object>>) env.getOrDefault("fab_list", new ArrayList<>());
        List<String> ids = new ArrayList<>();
        for (Map<String, Object> f : fabList) {
            Object id = f.get("id");
            if (id != null) ids.add(id.toString());
        }
        if (ids.isEmpty()) throw new RuntimeException("EnvConfig has no fab_list ids");
        return ids;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> pickWorkflow(Map<String, Object> envRoot) {
        Map<String, Object> env = getOrCreateMap(envRoot, "environment");
        List<Map<String, Object>> wfList =
                (List<Map<String, Object>>) env.getOrDefault("workflow_list", new ArrayList<>());
        if (wfList.isEmpty()) throw new RuntimeException("EnvConfig has no workflow_list");
        return wfList.get(0);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> buildPhaseTask(
            String eqId,
            Map<String, Object> phaseDict,
            List<Integer> opWorklengths,
            LocalDate startDay,
            LocalDate endDay
    ) {
        String phaseId = String.valueOf(phaseDict.get("id"));
        String phaseName = String.valueOf(
                phaseDict.getOrDefault("name", phaseId)
        );
        List<Map<String, Object>> ops =
                (List<Map<String, Object>>) phaseDict.getOrDefault("operation_list", new ArrayList<>());

        List<Map<String, Object>> opTaskList = new ArrayList<>();
        for (int i = 0; i < ops.size(); i++) {
            Map<String, Object> op = ops.get(i);
            int wl = opWorklengths.get(i);
            String opId = String.valueOf(op.get("id"));
            String opName = String.valueOf(op.getOrDefault("name", opId));

            Map<String, Object> opTask = new LinkedHashMap<>();
            opTask.put("id", eqId + opId);
            opTask.put("name", opName);
            opTask.put("operation", opId);
            opTask.put("workload_days", wl);
            opTaskList.add(opTask);
        }

        Map<String, Object> phaseTask = new LinkedHashMap<>();
        phaseTask.put("id", eqId + phaseId);
        phaseTask.put("name", phaseName);
        phaseTask.put("phase", phaseId);
        phaseTask.put("start_date", ymd(startDay));
        phaseTask.put("end_date", ymd(endDay));
        phaseTask.put("operation_task_list", opTaskList);

        return phaseTask;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> buildOneModule(
            int eqIndex,
            Map<String, Object> workflow,
            String fabId,
            LocalDate startDay,
            List<Object> worklength,
            HolidayCalendar holiday
    ) {
        String eqId = "e" + (eqIndex + 1);
        String name = "SU " + (1000 + eqIndex + 1) + "A";

        List<Map<String, Object>> phaseList =
                (List<Map<String, Object>>) workflow.getOrDefault("phase_list", new ArrayList<>());
        if (phaseList.size() != worklength.size()) {
            throw new RuntimeException("worklength length " + worklength.size()
                    + " does not match number of phases " + phaseList.size());
        }

        // Move module_start to the first working day for this fab's region
        LocalDate moduleStart = startDay;
        while (isFabHoliday(moduleStart, fabId, holiday)) {
            moduleStart = moduleStart.plusDays(1);
        }

        int cumulativeDays = 0;
        LocalDate finalEnd = moduleStart;
        List<Map<String, Object>> phaseTaskList = new ArrayList<>();

        for (int i = 0; i < phaseList.size(); i++) {
            Map<String, Object> ph = phaseList.get(i);
            List<Object> tuple = (List<Object>) worklength.get(i);
            int phaseDays = ((Number) tuple.get(0)).intValue();
            List<Integer> opWls = (List<Integer>) tuple.get(1);

            cumulativeDays += phaseDays;

            // inclusive working-day window: start = moduleStart, length = cumulativeDays
            LocalDate phaseEnd;
            if (cumulativeDays <= 1) {
                phaseEnd = moduleStart;
            } else {
                phaseEnd = addRegionWorkingDays(moduleStart, cumulativeDays - 1, fabId, holiday);
            }

            Map<String, Object> phaseTask =
                    buildPhaseTask(eqId, ph, opWls, moduleStart, phaseEnd);
            phaseTaskList.add(phaseTask);

            if (phaseEnd.isAfter(finalEnd)) finalEnd = phaseEnd;
        }

        Map<String, Object> eqDict = new LinkedHashMap<>();
        eqDict.put("id", eqId);
        eqDict.put("name", name);
        eqDict.put("workflow", String.valueOf(workflow.getOrDefault("id", "workflow")));
        eqDict.put("fab", fabId);
        eqDict.put("phase_task_list", phaseTaskList);
        eqDict.put("__END_DATE", finalEnd);
        return eqDict;
    }

    private static List<List<Object>> createWorklengthList() {
        // Python: return ([normal_worklength, vip_worklength], [0.8, 0.2])
        List<List<Object>> out = new ArrayList<>();
        out.add((List<Object>) (List<?>) NORMAL_WORKLENGTH);
        out.add((List<Object>) (List<?>) VIP_WORKLENGTH);
        return out;
    }

    private static int weightedIndex(Random rng, double[] weights) {
        double sum = 0;
        for (double w : weights) sum += w;
        double r = rng.nextDouble() * sum;
        double acc = 0;
        for (int i = 0; i < weights.length; i++) {
            acc += weights[i];
            if (r <= acc) return i;
        }
        return weights.length - 1;
    }

    private static List<Map<String, Object>> createNewModules(
            Map<String, Object> workflow,
            List<String> fabIds,
            int startIndex,
            int numToAdd,
            LocalDate startDay,
            LocalDate[] lastEndHolder,
            int moduleSeedOffset,
            HolidayCalendar holiday
    ) {
        if (numToAdd <= 0) {
            lastEndHolder[0] = startDay;
            return Collections.emptyList();
        }

        List<List<Object>> worklengthList = createWorklengthList();
        double[] worklengthWeights = new double[]{0.8, 0.2};

        Random rng = new Random(MODULE_SEED + moduleSeedOffset);
        List<Map<String, Object>> modules = new ArrayList<>();
        LocalDate lastEnd = startDay;

        for (int offset = 0; offset < numToAdd; offset++) {
            int idx = weightedIndex(rng, worklengthWeights);
            List<Object> worklength = worklengthList.get(idx);
            String fabId = fabIds.get(rng.nextInt(fabIds.size()));

            int eqIdx = startIndex + offset;
            Map<String, Object> eqDict =
                    buildOneModule(eqIdx, workflow, fabId, startDay, worklength, holiday);
            LocalDate eqEnd = (LocalDate) eqDict.remove("__END_DATE");
            modules.add(eqDict);
            if (eqEnd.isAfter(lastEnd)) lastEnd = eqEnd;
        }

        lastEndHolder[0] = lastEnd;
        return modules;
    }

    @SuppressWarnings("unchecked")
    private static int[] updateAssignments(Map<String, Object> schedRoot, LocalDate cutoff) {
        Map<String, Object> sched = getOrCreateMap(schedRoot, "schedule");
        List<Map<String, Object>> assignments =
                (List<Map<String, Object>>) sched.getOrDefault("assignment_list", new ArrayList<>());

        int changedFixed = 0;
        int total = 0;

        for (Map<String, Object> a : assignments) {
            total++;
            Object sdRaw = a.get("start_date");
            if (sdRaw == null) {
                String wdKey = a.containsKey("work_date_lsit") ? "work_date_lsit" : "work_date_list";
                List<Map<String, Object>> wdList =
                        (List<Map<String, Object>>) a.getOrDefault(wdKey, new ArrayList<>());
                if (!wdList.isEmpty()) {
                    sdRaw = wdList.get(0).get("date");
                }
            }
            if (sdRaw == null) continue;
            LocalDate sd;
            try {
                sd = parseDate(sdRaw);
            } catch (Exception e) {
                continue;
            }

            if (sd.isBefore(cutoff)) {
                if (!"Fixed".equals(a.get("plan_flexibility"))) {
                    a.put("plan_flexibility", "Fixed");
                    changedFixed++;
                }
            } else {
                if (!"Flexible".equals(a.get("plan_flexibility"))) {
                    a.put("plan_flexibility", "Flexible");
                }
            }
        }

        sched.put("assignment_list", assignments);
        return new int[]{total, changedFixed};
    }

    @SuppressWarnings("unchecked")
    private static Triple<List<Map<String, Object>>, LocalDate, LocalDate> collectExistingModules(
            Map<String, Object> schedRoot
    ) {
        Map<String, Object> sched = getOrCreateMap(schedRoot, "schedule");
        List<Map<String, Object>> wfList =
                (List<Map<String, Object>>) sched.getOrDefault("workflow_task_list", new ArrayList<>());
        List<Map<String, Object>> modules = new ArrayList<>();

        LocalDate lastStart = null;
        LocalDate lastEnd = null;

        for (Map<String, Object> mod : wfList) {
            Object midObj = mod.get("id");
            if (midObj == null) continue;
            String mid = midObj.toString();
            if (!(mid.startsWith("e") && mid.substring(1).matches("\\d+"))) continue;
            modules.add(mod);

            LocalDate modStart = null;
            LocalDate modEnd = null;

            List<Map<String, Object>> phaseTaskList =
                    (List<Map<String, Object>>) mod.getOrDefault("phase_task_list", new ArrayList<>());
            for (Map<String, Object> ph : phaseTaskList) {
                LocalDate s = parseDate(ph.get("start_date"));
                LocalDate e = parseDate(ph.get("end_date"));
                if (s != null) {
                    if (modStart == null || s.isBefore(modStart)) modStart = s;
                }
                if (e != null) {
                    if (modEnd == null || e.isAfter(modEnd)) modEnd = e;
                }
            }

            if (modStart != null) {
                if (lastStart == null || modStart.isAfter(lastStart)) lastStart = modStart;
            }
            if (modEnd != null) {
                if (lastEnd == null || modEnd.isAfter(lastEnd)) lastEnd = modEnd;
            }
        }

        modules.sort(Comparator.comparingInt(m -> {
            Object id = m.get("id");
            if (id == null) return 0;
            String s = id.toString();
            if (s.startsWith("e") && s.substring(1).matches("\\d+")) {
                return Integer.parseInt(s.substring(1));
            }
            return 0;
        }));

        return new Triple<>(modules, lastStart, lastEnd);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> getOrCreateMap(Map<String, Object> root, String key) {
        Object v = root.get(key);
        if (v instanceof Map) {
            return (Map<String, Object>) v;
        }
        Map<String, Object> m = new LinkedHashMap<>();
        root.put(key, m);
        return m;
    }

    // ------------------------------------------------------------------
    // MAIN LOOP (Java version of daily_run.py + update_schedule2.py)
    // ------------------------------------------------------------------

    public static void main(String[] args) throws Exception {
        Path projectRoot = findProjectRoot(Paths.get("").toAbsolutePath());

        Path envPath = projectRoot.resolve(ENV_PATH);
        Path schedPath = projectRoot.resolve(SCHEDULE_IN_PATH);
        Path schedOutPath = projectRoot.resolve(SCHEDULE_OUT_PATH);

        Path outDir = projectRoot.resolve("src/main/resource/schedule_outputs");
        Files.createDirectories(outDir);

        if (!Files.exists(envPath) || !Files.exists(schedPath)) {
            throw new IllegalStateException(
                    "Expected " + envPath + " and " + schedPath + " to exist."
            );
        }

        // Load initial schedule to get plan_range
        Map<String, Object> envRoot = loadYaml(envPath);
        HolidayCalendar holiday = buildHolidayCalendar(envRoot);

        Map<String, Object> schedRoot = loadYaml(schedPath);
        Map<String, Object> sched = getOrCreateMap(schedRoot, "schedule");
        Map<String, Object> planRange =
                (Map<String, Object>) sched.getOrDefault("plan_range", new LinkedHashMap<>());

        if (!planRange.containsKey("start_date") || !planRange.containsKey("end_date")) {
            throw new IllegalStateException("schedule.plan_range.start_date/end_date not found in Schedule.yaml");
        }

        LocalDate planStart = parseDate(planRange.get("start_date"));
        LocalDate planEnd = parseDate(planRange.get("end_date"));

        Triple<List<Map<String, Object>>, LocalDate, LocalDate> initialModTriple =
                collectExistingModules(schedRoot);
        LocalDate lastStart0 = initialModTriple.second;

        int stepDays = Math.max(1, EQ_EVAL_DAYS);

        // FIRST day of the first block = cutoff0 (first working day after lastStart0, or planStart)
        LocalDate cutoff0;
        if (lastStart0 != null) {
            cutoff0 = nextGlobalWorkingDay(lastStart0, holiday);
        } else {
            cutoff0 = planStart;
        }
        // ensure cutoff0 itself is not holiday
        while (isGlobalHoliday(cutoff0, holiday)) {
            cutoff0 = cutoff0.plusDays(1);
        }

        System.out.println("[INFO] Initial plan_range: " + planStart + " .. " + planEnd);
        System.out.println("[INFO] Lookback is DISABLED (no trimming).");

        // first block starts at cutoff0
        LocalDate current = cutoff0;
        int evalIndex = 0;

        while (true) {
            // 1) Build this block's working days [blockStart .. blockEnd] using global holidays
            //    (NO limit by initial planEnd; we keep going until module target is reached)
            List<LocalDate> blockDays = new ArrayList<>();
            LocalDate d = current;
            while (blockDays.size() < stepDays) {
                if (!isGlobalHoliday(d, holiday)) {
                    blockDays.add(d);
                }
                d = d.plusDays(1);
            }

            // stepDays >= 1, so blockDays should never be empty
            LocalDate blockStart = blockDays.get(0);
            LocalDate blockEnd   = blockDays.get(blockDays.size() - 1);

            // 2) Load latest Env & Schedule each BLOCK
            envRoot = loadYaml(envPath);
            holiday = buildHolidayCalendar(envRoot); // rebuild in case Env changed
            schedRoot = loadYaml(schedPath);
            sched = getOrCreateMap(schedRoot, "schedule");

            Triple<List<Map<String, Object>>, LocalDate, LocalDate> triple =
                    collectExistingModules(schedRoot);
            List<Map<String, Object>> modulesBefore = triple.first;
            LocalDate lastStartBefore = triple.second;
            LocalDate lastEndBefore   = triple.third;
            int beforeCount = modulesBefore.size();

            LocalDate cutoff = (lastStartBefore != null)
                    ? nextGlobalWorkingDay(lastStartBefore, holiday)
                    : planStart;

            System.out.println("\n==============================================");
            System.out.println("[BLOCK] " + blockStart + " .. " + blockEnd +
                    "  | cutoff=" + ymd(cutoff) +
                    "  | modules_before=" + beforeCount);
            System.out.println("==============================================");

            // 3) update_schedule2.main() equivalent (workers, assignments, modules)
            int[] workerCounts = extendWorkersIfNeeded(envRoot);
            int beforeWorkers  = workerCounts[0];
            int addedWorkers   = workerCounts[1];
            int afterWorkers   = beforeWorkers + addedWorkers;

            int[] assignCounts = updateAssignments(schedRoot, cutoff);
            int totalAssign    = assignCounts[0];
            int changedFixed   = assignCounts[1];

            List<Map<String, Object>> assignmentsNow =
                    (List<Map<String, Object>>) sched.getOrDefault("assignment_list", new ArrayList<>());
            boolean initialSchedule = assignmentsNow.isEmpty();

            triple = collectExistingModules(schedRoot);
            List<Map<String, Object>> modulesNow = triple.first;
            LocalDate lastStart = triple.second;
            LocalDate lastEnd   = triple.third;
            int currentN        = modulesNow.size();
            int targetN         = EQ_NUM;
            int remaining       = Math.max(0, targetN - currentN);

            Map<String, Object> workflow = pickWorkflow(envRoot);
            List<String> fabIds          = collectFabIds(envRoot);

            boolean hasAssignments = !assignmentsNow.isEmpty();

            // First evaluation (evalIndex == 0)
            if (evalIndex == 0 && !hasAssignments) {
                System.out.println("[INFO] First evaluation block with empty assignment_list: "
                        + "run initial full-horizon solve (no new modules, no cut-off).");

                schedRoot.put("schedule", sched);

                backupFile(envPath);
                saveYaml(envPath, envRoot);
                if (schedOutPath.equals(schedPath)) backupFile(schedPath);
                saveYaml(schedOutPath, schedRoot);

                runSolver(projectRoot, envPath, schedPath);

                String outName = "Schedule_" +
                        blockStart.format(DateTimeFormatter.ofPattern("yyyyMMdd")) + ".yaml";
                Path outPath = outDir.resolve(outName);
                Files.copy(schedPath, outPath, StandardCopyOption.REPLACE_EXISTING);
                System.out.println("[OUT] Wrote " + projectRoot.relativize(outPath));

                current = blockEnd.plusDays(1);
                while (isGlobalHoliday(current, holiday)) current = current.plusDays(1);
                evalIndex++;
                continue;
            }

            // Inside this block: simulate arrivals PER DAY
            LocalDate newLastEnd = lastEnd;
            int modulesAdded     = 0;
            int startIndex       = currentN;

            List<Map<String, Object>> newModules = new ArrayList<>();

            double eqPerDay    = EQ_PER_DAYS;
            double eqSigmaDay  = EQ_PER_DAYS_SIGMA;
            Random rngDay      = new Random(MODULE_SEED * 10007L + evalIndex);

            for (LocalDate currentSimDay : blockDays) {
                if (remaining <= 0) break;

                boolean allowExtendToday = true;
                if (!allowExtendToday) {
                    continue;
                }

                int modulesToday = 0;
                if (remaining > 0) {
                    double demand = Math.max(0.0, rngDay.nextGaussian() * eqSigmaDay + eqPerDay);

                    if (demand <= 0.0) {
                        modulesToday = 0;
                    } else if (demand < 1.0) {
                        if (rngDay.nextDouble() < demand) modulesToday = 1;
                    } else {
                        int whole = (int) demand;
                        double frac = demand - whole;
                        modulesToday = whole;
                        if (rngDay.nextDouble() < frac) modulesToday++;
                    }

                    if (modulesToday > remaining) modulesToday = remaining;
                }

                if (modulesToday <= 0) {
                    System.out.println("[BLOCK] " + currentSimDay + ": demand <= 0, no modules added.");
                    continue;
                }

                LocalDate[] lastEndHolder = new LocalDate[]{newLastEnd != null ? newLastEnd : cutoff};
                List<Map<String, Object>> modsForDay = createNewModules(
                        workflow,
                        fabIds,
                        startIndex,
                        modulesToday,
                        currentSimDay,
                        lastEndHolder,
                        evalIndex * 1000 + modulesAdded,
                        holiday
                );
                newLastEnd = lastEndHolder[0];

                startIndex        += modulesToday;
                remaining         -= modulesToday;
                modulesAdded      += modulesToday;
                newModules.addAll(modsForDay);

                System.out.println("[BLOCK] " + currentSimDay + ": added " + modulesToday + " modules.");
            }

            // Append any new modules to workflow_task_list
            if (!newModules.isEmpty()) {
                List<Map<String, Object>> wfList2 =
                        (List<Map<String, Object>>) sched.getOrDefault("workflow_task_list", new ArrayList<>());
                wfList2.addAll(newModules);
                sched.put("workflow_task_list", wfList2);
                schedRoot.put("schedule", sched);
            }

            // e) update plan_range (end_date only)
            Map<String, Object> pr =
                    (Map<String, Object>) sched.getOrDefault("plan_range", new LinkedHashMap<>());

            if (initialSchedule) {
                System.out.println("[INFO] Initial run: no modules added, keep plan_range as is.");
                if (!pr.containsKey("start_date")) {
                    pr.put("start_date", ymd(cutoff));
                }
                sched.put("plan_range", pr);

                // cutoff date for solver = first day of this block
                sched.put("cut_off_date", ymd(blockStart));

                schedRoot.put("schedule", sched);

                backupFile(envPath);
                saveYaml(envPath, envRoot);
                if (schedOutPath.equals(schedPath)) backupFile(schedPath);
                saveYaml(schedOutPath, schedRoot);

                System.out.println("[INFO] Exiting early after initial solve setup.");
                runSolver(projectRoot, envPath, schedPath);
                String outName = "Schedule_" + blockStart.format(DateTimeFormatter.ofPattern("yyyyMMdd")) + ".yaml";
                Path outPath = outDir.resolve(outName);
                Files.copy(schedPath, outPath, StandardCopyOption.REPLACE_EXISTING);
                System.out.println("[OUT] Wrote " + projectRoot.relativize(outPath));

                current = blockEnd.plusDays(1);
                while (isGlobalHoliday(current, holiday)) current = current.plusDays(1);
                evalIndex++;
                continue;
            }

            LocalDate currentEnd = parseDate(pr.get("end_date"));
            List<LocalDate> endCandidates = new ArrayList<>();
            if (currentEnd != null)   endCandidates.add(currentEnd);
            if (lastEndBefore != null) endCandidates.add(lastEndBefore);
            if (newLastEnd != null)   endCandidates.add(newLastEnd);
            LocalDate endBase = endCandidates.isEmpty() ? cutoff :
                    endCandidates.stream().max(LocalDate::compareTo).orElse(cutoff);
            LocalDate endFinal = endBase.plusDays(PLAN_RANGE_EXTRA_DAYS);
            pr.put("end_date", ymd(endFinal));
            sched.put("plan_range", pr);

            // cutoff date used by EmployeeSchedule = FIRST day of this block
            sched.put("cut_off_date", ymd(blockStart));

            schedRoot.put("schedule", sched);

            backupFile(envPath);
            saveYaml(envPath, envRoot);
            if (schedOutPath.equals(schedPath)) backupFile(schedPath);
            saveYaml(schedOutPath, schedRoot);

            System.out.println("Assignments: " + totalAssign + " processed, " + changedFixed + " set to Fixed.");
            System.out.println("Workers: before=" + beforeWorkers + ", added=" + addedWorkers +
                    ", after=" + afterWorkers + ", target=" + WORKER_NUM + ".");
            System.out.println("Modules: existing=" + currentN + ", added=" + modulesAdded +
                    ", target=" + targetN + ".");
            System.out.println("New plan_range: " + pr.get("start_date") + " .. " + pr.get("end_date"));
            System.out.println("cut_off_date for solver: " + ymd(blockStart));

            boolean hasExistingAssignmentsNow = hasExistingAssignments(schedRoot);

            boolean runSolverNow =
                    (!hasExistingAssignmentsNow) || (modulesAdded > 0);

            if (runSolverNow) {
                runSolver(projectRoot, envPath, schedPath);
                String outName = "Schedule_" + blockStart.format(DateTimeFormatter.ofPattern("yyyyMMdd")) + ".yaml";
                Path outPath = outDir.resolve(outName);
                Files.copy(schedPath, outPath, StandardCopyOption.REPLACE_EXISTING);
                System.out.println("[OUT] Wrote " + projectRoot.relativize(outPath));
            } else {
                System.out.println("[SKIP] No new modules in this block AND assignment_list already has rows -> skip solver/snapshot.");
            }

            int afterCount = currentN + modulesAdded;
            if (afterCount >= targetN) {
                System.out.println("[DONE] target modules reached: " + afterCount + " / " + targetN + ".");
                break;
            }

            current = blockEnd.plusDays(1);
            while (isGlobalHoliday(current, holiday)) current = current.plusDays(1);
            evalIndex++;
        }

        System.out.println("\n[DONE] Daily run finished (Java).");
    }

    private static void runSolver(Path projectRoot, Path envPath, Path schedPath) throws Exception {
        System.out.println("[RUN-JAVA] EmployeeSchedule.main(...)");
        EmployeeSchedule.main(new String[]{
                envPath.toString().replace("\\", "/"),
                schedPath.toString().replace("\\", "/")
        });
    }

    private static Path findProjectRoot(Path start) {
        Path p = start;
        while (p != null) {
            if (Files.exists(p.resolve("pom.xml"))) return p;
            p = p.getParent();
        }
        throw new RuntimeException("Could not find pom.xml above " + start);
    }

    private static class Triple<A, B, C> {
        final A first;
        final B second;
        final C third;
        Triple(A a, B b, C c) { first = a; second = b; third = c; }
    }
}
