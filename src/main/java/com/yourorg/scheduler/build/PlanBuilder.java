package com.yourorg.scheduler.build;

import java.io.*;
import java.nio.file.*;
import java.time.LocalDate;
import java.util.*;
import java.util.stream.Collectors;

import com.yourorg.scheduler.config.SolverConstants;
import com.yourorg.scheduler.calendar.SolveContext;
import com.yourorg.scheduler.calendar.WorkingDays;
import com.yourorg.scheduler.io.EnvConfigParser;
import com.yourorg.scheduler.io.LoadLog;
import com.yourorg.scheduler.io.ScheduleParser;
import com.yourorg.scheduler.model.*;

/** Turns parsed env + schedule into the SinglePassPlan (blocks, seats, pinned seats, candidates, warm start). */
public class PlanBuilder {

    public static int seatAwareWorkerOrderScore(CrewSeat s, EmployeeFact e) {
        if (e == null || e.id == 0) return Integer.MAX_VALUE;
        int score = 0;
        score += WorkerRules.workerTypeRank(e, s == null ? null : s.opId) * 1_000_000;
        score += WorkerRules.positiveSkillCount(e) * 1_000;
        if (s != null && !s.needManager) {
            score += e.isManager ? 100 : 0;
        }
        score += Math.max(0, e.histFixedTotalHours / 10);
        score += e.id;
        return score;
    }

    public static int autoHours(BlockDecision b) {
        List<Integer> allowed = (b.allowed == null || b.allowed.isEmpty())
                ? List.of(8) : b.allowed.stream().sorted().collect(Collectors.toList());
        int D = (b.startDay == null || b.days == null) ? 0 : WorkingDays.workingDaysCount(b.startDay, b.days, b.factory);
        if (D == 0) return allowed.get(0);
        int H = 1; // per-seat
        int R = Math.max(1, b.requiredHours);
        int best = allowed.get(0);
        int[] bestKey = null; // [mode, penalty, |h-8|, h]
        for (int h : allowed) {
            int prod = H * h * D;
            int[] key;
            if (prod < R) key = new int[]{0, R - prod, Math.abs(h - 8), h};
            else {
                int extra = Math.max(0, (prod - R) - H * h);
                key = new int[]{1, extra, Math.abs(h - 8), h};
            }
            if (bestKey == null || compareTuple(key, bestKey) < 0) { bestKey = key; best = h; }
        }
        return best;
    }
    public static int compareTuple(int[] a, int[] b) {
        for (int i = 0; i < Math.min(a.length, b.length); i++) {
            int c = Integer.compare(a[i], b[i]); if (c != 0) return c;
        }
        return Integer.compare(a.length, b.length);
    }

    public static void validateHorizonOverlaps(SinglePassPlan plan, int minDay, int maxDay) {
        if (plan == null || plan.seats == null || plan.blocks == null) return;

        Map<Integer, BlockDecision> byBlock = new HashMap<>();
        for (BlockDecision b : plan.blocks) byBlock.put(b.id, b);

        // empId -> dayId -> list of "module|op"
        Map<Integer, Map<Integer, List<String>>> empDayTasks = new HashMap<>();
        Map<Integer, Map<Integer, Integer>> empDayHours = new HashMap<>();

        for (CrewSeat s : plan.seats) {
            if (s == null || WorkerRules.isUnassigned(s.employee)) continue;

            BlockDecision b = byBlock.get(s.blockId);
            boolean pinned = s.pinnedFixed;

            Integer start = pinned ? s.pinnedStart : (b == null ? null : b.startDay);
            Integer days  = pinned ? s.pinnedDays  : (b == null ? null : b.days);
            int hours     = pinned ? (s.pinnedHours == null ? 0 : s.pinnedHours) : (b == null ? 0 : b.chosenHours());

            if (start == null || days == null || days <= 0) continue;

            int empId = s.employee.id;
            for (int di = 0; di < days; di++) {
                int dayId = start + di;
                if (dayId < minDay || dayId > maxDay) continue;
                if (!WorkingDays.isWorkingDay(dayId, s.factory)) continue;

                empDayTasks
                    .computeIfAbsent(empId, k -> new HashMap<>())
                    .computeIfAbsent(dayId, k -> new ArrayList<>())
                    .add(s.module + "|" + s.opId + (pinned ? "(FIXED)" : "(DYN)"));

                empDayHours
                    .computeIfAbsent(empId, k -> new HashMap<>())
                    .merge(dayId, hours, Integer::sum);
            }
        }

        // report
        for (var e1 : empDayTasks.entrySet()) {
            int empId = e1.getKey();
            for (var e2 : e1.getValue().entrySet()) {
                int dayId = e2.getKey();
                List<String> tasks = e2.getValue();
                int totH = empDayHours.getOrDefault(empId, Map.of()).getOrDefault(dayId, 0);

                if (tasks.size() > 1 || totH > SolverConstants.DAILY_CAP) {
                    LoadLog.appendLoadLog(String.format(
                        Locale.ROOT,
                        "[VALIDATE] empId=%d dayId=%d tasks=%d totalHours=%d tasks=%s",
                        empId, dayId, tasks.size(), totH, tasks.toString()
                    ));
                }
            }
        }
    }

        // ---------------- Build entities for single pass ----------------

    static class BuildOut {
        List<BlockDecision> blocks;
        List<CrewSeat> seats;
    }

    public static BuildOut buildEntitiesSinglePass(ParsedSchedule sch, ParsedEnv env) {
        Map<String, TaskWindow> keyToWin = new HashMap<>();
        Map<String, List<TaskWindow>> moduleWins = new HashMap<>();
        for (TaskWindow w : sch.windows) {
            keyToWin.putIfAbsent(w.module + "|" + w.opId, w);
            moduleWins.computeIfAbsent(w.module, k -> new ArrayList<>()).add(w);
        }

        // Infer factory for BOTH fixed and flexible rows
        for (FixedAssign fa : sch.fixedRows) {
            List<TaskWindow> ws = moduleWins.getOrDefault(fa.module, List.of());
            if (!ws.isEmpty()) fa.factory = ws.get(0).factory;
        }
        for (FlexibleAssign fa : sch.flexibleRows) {
            List<TaskWindow> ws = moduleWins.getOrDefault(fa.module, List.of());
            if (!ws.isEmpty()) fa.factory = ws.get(0).factory;
        }

        List<BlockDecision> blocks = new ArrayList<>();
        List<CrewSeat> seats = new ArrayList<>();

        // ---- dummy block for all pinned seats ----
        int minDayId = sch.daySlots.stream().mapToInt(d -> d.id).min().orElse(0);
        int maxDayId = sch.daySlots.stream().mapToInt(d -> d.id).max().orElse(0);

        BlockDecision fixedBlock = new BlockDecision();
        fixedBlock.id = 0;
        fixedBlock.module = "__FIXED__";
        fixedBlock.factory = null;
        fixedBlock.phaseId = "";
        fixedBlock.phaseNum = 0;
        fixedBlock.opId = "__FIXED__";

        // FIX: windowStart/windowEnd must be GLOBAL dayIds, not size-1
        fixedBlock.windowStart = minDayId;
        fixedBlock.windowEnd   = maxDayId;

        fixedBlock.requiredHours = 0;
        fixedBlock.allowed = List.of(8);
        fixedBlock.minHeads = 0;
        fixedBlock.maxHeads = 0;
        blocks.add(fixedBlock);

        int bid = 1;
        int sid = 1;

        // ---- dynamic block creation ----
        for (TaskWindow w : sch.windows) {
            int totalReq;
            if (w.workloadHours >= 0) {
                totalReq = w.workloadHours;
            } else {
                int baseline = (w.allowed.size() == 1 && w.allowed.get(0) == 4)
                        ? 4
                        : SolverConstants.REQUIRED_HOURS_PER_WORKLOAD_DAY;
                totalReq = w.workloadDays * baseline;
            }
            int fixedAll = sch.fixedHoursByKeyAll.getOrDefault(w.module + "|" + w.opId, 0);
            int req = Math.max(0, totalReq - fixedAll);

            if (req == 0) continue;

            BlockDecision b = new BlockDecision();
            b.id = bid++;
            b.module = w.module; b.factory = w.factory;
            b.phaseId = w.phaseId; b.phaseNum = w.phaseNum;
            b.opId = w.opId;
            int wStart = Math.max(w.startDayId, minDayId);
            int wEnd   = Math.min(w.endDayId,   maxDayId);
            if (wStart > wEnd) continue; // window outside solver horizon
            b.windowStart = wStart; b.windowEnd = wEnd;
            b.requiredHours = req;
            b.allowed = new ArrayList<>(w.allowed);
            b.minHeads = w.minHeads; b.maxHeads = w.maxHeads;
            b.recommendMinHeads = w.recommendMinHeads;
            b.recommendMaxHeads = w.recommendMaxHeads;

            // ADD THIS HERE
            String key = w.module + "|" + w.opId;
            b.fixedAssignedHoursActual = sch.fixedHoursByKeyAll.getOrDefault(key, 0);

            blocks.add(b);

            for (int sidx = 0; sidx < Math.max(1, w.maxHeads); sidx++) {
                CrewSeat cs = new CrewSeat();
                cs.id = sid++;
                cs.blockId = b.id;
                cs.module = w.module; cs.factory = w.factory;
                cs.phaseId = w.phaseId; cs.phaseNum = w.phaseNum;
                cs.opId = w.opId;
                cs.seatIndex = sidx;
                cs.needManager = (sidx == 0);
                cs.employee = CrewSeat.UNASSIGNED; // manager seats: normalizeInitialValues turns it into null
                seats.add(cs);
            }
        }

        Map<String,String> moduleToFactory = new HashMap<>();
        Map<String,String> moduleOpToPhase = new HashMap<>();
        Map<String,Integer> moduleOpToPhaseNum = new HashMap<>();
        for (TaskWindow w : sch.windows) {
            moduleToFactory.put(w.module, w.factory);
            moduleOpToPhase.put(w.module + "|" + w.opId, w.phaseId);
            moduleOpToPhaseNum.put(w.module + "|" + w.opId, w.phaseNum);
        }

        // ---- pinned seats (fixed assignments) ----
        for (FixedAssign fa : sch.fixedRows) {
            String factory = moduleToFactory.getOrDefault(fa.module, fa.factory);

            CrewSeat cs = new CrewSeat();
            cs.id = sid++;
            cs.blockId = fixedBlock.id; // 0

            cs.module = fa.module; cs.factory = factory;
            cs.phaseId = moduleOpToPhase.getOrDefault(fa.module + "|" + fa.opId, fa.phaseId);
            cs.phaseNum = moduleOpToPhaseNum.getOrDefault(fa.module + "|" + fa.opId, fa.phaseNum);
            cs.opId = fa.opId;
            cs.seatIndex = 0;
            cs.needManager = false;

            cs.pinnedFixed = true;
            Set<Integer> workDays = new HashSet<>();
            int maxHour = 0;

            for (var ent : fa.hoursByDay.entrySet()) {
                int dayId = ent.getKey();
                int h = ent.getValue() == null ? 0 : ent.getValue();
                if (h <= 0) continue;

                // keep ONLY days inside plan_range/horizon
                if (dayId < SolveContext.current().solveMinDay || dayId > SolveContext.current().solveMaxDay) continue;

                // keep ONLY working days
                if (!WorkingDays.isWorkingDay(dayId, factory)) continue;

                workDays.add(dayId);
                if (h > maxHour) maxHour = h;
            }

            if (workDays.isEmpty()) {
                // This fixed assignment is fully outside the horizon → don't create a pinned seat at all
                continue;
            }

            cs.pinnedWorkDays = workDays;

            // Keep these for export/debug, but they are NOT used for coverage anymore
            int minDid = workDays.stream().min(Integer::compareTo).orElse(SolveContext.current().solveMinDay);
            int maxDid = workDays.stream().max(Integer::compareTo).orElse(minDid);

            cs.pinnedStart = minDid;
            cs.pinnedDays  = Math.max(1, maxDid - minDid + 1);
            cs.pinnedHours = (maxHour > 0) ? maxHour : 8;


            EmployeeFact ef = env.byWid.get(fa.wid);
            if (ef == null) ef = CrewSeat.UNASSIGNED;
            cs.employee = ef;

            seats.add(cs);
        }
        int horizonStart = SolveContext.current().solveMinDay;

        for (FixedAssign fa : sch.fixedRows) {
            EmployeeFact ef = env.byWid.get(fa.wid);
            if (ef == null || ef.id == 0) continue;

            if (fa.factory == null || fa.factory.isBlank()) continue;
            String region = SolveContext.current().calendars.regionOfFab(fa.factory);
            if (region == null || region.isBlank()) continue;

            for (var ent : fa.hoursByDay.entrySet()) {
                int dayId = ent.getKey();
                int h = ent.getValue() == null ? 0 : ent.getValue();
                if (h <= 0) continue;

                // keep ONLY history strictly before the solver horizon
                if (dayId >= horizonStart) continue;

                if (dayId > ef.histLastFixedDayId) {
                    ef.histLastFixedDayId = dayId;
                    ef.histLastFixedFactory = fa.factory;
                    ef.histLastFixedRegion  = region;
                }
            }
        }
        BuildOut out = new BuildOut();
        out.blocks = blocks; out.seats = seats;
        return out;
    }

    // ---------------- Entity-dependent employee ranges ----------------

    public static void fillSeatCandidatesSinglePass(
            List<CrewSeat> seats,
            List<BlockDecision> blocks,
            List<EmployeeFact> employees) {

        Map<Integer, BlockDecision> byBlock = new HashMap<>();
        for (BlockDecision b : blocks) byBlock.put(b.id, b);

        Map<String, Set<Integer>> personalOff = SolveContext.current().calendars.workerOffByWid;

        for (CrewSeat s : seats) {
            if (s.pinnedFixed) {
                EmployeeFact pinnedEmp = null;
                for (EmployeeFact e : employees) {
                    if (e != null && s.pinnedWid != null && s.pinnedWid.equals(e.wid)) { pinnedEmp = e; break; }
                }
                s.setCandidateEmployees(pinnedEmp == null ? List.of() : List.of(pinnedEmp));
                continue;
            }

            BlockDecision b = byBlock.get(s.blockId);
            // Use current entity vars if set, else conservative estimate = entire window
            int estStart = (b != null && b.startDay != null) ? b.startDay : b.windowStart;
            int estDays  = (b != null && b.days     != null) ? b.days     : Math.max(1, b.windowEnd - b.windowStart + 1);

            List<EmployeeFact> cand = new ArrayList<>();
            for (EmployeeFact e : employees) {
                if (e == null || e.id == 0) continue;

                // Skill gate
                if (e.skills.getOrDefault(s.opId, 0) < 1) continue;

                //  preference gate (level 0 for region OR customer_company → disallow)
                String fabId     = s.factory;
                String regionId  = SolveContext.current().calendars.regionOfFab(fabId);
                String customerId = SolveContext.current().calendars.customerOfFab(fabId);
                int rp = WorkerRules.regionPref(e, regionId);
                int cp = WorkerRules.customerPref(e, customerId);
                if (rp == 0 || cp == 0) continue; // hate → never candidate

                cand.add(e);
            }

            // Manager gate (strict, no UNASSIGNED fallback)
            if (s.needManager) {
                cand = cand.stream().filter(emp -> emp != null && emp.isManager).toList();
                if (cand.isEmpty()) {
                    throw new IllegalStateException(
                        "No manager candidates for block " + s.blockId +
                        " seatIndex=" + s.seatIndex + " (" + s.module + " " + s.opId + ")."
                    );
                }
            }

            List<EmployeeFact> ordered = new ArrayList<>(cand);
            ordered.sort(Comparator
                    .comparingInt((EmployeeFact e) -> WorkerRules.workerTypeRank(e, s.opId))
                    .thenComparingInt(WorkerRules::positiveSkillCount)
                    .thenComparingInt(e -> (!s.needManager && e.isManager) ? 1 : 0)
                    .thenComparingInt(e -> Math.max(0, e.histFixedTotalHours))
                    .thenComparingInt(e -> e.id));

            s.setCandidateEmployees(ordered);

        }
    }

    public static void applyInitialPlan(ParsedSchedule sch, List<BlockDecision> blocks) {
         if (sch.initialRows == null || sch.initialRows.isEmpty()) return;

        // Earliest day index that actually exists as a DaySlot (after trimming)
        int minActiveDay = sch.daySlots.stream()
                .mapToInt(d -> d.id)
                .min()
                .orElse(0);

        // Group flexible rows by (module|opId)
        Map<String, List<InitialAssign>> byKey = new HashMap<>();
        for (InitialAssign ia : sch.initialRows) {
            if (ia == null) continue;
            if (sch.activeModules != null && !sch.activeModules.isEmpty()
                    && !sch.activeModules.contains(ia.module)) {
                continue;
            }
            byKey.computeIfAbsent(ia.module + "|" + ia.opId, k -> new ArrayList<>()).add(ia);
        }
        if (byKey.isEmpty()) return;

        for (BlockDecision b : blocks) {
            if (b == null || b.id == 0) continue; // skip dummy fixed block

            List<InitialAssign> list = byKey.get(b.module + "|" + b.opId);
            if (list == null || list.isEmpty()) continue;
            
            List<Integer> days = new ArrayList<>(); 
            int totalHours = 0;
            for (InitialAssign ia : list) {
                for (Map.Entry<Integer,Integer> e : ia.hoursByDay.entrySet()) {
                    int dayId = e.getKey();
                    if (dayId < minActiveDay) continue; // ignore very old days
                    days.add(dayId);
                    totalHours += e.getValue();
                    // b.warmPinned = true; // warm-start only; do not pin
                }
            }
            if (days.isEmpty()) continue;

            Collections.sort(days);
            int startDay = days.get(0);
            int endDay   = days.get(days.size() - 1);

            // Clamp the initial plan inside the legal window
            startDay = Math.max(startDay, b.windowStart);
            endDay   = Math.min(endDay,   b.windowEnd);
            // Phase-1 blocks start at windowStart by design (vrStartWithinWindow = [windowStart]); a later warm-start
            // day is moved back to windowStart (the warm end day is kept). Timefold 2.x rejects an initial value
            // outside the value range, 1.35 tolerated it through ValidationTolerantRange.
            if (b.phaseNum == 1 && startDay > b.windowStart) {
                LoadLog.appendLoadLog(String.format(
                    Locale.ROOT,
                    "[INIT-BLOCK-PHASE1-CLAMP] module=%s op=%s blockId=%d warmStart=%d -> windowStart=%d",
                    b.module, b.opId, b.id, startDay, b.windowStart
                ));
                startDay = b.windowStart;
            }
            if (startDay > endDay) continue;

            b.startDay = startDay;
            b.days     = Math.max(1, endDay - startDay + 1);

            // Choose hours close to average per-day usage, but inside allowed[]
            if (b.allowed != null && !b.allowed.isEmpty()) {
                int avgPerDay = totalHours / Math.max(1, days.size());
                int best = b.allowed.get(0);
                int bestDelta = Math.abs(best - avgPerDay);
                for (int h : b.allowed) {
                    int d = Math.abs(h - avgPerDay);
                    if (d < bestDelta) {
                        bestDelta = d;
                        best = h;
                    }
                }
                b.hours = best;
            }
            LoadLog.appendLoadLog(String.format(
                Locale.ROOT,
                "[INIT-BLOCK] module=%s op=%s blockId=%d window=%d..%d " +
                "flexDays=%d rawSpan=%d..%d -> initStart=%d initDays=%d initHours=%s",
                b.module, b.opId, b.id, b.windowStart, b.windowEnd,
                days.size(),
                days.get(0), days.get(days.size()-1),
                b.startDay, b.days, String.valueOf(b.hours)
            ));

            if (startDay > endDay) {
                LoadLog.appendLoadLog(String.format(
                    Locale.ROOT,
                    "[INIT-BLOCK-SKIP] module=%s op=%s blockId=%d window=%d..%d rawSpan=%d..%d (clamped invalid)",
                    b.module, b.opId, b.id, b.windowStart, b.windowEnd,
                    days.get(0), days.get(days.size()-1)
                ));
                continue;
            }
        }
    }
        static void applyInitialEmployees(ParsedSchedule sch, ParsedEnv env, List<CrewSeat> seats) {
            // key: module|opId -> flexible rows (keep full row, not just wid)
            Map<String, List<FlexibleAssign>> byKey = new HashMap<>();
            for (FlexibleAssign fa : sch.flexibleRows) {
                if (fa == null || fa.wid == null || fa.wid.isBlank()) continue;
                byKey.computeIfAbsent(fa.module + "|" + fa.opId, k -> new ArrayList<>()).add(fa);
            }

            // seats by key sorted by seatIndex for stable mapping
            Map<String, List<CrewSeat>> seatByKey = new HashMap<>();
            for (CrewSeat s : seats) {
                if (s == null) continue;
                if (s.blockId == 0) continue; // ignore dummy fixed block seats
                seatByKey.computeIfAbsent(s.module + "|" + s.opId, k -> new ArrayList<>()).add(s);
            }
            for (var e : seatByKey.entrySet()) {
                e.getValue().sort(Comparator.comparingInt(x -> x.seatIndex));
            }

            for (var e : byKey.entrySet()) {
                List<CrewSeat> keySeats = seatByKey.get(e.getKey());
                if (keySeats == null || keySeats.isEmpty()) continue;

                List<FlexibleAssign> rows = e.getValue();
                for (int i = 0; i < Math.min(rows.size(), keySeats.size()); i++) {
                    FlexibleAssign row = rows.get(i);
                    CrewSeat seat = keySeats.get(i);

                    EmployeeFact emp = env.byWid.get(row.wid);
                    if (emp == null) continue;

                    // ---- Warm-start (NOT pinned) ----
                    seat.employee = emp;
                    // seat.warmPinned = true; // do not pin
                    // seat.pinnedWid  = row.wid; // not used without pinning

                    if (!row.hoursByDay.isEmpty()) {
                        Integer minDid = row.hoursByDay.keySet().stream().min(Integer::compareTo).orElse(null);
                        Integer maxDid = row.hoursByDay.keySet().stream().max(Integer::compareTo).orElse(null);
                        // seat.pinnedStart = minDid;
                        // seat.pinnedDays  = (minDid == null || maxDid == null) ? null : Math.max(1, maxDid - minDid + 1);
                        // (kept as comments; warm-start is not pinned)
                    }
                }
            }
        }


    
    // ---------------- Warm start vs. value ranges (Timefold 2.x) ----------------

    /**
     * Brings every initial planning value into its STRICT value range. Timefold 2.x checks once, when a working
     * solution is set (solve, score calculation, every environment mode), that each initial value lies in its
     * entity's value range, pinned entities included, and fails otherwise; {@code null} is skipped and planned by the
     * construction heuristic. Ranges are never widened by UNASSIGNED (a manager range must never offer it), so:
     * <ul>
     *   <li>a real warm-start employee outside the range (fails the candidate gates, e.g. a non-manager on a manager
     *       seat) is KEPT: it is recorded in {@link CrewSeat#warmOutOfGateEmployee}, which appends exactly that
     *       employee to that seat's range (warm start carried over; residual: the search may re-select him for that
     *       seat later, which 1.27 could not once he was moved away);</li>
     *   <li>an UNASSIGNED copy (id 0, not the shared instance) on a seat whose range holds the shared
     *       {@link CrewSeat#UNASSIGNED} (non-manager seats) becomes the shared instance (same meaning);</li>
     *   <li>UNASSIGNED on a manager seat becomes {@code null}: the construction heuristic fills it with a
     *       manager;</li>
     *   <li>a block startDay / days / hours outside its range becomes {@code null} (re-planned). applyInitialPlan
     *       already clamps into the window, so this is only a safety net;</li>
     *   <li>pinned seats are left alone: their range is exactly their own employee.</li>
     * </ul>
     * Constraints treat a {@code null} employee as UNASSIGNED ({@code WorkerRules.isUnassigned}) and read seats with
     * {@code forEachIncludingUnassigned}, so the score of the plan is unchanged by this step. Idempotent. Call it after the
     * warm start and after the seat candidates are filled (the employee range depends on them).
     *
     * @return number of changed values per kind (also written to the load log); empty when nothing changed
     */
    public static Map<String, Integer> normalizeInitialValues(SinglePassPlan plan) {
        Map<String, Integer> counts = new TreeMap<>();
        if (plan == null) return counts;
        if (plan.seats != null) {
            for (CrewSeat s : plan.seats) {
                if (s == null || s.employee == null || s.isPinned()) continue;
                List<EmployeeFact> range = s.eligibleEmployeesForSeat();
                if (containsIdentity(range, s.employee)) continue;
                if (s.employee.id != 0) {
                    // a real warm-start employee failing the candidate gates: kept, and appended to this seat's range
                    s.warmOutOfGateEmployee = s.employee;
                    counts.merge(s.needManager
                            ? "manager seat warm employee outside candidates -> kept (range + employee)"
                            : "seat warm employee outside candidates -> kept (range + employee)", 1, Integer::sum);
                } else if (containsIdentity(range, CrewSeat.UNASSIGNED)) {
                    s.employee = CrewSeat.UNASSIGNED;
                    counts.merge("seat UNASSIGNED copy -> shared UNASSIGNED", 1, Integer::sum);
                } else {
                    counts.merge(s.needManager ? "manager seat UNASSIGNED -> null" : "seat UNASSIGNED outside range -> null",
                            1, Integer::sum);
                    s.employee = null;
                }
            }
        }
        if (plan.blocks != null) {
            for (BlockDecision b : plan.blocks) {
                if (b == null || b.isPinned()) continue;
                boolean changed = false;
                if (b.startDay != null && !b.vrStartWithinWindow().contains(b.startDay)) {
                    b.startDay = null;
                    counts.merge("block startDay outside range -> null", 1, Integer::sum);
                    changed = true;
                }
                if (b.days != null && !b.vrDaysWithinWindow().contains(b.days)) {
                    b.days = null;
                    counts.merge("block days outside range -> null", 1, Integer::sum);
                    changed = true;
                }
                if (b.hours != null && !b.vrAllowedHours().contains(b.hours)) {
                    b.hours = null;
                    counts.merge("block hours outside range -> null", 1, Integer::sum);
                    changed = true;
                }
                if (changed) b.coveredWorkingDays = WorkingDays.computeCoveredWorkingDays(b);
            }
        }
        if (!counts.isEmpty()) {
            LoadLog.appendLoadLog("[NORMALIZE-INITIAL] " + counts);
        }
        return counts;
    }

    private static <T> boolean containsIdentity(List<T> list, T value) {
        for (T t : list) if (t == value) return true;
        return false;
    }

    // ---------------- Debug / validation helpers ----------------

    public static Map<String,Integer> recomputeFixedHoursByKeyHorizon(List<FixedAssign> fixedRows, int minDayId, int maxDayId) {
        Map<String,Integer> out = new HashMap<>();
        if (fixedRows == null) return out;
        for (FixedAssign fa : fixedRows) {
            if (fa == null || fa.module == null || fa.opId == null) continue;
            int tot = 0;
            for (var e : fa.hoursByDay.entrySet()) {
                int dayId = e.getKey();
                int h = e.getValue() == null ? 0 : e.getValue();
                if (h <= 0) continue;
                if (dayId < minDayId || dayId > maxDayId) continue;
                tot += h;
            }
            if (tot > 0) {
                out.merge(fa.module + "|" + fa.opId, tot, Integer::sum);
            }
        }
        return out;
    }

    // ---------------- Test seam (characterisation tests) ----------------
    // buildProblem() is the unchanged "build" half of solveFromYaml (parse,
    // calendars, entities, static state, warm start, seat candidates), moved
    // out so tests can score the initial plan without solving. solveFromYaml
    // calls it first, so production behaviour is identical.

    public static final class BuiltProblem {
        public SinglePassPlan plan;
        public ParsedSchedule sch;
        public ParsedEnv env;
    }

    /** Same steps as solveFromYaml up to, but not including, solving. Sets the same static state. */
    public static SinglePassPlan buildPlanForTests(String envPath, String schedPath) throws IOException {
        return buildProblem(envPath, schedPath).plan;
    }

    public static BuiltProblem buildProblem(String envPath, String schedPath) throws IOException {
        ParsedEnv env = EnvConfigParser.parseEnv(envPath);
        String schedFixedPath = ScheduleParser.preprocessScheduleYaml(schedPath);
        ParsedSchedule sch = ScheduleParser.parseSchedule(schedFixedPath, env.opdef);
        EnvConfigParser.buildCalendars(envPath, sch.planStart, sch.planEnd);
        SolveContext.current().solveMinDay = sch.daySlots.stream().mapToInt(d -> d.id).min().orElse(0);
        SolveContext.current().solveMaxDay = sch.daySlots.stream().mapToInt(d -> d.id).max().orElse(0);

        // Performance redesign: cache dayId -> year and dayId -> year-month so
        // the overtime constraints no longer join DaySlot just to read a date.
        SolveContext.current().dayToYear.clear();
        SolveContext.current().dayToYm.clear();
        for (DaySlot d : sch.daySlots) {
            SolveContext.current().dayToYear.put(d.id, d.date.getYear());
            SolveContext.current().dayToYm.put(d.id, d.date.getYear() * 100 + d.date.getMonthValue());
        }

        // ---- DEBUG: parse summaries ----
        LoadLog.logParsedEnv(env);
        LoadLog.logParsedSchedule(sch);
        LoadLog.logModulePhaseCounts(sch);
        LoadLog.logAllWindowsDetail(sch);

        // Recompute fixed-hours per (module|op) INSIDE solver horizon only.
        int hMin = sch.daySlots.stream().mapToInt(d -> d.id).min().orElse(0);
        int hMax = sch.daySlots.stream().mapToInt(d -> d.id).max().orElse(0);
        sch.fixedHoursByKeyHorizon = recomputeFixedHoursByKeyHorizon(sch.fixedRows, hMin, hMax);


        int realEmp = Math.max(1, env.employees.size() - 1);
        int totalReq = sch.requiredByKey.values().stream().mapToInt(Integer::intValue).sum();
        SolveContext.current().targetHoursPerEmp = totalReq / (double) realEmp;

        BuildOut built = buildEntitiesSinglePass(sch, env);
        // --------------------------------------------------
        // Solver horizon (ONLY these dayIds exist as DaySlot)
        // --------------------------------------------------
        // clear history for all employees
        for (EmployeeFact e : env.employees) {
            if (e == null || e.id == 0) continue;
            e.histFixedTotalHours = 0;
            e.histFixedHoursByDay.clear();
            e.histFixedFactoriesByDay.clear();
        }

        // fill from fixedRows (HORIZON ONLY)
        for (FixedAssign fa : sch.fixedRows) {
            EmployeeFact ef = env.byWid.get(fa.wid);
            if (ef == null || ef.id == 0) continue;

            // hours + total (horizon-only)
            for (var ent : fa.hoursByDay.entrySet()) {
                int dayId = ent.getKey();
                int h     = ent.getValue();
                if (h <= 0) continue;

                if (dayId < SolveContext.current().solveMinDay || dayId > SolveContext.current().solveMaxDay) continue;

                ef.histFixedHoursByDay.merge(dayId, h, Integer::sum);
                ef.histFixedTotalHours += h;
            }

            // factories per day (horizon-only)
            if (fa.factory != null && !fa.factory.isBlank()) {
                for (Integer dayId : fa.hoursByDay.keySet()) {
                    if (dayId < SolveContext.current().solveMinDay || dayId > SolveContext.current().solveMaxDay) continue;

                    ef.histFixedFactoriesByDay
                        .computeIfAbsent(dayId, k -> new HashSet<>())
                        .add(fa.factory);
                }
            }
        }

        applyInitialPlan(sch, built.blocks);
        // Performance redesign: seed the coveredWorkingDays shadow variable from
        // the warm-started block spans so code that reads the plan before Timefold
        // touches it sees correct values. Timefold itself recomputes it through
        // BlockDecision.coveredWorkingDaysSupplier when a working solution is set.
        for (BlockDecision b : built.blocks) {
            b.coveredWorkingDays = WorkingDays.computeCoveredWorkingDays(b);
        }
        applyInitialEmployees(sch, env, built.seats);
        // --------------------------------------------------
        // Build background fixed-hours/factories per (emp, day)
        // BUT ONLY INSIDE solver horizon daySlots.
        // --------------------------------------------------
        SolveContext.current().fixedHoursByEmpDay.clear();
        SolveContext.current().fixedFactoriesByEmpDay.clear();
        SolveContext.current().fixedTaskcountByEmpDay.clear();

        for (FixedAssign fa : sch.fixedRows) {
            EmployeeFact ef = env.byWid.get(fa.wid);
            if (ef == null || ef.id == 0) continue;

            int empId = ef.id;

            for (Map.Entry<Integer, Integer> e : fa.hoursByDay.entrySet()) {
                int dayId = e.getKey();
                int h     = e.getValue();
                if (h <= 0) continue;

                if (dayId < SolveContext.current().solveMinDay || dayId > SolveContext.current().solveMaxDay) continue;

                // existing: fixed hours
                SolveContext.current().fixedHoursByEmpDay
                    .computeIfAbsent(empId, k -> new HashMap<>())
                    .merge(dayId, h, Integer::sum);

                // existing: fixed factories
                if (fa.factory != null && !fa.factory.isBlank()) {
                    SolveContext.current().fixedFactoriesByEmpDay
                        .computeIfAbsent(empId, k -> new HashMap<>())
                        .computeIfAbsent(dayId, k -> new HashSet<>())
                        .add(fa.factory);
                }

                // fixed task count (each fixed assignment row counts as 1 task on that day)
                SolveContext.current().fixedTaskcountByEmpDay
                    .computeIfAbsent(empId, k -> new HashMap<>())
                    .merge(dayId, 1, Integer::sum);
            }
        }

        // ----  aggregate fixed totals and overtime (HORIZON ONLY) ----
        SolveContext.current().fixedTotalHoursByEmp.clear();
        SolveContext.current().fixedAnnualOtByEmpYear.clear();
        SolveContext.current().fixedMonthlyOtByEmpYm.clear();

        for (FixedAssign fa : sch.fixedRows) {
            EmployeeFact ef = env.byWid.get(fa.wid);
            if (ef == null || ef.id == 0) continue;

            int empId = ef.id;

            for (Map.Entry<Integer, Integer> e : fa.hoursByDay.entrySet()) {
                int dayId = e.getKey();
                int h     = e.getValue();
                if (h <= 0) continue;

                if (dayId < SolveContext.current().solveMinDay || dayId > SolveContext.current().solveMaxDay) continue;

                // total fixed hours per employee
                SolveContext.current().fixedTotalHoursByEmp.merge(empId, h, Integer::sum);

                // overtime maps
                LocalDate d  = sch.planStart.plusDays(dayId);
                int year     = d.getYear();
                int month    = d.getMonthValue();
                int ym       = year * 100 + month;

                int ot = Math.max(0, h - SolverConstants.BASE_HOURS_PER_DAY);

                SolveContext.current().fixedAnnualOtByEmpYear
                    .computeIfAbsent(empId, k -> new HashMap<>())
                    .merge(year, ot, Integer::sum);

                SolveContext.current().fixedMonthlyOtByEmpYm
                    .computeIfAbsent(empId, k -> new HashMap<>())
                    .merge(ym, ot, Integer::sum);
            }
        }


        fillSeatCandidatesSinglePass(built.seats, built.blocks, env.employees);
        SinglePassPlan p = new SinglePassPlan();
        p.days = sch.daySlots;
        p.employees = env.employees;
        p.blocks = built.blocks;
        p.seats  = built.seats;
        normalizeInitialValues(p);

        BuiltProblem out = new BuiltProblem();
        out.plan = p;
        out.sch = sch;
        out.env = env;
        return out;
    }
}
