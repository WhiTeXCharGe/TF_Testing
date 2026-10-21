// mvn -DskipTests clean package   
// mvn -q exec:java -D"exec.args=src\main\resource\EnvConfig.yaml src\main\resource\Schedule.yaml"
package com.yourorg.scheduler;

import java.io.*;
import java.nio.file.*;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import ai.timefold.solver.core.api.score.Score;
import ai.timefold.solver.core.api.score.buildin.hardmediumsoft.HardMediumSoftScore;
import ai.timefold.solver.core.api.score.stream.Constraint;
import ai.timefold.solver.core.api.score.stream.ConstraintCollectors;
import ai.timefold.solver.core.api.score.stream.ConstraintFactory;
import ai.timefold.solver.core.api.score.stream.ConstraintProvider;
import ai.timefold.solver.core.api.score.stream.Joiners;
import ai.timefold.solver.core.api.solver.Solver;
import ai.timefold.solver.core.api.solver.SolverFactory;
import ai.timefold.solver.core.config.score.director.ScoreDirectorFactoryConfig;
import ai.timefold.solver.core.config.solver.SolverConfig;
import ai.timefold.solver.core.config.solver.termination.TerminationConfig;
import ai.timefold.solver.core.api.solver.SolutionManager;
import ai.timefold.solver.core.api.score.ScoreExplanation;

import com.yourorg.scheduler.calendar.SolveContext;
import com.yourorg.scheduler.build.PlanBuilder;
import com.yourorg.scheduler.calendar.WorkingDays;
import com.yourorg.scheduler.export.ScheduleExporter;
import com.yourorg.scheduler.io.YamlFiles;
import com.yourorg.scheduler.model.*;


/**
 * SINGLE PASS scheduler (Timefold 1.27-compatible).
 */
public class EmployeeSchedule {

    // ---------------- YAML I/O (moved to io.YamlFiles) ----------------



    // ---------------- Domain ----------------

    // ---------- Planning Entities (single pass) ----------


    // ---------------- Globals & helpers ----------------

    public static final int DAILY_CAP = 12;
    public static final int REQUIRED_HOURS_PER_WORKLOAD_DAY = 10;

    public static final boolean TRIM_FINISHED_MODULES = false;
    // if module ended more than this many days before cutOffDate,
    // it is considered finished and can be trimmed.
    public static final int MODULE_TRIM_GRACE_DAYS = 6;


    private static final java.time.format.DateTimeFormatter CLOCK =
            java.time.format.DateTimeFormatter.ofPattern("HH:mm:ss");
    private static String nowClock() { return java.time.LocalTime.now().format(CLOCK); }
    private static String fmt(java.time.Duration d) {
        long h = d.toHours(); long m = d.toMinutesPart(); long s = d.toSecondsPart(); long ms = d.toMillisPart();
        return String.format("%02d:%02d:%02d.%03d", h, m, s, ms);
    }






    // ---------------- Constraints ----------------

    public static class SinglePassConstraints implements ConstraintProvider {
        // soft weights
        static final int PREF_HOURS_WEIGHT = 3000;
        static final int SMALLER_HOURS_W   = 40;
        static final int EARLIER_START_W   = 1;
        static final int AFFINITY_W        = 5;
        static final int SKILL_DIVERSITY_W = 3;
        static final int SKILL_AVG_W       = 10;
        public static final int BASE_HOURS_PER_DAY = 8;
        static final int REGION_TRANSITION_W = 5;
        static final int FAB_PREFERENCE_W  = 5;
        static final int RECOMMEND_HEADS_W = 50; // tune
        static final int REGULAR_SPOT_W    = 40;

        @Override public Constraint[] defineConstraints(ConstraintFactory f) {
            return new Constraint[] {
                // block feasibility
                endWithinWindow(f),
                hoursValueAllowed(f),
                phaseOrder(f),

                // production & capacity (count only staffed seats)
                noUnderfillByBlock(f),
                overfillAtMostOneDayByBlock(f),

                // seat-level hard rules
                // employeeAvailableAllDays(f),
                // oneFactoryPerEmpPerDay(f),
                oneTaskPerEmpPerDay(f),
                dailyCap12h(f),
                regionTransitGap(f), 
                regionStayMaxOn(f), 
                regionAnnualStayMax(f),
                annualOvertimeLimit(f),  
                monthlyOvertimeLimit(f), 

                // softs
                preferSmallerHours(f), 
                preferEarlierStart(f), 
                softAffinityPenalty(f),
                softAffinityReward(f),
                softEncourageSkillVariety(f),
                softBalanceBlockAvgSkill(f), 
                softBalanceTotalHours(f), 
                softContinuousRegionStay(f),
                softFabPreference(f),
                preferRegularOverSpot(f),
                recommendHeadcount(f),
            };
        }

        // ---------- Block-level guards ----------
        
        Constraint endWithinWindow(ConstraintFactory f) {
            return f.forEach(BlockDecision.class)
                .filter(b -> b.startDay != null && b.days != null
                        && (b.startDay + b.days - 1) > b.windowEnd)
                .penalize(HardMediumSoftScore.ONE_HARD,
                    b -> (b.startDay + b.days - 1) - b.windowEnd)
                .asConstraint("block-end-within-window");
        }

        Constraint hoursValueAllowed(ConstraintFactory f) {
            return f.forEach(BlockDecision.class)
                .filter(b -> b.allowed == null || b.allowed.isEmpty() || !b.allowed.contains(b.chosenHours()))
                .penalize(HardMediumSoftScore.ONE_HARD)
                .asConstraint("block-hours-in-allowed");
        }

        Constraint phaseOrder(ConstraintFactory f) {
            return f.forEach(BlockDecision.class)
                .filter(a -> a.phaseNum > 0)
                .join(f.forEach(BlockDecision.class).filter(b -> b.phaseNum > 0),
                    Joiners.equal((BlockDecision a) -> a.module, (BlockDecision b) -> b.module),
                    Joiners.equal((BlockDecision a) -> a.phaseNum + 1, (BlockDecision b) -> b.phaseNum))
                .filter((a,b) -> a.startDay != null && a.days != null && b.startDay != null
                        && (a.startDay + a.days - 1) >= b.startDay)
                .penalize(HardMediumSoftScore.ONE_HARD, (a,b) -> (a.startDay + a.days - 1) - b.startDay + 1)
                .asConstraint("phase-order");
        }

        // ---------- helpers ----------
        private static int staffedCountForBlock(List<CrewSeat> seats) {
            int c = 0;
            for (CrewSeat s : seats) if (!WorkerRules.isUnassigned(s.employee)) c++;
            return c;
        }

        // (Removed seatCoversDayAndWorking: the day-based constraints no longer
        // join DaySlot. Per-seat coverage now comes from seatCoveredDays, which
        // reads the BlockDecision.coveredWorkingDays shadow variable.)

        private static boolean isWorkerUnavailable(EmployeeFact e, int dayId) {
            if (e == null || e.wid == null) return false;
            return SolveContext.current().calendars.workerOffByWid.getOrDefault(e.wid, Set.of()).contains(dayId);
        }

        private static int producedHoursForBlock(BlockDecision b, List<CrewSeat> seats) {
            if (b == null || seats == null || seats.isEmpty()) return 0;

            int total = b.fixedAssignedHoursActual;
            int hours = b.chosenHours();
            
            for (CrewSeat s : seats) {
                if (s == null || WorkerRules.isUnassigned(s.employee)) continue;
                if (s.pinnedFixed) continue;

                if (b.startDay == null || b.days == null || b.days <= 0) continue;

                for (int i = 0; i < b.days; i++) {
                    int did = b.startDay + i;
                    if (!WorkingDays.isWorkingDay(did, s.factory)) continue;
                    if (isWorkerUnavailable(s.employee, did)) continue; // zero on personal off-day
                    total += hours;
                }
            }
            return total;
        }
        // ---------- Production from staffed seats ----------
        static final int UNDERFILL_MULT = 10; // huge so it dominates other hard constraints
        private static int clampToInt(long v) {
            if (v > Integer.MAX_VALUE) return Integer.MAX_VALUE;
            if (v < Integer.MIN_VALUE) return Integer.MIN_VALUE;
            return (int) v;
        }
        Constraint noUnderfillByBlock(ConstraintFactory f) {
            var perBlock = f.forEach(BlockDecision.class)
                .join(f.forEach(CrewSeat.class),
                    Joiners.equal((BlockDecision b) -> b.id, (CrewSeat s) -> s.blockId))
                .groupBy((b, s) -> b,
                        ConstraintCollectors.toList((b, s) -> s));

            return perBlock
                .filter((b, seats) -> {
                    int prod = producedHoursForBlock(b, seats);
                    return prod < b.requiredHours;
                })
                .penalize(HardMediumSoftScore.ONE_HARD,
                    (b, seats) -> {
                        int prod = producedHoursForBlock(b, seats);
                        int gap = b.requiredHours - prod;
                        return clampToInt((long) gap * (long) UNDERFILL_MULT);
                    })
                .asConstraint("block-no-underfill");
        }

        Constraint overfillAtMostOneDayByBlock(ConstraintFactory f) {
            var perBlock = f.forEach(BlockDecision.class)
                .join(f.forEach(CrewSeat.class),
                    Joiners.equal((BlockDecision b) -> b.id, (CrewSeat s) -> s.blockId))
                .groupBy((b, s) -> b,
                        ConstraintCollectors.toList((b, s) -> s));

            return perBlock
                .filter((b, seats) -> {
                    int hours = b.chosenHours();
                    int staffed = (int) seats.stream()
                            .filter(s -> s != null && !WorkerRules.isUnassigned(s.employee) && !s.pinnedFixed)
                            .count();
                    int prod = producedHoursForBlock(b, seats);
                    int over = prod - b.requiredHours;
                    return over > staffed * hours;
                })
                .penalize(HardMediumSoftScore.ONE_HARD,
                    (b, seats) -> {
                        int hours = b.chosenHours();
                        int staffed = (int) seats.stream()
                                .filter(s -> s != null && !WorkerRules.isUnassigned(s.employee) && !s.pinnedFixed)
                                .count();
                        int prod = producedHoursForBlock(b, seats);
                        int over = prod - b.requiredHours;
                        return Math.max(0, over - staffed * hours);
                    })
                .asConstraint("block-overfill-at-most-one-day");
        }

        // ---------- Seat hard rules ----------

        Constraint employeeAvailableAllDays(ConstraintFactory f) {
            return f.forEach(CrewSeat.class)
                .join(f.forEach(BlockDecision.class),
                    Joiners.equal((CrewSeat s) -> s.blockId, (BlockDecision b) -> b.id))
                .filter((s, b) -> !WorkerRules.isUnassigned(s.employee))
                .filter((s, b) -> {
                    Set<Integer> off = SolveContext.current().calendars.workerOffByWid.getOrDefault(s.employee.wid, Set.of());

                    // FIXED seat: check only exact worked days
                    if (s.pinnedFixed) {
                        if (s.pinnedWorkDays == null) return false;
                        for (int did : s.pinnedWorkDays) {
                            if (off.contains(did)) return true;
                        }
                        return false;
                    }

                    // dynamic seat: check range
                    final Integer start = b.startDay;
                    final Integer days  = b.days;
                    if (start == null || days == null || days <= 0) return false;

                    for (int di = 0; di < days; di++) {
                        int did = start + di;
                        if (!WorkingDays.isWorkingDay(did, s.factory)) continue;
                        if (off.contains(did)) return true;
                    }
                    return false;
                })
                .penalize(HardMediumSoftScore.ONE_HARD)
                .asConstraint("seat-worker-available-all-days");
        }

        // helper key (put inside EmployeeSchedule class, anywhere static)
        static final class EmpDayKey {
            final int empId;
            final int dayId;
            EmpDayKey(int empId, int dayId) { this.empId = empId; this.dayId = dayId; }

            @Override public boolean equals(Object o) {
                if (this == o) return true;
                if (!(o instanceof EmpDayKey k)) return false;
                return empId == k.empId && dayId == k.dayId;
            }
            @Override public int hashCode() { return 31 * empId + dayId; }
        }

        // Carries (empId, dayId) for the task-count constraint.
        // Extends EmpDayKey semantics; EmpDayKey already has equals/hashCode.
        static final class EmpDayHours {
            final int empId; final int dayId; final int hours;
            EmpDayHours(int e, int d, int h) { empId=e; dayId=d; hours=h; }
        }

        // Expands one (seat, block) into one EmpDayKey per working day in the block range.
        // Used with flattenLast so Timefold scores only working days, not all 127 DaySlots.
        private static List<EmpDayKey> expandToEmpDays(int empId, int start, int days, String fab) {
            List<EmpDayKey> out = new ArrayList<>(days);
            for (int d = start, end = start + days - 1; d <= end; d++) {
                if (WorkingDays.isWorkingDay(d, fab)) out.add(new EmpDayKey(empId, d));
            }
            return out;
        }

        // Same as expandToEmpDays but also carries hours per day (for the cap constraint).
        private static List<EmpDayHours> expandToEmpDayHours(int empId, int start, int days, String fab, int hrs) {
            List<EmpDayHours> out = new ArrayList<>(days);
            for (int d = start, end = start + days - 1; d <= end; d++) {
                if (WorkingDays.isWorkingDay(d, fab)) out.add(new EmpDayHours(empId, d, hrs));
            }
            return out;
        }

        // Returns the dayIds a staffed seat actually works. Dynamic seats reuse
        // the block's coveredWorkingDays shadow variable; pinned (fixed-history)
        // seats use their explicit pinnedWorkDays set. Both collections already
        // exist, so this allocates nothing per call.
        private static Collection<Integer> seatCoveredDays(CrewSeat s, BlockDecision b) {
            if (s.pinnedFixed) {
                return (s.pinnedWorkDays == null) ? List.of() : s.pinnedWorkDays;
            }
            return (b == null || b.coveredWorkingDays == null) ? List.of() : b.coveredWorkingDays;
        }

        // One overtime contribution for a (seat, calendar-bucket) pair. The
        // overtime constraints pre-aggregate a block span by year (annual) or
        // year-month (monthly) before grouping, so each bucket is emitted once.
        static final class EmpBucketOt {
            final int empId; final String company; final int bucket; final int ot;
            EmpBucketOt(int empId, String company, int bucket, int ot) {
                this.empId = empId; this.company = company; this.bucket = bucket; this.ot = ot;
            }
        }

        Constraint oneTaskPerEmpPerDay(ConstraintFactory f) {
            // Strategy: map (seat, block) → List<EmpDayKey> (one entry per working day),
            // then flattenLast(list → list) → UniConstraintStream<EmpDayKey>.
            // This replaces the Joiners.filtering DaySlot×Seat cartesian product that was
            // O(127 DaySlots) per incremental move; now it's O(working_days_in_block).
            return f.forEach(CrewSeat.class)
                .filter(s -> !WorkerRules.isUnassigned(s.employee) && !s.isPinned())
                .join(f.forEach(BlockDecision.class),
                    Joiners.equal((CrewSeat s) -> s.blockId, (BlockDecision b) -> b.id))
                .filter((CrewSeat s, BlockDecision b) -> b.startDay != null && b.days != null && b.days > 0)
                .map((CrewSeat s, BlockDecision b) ->
                        expandToEmpDays(s.employee.id, b.startDay, b.days, s.factory))
                .flattenLast(list -> list)
                // UniConstraintStream<EmpDayKey>
                .groupBy(key -> key, ConstraintCollectors.count())
                .filter((key, dynCount) -> {
                    int fixedCount = SolveContext.current().fixedTaskcountByEmpDay
                        .getOrDefault(key.empId, Collections.emptyMap())
                        .getOrDefault(key.dayId, 0);
                    return (dynCount + fixedCount) > 1;
                })
                .penalize(HardMediumSoftScore.ONE_HARD, (key, dynCount) -> {
                    int fixedCount = SolveContext.current().fixedTaskcountByEmpDay
                        .getOrDefault(key.empId, Collections.emptyMap())
                        .getOrDefault(key.dayId, 0);
                    return (dynCount + fixedCount) - 1;
                })
                .asConstraint("emp-one-task-per-day");
        }

        Constraint dailyCap12h(ConstraintFactory f) {
            // Same flattenLast pattern. Pinned seats excluded — their hours are in FIXED_HOURS_BY_EMP_DAY.
            return f.forEach(CrewSeat.class)
                .filter(s -> !WorkerRules.isUnassigned(s.employee) && !s.isPinned())
                .join(f.forEach(BlockDecision.class),
                    Joiners.equal((CrewSeat s) -> s.blockId, (BlockDecision b) -> b.id))
                .filter((CrewSeat s, BlockDecision b) -> b.startDay != null && b.days != null && b.days > 0)
                .map((CrewSeat s, BlockDecision b) ->
                        expandToEmpDayHours(s.employee.id, b.startDay, b.days, s.factory, b.chosenHours()))
                .flattenLast(list -> list)
                // UniConstraintStream<EmpDayHours>
                .groupBy(
                    edh -> Arrays.asList(edh.empId, edh.dayId),
                    ConstraintCollectors.sum(edh -> edh.hours))
                .filter((key, dynamicHours) -> {
                    int empId = (int) key.get(0);
                    int dayId = (int) key.get(1);
                    int fixedHours = SolveContext.current().fixedHoursByEmpDay
                            .getOrDefault(empId, Collections.emptyMap())
                            .getOrDefault(dayId, 0);
                    return dynamicHours + fixedHours > DAILY_CAP;
                })
                .penalize(HardMediumSoftScore.ONE_HARD, (key, dynamicHours) -> {
                    int empId = (int) key.get(0);
                    int dayId = (int) key.get(1);
                    int fixedHours = SolveContext.current().fixedHoursByEmpDay
                            .getOrDefault(empId, Collections.emptyMap())
                            .getOrDefault(dayId, 0);
                    int total = dynamicHours + fixedHours;
                    return total - DAILY_CAP;
                })
                .asConstraint("seat-daily-cap-12h");
        }

        // Helper record
        static final class EmpDayRegion {
            final EmployeeFact emp;
            final int dayId;
            final String region;
            final boolean pinnedFixed;
            final int phaseNum;
            final int startDay;
            final String module;
            final String opId;

            EmpDayRegion(EmployeeFact emp, int dayId, String region,
                        boolean pinnedFixed, int phaseNum, int startDay,
                        String module, String opId) {
                this.emp = emp;
                this.dayId = dayId;
                this.region = region;
                this.pinnedFixed = pinnedFixed;
                this.phaseNum = phaseNum;
                this.startDay = startDay;
                this.module = module;
                this.opId = opId;
            }
        }
        private static List<EmpDayRegion> transitRelevantDays(List<EmpDayRegion> list) {
            if (list == null || list.isEmpty()) return List.of();

            boolean hasFlexible = list.stream().anyMatch(x -> x != null && !x.pinnedFixed);

            List<EmpDayRegion> out = new ArrayList<>();
            for (EmpDayRegion x : list) {
                if (x == null || x.region == null || x.region.isBlank()) continue;

                // only exclude fixed-vs-fixed
                if (!hasFlexible && x.pinnedFixed) {
                    continue;
                }

                out.add(x);
            }
            return out;
        }
        //Mark: No need to exclude Phase 1 and 2 start date compare with fixed schedule
        Constraint regionTransitGap(ConstraintFactory f) {

            // Performance redesign: the worked-day stream is derived from each
            // CrewSeat joined to its own BlockDecision and expanded through the
            // coveredWorkingDays shadow variable. The previous version joined
            // every DaySlot to every CrewSeat (a cartesian product), forcing
            // each incremental move to re-evaluate O(daySlots x seats) tuples.
            var worked = f.forEach(CrewSeat.class)
                .filter(s -> !WorkerRules.isUnassigned(s.employee))
                .join(BlockDecision.class,
                    Joiners.equal((CrewSeat s) -> s.blockId, (BlockDecision b) -> b.id))
                .map((CrewSeat s, BlockDecision b) -> {
                    String region = SolveContext.current().calendars.regionOfFab(s.factory);
                    List<EmpDayRegion> rows = new ArrayList<>();
                    for (int dayId : seatCoveredDays(s, b)) {
                        int startDay = s.pinnedFixed
                                ? (s.pinnedStart == null ? dayId : s.pinnedStart)
                                : (b.startDay == null ? dayId : b.startDay);
                        rows.add(new EmpDayRegion(
                                s.employee, dayId, region,
                                s.pinnedFixed, s.phaseNum, startDay,
                                s.module, s.opId));
                    }
                    return rows;
                })
                .flattenLast(rows -> rows)
                .filter(x -> x.region != null && !x.region.isBlank())
                .groupBy(x -> x.emp, ConstraintCollectors.toList(x -> x));

            return worked
                .filter((emp, list) -> emp != null && emp.id != 0 && list != null && !list.isEmpty())
                .filter((emp, list) -> {
                    List<EmpDayRegion> relevant = transitRelevantDays(list);

                    java.util.TreeMap<Integer, String> dayToRegion = new java.util.TreeMap<>();
                    for (EmpDayRegion x : relevant) {
                        dayToRegion.merge(
                            x.dayId,
                            x.region,
                            (a, b) -> (a.compareTo(b) <= 0) ? a : b
                        );
                    }
                    if (dayToRegion.isEmpty()) return false;

                    int prevDay = Integer.MIN_VALUE;
                    String prevRegion = null;

                    int firstDay = dayToRegion.firstKey();
                    if (emp.histLastFixedDayId != Integer.MIN_VALUE
                            && emp.histLastFixedRegion != null
                            && !emp.histLastFixedRegion.isBlank()
                            && emp.histLastFixedDayId < firstDay) {
                        prevDay = emp.histLastFixedDayId;
                        prevRegion = emp.histLastFixedRegion;
                    }

                    for (var e : dayToRegion.entrySet()) {
                        int day = e.getKey();
                        String region = e.getValue();

                        if (prevRegion != null && region != null && !region.equals(prevRegion)) {
                            int need = SolveContext.current().calendars.transitDays(prevRegion, region);
                            if (need > 0) {
                                int earliestLegal = prevDay + need + 1;
                                if (day < earliestLegal) {
                                    return true;
                                }
                            }
                        }
                        prevDay = day;
                        prevRegion = region;
                    }
                    return false;
                })
                .penalize(HardMediumSoftScore.ONE_HARD, (emp, list) -> {
                    List<EmpDayRegion> relevant = transitRelevantDays(list);

                    java.util.TreeMap<Integer, String> dayToRegion = new java.util.TreeMap<>();
                    for (EmpDayRegion x : relevant) {
                        dayToRegion.merge(
                            x.dayId,
                            x.region,
                            (a, b) -> (a.compareTo(b) <= 0) ? a : b
                        );
                    }
                    if (dayToRegion.isEmpty()) return 0;

                    int prevDay = Integer.MIN_VALUE;
                    String prevRegion = null;

                    int firstDay = dayToRegion.firstKey();
                    if (emp.histLastFixedDayId != Integer.MIN_VALUE
                            && emp.histLastFixedRegion != null
                            && !emp.histLastFixedRegion.isBlank()
                            && emp.histLastFixedDayId < firstDay) {
                        prevDay = emp.histLastFixedDayId;
                        prevRegion = emp.histLastFixedRegion;
                    }

                    int penalty = 0;
                    for (var e : dayToRegion.entrySet()) {
                        int day = e.getKey();
                        String region = e.getValue();

                        if (prevRegion != null && region != null && !region.equals(prevRegion)) {
                            int need = SolveContext.current().calendars.transitDays(prevRegion, region);
                            if (need > 0) {
                                int earliestLegal = prevDay + need + 1;
                                if (day < earliestLegal) {
                                    penalty += (earliestLegal - day);
                                }
                            }
                        }
                        prevDay = day;
                        prevRegion = region;
                    }
                    return Math.max(1, penalty);
                })
                .asConstraint("emp-region-transit-gap");
        }

        private static int maxSegmentSpanWithBreak(List<Integer> dayList, int offInterval) {
            if (dayList == null || dayList.isEmpty()) return 0;
            int brk = Math.max(1, offInterval);
            List<Integer> ds = new ArrayList<>(dayList);
            Collections.sort(ds);
            int best = 1, segStart = ds.get(0), prev = ds.get(0);
            for (int i=1;i<ds.size();i++){
                int d=ds.get(i); int gap=d-prev-1;
                if (gap >= brk) { best = Math.max(best, prev - segStart + 1); segStart = d; }
                prev = d;
            }
            best = Math.max(best, prev - segStart + 1);
            return best;
        }

        Constraint regionStayMaxOn(ConstraintFactory f) {
            // Performance redesign: replaced the DaySlot x CrewSeat cartesian
            // product with a CrewSeat -> BlockDecision join expanded through the
            // coveredWorkingDays shadow variable. region != null is enforced
            // while building rows, so the old post-groupBy null filter is gone.
            var items = f.forEach(CrewSeat.class)
                .filter(s -> !WorkerRules.isUnassigned(s.employee))
                .join(BlockDecision.class,
                    Joiners.equal((CrewSeat s) -> s.blockId, (BlockDecision b) -> b.id))
                .map((CrewSeat s, BlockDecision b) -> {
                    String region = SolveContext.current().calendars.regionOfFab(s.factory);
                    List<EmpDayRegion> rows = new ArrayList<>();
                    if (region != null) {
                        for (int dayId : seatCoveredDays(s, b)) {
                            rows.add(new EmpDayRegion(
                                    s.employee, dayId, region,
                                    s.pinnedFixed, s.phaseNum, 0,
                                    s.module, s.opId));
                        }
                    }
                    return rows;
                })
                .flattenLast(rows -> rows)
                .groupBy(x -> Arrays.asList(x.emp.id, x.region),
                         ConstraintCollectors.toList(x -> x.dayId));

            return items
                .filter((key, dayList) -> {
                    String regionId = (String) key.get(1);
                    int maxOn  = SolveContext.current().calendars.maxStayOn(regionId);
                    int offInt = SolveContext.current().calendars.stayOffInterval(regionId);
                    if (maxOn == Integer.MAX_VALUE) return false;
                    int maxSpan = maxSegmentSpanWithBreak(dayList, offInt);
                    return maxSpan > maxOn;
                })
                .penalize(HardMediumSoftScore.ONE_HARD,
                    (key, dayList) -> {
                        String regionId = (String) key.get(1);
                        int maxOn  = SolveContext.current().calendars.maxStayOn(regionId);
                        int offInt = SolveContext.current().calendars.stayOffInterval(regionId);
                        int maxSpan = maxSegmentSpanWithBreak(dayList, offInt);
                        return Math.max(1, maxSpan - maxOn);
                    })
                .asConstraint("emp-region-stay-max-on");
        }

        private static int totalSegmentSpanWithBreak(List<Integer> dayList, int offInterval) {
            if (dayList == null || dayList.isEmpty()) return 0;
            int brk = Math.max(1, offInterval);

            // deduplicate days just in case one person has multiple seats on same day/region
            List<Integer> ds = new ArrayList<>(new HashSet<>(dayList));
            Collections.sort(ds);

            int total = 0;
            int segStart = ds.get(0);
            int prev     = ds.get(0);

            for (int i = 1; i < ds.size(); i++) {
                int d   = ds.get(i);
                int gap = d - prev - 1;
                if (gap >= brk) {
                    // close previous segment
                    total += (prev - segStart + 1);
                    segStart = d;
                }
                prev = d;
            }
            // last segment
            total += (prev - segStart + 1);
            return total;
        }

        Constraint regionAnnualStayMax(ConstraintFactory f) {
            // Performance redesign: replaced the DaySlot x CrewSeat cartesian
            // product with a CrewSeat -> BlockDecision join expanded through the
            // coveredWorkingDays shadow variable.
            var items = f.forEach(CrewSeat.class)
                .filter(s -> !WorkerRules.isUnassigned(s.employee))
                .join(BlockDecision.class,
                    Joiners.equal((CrewSeat s) -> s.blockId, (BlockDecision b) -> b.id))
                .map((CrewSeat s, BlockDecision b) -> {
                    String region = SolveContext.current().calendars.regionOfFab(s.factory);
                    List<EmpDayRegion> rows = new ArrayList<>();
                    if (region != null) {
                        for (int dayId : seatCoveredDays(s, b)) {
                            rows.add(new EmpDayRegion(
                                    s.employee, dayId, region,
                                    s.pinnedFixed, s.phaseNum, 0,
                                    s.module, s.opId));
                        }
                    }
                    return rows;
                })
                .flattenLast(rows -> rows)
                .groupBy(x -> Arrays.asList(x.emp.id, x.region),
                        ConstraintCollectors.toList(x -> x.dayId));

            return items
                .filter((key, dayList) -> {
                    String regionId = (String) key.get(1);
                    int maxAnnual = SolveContext.current().calendars.annualMaxStay(regionId);
                    int offInt    = SolveContext.current().calendars.stayOffInterval(regionId);
                    if (maxAnnual == Integer.MAX_VALUE) return false;
                    int totalSpan = totalSegmentSpanWithBreak(dayList, offInt);
                    return totalSpan > maxAnnual;
                })
                .penalize(
                    HardMediumSoftScore.ONE_HARD,
                    (key, dayList) -> {
                        String regionId = (String) key.get(1);
                        int maxAnnual = SolveContext.current().calendars.annualMaxStay(regionId);
                        int offInt    = SolveContext.current().calendars.stayOffInterval(regionId);
                        int totalSpan = totalSegmentSpanWithBreak(dayList, offInt);
                        return Math.max(1, totalSpan - maxAnnual);
                    }
                )
                .asConstraint("emp-region-annual-stay-max");
        }


        Constraint annualOvertimeLimit(ConstraintFactory f) {
            // Performance redesign: dropped the DaySlot cartesian product. Each
            // seat's block span is expanded via the coveredWorkingDays shadow
            // variable and pre-aggregated by year inside map(); the year of a
            // dayId is resolved through DAY_TO_YEAR instead of a DaySlot join.
            var dynEmpYearOt = f.forEach(CrewSeat.class)
                .filter(s -> !WorkerRules.isUnassigned(s.employee))
                .join(BlockDecision.class,
                    Joiners.equal((CrewSeat s) -> s.blockId, (BlockDecision b) -> b.id))
                .map((CrewSeat s, BlockDecision b) -> {
                    // Pinned/warm-pinned overtime is loaded from
                    // FIXED_ANNUAL_OT_BY_EMP_YEAR, so those seats contribute 0
                    // here but still create their (emp, company, year) group key.
                    int otPerDay = (s.pinnedFixed || s.warmPinned)
                            ? 0
                            : Math.max(0, b.chosenHours() - BASE_HOURS_PER_DAY);
                    int empId = s.employee.id;
                    String comp = WorkerRules.company(s.employee);
                    Map<Integer,Integer> byYear = new HashMap<>();
                    for (int dayId : seatCoveredDays(s, b)) {
                        Integer year = SolveContext.current().dayToYear.get(dayId);
                        if (year == null) continue;
                        byYear.merge(year, otPerDay, Integer::sum);
                    }
                    List<EmpBucketOt> rows = new ArrayList<>(byYear.size());
                    for (Map.Entry<Integer,Integer> e : byYear.entrySet()) {
                        rows.add(new EmpBucketOt(empId, comp, e.getKey(), e.getValue()));
                    }
                    return rows;
                })
                .flattenLast(rows -> rows)
                .groupBy(x -> Arrays.asList(x.empId, x.company, x.bucket),
                         ConstraintCollectors.sum(x -> x.ot));

            return dynEmpYearOt
                .filter((key, dynOtHours) -> {
                    int empId      = (int) key.get(0);
                    String companyId = (String) key.get(1);
                    int year       = (int) key.get(2);

                    if (companyId == null || companyId.isBlank()) return false;
                    int limit = SolveContext.current().calendars.annualOtLimit(companyId);
                    if (limit == Integer.MAX_VALUE) return false;

                    int fixedOt = SolveContext.current().fixedAnnualOtByEmpYear
                            .getOrDefault(empId, Collections.emptyMap())
                            .getOrDefault(year, 0);

                    int totalOt = dynOtHours + fixedOt;
                    return totalOt > limit;
                })
                .penalize(
                    HardMediumSoftScore.ONE_HARD,
                    (key, dynOtHours) -> {
                        int empId      = (int) key.get(0);
                        String companyId = (String) key.get(1);
                        int year       = (int) key.get(2);

                        int limit = SolveContext.current().calendars.annualOtLimit(companyId);
                        int fixedOt = SolveContext.current().fixedAnnualOtByEmpYear
                                .getOrDefault(empId, Collections.emptyMap())
                                .getOrDefault(year, 0);

                        int totalOt = dynOtHours + fixedOt;
                        return Math.max(1, totalOt - limit);
                    }
                )
                .asConstraint("emp-annual-overtime-limit");
        }

        Constraint monthlyOvertimeLimit(ConstraintFactory f) {
            // Performance redesign: dropped the DaySlot cartesian product. Each
            // seat's block span is expanded via the coveredWorkingDays shadow
            // variable and pre-aggregated by year-month (ym = year*100+month)
            // through DAY_TO_YM instead of a DaySlot join. The group key now
            // carries ym directly, so the filter/penalize read key.get(2)=ym.
            var dynEmpMonthOt = f.forEach(CrewSeat.class)
                .filter(s -> !WorkerRules.isUnassigned(s.employee))
                .join(BlockDecision.class,
                    Joiners.equal((CrewSeat s) -> s.blockId, (BlockDecision b) -> b.id))
                .map((CrewSeat s, BlockDecision b) -> {
                    // Pinned overtime is loaded from FIXED_MONTHLY_OT_BY_EMP_YM,
                    // so pinned seats contribute 0 here but still create the key.
                    int otPerDay = s.pinnedFixed
                            ? 0
                            : Math.max(0, b.chosenHours() - BASE_HOURS_PER_DAY);
                    int empId = s.employee.id;
                    String comp = WorkerRules.company(s.employee);
                    Map<Integer,Integer> byYm = new HashMap<>();
                    for (int dayId : seatCoveredDays(s, b)) {
                        Integer ym = SolveContext.current().dayToYm.get(dayId);
                        if (ym == null) continue;
                        byYm.merge(ym, otPerDay, Integer::sum);
                    }
                    List<EmpBucketOt> rows = new ArrayList<>(byYm.size());
                    for (Map.Entry<Integer,Integer> e : byYm.entrySet()) {
                        rows.add(new EmpBucketOt(empId, comp, e.getKey(), e.getValue()));
                    }
                    return rows;
                })
                .flattenLast(rows -> rows)
                .groupBy(x -> Arrays.asList(x.empId, x.company, x.bucket),
                         ConstraintCollectors.sum(x -> x.ot));

            return dynEmpMonthOt
                .filter((key, dynOtHours) -> {
                    int empId      = (int) key.get(0);
                    String companyId = (String) key.get(1);
                    int ym         = (int) key.get(2);

                    if (companyId == null || companyId.isBlank()) return false;
                    int limit = SolveContext.current().calendars.monthlyOtLimit(companyId);
                    if (limit == Integer.MAX_VALUE) return false;

                    int fixedOt = SolveContext.current().fixedMonthlyOtByEmpYm
                            .getOrDefault(empId, Collections.emptyMap())
                            .getOrDefault(ym, 0);

                    int totalOt = dynOtHours + fixedOt;
                    return totalOt > limit;
                })
                .penalize(
                    HardMediumSoftScore.ONE_HARD,
                    (key, dynOtHours) -> {
                        int empId      = (int) key.get(0);
                        String companyId = (String) key.get(1);
                        int ym         = (int) key.get(2);

                        int limit = SolveContext.current().calendars.monthlyOtLimit(companyId);
                        int fixedOt = SolveContext.current().fixedMonthlyOtByEmpYm
                                .getOrDefault(empId, Collections.emptyMap())
                                .getOrDefault(ym, 0);

                        int totalOt = dynOtHours + fixedOt;
                        return Math.max(1, totalOt - limit);
                    }
                )
                .asConstraint("emp-monthly-overtime-limit");
        }



        // ---------- Softs ----------

        Constraint preferSmallerHours(ConstraintFactory f) {
            return f.forEach(BlockDecision.class)
                .penalize(HardMediumSoftScore.ONE_SOFT, b -> SMALLER_HOURS_W * b.chosenHours())
                .asConstraint("soft-smaller-hours");
        }

        // Constraint preferEarlierStart(ConstraintFactory f) {
        //     return f.forEach(BlockDecision.class)
        //         .penalize(HardMediumSoftScore.ONE_SOFT, b -> b.startDay == null ? 0 : EARLIER_START_W * b.startDay)
        //         .asConstraint("soft-earlier-start");
        // }

        Constraint preferEarlierStart(ConstraintFactory f) {
            return f.forEach(BlockDecision.class)
                    // only consider blocks that actually have a startDay
                    .filter(b -> b.startDay != null)
                    .penalize(HardMediumSoftScore.ONE_SOFT, b -> {
                        // delay relative to earliest allowed day in the window
                        int delay = b.startDay - b.windowStart;
                        if (delay <= 0) {
                            return 0; // already as early as window allows
                        }
                        return EARLIER_START_W * delay;
                    })
                    .asConstraint("soft-earlier-start");
        }



        Constraint softAffinityPenalty(ConstraintFactory f) {
            return f.forEach(CrewSeat.class)
                .filter(s -> !WorkerRules.isUnassigned(s.employee))
                .join(f.forEach(CrewSeat.class).filter(s -> !WorkerRules.isUnassigned(s.employee)),
                    Joiners.equal(s -> s.blockId),
                    Joiners.lessThan(s -> s.employee.id))
                .penalize(HardMediumSoftScore.ONE_SOFT,
                    (s1, s2) -> {
                        int edge = WorkerRules.computeAffinityEdge(s1.employee, s2.employee);
                        return AFFINITY_W * Math.max(0, -edge);
                    })
                .asConstraint("soft-affinity-penalty");
        }

        Constraint softAffinityReward(ConstraintFactory f) {
            return f.forEach(CrewSeat.class)
                .filter(s -> !WorkerRules.isUnassigned(s.employee))
                .join(f.forEach(CrewSeat.class).filter(s -> !WorkerRules.isUnassigned(s.employee)),
                    Joiners.equal(s -> s.blockId),
                    Joiners.lessThan(s -> s.employee.id))
                .reward(HardMediumSoftScore.ONE_SOFT,
                    (s1, s2) -> {
                        int edge = WorkerRules.computeAffinityEdge(s1.employee, s2.employee);
                        return AFFINITY_W * Math.max(0, edge);
                    })
                .asConstraint("soft-affinity-reward");
        }

        Constraint softEncourageSkillVariety(ConstraintFactory f) {
            return f.forEach(CrewSeat.class)
                .filter(s -> !WorkerRules.isUnassigned(s.employee))
                .groupBy(
                    // key: [blockId, opId, skillLevel]
                    s -> Arrays.asList(s.blockId, s.opId, WorkerRules.skill(s.employee, s.opId)),
                    ConstraintCollectors.count()
                )
                .filter((key, count) -> count > 1)
                .penalize(HardMediumSoftScore.ONE_SOFT,
                    (key, count) -> SKILL_DIVERSITY_W * (count * (count - 1) / 2)
                )
                .asConstraint("soft-encourage-skill-variety");
        }

        Constraint softBalanceBlockAvgSkill(ConstraintFactory f) {
            return f.forEach(CrewSeat.class)
                .filter(s -> !WorkerRules.isUnassigned(s.employee))
                .groupBy(s -> Arrays.asList(s.blockId, s.opId),
                         ConstraintCollectors.sum(s -> WorkerRules.skill(s.employee, s.opId)),
                         ConstraintCollectors.count())
                .filter((key, sumLv, cnt) -> cnt > 0)
                .penalize(HardMediumSoftScore.ONE_SOFT,
                    (key, sumLv, cnt) -> (int)(SKILL_AVG_W *
                        Math.abs((sumLv / Math.max(1.0, cnt)) - WorkerRules.avgSkill((String) key.get(1))) * 100))
                .asConstraint("soft-balance-block-avg-skill");
        }

        Constraint softBalanceTotalHours(ConstraintFactory f) {
            // Performance redesign: a seat's hours over its whole span equal
            // coveredDays * chosenHours, so the per-day DaySlot join and the
            // day flattening are unnecessary - one seat-to-block tuple suffices.
            return f.forEach(CrewSeat.class)
                .filter(s -> !WorkerRules.isUnassigned(s.employee))
                .join(BlockDecision.class,
                    Joiners.equal((CrewSeat s) -> s.blockId, (BlockDecision b) -> b.id))
                .groupBy((CrewSeat s, BlockDecision b) -> s.employee,
                        ConstraintCollectors.sum((CrewSeat s, BlockDecision b) ->
                                seatCoveredDays(s, b).size() * b.chosenHours()))
                .penalize(HardMediumSoftScore.ONE_SOFT, (emp, dynTot) -> {
                    int total = dynTot + emp.histFixedTotalHours;
                    return (int) Math.abs(total - SolveContext.current().targetHoursPerEmp);
                })
                .asConstraint("soft-balance-total-hours");
        }
        Constraint softContinuousRegionStay(ConstraintFactory f) {
            return f.forEach(CrewSeat.class)
                .filter(s -> !WorkerRules.isUnassigned(s.employee))
                .join(f.forEach(CrewSeat.class)
                    .filter(s -> !WorkerRules.isUnassigned(s.employee)),
                    Joiners.equal(s -> s.employee.id),
                    Joiners.lessThan(s -> s.blockId))
                .filter((s1, s2) -> {
                    String r1 = SolveContext.current().calendars.regionOfFab(s1.factory);
                    String r2 = SolveContext.current().calendars.regionOfFab(s2.factory);
                    return r1 != null && r2 != null && !r1.equals(r2);
                })
                .penalize(HardMediumSoftScore.ONE_SOFT,
                    (s1, s2) -> REGION_TRANSITION_W)
                .asConstraint("soft-minimize-region-transitions");
        }

        Constraint softFabPreference(ConstraintFactory f) {
            return f.forEach(CrewSeat.class)
                .join(f.forEach(BlockDecision.class),
                    Joiners.equal((CrewSeat s) -> s.blockId, (BlockDecision b) -> b.id))
                .filter((s, b) -> !WorkerRules.isUnassigned(s.employee))
                .reward(
                    HardMediumSoftScore.ONE_SOFT,
                    (s, b) -> {
                        String fabId      = s.factory;
                        String regionId   = SolveContext.current().calendars.regionOfFab(fabId);
                        String customerId = SolveContext.current().calendars.customerOfFab(fabId);

                        int rLevel = WorkerRules.regionPref(s.employee, regionId);
                        int cLevel = WorkerRules.customerPref(s.employee, customerId);

                        // level 0/1 → 0 reward; >=2 → n(n-1)
                        int rScore = (rLevel <= 1) ? 0 : (rLevel * (rLevel - 1));
                        int cScore = (cLevel <= 1) ? 0 : (cLevel * (cLevel - 1));

                        return FAB_PREFERENCE_W * (rScore + cScore);
                    }
                )
                .asConstraint("soft-fab-preference");
        }

        Constraint preferRegularOverSpot(ConstraintFactory f) {
            return f.forEach(CrewSeat.class)
                .join(f.forEach(BlockDecision.class),
                    Joiners.equal((CrewSeat s) -> s.blockId, (BlockDecision b) -> b.id))
                .filter((s, b) -> !WorkerRules.isUnassigned(s.employee))
                .filter((s, b) -> !s.employee.isRegularForOperation(s.opId))
                .penalize(
                    HardMediumSoftScore.ONE_SOFT,
                    (s, b) -> REGULAR_SPOT_W
                )
                .asConstraint("prefer-regular-over-spot");
        }
        static int expPenalty(int diff) {
            // exponential growth but clamped to int
            // diff=0 => 0
            if (diff <= 0) return 0;
            long v = 1;
            for (int i = 0; i < diff; i++) {
                v *= 3;          // base 3 exponential (3,9,27,...)
                if (v > 1_000_000_000L) break;
            }
            return clampToInt(v);
        }
        Constraint recommendHeadcount(ConstraintFactory f) {
            var perBlock = f.forEach(BlockDecision.class)
                .filter(b -> b.recommendMinHeads > 0 || b.recommendMaxHeads > 0)
                .join(f.forEach(CrewSeat.class),
                    Joiners.equal((BlockDecision b) -> b.id, (CrewSeat s) -> s.blockId))
                .groupBy((b, s) -> b, ConstraintCollectors.toList((b, s) -> s));

            return perBlock
                .penalize(HardMediumSoftScore.ONE_MEDIUM, (b, seats) -> {
                    int staffed = staffedCountForBlock(seats);

                    int min = b.recommendMinHeads;
                    int max = b.recommendMaxHeads;

                    if (min <= 0 && max > 0) min = max;
                    if (max <= 0 && min > 0) max = min;
                    if (min <= 0 && max <= 0) return 0;
                    if (max < min) max = min;

                    int diff = 0;
                    if (staffed < min) {
                        diff = min - staffed;
                    } else if (staffed > max) {
                        diff = staffed - max;
                    } else {
                        diff = 0;
                    }

                    if (diff == 0) return 0;

                    int expo = expPenalty(diff);
                    return clampToInt((long) RECOMMEND_HEADS_W * (long) expo);
                })
                .asConstraint("block-recommend-headcount-range");
        }

    }
    
    // ---------------- Parsing ----------------








    // ---------------- Solver builder ----------------

    static <S> SolverFactory<S> buildSolverFactory(Class<S> solutionClass,
                                                Class<?>[] entityClasses,
                                                Class<? extends ConstraintProvider> providerClass,
                                                String bestScoreLimit,
                                                Integer spentMinutes,
                                                Integer unimprovedSeconds) {
        SolverConfig cfg = new SolverConfig();
        cfg.withSolutionClass(solutionClass);
        cfg.withEntityClasses(entityClasses);
        cfg.withScoreDirectorFactory(
                new ScoreDirectorFactoryConfig().withConstraintProviderClass(providerClass)
        );

        TerminationConfig term = new TerminationConfig();
        if (bestScoreLimit != null) term.setBestScoreLimit(bestScoreLimit);
        if (spentMinutes != null && spentMinutes > 0) {
            term.setSpentLimit(java.time.Duration.ofMinutes(spentMinutes));
        }
        if (unimprovedSeconds != null && unimprovedSeconds > 0) {
            term.setUnimprovedSpentLimit(java.time.Duration.ofSeconds(unimprovedSeconds));
        }
        cfg.withTerminationConfig(term);

        // NOTE: multi-threaded solving (moveThreadCount) is intentionally NOT
        // set here. It requires Timefold Solver Enterprise Edition; on the
        // Community Edition it throws at solve time. The speed-up in this
        // redesign comes purely from the shadow variable and the removal of
        // the DaySlot cartesian-product joins, which need no Enterprise build.

        // NOTE:
        // Do NOT force a single WEAKEST_FIT_DECREASING construction heuristic here,
        // because this solver has multiple planning entity classes (BlockDecision and CrewSeat).
        // With an explicit CH phase and no entityClass on the placer, Timefold throws:
        // "QueuedEntityPlacerConfig ... has no entityClass configured ..."
        //
        // We still keep weakest-fit style through entity/value sorting support on the model:
        // - @PlanningEntity(difficultyComparatorClass=...) on BlockDecision and CrewSeat
        // - strengthComparatorClass on planning variables / value ranges
        //
        // If later want explicit per-entity-class CH phases, define one CH phase for
        // BlockDecision and another CH phase for CrewSeat, each with its own entity selector.

        return SolverFactory.create(cfg);
    }

    static boolean hardZero(Score<?> s) { return s != null && s.toString().startsWith("0hard"); }

    // ---------------- Logging helper ----------------

    static void appendSolveLog(
            SolverFactory<SinglePassPlan> factoryStage2,
            SinglePassPlan best2,
            ParsedSchedule sch,
            java.time.Duration stage1Duration,
            java.time.Duration stage2Duration) {

        Path logFile = Paths.get("solver_log.txt"); // will be created in working directory

        try (BufferedWriter bw = Files.newBufferedWriter(
                logFile,
                StandardOpenOption.CREATE,
                StandardOpenOption.APPEND)) {

            // SolutionManager + explanation
            SolutionManager<SinglePassPlan, HardMediumSoftScore> solutionManager =
                    SolutionManager.create(factoryStage2);
            ScoreExplanation<SinglePassPlan, HardMediumSoftScore> explanation =
                    solutionManager.explain(best2);

            // Separator
            bw.write("============================================================");
            bw.newLine();
            bw.write("Run at: " + java.time.LocalDateTime.now());
            bw.newLine();

            // Basic timing and score
            bw.write("Stage1 duration: " + fmt(stage1Duration));
            bw.newLine();
            bw.write("Stage2 duration: " + fmt(stage2Duration));
            bw.newLine();
            bw.write("Final score: " + String.valueOf(best2.getScore()));
            bw.newLine();

            // Plan and cut-off
            bw.write("Plan range (plan_range): " + sch.planStart + " .. " + sch.planEnd);
            bw.newLine();
            if (sch.cutOffDate != null) {
                bw.write("Cut-off date (cut_off_date): " + sch.cutOffDate +
                        "  (cutOffDayId=" + sch.cutOffDayId + ")");
            } else {
                bw.write("Cut-off date (cut_off_date): [none]");
            }
            bw.newLine();

            // DaySlot range actually used in solver
            int minDayId = Integer.MAX_VALUE;
            int maxDayId = Integer.MIN_VALUE;
            LocalDate minDate = null;
            LocalDate maxDate = null;
            int dayCount = 0;

            if (best2.days != null) {
                for (DaySlot d : best2.days) {
                    if (d == null) continue;
                    dayCount++;
                    if (d.id < minDayId) {
                        minDayId = d.id;
                        minDate = d.date;
                    }
                    if (d.id > maxDayId) {
                        maxDayId = d.id;
                        maxDate = d.date;
                    }
                }
            }

            if (dayCount > 0) {
                bw.write("DaySlots in solver: id=" + minDayId + ".." + maxDayId +
                        " (count=" + dayCount + "), dates=" + minDate + " .. " + maxDate);
            } else {
                bw.write("DaySlots in solver: [none]");
            }
            bw.newLine();

            // Module sets
            java.util.Set<String> activeMods =
                    (sch.activeModules == null) ? new java.util.HashSet<>() :
                            new java.util.TreeSet<>(sch.activeModules);

            java.util.Set<String> dynamicBlockMods = new java.util.TreeSet<>();
            if (best2.blocks != null) {
                for (BlockDecision b : best2.blocks) {
                    if (b == null) continue;
                    // id 0 is the dummy fixed block
                    if (b.id == 0) continue;
                    if (b.module != null && !b.module.isBlank()) {
                        dynamicBlockMods.add(b.module);
                    }
                }
            }

            java.util.Set<String> pinnedMods = new java.util.TreeSet<>();
            if (best2.seats != null) {
                for (CrewSeat s : best2.seats) {
                    if (s == null) continue;
                    if (!s.pinnedFixed) continue;
                    if (s.module != null && !s.module.isBlank()) {
                        pinnedMods.add(s.module);
                    }
                }
            }

            bw.write("Active modules by trim logic (gap <= " + MODULE_TRIM_GRACE_DAYS + " days): "
                    + (activeMods.isEmpty() ? "[none]" : String.join(", ", activeMods)));
            bw.newLine();

            bw.write("Modules with dynamic blocks (BlockDecision.id > 0): "
                    + (dynamicBlockMods.isEmpty() ? "[none]" : String.join(", ", dynamicBlockMods)));
            bw.newLine();

            bw.write("Modules with pinned seats (fixed assignments visible to constraints): "
                    + (pinnedMods.isEmpty() ? "[none]" : String.join(", ", pinnedMods)));
            bw.newLine();

            bw.write("Per-constraint scores:");
            bw.newLine();

            // Per-constraint scores
            explanation.getConstraintMatchTotalMap().forEach((constraintName, cmt) -> {
                try {
                    int matchCount = (cmt.getConstraintMatchSet() == null) ? -1 : cmt.getConstraintMatchSet().size();
                    bw.write("  " + constraintName
                            + " score=" + cmt.getScore()
                            + " matches=" + matchCount);
                    bw.newLine();
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            });

            bw.flush();
        } catch (IOException e) {
            System.err.println("Failed to write solver_log.txt: " + e.getMessage());
        }
    }


    // Appends one line per best-score improvement to score_progress.txt.
    // Call from a BestSolutionChangedEvent listener so you can watch which constraints
    // are still firing and how fast the score is improving over time.
    static void appendScoreProgress(String stage, long msElapsed, HardMediumSoftScore score,
                                    SolutionManager<SinglePassPlan, HardMediumSoftScore> sm,
                                    SinglePassPlan solution) {
        Path logFile = Paths.get("score_progress.txt");
        try (BufferedWriter bw = Files.newBufferedWriter(logFile,
                StandardOpenOption.CREATE, StandardOpenOption.APPEND)) {

            bw.write(String.format("%-7s | t+%5ds | total=%-40s%n",
                    stage, msElapsed / 1000, score));

            // Per-constraint breakdown (slightly expensive; kept because it's best-effort)
            try {
                ScoreExplanation<SinglePassPlan, HardMediumSoftScore> ex = sm.explain(solution);
                ex.getConstraintMatchTotalMap().forEach((name, cmt) -> {
                    HardMediumSoftScore cs = cmt.getScore();
                    if (cs.hardScore() == 0 && cs.mediumScore() == 0) return; // skip zero-hard
                    int matches = (cmt.getConstraintMatchSet() == null) ? -1
                            : cmt.getConstraintMatchSet().size();
                    try {
                        String shortName = name.contains("/") ? name.substring(name.lastIndexOf('/') + 1) : name;
                        bw.write(String.format("          %-45s score=%-30s matches=%d%n",
                                shortName, cs, matches));
                    } catch (IOException ex2) { throw new UncheckedIOException(ex2); }
                });
            } catch (Exception ignored) { /* explain may fail mid-solve; skip */ }

        } catch (IOException e) { /* best-effort; don't crash the solver */ }
    }

    // ---------------- Live status reporting (STATUS_FILE) ----------------
    //
    // entrypoint.sh writes status=Running once, right before launching this JVM,
    // then only writes again (Completed/Failed/Cancelled) after the JVM exits.
    // On Azure Batch that meant the status blob showed "queued" for the whole
    // solve — Batch only uploads outputFiles once the task command line exits
    // (see api-controller's uploadCondition: 'taskCompletion'), so there was
    // never an intermediate write for it to pick up. This ticker fills that gap
    // by updating STATUS_FILE itself every few seconds with the real stage and
    // an elapsed-time-based progress estimate, so a live reader (the API
    // controller reading the task's file straight off the node) has something
    // fresh to show while the task is still active/running.

    static final long STAGE1_BUDGET_MS = java.time.Duration.ofMinutes(300).toMillis();
    static final long STAGE2_BUDGET_MS = java.time.Duration.ofMinutes(60).toMillis();
    static final long TOTAL_BUDGET_MS  = STAGE1_BUDGET_MS + STAGE2_BUDGET_MS;

    private static final DateTimeFormatter STATUS_TS_FMT =
            DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'").withZone(java.time.ZoneOffset.UTC);
    private static String nowIsoUtc() { return STATUS_TS_FMT.format(java.time.Instant.now()); }

    private static volatile String runStartedAtIso = null;

    static void writeRunningStatus(int stage, double overallProgress) {
        try {
            String statusDir = System.getenv().getOrDefault("STATUS_DIR", "/work/status");
            String runId = System.getenv().getOrDefault("RUN_ID", "local");
            if (runStartedAtIso == null) runStartedAtIso = nowIsoUtc();
            double clamped = Math.max(0.0, Math.min(0.99, overallProgress));

            String json = String.format(Locale.ROOT,
                "{%n  \"runId\": \"%s\",%n  \"status\": \"Running\",%n  \"stage\": %d,%n" +
                "  \"progress\": %.4f,%n  \"startedAt\": \"%s\",%n  \"updatedAt\": \"%s\",%n" +
                "  \"finishedAt\": null,%n  \"error\": null,%n  \"output\": null%n}%n",
                runId, stage, clamped, runStartedAtIso, nowIsoUtc());

            Path dir = Paths.get(statusDir);
            Files.createDirectories(dir);
            Path statusFile = dir.resolve(runId + ".json");
            Path tmp = dir.resolve(runId + ".json.tmp");
            Files.writeString(tmp, json, java.nio.charset.StandardCharsets.UTF_8);
            Files.move(tmp, statusFile,
                    java.nio.file.StandardCopyOption.REPLACE_EXISTING,
                    java.nio.file.StandardCopyOption.ATOMIC_MOVE);
        } catch (Exception ignored) {
            // best-effort — never let status reporting break the solve
        }
    }

    /** overall progress = time spent in earlier stages' budgets + fraction of the current stage's budget. */
    static double overallProgressForStage(int stage, long msSpentInStage) {
        long priorBudgetMs = (stage <= 1) ? 0 : STAGE1_BUDGET_MS;
        long stageBudgetMs = (stage <= 1) ? STAGE1_BUDGET_MS : STAGE2_BUDGET_MS;
        double stageFrac = Math.min(1.0, msSpentInStage / (double) stageBudgetMs);
        return (priorBudgetMs + stageFrac * stageBudgetMs) / (double) TOTAL_BUDGET_MS;
    }

        static final class ScoreFiles {
            static void writeText(java.nio.file.Path path, String header, String body) {
                try (var bw = java.nio.file.Files.newBufferedWriter(
                        path,
                        java.nio.file.StandardOpenOption.CREATE,
                        java.nio.file.StandardOpenOption.TRUNCATE_EXISTING,
                        java.nio.file.StandardOpenOption.WRITE)) {
                    bw.write(header);
                    bw.newLine();
                    bw.write(body);
                    bw.newLine();
                } catch (java.io.IOException e) {
                    System.err.println("Failed to write " + path + ": " + e.getMessage());
                }
            }
        }


    public static void writeInitialScoreAnalysis(
        SolverFactory<SinglePassPlan> factory,
        SinglePassPlan initialPlan,
        String fileName,
        int topMatchesPerConstraints ) {
            try {
                var solutionManager = SolutionManager.create(factory);
                solutionManager.update(initialPlan);
                var analysis = solutionManager.analyze(initialPlan);
                String header = "========== INITIAL Score (pre-solve ) ==========\n" +
                    "Total initial score: " + analysis.score();
                
                String body = analysis.summarize(Math.max(1, topMatchesPerConstraints));
                ScoreFiles.writeText(java.nio.file.Paths.get(fileName), header, body);                
            } catch (Exception ex) {
                System.err.println("writeInitialScoreAnalysis failed: " + ex.getMessage());
            }
        }

    public static void writeFinalScoreAnalysis(
        SolverFactory<SinglePassPlan> factory,
        SinglePassPlan finalPlan,
        String fileName,
        int topMatchesPerConstraints) {
            try {
                var solutionManager = SolutionManager.create(factory);
                solutionManager.update(finalPlan);
                var analysis = solutionManager.analyze(finalPlan);
                String header = "========== Final Score (pre-solve ) ==========\n" +
                    "Total initial score: " + analysis.score();
                String body = analysis.summarize(Math.max(1,topMatchesPerConstraints));
                ScoreFiles.writeText(java.nio.file.Paths.get(fileName), header, body);
            } catch (Exception ex){
                System.err.println("writeInitialScoreAnalysis failed:  " + ex.getMessage());
            }
        }
    





// ---------------- Public API ----------------

    public static class RunResult { public SinglePassPlan plan; public LocalDate planStart; }


    /**
     * Solver factory with the production model (solution, entities, constraints, default phases) for tests:
     * optional step-count limit and random seed; no time-based termination.
     */
    static SolverFactory<SinglePassPlan> factoryForTests(Integer stepLimit, Long seed) {
        SolverConfig cfg = new SolverConfig();
        cfg.withSolutionClass(SinglePassPlan.class);
        cfg.withEntityClasses(BlockDecision.class, CrewSeat.class);
        cfg.withScoreDirectorFactory(
                new ScoreDirectorFactoryConfig().withConstraintProviderClass(SinglePassConstraints.class)
        );
        TerminationConfig term = new TerminationConfig();
        if (stepLimit != null) term.setStepCountLimit(stepLimit);
        cfg.withTerminationConfig(term);
        if (seed != null) cfg.withRandomSeed(seed);
        return SolverFactory.create(cfg);
    }


    public static RunResult solveFromYaml(String envPath, String schedPath) throws IOException {
        PlanBuilder.BuiltProblem problem = PlanBuilder.buildProblem(envPath, schedPath);
        ParsedEnv env = problem.env;
        ParsedSchedule sch = problem.sch;
        SinglePassPlan p = problem.plan;

        SolverFactory<SinglePassPlan> factoryForInitial = buildSolverFactory(
            SinglePassPlan.class,
            new Class<?>[]{ BlockDecision.class, CrewSeat.class},
            SinglePassConstraints.class,
            "0hard/*medium/*soft",
            1,
            1);
        writeInitialScoreAnalysis(factoryForInitial, p, "initial_score.txt",200);
        
        // Init score_progress.txt for this run
        try (BufferedWriter _pw = Files.newBufferedWriter(Paths.get("score_progress.txt"),
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
            _pw.write("Stage   | elapsed | total score                             | per-constraint (hard/medium only)");
            _pw.newLine();
            _pw.write("------- | ------- | --------------------------------------- | --------------------------------------------------");
            _pw.newLine();
        } catch (IOException ignored) {}

        // Ticks every few seconds so STATUS_FILE always has a fresh Running
        // stage/progress, independent of how often a new best score is found
        // (score improvements can plateau for many minutes on a hard instance).
        final java.util.concurrent.atomic.AtomicInteger tickerStage = new java.util.concurrent.atomic.AtomicInteger(1);
        final java.util.concurrent.atomic.AtomicLong tickerStageStartNanos =
                new java.util.concurrent.atomic.AtomicLong(System.nanoTime());
        ScheduledExecutorService progressTicker = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "status-progress-ticker");
            t.setDaemon(true);
            return t;
        });
        progressTicker.scheduleAtFixedRate(() -> {
            int stage = tickerStage.get();
            long msSpent = (System.nanoTime() - tickerStageStartNanos.get()) / 1_000_000L;
            writeRunningStatus(stage, overallProgressForStage(stage, msSpent));
        }, 5, 10, TimeUnit.SECONDS);

        System.out.println("Start SINGLE PASS (stage1) at " + nowClock());
        long t0 = System.nanoTime();
        // ---- Stage 1: current settings (e.g., 90/90) ----
        SolverFactory<SinglePassPlan> factoryStage1 = buildSolverFactory(
                SinglePassPlan.class,
                new Class<?>[]{ BlockDecision.class, CrewSeat.class },
                SinglePassConstraints.class,
                "0hard/*medium/*soft", 300, 600);
        Solver<SinglePassPlan> stage1 = factoryStage1.buildSolver();
        {
            SolutionManager<SinglePassPlan, HardMediumSoftScore> smS1 =
                    SolutionManager.create(factoryStage1);
            stage1.addEventListener(event -> {
                if (event.getNewBestScore() == null) return;
                appendScoreProgress("Stage1", event.getTimeMillisSpent(),
                        (HardMediumSoftScore) event.getNewBestScore(),
                        smS1, event.getNewBestSolution());
            });
        }
        SinglePassPlan best1 = stage1.solve(p);

        long t1 = System.nanoTime();
        System.out.printf("Stage1 done %s | duration=%s | score=%s | blocks=%d | seats=%d%n",
                nowClock(), fmt(java.time.Duration.ofNanos(t1 - t0)),
                String.valueOf(best1.getScore()),
                best1.blocks == null ? 0 : best1.blocks.size(),
                best1.seats == null ? 0 : best1.seats.size());

        // ---- Before stage2 ----
        // Warm-start is NOT pinned anymore, so there is nothing to "unpin".
        // Just refresh candidate lists based on the stage1 solution.
        PlanBuilder.fillSeatCandidatesSinglePass(best1.seats, best1.blocks, env.employees);
        writeFinalScoreAnalysis(factoryStage1, best1, "stage1_score.txt", 100);

        // ---- Stage 2 (polish): 60 minutes, start from stage1 result ----
        tickerStage.set(2);
        tickerStageStartNanos.set(System.nanoTime());
        System.out.println("Start POLISH (stage2, 60m) at " + nowClock());

        SolverFactory<SinglePassPlan> factoryStage2 = buildSolverFactory(
                SinglePassPlan.class,
                new Class<?>[]{ BlockDecision.class, CrewSeat.class },
                SinglePassConstraints.class,
                null /* bestScoreLimit */,
                60  /* spentMinutes */,
                60 /* unimprovedSeconds */);

        Solver<SinglePassPlan> stage2 = factoryStage2.buildSolver();
        {
            SolutionManager<SinglePassPlan, HardMediumSoftScore> smS2 =
                    SolutionManager.create(factoryStage2);
            stage2.addEventListener(event -> {
                if (event.getNewBestScore() == null) return;
                appendScoreProgress("Stage2", event.getTimeMillisSpent(),
                        (HardMediumSoftScore) event.getNewBestScore(),
                        smS2, event.getNewBestSolution());
            });
        }
        SinglePassPlan best2 = stage2.solve(best1);
        progressTicker.shutdownNow();

        long t2 = System.nanoTime();
        System.out.printf("Stage2 done %s | duration=%s | score=%s%n",
                nowClock(), fmt(java.time.Duration.ofNanos(t2 - t1)),
                String.valueOf(best2.getScore()));

        java.time.Duration stage1Dur = java.time.Duration.ofNanos(t1 - t0);
        java.time.Duration stage2Dur = java.time.Duration.ofNanos(t2 - t1);


        System.out.printf("Stage1 done %s | duration=%s | score=%s | blocks=%d | seats=%d%n",
                nowClock(), fmt(stage1Dur),
                String.valueOf(best1.getScore()),
                best1.blocks == null ? 0 : best1.blocks.size(),
                best1.seats == null ? 0 : best1.seats.size());

        System.out.printf("Stage2 done %s | duration=%s | score=%s%n",
                nowClock(), fmt(stage2Dur),
                String.valueOf(best2.getScore()));

        PlanBuilder.validateHorizonOverlaps(best2, SolveContext.current().solveMinDay, SolveContext.current().solveMaxDay);

        // Append log to file (score + durations + per-constraint score)
        writeFinalScoreAnalysis(factoryStage2, best2, "final_score.txt", 100);
        appendSolveLog(factoryStage2, best2, sch, stage1Dur, stage2Dur);

        RunResult rr = new RunResult();
        rr.plan = best2; rr.planStart = sch.planStart;
        // RunResult rr = new RunResult();
        // rr.plan = best1; rr.planStart = sch.planStart;
        return rr;

    }

    // ---------------- Export hook ----------------
    public static void main(String[] args) throws Exception {
        String envPath = args.length > 0 ? args[0] : "EnvConfig.yaml";
        String schedPath = args.length > 1 ? args[1] : "Schedule.yaml";

        RunResult rr = solveFromYaml(envPath, schedPath);

        ScheduleExporter.overwriteScheduleWithAssignments(
                rr.plan, rr.planStart, schedPath, envPath);

        System.out.println("Done.");
    }
}