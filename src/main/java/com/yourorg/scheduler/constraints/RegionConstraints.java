package com.yourorg.scheduler.constraints;

import java.util.*;

import ai.timefold.solver.core.api.score.HardMediumSoftScore;
import ai.timefold.solver.core.api.score.stream.Constraint;
import ai.timefold.solver.core.api.score.stream.ConstraintCollectors;
import ai.timefold.solver.core.api.score.stream.ConstraintFactory;
import ai.timefold.solver.core.api.score.stream.uni.UniConstraintStream;
import ai.timefold.solver.core.api.score.stream.Joiners;

import com.yourorg.scheduler.calendar.SolveContext;
import com.yourorg.scheduler.model.*;
import static com.yourorg.scheduler.constraints.ConstraintSupport.seatCoveredDays;

/** Region rules: transit gap between regions, maximum stay, annual stay (3 constraints). */
final class RegionConstraints {

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
    /**
     * Worked (employee, day, region) rows of the pinned MISC seats that have a region of their own (design
     * section 7). Before solving, they are the only pinned seats that take part in the region rules: every pinned seat
     * hangs on the dummy fixed block 0, which is uninitialised (no start/days) until the construction heuristic sets
     * it, so the inner join with BlockDecision in the streams below drops them (the golden initial totals pin this).
     * After construction block 0 is initialised and all pinned seats pass that join too; a pinned misc seat is then
     * counted twice, which changes none of the three rules (duplicate days change no span and no transit). With the
     * default misc_neutral_regions no pinned seat has a misc region, so this stream is empty.
     */
    private static UniConstraintStream<EmpDayRegion> pinnedMiscRows(ConstraintFactory f, boolean withStartDay) {
        return f.forEachIncludingUnassigned(CrewSeat.class)
            .filter(s -> s.pinnedFixed && s.regionFromMisc && !WorkerRules.isUnassigned(s.employee))
            .map(s -> {
                String region = s.regionId();
                List<EmpDayRegion> rows = new ArrayList<>();
                if (region != null && s.pinnedWorkDays != null) {
                    for (int dayId : s.pinnedWorkDays) {
                        rows.add(new EmpDayRegion(
                                s.employee, dayId, region,
                                true, s.phaseNum,
                                withStartDay ? (s.pinnedStart == null ? dayId : s.pinnedStart) : 0,
                                s.module, s.opId));
                    }
                }
                return rows;
            })
            .flattenLast(rows -> rows);
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
        var worked = f.forEachIncludingUnassigned(CrewSeat.class)
            .filter(s -> !WorkerRules.isUnassigned(s.employee))
            .join(BlockDecision.class,
                Joiners.equal((CrewSeat s) -> s.blockId, (BlockDecision b) -> b.id))
            .map((CrewSeat s, BlockDecision b) -> {
                String region = s.regionId();
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
            .concat(pinnedMiscRows(f, true))
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
        var items = f.forEachIncludingUnassigned(CrewSeat.class)
            .filter(s -> !WorkerRules.isUnassigned(s.employee))
            .join(BlockDecision.class,
                Joiners.equal((CrewSeat s) -> s.blockId, (BlockDecision b) -> b.id))
            .map((CrewSeat s, BlockDecision b) -> {
                String region = s.regionId();
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
            .concat(pinnedMiscRows(f, false))
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
        var items = f.forEachIncludingUnassigned(CrewSeat.class)
            .filter(s -> !WorkerRules.isUnassigned(s.employee))
            .join(BlockDecision.class,
                Joiners.equal((CrewSeat s) -> s.blockId, (BlockDecision b) -> b.id))
            .map((CrewSeat s, BlockDecision b) -> {
                String region = s.regionId();
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
            .concat(pinnedMiscRows(f, false))
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

    Constraint[] all(ConstraintFactory f) {
        return new Constraint[] {
            regionTransitGap(f),
            regionStayMaxOn(f),
            regionAnnualStayMax(f),
        };
    }
}
