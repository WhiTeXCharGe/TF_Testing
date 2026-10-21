package com.yourorg.scheduler.constraints;

import java.util.*;

import ai.timefold.solver.core.api.score.HardMediumSoftScore;
import ai.timefold.solver.core.api.score.stream.Constraint;
import ai.timefold.solver.core.api.score.stream.ConstraintCollectors;
import ai.timefold.solver.core.api.score.stream.ConstraintFactory;
import ai.timefold.solver.core.api.score.stream.uni.UniConstraintStream;
import ai.timefold.solver.core.api.score.stream.Joiners;

import com.yourorg.scheduler.calendar.SolveContext;
import com.yourorg.scheduler.calendar.TripCalendar.TripInterval;
import com.yourorg.scheduler.model.*;
import static com.yourorg.scheduler.constraints.ConstraintSupport.seatCoveredDays;

/**
 * Region rules: transit gap between regions, maximum stay, annual stay (3 constraints).
 *
 * <p>The transit gap reads worked days only. The two stay limits measure a stay from FI to FO, both included (design
 * section 4.1, decision D14): see {@link #longestStay} and {@link #totalStay}.
 */
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

    // ------------------------------------------------------------------ stay length (FI -> FO)

    /**
     * dayId of the marker row that {@link #tripRegionRows} adds per (worker, trip region), so that a trip without any
     * task of its region still forms a (worker, region) group. Never a real day; the stay helpers skip it.
     */
    static final int TRIP_MARKER = Integer.MIN_VALUE;

    /**
     * One marker row per (worker, region) of the worker's enforced Fixed trips (only workers with trips; a problem-fact
     * stream, so it never changes while solving). The stay rules then also see a (worker, region) without worked days.
     */
    private static UniConstraintStream<EmpDayRegion> tripRegionRows(ConstraintFactory f) {
        return f.forEach(EmployeeFact.class)
            .filter(e -> !SolveContext.current().tripCalendar.tripsOf(e.id).isEmpty())
            .map(e -> {
                SolveContext ctx = SolveContext.current();
                List<EmpDayRegion> rows = new ArrayList<>(2);
                Set<String> seen = new HashSet<>(4);
                for (TripInterval t : ctx.tripCalendar.tripsOf(e.id)) {
                    String region = ctx.regions.idOf(t.regionIdx());
                    if (region == null || !seen.add(region)) continue;
                    rows.add(new EmpDayRegion(e, TRIP_MARKER, region, true, 0, 0, null, null));
                }
                return rows;
            })
            .flattenLast(rows -> rows);
    }

    /** The worker's enforced Fixed trips in {@code regionId} (unclamped dates); the shared empty list for most workers. */
    private static List<TripInterval> tripsIn(int empId, String regionId) {
        SolveContext ctx = SolveContext.current();
        List<TripInterval> all = ctx.tripCalendar.tripsOf(empId);
        if (all.isEmpty()) return all;
        int r = ctx.regions.indexOf(regionId);
        if (r < 0) return List.of();
        List<TripInterval> out = new ArrayList<>(all.size());
        for (TripInterval t : all) if (t.regionIdx() == r) out.add(t);
        return out;
    }

    /** Worked days of a group, sorted, marker rows dropped. */
    private static int[] sortedDays(List<Integer> dayList) {
        int[] ds = new int[dayList.size()];
        int n = 0;
        for (Integer d : dayList) if (d != null && d != TRIP_MARKER) ds[n++] = d;
        if (n != ds.length) ds = Arrays.copyOf(ds, n);
        Arrays.sort(ds);
        return ds;
    }

    /**
     * Longest stay (FI..FO, both included) of one (worker, region) group. Worked days closer than the interval
     * ({@code gap = next - prev - 1 < max(1, offInterval)}) form one stay. Without a Fixed trip of that worker in that
     * region a stay is {@code (last - first + 1) + 2} days (the day before the first and the day after the last task);
     * otherwise see {@link #measureWithTrips}.
     */
    static int longestStay(List<Integer> dayList, int offInterval, int empId, String regionId) {
        List<TripInterval> trips = tripsIn(empId, regionId);
        if (!trips.isEmpty()) return measureWithTrips(dayList, offInterval, trips)[0];
        int[] ds = sortedDays(dayList);
        if (ds.length == 0) return 0;
        int brk = Math.max(1, offInterval);
        int best = 1, segStart = ds[0], prev = ds[0];
        for (int i = 1; i < ds.length; i++) {
            int d = ds[i];
            if (d - prev - 1 >= brk) { best = Math.max(best, prev - segStart + 1); segStart = d; }
            prev = d;
        }
        best = Math.max(best, prev - segStart + 1);
        return best + 2;
    }

    /**
     * Sum of the stay lengths (FI..FO) of one (worker, region) group; a day shared by the FO of one stay and the FI of
     * the next (possible with interval 1) counts once. Same stays as {@link #longestStay}.
     */
    static int totalStay(List<Integer> dayList, int offInterval, int empId, String regionId) {
        List<TripInterval> trips = tripsIn(empId, regionId);
        if (!trips.isEmpty()) return measureWithTrips(dayList, offInterval, trips)[1];
        int[] ds = sortedDays(dayList);
        if (ds.length == 0) return 0;
        int brk = Math.max(1, offInterval);
        int total = 0, segStart = ds[0], prev = ds[0];
        for (int i = 1; i < ds.length; i++) {
            int d = ds[i];
            if (d == prev) continue; // the same day from two seats counts once
            int gap = d - prev - 1;
            if (gap >= brk) {
                total += (prev - segStart + 1) + 2;
                // this stay's FO (prev + 1) and the next stay's FI (d - 1) coincide when gap == 1
                total -= Math.max(0, 2 - gap);
                segStart = d;
            }
            prev = d;
        }
        total += (prev - segStart + 1) + 2;
        return total;
    }

    /**
     * Stays of a worker with Fixed trips in the region. Present days = worked days plus every day of a trip
     * (FI..FO, unclamped, the empty days inside included). Present days closer than the interval form one stay; a stay
     * runs from {@code min(trip.start, first worked - 1)} to {@code max(trip.end, last worked + 1)} over its trips and
     * worked days. Only workers with trips in the region come here.
     *
     * @return {longest stay, sum of the stays with a shared FO/FI day counted once}
     */
    private static int[] measureWithTrips(List<Integer> dayList, int offInterval, List<TripInterval> trips) {
        int[] ds = sortedDays(dayList);
        // elements {presentStart, presentEnd, spanStart, spanEnd}
        int[][] el = new int[ds.length + trips.size()][];
        int n = 0;
        for (int d : ds) el[n++] = new int[] {d, d, d - 1, d + 1};
        for (TripInterval t : trips) el[n++] = new int[] {t.startDay(), t.endDay(), t.startDay(), t.endDay()};
        Arrays.sort(el, (a, b) -> a[0] != b[0] ? Integer.compare(a[0], b[0]) : Integer.compare(a[1], b[1]));

        int brk = Math.max(1, offInterval);
        int best = 0, total = 0;
        boolean hasPrevStay = false;
        int prevSpanEnd = 0;
        int presentEnd = el[0][1], spanStart = el[0][2], spanEnd = el[0][3];
        for (int i = 1; i <= n; i++) {
            if (i < n && el[i][0] - presentEnd - 1 < brk) { // same stay
                presentEnd = Math.max(presentEnd, el[i][1]);
                spanStart = Math.min(spanStart, el[i][2]);
                spanEnd = Math.max(spanEnd, el[i][3]);
                continue;
            }
            // close the current stay
            int len = spanEnd - spanStart + 1;
            best = Math.max(best, len);
            total += len - (hasPrevStay ? Math.max(0, prevSpanEnd - spanStart + 1) : 0);
            hasPrevStay = true;
            prevSpanEnd = spanEnd;
            if (i < n) { presentEnd = el[i][1]; spanStart = el[i][2]; spanEnd = el[i][3]; }
        }
        return new int[] {best, total};
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
            .concat(tripRegionRows(f))
            .groupBy(x -> Arrays.asList(x.emp.id, x.region),
                     ConstraintCollectors.toList(x -> x.dayId));

        return items
            .filter((key, dayList) -> {
                String regionId = (String) key.get(1);
                int maxOn  = SolveContext.current().calendars.maxStayOn(regionId);
                int offInt = SolveContext.current().calendars.stayOffInterval(regionId);
                if (maxOn == Integer.MAX_VALUE) return false;
                int maxSpan = longestStay(dayList, offInt, (Integer) key.get(0), regionId);
                return maxSpan > maxOn;
            })
            .penalize(HardMediumSoftScore.ONE_HARD,
                (key, dayList) -> {
                    String regionId = (String) key.get(1);
                    int maxOn  = SolveContext.current().calendars.maxStayOn(regionId);
                    int offInt = SolveContext.current().calendars.stayOffInterval(regionId);
                    int maxSpan = longestStay(dayList, offInt, (Integer) key.get(0), regionId);
                    return Math.max(1, maxSpan - maxOn);
                })
            .asConstraint("emp-region-stay-max-on");
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
            .concat(tripRegionRows(f))
            .groupBy(x -> Arrays.asList(x.emp.id, x.region),
                    ConstraintCollectors.toList(x -> x.dayId));

        return items
            .filter((key, dayList) -> {
                String regionId = (String) key.get(1);
                int maxAnnual = SolveContext.current().calendars.annualMaxStay(regionId);
                int offInt    = SolveContext.current().calendars.stayOffInterval(regionId);
                if (maxAnnual == Integer.MAX_VALUE) return false;
                int totalSpan = totalStay(dayList, offInt, (Integer) key.get(0), regionId);
                return totalSpan > maxAnnual;
            })
            .penalize(
                HardMediumSoftScore.ONE_HARD,
                (key, dayList) -> {
                    String regionId = (String) key.get(1);
                    int maxAnnual = SolveContext.current().calendars.annualMaxStay(regionId);
                    int offInt    = SolveContext.current().calendars.stayOffInterval(regionId);
                    int totalSpan = totalStay(dayList, offInt, (Integer) key.get(0), regionId);
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
