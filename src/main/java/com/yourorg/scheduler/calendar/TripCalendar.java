package com.yourorg.scheduler.calendar;

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.function.Function;

import com.yourorg.scheduler.trip.Flexibility;
import com.yourorg.scheduler.trip.Trip;
import com.yourorg.scheduler.trip.TripEntry;

/**
 * Per-worker day marks of the enforced Fixed trips (design section 5). Built once per solve, read-only afterwards.
 * Indexed by employee id ({@code EmployeeFact.id}) and dayId (days since the plan start, horizon
 * {@code [0, horizonDays-1]}); no boxing and no map lookups on the read path.
 *
 * <ul>
 *   <li><b>blocked</b>: the start (FI) and end (FO) day of a trip; closed for every task of that worker.</li>
 *   <li><b>lockRegion</b>: the days strictly between start and end; open only for tasks of the trip's region.</li>
 *   <li><b>presenceRegion</b>: start..end inclusive, clamped to the horizon (kept for compatibility; the stay
 *       limits do not read it).</li>
 *   <li><b>tripsOf</b>: the worker's trips with their UNCLAMPED FI/FO dayIds (may be negative or beyond the horizon) and
 *       region; read by the stay limits ({@code emp-region-stay-max-on}, {@code emp-region-annual-stay-max}, design
 *       section 4.1), which count a stay from FI to FO.</li>
 * </ul>
 * Only {@link TripEntry#enforced() enforced} entries leave marks (a valid Fixed trip that survived the overlap
 * check); Flexible and not-enforced entries are ignored. Marks outside the horizon are dropped (clamped). Touching
 * trips (first.end == second.start) block the shared day for both; on that day {@link #presenceRegion} returns the
 * region of the earlier trip (the one that ends there). Use {@link WorkerCalendar#isOpen} rather than reading the
 * marks directly.
 */
public final class TripCalendar {

    /**
     * One enforced Fixed trip of a worker: FI {@code startDay} .. FO {@code endDay} as dayIds relative to the plan
     * start, NOT clamped to the horizon; {@code regionIdx} = {@link RegionIndex} index of the trip region (-1 unknown).
     */
    public record TripInterval(int startDay, int endDay, int regionIdx) {}

    private static final List<TripInterval> NO_TRIPS = List.of();
    private static final TripCalendar EMPTY =
            new TripCalendar(new boolean[0][], new int[0][], new int[0][], new List[0], 0);

    /** emp -> day -> blocked; null row = worker without marks. */
    private final boolean[][] blocked;
    /** emp -> day -> region index of the trip interior, -1 = none; null row = worker without marks. */
    private final int[][] lock;
    /** emp -> day -> region index of start..end, -1 = none; null row = worker without marks. */
    private final int[][] presence;
    /** emp -> trips, earliest first (start, then end), unclamped; null = worker without trips. */
    private final List<TripInterval>[] trips;
    private final int horizonDays;

    private TripCalendar(boolean[][] blocked, int[][] lock, int[][] presence, List<TripInterval>[] trips,
                         int horizonDays) {
        this.blocked = blocked;
        this.lock = lock;
        this.presence = presence;
        this.trips = trips;
        this.horizonDays = horizonDays;
    }

    /** A calendar without any marks (every worker takes the fast path). */
    public static TripCalendar empty() {
        return EMPTY;
    }

    /** Same as {@link #build(List, Function, RegionIndex, int, int)} with the epoch day of {@code planStart}. */
    public static TripCalendar build(List<TripEntry> entries, Function<String, Integer> workerIndexOf,
                                     RegionIndex regions, LocalDate planStart, int horizonDays) {
        return build(entries, workerIndexOf, regions, Math.toIntExact(planStart.toEpochDay()), horizonDays);
    }

    /**
     * @param entries         parsed trips ({@code ParsedSchedule.tripEntries}); null = none
     * @param workerIndexOf   worker id -> {@code EmployeeFact.id}, null for an unknown worker (its trips are ignored)
     * @param regions         region index of the current solve (a trip region it does not know gets no region lock;
     *                        its start and end day are still blocked)
     * @param planStartOffset epoch day ({@code LocalDate.toEpochDay()}) of the plan start, i.e. of dayId 0:
     *                        {@code dayId = date.toEpochDay() - planStartOffset}
     * @param horizonDays     number of dayIds of the plan ({@code planEnd - planStart + 1}); marks are clamped to
     *                        {@code [0, horizonDays-1]}
     */
    public static TripCalendar build(List<TripEntry> entries, Function<String, Integer> workerIndexOf,
                                     RegionIndex regions, int planStartOffset, int horizonDays) {
        if (entries == null || entries.isEmpty() || horizonDays <= 0) return EMPTY;

        // enforced Fixed trips of known workers, earliest first (presence on a touching day: the earlier trip wins)
        List<int[]> trips = new ArrayList<>(); // {emp, startDay, endDay, regionIdx, rawStart, rawEnd}
        int maxEmp = -1;
        for (TripEntry e : entries) {
            if (e == null || !e.enforced()) continue;
            Trip t = e.trip();
            if (t == null || t.flexibility() != Flexibility.FIXED || t.start() == null || t.end() == null) continue;
            String wid = e.worker() != null ? e.worker() : t.worker();
            Integer emp = (wid == null) ? null : workerIndexOf.apply(wid);
            if (emp == null || emp < 0) continue;
            long s = t.start().toEpochDay() - planStartOffset;
            long en = t.end().toEpochDay() - planStartOffset;
            if (en < 0 || s > horizonDays - 1L || s > en) continue; // nothing inside the horizon
            int region = (regions == null) ? -1 : regions.indexOf(t.region());
            trips.add(new int[] {emp, (int) Math.max(s, -1L), (int) Math.min(en, horizonDays), region,
                    Math.toIntExact(s), Math.toIntExact(en)});
            maxEmp = Math.max(maxEmp, emp);
        }
        if (trips.isEmpty()) return EMPTY;
        trips.sort(Comparator.<int[]>comparingInt(x -> x[4]).thenComparingInt(x -> x[5]));

        boolean[][] blocked = new boolean[maxEmp + 1][];
        int[][] lock = new int[maxEmp + 1][];
        int[][] presence = new int[maxEmp + 1][];
        @SuppressWarnings("unchecked")
        List<TripInterval>[] intervals = new List[maxEmp + 1];
        int last = horizonDays - 1;
        for (int[] t : trips) {
            int emp = t[0], start = t[1], end = t[2], region = t[3];
            if (intervals[emp] == null) intervals[emp] = new ArrayList<>(2);
            intervals[emp].add(new TripInterval(t[4], t[5], region));
            if (blocked[emp] == null) {
                blocked[emp] = new boolean[horizonDays];
                lock[emp] = new int[horizonDays];
                presence[emp] = new int[horizonDays];
                Arrays.fill(lock[emp], -1);
                Arrays.fill(presence[emp], -1);
            }
            // start / end were clamped to [-1, horizonDays], so out-of-horizon ends are simply skipped here
            if (start >= 0 && start <= last) blocked[emp][start] = true;
            if (end >= 0 && end <= last) blocked[emp][end] = true;
            for (int d = Math.max(start + 1, 0); d <= Math.min(end - 1, last); d++) {
                if (lock[emp][d] == -1) lock[emp][d] = region;
            }
            for (int d = Math.max(start, 0); d <= Math.min(end, last); d++) {
                if (presence[emp][d] == -1) presence[emp][d] = region;
            }
        }
        // a blocked day never carries a region lock (touching trips: the shared day is closed for both)
        for (int emp = 0; emp <= maxEmp; emp++) {
            if (blocked[emp] == null) continue;
            for (int d = 0; d < horizonDays; d++) if (blocked[emp][d]) lock[emp][d] = -1;
        }
        for (int emp = 0; emp <= maxEmp; emp++) {
            if (intervals[emp] != null) intervals[emp] = List.copyOf(intervals[emp]);
        }
        return new TripCalendar(blocked, lock, presence, intervals, horizonDays);
    }

    /** False for a worker without marks (and for any id outside the calendar): the fast path of every reader. */
    public boolean hasMarks(int empIdx) {
        return empIdx >= 0 && empIdx < blocked.length && blocked[empIdx] != null;
    }

    /** The start (FI) or end (FO) day of an enforced Fixed trip of this worker. */
    public boolean isBlocked(int empIdx, int day) {
        return hasMarks(empIdx) && day >= 0 && day < horizonDays && blocked[empIdx][day];
    }

    /** Region index of the trip for a day strictly inside it, else -1. */
    public int lockRegion(int empIdx, int day) {
        return (hasMarks(empIdx) && day >= 0 && day < horizonDays) ? lock[empIdx][day] : -1;
    }

    /** Region index of the trip for a day from its start to its end (both included), else -1. */
    public int presenceRegion(int empIdx, int day) {
        return (hasMarks(empIdx) && day >= 0 && day < horizonDays) ? presence[empIdx][day] : -1;
    }

    /**
     * The worker's enforced Fixed trips with their unclamped dates, earliest first (start, then end); one shared empty
     * list (no allocation) for a worker without trips and for any id outside the calendar. Exactly the trips that leave
     * marks: a trip lying entirely outside the horizon is in neither.
     */
    public List<TripInterval> tripsOf(int empIdx) {
        if (empIdx < 0 || empIdx >= trips.length) return NO_TRIPS;
        List<TripInterval> l = trips[empIdx];
        return l == null ? NO_TRIPS : l;
    }

    /** Number of dayIds covered ({@code [0, horizonDays-1]}); 0 for the empty calendar. */
    public int horizonDays() {
        return horizonDays;
    }
}
