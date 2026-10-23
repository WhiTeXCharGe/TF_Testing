package com.yourorg.scheduler.trip;

import java.time.LocalDate;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.function.Function;

/**
 * Derives the FI/FO trips of the output {@code trip_list} from the final worked days (design section 6). Pure: no solver,
 * no file access. The caller merges {@link DeriveResult#derived()} with the Fixed trips it keeps.
 *
 * <p>Per worker: dedupe days; optionally cut Fixed trips with no same-region day strictly inside (section 6.2);
 * drop days inside the remaining Fixed windows; split the rest into runs (region change, off-days
 * {@code >= max(1, interval)}, or a remaining Fixed trip in between); every run becomes a FLEXIBLE trip
 * {@code first - 1 .. last + 1}; finally a trip may start on, but never before, the previous trip's end
 * ({@code tight_transit} warning when it has to be moved).
 */
public final class TripDeriver {

    private TripDeriver() {
    }

    /**
     * @param days             worked days that have a region
     * @param fixedTrips       the input Fixed trips
     * @param intervalOf       region to raw {@code stay_off_interval} (may be null, or return null / 0: treated as 1)
     * @param allowCutFixedTrip remove Fixed trips that have no same-region worked day strictly inside
     */
    public static DeriveResult derive(List<WorkedDay> days, List<Trip> fixedTrips,
                                      Function<String, Integer> intervalOf, boolean allowCutFixedTrip) {
        List<Trip> derived = new ArrayList<>();
        List<Trip> cutFixed = new ArrayList<>();
        List<String> warnings = new ArrayList<>();
        List<WorkedDay> dayList = days == null ? List.of() : days;
        List<Trip> fixedList = fixedTrips == null ? List.of() : fixedTrips;

        TreeSet<String> workers = new TreeSet<>();
        for (WorkedDay d : dayList) {
            if (d != null && d.worker() != null) workers.add(d.worker());
        }
        for (Trip t : fixedList) {
            if (t != null && t.worker() != null) workers.add(t.worker());
        }

        for (String worker : workers) {
            deriveWorker(worker, dayList, fixedList, intervalOf, allowCutFixedTrip, derived, cutFixed, warnings);
        }
        return new DeriveResult(derived, cutFixed, warnings);
    }

    private static void deriveWorker(String worker, List<WorkedDay> allDays, List<Trip> allFixed,
                                     Function<String, Integer> intervalOf, boolean allowCut,
                                     List<Trip> derivedOut, List<Trip> cutOut, List<String> warnings) {
        // 0. dedupe; two regions on one day -> the alphabetically smaller region wins
        TreeMap<LocalDate, String> dayRegion = new TreeMap<>();
        TreeSet<LocalDate> conflictWarned = new TreeSet<>();
        for (WorkedDay d : allDays) {
            if (d == null || !worker.equals(d.worker()) || d.date() == null || d.region() == null) continue;
            String old = dayRegion.get(d.date());
            if (old == null) {
                dayRegion.put(d.date(), d.region());
            } else if (!old.equals(d.region())) {
                String kept = old.compareTo(d.region()) <= 0 ? old : d.region();
                String dropped = old.equals(kept) ? d.region() : old;
                dayRegion.put(d.date(), kept);
                if (conflictWarned.add(d.date())) {
                    warnings.add("region_conflict_day worker=" + worker + " date=" + d.date() + " kept=" + kept
                            + " dropped=" + dropped);
                }
            }
        }

        List<Trip> fixed = new ArrayList<>();
        for (Trip t : allFixed) {
            if (t != null && worker.equals(t.worker())) fixed.add(t);
        }
        fixed.sort(Comparator.comparing(Trip::start).thenComparing(Trip::end));

        // 1. cut Fixed trips with no same-region worked day strictly inside
        List<Trip> remaining = new ArrayList<>();
        for (Trip t : fixed) {
            if (allowCut && !hasDayInside(dayRegion, t)) {
                cutOut.add(t);
                warnings.add("cut_fixed_trip worker=" + worker + " start=" + t.start() + " end=" + t.end()
                        + " region=" + t.region());
            } else {
                remaining.add(t);
            }
        }

        // 2. days inside a remaining Fixed window (inclusive) belong to that trip. A worked day that the trip's calendar
        // marks would have closed (FI/FO day, or another region strictly inside) is reported, the output is unchanged.
        List<Map.Entry<LocalDate, String>> free = new ArrayList<>();
        for (Map.Entry<LocalDate, String> e : dayRegion.entrySet()) {
            boolean inside = false;
            for (Trip t : remaining) {
                if (!e.getKey().isBefore(t.start()) && !e.getKey().isAfter(t.end())) {
                    inside = true;
                    boolean fiFo = e.getKey().equals(t.start()) || e.getKey().equals(t.end());
                    if (fiFo || !e.getValue().equals(t.region())) {
                        warnings.add("fixed_trip_conflict worker=" + worker + " date=" + e.getKey()
                                + " region=" + e.getValue() + " trip_region=" + t.region());
                    }
                    break;
                }
            }
            if (!inside) free.add(e);
        }

        // 3./4. runs -> derived trips (start is mutable: the neighbour check may move it)
        List<Item> items = new ArrayList<>();
        for (Trip t : remaining) items.add(new Item(t.start(), t.end(), t.region(), false));
        int i = 0;
        while (i < free.size()) {
            String region = free.get(i).getValue();
            LocalDate first = free.get(i).getKey();
            LocalDate last = first;
            int j = i + 1;
            while (j < free.size()) {
                LocalDate next = free.get(j).getKey();
                if (!region.equals(free.get(j).getValue())) break;
                long gap = ChronoUnit.DAYS.between(last, next) - 1;
                if (gap >= interval(intervalOf, region)) break;
                if (barrierBetween(remaining, last, next)) break;
                last = next;
                j++;
            }
            items.add(new Item(first.minusDays(1), last.plusDays(1), region, true));
            i = j;
        }

        // 5. neighbour check over derived + remaining Fixed trips in start order (Fixed first on equal start)
        items.sort(Comparator.<Item, LocalDate>comparing(it -> it.start)
                .thenComparing(it -> it.derived ? 1 : 0).thenComparing(it -> it.end));
        LocalDate prevEnd = null;
        for (Item it : items) {
            if (it.derived && prevEnd != null && it.start.isBefore(prevEnd)) {
                it.start = prevEnd;
                warnings.add("tight_transit worker=" + worker + " region=" + it.region + " start=" + it.start);
            }
            if (prevEnd == null || it.end.isAfter(prevEnd)) prevEnd = it.end;
        }
        items.sort(Comparator.<Item, LocalDate>comparing(it -> it.start).thenComparing(it -> it.end));
        for (Item it : items) {
            if (it.derived) {
                derivedOut.add(new Trip(worker, it.start, it.end, it.region, Flexibility.FLEXIBLE, null, Map.of()));
            }
        }
    }

    private static boolean hasDayInside(TreeMap<LocalDate, String> dayRegion, Trip t) {
        if (!t.start().isBefore(t.end())) return false;
        for (String region : dayRegion.subMap(t.start(), false, t.end(), false).values()) {
            if (region.equals(t.region())) return true;
        }
        return false;
    }

    /** A remaining Fixed trip whose window intersects the open interval (prev, next). */
    private static boolean barrierBetween(List<Trip> remaining, LocalDate prev, LocalDate next) {
        for (Trip t : remaining) {
            if (t.start().isBefore(next) && t.end().isAfter(prev)) return true;
        }
        return false;
    }

    private static int interval(Function<String, Integer> intervalOf, String region) {
        Integer v = intervalOf == null ? null : intervalOf.apply(region);
        return v == null ? 1 : Math.max(1, v);
    }

    private static final class Item {
        LocalDate start;
        final LocalDate end;
        final String region;
        final boolean derived;

        Item(LocalDate start, LocalDate end, String region, boolean derived) {
            this.start = start;
            this.end = end;
            this.region = region;
            this.derived = derived;
        }
    }
}
