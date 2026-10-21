package com.yourorg.scheduler.trip;

import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Builds the output {@code trip_list} (design section 3): the per-worker nested form
 * {@code [ {worker: w1, trips: [ ... ]}, ... ]}, workers sorted, each worker's trips sorted by start date.
 * Pure: no file access.
 */
public final class TripWriter {

    private TripWriter() {
    }

    private static final DateTimeFormatter DATE_FORMAT = DateTimeFormatter.ofPattern("yyyy/MM/dd");

    private record Out(LocalDate start, LocalDate end, int kind, Map<String, Object> map) {
    }

    /**
     * @param keptFixed   Fixed entries that stay in the output; copied from {@link TripEntry#raw()} unchanged (only the
     *                    {@code worker} key is dropped, the owner is {@link TripEntry#worker()}). An entry without a
     *                    worker cannot be placed and is skipped.
     * @param derived     derived trips; written as Flexible with an empty description
     * @param historyDays N &gt;= 0: derived trips with {@code end < planStart - N days} are dropped (Fixed entries never
     *                    are); negative keeps all
     * @param planStart   first day of the plan (only used with {@code historyDays >= 0})
     * @return the trip_list; empty when there is nothing to write
     */
    public static List<Map<String, Object>> toYaml(List<TripEntry> keptFixed, List<Trip> derived, int historyDays,
                                                   LocalDate planStart) {
        Map<String, List<Out>> byWorker = new TreeMap<>();

        if (keptFixed != null) {
            for (TripEntry e : keptFixed) {
                if (e == null || e.worker() == null || e.raw() == null) continue;
                Map<String, Object> map = new LinkedHashMap<>(e.raw());
                map.remove("worker");
                LocalDate start = e.trip() != null ? e.trip().start() : TripListParser.toDate(e.raw().get("start_date"));
                LocalDate end = e.trip() != null ? e.trip().end() : TripListParser.toDate(e.raw().get("end_date"));
                byWorker.computeIfAbsent(e.worker(), k -> new ArrayList<>()).add(new Out(start, end, 0, map));
            }
        }

        if (derived != null) {
            LocalDate oldest = (historyDays >= 0 && planStart != null) ? planStart.minusDays(historyDays) : null;
            for (Trip t : derived) {
                if (t == null || t.worker() == null) continue;
                if (oldest != null && t.end().isBefore(oldest)) continue;
                Map<String, Object> map = new LinkedHashMap<>();
                map.put("start_date", t.start().format(DATE_FORMAT));
                map.put("end_date", t.end().format(DATE_FORMAT));
                map.put("region", t.region());
                map.put("plan_flexibility", "Flexible");
                map.put("description", "");
                byWorker.computeIfAbsent(t.worker(), k -> new ArrayList<>()).add(new Out(t.start(), t.end(), 1, map));
            }
        }

        Comparator<Out> order = Comparator
                .comparing(Out::start, Comparator.nullsLast(Comparator.<LocalDate>naturalOrder()))
                .thenComparingInt(Out::kind)
                .thenComparing(Out::end, Comparator.nullsLast(Comparator.<LocalDate>naturalOrder()));

        List<Map<String, Object>> result = new ArrayList<>();
        for (Map.Entry<String, List<Out>> w : byWorker.entrySet()) {
            List<Out> trips = new ArrayList<>(w.getValue());
            trips.sort(order); // stable: equal keys keep input order
            List<Map<String, Object>> tripMaps = new ArrayList<>();
            for (Out o : trips) tripMaps.add(o.map());
            Map<String, Object> entry = new LinkedHashMap<>();
            entry.put("worker", w.getKey());
            entry.put("trips", tripMaps);
            result.add(entry);
        }
        return result;
    }
}
