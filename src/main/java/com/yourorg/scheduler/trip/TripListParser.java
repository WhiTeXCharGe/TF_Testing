package com.yourorg.scheduler.trip;

import java.time.LocalDate;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.time.format.ResolverStyle;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Date;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

import com.yourorg.scheduler.config.InvalidInputDataException;

/**
 * Reads and validates the {@code trip_list} of Schedule.yaml (design section 3, "Reading input safely"). Validation
 * only: nothing here changes how the solver works.
 *
 * <p>Input shape: {@code [ {worker: w1, trips: [ {start_date, end_date, region, plan_flexibility, description, ...}, ...]}, ...]}
 * (nested form) or a list of trip maps that carry {@code worker} themselves (flat form). Every problem becomes one
 * warning line; with {@code strict} any warning makes {@link #parse} throw {@link InvalidInputDataException}.
 *
 * <p>Result: one {@link TripEntry} per surviving trip, in input order. {@code enforced} is true only for a valid FIXED
 * trip that survives the overlap check. An invalid FIXED trip keeps an entry (not enforced) so that it can be copied to
 * the output unchanged; an invalid FLEXIBLE trip is dropped (warning only).
 */
public final class TripListParser {

    private TripListParser() {}

    /** Keys the parser reads itself; every other key of a trip is "extra". */
    static final Set<String> KNOWN_KEYS = Set.of(
            "worker", "start_date", "end_date", "region", "plan_flexibility", "flexibility", "description");

    private static final DateTimeFormatter DATE_FORMAT =
            DateTimeFormatter.ofPattern("uuuu/MM/dd").withResolverStyle(ResolverStyle.STRICT);

    public static TripParseResult parse(Object rawTripList, TripContext ctx, boolean strict) {
        List<String> warnings = new ArrayList<>();
        List<TripEntry> entries = parseLenient(rawTripList, ctx, warnings);
        if (strict && !warnings.isEmpty()) {
            throw new InvalidInputDataException(
                    "trip_list has " + warnings.size() + " problem(s): " + String.join("; ", warnings), warnings);
        }
        return new TripParseResult(entries, warnings);
    }

    // ------------------------------------------------------------------ one input trip

    private static final class Cand {
        Map<String, Object> raw;
        String worker;           // null when missing / blank
        LocalDate start, end;    // null when missing / unparseable
        String region;           // null when missing / blank
        Flexibility flex;        // effective: FLEXIBLE when missing or unknown
        String description;
        Map<String, Object> extra;
        /** problems that make the trip unusable for solving (the trip is then never enforced) */
        final List<String> blocking = new ArrayList<>();
        /** problems that do not (unknown / missing plan_flexibility) */
        final List<String> soft = new ArrayList<>();
        boolean duplicate;

        boolean valid() { return blocking.isEmpty(); }

        boolean buildable() { return worker != null && start != null && end != null && region != null; }

        Trip toTrip() {
            return new Trip(worker, start, end, region, flex, description, extra);
        }
    }

    private static List<TripEntry> parseLenient(Object rawTripList, TripContext ctx, List<String> warnings) {
        List<TripEntry> none = new ArrayList<>();
        if (rawTripList == null) return none;
        if (!(rawTripList instanceof List<?> list)) {
            warnings.add("trip_list is not a list (found " + rawTripList.getClass().getSimpleName()
                    + "); no trips read");
            return none;
        }

        // ---- 1. flatten the nested / flat forms into candidates
        List<Cand> cands = new ArrayList<>();
        int index = 0;
        for (Object el : list) {
            index++;
            if (!(el instanceof Map<?, ?> m)) {
                warnings.add("trip_list entry #" + index + " is not a map, ignored: " + el);
                continue;
            }
            if (m.containsKey("trips")) {
                String outer = text(m.get("worker"));
                Object trips = m.get("trips");
                if (trips == null) continue; // a worker without trips
                if (!(trips instanceof List<?> tl)) {
                    warnings.add("trip_list worker " + (outer == null ? "?" : outer)
                            + ": 'trips' is not a list, ignored");
                    continue;
                }
                int ti = 0;
                for (Object t : tl) {
                    ti++;
                    if (!(t instanceof Map<?, ?> tm)) {
                        warnings.add("trip_list worker " + (outer == null ? "?" : outer) + ": trip #" + ti
                                + " is not a map, ignored: " + t);
                        continue;
                    }
                    String inner = text(tm.get("worker"));
                    if (outer != null && inner != null && !outer.equals(inner)) {
                        warnings.add("trip_list worker " + outer + ": trip #" + ti + " has its own worker '" + inner
                                + "', which differs; the worker of the parent entry (" + outer + ") is used");
                    }
                    cands.add(candidate(tm, outer != null ? outer : inner, ctx));
                }
            } else {
                cands.add(candidate(m, text(m.get("worker")), ctx));
            }
        }

        // ---- 2. silent deduplication (first one wins)
        Set<String> seen = new HashSet<>();
        for (Cand c : cands) {
            String key = c.buildable()
                    ? c.worker + "|" + c.region + "|" + c.start + "|" + c.end + "|" + c.flex
                    : "raw|" + c.worker + "|" + c.raw;
            if (!seen.add(key)) c.duplicate = true;
        }

        // ---- 3. per-trip warnings, in input order
        for (Cand c : cands) {
            if (c.duplicate) continue;
            for (String p : c.blocking) {
                warnings.add(label(c) + ": " + p + (c.flex == Flexibility.FIXED
                        ? " (Fixed trip not enforced, copied to the output unchanged)"
                        : " (trip skipped)"));
            }
            for (String p : c.soft) warnings.add(label(c) + ": " + p);
        }

        // ---- 4. overlap of valid Fixed trips of one worker: the earlier (start, then end) one stays enforced
        Set<Cand> notEnforced = new HashSet<>(); // identity based: Cand does not override equals
        Map<String, List<Cand>> fixedByWorker = new LinkedHashMap<>();
        for (Cand c : cands) {
            if (!c.duplicate && c.valid() && c.flex == Flexibility.FIXED) {
                fixedByWorker.computeIfAbsent(c.worker, k -> new ArrayList<>()).add(c);
            }
        }
        for (List<Cand> fixed : fixedByWorker.values()) {
            List<Cand> sorted = new ArrayList<>(fixed); // stable sort: ties keep input order
            sorted.sort(Comparator.<Cand, LocalDate>comparing(c -> c.start).thenComparing(c -> c.end));
            LocalDate enforcedUntil = null; // latest end date among the enforced trips so far
            Cand latestEnforced = null;
            for (Cand c : sorted) {
                if (enforcedUntil != null && c.start.isBefore(enforcedUntil)) {
                    notEnforced.add(c);
                    warnings.add(label(c) + ": overlaps the Fixed trip " + latestEnforced.start + ".."
                            + latestEnforced.end + " " + latestEnforced.region
                            + " (the later trip is not enforced, copied to the output unchanged)");
                } else if (enforcedUntil == null || c.end.isAfter(enforcedUntil)) {
                    enforcedUntil = c.end; // touching (c.start == enforcedUntil) is allowed
                    latestEnforced = c;
                }
            }
        }

        // ---- 5. entries in input order
        List<TripEntry> entries = new ArrayList<>();
        for (Cand c : cands) {
            if (c.duplicate) continue;
            if (c.flex == Flexibility.FIXED) {
                boolean enforced = c.valid() && !notEnforced.contains(c);
                entries.add(new TripEntry(c.buildable() ? c.toTrip() : null, c.raw, enforced, c.worker));
            } else if (c.valid()) {
                entries.add(new TripEntry(c.toTrip(), c.raw, false, c.worker));
            } // an invalid Flexible trip is dropped
        }
        return entries;
    }

    private static Cand candidate(Map<?, ?> input, String worker, TripContext ctx) {
        Cand c = new Cand();
        c.raw = new LinkedHashMap<>();
        for (Map.Entry<?, ?> e : input.entrySet()) c.raw.put(String.valueOf(e.getKey()), plainValue(e.getValue()));
        c.extra = new LinkedHashMap<>();
        for (Map.Entry<String, Object> e : c.raw.entrySet()) {
            if (!KNOWN_KEYS.contains(e.getKey())) c.extra.put(e.getKey(), e.getValue());
        }
        c.worker = worker;
        c.region = text(c.raw.get("region"));
        c.description = c.raw.get("description") == null ? null : String.valueOf(c.raw.get("description"));

        // plan_flexibility (alias flexibility), case-insensitive; unknown or missing is treated as Flexible
        Object f = c.raw.get("plan_flexibility") != null ? c.raw.get("plan_flexibility") : c.raw.get("flexibility");
        String ft = text(f);
        if (ft == null) {
            c.flex = Flexibility.FLEXIBLE;
            c.soft.add("plan_flexibility is missing (treated as Flexible)");
        } else if (ft.toLowerCase(Locale.ROOT).equals("fixed")) {
            c.flex = Flexibility.FIXED;
        } else if (ft.toLowerCase(Locale.ROOT).equals("flexible")) {
            c.flex = Flexibility.FLEXIBLE;
        } else {
            c.flex = Flexibility.FLEXIBLE;
            c.soft.add("unknown plan_flexibility '" + ft + "' (treated as Flexible)");
        }

        // worker
        if (c.worker == null) {
            c.blocking.add("worker is missing");
        } else if (ctx.workerIds() == null || !ctx.workerIds().contains(c.worker)) {
            c.blocking.add("worker '" + c.worker + "' is not in EnvConfig");
        }

        // dates
        c.start = date(c, "start_date");
        c.end = date(c, "end_date");
        if (c.start != null && c.end != null) {
            if (c.start.isAfter(c.end)) {
                c.blocking.add("start_date is after end_date");
            } else if (c.start.equals(c.end)) {
                c.blocking.add("start_date equals end_date inside one trip (equal dates are only valid between two "
                        + "different trips)");
            }
        }

        // region
        if (c.region == null) {
            c.blocking.add("region is missing");
        } else if (ctx.regionIds() == null || !ctx.regionIds().contains(c.region)) {
            c.blocking.add("region '" + c.region + "' is unknown (not in region_list, the fabs or the transit map)");
        }
        return c;
    }

    /** Reads one date key; adds a blocking problem when it is missing or unparseable. */
    private static LocalDate date(Cand c, String key) {
        Object v = c.raw.get(key);
        if (v == null || (v instanceof CharSequence s && s.toString().isBlank())) {
            c.blocking.add(key + " is missing");
            return null;
        }
        LocalDate d = toDate(v);
        if (d == null) c.blocking.add(key + " '" + v + "' is not a date (use YYYY/MM/DD or YYYY-MM-DD)");
        return d;
    }

    /** LocalDate, SnakeYAML's java.util.Date (midnight UTC), or a YYYY/MM/DD / YYYY-MM-DD string; else null. */
    static LocalDate toDate(Object v) {
        if (v instanceof LocalDate d) return d;
        if (v instanceof Date d) return d.toInstant().atZone(ZoneOffset.UTC).toLocalDate();
        if (v instanceof CharSequence s) {
            try {
                return LocalDate.parse(s.toString().trim().replace('-', '/'), DATE_FORMAT);
            } catch (DateTimeParseException e) {
                return null;
            }
        }
        return null;
    }

    /** Dates SnakeYAML (or a caller) produced become yyyy/MM/dd strings; everything else is returned as is. */
    private static Object plainValue(Object v) {
        LocalDate d = (v instanceof LocalDate || v instanceof Date) ? toDate(v) : null;
        return d == null ? v : DATE_FORMAT.format(d);
    }

    private static String text(Object o) {
        if (o == null) return null;
        String s = String.valueOf(o).trim();
        return s.isEmpty() ? null : s;
    }

    private static String label(Cand c) {
        return "trip_list worker " + (c.worker == null ? "?" : c.worker) + " trip "
                + shown(c.start, c.raw.get("start_date")) + ".." + shown(c.end, c.raw.get("end_date"))
                + (c.region == null ? "" : " " + c.region);
    }

    private static String shown(LocalDate parsed, Object original) {
        if (parsed != null) return parsed.toString();
        return original == null ? "?" : String.valueOf(original);
    }
}
