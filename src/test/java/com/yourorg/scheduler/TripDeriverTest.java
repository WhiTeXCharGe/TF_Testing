package com.yourorg.scheduler;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.time.DayOfWeek;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeSet;
import java.util.function.Function;

import org.junit.jupiter.api.Test;

import com.yourorg.scheduler.trip.DeriveResult;
import com.yourorg.scheduler.trip.Flexibility;
import com.yourorg.scheduler.trip.Trip;
import com.yourorg.scheduler.trip.TripDeriver;
import com.yourorg.scheduler.trip.WorkedDay;

/** Output trip derivation (design section 6): runs, FI/FO, barrier, cut, neighbour check. */
class TripDeriverTest {

    private static final Function<String, Integer> I3 = r -> 3;

    private static WorkedDay wd(String w, String date, String region) {
        return new WorkedDay(w, LocalDate.parse(date), region);
    }

    private static Trip fixed(String w, String s, String e, String region) {
        return new Trip(w, LocalDate.parse(s), LocalDate.parse(e), region, Flexibility.FIXED, "fixed row", Map.of());
    }

    private static void assertTrip(Trip t, String w, String s, String e, String region) {
        assertEquals(w, t.worker());
        assertEquals(LocalDate.parse(s), t.start(), "start of " + t);
        assertEquals(LocalDate.parse(e), t.end(), "end of " + t);
        assertEquals(region, t.region());
        assertEquals(Flexibility.FLEXIBLE, t.flexibility());
        assertNull(t.description());
        assertTrue(t.extra().isEmpty());
    }

    private static DeriveResult run(List<WorkedDay> days, List<Trip> fixed, Function<String, Integer> iv,
                                    boolean cut) {
        return TripDeriver.derive(days, fixed, iv, cut);
    }

    @Test
    void example_from_design() {
        Trip f = fixed("w1", "2026-09-01", "2026-09-10", "r1");
        DeriveResult r = run(List.of(wd("w1", "2026-09-03", "r1"), wd("w1", "2026-09-04", "r1"),
                wd("w1", "2026-09-09", "r1"), wd("w1", "2026-09-12", "r2"), wd("w1", "2026-09-13", "r2"),
                wd("w1", "2026-09-14", "r2")), List.of(f), I3, true);
        assertEquals(1, r.derived().size());
        assertTrip(r.derived().get(0), "w1", "2026-09-11", "2026-09-15", "r2");
        assertTrue(r.cutFixed().isEmpty());
        assertTrue(r.warnings().isEmpty());
    }

    @Test
    void gap_below_interval_continues() {
        DeriveResult r = run(List.of(wd("w1", "2026-09-01", "r1"), wd("w1", "2026-09-03", "r1")), List.of(), I3, true);
        assertEquals(1, r.derived().size());
        assertTrip(r.derived().get(0), "w1", "2026-08-31", "2026-09-04", "r1");
    }

    @Test
    void gap_equal_interval_splits() {
        DeriveResult r = run(List.of(wd("w1", "2026-09-01", "r1"), wd("w1", "2026-09-05", "r1")), List.of(), I3, true);
        assertEquals(2, r.derived().size());
        assertTrip(r.derived().get(0), "w1", "2026-08-31", "2026-09-02", "r1");
        assertTrip(r.derived().get(1), "w1", "2026-09-04", "2026-09-06", "r1");
    }

    @Test
    void interval_one_splits_on_any_gap() {
        List<WorkedDay> days = List.of(wd("w1", "2026-09-01", "r1"), wd("w1", "2026-09-03", "r1"));
        assertEquals(2, run(days, List.of(), r -> 1, true).derived().size());
        assertEquals(2, run(days, List.of(), r -> 0, true).derived().size());
        assertEquals(2, run(days, List.of(), r -> null, true).derived().size());
        assertEquals(2, run(days, List.of(), null, true).derived().size());
        // consecutive days still continue with interval 1
        assertEquals(1, run(List.of(wd("w1", "2026-09-01", "r1"), wd("w1", "2026-09-02", "r1")), List.of(),
                r -> 1, true).derived().size());
    }

    @Test
    void region_change_shares_boundary_day() {
        DeriveResult r = run(List.of(wd("w1", "2026-09-01", "r1"), wd("w1", "2026-09-02", "r1"),
                wd("w1", "2026-09-03", "r1"), wd("w1", "2026-09-05", "r2"), wd("w1", "2026-09-06", "r2")),
                List.of(), I3, true);
        assertEquals(2, r.derived().size());
        assertTrip(r.derived().get(0), "w1", "2026-08-31", "2026-09-04", "r1");
        assertTrip(r.derived().get(1), "w1", "2026-09-04", "2026-09-07", "r2");
        assertTrue(r.warnings().isEmpty());
    }

    @Test
    void same_region_after_fixed_trip_is_separate() {
        DeriveResult r = run(List.of(wd("w1", "2026-09-11", "r1"), wd("w1", "2026-09-12", "r1"),
                wd("w1", "2026-09-13", "r1"), wd("w1", "2026-09-05", "r1")),
                List.of(fixed("w1", "2026-09-01", "2026-09-10", "r1")), I3, true);
        assertEquals(1, r.derived().size());
        assertTrip(r.derived().get(0), "w1", "2026-09-10", "2026-09-14", "r1");
        assertTrue(r.warnings().isEmpty());
    }

    @Test
    void fixed_trip_is_barrier_for_same_region_runs() {
        // plain gap rule: 09/03 .. 09/08 has gap 4 < interval 10, so without the Fixed trip they merge
        List<WorkedDay> days = List.of(wd("w1", "2026-09-03", "r1"), wd("w1", "2026-09-08", "r1"));
        assertEquals(1, run(days, List.of(), r -> 10, false).derived().size());
        DeriveResult r = run(days, List.of(fixed("w1", "2026-09-05", "2026-09-06", "r1")), r2 -> 10, false);
        assertEquals(2, r.derived().size());
        assertTrip(r.derived().get(0), "w1", "2026-09-02", "2026-09-04", "r1");
        assertTrip(r.derived().get(1), "w1", "2026-09-07", "2026-09-09", "r1");
        assertTrue(r.warnings().isEmpty());
    }

    @Test
    void weekend_fi_fo() {
        DeriveResult r = run(List.of(wd("w1", "2026-09-07", "r1"), wd("w1", "2026-09-11", "r1")), List.of(),
                x -> 10, true);
        assertEquals(1, r.derived().size());
        assertEquals(DayOfWeek.MONDAY, LocalDate.parse("2026-09-07").getDayOfWeek());
        assertTrip(r.derived().get(0), "w1", "2026-09-06", "2026-09-12", "r1");
        assertEquals(DayOfWeek.SUNDAY, r.derived().get(0).start().getDayOfWeek());
        assertEquals(DayOfWeek.SATURDAY, r.derived().get(0).end().getDayOfWeek());
    }

    @Test
    void days_inside_fixed_window_are_not_derived() {
        DeriveResult r = run(List.of(wd("w1", "2026-09-03", "r1"), wd("w1", "2026-09-04", "r1")),
                List.of(fixed("w1", "2026-09-01", "2026-09-10", "r1")), I3, true);
        assertTrue(r.derived().isEmpty());
        assertTrue(r.cutFixed().isEmpty());
    }

    @Test
    void fixed_trip_without_tasks_is_cut_when_allowed() {
        Trip f = fixed("w1", "2026-09-01", "2026-09-10", "r1");
        DeriveResult r = run(List.of(), List.of(f), I3, true);
        assertEquals(1, r.cutFixed().size());
        assertSame(f, r.cutFixed().get(0));
        assertTrue(r.derived().isEmpty());
        assertEquals(1, r.warnings().size());
        assertEquals("cut_fixed_trip worker=w1 start=2026-09-01 end=2026-09-10 region=r1", r.warnings().get(0));
    }

    @Test
    void fixed_trip_without_tasks_kept_when_not_allowed() {
        DeriveResult r = run(List.of(), List.of(fixed("w1", "2026-09-01", "2026-09-10", "r1")), I3, false);
        assertTrue(r.cutFixed().isEmpty());
        assertTrue(r.derived().isEmpty());
        assertTrue(r.warnings().isEmpty());
    }

    @Test
    void other_region_day_inside_does_not_count_as_inside() {
        Trip f = fixed("w1", "2026-09-01", "2026-09-10", "r1");
        List<WorkedDay> days = List.of(wd("w1", "2026-09-05", "r2"));
        DeriveResult cut = run(days, List.of(f), I3, true);
        assertEquals(List.of(f), cut.cutFixed());
        assertEquals(1, cut.derived().size()); // the r2 day is no longer inside any kept trip
        assertTrip(cut.derived().get(0), "w1", "2026-09-04", "2026-09-06", "r2");

        DeriveResult keep = run(days, List.of(f), I3, false);
        assertTrue(keep.cutFixed().isEmpty());
        assertTrue(keep.derived().isEmpty()); // removed as inside the window
    }

    @Test
    void day_on_fixed_trip_boundary_does_not_count_as_inside_for_cut() {
        Trip f = fixed("w1", "2026-09-01", "2026-09-10", "r1");
        DeriveResult r = run(List.of(wd("w1", "2026-09-01", "r1"), wd("w1", "2026-09-10", "r1")), List.of(f), I3,
                true);
        assertEquals(List.of(f), r.cutFixed());
        assertEquals(2, r.derived().size()); // gap 8 >= 3: two runs, no longer inside a kept trip
    }

    @Test
    void cut_trip_is_no_longer_a_barrier() {
        // the fixed trip has no r1 day inside -> cut; the two r1 days then merge into one run
        Trip f = fixed("w1", "2026-09-05", "2026-09-06", "r1");
        DeriveResult r = run(List.of(wd("w1", "2026-09-03", "r1"), wd("w1", "2026-09-08", "r1")), List.of(f),
                x -> 10, true);
        assertEquals(List.of(f), r.cutFixed());
        assertEquals(1, r.derived().size());
        assertTrip(r.derived().get(0), "w1", "2026-09-02", "2026-09-09", "r1");
    }

    @Test
    void tight_transit_warns() {
        DeriveResult r = run(List.of(wd("w1", "2026-09-01", "r1"), wd("w1", "2026-09-02", "r2")), List.of(), I3, true);
        assertEquals(2, r.derived().size());
        assertTrip(r.derived().get(0), "w1", "2026-08-31", "2026-09-02", "r1");
        assertTrip(r.derived().get(1), "w1", "2026-09-02", "2026-09-03", "r2");
        assertEquals(1, r.warnings().size());
        assertEquals("tight_transit worker=w1 region=r2 start=2026-09-02", r.warnings().get(0));
    }

    @Test
    void region_conflict_day_keeps_alphabetical() {
        DeriveResult r = run(List.of(wd("w1", "2026-09-03", "r2"), wd("w1", "2026-09-03", "r1")), List.of(), I3, true);
        assertEquals(1, r.derived().size());
        assertTrip(r.derived().get(0), "w1", "2026-09-02", "2026-09-04", "r1");
        assertEquals(1, r.warnings().size());
        assertTrue(r.warnings().get(0).startsWith("region_conflict_day worker=w1 date=2026-09-03"));
    }

    @Test
    void no_days_no_trips() {
        DeriveResult r = run(List.of(), List.of(), I3, true);
        assertTrue(r.derived().isEmpty() && r.cutFixed().isEmpty() && r.warnings().isEmpty());
    }

    @Test
    void workers_with_only_fixed_trips_produce_no_derived() {
        DeriveResult r = run(List.of(wd("w2", "2026-09-03", "r1")),
                List.of(fixed("w1", "2026-09-01", "2026-09-10", "r1")), I3, false);
        assertEquals(1, r.derived().size());
        assertEquals("w2", r.derived().get(0).worker());
    }

    @Test
    void workers_are_independent() {
        Trip f = fixed("w1", "2026-09-01", "2026-09-10", "r1");
        DeriveResult r = run(List.of(wd("w2", "2026-09-03", "r1"), wd("w2", "2026-09-05", "r2"),
                wd("w1", "2026-09-03", "r1"), wd("w1", "2026-09-20", "r1")), List.of(f), I3, true);
        // w2's r1 day does not count inside w1's trip; w1's derived trip is not affected by w2's regions
        List<Trip> w1 = r.derived().stream().filter(t -> t.worker().equals("w1")).toList();
        List<Trip> w2 = r.derived().stream().filter(t -> t.worker().equals("w2")).toList();
        assertEquals(1, w1.size());
        assertTrip(w1.get(0), "w1", "2026-09-19", "2026-09-21", "r1");
        assertEquals(2, w2.size());
        assertTrip(w2.get(0), "w2", "2026-09-02", "2026-09-04", "r1");
        assertTrip(w2.get(1), "w2", "2026-09-04", "2026-09-06", "r2");
        assertTrue(r.cutFixed().isEmpty());
    }

    @Test
    void cut_depends_only_on_own_worker_days() {
        Trip f = fixed("w1", "2026-09-01", "2026-09-10", "r1");
        DeriveResult r = run(List.of(wd("w2", "2026-09-05", "r1")), List.of(f), I3, true);
        assertEquals(List.of(f), r.cutFixed());
    }

    @Test
    void derived_are_sorted_by_worker_then_start() {
        DeriveResult r = run(List.of(wd("w2", "2026-09-03", "r1"), wd("w1", "2026-09-20", "r1"),
                wd("w1", "2026-09-03", "r1"), wd("w2", "2026-09-30", "r1")), List.of(), I3, true);
        List<String> keys = r.derived().stream().map(t -> t.worker() + "@" + t.start()).toList();
        assertEquals(List.of("w1@2026-09-02", "w1@2026-09-19", "w2@2026-09-02", "w2@2026-09-29"), keys);
    }

    @Test
    void duplicates_deduped() {
        DeriveResult r = run(List.of(wd("w1", "2026-09-03", "r1"), wd("w1", "2026-09-03", "r1")), List.of(), I3, true);
        assertEquals(1, r.derived().size());
        assertTrue(r.warnings().isEmpty());
    }

    @Test
    void null_and_empty_inputs() {
        DeriveResult r = TripDeriver.derive(null, null, null, true);
        assertTrue(r.derived().isEmpty() && r.cutFixed().isEmpty() && r.warnings().isEmpty());
        r = TripDeriver.derive(List.of(), null, I3, false);
        assertTrue(r.derived().isEmpty());
    }

    // ---------------------------------------------------------------- property

    private static List<WorkedDay> randomDays(Random rnd, String worker, LocalDate base) {
        List<WorkedDay> out = new ArrayList<>();
        int day = rnd.nextInt(4);
        String region = rnd.nextBoolean() ? "r1" : "r2";
        while (day < 90) {
            out.add(new WorkedDay(worker, base.plusDays(day), region));
            String next = rnd.nextInt(4) == 0 ? (region.equals("r1") ? "r2" : "r1") : region;
            int gap = next.equals(region) ? rnd.nextInt(6) : 3 + rnd.nextInt(3); // days off in between
            day += 1 + gap;
            region = next;
        }
        return out;
    }

    private static List<Trip> randomFixed(Random rnd, String worker, LocalDate base) {
        List<Trip> out = new ArrayList<>();
        int n = rnd.nextInt(3);
        int cursor = rnd.nextInt(20);
        for (int i = 0; i < n; i++) {
            int len = 2 + rnd.nextInt(9);
            if (cursor + len >= 90) break;
            out.add(new Trip(worker, base.plusDays(cursor), base.plusDays(cursor + len),
                    rnd.nextBoolean() ? "r1" : "r2", Flexibility.FIXED, "f" + i, Map.of("k", i)));
            cursor += len + 1 + rnd.nextInt(25);
        }
        return out;
    }

    @Test
    void invariants_property() {
        Random rnd = new Random(42);
        LocalDate base = LocalDate.parse("2026-07-01");
        for (int scenario = 0; scenario < 200; scenario++) {
            int workers = 1 + rnd.nextInt(3);
            List<WorkedDay> days = new ArrayList<>();
            List<Trip> fixedAll = new ArrayList<>();
            for (int w = 0; w < workers; w++) {
                String name = "w" + w;
                days.addAll(randomDays(rnd, name, base));
                fixedAll.addAll(randomFixed(rnd, name, base));
            }
            List<Trip> fixedBefore = List.copyOf(fixedAll);
            DeriveResult res = TripDeriver.derive(days, fixedAll, I3, true);
            String ctx = "scenario " + scenario;

            // pure: same input, same output; input untouched
            assertEquals(res, TripDeriver.derive(days, fixedAll, I3, true), ctx);
            assertEquals(fixedBefore, fixedAll, ctx);

            // cutFixed holds the very input instances
            for (Trip c : res.cutFixed()) {
                assertTrue(fixedAll.stream().anyMatch(f -> f == c), ctx);
            }
            assertFalse(res.warnings().stream().anyMatch(s -> s.startsWith("tight_transit")), ctx);
            assertFalse(res.warnings().stream().anyMatch(s -> s.startsWith("region_conflict_day")), ctx);

            // derived sorted by worker then start
            List<Trip> sorted = new ArrayList<>(res.derived());
            sorted.sort(Comparator.comparing(Trip::worker).thenComparing(Trip::start));
            assertEquals(sorted, res.derived(), ctx);

            for (int w = 0; w < workers; w++) {
                String name = "w" + w;
                List<Trip> remaining = new ArrayList<>();
                for (Trip f : fixedAll) {
                    if (f.worker().equals(name) && res.cutFixed().stream().noneMatch(c -> c == f)) remaining.add(f);
                }
                // cut rule: cut iff no same-region day strictly inside
                for (Trip f : fixedAll) {
                    if (!f.worker().equals(name)) continue;
                    boolean cut = res.cutFixed().stream().anyMatch(c -> c == f);
                    boolean has = days.stream().anyMatch(d -> d.worker().equals(name) && d.region().equals(f.region())
                            && d.date().isAfter(f.start()) && d.date().isBefore(f.end()));
                    assertEquals(!has, cut, ctx + " " + f);
                }
                List<Trip> mine = res.derived().stream().filter(t -> t.worker().equals(name)).toList();
                List<Trip> all = new ArrayList<>(mine);
                all.addAll(remaining);
                all.sort(Comparator.comparing(Trip::start).thenComparing(Trip::end));
                for (int i = 1; i < all.size(); i++) {
                    assertFalse(all.get(i).start().isBefore(all.get(i - 1).end()),
                            ctx + " overlap " + all.get(i - 1) + " / " + all.get(i));
                }
                for (Trip t : mine) {
                    assertTrue(t.start().isBefore(t.end()), ctx);
                }
                // every worked day outside the remaining windows lies strictly inside exactly one own-region trip
                TreeSet<LocalDate> covered = new TreeSet<>();
                for (WorkedDay d : days) {
                    if (!d.worker().equals(name)) continue;
                    boolean inside = remaining.stream()
                            .anyMatch(f -> !d.date().isBefore(f.start()) && !d.date().isAfter(f.end()));
                    List<Trip> hit = mine.stream()
                            .filter(t -> t.start().isBefore(d.date()) && t.end().isAfter(d.date())).toList();
                    if (inside) {
                        assertEquals(0, hit.size(), ctx + " " + d);
                    } else {
                        assertEquals(1, hit.size(), ctx + " " + d);
                        assertEquals(hit.get(0).region(), d.region(), ctx + " " + d);
                        covered.add(d.date());
                    }
                }
                // FI/FO are exactly one day outside the first/last worked day of each trip
                for (Trip t : mine) {
                    List<LocalDate> in = covered.stream()
                            .filter(x -> t.start().isBefore(x) && t.end().isAfter(x)).toList();
                    assertFalse(in.isEmpty(), ctx + " " + t);
                    assertEquals(t.start().plusDays(1), in.get(0), ctx + " " + t);
                    assertEquals(t.end().minusDays(1), in.get(in.size() - 1), ctx + " " + t);
                }
            }
        }
    }
}
