package com.yourorg.scheduler;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.time.LocalDate;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import com.yourorg.scheduler.calendar.RegionIndex;
import com.yourorg.scheduler.calendar.SolveContext;
import com.yourorg.scheduler.calendar.TripCalendar;
import com.yourorg.scheduler.calendar.WorkerCalendar;
import com.yourorg.scheduler.trip.Flexibility;
import com.yourorg.scheduler.trip.Trip;
import com.yourorg.scheduler.trip.TripEntry;

/**
 * Trip marks (design section 5): the start and end day of an enforced Fixed trip are blocked, the days strictly
 * between are open only for the trip's region, and {@link WorkerCalendar#isOpen} is the one test over them.
 */
class TripCalendarTest {

    /** dayId 0. */
    static final LocalDate PLAN_START = LocalDate.of(2026, 9, 1);
    static final int HORIZON = 30; // dayIds 0..29 = 09/01..09/30
    static final RegionIndex REGIONS = RegionIndex.of(List.of("r_other", "r1", "r2"));
    static final int R1 = REGIONS.indexOf("r1");
    static final int R2 = REGIONS.indexOf("r2");
    static final int W1 = 1, W2 = 2;

    static final Function<String, Integer> WORKERS;
    static {
        Map<String, Integer> m = new HashMap<>();
        m.put("w1", W1);
        m.put("w2", W2);
        WORKERS = m::get;
    }

    @BeforeEach
    @AfterEach
    void resetContext() {
        SolveContext.clear();
    }

    static TripEntry fixed(String worker, String start, String end, String region) {
        return new TripEntry(new Trip(worker, d(start), d(end), region, Flexibility.FIXED, null, Map.of()),
                Map.of(), true);
    }

    static LocalDate d(String mmdd) {
        return LocalDate.parse("2026-" + mmdd);
    }

    static int day(String mmdd) {
        return (int) (d(mmdd).toEpochDay() - PLAN_START.toEpochDay());
    }

    /** Builds the calendar and installs it in the current SolveContext (isOpen reads it from there). */
    static TripCalendar install(TripEntry... entries) {
        TripCalendar c = TripCalendar.build(List.of(entries), WORKERS, REGIONS,
                (int) PLAN_START.toEpochDay(), HORIZON);
        SolveContext.current().tripCalendar = c;
        return c;
    }

    @Test
    void blocked_on_start_and_end() {
        TripCalendar c = install(fixed("w1", "09-03", "09-08", "r1"));
        assertTrue(c.hasMarks(W1));
        assertTrue(c.isBlocked(W1, day("09-03")));
        assertTrue(c.isBlocked(W1, day("09-08")));
        assertFalse(c.isBlocked(W1, day("09-02")));
        assertFalse(c.isBlocked(W1, day("09-05")));
        assertFalse(c.isBlocked(W1, day("09-09")));
        // closed even for the trip's own region
        assertFalse(WorkerCalendar.isOpen(W1, day("09-03"), R1));
        assertFalse(WorkerCalendar.isOpen(W1, day("09-08"), R1));
        assertFalse(WorkerCalendar.isOpen(W1, day("09-08"), R2));
    }

    @Test
    void interior_locked_to_region() {
        TripCalendar c = install(fixed("w1", "09-03", "09-08", "r1"));
        for (int dd = day("09-04"); dd <= day("09-07"); dd++) {
            assertEquals(R1, c.lockRegion(W1, dd), "day " + dd);
            assertTrue(WorkerCalendar.isOpen(W1, dd, R1), "same region open on " + dd);
            assertFalse(WorkerCalendar.isOpen(W1, dd, R2), "other region closed on " + dd);
        }
        assertEquals(-1, c.lockRegion(W1, day("09-03")));
        assertEquals(-1, c.lockRegion(W1, day("09-08")));
        assertEquals(-1, c.lockRegion(W1, day("09-02")));
        assertEquals(-1, c.lockRegion(W1, day("09-09")));
        // outside [start, end] nothing is implied
        assertTrue(WorkerCalendar.isOpen(W1, day("09-02"), R2));
        assertTrue(WorkerCalendar.isOpen(W1, day("09-09"), R2));
        // another worker is not touched
        assertTrue(WorkerCalendar.isOpen(W2, day("09-05"), R2));
    }

    @Test
    void touching_trips_block_shared_day() {
        TripCalendar c = install(fixed("w1", "09-03", "09-08", "r1"), fixed("w1", "09-08", "09-12", "r2"));
        assertTrue(c.isBlocked(W1, day("09-08")));
        assertEquals(-1, c.lockRegion(W1, day("09-08")));
        assertFalse(WorkerCalendar.isOpen(W1, day("09-08"), R1));
        assertFalse(WorkerCalendar.isOpen(W1, day("09-08"), R2));
        assertEquals(R1, c.lockRegion(W1, day("09-07")));
        assertEquals(R2, c.lockRegion(W1, day("09-09")));
        assertTrue(c.isBlocked(W1, day("09-12")));
    }

    @Test
    void outside_horizon_clamped() {
        TripCalendar c = install(
                fixed("w1", "08-25", "09-04", "r1"),  // starts before the plan
                fixed("w1", "09-28", "10-10", "r2"),  // ends after the plan
                fixed("w2", "10-05", "10-09", "r1")); // fully after the plan
        assertTrue(c.isBlocked(W1, day("09-04")));
        assertEquals(R1, c.lockRegion(W1, 0));
        assertEquals(R1, c.lockRegion(W1, day("09-03")));
        assertTrue(c.isBlocked(W1, day("09-28")));
        assertEquals(R2, c.lockRegion(W1, day("09-29")));
        assertEquals(R2, c.lockRegion(W1, HORIZON - 1));
        // outside the horizon: no marks, no exception
        assertFalse(c.isBlocked(W1, -1));
        assertFalse(c.isBlocked(W1, HORIZON));
        assertFalse(c.isBlocked(W1, 999));
        assertEquals(-1, c.lockRegion(W1, -7));
        assertEquals(-1, c.lockRegion(W1, HORIZON + 3));
        assertEquals(-1, c.presenceRegion(W1, HORIZON));
        assertTrue(WorkerCalendar.isOpen(W1, -1, R2));
        assertTrue(WorkerCalendar.isOpen(W1, HORIZON, R1));
        // a trip that lies fully outside leaves nothing
        assertFalse(c.hasMarks(W2));
    }

    @Test
    void worker_without_trips_uses_fast_path() {
        TripCalendar c = install(fixed("w1", "09-03", "09-08", "r1"));
        assertFalse(c.hasMarks(W2));
        assertFalse(c.hasMarks(0));
        assertFalse(c.hasMarks(-1));
        assertFalse(c.hasMarks(999));
        for (int emp : new int[] {W2, 0, -1, 999}) {
            for (int dd : new int[] {-5, 0, day("09-03"), day("09-05"), HORIZON, 10_000}) {
                assertTrue(WorkerCalendar.isOpen(emp, dd, R2));
                assertTrue(WorkerCalendar.isOpen(emp, dd, -1));
                assertFalse(c.isBlocked(emp, dd));
                assertEquals(-1, c.lockRegion(emp, dd));
                assertEquals(-1, c.presenceRegion(emp, dd));
            }
        }
        // the default calendar of a fresh context has no marks at all
        SolveContext.clear();
        assertFalse(SolveContext.current().tripCalendar.hasMarks(W1));
        assertTrue(WorkerCalendar.isOpen(W1, day("09-03"), R1));
        assertTrue(WorkerCalendar.isOpen(W1, day("09-05"), R2));
    }

    @Test
    void flexible_and_unenforced_entries_leave_no_marks() {
        Trip flexible = new Trip("w1", d("09-03"), d("09-08"), "r1", Flexibility.FLEXIBLE, null, Map.of());
        Trip fixedTrip = new Trip("w2", d("09-03"), d("09-08"), "r1", Flexibility.FIXED, null, Map.of());
        TripCalendar c = install(
                new TripEntry(flexible, Map.of(), false),
                new TripEntry(fixedTrip, Map.of(), false),           // a Fixed trip that is not enforced
                new TripEntry(null, Map.of("worker", "w1"), false, "w1")); // a broken trip
        assertFalse(c.hasMarks(W1));
        assertFalse(c.hasMarks(W2));
        assertTrue(WorkerCalendar.isOpen(W1, day("09-03"), R2));
        assertTrue(WorkerCalendar.isOpen(W2, day("09-05"), R2));
    }

    @Test
    void presence_region_covers_start_to_end() {
        TripCalendar c = install(fixed("w1", "09-03", "09-08", "r1"));
        for (int dd = day("09-03"); dd <= day("09-08"); dd++) {
            assertEquals(R1, c.presenceRegion(W1, dd), "day " + dd);
        }
        assertEquals(-1, c.presenceRegion(W1, day("09-02")));
        assertEquals(-1, c.presenceRegion(W1, day("09-09")));
    }

    @Test
    void unknown_worker_ignored() {
        TripCalendar c = install(fixed("ghost", "09-03", "09-08", "r1"), fixed("w2", "09-10", "09-12", "r2"));
        assertFalse(c.hasMarks(W1));
        assertTrue(c.hasMarks(W2));
        assertTrue(c.isBlocked(W2, day("09-10")));
    }

    @Test
    void isOpen_region_less_seat_is_closed_inside_window() {
        install(fixed("w1", "09-03", "09-08", "r1"));
        assertFalse(WorkerCalendar.isOpen(W1, day("09-05"), -1));
        assertFalse(WorkerCalendar.isOpen(W1, day("09-03"), -1));
        assertTrue(WorkerCalendar.isOpen(W1, day("09-10"), -1));
    }
}
