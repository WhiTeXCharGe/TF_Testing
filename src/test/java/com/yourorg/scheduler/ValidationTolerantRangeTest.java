package com.yourorg.scheduler;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.junit.jupiter.api.Test;

import com.yourorg.scheduler.model.CrewSeat;

import ai.timefold.solver.core.api.domain.valuerange.ValueRange;
import com.yourorg.scheduler.model.EmployeeFact;

/**
 * The initial value of a seat is accepted by {@code contains} (Timefold 1.35 validation) but is invisible to the
 * search: not in the size, not in the original or random iterator, and no other instance with the same id is accepted.
 */
class ValidationTolerantRangeTest {

    private static EmployeeFact emp(int id, boolean mgr) {
        return new EmployeeFact(id, "w" + id, "n" + id, Map.of(), mgr, "c");
    }

    private static EmployeeFact freshUnassigned() {
        return new EmployeeFact(0, "__UNASSIGNED__", "__UNASSIGNED__", Map.of(), false, "");
    }

    private static List<EmployeeFact> everythingOffered(ValueRange<EmployeeFact> r, Random rnd) {
        List<EmployeeFact> seen = new ArrayList<>();
        for (Iterator<EmployeeFact> it = r.createOriginalIterator(); it.hasNext();) seen.add(it.next());
        Iterator<EmployeeFact> it = r.createRandomIterator(rnd);
        for (int i = 0; i < 500; i++) seen.add(it.next());
        for (long i = 0; i < r.getSize(); i++) seen.add(r.get(i));
        return seen;
    }

    @Test
    void manager_seat_accepts_its_initial_unassigned_copy_but_never_offers_it() {
        EmployeeFact m1 = emp(1, true), m2 = emp(2, true), worker = emp(3, false);
        CrewSeat seat = new CrewSeat();
        seat.needManager = true;
        seat.setCandidateEmployees(List.of(m1, m2, worker));
        EmployeeFact initial = freshUnassigned();
        seat.employee = initial;

        ValueRange<EmployeeFact> r = seat.eligibleEmployeesForSeat();
        assertTrue(r.contains(initial), "initial value must pass validation");
        assertFalse(r.contains(freshUnassigned()), "another id-0 instance must not be accepted");
        assertTrue(r.contains(m1));
        assertTrue(r.contains(m2));
        assertFalse(r.contains(worker), "non-manager is not in the manager range");
        assertEquals(2, r.getSize());
        for (EmployeeFact e : everythingOffered(r, new Random(1))) {
            assertTrue(e == m1 || e == m2, "range offered " + e.wid);
        }
    }

    @Test
    void pinned_seat_accepts_its_real_employee_when_pinnedWid_is_missing() {
        EmployeeFact real = emp(5, false);
        CrewSeat seat = new CrewSeat();
        seat.pinnedFixed = true; // pinnedWid stays null, as PlanBuilder builds it
        seat.setCandidateEmployees(List.of());
        seat.employee = real;
        ValueRange<EmployeeFact> r = seat.eligibleEmployeesForSeat();
        assertTrue(r.contains(real));
        assertEquals(1, r.getSize());
        for (EmployeeFact e : everythingOffered(r, new Random(2))) assertEquals(0, e.id);
    }

    @Test
    void warm_start_employee_failing_the_gates_is_accepted_but_not_offered() {
        EmployeeFact ok = emp(1, false), gated = emp(9, false);
        CrewSeat seat = new CrewSeat();
        seat.setCandidateEmployees(List.of(ok));
        seat.employee = gated;
        ValueRange<EmployeeFact> r = seat.eligibleEmployeesForSeat();
        assertTrue(r.contains(gated));
        assertEquals(2, r.getSize()); // ok + UNASSIGNED, exactly the original range
        for (EmployeeFact e : everythingOffered(r, new Random(3))) assertTrue(e != gated);
        // once the seat holds another value, a newly built range no longer tolerates the old one
        seat.employee = ok;
        assertFalse(seat.eligibleEmployeesForSeat().contains(gated));
    }
}
