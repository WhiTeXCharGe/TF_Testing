package com.yourorg.scheduler;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import com.yourorg.scheduler.build.PlanBuilder;
import com.yourorg.scheduler.model.BlockDecision;
import com.yourorg.scheduler.model.CrewSeat;
import com.yourorg.scheduler.model.DaySlot;
import com.yourorg.scheduler.model.EmployeeFact;
import com.yourorg.scheduler.model.InitialAssign;
import com.yourorg.scheduler.model.ParsedSchedule;
import com.yourorg.scheduler.model.SinglePassPlan;

/**
 * {@code PlanBuilder.normalizeInitialValues} (rulings R1/R1b) and the phase-1 clamp of {@code applyInitialPlan} (R2):
 * how warm-start values are brought into the STRICT value ranges that Timefold 2.x validates once per solve.
 */
class NormalizeInitialValuesTest {

    @TempDir
    Path tmp;

    private SinglePassPlan small(String name) throws Exception {
        return Characterization.buildFrom(CharacterizationTest.SMALL_FIXTURE, tmp.resolve(name));
    }

    private static EmployeeFact unassignedCopy() {
        return new EmployeeFact(0, "__UNASSIGNED__", "__UNASSIGNED__", Map.of(), false, "");
    }

    private static boolean containsIdentity(List<EmployeeFact> l, EmployeeFact e) {
        for (EmployeeFact x : l) if (x == e) return true;
        return false;
    }

    private static CrewSeat firstSeat(SinglePassPlan p, boolean manager) {
        for (CrewSeat s : p.seats) if (!s.pinnedFixed && s.blockId != 0 && s.needManager == manager) return s;
        return null;
    }

    /** A real employee outside the seat's strict range (fails its candidate gates). */
    private static EmployeeFact outOfGate(SinglePassPlan p, CrewSeat s, boolean wantManager) {
        List<EmployeeFact> strict = s.strictEligibleEmployees();
        for (EmployeeFact e : p.employees) {
            if (e != null && e.id != 0 && e.isManager == wantManager && !containsIdentity(strict, e)) return e;
        }
        return null;
    }

    @Test
    void unassigned_copy_on_non_manager_seat_becomes_the_shared_sentinel() throws Exception {
        SinglePassPlan p = small("copy");
        CrewSeat s = firstSeat(p, false);
        assertNotNull(s);
        EmployeeFact copy = unassignedCopy();
        s.employee = copy;
        assertFalse(containsIdentity(s.eligibleEmployeesForSeat(), copy), "a copy is never in the range");

        Map<String, Integer> c = PlanBuilder.normalizeInitialValues(p);
        assertSame(CrewSeat.UNASSIGNED, s.employee);
        assertTrue(containsIdentity(s.eligibleEmployeesForSeat(), CrewSeat.UNASSIGNED));
        assertEquals(1, c.get("seat UNASSIGNED copy -> shared UNASSIGNED"), String.valueOf(c));
    }

    @Test
    void unassigned_on_manager_seat_becomes_null() throws Exception {
        SinglePassPlan p = small("mgr");
        CrewSeat s = firstSeat(p, true);
        assertNotNull(s);
        for (EmployeeFact copy : new EmployeeFact[] {unassignedCopy(), CrewSeat.UNASSIGNED}) {
            s.employee = copy;
            Map<String, Integer> c = PlanBuilder.normalizeInitialValues(p);
            assertNull(s.employee, "UNASSIGNED on a manager seat is re-planned by the construction heuristic");
            assertNull(s.warmOutOfGateEmployee, "UNASSIGNED is never appended to a range");
            assertEquals(1, c.get("manager seat UNASSIGNED -> null"), String.valueOf(c));
            for (EmployeeFact e : s.eligibleEmployeesForSeat()) {
                assertTrue(e.id != 0 && e.isManager, "manager range offers " + e.wid);
            }
        }
    }

    @Test
    void out_of_gate_warm_employee_is_kept_and_appended_to_only_that_seats_range() throws Exception {
        SinglePassPlan p = small("warm");
        CrewSeat seat = firstSeat(p, false);
        assertNotNull(seat);
        EmployeeFact gated = outOfGate(p, seat, false);
        if (gated == null) gated = outOfGate(p, seat, true);
        assertNotNull(gated, "the small fixture has no employee failing the gates of seat " + seat.id);
        final EmployeeFact warm = gated;
        int strictSize = seat.strictEligibleEmployees().size();

        seat.employee = warm;
        Map<String, Integer> c = PlanBuilder.normalizeInitialValues(p);
        assertSame(warm, seat.employee, "kept");
        assertSame(warm, seat.warmOutOfGateEmployee);
        List<EmployeeFact> range = seat.eligibleEmployeesForSeat();
        assertEquals(strictSize + 1, range.size(), "exactly that one employee appended");
        assertSame(warm, range.get(range.size() - 1));
        assertEquals(1, c.get("seat warm employee outside candidates -> kept (range + employee)"), String.valueOf(c));

        // no other seat's range is widened
        for (CrewSeat other : p.seats) {
            if (other == seat || other.pinnedFixed) continue;
            assertNull(other.warmOutOfGateEmployee, "seat " + other.id);
            assertEquals(containsIdentity(other.strictEligibleEmployees(), warm),
                    containsIdentity(other.eligibleEmployeesForSeat(), warm), "seat " + other.id);
        }
        // idempotent: a second pass changes nothing
        assertTrue(PlanBuilder.normalizeInitialValues(p).isEmpty());
        // Timefold 2.x accepts it: score and a few steps run
        assertDoesNotThrow(() -> Characterization.analyze(p));
        assertDoesNotThrow(() -> Characterization.solveSeeded(p, 50, CharacterizationTest.SEED));
    }

    @Test
    void non_manager_warm_started_on_manager_seat_is_kept_but_unassigned_never_offered() throws Exception {
        SinglePassPlan p = small("warm-mgr");
        CrewSeat seat = firstSeat(p, true);
        assertNotNull(seat);
        EmployeeFact worker = outOfGate(p, seat, false);
        assertNotNull(worker, "no non-manager in the small fixture");
        seat.employee = worker;
        Map<String, Integer> c = PlanBuilder.normalizeInitialValues(p);
        assertSame(worker, seat.employee);
        assertEquals(1, c.get("manager seat warm employee outside candidates -> kept (range + employee)"),
                String.valueOf(c));
        List<EmployeeFact> range = seat.eligibleEmployeesForSeat();
        assertTrue(containsIdentity(range, worker));
        for (EmployeeFact e : range) assertTrue(e.id != 0, "UNASSIGNED in a manager range");
        assertDoesNotThrow(() -> Characterization.analyze(p));
    }

    @Test
    void pinned_seat_range_is_exactly_its_own_employee() throws Exception {
        SinglePassPlan p = small("pinned");
        int pinned = 0;
        for (CrewSeat s : p.seats) {
            if (!s.pinnedFixed) continue;
            pinned++;
            assertNotNull(s.employee);
            List<EmployeeFact> range = s.eligibleEmployeesForSeat();
            assertEquals(1, range.size(), "seat " + s.id);
            assertSame(s.employee, range.get(0), "seat " + s.id);
        }
        assertTrue(pinned > 0, "the small fixture has no pinned seat");
        List<EmployeeFact> before = new ArrayList<>();
        for (CrewSeat s : p.seats) before.add(s.employee);
        PlanBuilder.normalizeInitialValues(p);
        for (int i = 0; i < p.seats.size(); i++) {
            if (p.seats.get(i).pinnedFixed) assertSame(before.get(i), p.seats.get(i).employee);
        }
    }

    @Test
    void fixtures_build_already_normalised() throws Exception {
        for (Path fx : new Path[] {CharacterizationTest.TINY_FIXTURE, CharacterizationTest.SMALL_FIXTURE}) {
            SinglePassPlan p = Characterization.buildFrom(fx, tmp.resolve("n-" + fx.getFileName()));
            assertTrue(PlanBuilder.normalizeInitialValues(p).isEmpty(), "second pass on " + fx);
            for (CrewSeat s : p.seats) {
                if (s.employee == null) {
                    assertTrue(s.needManager, "only manager seats start null: seat " + s.id);
                    continue;
                }
                assertTrue(containsIdentity(s.eligibleEmployeesForSeat(), s.employee), "seat " + s.id);
            }
        }
    }

    // ------------------------------------------------------------------ phase-1 clamp (R2)

    private static ParsedSchedule schedule(int fromDay, int toDay, List<InitialAssign> rows) {
        ParsedSchedule sch = new ParsedSchedule();
        sch.daySlots = new ArrayList<>();
        for (int d = fromDay; d <= toDay; d++) sch.daySlots.add(new DaySlot(d, LocalDate.of(2026, 1, 1).plusDays(d)));
        sch.initialRows = rows;
        return sch;
    }

    private static BlockDecision block(int id, int phaseNum, String op) {
        BlockDecision b = new BlockDecision();
        b.id = id;
        b.module = "m";
        b.opId = op;
        b.phaseNum = phaseNum;
        b.windowStart = 10;
        b.windowEnd = 20;
        b.allowed = List.of(8);
        return b;
    }

    private static InitialAssign row(String op, int... days) {
        InitialAssign ia = new InitialAssign();
        ia.module = "m";
        ia.opId = op;
        for (int d : days) ia.hoursByDay.put(d, 8);
        return ia;
    }

    @Test
    void phase1_warm_start_after_window_start_is_clamped_to_window_start() {
        BlockDecision p1 = block(1, 1, "o1");
        BlockDecision p2 = block(2, 2, "o2");
        ParsedSchedule sch = schedule(0, 30, List.of(row("o1", 12, 13, 14), row("o2", 12, 13, 14)));
        PlanBuilder.applyInitialPlan(sch, List.of(p1, p2));

        assertEquals(10, p1.startDay, "phase 1 starts at windowStart");
        assertEquals(5, p1.days, "warm end day 14 kept: 10..14");
        assertEquals(List.of(10), p1.vrStartWithinWindow());
        assertTrue(p1.vrStartWithinWindow().contains(p1.startDay));

        assertEquals(12, p2.startDay, "other phases keep the warm start day");
        assertEquals(3, p2.days);
    }

    @Test
    void phase1_block_with_all_warm_days_after_window_end_is_not_warm_started() throws Exception {
        BlockDecision p1 = block(77, 1, "o1"); // window 10..20
        ParsedSchedule sch = schedule(0, 30, List.of(row("o1", 25, 26)));
        Path log = Path.of("load_log.txt");
        long before = Files.exists(log) ? Files.readAllLines(log).size() : 0;

        PlanBuilder.applyInitialPlan(sch, List.of(p1));

        assertNull(p1.startDay, "skipped as on 1.35: left for the construction heuristic");
        assertNull(p1.days);
        assertNull(p1.hours);
        List<String> lines = Files.exists(log) ? Files.readAllLines(log) : List.of();
        for (String l : lines.subList((int) Math.min(before, lines.size()), lines.size())) {
            assertFalse(l.contains("blockId=77"), "no INIT-BLOCK / PHASE1-CLAMP log for a skipped block: " + l);
        }
    }
}
