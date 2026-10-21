package com.yourorg.scheduler;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.HashSet;
import java.util.Set;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import com.yourorg.scheduler.model.CrewSeat;
import com.yourorg.scheduler.model.SinglePassPlan;

/**
 * Business rule: seat 0 of a block must be a manager. A manager seat that holds a manager at the start must never
 * be emptied (UNASSIGNED) by the search. Timefold 1.35 validates initial values against the value range, which
 * tempts one to put UNASSIGNED into the manager range; this test guards against that weakening the rule.
 */
class ManagerSeatKeepsManagerTest {

    @TempDir
    Path tmp;

    private void check(Path fixture, int steps) throws Exception {
        SinglePassPlan p = Characterization.buildFrom(fixture, tmp.resolve("p"));
        Set<Integer> heldManager = new HashSet<>();
        for (CrewSeat s : p.seats) {
            if (!s.pinnedFixed && s.needManager && s.employee != null && s.employee.id != 0 && s.employee.isManager) {
                heldManager.add(s.id);
            }
        }
        assertFalse(heldManager.isEmpty(), "fixture has no manager seat holding a manager at the start");

        SinglePassPlan best = Characterization.solveSeeded(p, steps, CharacterizationTest.SEED);
        for (CrewSeat s : best.seats) {
            if (heldManager.contains(s.id)) {
                assertTrue(s.employee != null && s.employee.id != 0 && s.employee.isManager,
                        "manager seat " + s.id + " was emptied or given a non-manager: " + s.employee);
            }
        }
    }

    @Test
    void small_fixture_2000_steps() throws Exception {
        check(CharacterizationTest.SMALL_FIXTURE, 2000);
    }

    /**
     * Manager seats that start UNASSIGNED (datagen has no warm start) may be filled by the search, but a seat that
     * holds a manager in one best solution must not be UNASSIGNED in a later one.
     */
    @Test
    void datagen_manager_seat_filled_by_search_is_never_emptied_again() throws Exception {
        SinglePassPlan p = Characterization.buildFrom(CharacterizationTest.DATAGEN_FIXTURE, tmp.resolve("hist"));
        Set<Integer> filled = new HashSet<>();
        java.util.List<String> violations = new java.util.ArrayList<>();
        ai.timefold.solver.core.api.solver.Solver<SinglePassPlan> solver = com.yourorg.scheduler.solver.SolverFactories
                .forTests(2000, CharacterizationTest.SEED).buildSolver();
        solver.addEventListener(ev -> {
            for (CrewSeat s : ev.getNewBestSolution().seats) {
                if (s.pinnedFixed || !s.needManager) continue;
                boolean holdsManager = s.employee != null && s.employee.id != 0;
                if (holdsManager) filled.add(s.id);
                else if (filled.contains(s.id)) violations.add("seat " + s.id);
            }
        });
        solver.solve(p);
        assertFalse(filled.isEmpty(), "search never filled a manager seat, test proves nothing");
        assertTrue(violations.isEmpty(), "manager seats emptied again after being filled: " + violations);
    }
}
