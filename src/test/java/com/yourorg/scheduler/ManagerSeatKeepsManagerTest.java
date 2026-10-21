package com.yourorg.scheduler;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import com.yourorg.scheduler.model.CrewSeat;
import com.yourorg.scheduler.model.SinglePassPlan;
import com.yourorg.scheduler.solver.SolverFactories;

import ai.timefold.solver.core.api.solver.Solver;

/**
 * Business rule: seat 0 of a block must be a manager. Once a manager seat holds a manager (from the warm start or
 * because the search filled it), the search must never empty it again (UNASSIGNED). Timefold 1.35 validates initial
 * values against the value range, which tempts one to widen the manager range with UNASSIGNED; that makes the search
 * re-empty manager seats, visible only after thousands of steps (datagen, seed 42: 3 seats at 6000 steps, 19 at 15000),
 * so the runs below are long enough to see it. Every best solution is inspected, not only the final one.
 */
class ManagerSeatKeepsManagerTest {

    static final int STEPS = 6000;

    @TempDir
    Path tmp;

    private static boolean staffed(CrewSeat s) {
        return s.employee != null && s.employee.id != 0;
    }

    /** Manager seats that held somebody (initially or in an earlier best solution) and are UNASSIGNED in a later one. */
    private List<String> reEmptied(Path fixture, int steps) throws Exception {
        SinglePassPlan p = Characterization.buildFrom(fixture, tmp.resolve("p"));
        Set<Integer> filled = new HashSet<>();
        for (CrewSeat s : p.seats) {
            if (!s.pinnedFixed && s.needManager && staffed(s)) filled.add(s.id);
        }
        List<String> violations = new ArrayList<>();
        Solver<SinglePassPlan> solver = SolverFactories.forTests(steps, CharacterizationTest.SEED).buildSolver();
        solver.addEventListener(ev -> {
            for (CrewSeat s : ev.getNewBestSolution().seats) {
                if (s.pinnedFixed || !s.needManager) continue;
                if (staffed(s)) filled.add(s.id);
                else if (filled.contains(s.id)) violations.add("seat " + s.id);
            }
        });
        SinglePassPlan best = solver.solve(p);
        for (CrewSeat s : best.seats) {
            if (!s.pinnedFixed && s.needManager && staffed(s)) {
                assertTrue(s.employee.isManager, "non-manager on manager seat " + s.id);
            }
        }
        assertFalse(filled.isEmpty(), "no manager seat was ever filled, the test proves nothing");
        return violations;
    }

    @Test
    void small_fixture_manager_seats_are_never_emptied_again() throws Exception {
        List<String> v = reEmptied(CharacterizationTest.SMALL_FIXTURE, STEPS);
        assertTrue(v.isEmpty(), "manager seats emptied again: " + v);
    }

    @Test
    void datagen_fixture_manager_seats_are_never_emptied_again() throws Exception {
        List<String> v = reEmptied(CharacterizationTest.DATAGEN_FIXTURE, STEPS);
        assertTrue(v.isEmpty(), "manager seats emptied again: " + v);
    }
}
