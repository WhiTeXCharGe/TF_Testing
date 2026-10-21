package com.yourorg.scheduler;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;

import org.junit.jupiter.api.Test;

import ai.timefold.solver.core.api.score.stream.ConstraintMetaModel;
import ai.timefold.solver.core.api.solver.SolverFactory;
import ai.timefold.solver.core.impl.score.stream.common.AbstractConstraintStreamScoreDirectorFactory;
import ai.timefold.solver.core.impl.solver.DefaultSolverFactory;

import com.yourorg.scheduler.constraints.SinglePassConstraints;
import com.yourorg.scheduler.model.SinglePassPlan;

/**
 * Pins the constraint ids and their registration order. The ids come from the solver's own constraint
 * meta-model, i.e. from what {@link SinglePassConstraints#defineConstraints} really registers.
 */
class ConstraintNamesTest {

    static final List<String> EXPECTED_IDS = List.of(
            // block feasibility
            "block-end-within-window",
            "block-hours-in-allowed",
            "phase-order",
            // production & capacity
            "block-no-underfill",
            "block-overfill-at-most-one-day",
            // seat-level hard rules
            "emp-one-task-per-day",
            "seat-daily-cap-12h",
            "emp-region-transit-gap",
            "emp-region-stay-max-on",
            "emp-region-annual-stay-max",
            "emp-annual-overtime-limit",
            "emp-monthly-overtime-limit",
            // softs
            "soft-smaller-hours",
            "soft-earlier-start",
            "soft-affinity-penalty",
            "soft-affinity-reward",
            "soft-encourage-skill-variety",
            "soft-balance-block-avg-skill",
            "soft-balance-total-hours",
            "soft-minimize-region-transitions",
            "soft-fab-preference",
            "prefer-regular-over-spot",
            "block-recommend-headcount-range");

    @Test
    @SuppressWarnings({"unchecked", "rawtypes"})
    void names_and_order_unchanged() throws Exception {
        SolverFactory<SinglePassPlan> factory = EmployeeSchedule.factoryForTests(null, null);
        Object sdf = ((DefaultSolverFactory) factory).getScoreDirectorFactory();
        assertTrue(sdf instanceof AbstractConstraintStreamScoreDirectorFactory, String.valueOf(sdf));
        ConstraintMetaModel meta = ((AbstractConstraintStreamScoreDirectorFactory) sdf).getConstraintMetaModel();
        List<String> actual = new ArrayList<>();
        meta.getConstraints().forEach(c -> actual.add(c.getConstraintRef().constraintName()));
        assertEquals(EXPECTED_IDS, actual);
        assertEquals(23, actual.size());
        assertTrue(!actual.contains("seat-worker-available-all-days"), "must stay unregistered");

        // same ids as the characterisation golden (which lists every registered constraint)
        Map<String, Object> golden = Characterization.readGolden(CharacterizationTest.SMALL_GOLDEN);
        Map<String, Object> initial = (Map<String, Object>) golden.get("initial");
        Map<String, Object> goldenConstraints = (Map<String, Object>) initial.get("constraints");
        assertEquals(new TreeSet<>(goldenConstraints.keySet()), new TreeSet<>(actual));
    }

    @Test
    void provider_has_a_no_arg_constructor() {
        new SinglePassConstraints();
    }
}
