package com.yourorg.scheduler;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import ai.timefold.solver.core.api.score.HardMediumSoftScore;
import ai.timefold.solver.core.api.solver.SolutionManager;

import com.yourorg.scheduler.Characterization.ConstraintTotal;
import com.yourorg.scheduler.Characterization.Snapshot;
import com.yourorg.scheduler.model.SinglePassPlan;
import com.yourorg.scheduler.solver.ConstraintBreakdown;
import com.yourorg.scheduler.solver.SolverFactories;

/**
 * Pins {@link ConstraintBreakdown}, the one place that uses Timefold internals (Community replacement of the
 * Enterprise-only score analysis). If a Timefold upgrade breaks it, this test says so first.
 */
class ConstraintBreakdownTest {

    @TempDir
    Path tmp;

    @Test
    @SuppressWarnings("unchecked")
    void small_fixture_matches_the_golden_initial_totals() throws Exception {
        SinglePassPlan p = Characterization.buildFrom(CharacterizationTest.SMALL_FIXTURE, tmp.resolve("a"));
        ConstraintBreakdown.Result r = ConstraintBreakdown.of(SolverFactories.forTests(null, null)).analyze(p);

        Snapshot golden = Characterization.snapshotFromJson(
                (Map<String, Object>) Characterization.readGolden(CharacterizationTest.SMALL_GOLDEN).get("initial"));
        assertEquals(golden.score(), String.valueOf(r.score()));
        assertEquals(golden.constraints().keySet(), r.totals().keySet(), "every registered constraint, by id");
        assertEquals(23, r.totals().size());
        r.totals().forEach((id, t) -> assertEquals(golden.constraints().get(id),
                new ConstraintTotal(String.valueOf(t.score()), t.matches()), id));
        assertTrue(r.matches().isEmpty(), "analyze() keeps no matches");

        // the total equals the public API's score
        HardMediumSoftScore update = SolutionManager.<SinglePassPlan, HardMediumSoftScore>create(
                SolverFactories.forTests(null, null)).update(
                        Characterization.buildFrom(CharacterizationTest.SMALL_FIXTURE, tmp.resolve("b")));
        assertEquals(update, r.score());
    }

    @Test
    void summary_lists_every_constraint_and_at_most_n_matches_each() throws Exception {
        SinglePassPlan p = Characterization.buildFrom(CharacterizationTest.SMALL_FIXTURE, tmp.resolve("c"));
        ConstraintBreakdown.Result r = ConstraintBreakdown.of(SolverFactories.forTests(null, null))
                .analyzeWithMatches(p, 2);
        r.totals().forEach((id, t) -> {
            List<ConstraintBreakdown.Match> ms = r.matches().get(id);
            assertEquals(Math.min(2, t.matches()), ms.size(), id);
            for (int i = 1; i < ms.size(); i++) {
                assertTrue(ms.get(i - 1).score().compareTo(ms.get(i).score()) <= 0, "worst first: " + id);
            }
        });
        String text = ConstraintBreakdown.summarize(r, 2);
        assertTrue(text.startsWith("Explanation of score (" + r.score() + "):"), text);
        long constraintLines = text.lines().filter(l -> l.contains(": constraint (")).count();
        assertEquals(23, constraintLines, text);
        for (String id : r.totals().keySet()) {
            assertTrue(text.contains("constraint (" + id + ") has " + r.totals().get(id).matches() + " matches"), id);
        }
    }
}
