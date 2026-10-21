package com.yourorg.scheduler.model;

import com.yourorg.scheduler.calendar.WorkingDays;

import ai.timefold.solver.core.api.domain.variable.ShadowVariable;
import ai.timefold.solver.core.api.domain.variable.VariableListener;
import ai.timefold.solver.core.api.score.director.ScoreDirector;

// Keeps BlockDecision.coveredWorkingDays in sync with the block's startDay
// and days planning variables. Centralising the working-day expansion in a
// single shadow variable means it is rebuilt once per block-span change,
// rather than being recomputed inside every day-based constraint and for
// every seat attached to the block.
// VariableListener is deprecated in Timefold 1.27+ but is a Community
// Edition feature and still fully functional; "removal" is suppressed for
// the same reason as the @ShadowVariable annotation above.
@SuppressWarnings({"deprecation", "removal"})
public class BlockSpanVariableListener
        implements VariableListener<SinglePassPlan, BlockDecision> {

    @Override
    public void beforeVariableChanged(ScoreDirector<SinglePassPlan> scoreDirector, BlockDecision block) {
        // no-op: the new value is fully recomputed in afterVariableChanged
    }

    @Override
    public void afterVariableChanged(ScoreDirector<SinglePassPlan> scoreDirector, BlockDecision block) {
        updateCoveredWorkingDays(scoreDirector, block);
    }

    @Override
    public void beforeEntityAdded(ScoreDirector<SinglePassPlan> scoreDirector, BlockDecision block) {
        // no-op
    }

    @Override
    public void afterEntityAdded(ScoreDirector<SinglePassPlan> scoreDirector, BlockDecision block) {
        updateCoveredWorkingDays(scoreDirector, block);
    }

    @Override
    public void beforeEntityRemoved(ScoreDirector<SinglePassPlan> scoreDirector, BlockDecision block) {
        // no-op
    }

    @Override
    public void afterEntityRemoved(ScoreDirector<SinglePassPlan> scoreDirector, BlockDecision block) {
        // no-op
    }

    private void updateCoveredWorkingDays(ScoreDirector<SinglePassPlan> scoreDirector, BlockDecision block) {
        // Timefold must be notified around a shadow-variable mutation so the
        // incremental score engine re-evaluates the constraints that read it.
        scoreDirector.beforeVariableChanged(block, "coveredWorkingDays");
        block.coveredWorkingDays = WorkingDays.computeCoveredWorkingDays(block);
        scoreDirector.afterVariableChanged(block, "coveredWorkingDays");
    }
}
