package com.yourorg.scheduler.solver;

import ai.timefold.solver.core.api.solver.SolverFactory;
import ai.timefold.solver.core.config.score.director.ScoreDirectorFactoryConfig;
import ai.timefold.solver.core.config.solver.SolverConfig;
import ai.timefold.solver.core.config.solver.termination.TerminationConfig;

import com.yourorg.scheduler.constraints.SinglePassConstraints;
import com.yourorg.scheduler.model.BlockDecision;
import com.yourorg.scheduler.model.CrewSeat;
import com.yourorg.scheduler.model.SinglePassPlan;

/** Builds the Timefold solver factories for the production model. */
public final class SolverFactories {
    private SolverFactories() {}

    /**
     * Production solver factory: time/unimproved/best-score termination; {@code seed} (when non-null) fixes the
     * random seed, otherwise no seed is set.
     */
    public static SolverFactory<SinglePassPlan> build(Class<?>[] entities,
                                                      Integer spentMinutes,
                                                      Integer unimprovedSeconds,
                                                      String bestScoreLimit,
                                                      Long seed) {
        SolverConfig cfg = new SolverConfig();
        cfg.withSolutionClass(SinglePassPlan.class);
        cfg.withEntityClasses(entities);
        cfg.withScoreDirectorFactory(
                new ScoreDirectorFactoryConfig().withConstraintProviderClass(SinglePassConstraints.class)
        );

        TerminationConfig term = new TerminationConfig();
        if (bestScoreLimit != null) term.setBestScoreLimit(bestScoreLimit);
        if (spentMinutes != null && spentMinutes > 0) {
            term.setSpentLimit(java.time.Duration.ofMinutes(spentMinutes));
        }
        if (unimprovedSeconds != null && unimprovedSeconds > 0) {
            term.setUnimprovedSpentLimit(java.time.Duration.ofSeconds(unimprovedSeconds));
        }
        cfg.withTerminationConfig(term);
        if (seed != null) cfg.withRandomSeed(seed);

        // NOTE: multi-threaded solving (moveThreadCount) is intentionally NOT
        // set here. It requires Timefold Solver Enterprise Edition; on the
        // Community Edition it throws at solve time. The speed-up in this
        // redesign comes purely from the shadow variable and the removal of
        // the DaySlot cartesian-product joins, which need no Enterprise build.

        // NOTE:
        // Do NOT force a single WEAKEST_FIT_DECREASING construction heuristic here,
        // because this solver has multiple planning entity classes (BlockDecision and CrewSeat).
        // With an explicit CH phase and no entityClass on the placer, Timefold throws:
        // "QueuedEntityPlacerConfig ... has no entityClass configured ..."
        //
        // We still keep weakest-fit style through entity/value sorting support on the model:
        // - @PlanningEntity(comparatorClass=...) on BlockDecision and CrewSeat
        // - comparatorClass on planning variables (2.x name of the 1.x strengthComparatorClass)
        //
        // If later want explicit per-entity-class CH phases, define one CH phase for
        // BlockDecision and another CH phase for CrewSeat, each with its own entity selector.

        return SolverFactory.create(cfg);
    }

    /**
     * Solver factory with the production model (solution, entities, constraints, default phases) for tests:
     * optional step-count limit and random seed; no time-based termination.
     */
    public static SolverFactory<SinglePassPlan> forTests(Integer stepLimit, Long seed) {
        SolverConfig cfg = new SolverConfig();
        cfg.withSolutionClass(SinglePassPlan.class);
        cfg.withEntityClasses(BlockDecision.class, CrewSeat.class);
        cfg.withScoreDirectorFactory(
                new ScoreDirectorFactoryConfig().withConstraintProviderClass(SinglePassConstraints.class)
        );
        TerminationConfig term = new TerminationConfig();
        if (stepLimit != null) term.setStepCountLimit(stepLimit);
        cfg.withTerminationConfig(term);
        if (seed != null) cfg.withRandomSeed(seed);
        return SolverFactory.create(cfg);
    }
}
