package com.yourorg.scheduler.solver;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import ai.timefold.solver.core.api.score.HardMediumSoftScore;
import ai.timefold.solver.core.api.solver.SolverFactory;
import ai.timefold.solver.core.impl.score.constraint.ConstraintMatch;
import ai.timefold.solver.core.impl.score.constraint.ConstraintMatchPolicy;
import ai.timefold.solver.core.impl.score.constraint.ConstraintMatchTotal;
import ai.timefold.solver.core.impl.score.director.InnerScoreDirector;
import ai.timefold.solver.core.impl.score.director.ScoreDirectorFactory;
import ai.timefold.solver.core.impl.solver.DefaultSolverFactory;

import com.yourorg.scheduler.model.SinglePassPlan;

/**
 * Per-constraint score breakdown (score and match count of every registered constraint) on Timefold Solver
 * <b>Community</b> 2.x.
 *
 * <p>Why this class exists: in 2.x {@code ScoreExplanation} / {@code SolutionManager.explain} were removed and
 * {@code SolutionManager.analyze} ({@code ScoreAnalysis}) is Enterprise-only (on Community it throws
 * {@code IllegalStateException: A commercial feature "Score analysis" was requested but it could not be loaded}).
 * {@code SolutionManager.update} still works but gives the total score only.
 *
 * <p>How: it builds Timefold's own score director with constraint matching enabled
 * ({@link ConstraintMatchPolicy#ENABLED}) and reads its constraint match totals. That is the mechanism
 * {@code ConstraintVerifier} uses on Community. It is <b>internal API</b> ({@code ai.timefold.solver.core.impl}:
 * {@link DefaultSolverFactory#getScoreDirectorFactory()}, {@link ScoreDirectorFactory#createScoreDirectorBuilder()},
 * {@code withConstraintMatchPolicy}, {@link InnerScoreDirector#getConstraintMatchTotalMap()},
 * {@link ConstraintMatchTotal}, {@link ConstraintMatch}) and may change in any 2.x minor release.
 *
 * <p><b>This is the single place to re-check on every Timefold upgrade.</b> It is pinned by the characterisation
 * tests ({@code CharacterizationTest}, {@code ConstraintBreakdownTest}: the per-constraint totals of the fixtures must
 * equal the golden files), so a break shows up there at once. Used by {@link ScoreLogger} (initial/stage1/final score
 * files, score_progress.txt, solver_log.txt) and by the test harness.
 *
 * <p>Public-API fallback if this ever stops working: per constraint, a solver factory whose constraint provider keeps
 * only that one constraint (via {@code ScoreDirectorFactoryConfig.withConstraintProviderCustomProperties}) gives the
 * constraint's score through {@code SolutionManager.update}; the match count can be read from a
 * {@code ConstraintVerifier} assertion ({@code penalizes(n)}/{@code rewards(n)}). Proven equal to the goldens in the
 * Task 10 spike, but slow (one solver factory per constraint), so it is only suitable for tests.
 */
public final class ConstraintBreakdown {

    /** Score and match count of one constraint. */
    public record Total(HardMediumSoftScore score, int matches) {}

    /** One constraint match: its score and the justification text (usually the matched facts). */
    public record Match(HardMediumSoftScore score, String justification) {}

    /**
     * Total score plus, per constraint id (sorted by id), its total and (only from {@link #analyzeWithMatches}) its
     * worst matches.
     */
    public record Result(HardMediumSoftScore score, Map<String, Total> totals, Map<String, List<Match>> matches) {}

    private final ScoreDirectorFactory<SinglePassPlan, HardMediumSoftScore> scoreDirectorFactory;

    private ConstraintBreakdown(ScoreDirectorFactory<SinglePassPlan, HardMediumSoftScore> scoreDirectorFactory) {
        this.scoreDirectorFactory = scoreDirectorFactory;
    }

    /** Breakdown for the model (solution, entities, constraints) of {@code factory}; build once, use many times. */
    public static ConstraintBreakdown of(SolverFactory<SinglePassPlan> factory) {
        if (!(factory instanceof DefaultSolverFactory<SinglePassPlan> defaultFactory)) {
            throw new IllegalStateException("Unsupported SolverFactory implementation " + factory.getClass().getName()
                    + "; ConstraintBreakdown relies on Timefold's internal DefaultSolverFactory");
        }
        return new ConstraintBreakdown(defaultFactory.getScoreDirectorFactory());
    }

    /**
     * Scores {@code plan} from scratch with constraint matching enabled. Like {@code SolutionManager.update} it sets
     * the plan's score and shadow variables; it changes no planning variable.
     */
    public Result analyze(SinglePassPlan plan) {
        return analyze(plan, 0);
    }

    /** As {@link #analyze}, and keeps up to {@code topMatchesPerConstraint} matches per constraint, worst first. */
    public Result analyzeWithMatches(SinglePassPlan plan, int topMatchesPerConstraint) {
        return analyze(plan, Math.max(1, topMatchesPerConstraint));
    }

    private Result analyze(SinglePassPlan plan, int keepMatches) {
        try (InnerScoreDirector<SinglePassPlan, HardMediumSoftScore> sd = scoreDirectorFactory
                .createScoreDirectorBuilder()
                .withConstraintMatchPolicy(ConstraintMatchPolicy.ENABLED)
                .build()) {
            sd.setWorkingSolution(plan);
            HardMediumSoftScore score = sd.calculateScore().raw();
            Map<String, Total> totals = new TreeMap<>();
            Map<String, List<Match>> matches = new TreeMap<>();
            for (Map.Entry<?, ConstraintMatchTotal<HardMediumSoftScore>> e : sd.getConstraintMatchTotalMap().entrySet()) {
                ConstraintMatchTotal<HardMediumSoftScore> cmt = e.getValue();
                String id = cmt.getConstraintRef().id();
                if (totals.put(id, new Total(cmt.getScore(), cmt.getConstraintMatchCount())) != null) {
                    throw new IllegalStateException("Duplicate constraint id " + id);
                }
                if (keepMatches > 0) {
                    List<ConstraintMatch<HardMediumSoftScore>> all = new ArrayList<>(cmt.getConstraintMatchSet());
                    all.sort(Comparator.comparing(ConstraintMatch::getScore));
                    int n = Math.min(keepMatches, all.size());
                    List<Match> ms = new ArrayList<>(n);
                    for (int i = 0; i < n; i++) {
                        ConstraintMatch<HardMediumSoftScore> cm = all.get(i);
                        ms.add(new Match(cm.getScore(), String.valueOf((Object) cm.getJustification())));
                    }
                    matches.put(id, Collections.unmodifiableList(ms));
                }
            }
            return new Result(score, Collections.unmodifiableMap(totals), Collections.unmodifiableMap(matches));
        }
    }

    /**
     * Text summary in the layout of the 1.x {@code ScoreAnalysis.summarize()}: constraints worst score first, each
     * with its score, id and match count, then up to {@code topMatchesPerConstraint} matches (worst first; only a
     * result of {@link #analyzeWithMatches} carries matches).
     */
    public static String summarize(Result r, int topMatchesPerConstraint) {
        int limit = Math.max(1, topMatchesPerConstraint);
        StringBuilder sb = new StringBuilder();
        sb.append("Explanation of score (").append(r.score()).append("):\n");
        sb.append("    Constraint matches:\n");
        List<String> ids = new ArrayList<>(r.totals().keySet());
        ids.sort(Comparator.comparing((String id) -> r.totals().get(id).score()).thenComparing(id -> id));
        for (String id : ids) {
            Total t = r.totals().get(id);
            sb.append(String.format("        %s: constraint (%s) has %d matches:%n", t.score().toShortString(), id,
                    t.matches()));
            List<Match> ms = r.matches().getOrDefault(id, List.of());
            for (int i = 0; i < Math.min(limit, ms.size()); i++) {
                Match m = ms.get(i);
                sb.append(String.format("            %s: justified with (%s)%n", m.score().toShortString(),
                        m.justification()));
            }
            if (!ms.isEmpty() && t.matches() > Math.min(limit, ms.size())) sb.append("            ...\n");
        }
        return sb.toString();
    }
}
