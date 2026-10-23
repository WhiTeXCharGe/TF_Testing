package com.yourorg.scheduler;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import com.yourorg.scheduler.Characterization.ConstraintTotal;
import com.yourorg.scheduler.Characterization.Snapshot;
import com.yourorg.scheduler.model.SinglePassPlan;

/**
 * Characterisation of the baseline (Timefold 1.27) behaviour.
 *
 * <ul>
 *   <li>{@code small} fixture (src/test/resources/fixtures/small): initial (warm-start) per-constraint totals and a
 *       seeded, step-limited run, golden {@code golden/small-1.27.json}.</li>
 *   <li>optional real sample, enabled by {@code -Dreal.sample=<dir with EnvConfig.yaml and Schedule.yaml>}:
 *       initial totals, golden {@code golden/real-1.27.json}.</li>
 * </ul>
 * Regenerate goldens only for an intended behaviour change: {@code mvn test -Dgolden.write=true}.
 *
 * <p>The golden files keep their {@code -1.27} names. {@code small-1.27.json} and {@code datagen-1.27.json} were
 * regenerated in the commit "fix: weigh phase-order and window violations 100 per day" (block weights change, see
 * README "Block weights"): in their {@code initial} sections only {@code phase-order}, {@code block-end-within-window}
 * and the total score may differ from the 1.27 values (in practice only {@code small}'s phase-order changed, -2 to
 * -200 hard), all other constraints are still the 1.27 values; the {@code seeded_run} sections follow the new search
 * trajectory. {@code real-1.27.json} was NOT regenerated (the real sample is not run in the normal workflow). Its
 * {@code phase-order} and {@code block-end-within-window} entries record 0 matches, which the new weights leave at 0,
 * so it is expected to still hold, but that is unverified: treat those two entries and the total as stale until
 * someone runs it with {@code -Dreal.sample=<dir>} (and {@code -Dgolden.write=true} if it reports a difference).
 *
 * <p>{@code real-1.27.json} was also EDITED BY HAND (not regenerated, not re-verified with {@code -Dreal.sample}) for the
 * conditional stay rules (25547bd): the real EnvConfig files have unlimited stay limits (10000), so
 * {@code emp-region-stay-max-on} and {@code emp-region-annual-stay-max} are no longer registered and were removed from
 * its {@code initial} section (21 constraint ids instead of 23). Both entries had 0 matches and score 0, so the total
 * and every other entry are unchanged.
 */
class CharacterizationTest {

    static final Path TINY_FIXTURE = Path.of("src", "test", "resources", "fixtures", "tiny");
    static final Path SMALL_FIXTURE = Path.of("src", "test", "resources", "fixtures", "small");
    static final Path DATAGEN_FIXTURE = Path.of("src", "test", "resources", "fixtures", "datagen");
    static final Path DATAGEN_GOLDEN = Characterization.GOLDEN_DIR.resolve("datagen-1.27.json");
    static final Path SMALL_GOLDEN = Characterization.GOLDEN_DIR.resolve("small-1.27.json");
    static final Path REAL_GOLDEN = Characterization.GOLDEN_DIR.resolve("real-1.27.json");

    static final int STEP_LIMIT = 200;
    static final long SEED = 42L;
    /** The datagen plan has far more entities than the small one; 200 steps do not even finish construction. */
    static final int DATAGEN_STEP_LIMIT = 2000;

    @TempDir
    Path tmp;

    @Test
    void initial_totals_match_golden() throws Exception {
        SinglePassPlan p = Characterization.buildFrom(SMALL_FIXTURE, tmp.resolve("initial"));
        Map<String, ConstraintTotal> totals = Characterization.initialTotals(p);
        Snapshot initial = Characterization.analyze(p);
        assertEquals(totals, initial.constraints());
        Characterization.checkSection(SMALL_GOLDEN, "initial", Map.of(), initial);
    }

    @Test
    void seeded_run_matches_golden() throws Exception {
        SinglePassPlan p = Characterization.buildFrom(SMALL_FIXTURE, tmp.resolve("seeded"));
        SinglePassPlan best = Characterization.solveSeeded(p, STEP_LIMIT, SEED);
        Snapshot fin = Characterization.analyze(best);
        assertEquals(String.valueOf(best.getScore()), fin.score(), "solver score vs. re-analysed score");
        Characterization.checkSection(SMALL_GOLDEN, "seeded_run", seededParams(STEP_LIMIT), fin);
    }

    /** Same input twice in one JVM must give identical initial totals and an identical seeded run. */
    @Test
    void build_and_seeded_run_are_deterministic_within_one_jvm() throws Exception {
        Snapshot a = Characterization.analyze(Characterization.buildFrom(SMALL_FIXTURE, tmp.resolve("a")));
        Snapshot b = Characterization.analyze(Characterization.buildFrom(SMALL_FIXTURE, tmp.resolve("b")));
        assertEquals("", Characterization.diff(a, b), "initial totals differ between two builds");

        String s1 = Characterization.seededRun(
                Characterization.buildFrom(SMALL_FIXTURE, tmp.resolve("c")), STEP_LIMIT, SEED);
        String s2 = Characterization.seededRun(
                Characterization.buildFrom(SMALL_FIXTURE, tmp.resolve("d")), STEP_LIMIT, SEED);
        assertEquals(s1, s2, "seeded run differs between two runs");
    }

    @Test
    void real_sample_initial_totals_match_golden() throws Exception {
        String dir = System.getProperty("real.sample");
        assumeTrue(dir != null && !dir.isBlank(),
                "real sample not configured (-Dreal.sample=<dir with EnvConfig.yaml and Schedule.yaml>)");
        SinglePassPlan p = Characterization.buildFrom(Path.of(dir), tmp.resolve("real"));
        Characterization.checkSection(REAL_GOLDEN, "initial", Map.of(), Characterization.analyze(p));
    }

    /** Fixture = unmodified output of logic/data_generator (EnvConfig.yaml + Schedule.yaml). */
    @Test
    void datagen_initial_totals_match_golden() throws Exception {
        SinglePassPlan p = Characterization.buildFrom(DATAGEN_FIXTURE, tmp.resolve("dg-initial"));
        Snapshot initial = Characterization.analyze(p);
        assertEquals(Characterization.initialTotals(p), initial.constraints());
        Characterization.checkSection(DATAGEN_GOLDEN, "initial", Map.of(), initial);
    }

    @Test
    void datagen_seeded_run_matches_golden() throws Exception {
        SinglePassPlan p = Characterization.buildFrom(DATAGEN_FIXTURE, tmp.resolve("dg-seeded"));
        SinglePassPlan best = Characterization.solveSeeded(p, DATAGEN_STEP_LIMIT, SEED);
        Snapshot fin = Characterization.analyze(best);
        assertEquals(String.valueOf(best.getScore()), fin.score(), "solver score vs. re-analysed score");
        Characterization.checkSection(DATAGEN_GOLDEN, "seeded_run", seededParams(DATAGEN_STEP_LIMIT), fin);
    }

    private static Map<String, Object> seededParams(int stepLimit) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("step_limit", stepLimit);
        m.put("seed", SEED);
        return m;
    }
}
