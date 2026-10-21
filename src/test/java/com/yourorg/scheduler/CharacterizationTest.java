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
 */
class CharacterizationTest {

    static final Path SMALL_FIXTURE = Path.of("src", "test", "resources", "fixtures", "small");
    static final Path SMALL_GOLDEN = Characterization.GOLDEN_DIR.resolve("small-1.27.json");
    static final Path REAL_GOLDEN = Characterization.GOLDEN_DIR.resolve("real-1.27.json");

    static final int STEP_LIMIT = 200;
    static final long SEED = 42L;

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
        Characterization.checkSection(SMALL_GOLDEN, "seeded_run", seededParams(), fin);
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

    private static Map<String, Object> seededParams() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("step_limit", STEP_LIMIT);
        m.put("seed", SEED);
        return m;
    }
}
