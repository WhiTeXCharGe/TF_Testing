package com.yourorg.scheduler;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import com.yourorg.scheduler.calendar.WorkingDays;
import com.yourorg.scheduler.model.BlockDecision;
import com.yourorg.scheduler.model.SinglePassPlan;

/**
 * PlanBuilder.applyInitialPlan may warm-start a phase-1 block on a day after windowStart, while the phase-1 start range
 * is only [windowStart]. Timefold 1.27 accepted that; 1.35 validates initial values against the range and must
 * not reject it.
 */
class Phase1WarmStartTest {

    @TempDir
    Path tmp;

    @Test
    void warm_started_phase1_block_beyond_window_start_builds_and_runs() throws Exception {
        for (Path fixture : new Path[] {CharacterizationTest.SMALL_FIXTURE, CharacterizationTest.DATAGEN_FIXTURE}) {
            SinglePassPlan p = Characterization.buildFrom(fixture, tmp.resolve(fixture.getFileName().toString()));
            BlockDecision target = null;
            for (BlockDecision b : p.blocks) {
                if (b.id != 0 && b.phaseNum == 1 && b.windowEnd > b.windowStart) { target = b; break; }
            }
            assertNotNull(target, "no phase-1 block with a window longer than one day in " + fixture);
            target.startDay = target.windowStart + 1;
            if (target.days == null) target.days = 1;
            target.coveredWorkingDays = WorkingDays.computeCoveredWorkingDays(target);

            assertDoesNotThrow(() -> Characterization.analyze(p), "analyze " + fixture);

            // the warm day passes validation (contains) but is not part of what the search can pick
            final Integer warmDay = target.startDay;
            assertTrue(target.vrStartWithinWindow().contains(warmDay));
            assertEquals(1, target.vrStartWithinWindow().getSize());
            java.util.Iterator<Integer> it = target.vrStartWithinWindow().createOriginalIterator();
            while (it.hasNext()) assertNotEquals(warmDay, it.next(), "warm day offered to the search");
            java.util.Iterator<Integer> rnd = target.vrStartWithinWindow().createRandomIterator(new java.util.Random(7));
            for (int i = 0; i < 100; i++) assertNotEquals(warmDay, rnd.next());
            if (fixture.equals(CharacterizationTest.SMALL_FIXTURE)) { // datagen needs far more steps to finish construction
                assertDoesNotThrow(() -> Characterization.solveSeeded(p, 200, CharacterizationTest.SEED), "solve " + fixture);
            }
        }
    }
}
