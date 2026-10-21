package com.yourorg.scheduler;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import com.yourorg.scheduler.build.PlanBuilder;
import com.yourorg.scheduler.calendar.WorkingDays;
import com.yourorg.scheduler.model.BlockDecision;
import com.yourorg.scheduler.model.InitialAssign;
import com.yourorg.scheduler.model.SinglePassPlan;

/**
 * A warm start (flexible rows of Schedule.yaml) may begin a phase-1 block on a day after its windowStart, while the
 * phase-1 start range is only [windowStart] (phase-1 blocks start at the window start by design). Timefold 2.x rejects
 * an initial value outside the value range, so {@code PlanBuilder.applyInitialPlan} clamps such a start day to
 * windowStart (keeping the warm end day). This test drives that path on the real fixtures: the block is clamped, the
 * plan scores without error, and (small fixture) a few solver steps run.
 */
class Phase1WarmStartTest {

    @TempDir
    Path tmp;

    static PlanBuilder.BuiltProblem build(Path fixture, Path workDir) throws Exception {
        Files.createDirectories(workDir);
        Path env = workDir.resolve("EnvConfig.yaml");
        Path sched = workDir.resolve("Schedule.yaml");
        Files.copy(fixture.resolve("EnvConfig.yaml"), env, StandardCopyOption.REPLACE_EXISTING);
        Files.copy(fixture.resolve("Schedule.yaml"), sched, StandardCopyOption.REPLACE_EXISTING);
        return PlanBuilder.buildProblem(env.toString(), sched.toString());
    }

    @Test
    void warm_started_phase1_block_beyond_window_start_is_clamped_and_runs() throws Exception {
        for (Path fixture : new Path[] {CharacterizationTest.SMALL_FIXTURE, CharacterizationTest.DATAGEN_FIXTURE}) {
            PlanBuilder.BuiltProblem bp = build(fixture, tmp.resolve(fixture.getFileName().toString()));
            SinglePassPlan p = bp.plan;
            BlockDecision target = null;
            for (BlockDecision b : p.blocks) {
                boolean active = bp.sch.activeModules == null || bp.sch.activeModules.isEmpty()
                        || bp.sch.activeModules.contains(b.module);
                if (b.id != 0 && b.phaseNum == 1 && b.windowEnd >= b.windowStart + 2 && active) { target = b; break; }
            }
            assertNotNull(target, "no phase-1 block with a window of at least three days in " + fixture);
            final BlockDecision t = target;

            // warm-start rows for the target block: days windowStart+1 .. windowStart+2 (after the window start)
            int warmFrom = t.windowStart + 1;
            int warmTo = t.windowStart + 2;
            InitialAssign row = new InitialAssign();
            row.module = t.module;
            row.opId = t.opId;
            row.hoursByDay.put(warmFrom, 8);
            row.hoursByDay.put(warmTo, 8);
            List<InitialAssign> rows = new ArrayList<>();
            if (bp.sch.initialRows != null) {
                for (InitialAssign ia : bp.sch.initialRows) {
                    if (ia != null && !(t.module.equals(ia.module) && t.opId.equals(ia.opId))) rows.add(ia);
                }
            }
            rows.add(row);
            bp.sch.initialRows = rows;
            t.startDay = null;
            t.days = null;
            t.hours = null;

            PlanBuilder.applyInitialPlan(bp.sch, p.blocks);
            for (BlockDecision b : p.blocks) b.coveredWorkingDays = WorkingDays.computeCoveredWorkingDays(b);

            assertEquals(t.windowStart, t.startDay, "phase-1 warm start clamped to windowStart in " + fixture);
            assertEquals(warmTo - t.windowStart + 1, t.days, "warm end day kept in " + fixture);
            assertTrue(t.vrStartWithinWindow().contains(t.startDay));
            assertEquals(List.of(t.windowStart), t.vrStartWithinWindow(), "phase-1 range is [windowStart]");

            // nothing left for the safety net
            Map<String, Integer> n = PlanBuilder.normalizeInitialValues(p);
            assertFalse(n.keySet().stream().anyMatch(k -> k.startsWith("block")), "block values normalised: " + n);

            assertDoesNotThrow(() -> Characterization.analyze(p), "analyze " + fixture);
            if (fixture.equals(CharacterizationTest.SMALL_FIXTURE)) { // datagen needs far more steps to finish construction
                SinglePassPlan best = Characterization.solveSeeded(p, 200, CharacterizationTest.SEED);
                for (BlockDecision b : best.blocks) {
                    if (b.id == t.id) assertEquals(t.windowStart, b.startDay, "phase-1 block moved off windowStart");
                }
            }
        }
    }
}
