package com.yourorg.scheduler.solver;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import ai.timefold.solver.core.api.score.HardMediumSoftScore;
import ai.timefold.solver.core.api.solver.SolutionManager;
import ai.timefold.solver.core.api.solver.Solver;
import ai.timefold.solver.core.api.solver.SolverFactory;

import com.yourorg.scheduler.build.PlanBuilder;
import com.yourorg.scheduler.calendar.SolveContext;
import com.yourorg.scheduler.model.BlockDecision;
import com.yourorg.scheduler.model.CrewSeat;
import com.yourorg.scheduler.model.ParsedEnv;
import com.yourorg.scheduler.model.ParsedSchedule;
import com.yourorg.scheduler.model.SinglePassPlan;

import static com.yourorg.scheduler.solver.ScoreLogger.appendScoreProgress;
import static com.yourorg.scheduler.solver.ScoreLogger.appendSolveLog;
import static com.yourorg.scheduler.solver.ScoreLogger.fmt;
import static com.yourorg.scheduler.solver.ScoreLogger.nowClock;
import static com.yourorg.scheduler.solver.ScoreLogger.writeFinalScoreAnalysis;
import static com.yourorg.scheduler.solver.ScoreLogger.writeInitialScoreAnalysis;

/** The two-stage solve: stage 1 (find a feasible plan), then stage 2 (polish from the stage 1 result). */
public final class TwoStageRunner {
    private TwoStageRunner() {}

    public static RunResult run(PlanBuilder.BuiltProblem problem, TimerConfig timer) {
        ParsedEnv env = problem.env;
        ParsedSchedule sch = problem.sch;
        SinglePassPlan p = problem.plan;
        StatusReporter status = new StatusReporter(timer);

        SolverFactory<SinglePassPlan> factoryForInitial = SolverFactories.build(
            new Class<?>[]{ BlockDecision.class, CrewSeat.class},
            1,
            1,
            "0hard/*medium/*soft",
            timer.seed());
        writeInitialScoreAnalysis(factoryForInitial, p, "initial_score.txt",200);

        // Init score_progress.txt for this run (skipped when write_score_files=false)
        if (ScoreLogger.writeScoreFiles()) {
            try (BufferedWriter _pw = Files.newBufferedWriter(Paths.get("score_progress.txt"),
                    StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
                _pw.write("Stage   | elapsed | total score                             | per-constraint (hard/medium only)");
                _pw.newLine();
                _pw.write("------- | ------- | --------------------------------------- | --------------------------------------------------");
                _pw.newLine();
            } catch (IOException ignored) {}
        }

        // Ticks every few seconds so STATUS_FILE always has a fresh Running
        // stage/progress, independent of how often a new best score is found
        // (score improvements can plateau for many minutes on a hard instance).
        final java.util.concurrent.atomic.AtomicInteger tickerStage = new java.util.concurrent.atomic.AtomicInteger(1);
        final java.util.concurrent.atomic.AtomicLong tickerStageStartNanos =
                new java.util.concurrent.atomic.AtomicLong(System.nanoTime());
        ScheduledExecutorService progressTicker = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "status-progress-ticker");
            t.setDaemon(true);
            return t;
        });
        progressTicker.scheduleAtFixedRate(() -> {
            int stage = tickerStage.get();
            long msSpent = (System.nanoTime() - tickerStageStartNanos.get()) / 1_000_000L;
            status.writeRunningStatus(stage, status.overallProgressForStage(stage, msSpent));
        }, 5, 10, TimeUnit.SECONDS);

        System.out.println("Start SINGLE PASS (stage1) at " + nowClock());
        long t0 = System.nanoTime();
        // ---- Stage 1: current settings (e.g., 90/90) ----
        SolverFactory<SinglePassPlan> factoryStage1 = SolverFactories.build(
                new Class<?>[]{ BlockDecision.class, CrewSeat.class },
                timer.stage1Minutes(), timer.stage1UnimprovedSeconds(),
                timer.stage1StopWhenFeasible() ? "0hard/*medium/*soft" : null,
                timer.seed());
        Solver<SinglePassPlan> stage1 = factoryStage1.buildSolver();
        {
            SolutionManager<SinglePassPlan, HardMediumSoftScore> smS1 =
                    SolutionManager.create(factoryStage1);
            stage1.addEventListener(event -> {
                if (event.getNewBestScore() == null) return;
                appendScoreProgress("Stage1", event.getTimeMillisSpent(),
                        (HardMediumSoftScore) event.getNewBestScore(),
                        smS1, event.getNewBestSolution());
            });
        }
        SinglePassPlan best1 = stage1.solve(p);

        long t1 = System.nanoTime();
        System.out.printf("Stage1 done %s | duration=%s | score=%s | blocks=%d | seats=%d%n",
                nowClock(), fmt(java.time.Duration.ofNanos(t1 - t0)),
                String.valueOf(best1.getScore()),
                best1.blocks == null ? 0 : best1.blocks.size(),
                best1.seats == null ? 0 : best1.seats.size());

        // ---- Before stage2 ----
        // Warm-start is NOT pinned anymore, so there is nothing to "unpin".
        // Just refresh candidate lists based on the stage1 solution.
        PlanBuilder.fillSeatCandidatesSinglePass(best1.seats, best1.blocks, env.employees);
        writeFinalScoreAnalysis(factoryStage1, best1, "stage1_score.txt", 100);

        // With stage 2 disabled the stage 1 result is the final result.
        SolverFactory<SinglePassPlan> factoryFinal = factoryStage1;
        SinglePassPlan best2 = best1;
        long t2 = t1;
        if (timer.stage2Enabled()) {
            // ---- Stage 2 (polish): start from stage1 result ----
            tickerStage.set(2);
            tickerStageStartNanos.set(System.nanoTime());
            System.out.println("Start POLISH (stage2, " + timer.stage2Minutes() + "m) at " + nowClock());

            SolverFactory<SinglePassPlan> factoryStage2 = SolverFactories.build(
                    new Class<?>[]{ BlockDecision.class, CrewSeat.class },
                    timer.stage2Minutes()  /* spentMinutes */,
                    timer.stage2UnimprovedSeconds() /* unimprovedSeconds */,
                    null /* bestScoreLimit */,
                    timer.seed());

            Solver<SinglePassPlan> stage2 = factoryStage2.buildSolver();
            {
                SolutionManager<SinglePassPlan, HardMediumSoftScore> smS2 =
                        SolutionManager.create(factoryStage2);
                stage2.addEventListener(event -> {
                    if (event.getNewBestScore() == null) return;
                    appendScoreProgress("Stage2", event.getTimeMillisSpent(),
                            (HardMediumSoftScore) event.getNewBestScore(),
                            smS2, event.getNewBestSolution());
                });
            }
            best2 = stage2.solve(best1);
            factoryFinal = factoryStage2;
            t2 = System.nanoTime();
        }
        progressTicker.shutdownNow();

        if (timer.stage2Enabled()) {
            System.out.printf("Stage2 done %s | duration=%s | score=%s%n",
                    nowClock(), fmt(java.time.Duration.ofNanos(t2 - t1)),
                    String.valueOf(best2.getScore()));
        }

        java.time.Duration stage1Dur = java.time.Duration.ofNanos(t1 - t0);
        java.time.Duration stage2Dur = java.time.Duration.ofNanos(t2 - t1);


        System.out.printf("Stage1 done %s | duration=%s | score=%s | blocks=%d | seats=%d%n",
                nowClock(), fmt(stage1Dur),
                String.valueOf(best1.getScore()),
                best1.blocks == null ? 0 : best1.blocks.size(),
                best1.seats == null ? 0 : best1.seats.size());

        if (timer.stage2Enabled()) {
            System.out.printf("Stage2 done %s | duration=%s | score=%s%n",
                    nowClock(), fmt(stage2Dur),
                    String.valueOf(best2.getScore()));
        }

        PlanBuilder.validateHorizonOverlaps(best2, SolveContext.current().solveMinDay, SolveContext.current().solveMaxDay);

        // Append log to file (score + durations + per-constraint score)
        writeFinalScoreAnalysis(factoryFinal, best2, "final_score.txt", 100);
        appendSolveLog(factoryFinal, best2, sch, stage1Dur, stage2Dur);

        RunResult rr = new RunResult();
        rr.plan = best2; rr.planStart = sch.planStart;
        return rr;
    }
}
