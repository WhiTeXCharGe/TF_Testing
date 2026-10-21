// mvn -DskipTests clean package   
// mvn -q exec:java -D"exec.args=src\main\resource\EnvConfig.yaml src\main\resource\Schedule.yaml"
package com.yourorg.scheduler;

import java.io.*;
import java.nio.file.*;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import ai.timefold.solver.core.api.score.Score;
import ai.timefold.solver.core.api.score.buildin.hardmediumsoft.HardMediumSoftScore;
import ai.timefold.solver.core.api.score.stream.Constraint;
import ai.timefold.solver.core.api.score.stream.ConstraintCollectors;
import ai.timefold.solver.core.api.score.stream.ConstraintFactory;
import ai.timefold.solver.core.api.score.stream.ConstraintProvider;
import ai.timefold.solver.core.api.score.stream.Joiners;
import ai.timefold.solver.core.api.solver.Solver;
import ai.timefold.solver.core.api.solver.SolverFactory;
import ai.timefold.solver.core.config.score.director.ScoreDirectorFactoryConfig;
import ai.timefold.solver.core.config.solver.SolverConfig;
import ai.timefold.solver.core.config.solver.termination.TerminationConfig;
import ai.timefold.solver.core.api.solver.SolutionManager;
import ai.timefold.solver.core.api.score.ScoreExplanation;

import com.yourorg.scheduler.calendar.SolveContext;
import com.yourorg.scheduler.build.PlanBuilder;
import com.yourorg.scheduler.constraints.SinglePassConstraints;
import com.yourorg.scheduler.calendar.WorkingDays;
import com.yourorg.scheduler.config.SolverConstants;
import com.yourorg.scheduler.export.ScheduleExporter;
import com.yourorg.scheduler.io.YamlFiles;
import com.yourorg.scheduler.model.*;


/**
 * SINGLE PASS scheduler (Timefold 1.27-compatible).
 */
public class EmployeeSchedule {

    // ---------------- YAML I/O (moved to io.YamlFiles) ----------------



    // ---------------- Domain ----------------

    // ---------- Planning Entities (single pass) ----------


    // ---------------- Globals & helpers ----------------

    // DAILY_CAP, REQUIRED_HOURS_PER_WORKLOAD_DAY, TRIM_FINISHED_MODULES, MODULE_TRIM_GRACE_DAYS and
    // BASE_HOURS_PER_DAY moved to config.SolverConstants.


    private static final java.time.format.DateTimeFormatter CLOCK =
            java.time.format.DateTimeFormatter.ofPattern("HH:mm:ss");
    private static String nowClock() { return java.time.LocalTime.now().format(CLOCK); }
    private static String fmt(java.time.Duration d) {
        long h = d.toHours(); long m = d.toMinutesPart(); long s = d.toSecondsPart(); long ms = d.toMillisPart();
        return String.format("%02d:%02d:%02d.%03d", h, m, s, ms);
    }






    // ---------------- Constraints (moved to constraints.SinglePassConstraints) ----------------
    
    // ---------------- Parsing ----------------








    // ---------------- Solver builder ----------------

    static <S> SolverFactory<S> buildSolverFactory(Class<S> solutionClass,
                                                Class<?>[] entityClasses,
                                                Class<? extends ConstraintProvider> providerClass,
                                                String bestScoreLimit,
                                                Integer spentMinutes,
                                                Integer unimprovedSeconds) {
        SolverConfig cfg = new SolverConfig();
        cfg.withSolutionClass(solutionClass);
        cfg.withEntityClasses(entityClasses);
        cfg.withScoreDirectorFactory(
                new ScoreDirectorFactoryConfig().withConstraintProviderClass(providerClass)
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
        // - @PlanningEntity(difficultyComparatorClass=...) on BlockDecision and CrewSeat
        // - strengthComparatorClass on planning variables / value ranges
        //
        // If later want explicit per-entity-class CH phases, define one CH phase for
        // BlockDecision and another CH phase for CrewSeat, each with its own entity selector.

        return SolverFactory.create(cfg);
    }

    static boolean hardZero(Score<?> s) { return s != null && s.toString().startsWith("0hard"); }

    // ---------------- Logging helper ----------------

    static void appendSolveLog(
            SolverFactory<SinglePassPlan> factoryStage2,
            SinglePassPlan best2,
            ParsedSchedule sch,
            java.time.Duration stage1Duration,
            java.time.Duration stage2Duration) {

        Path logFile = Paths.get("solver_log.txt"); // will be created in working directory

        try (BufferedWriter bw = Files.newBufferedWriter(
                logFile,
                StandardOpenOption.CREATE,
                StandardOpenOption.APPEND)) {

            // SolutionManager + explanation
            SolutionManager<SinglePassPlan, HardMediumSoftScore> solutionManager =
                    SolutionManager.create(factoryStage2);
            ScoreExplanation<SinglePassPlan, HardMediumSoftScore> explanation =
                    solutionManager.explain(best2);

            // Separator
            bw.write("============================================================");
            bw.newLine();
            bw.write("Run at: " + java.time.LocalDateTime.now());
            bw.newLine();

            // Basic timing and score
            bw.write("Stage1 duration: " + fmt(stage1Duration));
            bw.newLine();
            bw.write("Stage2 duration: " + fmt(stage2Duration));
            bw.newLine();
            bw.write("Final score: " + String.valueOf(best2.getScore()));
            bw.newLine();

            // Plan and cut-off
            bw.write("Plan range (plan_range): " + sch.planStart + " .. " + sch.planEnd);
            bw.newLine();
            if (sch.cutOffDate != null) {
                bw.write("Cut-off date (cut_off_date): " + sch.cutOffDate +
                        "  (cutOffDayId=" + sch.cutOffDayId + ")");
            } else {
                bw.write("Cut-off date (cut_off_date): [none]");
            }
            bw.newLine();

            // DaySlot range actually used in solver
            int minDayId = Integer.MAX_VALUE;
            int maxDayId = Integer.MIN_VALUE;
            LocalDate minDate = null;
            LocalDate maxDate = null;
            int dayCount = 0;

            if (best2.days != null) {
                for (DaySlot d : best2.days) {
                    if (d == null) continue;
                    dayCount++;
                    if (d.id < minDayId) {
                        minDayId = d.id;
                        minDate = d.date;
                    }
                    if (d.id > maxDayId) {
                        maxDayId = d.id;
                        maxDate = d.date;
                    }
                }
            }

            if (dayCount > 0) {
                bw.write("DaySlots in solver: id=" + minDayId + ".." + maxDayId +
                        " (count=" + dayCount + "), dates=" + minDate + " .. " + maxDate);
            } else {
                bw.write("DaySlots in solver: [none]");
            }
            bw.newLine();

            // Module sets
            java.util.Set<String> activeMods =
                    (sch.activeModules == null) ? new java.util.HashSet<>() :
                            new java.util.TreeSet<>(sch.activeModules);

            java.util.Set<String> dynamicBlockMods = new java.util.TreeSet<>();
            if (best2.blocks != null) {
                for (BlockDecision b : best2.blocks) {
                    if (b == null) continue;
                    // id 0 is the dummy fixed block
                    if (b.id == 0) continue;
                    if (b.module != null && !b.module.isBlank()) {
                        dynamicBlockMods.add(b.module);
                    }
                }
            }

            java.util.Set<String> pinnedMods = new java.util.TreeSet<>();
            if (best2.seats != null) {
                for (CrewSeat s : best2.seats) {
                    if (s == null) continue;
                    if (!s.pinnedFixed) continue;
                    if (s.module != null && !s.module.isBlank()) {
                        pinnedMods.add(s.module);
                    }
                }
            }

            bw.write("Active modules by trim logic (gap <= " + SolverConstants.MODULE_TRIM_GRACE_DAYS + " days): "
                    + (activeMods.isEmpty() ? "[none]" : String.join(", ", activeMods)));
            bw.newLine();

            bw.write("Modules with dynamic blocks (BlockDecision.id > 0): "
                    + (dynamicBlockMods.isEmpty() ? "[none]" : String.join(", ", dynamicBlockMods)));
            bw.newLine();

            bw.write("Modules with pinned seats (fixed assignments visible to constraints): "
                    + (pinnedMods.isEmpty() ? "[none]" : String.join(", ", pinnedMods)));
            bw.newLine();

            bw.write("Per-constraint scores:");
            bw.newLine();

            // Per-constraint scores
            explanation.getConstraintMatchTotalMap().forEach((constraintName, cmt) -> {
                try {
                    int matchCount = (cmt.getConstraintMatchSet() == null) ? -1 : cmt.getConstraintMatchSet().size();
                    bw.write("  " + constraintName
                            + " score=" + cmt.getScore()
                            + " matches=" + matchCount);
                    bw.newLine();
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            });

            bw.flush();
        } catch (IOException e) {
            System.err.println("Failed to write solver_log.txt: " + e.getMessage());
        }
    }


    // Appends one line per best-score improvement to score_progress.txt.
    // Call from a BestSolutionChangedEvent listener so you can watch which constraints
    // are still firing and how fast the score is improving over time.
    static void appendScoreProgress(String stage, long msElapsed, HardMediumSoftScore score,
                                    SolutionManager<SinglePassPlan, HardMediumSoftScore> sm,
                                    SinglePassPlan solution) {
        Path logFile = Paths.get("score_progress.txt");
        try (BufferedWriter bw = Files.newBufferedWriter(logFile,
                StandardOpenOption.CREATE, StandardOpenOption.APPEND)) {

            bw.write(String.format("%-7s | t+%5ds | total=%-40s%n",
                    stage, msElapsed / 1000, score));

            // Per-constraint breakdown (slightly expensive; kept because it's best-effort)
            try {
                ScoreExplanation<SinglePassPlan, HardMediumSoftScore> ex = sm.explain(solution);
                ex.getConstraintMatchTotalMap().forEach((name, cmt) -> {
                    HardMediumSoftScore cs = cmt.getScore();
                    if (cs.hardScore() == 0 && cs.mediumScore() == 0) return; // skip zero-hard
                    int matches = (cmt.getConstraintMatchSet() == null) ? -1
                            : cmt.getConstraintMatchSet().size();
                    try {
                        String shortName = name.contains("/") ? name.substring(name.lastIndexOf('/') + 1) : name;
                        bw.write(String.format("          %-45s score=%-30s matches=%d%n",
                                shortName, cs, matches));
                    } catch (IOException ex2) { throw new UncheckedIOException(ex2); }
                });
            } catch (Exception ignored) { /* explain may fail mid-solve; skip */ }

        } catch (IOException e) { /* best-effort; don't crash the solver */ }
    }

    // ---------------- Live status reporting (STATUS_FILE) ----------------
    //
    // entrypoint.sh writes status=Running once, right before launching this JVM,
    // then only writes again (Completed/Failed/Cancelled) after the JVM exits.
    // On Azure Batch that meant the status blob showed "queued" for the whole
    // solve — Batch only uploads outputFiles once the task command line exits
    // (see api-controller's uploadCondition: 'taskCompletion'), so there was
    // never an intermediate write for it to pick up. This ticker fills that gap
    // by updating STATUS_FILE itself every few seconds with the real stage and
    // an elapsed-time-based progress estimate, so a live reader (the API
    // controller reading the task's file straight off the node) has something
    // fresh to show while the task is still active/running.

    static final long STAGE1_BUDGET_MS = java.time.Duration.ofMinutes(300).toMillis();
    static final long STAGE2_BUDGET_MS = java.time.Duration.ofMinutes(60).toMillis();
    static final long TOTAL_BUDGET_MS  = STAGE1_BUDGET_MS + STAGE2_BUDGET_MS;

    private static final DateTimeFormatter STATUS_TS_FMT =
            DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'").withZone(java.time.ZoneOffset.UTC);
    private static String nowIsoUtc() { return STATUS_TS_FMT.format(java.time.Instant.now()); }

    private static volatile String runStartedAtIso = null;

    static void writeRunningStatus(int stage, double overallProgress) {
        try {
            String statusDir = System.getenv().getOrDefault("STATUS_DIR", "/work/status");
            String runId = System.getenv().getOrDefault("RUN_ID", "local");
            if (runStartedAtIso == null) runStartedAtIso = nowIsoUtc();
            double clamped = Math.max(0.0, Math.min(0.99, overallProgress));

            String json = String.format(Locale.ROOT,
                "{%n  \"runId\": \"%s\",%n  \"status\": \"Running\",%n  \"stage\": %d,%n" +
                "  \"progress\": %.4f,%n  \"startedAt\": \"%s\",%n  \"updatedAt\": \"%s\",%n" +
                "  \"finishedAt\": null,%n  \"error\": null,%n  \"output\": null%n}%n",
                runId, stage, clamped, runStartedAtIso, nowIsoUtc());

            Path dir = Paths.get(statusDir);
            Files.createDirectories(dir);
            Path statusFile = dir.resolve(runId + ".json");
            Path tmp = dir.resolve(runId + ".json.tmp");
            Files.writeString(tmp, json, java.nio.charset.StandardCharsets.UTF_8);
            Files.move(tmp, statusFile,
                    java.nio.file.StandardCopyOption.REPLACE_EXISTING,
                    java.nio.file.StandardCopyOption.ATOMIC_MOVE);
        } catch (Exception ignored) {
            // best-effort — never let status reporting break the solve
        }
    }

    /** overall progress = time spent in earlier stages' budgets + fraction of the current stage's budget. */
    static double overallProgressForStage(int stage, long msSpentInStage) {
        long priorBudgetMs = (stage <= 1) ? 0 : STAGE1_BUDGET_MS;
        long stageBudgetMs = (stage <= 1) ? STAGE1_BUDGET_MS : STAGE2_BUDGET_MS;
        double stageFrac = Math.min(1.0, msSpentInStage / (double) stageBudgetMs);
        return (priorBudgetMs + stageFrac * stageBudgetMs) / (double) TOTAL_BUDGET_MS;
    }

        static final class ScoreFiles {
            static void writeText(java.nio.file.Path path, String header, String body) {
                try (var bw = java.nio.file.Files.newBufferedWriter(
                        path,
                        java.nio.file.StandardOpenOption.CREATE,
                        java.nio.file.StandardOpenOption.TRUNCATE_EXISTING,
                        java.nio.file.StandardOpenOption.WRITE)) {
                    bw.write(header);
                    bw.newLine();
                    bw.write(body);
                    bw.newLine();
                } catch (java.io.IOException e) {
                    System.err.println("Failed to write " + path + ": " + e.getMessage());
                }
            }
        }


    public static void writeInitialScoreAnalysis(
        SolverFactory<SinglePassPlan> factory,
        SinglePassPlan initialPlan,
        String fileName,
        int topMatchesPerConstraints ) {
            try {
                var solutionManager = SolutionManager.create(factory);
                solutionManager.update(initialPlan);
                var analysis = solutionManager.analyze(initialPlan);
                String header = "========== INITIAL Score (pre-solve ) ==========\n" +
                    "Total initial score: " + analysis.score();
                
                String body = analysis.summarize(Math.max(1, topMatchesPerConstraints));
                ScoreFiles.writeText(java.nio.file.Paths.get(fileName), header, body);                
            } catch (Exception ex) {
                System.err.println("writeInitialScoreAnalysis failed: " + ex.getMessage());
            }
        }

    public static void writeFinalScoreAnalysis(
        SolverFactory<SinglePassPlan> factory,
        SinglePassPlan finalPlan,
        String fileName,
        int topMatchesPerConstraints) {
            try {
                var solutionManager = SolutionManager.create(factory);
                solutionManager.update(finalPlan);
                var analysis = solutionManager.analyze(finalPlan);
                String header = "========== Final Score (pre-solve ) ==========\n" +
                    "Total initial score: " + analysis.score();
                String body = analysis.summarize(Math.max(1,topMatchesPerConstraints));
                ScoreFiles.writeText(java.nio.file.Paths.get(fileName), header, body);
            } catch (Exception ex){
                System.err.println("writeInitialScoreAnalysis failed:  " + ex.getMessage());
            }
        }
    





// ---------------- Public API ----------------

    public static class RunResult { public SinglePassPlan plan; public LocalDate planStart; }


    /**
     * Solver factory with the production model (solution, entities, constraints, default phases) for tests:
     * optional step-count limit and random seed; no time-based termination.
     */
    static SolverFactory<SinglePassPlan> factoryForTests(Integer stepLimit, Long seed) {
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


    public static RunResult solveFromYaml(String envPath, String schedPath) throws IOException {
        PlanBuilder.BuiltProblem problem = PlanBuilder.buildProblem(envPath, schedPath);
        ParsedEnv env = problem.env;
        ParsedSchedule sch = problem.sch;
        SinglePassPlan p = problem.plan;

        SolverFactory<SinglePassPlan> factoryForInitial = buildSolverFactory(
            SinglePassPlan.class,
            new Class<?>[]{ BlockDecision.class, CrewSeat.class},
            SinglePassConstraints.class,
            "0hard/*medium/*soft",
            1,
            1);
        writeInitialScoreAnalysis(factoryForInitial, p, "initial_score.txt",200);
        
        // Init score_progress.txt for this run
        try (BufferedWriter _pw = Files.newBufferedWriter(Paths.get("score_progress.txt"),
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
            _pw.write("Stage   | elapsed | total score                             | per-constraint (hard/medium only)");
            _pw.newLine();
            _pw.write("------- | ------- | --------------------------------------- | --------------------------------------------------");
            _pw.newLine();
        } catch (IOException ignored) {}

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
            writeRunningStatus(stage, overallProgressForStage(stage, msSpent));
        }, 5, 10, TimeUnit.SECONDS);

        System.out.println("Start SINGLE PASS (stage1) at " + nowClock());
        long t0 = System.nanoTime();
        // ---- Stage 1: current settings (e.g., 90/90) ----
        SolverFactory<SinglePassPlan> factoryStage1 = buildSolverFactory(
                SinglePassPlan.class,
                new Class<?>[]{ BlockDecision.class, CrewSeat.class },
                SinglePassConstraints.class,
                "0hard/*medium/*soft", 300, 600);
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

        // ---- Stage 2 (polish): 60 minutes, start from stage1 result ----
        tickerStage.set(2);
        tickerStageStartNanos.set(System.nanoTime());
        System.out.println("Start POLISH (stage2, 60m) at " + nowClock());

        SolverFactory<SinglePassPlan> factoryStage2 = buildSolverFactory(
                SinglePassPlan.class,
                new Class<?>[]{ BlockDecision.class, CrewSeat.class },
                SinglePassConstraints.class,
                null /* bestScoreLimit */,
                60  /* spentMinutes */,
                60 /* unimprovedSeconds */);

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
        SinglePassPlan best2 = stage2.solve(best1);
        progressTicker.shutdownNow();

        long t2 = System.nanoTime();
        System.out.printf("Stage2 done %s | duration=%s | score=%s%n",
                nowClock(), fmt(java.time.Duration.ofNanos(t2 - t1)),
                String.valueOf(best2.getScore()));

        java.time.Duration stage1Dur = java.time.Duration.ofNanos(t1 - t0);
        java.time.Duration stage2Dur = java.time.Duration.ofNanos(t2 - t1);


        System.out.printf("Stage1 done %s | duration=%s | score=%s | blocks=%d | seats=%d%n",
                nowClock(), fmt(stage1Dur),
                String.valueOf(best1.getScore()),
                best1.blocks == null ? 0 : best1.blocks.size(),
                best1.seats == null ? 0 : best1.seats.size());

        System.out.printf("Stage2 done %s | duration=%s | score=%s%n",
                nowClock(), fmt(stage2Dur),
                String.valueOf(best2.getScore()));

        PlanBuilder.validateHorizonOverlaps(best2, SolveContext.current().solveMinDay, SolveContext.current().solveMaxDay);

        // Append log to file (score + durations + per-constraint score)
        writeFinalScoreAnalysis(factoryStage2, best2, "final_score.txt", 100);
        appendSolveLog(factoryStage2, best2, sch, stage1Dur, stage2Dur);

        RunResult rr = new RunResult();
        rr.plan = best2; rr.planStart = sch.planStart;
        // RunResult rr = new RunResult();
        // rr.plan = best1; rr.planStart = sch.planStart;
        return rr;

    }

    // ---------------- Export hook ----------------
    public static void main(String[] args) throws Exception {
        String envPath = args.length > 0 ? args[0] : "EnvConfig.yaml";
        String schedPath = args.length > 1 ? args[1] : "Schedule.yaml";

        RunResult rr = solveFromYaml(envPath, schedPath);

        ScheduleExporter.overwriteScheduleWithAssignments(
                rr.plan, rr.planStart, schedPath, envPath);

        System.out.println("Done.");
    }
}