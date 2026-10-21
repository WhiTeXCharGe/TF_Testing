package com.yourorg.scheduler.solver;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.LocalDate;

import ai.timefold.solver.core.api.score.Score;
import ai.timefold.solver.core.api.score.ScoreExplanation;
import ai.timefold.solver.core.api.score.buildin.hardmediumsoft.HardMediumSoftScore;
import ai.timefold.solver.core.api.solver.SolutionManager;
import ai.timefold.solver.core.api.solver.SolverFactory;

import com.yourorg.scheduler.config.SolverConstants;
import com.yourorg.scheduler.model.BlockDecision;
import com.yourorg.scheduler.model.CrewSeat;
import com.yourorg.scheduler.model.DaySlot;
import com.yourorg.scheduler.model.ParsedSchedule;
import com.yourorg.scheduler.model.SinglePassPlan;

/** Score logging: solver_log.txt, score_progress.txt and the initial/stage1/final score analysis files. */
public final class ScoreLogger {
    private ScoreLogger() {}

    private static final java.time.format.DateTimeFormatter CLOCK =
            java.time.format.DateTimeFormatter.ofPattern("HH:mm:ss");
    static String nowClock() { return java.time.LocalTime.now().format(CLOCK); }
    static String fmt(java.time.Duration d) {
        long h = d.toHours(); long m = d.toMinutesPart(); long s = d.toSecondsPart(); long ms = d.toMillisPart();
        return String.format("%02d:%02d:%02d.%03d", h, m, s, ms);
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
}
