package com.yourorg.scheduler.solver;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.LocalDate;
import java.util.Map;

import ai.timefold.solver.core.api.score.HardMediumSoftScore;
import ai.timefold.solver.core.api.score.Score;
import ai.timefold.solver.core.api.solver.SolverFactory;

import com.yourorg.scheduler.config.SolverConstants;
import com.yourorg.scheduler.io.LoadLog;
import com.yourorg.scheduler.model.BlockDecision;
import com.yourorg.scheduler.model.CrewSeat;
import com.yourorg.scheduler.model.DaySlot;
import com.yourorg.scheduler.model.ParsedSchedule;
import com.yourorg.scheduler.model.SinglePassPlan;

/** Score logging: solver_log.txt, score_progress.txt and the initial/stage1/final score analysis files. */
public final class ScoreLogger {
    private ScoreLogger() {}

    // write_score_files: false suppresses initial_score.txt, stage1_score.txt, final_score.txt and
    // score_progress.txt (solver_log.txt is unaffected). Set once by Main before the solve; default true.
    private static volatile boolean writeScoreFiles = true;

    public static void setWriteScoreFiles(boolean enabled) { writeScoreFiles = enabled; }

    public static boolean writeScoreFiles() { return writeScoreFiles; }

    private static final java.time.format.DateTimeFormatter CLOCK =
            java.time.format.DateTimeFormatter.ofPattern("HH:mm:ss");
    static String nowClock() { return java.time.LocalTime.now().format(CLOCK); }
    static String fmt(java.time.Duration d) {
        long h = d.toHours(); long m = d.toMinutesPart(); long s = d.toSecondsPart(); long ms = d.toMillisPart();
        return String.format("%02d:%02d:%02d.%03d", h, m, s, ms);
    }

    static boolean hardZero(Score<?> s) { return s != null && s.toString().startsWith("0hard"); }

    /**
     * A score-file writer failed, most likely inside {@link ConstraintBreakdown} (Timefold internal API). Reported on
     * stderr and in load_log.txt with the exception class, so the failure point is visible after an upgrade.
     */
    static void reportFailure(String where, Exception ex) {
        String msg = where + " failed: " + ex.getClass().getName() + ": " + ex.getMessage();
        System.err.println(msg);
        LoadLog.appendLoadLog("[SCORE-FILE-FAILED] " + msg);
    }

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

            // Per-constraint breakdown (Community replacement of SolutionManager.explain, see ConstraintBreakdown).
            // It uses Timefold internals and runs after the whole solve: a failure here must not skip the export.
            ConstraintBreakdown.Result breakdown = null;
            try {
                breakdown = ConstraintBreakdown.of(factoryStage2).analyze(best2);
            } catch (RuntimeException ex) {
                reportFailure("solver_log.txt constraint breakdown", ex);
                System.err.println("[WARN] solver_log.txt: per-constraint scores skipped (" + ex.getClass().getSimpleName() + ")");
            }

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

            // Per-constraint scores (sorted by constraint id; 2.x ids carry no package prefix)
            if (breakdown == null) {
                bw.write("  [unavailable: breakdown failed, see load_log.txt]");
                bw.newLine();
            } else {
                for (Map.Entry<String, ConstraintBreakdown.Total> e : breakdown.totals().entrySet()) {
                    bw.write("  " + e.getKey()
                            + " score=" + e.getValue().score()
                            + " matches=" + e.getValue().matches());
                    bw.newLine();
                }
            }

            bw.flush();
        } catch (IOException e) {
            System.err.println("Failed to write solver_log.txt: " + e.getMessage());
        }
    }


    // Appends one line per best-score improvement to score_progress.txt.
    // Call from a BestSolutionChangedEvent listener so you can watch which constraints
    // are still firing and how fast the score is improving over time.
    static void appendScoreProgress(String stage, long msElapsed, HardMediumSoftScore score,
                                    ConstraintBreakdown breakdown,
                                    SinglePassPlan solution) {
        if (!writeScoreFiles) return;
        Path logFile = Paths.get("score_progress.txt");
        try (BufferedWriter bw = Files.newBufferedWriter(logFile,
                StandardOpenOption.CREATE, StandardOpenOption.APPEND)) {

            bw.write(String.format("%-7s | t+%5ds | total=%-40s%n",
                    stage, msElapsed / 1000, score));

            // Per-constraint breakdown (slightly expensive; kept because it's best-effort)
            try {
                ConstraintBreakdown.Result ex = breakdown.analyze(solution);
                for (Map.Entry<String, ConstraintBreakdown.Total> e : ex.totals().entrySet()) {
                    HardMediumSoftScore cs = e.getValue().score();
                    if (cs.hardScore() == 0 && cs.mediumScore() == 0) continue; // skip zero-hard
                    bw.write(String.format("          %-45s score=%-30s matches=%d%n",
                            e.getKey(), cs, e.getValue().matches()));
                }
            } catch (Exception ignored) { /* the breakdown may fail mid-solve; skip */ }

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
            if (!writeScoreFiles) return;
            try {
                var analysis = ConstraintBreakdown.of(factory).analyzeWithMatches(initialPlan, topMatchesPerConstraints);
                String header = "========== INITIAL Score (pre-solve ) ==========\n" +
                    "Total initial score: " + analysis.score();

                String body = ConstraintBreakdown.summarize(analysis, topMatchesPerConstraints);
                ScoreFiles.writeText(java.nio.file.Paths.get(fileName), header, body);                
            } catch (Exception ex) {
                reportFailure("writeInitialScoreAnalysis", ex);
            }
        }

    public static void writeFinalScoreAnalysis(
        SolverFactory<SinglePassPlan> factory,
        SinglePassPlan finalPlan,
        String fileName,
        int topMatchesPerConstraints) {
            if (!writeScoreFiles) return;
            try {
                var analysis = ConstraintBreakdown.of(factory).analyzeWithMatches(finalPlan, topMatchesPerConstraints);
                String header = "========== Final Score (pre-solve ) ==========\n" +
                    "Total initial score: " + analysis.score();
                String body = ConstraintBreakdown.summarize(analysis, topMatchesPerConstraints);
                ScoreFiles.writeText(java.nio.file.Paths.get(fileName), header, body);
            } catch (Exception ex){
                reportFailure("writeFinalScoreAnalysis(" + fileName + ")", ex);
            }
        }
}
