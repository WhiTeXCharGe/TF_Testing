package com.yourorg.scheduler;

import org.yaml.snakeyaml.LoaderOptions;
import org.yaml.snakeyaml.Yaml;
import org.yaml.snakeyaml.constructor.SafeConstructor;
import org.yaml.snakeyaml.DumperOptions;

import java.io.*;
import java.nio.file.*;
import java.time.Duration;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

/**
 * v843_sp: Stress tool for EmployeeSchedule.
 *
 * - Uses the SAME EnvConfig.yaml / Schedule.yaml.
 * - Does NOT add modules, does NOT add workers,
 *   does NOT change env or schedule logic.
 * - Just keeps solving repeatedly with different seeds
 *   until a time budget is reached or maxRuns reached.
 *
 * Output:
 *   - Each run overwrites Schedule.yaml (as usual),
 *     then gets copied to:
 *         schedule_outputs/Schedule_runNN.yaml
 *   - Each run creates its own solver log:
 *         schedule_outputs/solver_log_runNN.txt
 *
 * Usage (examples):
 *   mvn -q exec:java -D"exec.args=EnvConfig.yaml Schedule.yaml 3600 50"
 *
 *   args:
 *     0: EnvConfig path (default: EnvConfig.yaml)
 *     1: Schedule path (default: Schedule.yaml)
 *     2: totalSeconds (optional, default: 3600)
 *     3: maxRuns      (optional, default: 100)
 */
public class SolverStressRunner {

    private static final DateTimeFormatter TS_FMT =
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    private static void log(String msg) {
        System.out.println("[" + LocalDateTime.now().format(TS_FMT) + "] " + msg);
    }

    public static void main(String[] args) throws Exception {
        // --------- 0. Resolve project root & paths ----------
        Path projectRoot = findProjectRoot(Paths.get("").toAbsolutePath());

        String envArg = (args.length > 0) ? args[0] : "EnvConfig.yaml";
        String schedArg = (args.length > 1) ? args[1] : "Schedule.yaml";

        Path envPath = projectRoot.resolve(envArg);
        Path schedPath = projectRoot.resolve(schedArg);

        // total seconds budget
        int totalSeconds = (args.length > 2) ? parseIntSafe(args[2], 3600) : 3600;
        // max number of runs
        int maxRuns = (args.length > 3) ? parseIntSafe(args[3], 100) : 100;

        if (!Files.exists(envPath) || !Files.exists(schedPath)) {
            throw new IllegalStateException(
                    "Expected " + envPath + " and " + schedPath + " to exist.");
        }

        Path outDir = projectRoot.resolve("src/main/resource/schedule_outputs");
        Files.createDirectories(outDir);

        log("v843_sp: Solver stress runner.");
        log("EnvConfig: " + projectRoot.relativize(envPath));
        log("Schedule : " + projectRoot.relativize(schedPath));
        log("Time budget (sec): " + totalSeconds);
        log("Max runs          : " + maxRuns);

        long t0 = System.nanoTime();
        int runIndex = 1;
        Random seedRng = new Random(123456789L); // base for per-run seed

        while (true) {
            long elapsedSec = Duration.ofNanos(System.nanoTime() - t0).getSeconds();
            if (elapsedSec >= totalSeconds) {
                log("Time budget reached: " + elapsedSec + " sec >= " + totalSeconds + " sec.");
                break;
            }
            if (runIndex > maxRuns) {
                log("Max runs reached: " + maxRuns);
                break;
            }

            long runSeed = seedRng.nextLong();
            log("==============================================");
            log("Run #" + runIndex + "  (seed=" + runSeed + ")");
            log("Elapsed: " + elapsedSec + " sec");
            log("==============================================");

            // 1) Restore original Schedule.yaml from a base copy if you want.
            //    For now, we just use current Schedule.yaml as input.
            //    (If you want to always go back to baseline, uncomment the copy logic.)

            // 2) Clear/rotate solver_log.txt so each run log is separate
            Path baseLogPath = projectRoot.resolve("solver_log.txt");
            if (Files.exists(baseLogPath)) {
                Files.delete(baseLogPath);
            }

            // 3) Run the solver (with seed)
            runSingleSolve(envPath, schedPath, runSeed);

            // 4) After EmployeeSchedule overwrites Schedule.yaml,
            //    copy it to a snapshot file
            String schedOutName = String.format("Schedule_run%03d.yaml", runIndex);
            Path schedOutPath = outDir.resolve(schedOutName);
            Files.copy(schedPath, schedOutPath, StandardCopyOption.REPLACE_EXISTING);
            log("[OUT] Wrote " + projectRoot.relativize(schedOutPath));

            // 5) Copy solver_log.txt to a separate file
            if (Files.exists(baseLogPath)) {
                String logOutName = String.format("solver_log_run%03d.txt", runIndex);
                Path logOutPath = outDir.resolve(logOutName);
                Files.copy(baseLogPath, logOutPath, StandardCopyOption.REPLACE_EXISTING);
                log("[OUT] Wrote " + projectRoot.relativize(logOutPath));
            } else {
                log("[WARN] solver_log.txt not found after run " + runIndex);
            }

            runIndex++;
        }

        long totalElapsedSec = Duration.ofNanos(System.nanoTime() - t0).getSeconds();
        log("Done. Total elapsed ~" + totalElapsedSec + " sec, runs executed: " + (runIndex - 1));
    }

    private static int parseIntSafe(String s, int def) {
        if (s == null) return def;
        try { return Integer.parseInt(s.trim()); }
        catch (Exception e) { return def; }
    }

    private static void runSingleSolve(Path envPath, Path schedPath, long seed) throws Exception {
        // Hook for passing seed to EmployeeSchedule.
        // If you don't want to change EmployeeSchedule, you can ignore `seed`.
        // Here we call a new helper that accepts a seed.
        EmployeeSchedule.solveWithSeed(
                envPath.toString().replace("\\", "/"),
                schedPath.toString().replace("\\", "/"),
                seed
        );
    }

    private static Path findProjectRoot(Path start) {
        Path p = start;
        while (p != null) {
            if (Files.exists(p.resolve("pom.xml"))) return p;
            p = p.getParent();
        }
        throw new RuntimeException("Could not find pom.xml above " + start);
    }
}
