package com.yourorg.scheduler;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.yourorg.scheduler.build.PlanBuilder;
import com.yourorg.scheduler.calendar.SolveContext;
import com.yourorg.scheduler.config.SolverParams;
import com.yourorg.scheduler.export.ScheduleExporter;
import com.yourorg.scheduler.solver.RunResult;
import com.yourorg.scheduler.solver.ScoreLogger;
import com.yourorg.scheduler.solver.TimerConfig;
import com.yourorg.scheduler.solver.TwoStageRunner;

/**
 * SINGLE PASS scheduler (Timefold 1.x-compatible): reads EnvConfig.yaml + Schedule.yaml, solves in two stages
 * and writes the assignments back into the Schedule file.
 *
 * <pre>
 * mvn -DskipTests clean package
 * mvn -q exec:java -D"exec.args=EnvConfig.yaml Schedule.yaml"
 * </pre>
 */
public class Main {

    /** Solves with the production default timers. */
    public static RunResult solveFromYaml(String envPath, String schedPath) throws IOException {
        return solveFromYaml(envPath, schedPath, TimerConfig.defaults());
    }

    public static RunResult solveFromYaml(String envPath, String schedPath, TimerConfig timers) throws IOException {
        PlanBuilder.BuiltProblem problem = PlanBuilder.buildProblem(envPath, schedPath);
        return TwoStageRunner.run(problem, timers);
    }

    /**
     * Command line split into positional arguments (envPath, schedPath) and parameter flags. A flag is a token
     * starting with {@code --}; for a known parameter written as {@code --key value} the value token belongs to the
     * flag. Put positional arguments before flags.
     */
    public record SplitArgs(List<String> positional, String[] flags) {}

    public static SplitArgs splitArgs(String[] args) {
        List<String> positional = new ArrayList<>();
        List<String> flags = new ArrayList<>();
        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            if (a != null && a.startsWith("--")) {
                flags.add(a);
                if (!a.contains("=") && SolverParams.canonicalKey(a.substring(2)) != null
                        && i + 1 < args.length && args[i + 1] != null && !args[i + 1].startsWith("--")) {
                    flags.add(args[++i]); // the value of "--key value"
                }
            } else {
                positional.add(a);
            }
        }
        return new SplitArgs(positional, flags.toArray(new String[0]));
    }

    /** Params file: TF_PARAMS_FILE, else ${INPUT_DIR or /work/input}/SolverParams.yaml. */
    public static Path paramsFile(Map<String, String> env) {
        String explicit = env.get("TF_PARAMS_FILE");
        if (explicit != null && !explicit.isBlank()) return Path.of(explicit);
        String inputDir = env.get("INPUT_DIR");
        return Path.of(inputDir == null || inputDir.isBlank() ? "/work/input" : inputDir, "SolverParams.yaml");
    }

    public static void main(String[] args) throws Exception {
        SplitArgs split = splitArgs(args);
        List<String> pos = split.positional();
        String envPath = pos.size() > 0 ? pos.get(0) : "EnvConfig.yaml";
        String schedPath = pos.size() > 1 ? pos.get(1) : "Schedule.yaml";

        Map<String, String> env = System.getenv();
        SolverParams params = SolverParams.load(split.flags(), env, paramsFile(env));
        params.warnings().forEach(w -> System.out.println("[params] WARN " + w));
        System.out.println("[params] " + params.describe());
        ScoreLogger.setWriteScoreFiles(params.writeScoreFiles());
        // misc rows whose region is listed here count as region-less (design section 7); read by the build step
        SolveContext.current().miscNeutralRegions = params.miscNeutralRegions();
        // trip_list problems fail the run only in strict mode (read by the build step)
        SolveContext.current().tripValidationStrict = "strict".equals(params.tripValidation());

        RunResult rr = solveFromYaml(envPath, schedPath, params.timers());

        ScheduleExporter.overwriteScheduleWithAssignments(
                rr.plan, rr.planStart, schedPath, envPath);

        System.out.println("Done.");
    }
}
